"""Cross-fitted episodic factual adaptation for longitudinal causal study models.

Adapts a pretrained ``CausalStudyModel`` to a single applied dataset using strictly
observable ``(X, T, Y)`` factual losses inside K-fold cross-fitting:
  - Freezes synthetic-pretrained CATE shrinkage and posterior calibration heads so
    unobserved treatment effect heterogeneity cannot overfit factual noise.
  - Adapts patient representation and nuisance-weighting layers via out-of-sample
    inner episodes ``(ctx_inner -> qry_inner)`` with an L2 anchor to pretrained weights.
  - Gates every fold automatically against step-0 zero-shot inner-validation factual
    loss, falling back seamlessly to zero-shot when ``N`` is too small for adaptation
    to improve observable ``(T, Y)`` likelihood.
"""

from contextlib import nullcontext
import copy
import os
from pathlib import Path
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F

from .events import binary_observations, read_patient_table
from .model import SentenceBatch
from .training import OBSERVED, _autocast

OVERLAP_MARGIN = 0.05

# Parameter name prefixes frozen during applied adaptation to prevent CATE overfitting.
_FROZEN_PREFIXES = (
    "log_lam_tau",
    "log_text_mult_tau",
    "log_cv_temp",
    "arm_cal.",
    "prop_cal.",
    "layers.",
    "head.",
    "propensity_head.",
    "attn_residual_scale",
    "prop_residual_scale",
    "outcome_token.",
    "slot_proj.",
    "slot_embed",
)


def _stratified_folds(t: np.ndarray, y: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    """Deterministic stratified K-fold split on (T, Y) (falling back to T if sparse)."""
    n = len(t)
    if n_folds < 2 or n_folds > n:
        raise ValueError(f"Expected 2 <= folds <= N ({n}), got {n_folds}")
    rng = np.random.default_rng(seed)
    t_i = (t > 0.5).astype(np.int64)
    y_i = (y > 0.5).astype(np.int64)
    strata = t_i * 2 + y_i
    counts = np.bincount(strata, minlength=4)
    if np.any((counts > 0) & (counts < n_folds)):
        strata = t_i
    folds = [[] for _ in range(n_folds)]
    for cls in np.unique(strata):
        idx = np.where(strata == cls)[0]
        rng.shuffle(idx)
        for rank, item in enumerate(idx):
            folds[rank % n_folds].append(int(item))
    return [np.array(sorted(f), dtype=np.int64) for f in folds]


def _stratified_split(
    t: np.ndarray, y: np.ndarray, val_frac: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified train/val index split keeping at least one sample per stratum when possible."""
    t_i = (t > 0.5).astype(np.int64)
    y_i = (y > 0.5).astype(np.int64)
    strata = t_i * 2 + y_i
    tr_idx, va_idx = [], []
    for cls in np.unique(strata):
        idx = np.where(strata == cls)[0]
        rng.shuffle(idx)
        if len(idx) < 2:
            tr_idx.extend(idx.tolist())
            continue
        n_va = max(1, min(len(idx) - 1, int(round(len(idx) * val_frac))))
        va_idx.extend(idx[:n_va].tolist())
        tr_idx.extend(idx[n_va:].tolist())
    if not va_idx:
        all_idx = np.arange(len(t))
        rng.shuffle(all_idx)
        n_va = max(1, min(len(t) - 1, int(round(len(t) * val_frac))))
        return np.sort(all_idx[n_va:]), np.sort(all_idx[:n_va])
    return np.array(sorted(tr_idx), dtype=np.int64), np.array(sorted(va_idx), dtype=np.int64)


def _subset_sentence_batch(batch: SentenceBatch, indices: np.ndarray) -> SentenceBatch:
    """Slice a SentenceBatch to a subset of patients with zero leakage from excluded patients."""
    device = batch.unique_embs.device
    hidden = batch.unique_embs.shape[1]
    n_sub = len(indices)
    if n_sub == 0 or len(batch.flat_idx) == 0:
        return SentenceBatch(
            unique_embs=torch.empty((0, hidden), device=device, dtype=torch.float32),
            flat_idx=torch.empty((0,), device=device, dtype=torch.long),
            patient_ids=torch.empty((0,), device=device, dtype=torch.long),
            num_patients=n_sub,
            unique_texts=(),
        )

    remap_pat = torch.full((batch.num_patients,), -1, device=device, dtype=torch.long)
    idx_t = torch.as_tensor(indices, device=device, dtype=torch.long)
    remap_pat[idx_t] = torch.arange(n_sub, device=device, dtype=torch.long)

    new_pat_ids = remap_pat[batch.patient_ids]
    keep = new_pat_ids >= 0
    if not bool(keep.any().item()):
        return SentenceBatch(
            unique_embs=torch.empty((0, hidden), device=device, dtype=torch.float32),
            flat_idx=torch.empty((0,), device=device, dtype=torch.long),
            patient_ids=torch.empty((0,), device=device, dtype=torch.long),
            num_patients=n_sub,
            unique_texts=(),
        )

    kept_pat = new_pat_ids[keep]
    kept_flat = batch.flat_idx[keep]
    used_u = torch.unique(kept_flat, sorted=True)
    remap_u = torch.full((len(batch.unique_embs),), -1, device=device, dtype=torch.long)
    remap_u[used_u] = torch.arange(len(used_u), device=device, dtype=torch.long)
    new_flat = remap_u[kept_flat]
    new_embs = batch.unique_embs[used_u]
    new_texts = (
        tuple(batch.unique_texts[i] for i in used_u.cpu().tolist())
        if len(batch.unique_texts) == len(batch.unique_embs)
        else ()
    )
    return SentenceBatch(
        unique_embs=new_embs,
        flat_idx=new_flat,
        patient_ids=kept_pat,
        num_patients=n_sub,
        unique_texts=new_texts,
    )


def _configure_adaptation_params(model, *, adapt_backbone: bool = False):
    """Freeze CATE/calibration heads and return trainable representation/nuisance parameters."""
    backbone_params = []
    adapter_params = []
    for name, param in model.named_parameters():
        if name.startswith(_FROZEN_PREFIXES):
            param.requires_grad = False
            continue
        if name.startswith("patient.backbone."):
            if adapt_backbone:
                param.requires_grad = True
                backbone_params.append((name, param))
            else:
                param.requires_grad = False
            continue
        param.requires_grad = True
        adapter_params.append((name, param))
    return adapter_params, backbone_params


def _observable_factual_loss(model, outcome_logits, prop_logits, scale, t_q, y_q):
    """Strictly observable factual BCE on observed arm outcomes and treatment propensities."""
    mean_risk, _ = model.risk_stats(outcome_logits, scale)
    arm_idx = t_q.long()
    p_factual = mean_risk[torch.arange(len(arm_idx), device=arm_idx.device), arm_idx].clamp(
        1e-5, 1.0 - 1e-5
    )
    out_bce = F.binary_cross_entropy(p_factual, y_q.float())
    prop_bce = F.binary_cross_entropy_with_logits(prop_logits.float(), t_q.float())
    return out_bce + 0.5 * prop_bce


def _encode_rows_chunked(
    model, rows, *, batch_size: int, precision: str, freeze_backbone: bool
):
    device = model.centers.device
    chunks = []
    for start in range(0, len(rows), batch_size):
        sub = rows[start : start + batch_size]
        grad_mask = [False] * len(sub) if freeze_backbone else None
        with _autocast(device, precision):
            chunks.append(model.patient(sub, backbone_grad_mask=grad_mask))
    return torch.cat(chunks, dim=0)


def _forward_with_cached_text(
    model,
    ctx_rows,
    qry_rows,
    *,
    ctx_lex=None,
    qry_lex=None,
    ctx_sent=None,
    qry_sent=None,
    batch_size: int,
    precision: str,
    freeze_backbone: bool,
):
    device = model.centers.device
    t_ctx, y_ctx = binary_observations(ctx_rows, device)
    ctx_emb = _encode_rows_chunked(
        model, ctx_rows, batch_size=batch_size, precision=precision, freeze_backbone=freeze_backbone
    )
    qry_emb = _encode_rows_chunked(
        model, qry_rows, batch_size=batch_size, precision=precision, freeze_backbone=freeze_backbone
    )
    with _autocast(device, precision):
        memories, scale = model.condition(
            ctx_emb, t_ctx, y_ctx, lexical=ctx_lex, sentences=ctx_sent
        )
        outcome_logits = model.query_logits(
            qry_emb, memories, scale, lexical=qry_lex, sentences=qry_sent
        )
        prop_logits = model.query_propensity_logits(
            qry_emb, memories, scale, lexical=qry_lex, sentences=qry_sent
        )
    return outcome_logits, prop_logits, scale


def adapt_fold_model(
    base_model,
    train_rows: list[dict],
    *,
    adapt_steps: int = 30,
    lr: float = 1e-3,
    backbone_lr: float = 1e-5,
    anchor_weight: float = 1e-2,
    adapt_backbone: bool = False,
    batch_size: int = 64,
    eval_every: int = 5,
    precision: str = "auto",
    seed: int = 42,
    train_lex: torch.Tensor | None = None,
    train_sent: SentenceBatch | None = None,
):
    """Adapt a copy of ``base_model`` on ``train_rows`` via episodic observable factual losses.

    Automatically gates against step-0 zero-shot inner-validation loss: if adaptation does
    not improve observable ``(T, Y)`` log-loss on the inner validation split, returns the
    unmodified step-0 zero-shot weights.
    """
    model = copy.deepcopy(base_model)
    device = model.centers.device
    use_lex = model.cfg.lexical_dim > 0
    use_sent = model.cfg.sentence_concept_dim > 0

    if len(train_rows) < 10 or adapt_steps <= 0:
        model.eval()
        return model, {
            "used_adaptation": False,
            "best_step": 0,
            "val_loss_step0": float("nan"),
            "val_loss_best": float("nan"),
        }

    if use_lex and train_lex is None:
        train_lex = model.patient.lexical_features(train_rows)
    if use_sent and train_sent is None and not adapt_backbone:
        with _autocast(device, precision):
            train_sent = model.patient.sentence_features(train_rows)

    t_all = np.array([float(r["treatment_indicator"]) for r in train_rows], dtype=np.float64)
    y_all = np.array([float(r["outcome_indicator"]) for r in train_rows], dtype=np.float64)
    rng = np.random.default_rng(seed)
    tr_idx, va_idx = _stratified_split(t_all, y_all, val_frac=0.2, rng=rng)
    if len(tr_idx) < 6 or len(va_idx) < 2:
        model.eval()
        return model, {
            "used_adaptation": False,
            "best_step": 0,
            "val_loss_step0": float("nan"),
            "val_loss_best": float("nan"),
        }

    inner_train = [train_rows[i] for i in tr_idx]
    inner_val = [train_rows[i] for i in va_idx]
    t_inner = t_all[tr_idx]
    y_inner = y_all[tr_idx]

    tr_idx_t = torch.as_tensor(tr_idx, device=device, dtype=torch.long)
    va_idx_t = torch.as_tensor(va_idx, device=device, dtype=torch.long)
    lex_tr = train_lex[tr_idx_t] if (use_lex and train_lex is not None) else None
    lex_va = train_lex[va_idx_t] if (use_lex and train_lex is not None) else None
    sent_tr = (
        _subset_sentence_batch(train_sent, tr_idx)
        if (use_sent and train_sent is not None)
        else None
    )
    sent_va = (
        _subset_sentence_batch(train_sent, va_idx)
        if (use_sent and train_sent is not None)
        else None
    )
    t_va_t, y_va_t = binary_observations(inner_val, device)

    def _eval_inner_val() -> float:
        model.eval()
        with torch.inference_mode():
            c_sent = (
                model.patient.sentence_features(inner_train)
                if (use_sent and adapt_backbone)
                else sent_tr
            )
            q_sent = (
                model.patient.sentence_features(inner_val)
                if (use_sent and adapt_backbone)
                else sent_va
            )
            ol, pl, sc = _forward_with_cached_text(
                model,
                inner_train,
                inner_val,
                ctx_lex=lex_tr,
                qry_lex=lex_va,
                ctx_sent=c_sent,
                qry_sent=q_sent,
                batch_size=batch_size,
                precision=precision,
                freeze_backbone=True,
            )
            return float(_observable_factual_loss(model, ol, pl, sc, t_va_t, y_va_t).item())

    adapter_params, backbone_params = _configure_adaptation_params(
        model, adapt_backbone=adapt_backbone
    )
    trainable = [*adapter_params, *backbone_params]
    if not trainable:
        model.eval()
        val0 = _eval_inner_val()
        return model, {
            "used_adaptation": False,
            "best_step": 0,
            "val_loss_step0": val0,
            "val_loss_best": val0,
        }

    anchor_ref = {name: p.detach().clone() for name, p in trainable}
    best_state = {name: p.detach().clone() for name, p in trainable}
    val_loss_0 = _eval_inner_val()
    best_val = val_loss_0
    best_step = 0

    param_groups = []
    if adapter_params:
        param_groups.append({"params": [p for _, p in adapter_params], "lr": lr})
    if backbone_params:
        param_groups.append({"params": [p for _, p in backbone_params], "lr": backbone_lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.0)

    for step in range(1, adapt_steps + 1):
        model.train()
        ep_ctx_i, ep_qry_i = _stratified_split(t_inner, y_inner, val_frac=0.25, rng=rng)
        ep_ctx = [inner_train[i] for i in ep_ctx_i]
        ep_qry = [inner_train[i] for i in ep_qry_i]
        t_q_t, y_q_t = binary_observations(ep_qry, device)

        ep_ctx_t = torch.as_tensor(ep_ctx_i, device=device, dtype=torch.long)
        ep_qry_t = torch.as_tensor(ep_qry_i, device=device, dtype=torch.long)
        c_lex = lex_tr[ep_ctx_t] if lex_tr is not None else None
        q_lex = lex_tr[ep_qry_t] if lex_tr is not None else None
        if use_sent:
            if adapt_backbone:
                c_sent = model.patient.sentence_features(ep_ctx)
                q_sent = model.patient.sentence_features(ep_qry)
            else:
                c_sent = _subset_sentence_batch(sent_tr, ep_ctx_i)
                q_sent = _subset_sentence_batch(sent_tr, ep_qry_i)
        else:
            c_sent = q_sent = None

        optimizer.zero_grad(set_to_none=True)
        ol, pl, sc = _forward_with_cached_text(
            model,
            ep_ctx,
            ep_qry,
            ctx_lex=c_lex,
            qry_lex=q_lex,
            ctx_sent=c_sent,
            qry_sent=q_sent,
            batch_size=batch_size,
            precision=precision,
            freeze_backbone=not adapt_backbone,
        )
        loss = _observable_factual_loss(model, ol, pl, sc, t_q_t, y_q_t)
        if anchor_weight > 0:
            sq_diff = sum((p - anchor_ref[n]).float().square().sum() for n, p in trainable)
            n_elem = sum(p.numel() for _, p in trainable)
            loss = loss + anchor_weight * (sq_diff / max(n_elem, 1))
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
            optimizer.step()

        if step % eval_every == 0 or step == adapt_steps:
            val_loss = _eval_inner_val()
            if np.isfinite(val_loss) and val_loss < best_val - 1e-4:
                best_val = val_loss
                best_step = step
                best_state = {name: p.detach().clone() for name, p in trainable}

    with torch.no_grad():
        for name, param in trainable:
            param.copy_(best_state[name])
    model.eval()
    return model, {
        "used_adaptation": best_step > 0,
        "best_step": best_step,
        "val_loss_step0": val_loss_0,
        "val_loss_best": best_val,
    }


@torch.inference_mode()
def _score_conditioned(
    model,
    ctx_rows: list[dict],
    qry_rows: list[dict],
    *,
    batch_size: int,
    precision: str,
    ctx_lex=None,
    qry_lex=None,
    ctx_sent=None,
    qry_sent=None,
):
    model.eval()
    device = model.centers.device
    use_lex = model.cfg.lexical_dim > 0
    use_sent = model.cfg.sentence_concept_dim > 0

    if use_lex and ctx_lex is None:
        ctx_lex = model.patient.lexical_features(ctx_rows)
    if use_lex and qry_lex is None:
        qry_lex = model.patient.lexical_features(qry_rows)
    if use_sent and ctx_sent is None:
        with _autocast(device, precision):
            ctx_sent = model.patient.sentence_features(ctx_rows)
    if use_sent and qry_sent is None:
        with _autocast(device, precision):
            qry_sent = model.patient.sentence_features(qry_rows)

    ctx_emb = _encode_rows_chunked(
        model, ctx_rows, batch_size=batch_size, precision=precision, freeze_backbone=True
    )
    t_ctx, y_ctx = binary_observations(ctx_rows, device)
    with _autocast(device, precision):
        memories, scale = model.condition(
            ctx_emb, t_ctx, y_ctx, lexical=ctx_lex, sentences=ctx_sent
        )

    p0_list, p1_list, s0_list, s1_list, prop_list = [], [], [], [], []
    for start in range(0, len(qry_rows), batch_size):
        sub = qry_rows[start : start + batch_size]
        sub_idx = np.arange(start, min(start + batch_size, len(qry_rows)), dtype=np.int64)
        q_lex = qry_lex[start : start + batch_size] if qry_lex is not None else None
        q_sent = _subset_sentence_batch(qry_sent, sub_idx) if qry_sent is not None else None
        with _autocast(device, precision):
            q_emb = model.patient(sub)
            logits = model.query_logits(q_emb, memories, scale, lexical=q_lex, sentences=q_sent)
            prop_logits = model.query_propensity_logits(
                q_emb, memories, scale, lexical=q_lex, sentences=q_sent
            )
        mean, sd = model.risk_stats(logits, scale)
        p0, p1 = mean.float().cpu().unbind(dim=1)
        s0, s1 = sd.float().cpu().unbind(dim=1)
        prop = prop_logits.float().sigmoid().cpu()
        p0_list.append(p0.numpy())
        p1_list.append(p1.numpy())
        s0_list.append(s0.numpy())
        s1_list.append(s1.numpy())
        prop_list.append(prop.numpy())

    return (
        np.concatenate(p0_list),
        np.concatenate(p1_list),
        np.concatenate(s0_list),
        np.concatenate(s1_list),
        np.concatenate(prop_list),
    )


def cross_fit_adapt_rows(
    base_model,
    rows: list[dict],
    *,
    folds: int = 5,
    adapt_steps: int = 30,
    lr: float = 1e-3,
    backbone_lr: float = 1e-5,
    anchor_weight: float = 1e-2,
    adapt_backbone: bool = False,
    batch_size: int = 64,
    eval_every: int = 5,
    precision: str = "auto",
    seed: int = 42,
):
    """Run strict K-fold cross-fitted episodic adaptation over a single labeled cohort."""
    n = len(rows)
    device = base_model.centers.device
    use_lex = base_model.cfg.lexical_dim > 0
    use_sent = base_model.cfg.sentence_concept_dim > 0

    all_lex = base_model.patient.lexical_features(rows) if use_lex else None
    all_sent = None
    if use_sent and not adapt_backbone:
        with torch.inference_mode(), _autocast(device, precision):
            all_sent = base_model.patient.sentence_features(rows)

    t = np.array([float(r["treatment_indicator"]) for r in rows], dtype=np.float64)
    y = np.array([float(r["outcome_indicator"]) for r in rows], dtype=np.float64)
    fold_indices = _stratified_folds(t, y, folds, seed)

    p0 = np.empty(n, dtype=np.float32)
    p1 = np.empty(n, dtype=np.float32)
    s0 = np.empty(n, dtype=np.float32)
    s1 = np.empty(n, dtype=np.float32)
    prop = np.empty(n, dtype=np.float32)
    fold_stats = []

    for k, val_idx in enumerate(fold_indices):
        val_set = set(val_idx.tolist())
        train_idx = np.array([i for i in range(n) if i not in val_set], dtype=np.int64)
        train_rows = [rows[i] for i in train_idx]
        val_rows = [rows[i] for i in val_idx]

        tr_t = torch.as_tensor(train_idx, device=device, dtype=torch.long)
        va_t = torch.as_tensor(val_idx, device=device, dtype=torch.long)
        tr_lex = all_lex[tr_t] if all_lex is not None else None
        va_lex = all_lex[va_t] if all_lex is not None else None
        tr_sent = _subset_sentence_batch(all_sent, train_idx) if all_sent is not None else None
        va_sent = _subset_sentence_batch(all_sent, val_idx) if all_sent is not None else None

        adapted, info = adapt_fold_model(
            base_model,
            train_rows,
            adapt_steps=adapt_steps,
            lr=lr,
            backbone_lr=backbone_lr,
            anchor_weight=anchor_weight,
            adapt_backbone=adapt_backbone,
            batch_size=batch_size,
            eval_every=eval_every,
            precision=precision,
            seed=seed + 101 * k,
            train_lex=tr_lex,
            train_sent=tr_sent,
        )
        fp0, fp1, fs0, fs1, fprop = _score_conditioned(
            adapted,
            train_rows,
            val_rows,
            batch_size=batch_size,
            precision=precision,
            ctx_lex=tr_lex,
            qry_lex=va_lex,
            ctx_sent=tr_sent if not adapt_backbone else None,
            qry_sent=va_sent if not adapt_backbone else None,
        )
        p0[val_idx] = fp0
        p1[val_idx] = fp1
        s0[val_idx] = fs0
        s1[val_idx] = fs1
        prop[val_idx] = fprop
        fold_stats.append({"fold": k, **info})
        del adapted
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "risk_control": p0,
        "risk_treated": p1,
        "risk_difference": p1 - p0,
        "risk_control_sd": s0,
        "risk_treated_sd": s1,
        "propensity_score": prop,
        "folds": fold_stats,
    }


def _write_predictions(output: Path, id_field: pa.Field, patient_ids: list, preds: dict) -> None:
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    p0 = preds["risk_control"]
    p1 = preds["risk_treated"]
    s0 = preds["risk_control_sd"]
    s1 = preds["risk_treated_sd"]
    prop = preds["propensity_score"]
    outside = (prop < OVERLAP_MARGIN) | (prop > 1.0 - OVERLAP_MARGIN)
    schema = pa.schema(
        [
            id_field,
            ("risk_control", pa.float32()),
            ("risk_treated", pa.float32()),
            ("risk_difference", pa.float32()),
            ("risk_control_sd", pa.float32()),
            ("risk_treated_sd", pa.float32()),
            ("propensity_score", pa.float32()),
            ("outside_overlap", pa.bool_()),
        ]
    )
    table = pa.Table.from_arrays(
        [
            patient_ids,
            p0.astype(np.float32),
            p1.astype(np.float32),
            (p1 - p0).astype(np.float32),
            s0.astype(np.float32),
            s1.astype(np.float32),
            prop.astype(np.float32),
            outside,
        ],
        schema=schema,
    )
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        temporary = Path(directory) / "predictions.parquet"
        pq.write_table(table, temporary)
        fd = os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.close(fd)
        try:
            os.replace(temporary, output)
        except Exception:
            output.unlink(missing_ok=True)
            raise


def adapt_and_predict(
    model,
    context_file,
    query_file,
    output,
    *,
    folds: int = 5,
    adapt_steps: int = 30,
    lr: float = 1e-3,
    backbone_lr: float = 1e-5,
    anchor_weight: float = 1e-2,
    adapt_backbone: bool = False,
    batch_size: int = 32,
    eval_every: int = 5,
    seed: int = 42,
    precision: str = "auto",
):
    """CLI backend for ``adapt-predict`` (single-cohort K-fold cross-fit or context->queries)."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    context = read_patient_table(context_file, model.patient.streams, OBSERVED)
    ctx_ids = context["patient_id"].to_pylist()
    if None in ctx_ids or len(set(ctx_ids)) != len(ctx_ids):
        raise ValueError("Patient IDs must be present and unique within each cohort")
    ctx_rows = context.to_pylist()

    if query_file is None:
        preds = cross_fit_adapt_rows(
            model,
            ctx_rows,
            folds=folds,
            adapt_steps=adapt_steps,
            lr=lr,
            backbone_lr=backbone_lr,
            anchor_weight=anchor_weight,
            adapt_backbone=adapt_backbone,
            batch_size=batch_size,
            eval_every=eval_every,
            precision=precision,
            seed=seed,
        )
        _write_predictions(Path(output), context.schema.field("patient_id"), ctx_ids, preds)
        return len(ctx_rows), preds["folds"]

    queries = read_patient_table(query_file, model.patient.streams)
    q_ids = queries["patient_id"].to_pylist()
    if None in q_ids or len(set(q_ids)) != len(q_ids):
        raise ValueError("Patient IDs must be present and unique within each cohort")
    if set(ctx_ids) & set(q_ids):
        raise ValueError("Context and query patient IDs must be disjoint")
    q_rows = queries.to_pylist()

    adapted, info = adapt_fold_model(
        model,
        ctx_rows,
        adapt_steps=adapt_steps,
        lr=lr,
        backbone_lr=backbone_lr,
        anchor_weight=anchor_weight,
        adapt_backbone=adapt_backbone,
        batch_size=batch_size,
        eval_every=eval_every,
        precision=precision,
        seed=seed,
    )
    p0, p1, s0, s1, prop = _score_conditioned(
        adapted,
        ctx_rows,
        q_rows,
        batch_size=batch_size,
        precision=precision,
    )
    preds = {
        "risk_control": p0,
        "risk_treated": p1,
        "risk_control_sd": s0,
        "risk_treated_sd": s1,
        "propensity_score": prop,
    }
    _write_predictions(Path(output), queries.schema.field("patient_id"), q_ids, preds)
    return len(q_rows), [{"fold": 0, **info}]
