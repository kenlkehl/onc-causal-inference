"""Training, resuming, and high-P stress-test evaluation for SparseTabularCausalPFN."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from .model import CHECKPOINT_FORMAT_VERSION, SparseTabularCausalPFN, TabularModelConfig
from .prior import DGP_FAMILIES, ROLE_TO_ID, PriorConfig, sample_tabular_study


def _sample_episode_worker(args):
    seed, n_ctx, n_q, num_features, prior_dict, family = args
    prior = PriorConfig(**prior_dict)
    return sample_tabular_study(
        seed, n_ctx + n_q, num_features=num_features, prior=prior, family=family
    )


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or x.std() < 1e-9 or y.std() < 1e-9:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])


def _auroc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = y_true > 0.5
    n_pos = int(pos.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _fast_loocv_ridge_np(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_test: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    alphas=(1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Exact observational LOO-CV Ridge using economy SVD (fast for any N, P)."""
    n = len(y_fit)
    sw = np.ones(n, dtype=np.float64) if sample_weight is None else np.maximum(sample_weight, 1e-6)
    sw = sw / sw.mean()
    w_sum = sw.sum()
    y_mean = float((sw * y_fit).sum() / w_sum)
    yc = y_fit - y_mean

    mu = (sw[:, None] * X_fit).sum(axis=0, keepdims=True) / w_sum
    var = (sw[:, None] * (X_fit - mu) ** 2).sum(axis=0, keepdims=True) / w_sum
    sd = np.sqrt(np.where(var > 1e-10, var, 1.0))
    Xf = (X_fit - mu) / sd
    Xt = (X_test - mu) / sd
    sqrt_w = np.sqrt(sw)
    Xw = Xf * sqrt_w[:, None]
    yw = yc * sqrt_w

    U, s, Vt = np.linalg.svd(Xw, full_matrices=False)
    s2 = s * s
    Uty = U.T @ yw

    best_alpha = alphas[-1]
    best_loo = float("inf")
    best_loo_pred = np.full(n, y_mean, dtype=np.float64)
    for alpha in alphas:
        filt = s2 / (s2 + alpha)
        yhat_w = U @ (filt * Uty)
        h_diag = np.clip((U * U) @ filt + sw / w_sum, 0.0, 0.99)
        resid_w = (yw - yhat_w) / (1.0 - h_diag)
        loo = float(np.mean(resid_w * resid_w))
        if loo < best_loo:
            best_loo = loo
            best_alpha = alpha
            best_loo_pred = y_mean + (yw - resid_w) / np.maximum(sqrt_w, 1e-6)

    beta = Vt.T @ ((s / (s2 + best_alpha)) * Uty)
    return best_loo_pred, y_mean + Xt @ beta


def _baseline_r_learner(X_ctx, t_ctx, y_ctx, X_q, *, screen_k: int | None = None):
    """Observational LOO-CV R-learner baseline (dense or top-K observationally screened)."""
    Xc = np.nan_to_num(X_ctx.astype(np.float64), nan=0.0)
    Xq = np.nan_to_num(X_q.astype(np.float64), nan=0.0)
    # Fill NaNs with context column means
    nan_c = ~np.isfinite(X_ctx)
    nan_q = ~np.isfinite(X_q)
    obs = (~nan_c).astype(np.float64)
    col_mean = (np.where(nan_c, 0.0, X_ctx) * obs).sum(axis=0) / np.maximum(obs.sum(axis=0), 1.0)
    Xc = np.where(nan_c, col_mean[None, :], X_ctx.astype(np.float64))
    Xq = np.where(nan_q, col_mean[None, :], X_q.astype(np.float64))
    sd = Xc.std(axis=0, keepdims=True)
    Xc = (Xc - Xc.mean(axis=0, keepdims=True)) / np.where(sd > 1e-6, sd, 1.0)
    Xq = (Xq - Xc.mean(axis=0, keepdims=True)) / np.where(sd > 1e-6, sd, 1.0)

    if screen_k is not None and Xc.shape[1] > screen_k:
        # Strictly observational univariate screening on (T, Y)
        t_c = t_ctx - t_ctx.mean()
        y_c = y_ctx - y_ctx.mean()
        rt = np.abs(Xc.T @ t_c) / np.maximum(np.linalg.norm(Xc, axis=0) * np.linalg.norm(t_c), 1e-8)
        ry = np.abs(Xc.T @ y_c) / np.maximum(np.linalg.norm(Xc, axis=0) * np.linalg.norm(y_c), 1e-8)
        top = np.argsort(-(rt + 2.0 * ry))[:screen_k]
        Xc = Xc[:, top]
        Xq = Xq[:, top]

    e_loo, _ = _fast_loocv_ridge_np(Xc, t_ctx.astype(np.float64), Xq)
    m_loo, _ = _fast_loocv_ridge_np(Xc, y_ctx.astype(np.float64), Xq)
    e_loo = np.clip(e_loo, 0.03, 0.97)
    w_res = t_ctx - e_loo
    y_res = y_ctx - m_loo
    sw = np.maximum(w_res * w_res, 1e-4)
    pseudo = np.clip(
        y_res / np.where(np.abs(w_res) > 1e-2, w_res, np.sign(w_res + 1e-9) * 1e-2), -5.0, 5.0
    )
    _, tau_q = _fast_loocv_ridge_np(
        Xc,
        pseudo,
        Xq,
        sample_weight=sw,
        alphas=(10.0, 100.0, 500.0, 2000.0, 10000.0, 50000.0, 200000.0),
    )
    return np.clip(tau_q, -0.95, 0.95)


def save_tabular_checkpoint(
    path: Path,
    model: SparseTabularCausalPFN,
    optimizer,
    step: int,
    settings: dict,
    prior: PriorConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "step": int(step),
            "model": asdict(model.cfg),
            "prior": asdict(prior),
            "settings": settings,
            "weights": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
        },
        tmp,
    )
    os.replace(tmp, path)


def load_tabular_run(
    run_dir: str | Path,
    device: str = "cpu",
    *,
    prefer_best: bool = True,
) -> tuple[SparseTabularCausalPFN, dict]:
    run_dir = Path(run_dir)
    ckpt_path = run_dir / "best_checkpoint.pt"
    if not prefer_best or not ckpt_path.is_file():
        ckpt_path = run_dir / "checkpoint.pt"
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = TabularModelConfig(**saved["model"])
    model = SparseTabularCausalPFN(cfg).to(device)
    model.load_state_dict(saved["weights"], strict=True)
    model.eval()
    return model, saved


@torch.inference_mode()
def evaluate_tabular(
    model: SparseTabularCausalPFN,
    *,
    studies: int = 64,
    context_size: int = 800,
    query_size: int = 128,
    num_features: int | None = None,
    prior: PriorConfig = PriorConfig(),
    family: str | None = None,
    seed: int = 100_001,
    compute_baselines: bool = True,
    num_workers: int = 0,
) -> dict:
    """Evaluate SparseTabularCausalPFN on held-out sparse high-P synthetic studies."""
    was_training = model.training
    model.eval()
    device = model.centers.device

    # Strictly enforce held_out design split during evaluation (zero design leakage!)
    eval_prior = replace(prior, design_split="held_out")
    prior_dict = asdict(eval_prior)
    # Odd seeds reserved for held-out evaluation (even seeds for training)
    tasks = [
        (seed + 2 * i + 1, context_size, query_size, num_features, prior_dict, family)
        for i in range(studies)
    ]

    if num_workers > 0 and studies > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            episodes = list(pool.map(_sample_episode_worker, tasks))
    else:
        episodes = [_sample_episode_worker(t) for t in tasks]

    all_p0, all_p1, all_t0, all_t1 = [], [], [], []
    all_cate, all_tcate, all_e, all_te = [], [], [], []
    all_arm_cate, all_ridge_cate, all_screen_cate = [], [], []
    within_rs, ate_errs = [], []
    null_cates, non_null_cates = [], []
    role_aucs = []
    conf_ge, inst_ge = [], []
    fam_sq: dict[str, list[np.ndarray]] = {}
    fam_wr: dict[str, list[float]] = {}
    fam_arm_sq: dict[str, list[np.ndarray]] = {}

    for idx_ep, ep in enumerate(episodes, 1):
        X = ep["X"]
        M = ep["M"]
        t = ep["t"]
        y = ep["y"]
        roles = ep["roles"]
        X_ctx, M_ctx, t_ctx, y_ctx = X[:context_size], M[:context_size], t[:context_size], y[:context_size]
        X_q, M_q = X[context_size:], M[context_size:]
        t0_q = ep["true_y0"][context_size:]
        t1_q = ep["true_y1"][context_size:]
        tcate_q = ep["true_ite"][context_size:]
        te_q = ep["true_propensity"][context_size:]

        X_ctx_t = torch.from_numpy(X_ctx).to(device)
        M_ctx_t = torch.from_numpy(M_ctx).to(device)
        t_ctx_t = torch.from_numpy(t_ctx).to(device)
        y_ctx_t = torch.from_numpy(y_ctx).to(device)
        X_q_t = torch.from_numpy(X_q).to(device)
        M_q_t = torch.from_numpy(M_q).to(device)

        outcome_logits, prop_logits, scale, mem = model(
            X_ctx_t, t_ctx_t, y_ctx_t, X_q_t, M_ctx=M_ctx_t, M_q=M_q_t
        )
        mean_risk, _ = model.risk_stats(outcome_logits, scale)
        p0, p1 = mean_risk.cpu().unbind(dim=1)
        p0_np, p1_np = p0.numpy(), p1.numpy()
        cate_np = p1_np - p0_np
        e_np = prop_logits.float().sigmoid().cpu().numpy()

        all_p0.append(p0_np)
        all_p1.append(p1_np)
        all_t0.append(t0_q)
        all_t1.append(t1_q)
        all_cate.append(cate_np)
        all_tcate.append(tcate_q)
        all_e.append(e_np)
        all_te.append(te_q)

        fam = str(ep["meta"].get("family", "sparse_oci"))
        if not ep["meta"]["is_null_hte"] and cate_np.std() > 1e-8 and tcate_q.std() > 1e-8:
            r_ep = float(np.corrcoef(cate_np, tcate_q)[0, 1])
            within_rs.append(r_ep)
            fam_wr.setdefault(fam, []).append(r_ep)
        ate_errs.append(float(cate_np.mean() - tcate_q.mean()))
        err_sq = (cate_np - tcate_q) ** 2
        fam_sq.setdefault(fam, []).append(err_sq)
        if ep["meta"]["is_null_hte"]:
            null_cates.append(err_sq)
        else:
            non_null_cates.append(err_sq)

        # Role detection AUROC & instrument suppression diagnostic
        role_probs = F.softmax(mem.role_logits.float(), dim=-1).cpu().numpy()
        p_signal = 1.0 - role_probs[:, 0]
        auc_s = _auroc_binary((roles != 0).astype(np.float64), p_signal)
        if math.isfinite(auc_s):
            role_aucs.append(auc_s)

        p_total = mem.n_features
        full_ge = np.zeros(p_total, dtype=np.float32)
        full_ge[mem.nuis_idx.cpu().numpy()] = torch.maximum(mem.g_er_lin, mem.g_er_nl).cpu().numpy()
        conf_mask = (roles == ROLE_TO_ID["confounder"]) | (roles == ROLE_TO_ID["dual_role"])
        inst_mask = roles == ROLE_TO_ID["instrument"]
        if conf_mask.any():
            conf_ge.append(float(full_ge[conf_mask].mean()))
        if inst_mask.any():
            inst_ge.append(float(full_ge[inst_mask].mean()))

        # Arm-mean baseline
        y0_bar = float(y_ctx[t_ctx < 0.5].mean()) if (t_ctx < 0.5).any() else float(y_ctx.mean())
        y1_bar = float(y_ctx[t_ctx > 0.5].mean()) if (t_ctx > 0.5).any() else float(y_ctx.mean())
        arm_ep = np.full_like(tcate_q, y1_bar - y0_bar)
        all_arm_cate.append(arm_ep)
        fam_arm_sq.setdefault(fam, []).append((arm_ep - tcate_q) ** 2)

        if compute_baselines:
            # Dense Ridge only when P <= 600 (keep eval snappy); screened top-32 always runs
            if X_ctx.shape[1] <= 600:
                all_ridge_cate.append(_baseline_r_learner(X_ctx, t_ctx, y_ctx, X_q, screen_k=None))
            all_screen_cate.append(_baseline_r_learner(X_ctx, t_ctx, y_ctx, X_q, screen_k=32))

        if studies >= 50 and idx_ep % 25 == 0:
            print(f"[tabular-eval] {idx_ep}/{studies} studies done", file=sys.stderr, flush=True)

    if was_training:
        model.train()

    p0_a = np.concatenate(all_p0)
    p1_a = np.concatenate(all_p1)
    t0_a = np.concatenate(all_t0)
    t1_a = np.concatenate(all_t1)
    cate_a = np.concatenate(all_cate)
    tcate_a = np.concatenate(all_tcate)
    e_a = np.concatenate(all_e)
    te_a = np.concatenate(all_te)
    arm_a = np.concatenate(all_arm_cate)

    out = {
        "studies": studies,
        "context_size": context_size,
        "num_features": num_features if num_features is not None else "log_uniform",
        "cate_rmse": float(np.sqrt(np.mean((cate_a - tcate_a) ** 2))),
        "risk_rmse": float(np.sqrt(0.5 * (np.mean((p0_a - t0_a) ** 2) + np.mean((p1_a - t1_a) ** 2)))),
        "propensity_rmse": float(np.sqrt(np.mean((e_a - te_a) ** 2))),
        "ate_rmse": float(np.sqrt(np.mean(np.square(ate_errs)))),
        "ate_bias": float(np.mean(ate_errs)),
        "cate_pearson": float(np.corrcoef(cate_a, tcate_a)[0, 1]) if cate_a.std() > 1e-8 else 0.0,
        "cate_spearman": _spearman(cate_a, tcate_a),
        "within_study_cate_pearson": float(np.mean(within_rs)) if within_rs else 0.0,
        "null_hte_cate_rmse": float(np.sqrt(np.mean(np.concatenate(null_cates)))) if null_cates else float("nan"),
        "non_null_cate_rmse": float(np.sqrt(np.mean(np.concatenate(non_null_cates)))) if non_null_cates else float("nan"),
        "role_signal_auroc": float(np.mean(role_aucs)) if role_aucs else float("nan"),
        "confounder_prop_gate": float(np.mean(conf_ge)) if conf_ge else float("nan"),
        "instrument_prop_gate": float(np.mean(inst_ge)) if inst_ge else float("nan"),
        "arm_mean_cate_rmse": float(np.sqrt(np.mean((arm_a - tcate_a) ** 2))),
    }
    if all_screen_cate:
        scr_a = np.concatenate(all_screen_cate)
        out["screened_ridge_r_cate_rmse"] = float(np.sqrt(np.mean((scr_a - tcate_a) ** 2)))
    if all_ridge_cate and len(all_ridge_cate) == len(all_tcate):
        rdg_a = np.concatenate(all_ridge_cate)
        out["dense_ridge_r_cate_rmse"] = float(np.sqrt(np.mean((rdg_a - tcate_a) ** 2)))
    out["by_family"] = {
        fam_name: {
            "studies": len(sq_list),
            "cate_rmse": float(np.sqrt(np.mean(np.concatenate(sq_list)))),
            "arm_mean_cate_rmse": float(np.sqrt(np.mean(np.concatenate(fam_arm_sq[fam_name])))),
            "within_study_cate_pearson": float(np.mean(fam_wr[fam_name])) if fam_wr.get(fam_name) else 0.0,
        }
        for fam_name, sq_list in sorted(fam_sq.items())
    }
    return out


def run_tabular_training(
    run_dir: str | Path,
    steps: int,
    device: str = "cuda",
    *,
    cfg: TabularModelConfig = TabularModelConfig(),
    prior: PriorConfig = PriorConfig(),
    min_context_size: int = 80,
    context_size: int = 3000,
    query_size: int = 64,
    grad_accum_steps: int = 8,
    lr: float = 5e-4,
    weight_decay: float = 1e-2,
    warmup_steps: int = 100,
    save_every: int = 250,
    eval_every: int = 250,
    eval_studies: int = 48,
    eval_context_size: int = 800,
    eval_num_features: int | None = 500,
    num_workers: int = 0,
    seed: int = 42,
    resume: bool = False,
) -> Path:
    """Train or resume SparseTabularCausalPFN on high-P sparse synthetic causal studies."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")

    start_step = 0
    best_cate = float("inf")
    settings = {
        "min_context_size": min_context_size,
        "context_size": context_size,
        "query_size": query_size,
        "grad_accum_steps": grad_accum_steps,
        "lr": lr,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "seed": seed,
    }

    if resume and (run_dir / "checkpoint.pt").is_file():
        saved = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
        cfg = TabularModelConfig(**saved["model"])
        prior = PriorConfig(**saved["prior"])
        start_step = int(saved["step"])
        model = SparseTabularCausalPFN(cfg).to(dev)
        model.load_state_dict(saved["weights"], strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        if saved.get("optimizer") is not None:
            optimizer.load_state_dict(saved["optimizer"])
    else:
        torch.manual_seed(seed)
        model = SparseTabularCausalPFN(cfg).to(dev)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Strictly enforce train split for Nemotron oncology designs during training
    prior = replace(prior, design_split="train")
    prior_dict = asdict(prior)
    rng = np.random.default_rng(seed + start_step * 997)
    metrics_path = run_dir / "metrics.jsonl"
    pool = ProcessPoolExecutor(max_workers=num_workers) if num_workers > 0 else None

    def _lr_for_step(s: int) -> float:
        if warmup_steps > 0 and s <= warmup_steps:
            return lr * (s / warmup_steps)
        prog = (s - warmup_steps) / max(steps - warmup_steps, 1)
        prog = min(max(prog, 0.0), 1.0)
        return lr * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog)))

    def _make_tasks(step_num: int):
        tasks = []
        for ep_i in range(grad_accum_steps):
            # Even seeds for training; log-uniform N_ctx and P across all 3 OCI DGP families
            ep_seed = int(seed * 100_000 + (step_num * grad_accum_steps + ep_i) * 2)
            log_n = rng.uniform(math.log(min_context_size), math.log(context_size))
            n_ctx = int(round(math.exp(log_n)))
            tasks.append((ep_seed, n_ctx, query_size, None, prior_dict, None))
        return tasks

    t_start = time.time()
    try:
        # 1-step async prefetch so CPU high-P episode synthesis overlaps with GPU forward/backward
        pending_futures = None
        if pool is not None and start_step < steps:
            pending_futures = [pool.submit(_sample_episode_worker, t) for t in _make_tasks(start_step + 1)]

        for step in range(start_step + 1, steps + 1):
            cur_lr = _lr_for_step(step)
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            if pool is not None:
                episodes = [f.result() for f in pending_futures]
                if step < steps:
                    pending_futures = [pool.submit(_sample_episode_worker, t) for t in _make_tasks(step + 1)]
                else:
                    pending_futures = None
            else:
                episodes = [_sample_episode_worker(t) for t in _make_tasks(step)]

            model.train()
            optimizer.zero_grad(set_to_none=True)
            acc = {"loss": 0.0, "outcome_loss": 0.0, "cate_loss": 0.0, "propensity_loss": 0.0, "role_loss": 0.0}
            p_list, n_list = [], []

            for ep in episodes:
                n_ctx = len(ep["t"]) - query_size
                p_list.append(ep["meta"]["p"])
                n_list.append(n_ctx)
                X_ctx = torch.from_numpy(ep["X"][:n_ctx]).to(dev)
                M_ctx = torch.from_numpy(ep["M"][:n_ctx]).to(dev)
                t_ctx = torch.from_numpy(ep["t"][:n_ctx]).to(dev)
                y_ctx = torch.from_numpy(ep["y"][:n_ctx]).to(dev)
                X_q = torch.from_numpy(ep["X"][n_ctx:]).to(dev)
                M_q = torch.from_numpy(ep["M"][n_ctx:]).to(dev)
                oracle = torch.from_numpy(
                    np.stack([ep["true_y0"][n_ctx:], ep["true_y1"][n_ctx:]], axis=1)
                ).to(dev)
                t_prop = torch.from_numpy(ep["true_propensity"][n_ctx:]).to(dev)
                roles = torch.from_numpy(ep["roles"]).to(dev)

                out_logits, prop_logits, scale, mem = model(
                    X_ctx, t_ctx, y_ctx, X_q, M_ctx=M_ctx, M_q=M_q
                )
                loss, comps = model.loss_components(
                    out_logits,
                    prop_logits,
                    scale,
                    oracle,
                    t_prop,
                    role_logits=mem.role_logits,
                    nuis_screen_logits=mem.nuis_screen_logits,
                    mod_screen_logits=mem.mod_screen_logits,
                    true_roles=roles,
                )
                if torch.isfinite(loss):
                    (loss / grad_accum_steps).backward()
                    acc["loss"] += float(loss.item()) / grad_accum_steps
                    for k, v in comps.items():
                        acc[k] += float(v.item()) / grad_accum_steps

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            row = {
                "step": step,
                "lr": cur_lr,
                **acc,
                "mean_n_ctx": float(np.mean(n_list)),
                "mean_p": float(np.mean(p_list)),
                "elapsed_s": round(time.time() - t_start, 2),
            }

            if eval_every > 0 and (step % eval_every == 0 or step == steps):
                ev = evaluate_tabular(
                    model,
                    studies=eval_studies,
                    context_size=eval_context_size,
                    query_size=query_size,
                    num_features=eval_num_features,
                    prior=prior,
                    seed=seed + 500_001,
                    compute_baselines=True,
                    num_workers=num_workers,
                )
                for k, v in ev.items():
                    if isinstance(v, (int, float)):
                        row[f"eval_{k}"] = v
                if ev["cate_rmse"] < best_cate:
                    best_cate = ev["cate_rmse"]
                    save_tabular_checkpoint(
                        run_dir / "best_checkpoint.pt", model, optimizer, step, settings, prior
                    )

            with metrics_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")

            if step % 25 == 0 or "eval_cate_rmse" in row or step == steps:
                msg = (
                    f"[step {step:4d}/{steps}] loss={row['loss']:.4f} "
                    f"(out={row['outcome_loss']:.3f} cate={row['cate_loss']:.4f} "
                    f"prop={row['propensity_loss']:.3f} role={row['role_loss']:.3f}) "
                    f"N={row['mean_n_ctx']:.0f} P={row['mean_p']:.0f}"
                )
                if "eval_cate_rmse" in row:
                    msg += (
                        f" | EVAL cate={row['eval_cate_rmse']:.4f} "
                        f"(arm={row['eval_arm_mean_cate_rmse']:.4f} "
                        f"scrR={row.get('eval_screened_ridge_r_cate_rmse', float('nan')):.4f}) "
                        f"wr={row['eval_within_study_cate_pearson']:+.3f} "
                        f"null={row['eval_null_hte_cate_rmse']:.4f} "
                        f"roleAUC={row['eval_role_signal_auroc']:.3f} "
                        f"gE(c/i)={row['eval_confounder_prop_gate']:.2f}/{row['eval_instrument_prop_gate']:.2f}"
                    )
                print(msg, flush=True)

            if save_every > 0 and (step % save_every == 0 or step == steps):
                save_tabular_checkpoint(
                    run_dir / "checkpoint.pt", model, optimizer, step, settings, prior
                )
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    return run_dir / "checkpoint.pt"
