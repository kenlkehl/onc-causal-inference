"""Study sampling, oracle-supervised training, evaluation baselines, and portable checkpoints."""

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict
import fcntl
import gzip
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.stats import pearsonr, spearmanr
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from synthetic_data.longitudinal import draw_longitudinal_dgp, sample_longitudinal_patient
from synthetic_data.study_designs import DEFAULT_BANK, load_bank
from .events import binary_observations, patient_summary_vector, read_patient_table
from .model import (
    BASE_MODEL,
    CHECKPOINT_FORMAT_VERSION,
    PINNED_REVISIONS,
    ArmScale,
    CausalStudyModel,
    ModelConfig,
)

OBSERVED = ["treatment_indicator", "outcome_indicator"]
TARGETS = ["true_y0", "true_y1"]
PROPENSITY_COL = "true_treatment_prob"
DESIGN_BANK_FILENAME = "design_bank.json.gz"


def _select_episode_design(seed, step, held_out, designs, micro_step=0):
    if designs is None:
        return None
    if not designs:
        raise ValueError("The requested study-design split is empty")
    seed_parts = [seed, step] if micro_step == 0 else [seed, step, micro_step]
    index = (
        (step - 1) % len(designs)
        if held_out
        else int(np.random.default_rng([*seed_parts, 721]).integers(len(designs)))
    )
    return designs[index]


def _simulate_for_design(
    design, seed, step, held_out, size, schema_version, micro_step, augment
):
    """Top-level picklable worker for ProcessPoolExecutor episode prefetching."""
    seed_parts = [seed, step] if micro_step == 0 else [seed, step, micro_step]
    state = np.random.SeedSequence(seed_parts).generate_state(2, dtype=np.uint64)
    # Even seeds for training; odd seeds for validation, for both DGPs and patients.
    dgp_seed, patient_seed = [2 * int(s) + int(held_out) for s in state]
    provenance = {"dgp_seed": dgp_seed, "patient_seed": patient_seed}
    if design is not None:
        provenance.update(
            design_id=design["id"],
            question_id=design["question"]["id"],
            cancer=design["question"]["cancer"],
            width=len(design["candidates"]["measurements"])
            + len(design["candidates"]["note_findings"]),
        )
    sim_schema = 1 if schema_version == 1 else 2
    dgp = draw_longitudinal_dgp(
        dgp_seed, "design" if design else "mixed", schema_version=sim_schema, design=design
    )
    omit_absent = 0.0
    style_var = False
    if augment and design is not None:
        aug_rng = np.random.default_rng([*seed_parts, 991])
        # Mix explicit-negation and real-EHR positive-only charting across studies.
        omit_absent = float(aug_rng.uniform(0.0, 0.65)) if aug_rng.random() < 0.7 else 0.0
        style_var = True
    rows = [
        sample_longitudinal_patient(
            i,
            seed=patient_seed,
            dgp=dgp,
            omit_absent_prob=omit_absent,
            style_variation=style_var,
        )
        for i in range(size)
    ]
    return rows, provenance


def simulate_episode(
    seed,
    step,
    held_out,
    size,
    *,
    schema_version=CHECKPOINT_FORMAT_VERSION,
    designs=None,
    micro_step=0,
    augment=False,
):
    design = _select_episode_design(seed, step, held_out, designs, micro_step=micro_step)
    return _simulate_for_design(
        design, seed, step, held_out, size, schema_version, micro_step, augment
    )


class _EpisodePrefetcher:
    """Overlap CPU episode simulation with GPU compute in exact deterministic order."""

    def __init__(self, jobs, *, num_workers=0):
        self._jobs = iter(jobs)
        self._num_workers = int(num_workers)
        self._executor = None
        self._pending = deque()
        self._max_inflight = max(2 * self._num_workers, 4) if self._num_workers > 0 else 0

    def __enter__(self):
        if self._num_workers > 0:
            ctx = multiprocessing.get_context("spawn")
            self._executor = ProcessPoolExecutor(max_workers=self._num_workers, mp_context=ctx)
            self._fill()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        self._pending.clear()

    def _fill(self):
        while len(self._pending) < self._max_inflight:
            try:
                meta, args = next(self._jobs)
            except StopIteration:
                break
            self._pending.append((meta, self._executor.submit(_simulate_for_design, *args)))

    def next_item(self):
        if self._executor is None:
            meta, args = next(self._jobs)
            rows, provenance = _simulate_for_design(*args)
            return meta, rows, provenance
        meta, future = self._pending.popleft()
        self._fill()
        rows, provenance = future.result()
        return meta, rows, provenance


def split_designs(bank, split):
    if bank is None:
        return None
    selected = [d for d in bank["designs"] if d["question"]["split"] == split]
    if not selected:
        raise ValueError(f"Design bank has no {split} designs")
    return selected


def _autocast(device, precision):
    dev = torch.device(device)
    if dev.type == "cuda" and precision in ("auto", "bf16") and torch.cuda.is_bf16_supported():
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def fit_episode(
    model,
    rows,
    context_size,
    *,
    precision="fp32",
    note_grad_patients=None,
    freeze_note_backbone=False,
):
    device = model.centers.device
    query_rows = rows[context_size:]
    oracle = torch.tensor(
        [[r[k] for k in TARGETS] for r in query_rows], device=device, dtype=torch.float32
    )
    true_prop = None
    if all(r.get(PROPENSITY_COL) is not None for r in query_rows):
        true_prop = torch.tensor(
            [r[PROPENSITY_COL] for r in query_rows], device=device, dtype=torch.float32
        )
    with _autocast(device, precision):
        outcome_logits, prop_logits, scale = model(
            rows[:context_size],
            query_rows,
            note_grad_patients=note_grad_patients,
            freeze_note_backbone=freeze_note_backbone,
        )
        loss, parts = model.loss_components(
            outcome_logits,
            scale,
            oracle,
            propensity_logits=prop_logits,
            true_propensity=true_prop,
        )
    risks = model.risks(outcome_logits, scale)
    prop_probs = prop_logits.float().sigmoid()
    return loss, risks, oracle, prop_probs, true_prop, parts


def _load_backbone(config):
    spec = dict(config)
    return AutoModel.from_config(AutoConfig.for_model(spec.pop("model_type"), **spec))


def _write_gzip_json(path, payload):
    path = Path(path)
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    digest = hashlib.sha256(raw).hexdigest()
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".tmp") as handle:
        tmp = Path(handle.name)
    try:
        with tmp.open("wb") as out, gzip.GzipFile(filename="", mode="wb", fileobj=out, mtime=0) as gz:
            gz.write(raw)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return digest


def load_run_design_bank(directory, saved):
    """Load the study design bank stored beside (or embedded inside) a checkpoint."""
    settings = saved.get("settings") or {}
    if "design_bank_file" in settings and settings["design_bank_file"] is not None:
        bank_path = Path(directory) / settings["design_bank_file"]
        with gzip.open(bank_path, "rb") as handle:
            raw = handle.read()
        expected = settings.get("design_bank_sha256")
        if expected and hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Design bank checksum mismatch: {bank_path}")
        return load_bank(bank_path)
    return settings.get("design_bank")


def load_run(directory, device, *, note_batch_size=None):
    directory = Path(directory)
    saved = torch.load(
        directory / "checkpoint.pt", map_location="cpu", mmap=True, weights_only=True
    )
    if saved["format_version"] not in (1, CHECKPOINT_FORMAT_VERSION):
        raise ValueError(
            f"Unsupported longitudinal checkpoint format {saved['format_version']}; "
            f"expected 1 or {CHECKPOINT_FORMAT_VERSION}"
        )
    tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer", local_files_only=True)
    cfg_dict = dict(saved["model"])
    if note_batch_size is not None:
        cfg_dict["note_batch_size"] = note_batch_size
    model = CausalStudyModel(
        _load_backbone(saved["backbone"]),
        tokenizer,
        ModelConfig(**cfg_dict),
        include_pros=saved["format_version"] != 1,
    )
    model.load_state_dict(saved["weights"])
    return model.to(device), saved


def _save(directory, model, optimizer, settings, step, *, filename="checkpoint.pt"):
    payload = dict(
        format_version=model.schema_version,
        backbone=model.patient.backbone.config.to_dict(),
        model=asdict(model.cfg),
        settings=settings,
        step=step,
        weights=model.state_dict(),
        optimizer=optimizer.state_dict(),
        rng_state=torch.get_rng_state(),
        cuda_rng_state=(
            torch.cuda.get_rng_state(model.centers.device) if model.centers.is_cuda else None
        ),
    )
    with tempfile.NamedTemporaryFile(dir=directory, delete=False, suffix=".tmp") as handle:
        tmp = Path(handle.name)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, directory / filename)
    finally:
        tmp.unlink(missing_ok=True)


def _is_no_decay(name, param):
    if param.ndim < 2:
        return True
    lowered = name.lower()
    return any(
        token in lowered
        for token in (
            "bias",
            "norm",
            "embed",
            "codes",
            "extra_labs",
            "extra_pros",
            "note_queries",
            "note_slot_time_q",
            ".summary",
        )
    )


def _sample_context_size(rng, min_context, max_context, sampling):
    """Sample a study context size in ``[min_context, max_context]``."""
    if min_context >= max_context:
        return max_context
    if sampling == "uniform":
        return int(rng.integers(min_context, max_context + 1))
    if sampling == "log_uniform":
        u = float(rng.uniform(math.log(min_context), math.log(max_context)))
        return int(min(max_context, max(min_context, round(math.exp(u)))))
    raise ValueError(f"Unknown context_size_sampling: {sampling}")


def _optimizer(model, settings):
    backbone_decay, backbone_no_decay = [], []
    new_decay, new_no_decay = [], []
    backbone_wd = float(settings.get("backbone_weight_decay", 0.0))
    head_wd = float(settings.get("weight_decay", 0.01))
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_backbone = name.startswith("patient.backbone.")
        no_decay = _is_no_decay(name, param)
        if is_backbone:
            (backbone_no_decay if no_decay else backbone_decay).append(param)
        else:
            (new_no_decay if no_decay else new_decay).append(param)
    groups = [
        {"params": backbone_decay, "lr": settings["note_lr"], "weight_decay": backbone_wd},
        {"params": backbone_no_decay, "lr": settings["note_lr"], "weight_decay": 0.0},
        {"params": new_decay, "lr": settings["lr"], "weight_decay": head_wd},
        {"params": new_no_decay, "lr": settings["lr"], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups)


def _build_scheduler(optimizer, settings, steps, start):
    kind = settings.get("lr_scheduler", "constant")
    warmup = int(settings.get("warmup_steps", 0))
    if kind == "constant":
        return None
    settings["lr_schedule_steps"] = steps
    if kind == "linear":
        return get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup, num_training_steps=steps, last_epoch=start - 1
        )
    if kind == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup, num_training_steps=steps, last_epoch=start - 1
        )
    raise ValueError(f"Unknown lr_scheduler: {kind}")


def run_training(
    directory, steps, device, save_every, *, settings=None, cfg=None, resume_overrides=None
):
    directory = Path(directory)
    if settings is not None:
        directory.mkdir(parents=True, exist_ok=True)
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(directory_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _train_locked(directory, steps, device, save_every, settings, cfg, resume_overrides)
    finally:
        os.close(directory_fd)


def _load_cohort_tables(paths, streams, min_rows):
    tables = []
    fingerprints = {}
    for path in paths:
        with open(path, "rb") as source:
            fingerprints[path] = hashlib.file_digest(source, "sha256").hexdigest()
        # Include propensity column when available; fall back cleanly if absent.
        try:
            table = read_patient_table(path, streams, [*OBSERVED, *TARGETS, PROPENSITY_COL])
        except Exception:
            table = read_patient_table(path, streams, [*OBSERVED, *TARGETS])
        ids = table["patient_id"]
        if table.num_rows < min_rows or ids.null_count or len(set(ids.to_pylist())) != table.num_rows:
            raise ValueError(
                f"Training cohort must contain at least {min_rows} unique patients: {path}"
            )
        tables.append(table)
    return tables, fingerprints


def _train_locked(directory, steps, device, save_every, settings, cfg, resume_overrides=None):
    if min(steps, save_every) < 1:
        raise ValueError("steps and save_every must be positive")
    fresh = settings is not None
    bank = None
    if not fresh:
        model, saved = load_run(directory, device)
        settings, start = dict(saved["settings"]), saved["step"]
        lr_overridden = False
        if resume_overrides:
            old_ctx = settings["context_size"]
            for key, value in resume_overrides.items():
                if value is not None:
                    if key in ("lr", "note_lr"):
                        lr_overridden = True
                    settings[key] = value
            # Keep min_context_size aligned if context_size changed without an explicit lower bound.
            if (
                resume_overrides.get("context_size") is not None
                and resume_overrides.get("min_context_size") is None
                and (
                    settings.get("min_context_size") is None
                    or settings.get("min_context_size") == old_ctx
                )
            ):
                settings["min_context_size"] = settings["context_size"]
        bank = load_run_design_bank(directory, saved)
        # Never re-embed large bank dictionaries inside checkpoint.pt.
        settings.pop("design_bank", None)
        optimizer = _optimizer(model, settings)
        optimizer.load_state_dict(saved["optimizer"])
        if lr_overridden:
            for idx, group in enumerate(optimizer.param_groups):
                target_lr = settings["note_lr"] if idx < 2 else settings["lr"]
                group["lr"] = target_lr
                group["initial_lr"] = target_lr
        if "rng_state" in saved:
            torch.set_rng_state(saved["rng_state"])
        if model.centers.is_cuda and saved.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(saved["cuda_rng_state"], model.centers.device)
        del saved
        if steps <= start:
            raise ValueError(f"Requested total steps must exceed completed step {start}")
    else:
        settings = dict(settings)
        bank = settings.pop("design_bank", None)
        use_default_bank = not settings["cohorts"] and not settings.get("mixed_prior", False)
        if bank is None and use_default_bank:
            bank = load_bank(DEFAULT_BANK)
        min_ctx = settings.get("min_context_size")
        if min_ctx is None:
            min_ctx = settings["context_size"]
        if (
            min(
                settings["context_size"],
                min_ctx,
                settings["query_size"],
                settings["lr"],
                settings["note_lr"],
                settings.get("grad_accum_steps", 1),
            )
            <= 0
            or min_ctx > settings["context_size"]
            or settings.get("warmup_steps", 0) < 0
            or settings.get("num_workers", 0) < 0
            or (
                settings.get("note_grad_patients") is not None
                and settings["note_grad_patients"] < 0
            )
        ):
            raise ValueError(
                "Invalid training sizes, accumulation, warmup, workers, note_grad_patients, or LRs"
            )
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Use a fresh run directory or resume: {directory}")
        torch.manual_seed(settings["seed"])
        revision = PINNED_REVISIONS.get(settings["note_model"])
        tokenizer = AutoTokenizer.from_pretrained(settings["note_model"], revision=revision)
        backbone = AutoModel.from_pretrained(
            settings["note_model"], revision=revision, dtype=torch.float32
        )
        model = CausalStudyModel(backbone, tokenizer, cfg).to(device)
        optimizer = _optimizer(model, settings)
        start = 0

    max_context = int(settings["context_size"])
    min_context = int(settings.get("min_context_size") or max_context)
    context_sampling = settings.get("context_size_sampling", "uniform")
    query_size = int(settings["query_size"])
    grad_accum = int(settings.get("grad_accum_steps", 1))
    precision = settings.get("precision", "auto")
    augment = bool(settings.get("augment_notes", False))
    eval_every = int(settings.get("eval_every", 0))
    eval_studies = int(settings.get("eval_studies", 16))
    eval_context = int(settings.get("eval_context_size") or max_context)
    best_eval_rmse = settings.get("best_eval_cate_rmse")
    note_grad_patients = settings.get("note_grad_patients")
    freeze_note_backbone = bool(settings.get("freeze_note_backbone", False))
    num_workers = int(settings.get("num_workers", 0))
    if (
        min(
            max_context,
            min_context,
            eval_context,
            query_size,
            grad_accum,
            settings["lr"],
            settings["note_lr"],
        )
        <= 0
        or min_context > max_context
        or context_sampling not in ("uniform", "log_uniform")
        or eval_every < 0
        or eval_studies < 1
        or num_workers < 0
        or (note_grad_patients is not None and note_grad_patients < 0)
    ):
        raise ValueError(
            "Invalid context/query sizes, sampling mode, grad_accum_steps, eval settings, LRs, or workers"
        )

    designs = split_designs(bank, "train")
    if designs and settings["cohorts"]:
        raise ValueError("Choose a design bank or materialized training cohorts")
    tables, fingerprints = _load_cohort_tables(
        settings["cohorts"], model.patient.streams, max_context + query_size
    )
    if not fresh and fingerprints != settings.get("cohort_sha256", {}):
        raise ValueError("Training cohort contents have changed since the checkpoint")

    scheduler = _build_scheduler(optimizer, settings, steps, start)
    if fresh:
        settings["cohort_sha256"] = fingerprints
        if bank is not None:
            settings["design_bank_sha256"] = _write_gzip_json(
                directory / DESIGN_BANK_FILENAME, bank
            )
            settings["design_bank_file"] = DESIGN_BANK_FILENAME
        else:
            settings["design_bank_file"] = None
        model.patient.tokenizer.save_pretrained(directory / "tokenizer")
        _save(directory, model, optimizer, settings, step=0)

    def _set_train_modes():
        model.train()
        if freeze_note_backbone:
            model.patient.backbone.eval()

    def _episode_jobs():
        for step in range(start + 1, steps + 1):
            for micro in range(grad_accum):
                if min_context < max_context:
                    ctx_rng = np.random.default_rng([settings["seed"], step, micro, 313])
                    n_context = _sample_context_size(
                        ctx_rng, min_context, max_context, context_sampling
                    )
                else:
                    n_context = max_context
                design = _select_episode_design(
                    settings["seed"], step, False, designs, micro_step=micro
                )
                yield (
                    (step, micro, n_context),
                    (
                        design,
                        settings["seed"],
                        step,
                        False,
                        n_context + query_size,
                        model.schema_version,
                        micro,
                        augment,
                    ),
                )

    metrics_path = directory / "metrics.jsonl"
    _set_train_modes()
    prefetch_cm = (
        _EpisodePrefetcher(_episode_jobs(), num_workers=num_workers)
        if not tables
        else nullcontext(None)
    )
    with prefetch_cm as prefetcher:
        for step in range(start + 1, steps + 1):
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            outcome_sum = cate_sum = prop_sum = 0.0
            last_provenance = {}
            last_ctx = max_context
            for micro in range(grad_accum):
                if tables:
                    if min_context < max_context:
                        ctx_rng = np.random.default_rng([settings["seed"], step, micro, 313])
                        n_context = _sample_context_size(
                            ctx_rng, min_context, max_context, context_sampling
                        )
                    else:
                        n_context = max_context
                    size = n_context + query_size
                    seed_parts = (
                        [settings["seed"], step] if micro == 0 else [settings["seed"], step, micro]
                    )
                    rng = np.random.default_rng(seed_parts)
                    idx = int(rng.integers(len(tables)))
                    rows = (
                        tables[idx]
                        .take(rng.choice(len(tables[idx]), size, replace=False))
                        .to_pylist()
                    )
                    provenance = {"cohort": settings["cohorts"][idx]}
                else:
                    (_, _, n_context), rows, provenance = prefetcher.next_item()
                last_ctx = n_context
                last_provenance = provenance
                loss, _, _, _, _, parts = fit_episode(
                    model,
                    rows,
                    n_context,
                    precision=precision,
                    note_grad_patients=note_grad_patients,
                    freeze_note_backbone=freeze_note_backbone,
                )
                (loss / grad_accum).backward()
                total_loss += loss.item() / grad_accum
                outcome_sum += parts["outcome_loss"].item() / grad_accum
                cate_sum += parts["cate_loss"].item() / grad_accum
                prop_sum += parts["propensity_loss"].item() / grad_accum

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            note_lr = optimizer.param_groups[0]["lr"]
            lr = optimizer.param_groups[2]["lr"]
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            record = {
                "step": step,
                "loss": total_loss,
                "outcome_loss": outcome_sum,
                "cate_loss": cate_sum,
                "propensity_loss": prop_sum,
                "context_size": last_ctx,
                "note_lr": 0.0 if freeze_note_backbone else note_lr,
                "lr": lr,
                "freeze_note_backbone": freeze_note_backbone,
                **last_provenance,
            }
            if eval_every > 0 and (step % eval_every == 0 or step == steps) and not tables:
                val = evaluate(
                    model,
                    settings["seed"],
                    eval_studies,
                    eval_context,
                    query_size,
                    bank=bank,
                    precision=precision,
                    num_workers=num_workers,
                )
                _set_train_modes()
                record["eval_context_size"] = eval_context
                record["eval_cate_rmse"] = val["cate_rmse"]
                record["eval_risk_rmse"] = val["risk_rmse"]
                record["eval_ate_rmse"] = val["ate_rmse"]
                record["eval_cate_pearson"] = val["cate_pearson"]
                record["eval_within_study_cate_pearson"] = val["mean_within_study_cate_pearson"]
                record["eval_propensity_rmse"] = val.get("propensity_rmse")
                record["eval_arm_mean_cate_rmse"] = val["arm_mean_cate_rmse"]
                record["eval_zero_cate_rmse"] = val["zero_cate_rmse"]
                record["eval_linear_t_learner_cate_rmse"] = val["linear_t_learner_cate_rmse"]
                if best_eval_rmse is None or val["cate_rmse"] < best_eval_rmse:
                    best_eval_rmse = val["cate_rmse"]
                    settings["best_eval_cate_rmse"] = best_eval_rmse
                    _save(
                        directory, model, optimizer, settings, step, filename="best_checkpoint.pt"
                    )

            line = json.dumps(record)
            with metrics_path.open("a", encoding="utf-8") as log_file:
                log_file.write(line + "\n")
            print(line, flush=True)
            if step == 1 or step % save_every == 0 or step == steps:
                _save(directory, model, optimizer, settings, step)


def _ridge_t_learner_cate(
    context_rows, query_rows, *, alphas=(1.0, 10.0, 100.0, 1000.0, 10000.0)
):
    """Observational LOO-CV Ridge T-learner baseline on structured patient summaries.

    Selects per-arm regularization strictly by closed-form leave-one-out cross-validation
    on observed context (X, T, Y) with zero oracle information.
    """
    x_ctx = np.asarray([patient_summary_vector(r) for r in context_rows], dtype=np.float64)
    x_q = np.asarray([patient_summary_vector(r) for r in query_rows], dtype=np.float64)
    t_ctx = np.asarray([r["treatment_indicator"] for r in context_rows], dtype=np.int64)
    y_ctx = np.asarray([r["outcome_indicator"] for r in context_rows], dtype=np.float64)

    mean = x_ctx.mean(axis=0)
    std = x_ctx.std(axis=0)
    std = np.where(std > 1e-6, std, 1.0)
    z_ctx = (x_ctx - mean) / std
    z_q = (x_q - mean) / std
    d = z_ctx.shape[1]
    eye = np.eye(d + 1, dtype=np.float64)
    eye[0, 0] = 0.0  # do not penalize intercept
    xq = np.hstack([np.ones((len(query_rows), 1)), z_q])

    preds = []
    for arm in (0, 1):
        mask = t_ctx == arm
        n_arm = int(mask.sum())
        if n_arm == 0:
            preds.append(np.full(len(query_rows), 0.5, dtype=np.float64))
            continue
        if n_arm < 3:
            rate = (y_ctx[mask].sum() + 0.5) / (n_arm + 1.0)
            preds.append(np.full(len(query_rows), rate, dtype=np.float64))
            continue
        xa = np.hstack([np.ones((n_arm, 1)), z_ctx[mask]])
        ya = y_ctx[mask]
        gram = xa.T @ xa
        rhs = xa.T @ ya
        best_loo = float("inf")
        best_coef = None
        for alpha in alphas:
            reg = alpha * eye
            reg[0, 0] = 1e-6
            chol = np.linalg.cholesky(gram + reg)
            coef = cho_solve((chol, True), rhs)
            v = solve_triangular(chol, xa.T, lower=True)
            h = np.sum(v * v, axis=0)
            denom = np.maximum(1.0 - h, 1e-3)
            loo_mse = float(np.mean(((ya - xa @ coef) / denom) ** 2))
            if loo_mse < best_loo:
                best_loo = loo_mse
                best_coef = coef
        preds.append(np.clip(xq @ best_coef, 0.0, 1.0))
    return preds[1] - preds[0]


def _safe_corr(a, b, fn):
    if len(a) < 2 or float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return None
    val = float(fn(a, b)[0])
    return val if math.isfinite(val) else None


def _width_bucket(width):
    if width <= 20:
        return "3-20"
    if width <= 50:
        return "21-50"
    if width <= 80:
        return "51-80"
    return "81-100"


def _aggregate_eval(records):
    pred_ites = np.concatenate([r["pred_ite"] for r in records])
    true_ites = np.concatenate([r["true_ite"] for r in records])
    arm_errors = np.concatenate([r["arm_err"] for r in records])
    arm_mean_errs = np.concatenate([r["arm_mean_err"] for r in records])
    ridge_errs = np.concatenate([r["ridge_err"] for r in records])
    ate_errs = np.array([float(np.mean(r["pred_ite"]) - np.mean(r["true_ite"])) for r in records])
    within_corrs = [
        c for r in records if (c := _safe_corr(r["pred_ite"], r["true_ite"], pearsonr)) is not None
    ]
    prop_errs = [r["prop_err"] for r in records if r["prop_err"] is not None]
    out = dict(
        studies=len(records),
        query_patients=int(len(true_ites)),
        risk_rmse=float(np.sqrt(np.mean(arm_errors**2))),
        risk_brier=float(np.mean(arm_errors**2)),
        cate_rmse=float(np.sqrt(np.mean((pred_ites - true_ites) ** 2))),
        ate_rmse=float(np.sqrt(np.mean(ate_errs**2))),
        ate_bias=float(np.mean(ate_errs)),
        cate_pearson=_safe_corr(pred_ites, true_ites, pearsonr),
        cate_spearman=_safe_corr(pred_ites, true_ites, spearmanr),
        mean_within_study_cate_pearson=float(np.mean(within_corrs)) if within_corrs else None,
        arm_mean_cate_rmse=float(np.sqrt(np.mean(arm_mean_errs**2))),
        zero_cate_rmse=float(np.sqrt(np.mean(true_ites**2))),
        linear_t_learner_cate_rmse=float(np.sqrt(np.mean(ridge_errs**2))),
    )
    if prop_errs:
        all_pe = np.concatenate(prop_errs)
        out["propensity_rmse"] = float(np.sqrt(np.mean(all_pe**2)))
    return out


def _score_episode(model, rows, context_size, *, precision="auto"):
    _, pred, truth, prop_pred, prop_true, _ = fit_episode(
        model, rows, context_size, precision=precision
    )
    t, y = binary_observations(rows[:context_size], model.centers.device)
    rates = ArmScale.from_context(t, y).rate
    pred_np = pred.detach().cpu().numpy()
    truth_np = truth.detach().cpu().numpy()
    pred_ite = pred_np[:, 1] - pred_np[:, 0]
    true_ite = truth_np[:, 1] - truth_np[:, 0]
    arm_mean_ite = float((rates[1] - rates[0]).item())
    ridge_ite = _ridge_t_learner_cate(rows[:context_size], rows[context_size:])
    prop_err = None
    if prop_true is not None:
        prop_err = (prop_pred - prop_true).detach().cpu().numpy()
    return dict(
        pred_ite=pred_ite,
        true_ite=true_ite,
        arm_err=(pred_np - truth_np).ravel(),
        arm_mean_err=arm_mean_ite - true_ite,
        ridge_err=ridge_ite - true_ite,
        prop_err=prop_err,
    )


@torch.inference_mode()
def evaluate(
    model, seed, studies, context_size, query_size, *, bank=None, precision="auto", num_workers=0
):
    if min(studies, context_size, query_size) < 1 or num_workers < 0:
        raise ValueError("Evaluation study/patient counts must be positive and workers nonnegative")
    model.eval()
    designs = split_designs(bank, "held_out")
    train_cancers = (
        {d["question"]["cancer"] for d in bank["designs"] if d["question"]["split"] == "train"}
        if bank is not None
        else set()
    )
    records = []
    by_design = {}
    by_width = {}
    by_gen = {}

    def _eval_jobs():
        for step in range(1, studies + 1):
            design = _select_episode_design(seed, step, True, designs)
            yield (
                step,
                (
                    design,
                    seed,
                    step,
                    True,
                    context_size + query_size,
                    model.schema_version,
                    0,
                    False,
                ),
            )

    progress_every = max(studies // 20, 25) if studies >= 50 else 0
    t_eval_start = time.monotonic()
    with _EpisodePrefetcher(_eval_jobs(), num_workers=num_workers) as prefetcher:
        for done_idx in range(1, studies + 1):
            _, rows, provenance = prefetcher.next_item()
            rec = _score_episode(model, rows, context_size, precision=precision)
            records.append(rec)
            if designs:
                by_design.setdefault(provenance["design_id"], []).append(rec)
                by_width.setdefault(_width_bucket(provenance["width"]), []).append(rec)
                gen_key = (
                    "seen_cancer_new_question"
                    if provenance["cancer"] in train_cancers
                    else "unseen_cancer"
                )
                by_gen.setdefault(gen_key, []).append(rec)
            if progress_every and (done_idx % progress_every == 0 or done_idx == studies):
                elapsed = time.monotonic() - t_eval_start
                eta = elapsed * (studies - done_idx) / done_idx
                print(
                    f"[evaluate] {done_idx}/{studies} studies "
                    f"({elapsed / 60.0:.1f}m elapsed, ~{eta / 60.0:.1f}m left)",
                    file=sys.stderr,
                    flush=True,
                )
    result = _aggregate_eval(records)
    if designs:
        result["by_width"] = {k: _aggregate_eval(v) for k, v in sorted(by_width.items())}
        result["by_generalization"] = {k: _aggregate_eval(v) for k, v in sorted(by_gen.items())}
        result["held_out_designs"] = {
            key: dict(
                studies=len(vals),
                cate_rmse=float(
                    np.sqrt(
                        np.mean(
                            np.concatenate([r["pred_ite"] - r["true_ite"] for r in vals]) ** 2
                        )
                    )
                ),
            )
            for key, vals in by_design.items()
        }
    return result


@torch.inference_mode()
def evaluate_cohorts(
    model, cohort_paths, seed, studies, context_size, query_size, *, precision="auto"
):
    """Evaluate a trained model on materialized Parquet cohorts (e.g., LLM-rendered studies)."""
    if min(studies, context_size, query_size) < 1 or not cohort_paths:
        raise ValueError("Provide at least one cohort and positive study/patient counts")
    model.eval()
    tables, _ = _load_cohort_tables(
        [str(Path(p).resolve()) for p in cohort_paths],
        model.patient.streams,
        context_size + query_size,
    )
    records = []
    by_cohort = {}
    size = context_size + query_size
    for step in range(1, studies + 1):
        rng = np.random.default_rng([seed, step, 509])
        idx = (step - 1) % len(tables)
        rows = tables[idx].take(rng.choice(len(tables[idx]), size, replace=False)).to_pylist()
        rec = _score_episode(model, rows, context_size, precision=precision)
        records.append(rec)
        by_cohort.setdefault(str(cohort_paths[idx]), []).append(rec)
    result = _aggregate_eval(records)
    result["cohorts"] = {k: _aggregate_eval(v) for k, v in by_cohort.items()}
    return result
