"""Study sampling, oracle-supervised training, and portable checkpoints."""

from dataclasses import asdict
import fcntl
import hashlib
from pathlib import Path
import json
import os
import tempfile

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from synthetic_data.longitudinal import draw_longitudinal_dgp, sample_longitudinal_patient
from synthetic_data.study_designs import DEFAULT_BANK, load_bank
from .events import binary_observations, read_patient_table
from .model import BASE_MODEL, BASE_REVISION, ArmScale, CausalStudyModel, ModelConfig

OBSERVED = ["treatment_indicator", "outcome_indicator"]
TARGETS = ["true_y0", "true_y1"]


def simulate_episode(seed, step, held_out, size, *, schema_version=2, designs=None):
    state = np.random.SeedSequence([seed, step]).generate_state(2, dtype=np.uint64)
    # Even seeds for training; odd seeds for validation, for both DGPs and patients.
    dgp_seed, patient_seed = [2 * int(s) + int(held_out) for s in state]
    provenance = {"dgp_seed": dgp_seed, "patient_seed": patient_seed}
    design = None
    if designs is not None:
        if not designs:
            raise ValueError("The requested study-design split is empty")
        index = (
            (step - 1) % len(designs)
            if held_out
            else int(np.random.default_rng([seed, step, 721]).integers(len(designs)))
        )
        design = designs[index]
        provenance.update(design_id=design["id"], question_id=design["question"]["id"])
    dgp = draw_longitudinal_dgp(
        dgp_seed, "design" if design else "mixed", schema_version=schema_version, design=design
    )
    rows = [sample_longitudinal_patient(i, seed=patient_seed, dgp=dgp) for i in range(size)]
    return rows, provenance


def split_designs(bank, split):
    if bank is None:
        return None
    selected = [d for d in bank["designs"] if d["question"]["split"] == split]
    if not selected:
        raise ValueError(f"Design bank has no {split} designs")
    return selected


def fit_episode(model, rows, context_size):
    device = model.centers.device
    oracle = torch.tensor(
        [[r[k] for k in TARGETS] for r in rows[context_size:]], device=device, dtype=torch.float32
    )
    logits, scale = model(rows[:context_size], rows[context_size:])
    return model.loss(logits, scale, oracle), model.risks(logits, scale), oracle


def _load_backbone(config):
    spec = dict(config)
    return AutoModel.from_config(AutoConfig.for_model(spec.pop("model_type"), **spec))


def load_run(directory, device):
    directory = Path(directory)
    saved = torch.load(
        directory / "checkpoint.pt", map_location="cpu", mmap=True, weights_only=True
    )
    if saved["format_version"] not in (1, 2):
        raise ValueError("Unknown longitudinal checkpoint format")
    tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer", local_files_only=True)
    model = CausalStudyModel(
        _load_backbone(saved["backbone"]),
        tokenizer,
        ModelConfig(**saved["model"]),
        include_pros=saved["format_version"] == 2,
    )
    model.load_state_dict(saved["weights"])
    return model.to(device), saved


def _save(directory, model, optimizer, settings, step):
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
        os.replace(tmp, directory / "checkpoint.pt")
    finally:
        tmp.unlink(missing_ok=True)


def _optimizer(model, settings):
    backbone = list(model.patient.backbone.parameters())
    ids = {id(p) for p in backbone}
    new = [p for p in model.parameters() if id(p) not in ids]
    return torch.optim.AdamW(
        [{"params": backbone, "lr": settings["note_lr"]}, {"params": new, "lr": settings["lr"]}]
    )


def run_training(directory, steps, device, save_every, *, settings=None, cfg=None):
    directory = Path(directory)
    if settings is not None:
        directory.mkdir(parents=True, exist_ok=True)
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(directory_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _train_locked(directory, steps, device, save_every, settings, cfg)
    finally:
        os.close(directory_fd)


def _train_locked(directory, steps, device, save_every, settings, cfg):
    if min(steps, save_every) < 1:
        raise ValueError("steps and save_every must be positive")
    fresh = settings is not None
    if not fresh:
        model, saved = load_run(directory, device)
        settings, start = saved["settings"], saved["step"]
        optimizer = _optimizer(model, settings)
        optimizer.load_state_dict(saved["optimizer"])
        if "rng_state" in saved:
            torch.set_rng_state(saved["rng_state"])
        if model.centers.is_cuda and saved.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(saved["cuda_rng_state"], model.centers.device)
        del saved
        if steps <= start:
            raise ValueError(f"Requested total steps must exceed completed step {start}")
    else:
        if not settings["cohorts"] and "design_bank" not in settings:
            settings["design_bank"] = load_bank(DEFAULT_BANK)
        if (
            min(
                settings["context_size"],
                settings["query_size"],
                settings["lr"],
                settings["note_lr"],
            )
            <= 0
        ):
            raise ValueError("Training sizes and learning rates must be positive")
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Use a fresh run directory or resume: {directory}")
        torch.manual_seed(settings["seed"])
        revision = BASE_REVISION if settings["note_model"] == BASE_MODEL else None
        tokenizer = AutoTokenizer.from_pretrained(settings["note_model"], revision=revision)
        backbone = AutoModel.from_pretrained(
            settings["note_model"], revision=revision, dtype=torch.float32
        )
        model = CausalStudyModel(backbone, tokenizer, cfg).to(device)
        optimizer = _optimizer(model, settings)
        start = 0
    n_context = settings["context_size"]
    size = n_context + settings["query_size"]
    designs = split_designs(settings.get("design_bank"), "train")
    if designs and settings["cohorts"]:
        raise ValueError("Choose a design bank or materialized training cohorts")
    tables = []
    fingerprints = {}
    for path in settings["cohorts"]:
        with open(path, "rb") as source:
            fingerprints[path] = hashlib.file_digest(source, "sha256").hexdigest()
        table = read_patient_table(path, model.patient.streams, [*OBSERVED, *TARGETS])
        ids = table["patient_id"]
        if table.num_rows < size or ids.null_count or len(set(ids.to_pylist())) != table.num_rows:
            raise ValueError(
                f"Training cohort must contain at least {size} unique patients: {path}"
            )
        tables.append(table)
    if not fresh and fingerprints != settings.get("cohort_sha256", {}):
        raise ValueError("Training cohort contents have changed since the checkpoint")
    if fresh:
        settings["cohort_sha256"] = fingerprints
        model.patient.tokenizer.save_pretrained(directory / "tokenizer")
        _save(directory, model, optimizer, settings, step=0)
    model.train()
    for step in range(start + 1, steps + 1):
        if tables:
            rng = np.random.default_rng([settings["seed"], step])
            idx = int(rng.integers(len(tables)))
            rows = tables[idx].take(rng.choice(len(tables[idx]), size, replace=False)).to_pylist()
            provenance = {"cohort": settings["cohorts"][idx]}
        else:
            rows, provenance = simulate_episode(
                settings["seed"],
                step,
                False,
                size,
                schema_version=model.schema_version,
                designs=designs,
            )
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = fit_episode(model, rows, n_context)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        print(json.dumps({"step": step, "loss": loss.item(), **provenance}), flush=True)
        if step == 1 or step % save_every == 0 or step == steps:
            _save(directory, model, optimizer, settings, step)


@torch.inference_mode()
def evaluate(model, seed, studies, context_size, query_size, *, bank=None):
    if min(studies, context_size, query_size) < 1:
        raise ValueError("Evaluation study and patient counts must be positive")
    model.eval()
    errors, constant_errors, arm_errors = [], [], []
    designs = split_designs(bank, "held_out")
    by_design = {}
    for step in range(1, studies + 1):
        rows, provenance = simulate_episode(
            seed,
            step,
            True,
            context_size + query_size,
            schema_version=model.schema_version,
            designs=designs,
        )
        _, pred, truth = fit_episode(model, rows, context_size)
        t, y = binary_observations(rows[:context_size], model.centers.device)
        rates = ArmScale.from_context(t, y).rate
        true_effect = truth[:, 1] - truth[:, 0]
        errors.append(pred[:, 1] - pred[:, 0] - true_effect)
        constant_errors.append(rates[1] - rates[0] - true_effect)
        arm_errors.append(pred - truth)
        if designs:
            by_design.setdefault(provenance["design_id"], []).append(errors[-1])
    result = dict(
        studies=studies,
        query_patients=studies * query_size,
        risk_rmse=torch.cat(arm_errors).square().mean().sqrt().item(),
        cate_rmse=torch.cat(errors).square().mean().sqrt().item(),
        arm_mean_cate_rmse=torch.cat(constant_errors).square().mean().sqrt().item(),
    )
    if designs:
        result["held_out_designs"] = {
            key: dict(
                studies=len(values), cate_rmse=torch.cat(values).square().mean().sqrt().item()
            )
            for key, values in by_design.items()
        }
    return result
