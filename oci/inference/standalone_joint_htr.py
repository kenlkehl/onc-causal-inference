"""Standalone joint HTR: one network for the propensity, the outcome, and tau.

Per outer fold, one hierarchical transformer (the Stage 1 HTR extractor) is
trained with three heads on its text representation:

* propensity ``e(x) = P(T=1 | x)``, binary cross-entropy;
* outcome ``m(x) = E[Y | x]``, binary cross-entropy or squared error;
* effect ``tau(x)``, the pseudo-outcome objective of the Stage 1 effect models
  (``pseudo_outcome_mse``): squared error against ``(Y - m(x)) / (T - e(x))``.

The pseudo-outcome uses the network's own current ``e`` and ``m`` on each
batch, detached, with ``e`` clipped to ``[e_clip, 1 - e_clip]``. The effect
loss therefore never moves the nuisance heads directly, but it does train the
shared extractor. The effect loss is off for the first
``effect_warmup_epochs`` epochs. An optional validation subset of the outer
training rows selects the epoch and calibrates the held-out propensity and
outcome probabilities; the outer held-out fold is used only for prediction.

This path runs outside the two-stage workflow: no LLM, handoff, or Stage 2.
Fitting reads only the text, treatment, and outcome columns of the dataset.
Oracle columns are read by :func:`evaluate_against_oracle`, which first
verifies the SHA-256 of the frozen predictions.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import multiprocessing
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import AppliedInferenceConfig, ModelArchitectureConfig, TrainingConfig

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "oci_standalone_joint_htr_v1"
ROW_ID = "_oci_row_id"
ORACLE_NAME_PREFIXES = ("true_", "oracle_", "ground_truth")
DEFAULT_TEMPLATE = (
    Path(__file__).resolve().parents[2] / "example_configs" / "all_evidence_stage1_model.json"
)
DEFAULT_HTR_MODEL = "prajjwal1/bert-tiny"
SELECTION_RULES = ("validation_loss", "last_epoch")
CALIBRATION_METHODS = ("none", "temperature", "isotonic", "temperature_isotonic")


@dataclass
class JointHTRConfig:
    """Settings of the joint model that the shared HTR configuration lacks.

    Optimization (learning rate, schedule, batch size, weight decay, gradient
    clipping, data-loader workers) and the propensity loss weight
    (``alpha_propensity``) come from :class:`TrainingConfig`; ``epochs`` here
    replaces ``training.epochs``.
    """

    outer_folds: int = 5
    seed: int = 42
    epochs: int = 30
    # Weight of the pseudo-outcome loss relative to the outcome loss.
    effect_loss_weight: float = 1.0
    # Epochs trained on the nuisance losses alone before the effect loss starts.
    effect_warmup_epochs: int = 5
    # Propensity clip in the pseudo-outcome denominator (Stage 1's e_clip).
    e_clip: float = 0.01
    label_smoothing: float = 0.02
    # Share of each outer training fold held out for epoch selection and
    # calibration; 0 trains on every outer training row.
    validation_fraction: float = 0.1
    # "validation_loss": the epoch after warmup with the lowest validation
    # outcome loss + alpha_propensity * propensity loss + effect_loss_weight *
    # R-loss. The R-loss replaces the pseudo-outcome MSE in this criterion
    # because it is bounded by the squared outcome residual.
    selection: str = "validation_loss"
    # Fit on the validation rows and applied to the held-out e and m. Temperature
    # scaling corrects the overconfidence of in-sample training without the
    # step-function (and 0/1) probabilities isotonic regression gives on a few
    # hundred rows.
    nuisance_calibration: str = "temperature"
    # Propensity clip in the AIPW scores (Stage 2's propensity_clip).
    aipw_clip: float = 0.02

    def __post_init__(self) -> None:
        if int(self.outer_folds) < 2:
            raise ValueError("joint.outer_folds must be >= 2")
        if int(self.epochs) < 1:
            raise ValueError("joint.epochs must be >= 1")
        if not 0 <= int(self.effect_warmup_epochs) < int(self.epochs):
            raise ValueError("joint.effect_warmup_epochs must be in [0, epochs)")
        if float(self.effect_loss_weight) <= 0:
            raise ValueError("joint.effect_loss_weight must be positive")
        if not 0.0 < float(self.e_clip) < 0.5:
            raise ValueError("joint.e_clip must be in (0, 0.5)")
        if not 0.0 < float(self.aipw_clip) < 0.5:
            raise ValueError("joint.aipw_clip must be in (0, 0.5)")
        if not 0.0 <= float(self.label_smoothing) < 1.0:
            raise ValueError("joint.label_smoothing must be in [0, 1)")
        if not 0.0 <= float(self.validation_fraction) < 0.5:
            raise ValueError("joint.validation_fraction must be in [0, 0.5)")
        self.selection = str(self.selection).strip().lower()
        if self.selection not in SELECTION_RULES:
            raise ValueError(f"joint.selection must be one of {list(SELECTION_RULES)}")
        self.nuisance_calibration = str(self.nuisance_calibration).strip().lower()
        if self.nuisance_calibration not in CALIBRATION_METHODS:
            raise ValueError(
                f"joint.nuisance_calibration must be one of {list(CALIBRATION_METHODS)}"
            )
        if float(self.validation_fraction) == 0 and (
            self.selection != "last_epoch" or self.nuisance_calibration != "none"
        ):
            raise ValueError(
                "joint.validation_fraction=0 requires joint.selection='last_epoch' and "
                "joint.nuisance_calibration='none'"
            )


# ---------------------------------------------------------------------------
# Network and losses (torch is imported lazily so `--help` stays fast)


def _torch():
    import torch

    return torch


def build_joint_net(extractor: Any, hidden_dim: int, outcome_type: str):
    """A shared-extractor network with propensity, outcome, and effect heads."""

    import torch.nn as nn

    from .htr_trainer import _configured_hidden_head

    class JointHTRNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.extractor = extractor
            self.outcome_type = outcome_type
            # Same head layout as NuisanceNet / EffectNet: one hidden layer
            # shared by e and m, and a separate hidden layer for tau.
            self.shared = _configured_hidden_head(
                input_dim=int(extractor.output_dim),
                hidden_dim=hidden_dim,
                depth=1,
                activation="relu",
                dropout=0.1,
                layer_norm=False,
                bias=True,
            )
            self.propensity = nn.Linear(hidden_dim, 1)
            self.outcome = nn.Linear(hidden_dim, 1)
            self.effect_hidden = _configured_hidden_head(
                input_dim=int(extractor.output_dim),
                hidden_dim=hidden_dim,
                depth=1,
                activation="relu",
                dropout=0.1,
                layer_norm=False,
                bias=True,
            )
            self.effect = nn.Linear(hidden_dim, 1)

        def forward(self, texts_or_batch):
            features = self.extractor(
                texts_or_batch if isinstance(texts_or_batch, dict) else list(texts_or_batch)
            )
            hidden = self.shared(features)
            return (
                self.propensity(hidden).squeeze(-1),
                self.outcome(hidden).squeeze(-1),
                self.effect(self.effect_hidden(features)).squeeze(-1),
            )

    return JointHTRNet()


def joint_losses(
    t_logit,
    m_output,
    tau,
    t,
    y,
    *,
    outcome_type: str,
    e_clip: float,
    label_smoothing: float,
) -> Dict[str, Any]:
    """Per-batch losses; ``effect`` is the pseudo-outcome MSE with detached nuisances."""

    torch = _torch()
    import torch.nn.functional as F

    from .htr_trainer import torch_pseudo_outcome_mse_loss_vector

    def smoothed(target):
        if label_smoothing > 0:
            return target * (1.0 - label_smoothing) + 0.5 * label_smoothing
        return target

    propensity = F.binary_cross_entropy_with_logits(t_logit, smoothed(t))
    if outcome_type == "continuous":
        outcome = F.mse_loss(m_output, y)
        m_hat = m_output
    else:
        outcome = F.binary_cross_entropy_with_logits(m_output, smoothed(y))
        m_hat = torch.sigmoid(m_output)
    e_raw = torch.sigmoid(t_logit).detach()
    e_hat = e_raw.clamp(e_clip, 1.0 - e_clip)
    y_residual = y - m_hat.detach()
    t_residual = t - e_hat
    loss_vector, valid = torch_pseudo_outcome_mse_loss_vector(tau, y_residual, t_residual)
    valid_count = valid.sum()
    effect = torch.where(
        valid_count > 0,
        torch.where(valid, loss_vector, 0.0).sum() / valid_count.clamp_min(1),
        loss_vector.mean(),
    )
    pseudo = torch.where(valid, y_residual, 0.0) / torch.where(valid, t_residual, 1.0)
    return {
        "propensity": propensity,
        "outcome": outcome,
        "effect": effect,
        # Monitoring only (no gradient).
        "r_loss": torch.mean(torch.square(y_residual - tau.detach() * t_residual)),
        "abs_t_residual": torch.mean(torch.abs(t_residual)),
        "e_clipped_fraction": torch.mean(
            ((e_raw <= e_clip) | (e_raw >= 1.0 - e_clip)).to(t.dtype)
        ),
        "pseudo_outcome_rms": torch.sqrt(torch.mean(torch.square(pseudo))),
    }


TRAIN_STATS = (
    "loss",
    "propensity",
    "outcome",
    "effect",
    "r_loss",
    "abs_t_residual",
    "e_clipped_fraction",
    "pseudo_outcome_rms",
)


def _bce(probability: np.ndarray, target: np.ndarray) -> float:
    p = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return float(-np.mean(target * np.log(p) + (1.0 - target) * np.log(1.0 - p)))


def _auroc(score: np.ndarray, target: np.ndarray) -> Optional[float]:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(target)) < 2:
        return None
    return float(roc_auc_score(target, score))


def _format_metric(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def nuisance_and_effect_metrics(
    e_raw: np.ndarray,
    m_hat: np.ndarray,
    tau: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    *,
    outcome_type: str,
    e_clip: float,
    alpha_propensity: float,
    effect_loss_weight: float,
) -> Dict[str, Any]:
    """Unsmoothed held-out losses for predictions the model never trained on."""

    from .htr_trainer import r_pseudo_outcome

    e_hat = np.clip(e_raw, e_clip, 1.0 - e_clip)
    y_residual = y - m_hat
    t_residual = t - e_hat
    pseudo = r_pseudo_outcome(y_residual, t_residual)
    propensity = _bce(e_raw, t)
    outcome = _bce(m_hat, y) if outcome_type == "binary" else float(np.mean((y - m_hat) ** 2))
    r_loss = float(np.mean((y_residual - tau * t_residual) ** 2))
    return {
        "rows": int(len(t)),
        "propensity_loss": propensity,
        "propensity_auroc": _auroc(e_raw, t),
        "outcome_loss": outcome,
        "outcome_auroc": _auroc(m_hat, y) if outcome_type == "binary" else None,
        "pseudo_outcome_mse": float(np.nanmean((tau - pseudo) ** 2)),
        "r_loss": r_loss,
        "r_loss_at_zero_tau": float(np.mean(y_residual**2)),
        "tau_mean": float(np.mean(tau)),
        "tau_sd": float(np.std(tau)),
        "selection_loss": outcome + alpha_propensity * propensity + effect_loss_weight * r_loss,
    }


# ---------------------------------------------------------------------------
# Trainer


def _trainer_class():
    from .htr_trainer import HTRTrainer, current_lr, make_linear_lr_scheduler

    class JointHTRTrainer(HTRTrainer):
        """Train and score one joint network on one device."""

        def __init__(self, *, config, joint: JointHTRConfig, device, num_workers=None):
            super().__init__(config=config, device=device, num_workers=num_workers)
            self.joint = joint

        def create_model(self):
            return build_joint_net(
                self._create_extractor(),
                int(self.config.architecture.htr_prediction_head_hidden_dim),
                str(self.config.outcome_type),
            ).to(self.device)

        def _fields(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
            return {
                "t": df[self.config.treatment_column].to_numpy(dtype=np.float32),
                "y": df[self.config.outcome_column].to_numpy(dtype=np.float32),
            }

        def predict(self, model, df: pd.DataFrame, loader=None):
            """Raw ``e``, ``m`` (probability for binary outcomes), and ``tau``."""

            torch = _torch()
            model.eval()
            if loader is None:
                loader = self._make_text_loader(model, df, np.arange(len(df), dtype=int))
            e, m, tau = [], [], []
            with torch.no_grad():
                for batch in loader:
                    t_logit, m_output, effect = model(batch["model_input"])
                    e.append(torch.sigmoid(t_logit).cpu().numpy())
                    if self.config.outcome_type == "continuous":
                        m.append(m_output.cpu().numpy())
                    else:
                        m.append(torch.sigmoid(m_output).cpu().numpy())
                    tau.append(effect.cpu().numpy())
            return (
                np.concatenate(e).astype(float),
                np.concatenate(m).astype(float),
                np.concatenate(tau).astype(float),
            )

        def train_joint(
            self,
            model,
            df: pd.DataFrame,
            fit_positions: np.ndarray,
            validation_positions: np.ndarray,
            *,
            fold: int,
        ) -> Dict[str, Any]:
            torch = _torch()
            joint = self.joint
            train_config = self.config.training
            alpha = float(train_config.alpha_propensity)
            model.extractor.fit_tokenizer(
                df.iloc[fit_positions][self.config.text_column].astype(str).tolist()
            )
            self._assert_htr_sentence_encoder_training_state(model.extractor)
            train_loader = self._make_text_loader(
                model, df, fit_positions, fields=self._fields(df), shuffle=True
            )
            validation_df = df.iloc[validation_positions]
            validation_loader = (
                self._make_text_loader(
                    model, validation_df, np.arange(len(validation_df), dtype=int)
                )
                if len(validation_positions)
                else None
            )
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=train_config.learning_rate,
                weight_decay=getattr(train_config, "weight_decay", 0.01),
            )
            self._assert_htr_sentence_encoder_optimizer_coverage(model.extractor, optimizer)
            num_batches = max(1, len(train_loader))
            scheduler = make_linear_lr_scheduler(
                optimizer, train_config, num_batches, epochs_override=int(joint.epochs)
            )
            logger.info(
                "Fold %s: joint HTR training for %s epoch(s) (effect loss from epoch %s), "
                "fit rows=%s, validation rows=%s, batch_size=%s, batches/epoch=%s, "
                "dataloader_workers=%s, lr=%.3g%s",
                fold,
                joint.epochs,
                joint.effect_warmup_epochs + 1,
                len(fit_positions),
                len(validation_positions),
                train_config.batch_size,
                num_batches,
                train_loader.num_workers,
                current_lr(optimizer),
                self._cuda_memory_summary(),
            )
            t_val = validation_df[self.config.treatment_column].to_numpy(dtype=float)
            y_val = validation_df[self.config.outcome_column].to_numpy(dtype=float)
            history: List[Dict[str, Any]] = []
            best: Optional[Tuple[float, int, Dict[str, Any]]] = None
            progress_every = max(1, num_batches // 5)
            for epoch in range(1, int(joint.epochs) + 1):
                effect_active = epoch > int(joint.effect_warmup_epochs)
                model.train()
                sums = torch.zeros(len(TRAIN_STATS), device=self.device, dtype=torch.float64)
                batch_count = 0
                for batch_idx, batch in enumerate(train_loader, start=1):
                    t = batch["t"].to(self.device, non_blocking=True)
                    y = batch["y"].to(self.device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    t_logit, m_output, tau = model(batch["model_input"])
                    parts = joint_losses(
                        t_logit,
                        m_output,
                        tau,
                        t,
                        y,
                        outcome_type=str(self.config.outcome_type),
                        e_clip=float(joint.e_clip),
                        label_smoothing=float(joint.label_smoothing),
                    )
                    loss = parts["outcome"] + alpha * parts["propensity"]
                    if effect_active:
                        loss = loss + float(joint.effect_loss_weight) * parts["effect"]
                    loss.backward()
                    self._clip_and_step(model, optimizer, scheduler)
                    batch_count += 1
                    sums.add_(
                        torch.stack(
                            [loss.detach()] + [parts[name].detach() for name in TRAIN_STATS[1:]]
                        ).to(torch.float64)
                    )
                    if batch_idx == 1 or batch_idx == num_batches or batch_idx % progress_every == 0:
                        logger.info(
                            "Fold %s epoch %s/%s batch %s/%s lr=%.3g%s",
                            fold,
                            epoch,
                            joint.epochs,
                            batch_idx,
                            num_batches,
                            current_lr(optimizer),
                            self._cuda_memory_summary(),
                        )
                record: Dict[str, Any] = {
                    "epoch": epoch,
                    "effect_loss_active": effect_active,
                    "lr_end": current_lr(optimizer),
                    "train": dict(
                        zip(TRAIN_STATS, (sums / max(1, batch_count)).cpu().tolist(), strict=True)
                    ),
                }
                if validation_loader is not None:
                    e_val, m_val, tau_val = self.predict(model, validation_df, validation_loader)
                    record["validation"] = nuisance_and_effect_metrics(
                        e_val,
                        m_val,
                        tau_val,
                        t_val,
                        y_val,
                        outcome_type=str(self.config.outcome_type),
                        e_clip=float(joint.e_clip),
                        alpha_propensity=alpha,
                        effect_loss_weight=float(joint.effect_loss_weight),
                    )
                    score = float(record["validation"]["selection_loss"])
                    if (
                        joint.selection == "validation_loss"
                        and effect_active
                        and math.isfinite(score)
                        and (best is None or score < best[0])
                    ):
                        state = {
                            key: value.detach().to("cpu", copy=True)
                            for key, value in model.state_dict().items()
                        }
                        best = (score, epoch, state)
                history.append(record)
                validation = record.get("validation") or {}
                logger.info(
                    "Fold %s epoch %s/%s complete: loss=%.4f outcome=%.4f propensity=%.4f "
                    "effect=%.4f%s | |t-e|=%.3f e_clipped=%.3f pseudo_rms=%.3f%s",
                    fold,
                    epoch,
                    joint.epochs,
                    record["train"]["loss"],
                    record["train"]["outcome"],
                    record["train"]["propensity"],
                    record["train"]["effect"],
                    "" if effect_active else " (warmup)",
                    record["train"]["abs_t_residual"],
                    record["train"]["e_clipped_fraction"],
                    record["train"]["pseudo_outcome_rms"],
                    (
                        f" | validation selection_loss={validation['selection_loss']:.4f}"
                        f" r_loss={validation['r_loss']:.4f}"
                        f" (zero tau {validation['r_loss_at_zero_tau']:.4f})"
                        f" e_auroc={_format_metric(validation['propensity_auroc'])}"
                        + (
                            f" m_auroc={_format_metric(validation['outcome_auroc'])}"
                            if self.config.outcome_type == "binary"
                            else f" m_mse={validation['outcome_loss']:.4f}"
                        )
                        if validation
                        else ""
                    ),
                )
            selected_epoch = int(joint.epochs)
            if best is not None:
                selected_epoch = best[1]
                model.load_state_dict(best[2])
            return {"history": history, "selected_epoch": selected_epoch}

    return JointHTRTrainer


# ---------------------------------------------------------------------------
# Data, splits, and identities


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _row_fingerprint(rows: Sequence[int]) -> str:
    return _sha256_json(sorted(int(row) for row in rows))


def _write_json(path: Path, value: Any) -> None:
    from .stage2_diagnostics import json_safe

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def is_oracle_column(name: Any) -> bool:
    return str(name).strip().lower().startswith(ORACLE_NAME_PREFIXES)


def load_training_frame(dataset_path: Path, config: AppliedInferenceConfig) -> pd.DataFrame:
    """Read only the text, treatment, and outcome columns; rows keep file order."""

    columns = [config.text_column, config.treatment_column, config.outcome_column]
    oracle = [column for column in columns if is_oracle_column(column)]
    if oracle:
        raise ValueError(f"oracle columns cannot be model inputs: {oracle}")
    frame = pd.read_parquet(dataset_path, columns=columns).reset_index(drop=True)
    # Treatment and outcome keep their stored dtypes so the default outer folds
    # equal the workflow's folds for the same seed.
    for column in (config.treatment_column, config.outcome_column):
        if pd.to_numeric(frame[column], errors="coerce").isna().any():
            raise ValueError(f"{column} has missing or non-numeric values")
    if not set(pd.to_numeric(frame[config.treatment_column]).unique()) <= {0, 1}:
        raise ValueError(f"{config.treatment_column} must be binary (0/1)")
    if config.outcome_type == "binary" and not set(
        pd.to_numeric(frame[config.outcome_column]).unique()
    ) <= {0, 1}:
        raise ValueError(f"{config.outcome_column} must be 0/1 for a binary outcome")
    frame[config.text_column] = frame[config.text_column].fillna("").astype(str)
    frame[ROW_ID] = np.arange(len(frame), dtype=int)
    return frame


def infer_outcome_type(dataset_path: Path, outcome_column: str) -> str:
    values = pd.read_parquet(dataset_path, columns=[outcome_column])[outcome_column]
    unique = set(pd.to_numeric(values, errors="coerce").dropna().unique().tolist())
    return "binary" if unique <= {0, 1} else "continuous"


def outer_splits(
    frame: pd.DataFrame,
    config: AppliedInferenceConfig,
    joint: JointHTRConfig,
    splits_path: Optional[Path] = None,
) -> Tuple[List[Tuple[int, np.ndarray, np.ndarray]], Dict[str, Any]]:
    """The workflow's stratified outer folds, or the folds of a split-provenance file."""

    n = len(frame)
    if splits_path is None:
        from .tfidf_topic_stage1 import make_joint_treatment_outcome_splits

        splits, metadata = make_joint_treatment_outcome_splits(
            frame,
            treatment_column=config.treatment_column,
            outcome_column=config.outcome_column,
            outcome_type=config.outcome_type,
            n_splits=int(joint.outer_folds),
            seed=int(joint.seed),
        )
        folds = [(fold, fit, test) for fold, (fit, test) in enumerate(splits, start=1)]
    else:
        folds = []
        for line in Path(splits_path).read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            folds.append(
                (
                    int(record["outer_fold"]),
                    np.asarray(record["fit_row_ids"], dtype=int),
                    np.asarray(record["heldout_row_ids"], dtype=int),
                )
            )
        folds.sort(key=lambda item: item[0])
        metadata = {
            "method": "split_provenance_file",
            "path": str(Path(splits_path).resolve()),
            "sha256": _sha256_file(Path(splits_path)),
            "n_splits": len(folds),
        }
    heldout = np.concatenate([test for _, _, test in folds]) if folds else np.array([], int)
    if len(folds) < 2 or sorted(heldout.tolist()) != list(range(n)):
        raise ValueError("outer folds must have disjoint held-out rows that cover every row")
    for fold, fit, test in folds:
        if len(np.intersect1d(fit, test)):
            raise ValueError(f"outer fold {fold} fit rows overlap its held-out rows")
        if len(fit) == 0 or fit.min() < 0 or fit.max() >= n:
            raise ValueError(f"outer fold {fold} fit rows are empty or outside the dataset")
    return folds, metadata


def validation_split(
    frame: pd.DataFrame,
    fit_idx: np.ndarray,
    config: AppliedInferenceConfig,
    joint: JointHTRConfig,
    fold: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified (treatment x binary outcome) validation subset of an outer training fold."""

    from sklearn.model_selection import train_test_split

    fit_idx = np.asarray(fit_idx, dtype=int)
    if float(joint.validation_fraction) == 0:
        return fit_idx, np.array([], dtype=int)
    t = frame[config.treatment_column].to_numpy(dtype=float)[fit_idx]
    strata = (
        t * 2 + frame[config.outcome_column].to_numpy(dtype=float)[fit_idx]
        if config.outcome_type == "binary"
        else t
    )
    seed = int(joint.seed) + 7_919 * int(fold)
    try:
        train, validation = train_test_split(
            fit_idx, test_size=float(joint.validation_fraction), random_state=seed, stratify=strata
        )
    except ValueError:
        train, validation = train_test_split(
            fit_idx, test_size=float(joint.validation_fraction), random_state=seed
        )
    return np.sort(train), np.sort(validation)


def _config_record(config: AppliedInferenceConfig, joint: JointHTRConfig) -> Dict[str, Any]:
    architecture = asdict(config.architecture)
    return {
        "outcome_type": config.outcome_type,
        "text_column": config.text_column,
        "treatment_column": config.treatment_column,
        "outcome_column": config.outcome_column,
        "architecture": {
            key: value
            for key, value in architecture.items()
            if key.startswith("htr_") or key == "feature_extractor_type"
        },
        "training": asdict(config.training),
        "joint": asdict(joint),
    }


# ---------------------------------------------------------------------------
# Folds


def run_fold(
    *,
    frame: pd.DataFrame,
    config: AppliedInferenceConfig,
    joint: JointHTRConfig,
    fold: int,
    fit_idx: np.ndarray,
    test_idx: np.ndarray,
    device: str,
    fold_dir: Path,
    identity: Mapping[str, Any],
) -> None:
    """Train one joint network on an outer training fold and freeze its held-out predictions."""

    from ..utils.calibration import BinaryProbabilityCalibrator
    from .stage1_checkpointing import seed_everything
    from .stage2_diagnostics import dr_scores

    seed_everything(int(joint.seed) + 1_000 * int(fold))
    train_idx, validation_idx = validation_split(frame, fit_idx, config, joint, fold)
    trainer = _trainer_class()(config=config, joint=joint, device=device)
    model = trainer.create_model()
    try:
        result = trainer.train_joint(model, frame, train_idx, validation_idx, fold=fold)
        test_df = frame.iloc[test_idx]
        e_raw, m_raw, tau = trainer.predict(model, test_df)
        calibration: Dict[str, Any] = {"method": joint.nuisance_calibration}
        e_hat, m_hat = e_raw, m_raw
        if len(validation_idx) and joint.nuisance_calibration != "none":
            validation_df = frame.iloc[validation_idx]
            e_val, m_val, _ = trainer.predict(model, validation_df)
            propensity_calibrator = BinaryProbabilityCalibrator.fit(
                e_val,
                validation_df[config.treatment_column].to_numpy(dtype=float),
                method=joint.nuisance_calibration,
            )
            e_hat = propensity_calibrator.transform(e_raw)
            calibration.update(propensity_calibrator.metadata("propensity"))
            if config.outcome_type == "binary":
                outcome_calibrator = BinaryProbabilityCalibrator.fit(
                    m_val,
                    validation_df[config.outcome_column].to_numpy(dtype=float),
                    method=joint.nuisance_calibration,
                )
                m_hat = outcome_calibrator.transform(m_raw)
                calibration.update(outcome_calibrator.metadata("outcome"))
    finally:
        trainer._cleanup_model(model)
    t = test_df[config.treatment_column].to_numpy(dtype=float)
    y = test_df[config.outcome_column].to_numpy(dtype=float)
    # tau is the effect head; mu0/mu1 follow from m = e * mu1 + (1 - e) * mu0.
    mu0 = m_hat - e_hat * tau
    mu1 = m_hat + (1.0 - e_hat) * tau
    e_clipped = np.clip(e_hat, joint.e_clip, 1.0 - joint.e_clip)
    predictions = pd.DataFrame(
        {
            ROW_ID: test_df[ROW_ID].to_numpy(dtype=int),
            "outer_fold": int(fold),
            "treatment": t,
            "outcome": y,
            "propensity_raw": e_raw,
            "propensity": e_hat,
            "m_hat_raw": m_raw,
            "m_hat": m_hat,
            "estimated_cate": tau,
            "mu0": mu0,
            "mu1": mu1,
            "y_residual": y - m_hat,
            "t_residual": t - e_clipped,
            "aipw_score": dr_scores(y, t, e_hat, mu0, mu1, clip=float(joint.aipw_clip)),
        }
    )
    fold_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(fold_dir / "predictions.parquet", index=False)
    _write_json(fold_dir / "history.json", result["history"])
    _write_json(
        fold_dir / "fold.json",
        {
            "schema_version": SCHEMA_VERSION,
            "identity": dict(identity),
            "identity_sha256": _sha256_json(identity),
            "selected_epoch": result["selected_epoch"],
            "rows": {
                "train": len(train_idx),
                "validation": len(validation_idx),
                "heldout": len(test_idx),
            },
            "calibration": calibration,
            "predictions_sha256": _sha256_file(fold_dir / "predictions.parquet"),
            "device": str(device),
        },
    )


def _fold_identity(
    run_identity: Mapping[str, Any], fold: int, fit_idx: np.ndarray, test_idx: np.ndarray
) -> Dict[str, Any]:
    return {
        "run_identity_sha256": _sha256_json(run_identity),
        "outer_fold": int(fold),
        "fit_rows": _row_fingerprint(fit_idx),
        "heldout_rows": _row_fingerprint(test_idx),
    }


def fold_is_complete(fold_dir: Path, identity: Mapping[str, Any]) -> bool:
    """Finished folds are reused; a fold written under another identity fails closed."""

    record_path = fold_dir / "fold.json"
    predictions_path = fold_dir / "predictions.parquet"
    if not record_path.exists() or not predictions_path.exists():
        return False
    record = json.loads(record_path.read_text())
    if record.get("identity_sha256") != _sha256_json(identity):
        raise RuntimeError(
            f"{fold_dir} was written by a different configuration, dataset, or split; "
            "use a new output directory"
        )
    if record.get("predictions_sha256") != _sha256_file(predictions_path):
        raise RuntimeError(f"{predictions_path} changed after it was written")
    return True


def _configure_logging(log_path: Optional[Path] = None, level: str = "INFO") -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def _device_worker(payload: Mapping[str, Any]) -> None:
    """Spawned per device: run its folds serially."""

    output_dir = Path(payload["output_dir"])
    device = str(payload["device"])
    _configure_logging(output_dir / "logs" / f"worker_{device.replace(':', '_')}.log")
    config = payload["config"]
    joint = payload["joint"]
    frame = load_training_frame(Path(payload["dataset"]), config)
    for fold, fit_idx, test_idx, identity in payload["folds"]:
        run_fold(
            frame=frame,
            config=config,
            joint=joint,
            fold=fold,
            fit_idx=fit_idx,
            test_idx=test_idx,
            device=device,
            fold_dir=output_dir / "folds" / f"fold_{int(fold):02d}",
            identity=identity,
        )


# ---------------------------------------------------------------------------
# Run


def run_standalone_joint_htr(
    *,
    dataset_path: Path,
    output_dir: Path,
    config: AppliedInferenceConfig,
    joint: JointHTRConfig,
    devices: Sequence[str] = ("cpu",),
    splits_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Fit every outer fold (reusing finished ones), freeze predictions, and summarize."""

    dataset_path = Path(dataset_path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_training_frame(dataset_path, config)
    folds, split_metadata = outer_splits(frame, config, joint, splits_path)
    run_identity = {
        "schema_version": SCHEMA_VERSION,
        "dataset_sha256": _sha256_file(dataset_path),
        "rows": len(frame),
        "config": _config_record(config, joint),
        "splits": {
            "metadata": split_metadata,
            "folds": {
                str(fold): {"fit": _row_fingerprint(fit), "heldout": _row_fingerprint(test)}
                for fold, fit, test in folds
            },
        },
    }
    run_path = output_dir / "run.json"
    if run_path.exists():
        previous = json.loads(run_path.read_text())
        if previous.get("identity_sha256") != _sha256_json(run_identity):
            raise RuntimeError(
                f"{output_dir} holds a run with a different configuration, dataset, or split; "
                "use a new output directory"
            )
    _write_json(
        run_path,
        {
            "dataset": str(dataset_path),
            "identity": run_identity,
            "identity_sha256": _sha256_json(run_identity),
        },
    )
    pending = []
    for fold, fit_idx, test_idx in folds:
        identity = _fold_identity(run_identity, fold, fit_idx, test_idx)
        fold_dir = output_dir / "folds" / f"fold_{int(fold):02d}"
        if fold_is_complete(fold_dir, identity):
            logger.info("Fold %s is complete; reusing %s", fold, fold_dir)
        else:
            pending.append((fold, fit_idx, test_idx, identity))
    devices = [str(device) for device in devices] or ["cpu"]
    if len(devices) == 1 or len(pending) <= 1:
        for fold, fit_idx, test_idx, identity in pending:
            run_fold(
                frame=frame,
                config=config,
                joint=joint,
                fold=fold,
                fit_idx=fit_idx,
                test_idx=test_idx,
                device=devices[0],
                fold_dir=output_dir / "folds" / f"fold_{int(fold):02d}",
                identity=identity,
            )
    else:
        context = multiprocessing.get_context("spawn")
        workers = []
        for offset, device in enumerate(devices):
            assigned = pending[offset :: len(devices)]
            if not assigned:
                continue
            process = context.Process(
                target=_device_worker,
                args=(
                    {
                        "dataset": str(dataset_path),
                        "output_dir": str(output_dir),
                        "device": device,
                        "config": config,
                        "joint": joint,
                        "folds": assigned,
                    },
                ),
            )
            process.start()
            logger.info("Started %s for folds %s", device, [item[0] for item in assigned])
            workers.append((device, process))
        failed = []
        for device, process in workers:
            process.join()
            if process.exitcode != 0:
                failed.append(device)
        if failed:
            raise RuntimeError(
                f"fold workers on {failed} failed; see {output_dir / 'logs'} and rerun to resume"
            )
    parts = []
    for fold, fit_idx, test_idx in folds:
        fold_dir = output_dir / "folds" / f"fold_{int(fold):02d}"
        if not fold_is_complete(fold_dir, _fold_identity(run_identity, fold, fit_idx, test_idx)):
            raise RuntimeError(f"fold {fold} did not finish")
        parts.append(pd.read_parquet(fold_dir / "predictions.parquet"))
    predictions = pd.concat(parts, ignore_index=True).sort_values(ROW_ID, ignore_index=True)
    predictions_path = output_dir / "predictions.parquet"
    predictions.to_parquet(predictions_path, index=False)
    _write_json(
        output_dir / "predictions_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "predictions_sha256": _sha256_file(predictions_path),
            "run_identity_sha256": _sha256_json(run_identity),
            "rows": len(predictions),
        },
    )
    summary = summarize_predictions(predictions, output_dir, config, joint)
    _write_json(output_dir / "results.json", summary)
    return summary


def _mean_with_interval(values: np.ndarray) -> Dict[str, Any]:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return {"estimate": None, "standard_error": None, "confidence_interval_95": None}
    estimate = float(np.mean(values))
    se = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    return {
        "estimate": estimate,
        "standard_error": se,
        "confidence_interval_95": [estimate - 1.959964 * se, estimate + 1.959964 * se],
        "rows": int(len(values)),
    }


def summarize_predictions(
    predictions: pd.DataFrame,
    output_dir: Path,
    config: AppliedInferenceConfig,
    joint: JointHTRConfig,
) -> Dict[str, Any]:
    """Oracle-free held-out summary: ATE, BLP heterogeneity test, losses, nuisance fit."""

    from .stage2_diagnostics import best_linear_predictor, pool_blp

    tau = predictions["estimated_cate"].to_numpy(dtype=float)
    aipw = _mean_with_interval(predictions["aipw_score"].to_numpy(dtype=float))
    blp = {
        str(fold): best_linear_predictor(part["aipw_score"], part["estimated_cate"])
        for fold, part in predictions.groupby("outer_fold")
    }
    per_fold = {}
    for fold, part in predictions.groupby("outer_fold"):
        metrics = nuisance_and_effect_metrics(
            part["propensity"].to_numpy(dtype=float),
            part["m_hat"].to_numpy(dtype=float),
            part["estimated_cate"].to_numpy(dtype=float),
            part["treatment"].to_numpy(dtype=float),
            part["outcome"].to_numpy(dtype=float),
            outcome_type=str(config.outcome_type),
            e_clip=float(joint.e_clip),
            alpha_propensity=float(config.training.alpha_propensity),
            effect_loss_weight=float(joint.effect_loss_weight),
        )
        metrics.pop("selection_loss")
        record = json.loads((output_dir / "folds" / f"fold_{int(fold):02d}" / "fold.json").read_text())
        per_fold[str(fold)] = {"selected_epoch": record["selected_epoch"], **metrics}
    y_residual = predictions["y_residual"].to_numpy(dtype=float)
    t_residual = predictions["t_residual"].to_numpy(dtype=float)
    ate = aipw["estimate"]
    return {
        "schema_version": SCHEMA_VERSION,
        "rows": int(len(predictions)),
        "outcome_type": config.outcome_type,
        "ate": {"aipw": aipw, "plug_in_mean_tau": float(np.mean(tau))},
        "heterogeneity": {"blp_pooled": pool_blp(blp), "blp_per_fold": blp},
        "heldout_r_loss": {
            "estimated_tau": float(np.mean((y_residual - tau * t_residual) ** 2)),
            "constant_aipw_ate": (
                float(np.mean((y_residual - ate * t_residual) ** 2)) if ate is not None else None
            ),
            "zero_tau": float(np.mean(y_residual**2)),
        },
        "tau_distribution": {
            "mean": float(np.mean(tau)),
            "sd": float(np.std(tau)),
            "quantiles": {
                str(q): float(np.quantile(tau, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)
            },
        },
        "per_fold": per_fold,
    }


# ---------------------------------------------------------------------------
# Oracle evaluation (synthetic data only, after predictions are frozen)


def evaluate_against_oracle(output_dir: Path) -> Dict[str, Any]:
    """Compare frozen held-out predictions with the dataset's oracle columns."""

    import pyarrow.parquet as pq

    from ..evaluation.benchmark import ate_metrics, cate_metrics, oracle_ite_column

    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "predictions_manifest.json").read_text())
    predictions_path = output_dir / "predictions.parquet"
    if _sha256_file(predictions_path) != manifest["predictions_sha256"]:
        raise RuntimeError(f"{predictions_path} does not match predictions_manifest.json")
    run = json.loads((output_dir / "run.json").read_text())
    dataset_path = Path(run["dataset"])
    if _sha256_file(dataset_path) != run["identity"]["dataset_sha256"]:
        raise RuntimeError(f"{dataset_path} changed since the run")
    predictions = pd.read_parquet(predictions_path)
    outcome_type = run["identity"]["config"]["outcome_type"]
    available = set(pq.read_schema(dataset_path).names)
    ite_column = oracle_ite_column(available, outcome_type)
    if ite_column is None:
        raise ValueError(f"{dataset_path} has no true_ite column")
    wanted = [ite_column] + [
        column
        for column in ("true_treatment_prob", "true_y0_prob", "true_y1_prob", "true_y0", "true_y1")
        if column in available
    ]
    oracle = pd.read_parquet(dataset_path, columns=wanted)
    rows = predictions[ROW_ID].to_numpy(dtype=int)
    truth = oracle[ite_column].to_numpy(dtype=float)[rows]
    aipw = json.loads((output_dir / "results.json").read_text())["ate"]["aipw"]
    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "predictions_sha256": manifest["predictions_sha256"],
        "ite_column": ite_column,
        "ate_aipw": ate_metrics(predictions, truth, aipw),
        "cate": cate_metrics(predictions, truth, aipw.get("estimate")),
    }
    plug_in = float(predictions["estimated_cate"].mean())
    report["ate_plug_in_mean_tau"] = {
        "estimate": plug_in,
        "true": float(np.mean(truth)),
        "bias": plug_in - float(np.mean(truth)),
    }
    report["cate_per_fold"] = {
        str(fold): cate_metrics(part, truth[part.index.to_numpy()], float(part["aipw_score"].mean()))
        for fold, part in predictions.groupby("outer_fold")
    }

    def agreement(estimate: np.ndarray, target: np.ndarray) -> Dict[str, Any]:
        finite = np.isfinite(estimate) & np.isfinite(target)
        estimate, target = estimate[finite], target[finite]
        return {
            "rows": int(finite.sum()),
            "rmse": float(np.sqrt(np.mean((estimate - target) ** 2))),
            "pearson": (
                float(np.corrcoef(estimate, target)[0, 1])
                if np.std(estimate) > 0 and np.std(target) > 0
                else None
            ),
        }

    if "true_treatment_prob" in oracle:
        true_e = oracle["true_treatment_prob"].to_numpy(dtype=float)[rows]
        report["propensity"] = agreement(predictions["propensity"].to_numpy(dtype=float), true_e)
        potential = (
            ("true_y0_prob", "true_y1_prob")
            if outcome_type == "binary" and {"true_y0_prob", "true_y1_prob"} <= set(oracle)
            else ("true_y0", "true_y1")
        )
        if set(potential) <= set(oracle):
            y0 = oracle[potential[0]].to_numpy(dtype=float)[rows]
            y1 = oracle[potential[1]].to_numpy(dtype=float)[rows]
            report["outcome_m"] = agreement(
                predictions["m_hat"].to_numpy(dtype=float), true_e * y1 + (1.0 - true_e) * y0
            )
            report["mu0"] = agreement(predictions["mu0"].to_numpy(dtype=float), y0)
            report["mu1"] = agreement(predictions["mu1"].to_numpy(dtype=float), y1)
    _write_json(output_dir / "evaluation" / "oracle_metrics.json", report)
    return report


# ---------------------------------------------------------------------------
# Configuration and CLI


def _parse_override(raw: str) -> Tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(f"--set must use KEY=VALUE syntax: {raw!r}")
    key, text = raw.split("=", 1)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = text
    return key.strip(), value


def build_configs(
    *,
    template_path: Path = DEFAULT_TEMPLATE,
    outcome_type: str,
    htr_model: str = DEFAULT_HTR_MODEL,
    text_column: Optional[str] = None,
    treatment_column: Optional[str] = None,
    outcome_column: Optional[str] = None,
    overrides: Sequence[str] = (),
) -> Tuple[AppliedInferenceConfig, JointHTRConfig]:
    """HTR and optimizer settings from the Stage 1 template, then ``--set`` overrides.

    Keys are ``architecture.<htr field>``, ``training.<field>``, or ``joint.<field>``.
    """

    template = json.loads(Path(template_path).read_text())["config"]
    architecture = {
        key: value
        for key, value in template.get("architecture", {}).items()
        if key.startswith("htr_") or key == "feature_extractor_type"
    }
    architecture["htr_sentence_model"] = htr_model
    sections: Dict[str, Dict[str, Any]] = {
        "architecture": architecture,
        "training": dict(template.get("training", {})),
        "joint": {},
    }
    for raw in overrides:
        key, value = _parse_override(raw)
        section, _, name = key.partition(".")
        if section not in sections or not name or "." in name:
            raise ValueError(
                f"--set key {key!r} must be architecture.<field>, training.<field>, "
                "or joint.<field>"
            )
        sections[section][name] = value
    known = {
        "architecture": {field.name for field in dataclasses.fields(ModelArchitectureConfig)},
        "training": {field.name for field in dataclasses.fields(TrainingConfig)},
        "joint": {field.name for field in dataclasses.fields(JointHTRConfig)},
    }
    for section, values in sections.items():
        unknown = sorted(set(values) - known[section])
        if unknown:
            raise ValueError(f"unknown {section} settings: {unknown}")
    config = AppliedInferenceConfig(
        seed=int(sections["joint"].get("seed", JointHTRConfig.seed)),
        outcome_type=outcome_type,
        text_column=text_column or template.get("text_column", "clinical_text"),
        treatment_column=treatment_column
        or template.get("treatment_column", "treatment_indicator"),
        outcome_column=outcome_column or template.get("outcome_column", "outcome_indicator"),
        architecture=ModelArchitectureConfig(**sections["architecture"]),
        training=TrainingConfig(**sections["training"]),
    )
    return config, JointHTRConfig(**sections["joint"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone joint HTR: one network per outer fold for the propensity, the "
            "outcome, and tau (pseudo-outcome objective), with cross-fitted held-out "
            "predictions and an optional oracle evaluation."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="fit, freeze predictions, and summarize (resumable)")
    run.add_argument("--dataset", required=True, type=Path, help="cohort parquet file")
    run.add_argument("--out", required=True, type=Path, help="output directory")
    run.add_argument(
        "--outcome-type",
        choices=("auto", "binary", "continuous"),
        default="auto",
        help="auto: binary when every outcome is 0 or 1",
    )
    run.add_argument(
        "--devices",
        help=(
            "comma-separated torch devices; folds run in parallel, one process per device "
            "(default: cuda:0 when CUDA is available, else cpu)"
        ),
    )
    run.add_argument("--htr-model", default=DEFAULT_HTR_MODEL, help="HTR sentence encoder")
    run.add_argument(
        "--splits",
        type=Path,
        help=(
            "a run's components/tfidf/split_provenance.jsonl, to reuse its outer folds "
            "(default: the workflow's stratified folds for --seed)"
        ),
    )
    run.add_argument("--seed", type=int, help="shorthand for --set joint.seed=N")
    run.add_argument("--folds", type=int, help="shorthand for --set joint.outer_folds=K")
    run.add_argument("--epochs", type=int, help="shorthand for --set joint.epochs=N")
    run.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE, help="Stage 1 template")
    run.add_argument("--text-column")
    run.add_argument("--treatment-column")
    run.add_argument("--outcome-column")
    run.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="architecture.<field>, training.<field>, or joint.<field> (JSON values)",
    )
    run.add_argument(
        "--evaluate",
        action="store_true",
        help="after freezing predictions, run the oracle evaluation (synthetic data)",
    )
    run.add_argument("--log-level", default="INFO")
    evaluate = commands.add_parser(
        "evaluate", help="compare a finished run's frozen predictions with oracle columns"
    )
    evaluate.add_argument("--out", required=True, type=Path, help="output directory of a run")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "evaluate":
        _configure_logging()
        print(json.dumps(evaluate_against_oracle(args.out), indent=2, default=str))
        return 0
    _configure_logging(args.out / "logs" / "run.log", args.log_level)
    overrides = list(args.overrides)
    for flag, key in (("seed", "seed"), ("folds", "outer_folds"), ("epochs", "epochs")):
        if getattr(args, flag) is not None:
            overrides.append(f"joint.{key}={getattr(args, flag)}")
    settings = dict(
        template_path=args.template,
        htr_model=args.htr_model,
        text_column=args.text_column,
        treatment_column=args.treatment_column,
        outcome_column=args.outcome_column,
        overrides=overrides,
    )
    outcome_type = args.outcome_type
    if outcome_type == "auto":
        outcome_column = build_configs(outcome_type="binary", **settings)[0].outcome_column
        outcome_type = infer_outcome_type(args.dataset, outcome_column)
    config, joint = build_configs(outcome_type=outcome_type, **settings)
    devices = args.devices
    if devices is None:
        devices = "cuda:0" if _torch().cuda.is_available() else "cpu"
    summary = run_standalone_joint_htr(
        dataset_path=args.dataset,
        output_dir=args.out,
        config=config,
        joint=joint,
        devices=[device.strip() for device in str(devices).split(",") if device.strip()],
        splits_path=args.splits,
    )
    print(json.dumps(summary["ate"], indent=2, default=str))
    print(json.dumps(summary["heterogeneity"]["blp_pooled"], indent=2, default=str))
    if args.evaluate:
        report = evaluate_against_oracle(args.out)
        print(json.dumps({key: report[key] for key in ("ate_aipw", "cate")}, indent=2, default=str))
    return 0


__all__ = [
    "JointHTRConfig",
    "build_configs",
    "build_joint_net",
    "evaluate_against_oracle",
    "joint_losses",
    "load_training_frame",
    "main",
    "outer_splits",
    "run_standalone_joint_htr",
    "summarize_predictions",
]
