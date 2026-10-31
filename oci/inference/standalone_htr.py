"""Standalone HTR treatment-effect estimators, outside the two-stage workflow.

Per outer fold, on the outer training rows only, one of two methods fits HTRs
(the Stage 1 hierarchical transformer) and predicts the outer held-out rows:

* ``two_stage`` (default) runs the Stage 1 ``htr_neural`` lane's training
  unchanged, through ``MultiModelForestStage1HTRProvider``: ``nuisance_folds``
  cross-fitted HTR nuisance networks (propensity and outcome), then
  ``effect_folds`` HTR effect networks with the Stage 1 effect objective
  (``dina`` for a binary outcome, ``pseudo_outcome_mse`` by default for a
  continuous one). A DINA effect network is fitted against the sparse
  bag-of-words nuisances that Stage 1 cross-fits inside each effect fold
  (``stage1_dina.prepare_neural_nuisance``), and its ``tau = mu1 - mu0`` uses
  them; the HTR nuisance networks set effect-row eligibility
  (``r_stage_*_propensity``) and give the held-out ``e`` and ``m``. Held-out
  ``tau`` is the mean of the effect networks' predictions, and held-out ``e``
  and ``m`` are the means of the calibrated nuisance networks' predictions.
  Every inner network is a Stage 1 checkpoint unit, so an interrupted fold
  resumes at its next network.
* ``joint`` trains one network with propensity, outcome, and effect heads. The
  effect head fits the pseudo-outcome ``(Y - m) / (T - e)`` built from the
  network's own detached, in-sample nuisances (see :class:`JointHTRConfig`).

Fitting reads only the text, treatment, and outcome columns of the dataset.
Oracle columns are read by :func:`evaluate_against_oracle`, which first
verifies the SHA-256 of the frozen predictions.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import logging
import math
import multiprocessing
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import (
    AgenticAttentionVariableForestConfig,
    AppliedInferenceConfig,
    ModelArchitectureConfig,
    TrainingConfig,
    omit_inactive_htr_numeric_settings,
    resolve_effect_objective,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "oci_standalone_htr_v1"
ROW_ID = "_oci_row_id"
ORACLE_NAME_PREFIXES = ("true_", "oracle_", "ground_truth")
DEFAULT_TEMPLATE = (
    Path(__file__).resolve().parents[2] / "example_configs" / "all_evidence_stage1_model.json"
)
DEFAULT_HTR_MODEL = "prajjwal1/bert-tiny"
METHODS = ("two_stage", "joint")
DINA_NUISANCE_SOURCES = ("htr", "bow")
NUISANCE_NETWORKS = ("shared", "separate")
MLM_SETTINGS = (
    "mlm_epochs",
    "mlm_mask_probability",
    "mlm_learning_rate",
    "mlm_weight_decay",
    "mlm_batch_size",
)
MLM_SCOPES = ("network", "outer_fold")
DEFAULT_MLM_EPOCHS = 3
SELECTION_RULES = ("validation_loss", "last_epoch")
CALIBRATION_METHODS = ("none", "temperature", "isotonic", "temperature_isotonic")


@dataclass
class StandaloneHTRConfig:
    """Run settings shared by both methods."""

    method: str = "two_stage"
    outer_folds: int = 5
    seed: int = 42
    # Propensity clip in the AIPW scores (Stage 2's propensity_clip).
    aipw_clip: float = 0.02
    # two_stage with DINA: "htr" fits the effect networks against cross-fitted
    # HTR nuisances (propensity, mu0, mu1); "bow" keeps Stage 1's sparse
    # bag-of-words nuisances, cross-fitted inside each effect fold.
    dina_nuisances: str = "htr"
    # HTR DINA nuisance probabilities are clipped to [c, 1 - c]: DINA's baseline
    # nu = logit(mu) is unbounded, and isotonic calibration returns exact 0 and
    # 1 in pure tail blocks.
    dina_nuisance_clip: float = 0.01
    # HTR DINA nuisances: "shared" trains one network per nuisance fold with a
    # propensity head and one outcome head per arm; "separate" trains three
    # single-head networks, the propensity on every row and each arm's outcome
    # on that arm's rows only.
    nuisance_networks: str = "shared"
    # two_stage: masked-language-model pretraining of the chunk encoder before
    # each HTR network trains (htr_mlm_pretraining.py); 0 epochs turns it off.
    # Unset, it is 3 epochs for two_stage and 0 for joint, which does not
    # pretrain. mlm_scope "outer_fold" pretrains once on the outer training
    # fold's text, and every nuisance and effect network of the fold starts from
    # that encoder; "network" pretrains each network's encoder on the text of
    # that network's training rows only. Neither reads outer held-out text.
    mlm_epochs: Optional[int] = None
    mlm_scope: str = "outer_fold"
    mlm_mask_probability: float = 0.15
    mlm_learning_rate: float = 1e-4
    mlm_weight_decay: float = 0.01
    mlm_batch_size: int = 32

    def __post_init__(self) -> None:
        self.method = str(self.method).strip().lower()
        if self.method not in METHODS:
            raise ValueError(f"run.method must be one of {list(METHODS)}")
        self.dina_nuisances = str(self.dina_nuisances).strip().lower()
        if self.dina_nuisances not in DINA_NUISANCE_SOURCES:
            raise ValueError(
                f"run.dina_nuisances must be one of {list(DINA_NUISANCE_SOURCES)}"
            )
        self.nuisance_networks = str(self.nuisance_networks).strip().lower()
        if self.nuisance_networks not in NUISANCE_NETWORKS:
            raise ValueError(f"run.nuisance_networks must be one of {list(NUISANCE_NETWORKS)}")
        if int(self.outer_folds) < 2:
            raise ValueError("run.outer_folds must be >= 2")
        if not 0.0 < float(self.aipw_clip) < 0.5:
            raise ValueError("run.aipw_clip must be in (0, 0.5)")
        if not 0.0 < float(self.dina_nuisance_clip) < 0.5:
            raise ValueError("run.dina_nuisance_clip must be in (0, 0.5)")
        if self.mlm_epochs is None:
            self.mlm_epochs = DEFAULT_MLM_EPOCHS if self.method == "two_stage" else 0
        if isinstance(self.mlm_epochs, bool) or int(self.mlm_epochs) < 0:
            raise ValueError("run.mlm_epochs must be a nonnegative integer")
        self.mlm_scope = str(self.mlm_scope).strip().lower()
        if self.mlm_scope not in MLM_SCOPES:
            raise ValueError(f"run.mlm_scope must be one of {list(MLM_SCOPES)}")
        if not 0.0 < float(self.mlm_mask_probability) < 1.0:
            raise ValueError("run.mlm_mask_probability must be in (0, 1)")
        if float(self.mlm_learning_rate) <= 0 or float(self.mlm_weight_decay) < 0:
            raise ValueError("run.mlm_learning_rate must be positive and mlm_weight_decay >= 0")
        if int(self.mlm_batch_size) < 1:
            raise ValueError("run.mlm_batch_size must be >= 1")


@dataclass
class JointHTRConfig:
    """Settings of the ``joint`` method that the shared HTR configuration lacks.

    Optimization (learning rate, schedule, batch size, weight decay, gradient
    clipping, data-loader workers) and the propensity loss weight
    (``alpha_propensity``) come from :class:`TrainingConfig`; ``epochs`` here
    replaces ``training.epochs``.
    """

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

    def __post_init__(self) -> None:
        if int(self.epochs) < 1:
            raise ValueError("joint.epochs must be >= 1")
        if not 0 <= int(self.effect_warmup_epochs) < int(self.epochs):
            raise ValueError("joint.effect_warmup_epochs must be in [0, epochs)")
        if float(self.effect_loss_weight) <= 0:
            raise ValueError("joint.effect_loss_weight must be positive")
        if not 0.0 < float(self.e_clip) < 0.5:
            raise ValueError("joint.e_clip must be in (0, 0.5)")
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


@contextlib.contextmanager
def _preserved_random_state(device: Any):
    """Leave Python's, NumPy's, and Torch's global random streams as they were."""

    torch = _torch()
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    cuda = torch.device(device).type == "cuda" and torch.cuda.is_available()
    cuda_state = torch.cuda.get_rng_state(device) if cuda else None
    try:
        yield
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)


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
    settings: StandaloneHTRConfig,
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
            n_splits=int(settings.outer_folds),
            seed=int(settings.seed),
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
    *,
    seed: int,
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
    seed = int(seed) + 7_919 * int(fold)
    try:
        train, validation = train_test_split(
            fit_idx, test_size=float(joint.validation_fraction), random_state=seed, stratify=strata
        )
    except ValueError:
        train, validation = train_test_split(
            fit_idx, test_size=float(joint.validation_fraction), random_state=seed
        )
    return np.sort(train), np.sort(validation)


def _config_record(
    config: AppliedInferenceConfig, settings: StandaloneHTRConfig, joint: JointHTRConfig
) -> Dict[str, Any]:
    architecture = asdict(config.architecture)
    record: Dict[str, Any] = {
        "outcome_type": config.outcome_type,
        "text_column": config.text_column,
        "treatment_column": config.treatment_column,
        "outcome_column": config.outcome_column,
        "architecture": omit_inactive_htr_numeric_settings(
            {
                key: value
                for key, value in architecture.items()
                if key.startswith("htr_") or key == "feature_extractor_type"
            }
        ),
        "training": asdict(config.training),
        "run": asdict(settings),
    }
    # Only a two-stage DINA fit with HTR nuisances reads these. Stage 1's
    # bag-of-words nuisances leave them out, as before the settings existed, so
    # those runs keep their identities and checkpoints.
    if not _dina_uses_htr_nuisances(config, settings):
        record["run"].pop("dina_nuisances")
        record["run"].pop("dina_nuisance_clip")
    # Likewise for the shared nuisance networks, the only kind before the
    # setting existed.
    if settings.nuisance_networks == "shared":
        record["run"].pop("nuisance_networks")
    if int(settings.mlm_epochs) == 0:
        for key in MLM_SETTINGS:
            record["run"].pop(key)
    # Per-network pretraining was the only scope before the setting existed.
    if int(settings.mlm_epochs) == 0 or settings.mlm_scope == "network":
        record["run"].pop("mlm_scope")
    if settings.method == "two_stage":
        stage1_htr = config.architecture.agentic_attention_variable_forest
        record["stage1_htr"] = asdict(stage1_htr)
        record["effect_objective"] = resolve_effect_objective(
            stage1_htr.effect_objective, outcome_type=config.outcome_type
        )
    else:
        record["joint"] = asdict(joint)
    return record


# ---------------------------------------------------------------------------
# Folds


def _fit_joint_fold(
    *,
    frame: pd.DataFrame,
    config: AppliedInferenceConfig,
    settings: StandaloneHTRConfig,
    joint: JointHTRConfig,
    fold: int,
    fit_idx: np.ndarray,
    test_df: pd.DataFrame,
    device: str,
    fold_dir: Path,
):
    """One joint network: calibrated held-out e and m, and tau from its effect head."""

    from ..utils.calibration import BinaryProbabilityCalibrator
    from .stage1_checkpointing import seed_everything

    seed_everything(int(settings.seed) + 1_000 * int(fold))
    train_idx, validation_idx = validation_split(
        frame, fit_idx, config, joint, fold, seed=int(settings.seed)
    )
    trainer = _trainer_class()(config=config, joint=joint, device=device)
    model = trainer.create_model()
    try:
        result = trainer.train_joint(model, frame, train_idx, validation_idx, fold=fold)
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
    _write_json(fold_dir / "history.json", result["history"])
    record = {
        "selected_epoch": result["selected_epoch"],
        "rows": {
            "train": len(train_idx),
            "validation": len(validation_idx),
            "heldout": len(test_df),
        },
        "calibration": calibration,
    }
    return e_hat, m_hat, tau, {"propensity_raw": e_raw, "m_hat_raw": m_raw}, record


def build_arm_nuisance_net(extractor: Any, hidden_dim: int):
    """An HTR nuisance network with a propensity head and one outcome head per arm."""

    import torch.nn as nn

    from .htr_trainer import _configured_hidden_head

    class ArmNuisanceNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.extractor = extractor
            # NuisanceNet's shared hidden layer, with mu0 and mu1 heads in place of m.
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
            self.outcome0 = nn.Linear(hidden_dim, 1)
            self.outcome1 = nn.Linear(hidden_dim, 1)

        def forward(self, texts_or_batch):
            features = self.extractor(
                texts_or_batch if isinstance(texts_or_batch, dict) else list(texts_or_batch)
            )
            hidden = self.shared(features)
            return (
                self.propensity(hidden).squeeze(-1),
                self.outcome0(hidden).squeeze(-1),
                self.outcome1(hidden).squeeze(-1),
            )

    return ArmNuisanceNet()


def build_single_head_net(extractor: Any, hidden_dim: int):
    """An HTR network with one logit head, for one nuisance."""

    import torch.nn as nn

    from .htr_trainer import _configured_hidden_head

    class SingleHeadNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.extractor = extractor
            # ArmNuisanceNet's hidden layer, with a single head.
            self.hidden = _configured_hidden_head(
                input_dim=int(extractor.output_dim),
                hidden_dim=hidden_dim,
                depth=1,
                activation="relu",
                dropout=0.1,
                layer_norm=False,
                bias=True,
            )
            self.output = nn.Linear(hidden_dim, 1)

        def forward(self, texts_or_batch):
            features = self.extractor(
                texts_or_batch if isinstance(texts_or_batch, dict) else list(texts_or_batch)
            )
            return self.output(self.hidden(features)).squeeze(-1)

    return SingleHeadNet()


def _two_stage_trainer_class():
    from .htr_trainer import HTRTrainer, current_lr, make_linear_lr_scheduler

    class TwoStageHTRTrainer(HTRTrainer):
        """Stage 1's HTR trainer without attention evidence, with optional HTR DINA nuisances.

        When ``dina_nuisance_table`` (``e``, ``mu0``, ``mu1`` indexed by row ID) is
        set, DINA effect networks train and predict against it in place of Stage
        1's per-effect-fold bag-of-words nuisances. When ``mlm`` (the checkpoint
        store and run settings) is set, every network's chunk encoder starts
        from masked-LM pretraining, on its own training rows' text or, with
        ``run.mlm_scope=outer_fold``, on ``mlm_corpus`` (the outer training fold).
        """

        dina_nuisance_table: Optional[pd.DataFrame] = None
        mlm: Optional[Tuple[Any, StandaloneHTRConfig]] = None
        mlm_corpus: Optional[pd.DataFrame] = None

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.mlm_records: Dict[str, Dict[str, Any]] = {}

        def _attention_evidence(self, *args, **kwargs):
            # Attention records feed Stage 2 interpretation; they never change a fit.
            return []

        def apply_mlm_encoder(self, model, df: pd.DataFrame, positions, *, unit: str) -> None:
            """Load masked-LM pretrained encoder weights for the rows at ``positions``.

            The pretraining is its own checkpoint unit, keyed by those rows, and
            runs with every global random stream saved and restored, so the
            network draws the same numbers whether it was computed or reloaded.
            """

            if self.mlm is None:
                return
            from .htr_mlm_pretraining import load_encoder_weights, pretrain_chunk_encoder
            from .stage1_checkpointing import row_id_identity

            store, settings = self.mlm
            corpus = df.iloc[np.asarray(positions, dtype=int)]

            def compute() -> Dict[str, Any]:
                extractor = self._create_extractor()
                try:
                    return pretrain_chunk_encoder(
                        extractor,
                        corpus[self.config.text_column].astype(str).tolist(),
                        epochs=int(settings.mlm_epochs),
                        mask_probability=float(settings.mlm_mask_probability),
                        learning_rate=float(settings.mlm_learning_rate),
                        weight_decay=float(settings.mlm_weight_decay),
                        batch_size=int(settings.mlm_batch_size),
                        gradient_clip_norm=float(self.config.training.gradient_clip_norm),
                        label=unit,
                    )
                finally:
                    self._cleanup_model(extractor)

            with _preserved_random_state(self.device):
                value = store.run(
                    unit,
                    compute,
                    parameters={
                        "objective": "masked_language_model",
                        **{key: getattr(settings, key) for key in MLM_SETTINGS},
                    },
                    rows={"corpus": row_id_identity(corpus[ROW_ID].to_numpy())},
                    deterministic_seed=True,
                ).value
            load_encoder_weights(model.extractor, value["encoder"])
            self.mlm_records[unit] = {
                "rows": int(len(corpus)),
                "chunks": int(value["chunks"]),
                "head": value["head"],
                "history": value["history"],
            }

        def apply_network_mlm_encoder(
            self, model, df: pd.DataFrame, positions, *, kind: str, fold: int
        ) -> None:
            """Pretrained encoder weights for one nuisance or effect network of inner ``fold``."""

            if self.mlm is None:
                return
            if self.mlm[1].mlm_scope == "outer_fold":
                corpus = self.mlm_corpus
                unit = "htr/mlm/outer_fold"
                self.apply_mlm_encoder(model, corpus, np.arange(len(corpus)), unit=unit)
            else:
                unit = f"htr/mlm/{kind}/fold_{int(fold):03d}"
                self.apply_mlm_encoder(model, df, positions, unit=unit)

        def _train_nuisance_model(self, model, df, positions, outer_fold, fold, total_folds):
            self.apply_network_mlm_encoder(model, df, positions, kind="nuisance", fold=fold)
            return super()._train_nuisance_model(
                model, df, positions, outer_fold=outer_fold, fold=fold, total_folds=total_folds
            )

        def _train_effect_model(
            self,
            model,
            df,
            positions,
            outcomes,
            treatments,
            e_hat,
            m_hat,
            y_residuals,
            t_residuals,
            outer_fold,
            fold,
            total_folds,
        ):
            self.apply_network_mlm_encoder(model, df, positions, kind="effect", fold=fold)
            return super()._train_effect_model(
                model,
                df,
                positions,
                outcomes,
                treatments,
                e_hat,
                m_hat,
                y_residuals,
                t_residuals,
                outer_fold=outer_fold,
                fold=fold,
                total_folds=total_folds,
            )

        def _dina_rows(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
            from ..models import dina

            values = self.dina_nuisance_table.loc[df[ROW_ID].to_numpy(dtype=int)]
            return dina.nuisances(values["e"], values["mu0"], values["mu1"])

        def _dina_effect_nuisance(self, model, df, positions, *, seed):
            if self.dina_nuisance_table is None:
                return super()._dina_effect_nuisance(model, df, positions, seed=seed)
            positions = np.asarray(positions, dtype=int)
            nuisance = self._dina_rows(df.iloc[positions])
            fields = {key: np.zeros(len(df), np.float32) for key in ("dina_a", "dina_nu", "dina_e")}
            for key in fields:
                fields[key][positions] = nuisance[key[len("dina_") :]]
            return fields

        def _dina_effect_predictions(self, model, df, delta):
            if self.dina_nuisance_table is None:
                return super()._dina_effect_predictions(model, df, delta)
            from ..models import dina

            nuisance = self._dina_rows(df)
            mu0, mu1 = dina.counterfactuals(delta, nuisance["a"], nuisance["nu"])
            return {
                **nuisance,
                "tau": mu1 - mu0,
                "delta": np.asarray(delta),
                "mu0": mu0,
                "mu1": mu1,
            }

        def train_arm_nuisance(self, model, df, positions, *, outer_fold, fold, total_folds):
            """Stage 1's nuisance training, with each outcome head fitted on its own arm."""

            torch = _torch()
            import torch.nn.functional as F

            train_config = self.config.training
            stage1_htr = self.avf_config
            epochs = int(
                stage1_htr.nuisance_epochs
                if stage1_htr.nuisance_epochs is not None
                else train_config.epochs
            )
            weight_decay = float(
                stage1_htr.nuisance_weight_decay
                if stage1_htr.nuisance_weight_decay is not None
                else train_config.weight_decay
            )
            smoothing = float(stage1_htr.nuisance_label_smoothing)
            alpha = float(train_config.alpha_propensity)
            model.extractor.fit_tokenizer(
                df.iloc[positions][self.config.text_column].astype(str).tolist()
            )
            self._assert_htr_sentence_encoder_training_state(model.extractor)
            loader = self._make_text_loader(
                model,
                df,
                positions,
                fields={
                    "t": df[self.config.treatment_column].to_numpy(dtype=np.float32),
                    "y": df[self.config.outcome_column].to_numpy(dtype=np.float32),
                },
                shuffle=True,
                total_folds=total_folds,
            )
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=train_config.learning_rate,
                weight_decay=weight_decay,
            )
            self._assert_htr_sentence_encoder_optimizer_coverage(model.extractor, optimizer)
            num_batches = max(1, len(loader))
            scheduler = make_linear_lr_scheduler(
                optimizer, train_config, num_batches, epochs_override=epochs
            )
            progress_every = max(1, num_batches // 5)
            logger.info(
                "Outer fold %s arm nuisance fold %s/%s: training for %s epoch(s), "
                "batch_size=%s, batches/epoch=%s, dataloader_workers=%s, lr=%.3g, "
                "weight_decay=%.3g, label_smoothing=%.3g%s",
                outer_fold,
                fold,
                total_folds,
                epochs,
                train_config.batch_size,
                num_batches,
                loader.num_workers,
                current_lr(optimizer),
                weight_decay,
                smoothing,
                self._cuda_memory_summary(),
            )
            for epoch in range(1, epochs + 1):
                model.train()
                sums = torch.zeros(3, device=self.device, dtype=torch.float64)
                batch_count = 0
                for batch_idx, batch in enumerate(loader, start=1):
                    t = batch["t"].to(self.device, non_blocking=True)
                    y = batch["y"].to(self.device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    t_logit, y0_logit, y1_logit = model(batch["model_input"])
                    t_target = t * (1.0 - smoothing) + 0.5 * smoothing
                    y_target = y * (1.0 - smoothing) + 0.5 * smoothing
                    propensity_loss = F.binary_cross_entropy_with_logits(t_logit, t_target)
                    # Each row trains only the outcome head of the arm it received.
                    outcome_loss = F.binary_cross_entropy_with_logits(
                        torch.where(t > 0.5, y1_logit, y0_logit), y_target
                    )
                    loss = outcome_loss + alpha * propensity_loss
                    loss.backward()
                    self._clip_and_step(model, optimizer, scheduler)
                    batch_count += 1
                    sums.add_(
                        torch.stack(
                            (loss.detach(), propensity_loss.detach(), outcome_loss.detach())
                        ).to(torch.float64)
                    )
                    if batch_idx == 1 or batch_idx == num_batches or batch_idx % progress_every == 0:
                        logger.info(
                            "Outer fold %s arm nuisance fold %s/%s epoch %s/%s batch %s/%s "
                            "lr=%.3g%s",
                            outer_fold,
                            fold,
                            total_folds,
                            epoch,
                            epochs,
                            batch_idx,
                            num_batches,
                            current_lr(optimizer),
                            self._cuda_memory_summary(),
                        )
                loss_mean, propensity_mean, outcome_mean = (
                    (sums / max(1, batch_count)).cpu().tolist()
                )
                logger.info(
                    "Outer fold %s arm nuisance fold %s/%s epoch %s/%s complete: "
                    "loss=%.4f outcome=%.4f propensity=%.4f lr=%.3g%s",
                    outer_fold,
                    fold,
                    total_folds,
                    epoch,
                    epochs,
                    loss_mean,
                    outcome_mean,
                    propensity_mean,
                    current_lr(optimizer),
                    self._cuda_memory_summary(),
                )

        def predict_arm_nuisance(self, model, df: pd.DataFrame) -> Dict[str, np.ndarray]:
            torch = _torch()
            model.eval()
            loader = self._make_text_loader(model, df, np.arange(len(df), dtype=int))
            parts: Dict[str, List[np.ndarray]] = {"e": [], "mu0": [], "mu1": []}
            with torch.no_grad():
                for batch in loader:
                    for key, logit in zip(parts, model(batch["model_input"]), strict=True):
                        parts[key].append(torch.sigmoid(logit).cpu().numpy())
            return {key: np.concatenate(values).astype(float) for key, values in parts.items()}

        def train_single_head(
            self, model, df, positions, target_column, *, name, outer_fold, fold, total_folds
        ):
            """Nuisance training of a single-head network on ``positions`` against one column."""

            torch = _torch()
            import torch.nn.functional as F

            train_config = self.config.training
            stage1_htr = self.avf_config
            epochs = int(
                stage1_htr.nuisance_epochs
                if stage1_htr.nuisance_epochs is not None
                else train_config.epochs
            )
            weight_decay = float(
                stage1_htr.nuisance_weight_decay
                if stage1_htr.nuisance_weight_decay is not None
                else train_config.weight_decay
            )
            smoothing = float(stage1_htr.nuisance_label_smoothing)
            model.extractor.fit_tokenizer(
                df.iloc[positions][self.config.text_column].astype(str).tolist()
            )
            self._assert_htr_sentence_encoder_training_state(model.extractor)
            loader = self._make_text_loader(
                model,
                df,
                positions,
                fields={"target": df[target_column].to_numpy(dtype=np.float32)},
                shuffle=True,
                total_folds=total_folds,
            )
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=train_config.learning_rate,
                weight_decay=weight_decay,
            )
            self._assert_htr_sentence_encoder_optimizer_coverage(model.extractor, optimizer)
            num_batches = max(1, len(loader))
            scheduler = make_linear_lr_scheduler(
                optimizer, train_config, num_batches, epochs_override=epochs
            )
            progress_every = max(1, num_batches // 5)
            logger.info(
                "Outer fold %s %s nuisance fold %s/%s: training for %s epoch(s) on %s rows, "
                "batch_size=%s, batches/epoch=%s, dataloader_workers=%s, lr=%.3g, "
                "weight_decay=%.3g, label_smoothing=%.3g%s",
                outer_fold,
                name,
                fold,
                total_folds,
                epochs,
                len(positions),
                train_config.batch_size,
                num_batches,
                loader.num_workers,
                current_lr(optimizer),
                weight_decay,
                smoothing,
                self._cuda_memory_summary(),
            )
            for epoch in range(1, epochs + 1):
                model.train()
                total = torch.zeros((), device=self.device, dtype=torch.float64)
                batch_count = 0
                for batch_idx, batch in enumerate(loader, start=1):
                    target = batch["target"].to(self.device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.binary_cross_entropy_with_logits(
                        model(batch["model_input"]), target * (1.0 - smoothing) + 0.5 * smoothing
                    )
                    loss.backward()
                    self._clip_and_step(model, optimizer, scheduler)
                    batch_count += 1
                    total.add_(loss.detach().to(torch.float64))
                    if batch_idx == 1 or batch_idx == num_batches or batch_idx % progress_every == 0:
                        logger.info(
                            "Outer fold %s %s nuisance fold %s/%s epoch %s/%s batch %s/%s "
                            "lr=%.3g%s",
                            outer_fold,
                            name,
                            fold,
                            total_folds,
                            epoch,
                            epochs,
                            batch_idx,
                            num_batches,
                            current_lr(optimizer),
                            self._cuda_memory_summary(),
                        )
                logger.info(
                    "Outer fold %s %s nuisance fold %s/%s epoch %s/%s complete: loss=%.4f "
                    "lr=%.3g%s",
                    outer_fold,
                    name,
                    fold,
                    total_folds,
                    epoch,
                    epochs,
                    float(total.item()) / max(1, batch_count),
                    current_lr(optimizer),
                    self._cuda_memory_summary(),
                )

        def predict_single_head(self, model, df: pd.DataFrame) -> np.ndarray:
            torch = _torch()
            model.eval()
            loader = self._make_text_loader(model, df, np.arange(len(df), dtype=int))
            parts: List[np.ndarray] = []
            with torch.no_grad():
                for batch in loader:
                    parts.append(torch.sigmoid(model(batch["model_input"])).cpu().numpy())
            return np.concatenate(parts).astype(float)

    return TwoStageHTRTrainer


def _stage1_htr_provider(config: AppliedInferenceConfig, device: str, store: Any):
    """Stage 1's HTR provider, running :func:`_two_stage_trainer_class`'s trainer."""

    from .multi_model_forest_stage1 import MultiModelForestStage1HTRProvider

    trainer_class = _two_stage_trainer_class()

    class _Provider(MultiModelForestStage1HTRProvider):
        def _ensure_runner(self, discovery_df):
            del discovery_df
            if self._runner is None:
                self._runner = trainer_class(
                    config=self.config, device=self.device, num_workers=self.num_workers
                )
            return self._runner

    # The trainer reserves one CPU and gives the rest to data-loader workers.
    return _Provider(
        config=config,
        device=device,
        checkpoint_store=store,
        num_workers=1 + int(config.training.dataloader_workers),
    )


def _fit_arm_nuisance_ensemble(
    trainer: Any,
    store: Any,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    outer_fold: int,
    *,
    networks: str = "shared",
) -> Dict[str, Any]:
    """Cross-fitted HTR e, mu0, and mu1 on Stage 1's nuisance folds, calibrated out of fold.

    ``networks`` is ``run.nuisance_networks``: one three-head network per fold
    (``shared``), or one single-head network per nuisance and fold
    (``separate``).
    """

    from sklearn.model_selection import KFold

    from .htr_trainer import cross_fitted_calibrators
    from .multi_model_forest_stage1 import _bounded_fold_count
    from .stage1_checkpointing import row_id_identity

    config = trainer.config
    stage1_htr = trainer.avf_config
    folds = _bounded_fold_count(stage1_htr.nuisance_folds, len(train_df))
    splits = list(
        enumerate(
            # Stage 1's HTR nuisance folds for this outer fold.
            KFold(n_splits=folds, shuffle=True, random_state=10_000 + int(outer_fold)).split(
                train_df
            ),
            start=1,
        )
    )
    hidden_dim = int(config.architecture.htr_prediction_head_hidden_dim)

    def compute(fold: int, fit_pos: np.ndarray, heldout_pos: np.ndarray) -> Dict[str, Any]:
        model = build_arm_nuisance_net(trainer._create_extractor(), hidden_dim).to(trainer.device)
        try:
            trainer.apply_network_mlm_encoder(model, train_df, fit_pos, kind="nuisance", fold=fold)
            trainer.train_arm_nuisance(
                model, train_df, fit_pos, outer_fold=outer_fold, fold=fold, total_folds=folds
            )
            return {
                "fold": int(fold),
                "heldout_pos": heldout_pos,
                "heldout": trainer.predict_arm_nuisance(model, train_df.iloc[heldout_pos]),
                "test": trainer.predict_arm_nuisance(model, test_df),
            }
        finally:
            trainer._cleanup_model(model)

    def run(fold: int, fit_pos: np.ndarray, heldout_pos: np.ndarray) -> Dict[str, Any]:
        fit_pos = np.asarray(fit_pos, dtype=int)
        heldout_pos = np.asarray(heldout_pos, dtype=int)
        return store.run(
            f"htr/arm_nuisance/fold_{int(fold):03d}",
            lambda: compute(fold, fit_pos, heldout_pos),
            parameters={
                "outer_fold": int(outer_fold),
                "inner_fold": int(fold),
                "total_folds": int(folds),
                "objective": "treatment_and_arm_outcome_nuisance",
            },
            rows={
                "fit": row_id_identity(train_df.iloc[fit_pos][ROW_ID].to_numpy()),
                "validation": row_id_identity(train_df.iloc[heldout_pos][ROW_ID].to_numpy()),
                "heldout": row_id_identity(test_df[ROW_ID].to_numpy()),
            },
            deterministic_seed=True,
        ).value

    t_all = train_df[config.treatment_column].to_numpy(dtype=float)
    # Each separate network: its unit objective, the treatment arm it trains on
    # (None for every row), and its target column.
    single_heads = {
        "e": ("propensity_nuisance", None, config.treatment_column),
        "mu0": ("control_outcome_nuisance", 0, config.outcome_column),
        "mu1": ("treated_outcome_nuisance", 1, config.outcome_column),
    }

    def compute_single(
        name: str, fold: int, fit_pos: np.ndarray, train_pos: np.ndarray, heldout_pos: np.ndarray
    ) -> Dict[str, np.ndarray]:
        model = build_single_head_net(trainer._create_extractor(), hidden_dim).to(trainer.device)
        try:
            # The fold's three networks share one pretraining (of all its fit rows
            # under the network scope).
            trainer.apply_network_mlm_encoder(model, train_df, fit_pos, kind="nuisance", fold=fold)
            trainer.train_single_head(
                model,
                train_df,
                train_pos,
                single_heads[name][2],
                name=name,
                outer_fold=outer_fold,
                fold=fold,
                total_folds=folds,
            )
            return {
                "heldout": trainer.predict_single_head(model, train_df.iloc[heldout_pos]),
                "test": trainer.predict_single_head(model, test_df),
            }
        finally:
            trainer._cleanup_model(model)

    def run_single(
        name: str, fold: int, fit_pos: np.ndarray, heldout_pos: np.ndarray
    ) -> Dict[str, np.ndarray]:
        objective, arm, _ = single_heads[name]
        train_pos = fit_pos if arm is None else fit_pos[t_all[fit_pos] == arm]
        if len(train_pos) == 0:
            raise ValueError(
                f"outer fold {outer_fold} nuisance fold {fold} has no rows in treatment arm {arm}"
            )
        return store.run(
            f"htr/single_head/{name}/fold_{int(fold):03d}",
            lambda: compute_single(name, fold, fit_pos, train_pos, heldout_pos),
            parameters={
                "outer_fold": int(outer_fold),
                "inner_fold": int(fold),
                "total_folds": int(folds),
                "objective": objective,
            },
            rows={
                "fit": row_id_identity(train_df.iloc[train_pos][ROW_ID].to_numpy()),
                "validation": row_id_identity(train_df.iloc[heldout_pos][ROW_ID].to_numpy()),
                "heldout": row_id_identity(test_df[ROW_ID].to_numpy()),
            },
            deterministic_seed=True,
        ).value

    def run_separate(fold: int, fit_pos: np.ndarray, heldout_pos: np.ndarray) -> Dict[str, Any]:
        fit_pos = np.asarray(fit_pos, dtype=int)
        heldout_pos = np.asarray(heldout_pos, dtype=int)
        values = {name: run_single(name, fold, fit_pos, heldout_pos) for name in single_heads}
        return {
            "fold": int(fold),
            "heldout_pos": heldout_pos,
            "heldout": {name: value["heldout"] for name, value in values.items()},
            "test": {name: value["test"] for name, value in values.items()},
        }

    if networks == "separate":
        results = [
            run_separate(fold, fit_pos, heldout_pos) for fold, (fit_pos, heldout_pos) in splits
        ]
        units = [
            f"htr/single_head/{name}/fold_{fold:03d}" for fold, _ in splits for name in single_heads
        ]
    else:
        results = [run(fold, fit_pos, heldout_pos) for fold, (fit_pos, heldout_pos) in splits]
        units = [f"htr/arm_nuisance/fold_{fold:03d}" for fold, _ in splits]
    names = ("e", "mu0", "mu1")
    raw = {name: np.full(len(train_df), np.nan) for name in names}
    fold_of_row = np.zeros(len(train_df), dtype=int)
    for result in results:
        for name in names:
            raw[name][result["heldout_pos"]] = result["heldout"][name]
        fold_of_row[result["heldout_pos"]] = result["fold"]
    t = train_df[config.treatment_column].to_numpy(dtype=float)
    y = train_df[config.outcome_column].to_numpy(dtype=float)
    method = stage1_htr.nuisance_calibration
    # Fold k's calibrators see only other folds' out-of-fold predictions; each
    # outcome head is calibrated on the rows of its own arm.
    calibrators = {
        "e": cross_fitted_calibrators(raw["e"], t, fold_of_row, method=method),
        "mu0": cross_fitted_calibrators(raw["mu0"][t == 0], y[t == 0], fold_of_row[t == 0], method=method),
        "mu1": cross_fitted_calibrators(raw["mu1"][t == 1], y[t == 1], fold_of_row[t == 1], method=method),
    }
    calibrated = {name: np.full(len(train_df), np.nan) for name in names}
    test_parts: Dict[str, List[np.ndarray]] = {name: [] for name in names}
    for result in results:
        fold = result["fold"]
        for name in names:
            calibrated[name][result["heldout_pos"]] = calibrators[name][fold].transform(
                result["heldout"][name]
            )
            test_parts[name].append(calibrators[name][fold].transform(result["test"][name]))
    test_values = {name: np.mean(np.vstack(parts), axis=0) for name, parts in test_parts.items()}

    def frame(rows: np.ndarray, values: Mapping[str, np.ndarray]) -> pd.DataFrame:
        e, mu0, mu1 = values["e"], values["mu0"], values["mu1"]
        return pd.DataFrame(
            {
                ROW_ID: rows,
                "e_hat": e,
                "mu0_hat": mu0,
                "mu1_hat": mu1,
                "m_hat": e * mu1 + (1.0 - e) * mu0,
            }
        )

    train_predictions = frame(train_df[ROW_ID].to_numpy(dtype=int), calibrated)
    for name in names:
        train_predictions[f"{name}_hat_raw"] = raw[name]
    train_predictions["nuisance_fold"] = fold_of_row
    test_predictions = frame(test_df[ROW_ID].to_numpy(dtype=int), test_values)
    return {
        "train": {"predictions": train_predictions},
        "test_predictions": test_predictions,
        "units": units,
    }


def _dina_uses_htr_nuisances(config: AppliedInferenceConfig, settings: StandaloneHTRConfig) -> bool:
    if settings.method != "two_stage" or settings.dina_nuisances != "htr":
        return False
    objective = resolve_effect_objective(
        config.architecture.agentic_attention_variable_forest.effect_objective,
        outcome_type=config.outcome_type,
    )
    return objective == "dina"


def _fit_two_stage_fold(
    *,
    frame: pd.DataFrame,
    config: AppliedInferenceConfig,
    settings: StandaloneHTRConfig,
    fold: int,
    fit_idx: np.ndarray,
    test_df: pd.DataFrame,
    device: str,
    fold_dir: Path,
    identity: Mapping[str, Any],
):
    """Cross-fitted HTR nuisance and effect ensembles on one outer training fold."""

    from .multi_model_forest_stage1 import _bounded_fold_count
    from .stage1_checkpointing import Stage1CheckpointStore, row_id_identity

    train_df = frame.iloc[fit_idx].reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    stage1_htr = config.architecture.agentic_attention_variable_forest
    effect_objective = resolve_effect_objective(
        stage1_htr.effect_objective, outcome_type=config.outcome_type
    )
    htr_dina_nuisances = _dina_uses_htr_nuisances(config, settings)
    store = Stage1CheckpointStore(
        fold_dir / "stage1_checkpoints",
        context_identity={"standalone_htr_fold_sha256": _sha256_json(identity)},
        base_seed=int(settings.seed),
    )
    provider = _stage1_htr_provider(config, device, store)
    nuisance_folds = _bounded_fold_count(stage1_htr.nuisance_folds, len(train_df))
    trainer = provider._ensure_runner(train_df)
    if int(settings.mlm_epochs) > 0:
        trainer.mlm = (store, settings)
        trainer.mlm_corpus = train_df
    with store.context_lock():
        if htr_dina_nuisances:
            nuisance = _fit_arm_nuisance_ensemble(
                trainer,
                store,
                train_df,
                test_df,
                int(fold),
                networks=settings.nuisance_networks,
            )
            nuisance_units = nuisance["units"]
        else:
            nuisance = provider.fit_nuisance_inner_ensemble_predict(train_df, test_df, int(fold))
            nuisance_units = [
                f"htr/nuisance/fold_{inner:03d}" for inner in range(1, nuisance_folds + 1)
            ]
        # Effect units depend on this unit, so changed nuisance networks refit the effects.
        nuisance_train = store.run(
            "assembly/nuisance_ensemble",
            lambda: nuisance["train"]["predictions"],
            parameters={
                "outer_fold": int(fold),
                "sources": (
                    ["htr"]
                    if not htr_dina_nuisances
                    else ["htr_arms"]
                    if settings.nuisance_networks == "shared"
                    else ["htr_single_heads"]
                ),
            },
            dependencies=tuple(nuisance_units),
            rows={
                "train": row_id_identity(train_df[ROW_ID].to_numpy()),
                "heldout": row_id_identity(test_df[ROW_ID].to_numpy()),
            },
        ).value
        test_nuisance = nuisance["test_predictions"]
        if htr_dina_nuisances:
            # Out-of-fold values for outer training rows, ensemble means for held-out rows.
            clip = float(settings.dina_nuisance_clip)
            trainer.dina_nuisance_table = (
                pd.concat([nuisance_train, test_nuisance], ignore_index=True)
                .set_index(ROW_ID)[["e_hat", "mu0_hat", "mu1_hat"]]
                .rename(columns={"e_hat": "e", "mu0_hat": "mu0", "mu1_hat": "mu1"})
                .clip(lower=clip, upper=1.0 - clip)
            )
        effect = provider.fit_effect_variant_inner_ensemble_predict(
            train_df, test_df, nuisance_train, int(fold), effect_objective=effect_objective
        )
        store.mark_context_complete()
    test_effect = effect["test_predictions"]
    for name, part in (("nuisance", test_nuisance), ("effect", test_effect)):
        if not np.array_equal(part[ROW_ID].to_numpy(), test_df[ROW_ID].to_numpy()):
            raise RuntimeError(f"Stage 1 HTR {name} predictions are not aligned to the fold")
    # Outer-training rows' inner out-of-fold nuisances and effects, for diagnostics.
    effect["train"]["predictions"].to_parquet(fold_dir / "inner_oof.parquet", index=False)
    record = {
        "effect_objective": effect_objective,
        "inner_folds": {
            "nuisance": int(nuisance_folds),
            "effect": int(_bounded_fold_count(stage1_htr.effect_folds, len(train_df))),
        },
        "rows": {"train": len(train_df), "heldout": len(test_df)},
        "reused_checkpoint_units": list(store.reused_units),
    }
    if effect_objective == "dina":
        record["dina_nuisances"] = "htr" if htr_dina_nuisances else "bow"
    if htr_dina_nuisances:
        record["nuisance_networks"] = settings.nuisance_networks
    if int(settings.mlm_epochs) > 0:
        # Pretraining computed or reloaded in this session, added to earlier sessions'.
        mlm_path = fold_dir / "mlm_pretraining.json"
        records = json.loads(mlm_path.read_text()) if mlm_path.exists() else {}
        records.update(trainer.mlm_records)
        _write_json(mlm_path, dict(sorted(records.items())))
        record["mlm_units"] = sorted(records)
    return (
        test_nuisance["e_hat"].to_numpy(dtype=float),
        test_nuisance["m_hat"].to_numpy(dtype=float),
        test_effect["tau_hat_r_stage"].to_numpy(dtype=float),
        {},
        record,
    )


def _method_e_clip(
    config: AppliedInferenceConfig, settings: StandaloneHTRConfig, joint: JointHTRConfig
) -> float:
    if settings.method == "two_stage":
        return float(config.architecture.agentic_attention_variable_forest.e_clip)
    return float(joint.e_clip)


def run_fold(
    *,
    frame: pd.DataFrame,
    config: AppliedInferenceConfig,
    settings: StandaloneHTRConfig,
    joint: JointHTRConfig,
    fold: int,
    fit_idx: np.ndarray,
    test_idx: np.ndarray,
    device: str,
    fold_dir: Path,
    identity: Mapping[str, Any],
) -> None:
    """Fit one outer training fold with the configured method and freeze its held-out rows."""

    from .stage2_diagnostics import dr_scores

    fold_dir.mkdir(parents=True, exist_ok=True)
    test_df = frame.iloc[test_idx]
    common = dict(
        frame=frame,
        config=config,
        settings=settings,
        fold=fold,
        fit_idx=fit_idx,
        test_df=test_df,
        device=device,
        fold_dir=fold_dir,
    )
    if settings.method == "two_stage":
        e_hat, m_hat, tau, extra, record = _fit_two_stage_fold(**common, identity=identity)
    else:
        e_hat, m_hat, tau, extra, record = _fit_joint_fold(**common, joint=joint)
    t = test_df[config.treatment_column].to_numpy(dtype=float)
    y = test_df[config.outcome_column].to_numpy(dtype=float)
    # mu0 and mu1 follow from m = e * mu1 + (1 - e) * mu0 and tau = mu1 - mu0.
    mu0 = m_hat - e_hat * tau
    mu1 = m_hat + (1.0 - e_hat) * tau
    e_clip = _method_e_clip(config, settings, joint)
    predictions = pd.DataFrame(
        {
            ROW_ID: test_df[ROW_ID].to_numpy(dtype=int),
            "outer_fold": int(fold),
            "treatment": t,
            "outcome": y,
            "propensity": e_hat,
            "m_hat": m_hat,
            "estimated_cate": tau,
            "mu0": mu0,
            "mu1": mu1,
            "y_residual": y - m_hat,
            "t_residual": t - np.clip(e_hat, e_clip, 1.0 - e_clip),
            "aipw_score": dr_scores(y, t, e_hat, mu0, mu1, clip=float(settings.aipw_clip)),
            **extra,
        }
    )
    predictions.to_parquet(fold_dir / "predictions.parquet", index=False)
    _write_json(
        fold_dir / "fold.json",
        {
            "schema_version": SCHEMA_VERSION,
            "method": settings.method,
            "identity": dict(identity),
            "identity_sha256": _sha256_json(identity),
            **record,
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
    """Spawned per worker slot: run its folds serially on its device."""

    output_dir = Path(payload["output_dir"])
    device = str(payload["device"])
    _configure_logging(
        output_dir / "logs" / f"worker_{int(payload['slot']):02d}_{device.replace(':', '_')}.log"
    )
    config = payload["config"]
    frame = load_training_frame(Path(payload["dataset"]), config)
    for fold, fit_idx, test_idx, identity in payload["folds"]:
        run_fold(
            frame=frame,
            config=config,
            settings=payload["settings"],
            joint=payload["joint"],
            fold=fold,
            fit_idx=fit_idx,
            test_idx=test_idx,
            device=device,
            fold_dir=output_dir / "folds" / f"fold_{int(fold):02d}",
            identity=identity,
        )


# ---------------------------------------------------------------------------
# Run


def run_standalone_htr(
    *,
    dataset_path: Path,
    output_dir: Path,
    config: AppliedInferenceConfig,
    settings: StandaloneHTRConfig,
    joint: Optional[JointHTRConfig] = None,
    devices: Sequence[str] = ("cpu",),
    splits_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Fit every outer fold (reusing finished ones), freeze predictions, and summarize.

    ``devices`` lists one worker slot per entry; repeat a device to run several
    folds on it at once. Folds are assigned to slots in turn.
    """

    joint = joint or JointHTRConfig()
    dataset_path = Path(dataset_path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_training_frame(dataset_path, config)
    folds, split_metadata = outer_splits(frame, config, settings, splits_path)
    run_identity = {
        "schema_version": SCHEMA_VERSION,
        "dataset_sha256": _sha256_file(dataset_path),
        "rows": len(frame),
        "config": _config_record(config, settings, joint),
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
                settings=settings,
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
        for slot, device in enumerate(devices):
            assigned = pending[slot :: len(devices)]
            if not assigned:
                continue
            process = context.Process(
                target=_device_worker,
                args=(
                    {
                        "dataset": str(dataset_path),
                        "output_dir": str(output_dir),
                        "slot": slot,
                        "device": device,
                        "config": config,
                        "settings": settings,
                        "joint": joint,
                        "folds": assigned,
                    },
                ),
            )
            process.start()
            logger.info(
                "Started worker %s on %s for folds %s",
                slot,
                device,
                [item[0] for item in assigned],
            )
            workers.append((slot, device, process))
        failed = []
        for slot, device, process in workers:
            process.join()
            if process.exitcode != 0:
                failed.append(f"{slot} ({device})")
        if failed:
            raise RuntimeError(
                f"fold workers {failed} failed; see {output_dir / 'logs'} and rerun to resume"
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
    summary = summarize_predictions(predictions, output_dir, config, settings, joint)
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
    settings: StandaloneHTRConfig,
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
            e_clip=_method_e_clip(config, settings, joint),
            alpha_propensity=float(config.training.alpha_propensity),
            effect_loss_weight=float(joint.effect_loss_weight),
        )
        metrics.pop("selection_loss")
        record = json.loads(
            (output_dir / "folds" / f"fold_{int(fold):02d}" / "fold.json").read_text()
        )
        per_fold[str(fold)] = {
            **{
                key: record[key]
                for key in ("selected_epoch", "effect_objective", "inner_folds")
                if key in record
            },
            **metrics,
        }
    y_residual = predictions["y_residual"].to_numpy(dtype=float)
    t_residual = predictions["t_residual"].to_numpy(dtype=float)
    ate = aipw["estimate"]
    return {
        "schema_version": SCHEMA_VERSION,
        "method": settings.method,
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


def _load_template(template_path: Path) -> Dict[str, Any]:
    return json.loads(Path(template_path).read_text())["config"]


def _is_htr_architecture_key(key: str) -> bool:
    return key.startswith("htr_") or key == "feature_extractor_type"


def build_configs(
    *,
    template_path: Path = DEFAULT_TEMPLATE,
    outcome_type: str,
    htr_model: str = DEFAULT_HTR_MODEL,
    text_column: Optional[str] = None,
    treatment_column: Optional[str] = None,
    outcome_column: Optional[str] = None,
    overrides: Sequence[str] = (),
) -> Tuple[AppliedInferenceConfig, StandaloneHTRConfig, JointHTRConfig]:
    """Settings from the Stage 1 template, then ``--set`` overrides.

    Keys are ``run.<field>``, ``architecture.<htr field>``, ``training.<field>``,
    ``stage1_htr.<field>`` (the template's
    ``architecture.agentic_attention_variable_forest``), or ``joint.<field>``.
    """

    template = _load_template(template_path)
    template_architecture = template.get("architecture", {})
    architecture = {
        key: value for key, value in template_architecture.items() if _is_htr_architecture_key(key)
    }
    architecture["htr_sentence_model"] = htr_model
    sections: Dict[str, Dict[str, Any]] = {
        "run": {},
        "architecture": architecture,
        "training": dict(template.get("training", {})),
        "stage1_htr": dict(template_architecture.get("agentic_attention_variable_forest", {})),
        "joint": {},
    }
    for raw in overrides:
        key, value = _parse_override(raw)
        section, _, name = key.partition(".")
        if section not in sections or not name or "." in name:
            raise ValueError(
                f"--set key {key!r} must be run.<field>, architecture.<field>, "
                "training.<field>, stage1_htr.<field>, or joint.<field>"
            )
        sections[section][name] = value

    def names(cls) -> set:
        return {field.name for field in dataclasses.fields(cls)}

    known = {
        "run": names(StandaloneHTRConfig),
        "architecture": {
            name for name in names(ModelArchitectureConfig) if _is_htr_architecture_key(name)
        },
        "training": names(TrainingConfig),
        "stage1_htr": names(AgenticAttentionVariableForestConfig),
        "joint": names(JointHTRConfig),
    }
    for section, values in sections.items():
        unknown = sorted(set(values) - known[section])
        if unknown:
            raise ValueError(f"unknown {section} settings: {unknown}")
    settings = StandaloneHTRConfig(**sections["run"])
    config = AppliedInferenceConfig(
        seed=int(settings.seed),
        outcome_type=outcome_type,
        text_column=text_column or template.get("text_column", "clinical_text"),
        treatment_column=treatment_column
        or template.get("treatment_column", "treatment_indicator"),
        outcome_column=outcome_column or template.get("outcome_column", "outcome_indicator"),
        architecture=ModelArchitectureConfig(
            **sections["architecture"],
            agentic_attention_variable_forest=AgenticAttentionVariableForestConfig(
                **sections["stage1_htr"]
            ),
        ),
        training=TrainingConfig(**sections["training"]),
    )
    if settings.nuisance_networks == "separate" and not _dina_uses_htr_nuisances(config, settings):
        raise ValueError(
            "run.nuisance_networks=separate needs two_stage DINA with run.dina_nuisances=htr"
        )
    if int(settings.mlm_epochs) > 0 and settings.method != "two_stage":
        raise ValueError("run.mlm_epochs needs run.method=two_stage")
    return config, settings, JointHTRConfig(**sections["joint"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone HTR treatment-effect estimation per outer fold, with cross-fitted "
            "held-out predictions and an optional oracle evaluation. two_stage (default): "
            "Stage 1's cross-fitted HTR nuisance and effect ensembles; joint: one network "
            "with propensity, outcome, and tau heads."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="fit, freeze predictions, and summarize (resumable)")
    run.add_argument("--dataset", required=True, type=Path, help="cohort parquet file")
    run.add_argument("--out", required=True, type=Path, help="output directory")
    run.add_argument("--method", choices=METHODS, help="shorthand for --set run.method=NAME")
    run.add_argument(
        "--dina-nuisances",
        choices=DINA_NUISANCE_SOURCES,
        help="two_stage with DINA: shorthand for --set run.dina_nuisances=NAME (default htr)",
    )
    run.add_argument(
        "--outcome-type",
        choices=("auto", "binary", "continuous"),
        default="auto",
        help="auto: binary when every outcome is 0 or 1",
    )
    run.add_argument(
        "--devices",
        help=(
            "comma-separated torch devices, one worker process each "
            "(default: cuda:0 when CUDA is available, else cpu)"
        ),
    )
    run.add_argument(
        "--jobs-per-device",
        type=int,
        default=1,
        help="worker processes per device; outer folds run in parallel across all workers",
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
    run.add_argument("--seed", type=int, help="shorthand for --set run.seed=N")
    run.add_argument("--folds", type=int, help="shorthand for --set run.outer_folds=K")
    run.add_argument(
        "--inner-folds",
        type=int,
        help="two_stage: shorthand for stage1_htr.nuisance_folds and effect_folds",
    )
    run.add_argument(
        "--epochs",
        type=int,
        help=(
            "two_stage: stage1_htr.nuisance_epochs and effect_epochs; joint: joint.epochs"
        ),
    )
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
        help=(
            "run.<field>, architecture.<field>, training.<field>, stage1_htr.<field>, "
            "or joint.<field> (JSON values)"
        ),
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
    if args.method is not None:
        overrides.append(f"run.method={args.method}")
    if args.dina_nuisances is not None:
        overrides.append(f"run.dina_nuisances={args.dina_nuisances}")
    method = StandaloneHTRConfig.method
    for raw in overrides:
        key, value = _parse_override(raw)
        if key == "run.method":
            method = str(value)
    for flag, key in (("seed", "run.seed"), ("folds", "run.outer_folds")):
        if getattr(args, flag) is not None:
            overrides.append(f"{key}={getattr(args, flag)}")
    if args.inner_folds is not None:
        overrides += [f"stage1_htr.{key}={args.inner_folds}" for key in ("nuisance_folds", "effect_folds")]
    if args.epochs is not None:
        keys = ("joint.epochs",) if method == "joint" else (
            "stage1_htr.nuisance_epochs",
            "stage1_htr.effect_epochs",
        )
        overrides += [f"{key}={args.epochs}" for key in keys]
    outcome_type = args.outcome_type
    if outcome_type == "auto":
        outcome_column = args.outcome_column or _load_template(args.template).get(
            "outcome_column", "outcome_indicator"
        )
        outcome_type = infer_outcome_type(args.dataset, outcome_column)
    config, settings, joint = build_configs(
        template_path=args.template,
        outcome_type=outcome_type,
        htr_model=args.htr_model,
        text_column=args.text_column,
        treatment_column=args.treatment_column,
        outcome_column=args.outcome_column,
        overrides=overrides,
    )
    devices = args.devices
    if devices is None:
        devices = "cuda:0" if _torch().cuda.is_available() else "cpu"
    devices = [device.strip() for device in str(devices).split(",") if device.strip()]
    slots = [device for _ in range(max(1, int(args.jobs_per_device))) for device in devices]
    summary = run_standalone_htr(
        dataset_path=args.dataset,
        output_dir=args.out,
        config=config,
        settings=settings,
        joint=joint,
        devices=slots,
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
    "StandaloneHTRConfig",
    "build_configs",
    "build_arm_nuisance_net",
    "build_joint_net",
    "evaluate_against_oracle",
    "joint_losses",
    "load_training_frame",
    "main",
    "outer_splits",
    "run_standalone_htr",
    "summarize_predictions",
]
