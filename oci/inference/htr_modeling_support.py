"""HTR heads and loss transforms used by the active Stage 1 producer."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..config import AgenticAttentionVariableForestConfig
from ..utils.calibration import clip_probability

EFFECT_OBJECTIVES = {"squared_r_loss", "logistic_r_loss", "pseudo_outcome_mse", "dina"}


def effect_objective_name(config: AgenticAttentionVariableForestConfig) -> str:
    value = str(getattr(config, "effect_objective", "pseudo_outcome_mse")).strip().lower()
    if value not in EFFECT_OBJECTIVES:
        raise ValueError(
            "agentic_attention_variable_forest.effect_objective must be one of "
            "'dina', 'squared_r_loss', 'logistic_r_loss', or 'pseudo_outcome_mse'"
        )
    return value


def r_pseudo_outcome(
    y_residual: Any,
    t_residual: Any,
    *,
    min_abs_t_residual: float = 1e-8,
) -> np.ndarray:
    y_arr = np.asarray(y_residual, dtype=float)
    t_arr = np.asarray(t_residual, dtype=float)
    result = np.full_like(y_arr, np.nan, dtype=float)
    valid = np.isfinite(y_arr) & np.isfinite(t_arr) & (np.abs(t_arr) > min_abs_t_residual)
    np.divide(y_arr, t_arr, out=result, where=valid)
    return result


def binary_log_loss_from_logits(logits: Any, target: Any) -> np.ndarray:
    z = np.asarray(logits, dtype=float)
    y = np.asarray(target, dtype=float)
    return np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))


def _probability_logit(probability: Any) -> np.ndarray:
    clipped = clip_probability(probability)
    return np.log(clipped / (1.0 - clipped))


def logistic_r_logits(
    delta: Any,
    treatment: Any,
    e_hat: Any,
    m_hat: Any,
    *,
    e_clip: float,
) -> np.ndarray:
    delta_arr = np.asarray(delta, dtype=float)
    treatment_arr = np.asarray(treatment, dtype=float)
    propensity = np.clip(np.asarray(e_hat, dtype=float), e_clip, 1.0 - e_clip)
    return _probability_logit(m_hat) + (treatment_arr - propensity) * delta_arr


def logistic_r_tau_from_delta(
    delta: Any,
    e_hat: Any,
    m_hat: Any,
    *,
    e_clip: float,
) -> np.ndarray:
    delta_arr = np.asarray(delta, dtype=float)
    propensity = np.clip(np.asarray(e_hat, dtype=float), e_clip, 1.0 - e_clip)
    baseline = _probability_logit(m_hat)
    p1 = 1.0 / (1.0 + np.exp(-np.clip(baseline + (1.0 - propensity) * delta_arr, -50.0, 50.0)))
    p0 = 1.0 / (1.0 + np.exp(-np.clip(baseline - propensity * delta_arr, -50.0, 50.0)))
    return p1 - p0


def _configured_head_activation(name: str) -> nn.Module:
    implementations = {
        "gelu_exact": lambda: nn.GELU(approximate="none"),
        "gelu_tanh": lambda: nn.GELU(approximate="tanh"),
        "relu": lambda: nn.ReLU(inplace=False),
        "silu": lambda: nn.SiLU(inplace=False),
        "tanh": nn.Tanh,
    }
    key = str(name)
    if key not in implementations:
        raise ValueError(
            "HTR head activation must be one of: " + ", ".join(sorted(implementations))
        )
    return implementations[key]()


def _configured_hidden_head(
    *,
    input_dim: int,
    hidden_dim: int,
    depth: int,
    activation: str,
    dropout: float,
    layer_norm: bool,
    bias: bool,
) -> nn.Sequential:
    if isinstance(depth, bool) or int(depth) < 1:
        raise ValueError("HTR head depth must be a positive integer")
    if int(hidden_dim) < 1:
        raise ValueError("HTR head hidden_dim must be positive")
    if not 0.0 <= float(dropout) < 1.0:
        raise ValueError("HTR head dropout must be in [0, 1)")
    if type(layer_norm) is not bool or type(bias) is not bool:
        raise TypeError("HTR head norm/bias settings must be exact booleans")
    _configured_head_activation(activation)
    layers: list[nn.Module] = []
    source_dim = int(input_dim)
    for _ in range(int(depth)):
        layers.append(nn.Linear(source_dim, int(hidden_dim), bias=bias))
        if layer_norm:
            layers.append(nn.LayerNorm(int(hidden_dim)))
        layers.append(_configured_head_activation(activation))
        layers.append(nn.Dropout(float(dropout), inplace=False))
        source_dim = int(hidden_dim)
    return nn.Sequential(*layers)


class NuisanceNet(nn.Module):
    def __init__(
        self,
        extractor: nn.Module,
        hidden_dim: int,
        outcome_type: str,
        *,
        head_depth: int = 1,
        head_activation: str = "relu",
        head_dropout: float = 0.1,
        head_layer_norm: bool = False,
        head_bias: bool = True,
    ) -> None:
        super().__init__()
        self.extractor = extractor
        self.outcome_type = outcome_type
        self._head_configuration = {
            "hidden_dim": int(hidden_dim),
            "depth": int(head_depth),
            "activation": str(head_activation),
            "dropout": float(head_dropout),
            "layer_norm": head_layer_norm,
            "bias": head_bias,
        }
        self.shared = _configured_hidden_head(
            input_dim=int(extractor.output_dim),
            hidden_dim=hidden_dim,
            depth=head_depth,
            activation=head_activation,
            dropout=head_dropout,
            layer_norm=head_layer_norm,
            bias=head_bias,
        )
        self.propensity = nn.Linear(hidden_dim, 1, bias=head_bias)
        self.outcome = nn.Linear(hidden_dim, 1, bias=head_bias)

    def head_configuration(self) -> Dict[str, Any]:
        return dict(self._head_configuration)

    def forward(self, texts_or_batch):
        features = self.extractor(
            texts_or_batch if isinstance(texts_or_batch, dict) else list(texts_or_batch)
        )
        hidden = self.shared(features)
        return self.propensity(hidden).squeeze(-1), self.outcome(hidden).squeeze(-1)


class EffectNet(nn.Module):
    def __init__(
        self,
        extractor: nn.Module,
        hidden_dim: int,
        *,
        head_depth: int = 1,
        head_activation: str = "relu",
        head_dropout: float = 0.1,
        head_layer_norm: bool = False,
        head_bias: bool = True,
    ) -> None:
        super().__init__()
        self.extractor = extractor
        self._head_configuration = {
            "hidden_dim": int(hidden_dim),
            "depth": int(head_depth),
            "activation": str(head_activation),
            "dropout": float(head_dropout),
            "layer_norm": head_layer_norm,
            "bias": head_bias,
        }
        self.hidden = _configured_hidden_head(
            input_dim=int(extractor.output_dim),
            hidden_dim=hidden_dim,
            depth=head_depth,
            activation=head_activation,
            dropout=head_dropout,
            layer_norm=head_layer_norm,
            bias=head_bias,
        )
        self.output = nn.Linear(hidden_dim, 1, bias=head_bias)

    def head_configuration(self) -> Dict[str, Any]:
        return dict(self._head_configuration)

    def forward(self, texts_or_batch):
        features = self.extractor(
            texts_or_batch if isinstance(texts_or_batch, dict) else list(texts_or_batch)
        )
        return self.output(self.hidden(features)).squeeze(-1)


def run_crossfit_fold_tasks(
    run_fold,
    split_items,
    n_jobs: int,
    device_context=None,
) -> List[Dict[str, Any]]:
    def call_fold(fold, fit_pos, heldout_pos):
        if device_context is None:
            return run_fold(fold, fit_pos, heldout_pos)
        with device_context(fold):
            return run_fold(fold, fit_pos, heldout_pos)

    if n_jobs <= 1:
        return [
            call_fold(fold, fit_pos, heldout_pos) for fold, (fit_pos, heldout_pos) in split_items
        ]
    with ThreadPoolExecutor(
        max_workers=int(n_jobs),
        thread_name_prefix="avf-fold",
    ) as executor:
        futures = [
            executor.submit(call_fold, fold, fit_pos, heldout_pos)
            for fold, (fit_pos, heldout_pos) in split_items
        ]
        return [future.result() for future in futures]


_EffectNet = EffectNet
_NuisanceNet = NuisanceNet
_binary_log_loss_from_logits = binary_log_loss_from_logits
_effect_objective_name = effect_objective_name
_logistic_r_logits = logistic_r_logits
_logistic_r_tau_from_delta = logistic_r_tau_from_delta
_r_pseudo_outcome = r_pseudo_outcome
_run_crossfit_fold_tasks = run_crossfit_fold_tasks


__all__ = [
    "EffectNet",
    "NuisanceNet",
    "binary_log_loss_from_logits",
    "clip_probability",
    "effect_objective_name",
    "logistic_r_logits",
    "logistic_r_tau_from_delta",
    "r_pseudo_outcome",
    "run_crossfit_fold_tasks",
]
