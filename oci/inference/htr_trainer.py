"""HTR training core for the Stage 1 neural lane.

The networks, loaders, and training loops live here so that Stage 1 never
imports the agentic runners or constructs their LLM agent and extraction
clients. Only the HTR training settings are read from the applied
configuration. The standalone agentic attention-variable forest runner
subclasses :class:`HTRTrainer` to reuse this core.
"""

from __future__ import annotations

import copy
import gc
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..config import (
    HTR_EFFECT_OBJECTIVES,
    AgenticAttentionVariableForestConfig,
    AppliedInferenceConfig,
    resolve_effect_objective,
)
from ..utils.calibration import BinaryProbabilityCalibrator
from ..models.hierarchical_transformer_extractor import (
    HTR_SENTENCE_ENCODER_TRAINING_AUDIT_SCHEMA,
    HierarchicalTransformerExtractor,
)

logger = logging.getLogger(__name__)

EFFECT_OBJECTIVES = set(HTR_EFFECT_OBJECTIVES)


def effect_objective_name(config: AgenticAttentionVariableForestConfig) -> str:
    """Return the resolved effect objective of an HTR trainer's config.

    ``HTRTrainer.__init__`` resolves ``"auto"`` against the outcome type, so an
    unresolved value here is a programming error rather than a default.
    """

    value = str(getattr(config, "effect_objective", "auto")).strip().lower()
    if value == "auto":
        raise ValueError(
            "effect_objective='auto' must be resolved against the outcome type "
            "with resolve_effect_objective() before training"
        )
    if value not in EFFECT_OBJECTIVES:
        raise ValueError(
            "agentic_attention_variable_forest.effect_objective must be one of "
            "'dina', 'squared_r_loss', or 'pseudo_outcome_mse'"
        )
    return value


def effect_loss_label(effect_objective: str) -> str:
    if effect_objective == "dina":
        return "bernoulli_dina_loss"
    if effect_objective == "pseudo_outcome_mse":
        return "pseudo_outcome_mse"
    return "r_loss"


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


def torch_pseudo_outcome_mse_loss_vector(
    effect: torch.Tensor,
    y_residual: torch.Tensor,
    t_residual: torch.Tensor,
    *,
    min_abs_t_residual: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    valid = (
        torch.isfinite(y_residual)
        & torch.isfinite(t_residual)
        & (torch.abs(t_residual) > min_abs_t_residual)
    )
    pseudo_target = torch.zeros_like(y_residual)
    pseudo_target[valid] = y_residual[valid] / t_residual[valid]
    return torch.square(effect - pseudo_target), valid


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

    def forward(self, texts_or_batch):
        features = self.extractor(
            texts_or_batch if isinstance(texts_or_batch, dict) else list(texts_or_batch)
        )
        return self.output(self.hidden(features)).squeeze(-1)


class _FoldTextDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        positions: Sequence[int],
        fields: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.texts = [str(text or "") for text in texts]
        self.positions = np.asarray(positions, dtype=int)
        self.fields = {name: np.asarray(values) for name, values in (fields or {}).items()}

    def __len__(self) -> int:
        return int(len(self.positions))

    def __getitem__(self, index: int) -> Dict[str, Any]:
        position = int(self.positions[index])
        item: Dict[str, Any] = {
            "position": position,
            "text": self.texts[position],
        }
        for name, values in self.fields.items():
            item[name] = float(values[position])
        return item


class _FoldTextBatchCollator:
    def __init__(self, text_preprocessor: Optional[Any] = None):
        self.text_preprocessor = text_preprocessor

    def __call__(self, items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [str(item["text"]) for item in items]
        batch: Dict[str, Any] = {
            "model_input": (
                self.text_preprocessor(texts) if self.text_preprocessor is not None else texts
            ),
            "position": torch.as_tensor(
                [int(item["position"]) for item in items],
                dtype=torch.long,
            ),
        }
        field_names = (
            [key for key in items[0].keys() if key not in {"position", "text"}] if items else []
        )
        for name in field_names:
            batch[name] = torch.as_tensor(
                [float(item[name]) for item in items],
                dtype=torch.float32,
            )
        return batch


class HTRTrainer:
    """Create, train, and score HTR nuisance and effect networks on one device.

    Stage 1 fits folds serially inside one exact discovery context, so the
    trainer owns a single device and uses in-process data loading
    (``_data_loader_workers`` returns 0).
    """

    def __init__(
        self,
        *,
        config: AppliedInferenceConfig,
        device: torch.device | str,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.avf_config: AgenticAttentionVariableForestConfig = copy.deepcopy(
            getattr(
                config.architecture,
                "agentic_attention_variable_forest",
                AgenticAttentionVariableForestConfig(),
            )
        )
        # "auto" becomes "dina" for binary and "pseudo_outcome_mse" for
        # continuous outcomes; an explicit mismatch raises (never overridden).
        self.avf_config.effect_objective = resolve_effect_objective(
            self.avf_config.effect_objective,
            outcome_type=config.outcome_type,
        )

    def _effect_epochs(self) -> int:
        value = getattr(self.avf_config, "effect_epochs", None)
        if value is None:
            value = getattr(self.config.training, "epochs", 1)
        return max(1, int(value))

    def _create_extractor(self) -> nn.Module:
        arch = self.config.architecture
        extractor_type = str(
            getattr(arch, "feature_extractor_type", "hierarchical_transformer")
        ).strip().lower()
        if extractor_type not in {
            "hierarchical_transformer",
            "hier_transformer",
            "htr",
            "short_chunk_transformer",
        }:
            raise ValueError(
                "Stage 1 HTR evidence requires feature_extractor_type="
                "'hierarchical_transformer'"
            )
        return HierarchicalTransformerExtractor(
            device=self.device,
            sentence_encoder_model=getattr(arch, "htr_sentence_model", "prajjwal1/bert-tiny"),
            freeze_sentence_encoder=getattr(arch, "htr_freeze_sentence_encoder", False),
            chunk_size_words=getattr(arch, "htr_chunk_size_words", 96),
            chunk_overlap_words=getattr(arch, "htr_chunk_overlap_words", 24),
            max_chunks=getattr(arch, "htr_max_chunks", 512),
            max_chunk_length=getattr(arch, "htr_max_chunk_length", 128),
            num_transformer_layers=getattr(arch, "htr_num_layers", 2),
            num_attention_heads=getattr(arch, "htr_num_heads", 4),
            transformer_dim=getattr(arch, "htr_transformer_dim", 256),
            transformer_dropout=getattr(arch, "htr_dropout", 0.05),
            projection_dim=getattr(arch, "htr_projection_dim", 128),
            hash_embedding_dim=getattr(arch, "htr_hash_embedding_dim", 256),
            sentence_encoder_batch_size=getattr(
                arch,
                "htr_sentence_encoder_batch_size",
                128,
            ),
            sentence_encoder_backend=getattr(
                arch,
                "htr_sentence_encoder_backend",
                "auto",
            ),
            sentence_pooling=getattr(arch, "htr_sentence_pooling", "auto"),
            normalize_sentence_embeddings=getattr(
                arch,
                "htr_normalize_sentence_embeddings",
                True,
            ),
            trainable_sentence_encoder_layers=getattr(
                arch,
                "htr_trainable_sentence_encoder_layers",
                0,
            ),
        )

    def _assert_htr_sentence_encoder_training_state(
        self,
        extractor: nn.Module,
    ) -> Dict[str, Any]:
        """Fail before optimization if the live HTR encoder state is inexact."""

        configured_type = (
            str(
                getattr(
                    self.config.architecture,
                    "feature_extractor_type",
                    "hierarchical_transformer",
                )
            )
            .strip()
            .lower()
        )
        htr_requested = configured_type in {
            "hierarchical_transformer",
            "htr",
        }
        live_encoder_required = bool(
            getattr(
                self.config.architecture,
                "htr_require_live_unfrozen_encoder",
                False,
            )
        )
        if not htr_requested:
            return {
                "schema_version": HTR_SENTENCE_ENCODER_TRAINING_AUDIT_SCHEMA,
                "htr_requested": False,
            }
        if type(extractor) is not HierarchicalTransformerExtractor:
            if not live_encoder_required:
                return {
                    "schema_version": HTR_SENTENCE_ENCODER_TRAINING_AUDIT_SCHEMA,
                    "htr_requested": True,
                    "test_double_without_live_encoder": True,
                }
            raise TypeError("HTR training requires the exact HierarchicalTransformerExtractor")
        audit = extractor.sentence_encoder_training_audit()
        if audit.get("schema_version") != HTR_SENTENCE_ENCODER_TRAINING_AUDIT_SCHEMA:
            raise RuntimeError("HTR sentence-encoder training audit schema changed")
        requested_freeze = getattr(
            self.config.architecture,
            "htr_freeze_sentence_encoder",
            None,
        )
        if audit.get("requested_freeze_sentence_encoder") is not requested_freeze:
            raise RuntimeError("live HTR extractor freeze state differs from its effective config")
        if requested_freeze is False:
            if audit.get("hash_backend_without_sentence_encoder") is True:
                # Deterministic hash extractors are retained for lightweight
                # tests; the production backend authenticates a concrete
                # private transformer model tree instead.
                if live_encoder_required:
                    raise RuntimeError(
                        "HTR requires a live trainable transformer "
                        "sentence encoder; hash fallback is forbidden"
                    )
                return audit
            if (
                audit.get("effective_backend") != "transformers"
                or audit.get("encoder_initialized") is not True
                or audit.get("sentence_encoder_present") is not True
                or int(audit.get("sentence_encoder_parameter_tensors", 0)) <= 0
                or int(audit.get("sentence_encoder_parameters", 0)) <= 0
                or audit.get("all_sentence_encoder_parameters_trainable") is not True
            ):
                raise RuntimeError(
                    "unfrozen HTR policy is not reflected in the initialized "
                    "sentence-encoder parameters"
                )
        return audit

    def _assert_htr_sentence_encoder_optimizer_coverage(
        self,
        extractor: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Prove the live unfrozen encoder tensors are optimizer members."""

        audit = self._assert_htr_sentence_encoder_training_state(extractor)
        if (
            audit.get("htr_requested") is False
            or audit.get("hash_backend_without_sentence_encoder") is True
            or audit.get("test_double_without_live_encoder") is True
        ):
            return
        if audit.get("requested_freeze_sentence_encoder") is not False:
            return
        assert type(extractor) is HierarchicalTransformerExtractor
        encoder = extractor._sentence_encoder
        if encoder is None:
            raise RuntimeError("initialized unfrozen HTR sentence encoder is missing")
        expected = {id(parameter) for parameter in encoder.parameters()}
        observed = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group.get("params", ())
        }
        if not expected or not expected.issubset(observed):
            raise RuntimeError(
                "HTR nuisance optimizer omits one or more unfrozen sentence-encoder " "parameters"
            )

    def _data_loader_workers(self, total_folds: Optional[int] = None) -> int:
        """Return the DataLoader worker count; Stage 1 always loads in-process.

        The standalone attention-variable forest runner overrides this hook
        to use multiprocessing loader workers.
        """

        del total_folds
        return 0

    def _make_text_loader(
        self,
        model: nn.Module,
        df: pd.DataFrame,
        positions: Sequence[int],
        *,
        fields: Optional[Dict[str, np.ndarray]] = None,
        shuffle: bool = False,
        total_folds: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> DataLoader:
        extractor = getattr(model, "extractor", None)
        text_preprocessor = None
        if extractor is not None and hasattr(extractor, "make_batch_preprocessor"):
            text_preprocessor = extractor.make_batch_preprocessor()
        workers = self._data_loader_workers(total_folds=total_folds)
        effective_batch_size = self.config.training.batch_size if batch_size is None else batch_size
        loader_kwargs: Dict[str, Any] = {
            "batch_size": max(1, int(effective_batch_size)),
            "shuffle": bool(shuffle),
            "collate_fn": _FoldTextBatchCollator(text_preprocessor),
            "num_workers": workers,
            "pin_memory": self.device.type == "cuda",
        }
        if workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2
        return DataLoader(
            _FoldTextDataset(
                texts=df[self.config.text_column].astype(str).tolist(),
                positions=positions,
                fields=fields,
            ),
            **loader_kwargs,
        )

    def _train_nuisance_model(
        self,
        model: NuisanceNet,
        df: pd.DataFrame,
        positions,
        outer_fold: int,
        fold: int,
        total_folds: int,
    ):
        train_config = self.config.training
        nuisance_epochs = int(
            self.avf_config.nuisance_epochs
            if self.avf_config.nuisance_epochs is not None
            else train_config.epochs
        )
        nuisance_weight_decay = float(
            self.avf_config.nuisance_weight_decay
            if self.avf_config.nuisance_weight_decay is not None
            else getattr(train_config, "weight_decay", 0.01)
        )
        nuisance_label_smoothing = float(self.avf_config.nuisance_label_smoothing)
        model.extractor.fit_tokenizer(
            df.iloc[positions][self.config.text_column].astype(str).tolist()
        )
        self._assert_htr_sentence_encoder_training_state(model.extractor)
        train_loader = self._make_text_loader(
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
            weight_decay=nuisance_weight_decay,
        )
        self._assert_htr_sentence_encoder_optimizer_coverage(
            model.extractor,
            optimizer,
        )
        num_batches = max(1, len(train_loader))
        scheduler = make_linear_lr_scheduler(
            optimizer,
            train_config,
            num_batches,
            epochs_override=nuisance_epochs,
        )
        progress_every = max(1, num_batches // 5)
        logger.info(
            "Outer fold %s nuisance fold %s/%s: training for %s epoch(s), "
            "batch_size=%s, batches/epoch=%s, dataloader_workers=%s, "
            "lr=%.3g, weight_decay=%.3g, label_smoothing=%.3g, "
            "lr_schedule=%s%s",
            outer_fold,
            fold,
            total_folds,
            nuisance_epochs,
            train_config.batch_size,
            num_batches,
            train_loader.num_workers,
            current_lr(optimizer),
            nuisance_weight_decay,
            nuisance_label_smoothing,
            "linear" if scheduler is not None else "none",
            self._cuda_memory_summary(),
        )
        for epoch in range(1, nuisance_epochs + 1):
            model.train()
            loss_sum = 0.0
            prop_sum = 0.0
            outcome_sum = 0.0
            batch_count = 0
            for batch_idx, batch in enumerate(train_loader, start=1):
                t = batch["t"].to(self.device, non_blocking=True)
                y = batch["y"].to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                t_logit, y_pred = model(batch["model_input"])
                if nuisance_label_smoothing > 0:
                    t_target = t * (1.0 - nuisance_label_smoothing) + 0.5 * nuisance_label_smoothing
                else:
                    t_target = t
                prop_loss = F.binary_cross_entropy_with_logits(t_logit, t_target)
                if self.config.outcome_type == "continuous":
                    outcome_loss = F.mse_loss(y_pred, y)
                else:
                    if nuisance_label_smoothing > 0:
                        y_target = (
                            y * (1.0 - nuisance_label_smoothing) + 0.5 * nuisance_label_smoothing
                        )
                    else:
                        y_target = y
                    outcome_loss = F.binary_cross_entropy_with_logits(y_pred, y_target)
                loss = outcome_loss + self.config.training.alpha_propensity * prop_loss
                loss.backward()
                self._clip_and_step(model, optimizer, scheduler)
                batch_count += 1
                loss_value = float(loss.detach().cpu())
                prop_value = float(prop_loss.detach().cpu())
                outcome_value = float(outcome_loss.detach().cpu())
                loss_sum += loss_value
                prop_sum += prop_value
                outcome_sum += outcome_value
                if batch_idx == 1 or batch_idx == num_batches or batch_idx % progress_every == 0:
                    logger.info(
                        "Outer fold %s nuisance fold %s/%s epoch %s/%s "
                        "batch %s/%s loss=%.4f outcome=%.4f propensity=%.4f lr=%.3g%s",
                        outer_fold,
                        fold,
                        total_folds,
                        epoch,
                        nuisance_epochs,
                        batch_idx,
                        num_batches,
                        loss_value,
                        outcome_value,
                        prop_value,
                        current_lr(optimizer),
                        self._cuda_memory_summary(),
                    )
            denom = max(1, batch_count)
            logger.info(
                "Outer fold %s nuisance fold %s/%s epoch %s/%s complete: "
                "loss=%.4f outcome=%.4f propensity=%.4f lr=%.3g%s",
                outer_fold,
                fold,
                total_folds,
                epoch,
                nuisance_epochs,
                loss_sum / denom,
                outcome_sum / denom,
                prop_sum / denom,
                current_lr(optimizer),
                self._cuda_memory_summary(),
            )

    def _train_effect_model(
        self,
        model: EffectNet,
        df: pd.DataFrame,
        positions,
        outcomes: np.ndarray,
        treatments: np.ndarray,
        e_hat: np.ndarray,
        m_hat: np.ndarray,
        y_residuals: np.ndarray,
        t_residuals: np.ndarray,
        outer_fold: int,
        fold: int,
        total_folds: int,
    ):
        train_config = self.config.training
        effect_epochs = self._effect_epochs()
        model.extractor.fit_tokenizer(
            df.iloc[positions][self.config.text_column].astype(str).tolist()
        )
        dina_fields = {}
        if str(self.config.outcome_type).lower() == "binary":
            from .stage1_dina import prepare_neural_nuisance
            from ..models import dina

            dina_fields = prepare_neural_nuisance(
                self, model, df, positions, seed=70_000 + 100 * int(outer_fold) + int(fold)
            )
        train_loader = self._make_text_loader(
            model,
            df,
            positions,
            fields={
                **dina_fields,
                "outcome": np.asarray(outcomes, dtype=np.float32),
                "treatment": np.asarray(treatments, dtype=np.float32),
                "e_hat": np.asarray(e_hat, dtype=np.float32),
                "m_hat": np.asarray(m_hat, dtype=np.float32),
                "y_residual": np.asarray(y_residuals, dtype=np.float32),
                "t_residual": np.asarray(t_residuals, dtype=np.float32),
            },
            shuffle=True,
            total_folds=total_folds,
            batch_size=getattr(train_config, "effect_batch_size", None),
        )
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=train_config.learning_rate,
            weight_decay=getattr(train_config, "weight_decay", 0.01),
        )
        num_batches = max(1, len(train_loader))
        scheduler = make_linear_lr_scheduler(
            optimizer,
            train_config,
            num_batches,
            epochs_override=effect_epochs,
        )
        progress_every = max(1, num_batches // 5)
        effect_objective = effect_objective_name(self.avf_config)
        logger.info(
            "Outer fold %s effect fold %s/%s: training for %s epoch(s), "
            "objective=%s, batch_size=%s, batches/epoch=%s, dataloader_workers=%s, "
            "lr=%.3g, lr_schedule=%s%s",
            outer_fold,
            fold,
            total_folds,
            effect_epochs,
            effect_objective,
            train_loader.batch_size,
            num_batches,
            train_loader.num_workers,
            current_lr(optimizer),
            "linear" if scheduler is not None else "none",
            self._cuda_memory_summary(),
        )
        for epoch in range(1, effect_epochs + 1):
            model.train()
            loss_sum = 0.0
            batch_count = 0
            for batch_idx, batch in enumerate(train_loader, start=1):
                y_residual = batch["y_residual"].to(self.device, non_blocking=True)
                t_residual = batch["t_residual"].to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                effect = model(batch["model_input"])
                if dina_fields:
                    y = batch["outcome"].to(self.device, non_blocking=True)
                    t = batch["treatment"].to(self.device, non_blocking=True)
                    da = batch["dina_a"].to(self.device, non_blocking=True)
                    nu = batch["dina_nu"].to(self.device, non_blocking=True)
                    loss = dina.torch_loss(y, t, da, nu, effect)
                elif effect_objective == "pseudo_outcome_mse":
                    loss_vector, valid = torch_pseudo_outcome_mse_loss_vector(
                        effect,
                        y_residual,
                        t_residual,
                    )
                    loss = loss_vector[valid].mean() if torch.any(valid) else loss_vector.mean()
                else:
                    residual = y_residual - effect * t_residual
                    # Direct R-loss avoids high-variance y_residual / t_residual pseudo-labels.
                    loss = torch.mean(torch.square(residual))
                loss.backward()
                self._clip_and_step(model, optimizer, scheduler)
                batch_count += 1
                loss_value = float(loss.detach().cpu())
                loss_sum += loss_value
                if batch_idx == 1 or batch_idx == num_batches or batch_idx % progress_every == 0:
                    logger.info(
                        "Outer fold %s effect fold %s/%s epoch %s/%s "
                        "batch %s/%s %s=%.4f lr=%.3g%s",
                        outer_fold,
                        fold,
                        total_folds,
                        epoch,
                        effect_epochs,
                        batch_idx,
                        num_batches,
                        effect_loss_label(effect_objective),
                        loss_value,
                        current_lr(optimizer),
                        self._cuda_memory_summary(),
                    )
            logger.info(
                "Outer fold %s effect fold %s/%s epoch %s/%s complete: " "%s=%.4f lr=%.3g%s",
                outer_fold,
                fold,
                total_folds,
                epoch,
                effect_epochs,
                effect_loss_label(effect_objective),
                loss_sum / max(1, batch_count),
                current_lr(optimizer),
                self._cuda_memory_summary(),
            )

    def _clip_and_step(self, model: nn.Module, optimizer, scheduler=None) -> None:
        clip_norm = getattr(self.config.training, "gradient_clip_norm", 0.0)
        if clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    def _predict_nuisance_model(
        self, model: NuisanceNet, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        model.eval()
        prop = []
        outcome = []
        loader = self._make_text_loader(
            model,
            df,
            np.arange(len(df), dtype=int),
            shuffle=False,
        )
        with torch.no_grad():
            for batch in loader:
                t_logit, y_pred = model(batch["model_input"])
                prop.append(torch.sigmoid(t_logit).cpu().numpy())
                if self.config.outcome_type == "continuous":
                    outcome.append(y_pred.cpu().numpy())
                else:
                    outcome.append(torch.sigmoid(y_pred).cpu().numpy())
        return np.concatenate(prop), np.concatenate(outcome)

    def _predict_effect_model(self, model: EffectNet, df: pd.DataFrame) -> np.ndarray:
        model.eval()
        tau = []
        loader = self._make_text_loader(
            model,
            df,
            np.arange(len(df), dtype=int),
            shuffle=False,
        )
        with torch.no_grad():
            for batch in loader:
                tau.append(model(batch["model_input"]).cpu().numpy())
        return np.concatenate(tau)

    def _attention_evidence(
        self,
        extractor: nn.Module,
        df: pd.DataFrame,
        fold: int,
        outer_fold: int,
        stage: str,
        extra: Dict[str, np.ndarray],
    ) -> List[Dict[str, Any]]:
        texts = df[self.config.text_column].tolist()
        row_ids = df["_oci_row_id"].tolist()
        metadata = []
        for offset in range(len(df)):
            item = {"outer_fold": outer_fold}
            for key, values in extra.items():
                item[key] = _metadata_value(values, offset)
            metadata.append(item)
        batch_size = max(1, int(self.config.training.batch_size))
        total_batches = max(1, int(np.ceil(len(texts) / batch_size)))
        progress_every = max(1, total_batches // 5)
        records: List[Dict[str, Any]] = []
        for batch_idx, start in enumerate(range(0, len(texts), batch_size), start=1):
            end = min(start + batch_size, len(texts))
            records.extend(
                extractor.get_attention_evidence(
                    texts[start:end],
                    row_ids=row_ids[start:end],
                    fold=fold,
                    stage=stage,
                    top_k=self.avf_config.attention_top_k_chunks,
                    metadata=metadata[start:end],
                )
            )
            if batch_idx == 1 or batch_idx == total_batches or batch_idx % progress_every == 0:
                logger.info(
                    "Outer fold %s %s fold %s: attention batch %s/%s rows=%s/%s%s",
                    outer_fold,
                    stage,
                    fold,
                    batch_idx,
                    total_batches,
                    end,
                    len(texts),
                    self._cuda_memory_summary(),
                )
        return records

    def _cleanup_model(self, model: nn.Module) -> None:
        model.cpu()
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _cuda_memory_summary(self) -> str:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return ""
        try:
            device_index = self.device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(device_index) / 1e9
            reserved = torch.cuda.memory_reserved(device_index) / 1e9
            peak = torch.cuda.max_memory_allocated(device_index) / 1e9
            return (
                f" cuda_alloc={allocated:.2f}GB"
                f" cuda_reserved={reserved:.2f}GB"
                f" cuda_peak={peak:.2f}GB"
            )
        except Exception:
            return ""


def cross_fitted_calibrators(
    raw_predictions: Any,
    target: Any,
    fold_ids: Any,
    *,
    method: Optional[str],
) -> Dict[int, BinaryProbabilityCalibrator]:
    """Fit one nuisance probability calibrator per cross-fitting fold, out of fold.

    ``raw_predictions[i]`` must be the uncalibrated prediction for row ``i``
    from the model of fold ``fold_ids[i]``, i.e. a model that never trained on
    row ``i``. The calibrator returned for fold ``k`` is fit on the rows with
    ``fold_ids != k`` (fold ``k``'s fit partition) using those out-of-fold
    predictions, and is then applied to fold ``k``'s held-out and external
    predictions. Fold ``k``'s model's in-sample predictions, which are
    overconfident after training, and fold ``k``'s held-out labels never
    enter fold ``k``'s calibrator.
    """

    raw = np.asarray(raw_predictions, dtype=float).reshape(-1)
    labels = np.asarray(target, dtype=float).reshape(-1)
    folds = np.asarray(fold_ids).reshape(-1)
    if not (len(raw) == len(labels) == len(folds)):
        raise ValueError("raw predictions, targets, and fold ids must align row by row")
    unique_folds = np.unique(folds)
    if len(unique_folds) < 2:
        raise ValueError("out-of-fold calibration requires at least two folds")
    return {
        int(fold): BinaryProbabilityCalibrator.fit(
            raw[folds != fold],
            labels[folds != fold],
            method=method,
        )
        for fold in unique_folds
    }


def _metadata_value(values: Any, offset: int) -> Any:
    array = np.asarray(values, dtype=object)
    value = array.item() if array.ndim == 0 else array[offset]
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    return str(value)


def make_linear_lr_scheduler(
    optimizer,
    train_config,
    steps_per_epoch: int,
    *,
    epochs_override: Optional[int] = None,
):
    lr_schedule = str(getattr(train_config, "lr_schedule", "linear") or "").lower()
    if lr_schedule != "linear":
        return None
    epochs = int(
        epochs_override if epochs_override is not None else getattr(train_config, "epochs", 1)
    )
    total_steps = max(1, int(steps_per_epoch) * epochs)
    return torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.1,
        total_iters=total_steps,
    )


def current_lr(optimizer) -> float:
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))


__all__ = [
    "EFFECT_OBJECTIVES",
    "EffectNet",
    "HTRTrainer",
    "NuisanceNet",
    "cross_fitted_calibrators",
    "current_lr",
    "effect_loss_label",
    "effect_objective_name",
    "make_linear_lr_scheduler",
    "r_pseudo_outcome",
    "torch_pseudo_outcome_mse_loss_vector",
]
