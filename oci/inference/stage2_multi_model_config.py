"""Lightweight scientific policy for the Stage 2 multi-model evidence ensemble."""

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping

SCHEMA_VERSION = "stage2_multi_model_selection_v8_opt_in_diagnostic_families"
FAMILIES = (
    "univariable",
    "penalized_main",
    "penalized_interactions",
    "orthogonal_linear",
    "univariable_rlearner",
    "predictive_forest",
    "causal_forest",
    "dina",
    "matched_batch_contrast",
)
# Treatment/outcome prediction families that write only treatment- and
# outcome-role records. Modifier ranking reads effect-role records only,
# nuisance selection reads the fold elastic-net supports, and concept review
# prompts list effect-role rows only, so these run only when requested.
DIAGNOSTIC_FAMILIES = ("penalized_main", "predictive_forest")
EVIDENCE_FAMILY_POLICY = {
    "default_active": [
        "univariable",
        "penalized_interactions",
        "orthogonal_linear",
        "univariable_rlearner",
        "causal_forest",
        "dina",
    ],
    "opt_in": {
        "penalized_main": "listed in diagnostic_families",
        "predictive_forest": "listed in diagnostic_families",
        "matched_batch_contrast": "matched_batch.enabled is true",
    },
    "rule": (
        "dina runs for binary outcomes only. penalized_main and predictive_forest "
        "produce treatment/outcome prediction diagnostics that no modifier ranking, "
        "nuisance selection, modifier-count, or concept-review step reads; they run "
        "only when listed in diagnostic_families."
    ),
}
FINAL_ESTIMATORS = ("causal_forest", "linear_interactions", "dina")


@dataclass(frozen=True)
class MatchedBatchConfig:
    """Additional evidence only; no ontology search or modifier budget."""

    enabled: bool = False
    batch_size: int = 8
    bins_per_nuisance: int = 2
    min_reference_arm: int = 8
    max_smd: float = 0.25
    min_delta: float = 0.10
    z_threshold: float = 1.0
    filter_training_batches: bool = True
    train_passes: int = 100
    validation_passes: int = 50
    ridge_penalty: float = 0.01

    def validate(self):
        for name in ("enabled", "filter_training_batches"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"matched_batch.{name} must be boolean")
        for name in ("batch_size", "bins_per_nuisance", "min_reference_arm",
                     "train_passes", "validation_passes"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"matched_batch.{name} must be a positive integer")
        if self.batch_size < 4 or self.batch_size % 2:
            raise ValueError("matched_batch.batch_size must be even and at least four")
        if self.min_reference_arm < self.batch_size // 2:
            raise ValueError("matched_batch.min_reference_arm must support a complete batch")
        for name in ("max_smd", "min_delta", "z_threshold", "ridge_penalty"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"matched_batch.{name} must be finite and nonnegative")
        if self.ridge_penalty == 0:
            raise ValueError("matched_batch.ridge_penalty must be positive")


@dataclass(frozen=True)
class ModifierCountConfig:
    """Nested predictive/interaction selection of nonlocked modifier candidates."""

    candidate_counts: tuple[int, ...] = (0, 4, 8, 12, 16, 24, 32)
    max_ranked_modifiers: int = 64
    selection_rule: str = "balanced_evidence"
    relative_r_loss_tolerance: float = 1e-4
    forest_seeds: int = 3
    estimators: tuple[str, ...] = FINAL_ESTIMATORS
    concept_review: bool = False
    concept_top_n_per_fold: int = 100

    def validate(self) -> None:
        if not isinstance(self.concept_review, bool):
            raise ValueError("multi_model.modifier_count.concept_review must be boolean")
        if (
            not isinstance(self.estimators, tuple)
            or not self.estimators
            or any(not isinstance(v, str) or v not in FINAL_ESTIMATORS for v in self.estimators)
            or len(set(self.estimators)) != len(self.estimators)
        ):
            raise ValueError(
                "modifier_count.estimators must be a nonempty unique list of causal_forest, linear_interactions, and/or dina"
            )
        for name in ("max_ranked_modifiers", "forest_seeds", "concept_top_n_per_fold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"modifier_count.{name} must be a positive integer")
        if (
            not isinstance(self.candidate_counts, tuple)
            or not self.candidate_counts
            or any(
                isinstance(k, bool) or not isinstance(k, int) or k < 0
                for k in self.candidate_counts
            )
            or tuple(sorted(set(self.candidate_counts))) != self.candidate_counts
            or self.candidate_counts[0] != 0
            or self.candidate_counts[-1] > self.max_ranked_modifiers
        ):
            raise ValueError(
                "modifier_count.candidate_counts must be sorted unique integers starting at 0 and at most max_ranked_modifiers"
            )
        if self.selection_rule != "balanced_evidence":
            raise ValueError("modifier_count.selection_rule must be balanced_evidence")
        if (isinstance(self.relative_r_loss_tolerance, bool)
                or not isinstance(self.relative_r_loss_tolerance, (int, float))
                or not math.isfinite(self.relative_r_loss_tolerance)
                or not 0 <= self.relative_r_loss_tolerance <= 1):
            raise ValueError("modifier_count.relative_r_loss_tolerance must be in [0, 1]")

    def public_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "candidate_counts": list(self.candidate_counts),
            "estimators": list(self.estimators),
        }


@dataclass(frozen=True)
class EvidenceStabilityConfig:
    """Training-only resampling and conservative alternate-measurement groups."""

    partition_repeats: int = 3
    subset_partitions: int = 4
    group_equivalent_measurements: bool = True
    group_min_observations: int = 30
    group_min_overlap: float = 0.8
    group_min_association: float = 0.95
    group_min_name_similarity: float = 0.8

    def validate(self):
        for name in ('partition_repeats', 'subset_partitions', 'group_min_observations'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'multi_model.stability.{name} must be a positive integer')
        if type(self.group_equivalent_measurements) is not bool:
            raise ValueError('group_equivalent_measurements must be boolean')
        for name in ('group_min_overlap', 'group_min_association', 'group_min_name_similarity'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f'multi_model.stability.{name} must be in (0, 1]')


@dataclass(frozen=True)
class Stage2MultiModelConfig:
    repeats: int = 3
    row_fraction: float = 0.8
    l1_ratios: tuple[float, ...] = (0.2, 0.8, 1.0)
    regularization_grid_size: int = 8
    forest_trees: int = 800
    forest_min_samples_leaf: int = 10
    feature_subset_size: int = 32
    permutation_repeats: int = 10
    stability: EvidenceStabilityConfig = field(default_factory=EvidenceStabilityConfig)
    nominal_p_threshold: float = 0.05
    q_threshold: float = 0.1
    interaction_ranking_weight: float = 0.5
    max_prompt_chars: int = 100_000
    modifier_count: ModifierCountConfig = field(default_factory=ModifierCountConfig)
    matched_batch: MatchedBatchConfig = MatchedBatchConfig()
    diagnostic_families: tuple[str, ...] = ()

    def validate(self) -> None:
        if (
            not isinstance(self.diagnostic_families, tuple)
            or any(not isinstance(v, str) or v not in DIAGNOSTIC_FAMILIES
                   for v in self.diagnostic_families)
            or self.diagnostic_families
            != tuple(f for f in DIAGNOSTIC_FAMILIES if f in self.diagnostic_families)
        ):
            raise ValueError(
                "multi_model.diagnostic_families must list unique values from "
                f"{list(DIAGNOSTIC_FAMILIES)} in that order"
            )
        if not isinstance(self.stability, EvidenceStabilityConfig):
            raise ValueError('multi_model.stability must be an EvidenceStabilityConfig object')
        self.stability.validate()
        if not isinstance(self.matched_batch, MatchedBatchConfig):
            raise ValueError("multi_model.matched_batch must be a MatchedBatchConfig object")
        self.matched_batch.validate()
        if not isinstance(self.modifier_count, ModifierCountConfig):
            raise ValueError("multi_model.modifier_count must be a ModifierCountConfig object")
        self.modifier_count.validate()
        for name in (
            "repeats",
            "regularization_grid_size",
            "forest_trees",
            "forest_min_samples_leaf",
            "feature_subset_size",
            "permutation_repeats",
            "max_prompt_chars",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"stage2.statistical_selection.multi_model.{name} must be positive integer"
                )
        if self.regularization_grid_size < 3 or self.forest_trees < 4:
            raise ValueError("multi_model requires at least 3 regularization values and 4 trees")
        if self.max_prompt_chars < 4_000:
            raise ValueError("multi_model.max_prompt_chars must be at least 4000")
        for name in ("row_fraction", "nominal_p_threshold", "q_threshold"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 < value <= 1
            ):
                raise ValueError(f"multi_model.{name} must be in (0, 1]")
        if not self.l1_ratios or any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or not 0 < v <= 1
            for v in self.l1_ratios
        ):
            raise ValueError("multi_model.l1_ratios must contain finite values in (0, 1]")
        if (isinstance(self.interaction_ranking_weight, bool)
                or not isinstance(self.interaction_ranking_weight, (int, float))
                or not math.isfinite(self.interaction_ranking_weight)
                or not 0 <= self.interaction_ranking_weight <= 1):
            raise ValueError("multi_model.interaction_ranking_weight must be in [0, 1]")

    def public_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": SCHEMA_VERSION,
            **asdict(self),
            "modifier_count": self.modifier_count.public_dict(),
            "diagnostic_families": list(self.diagnostic_families),
            "evidence_family_policy": deepcopy(EVIDENCE_FAMILY_POLICY),
        }
        # Existing saved runs keep their policy unless this family is requested.
        if self.matched_batch == MatchedBatchConfig():
            result.pop("matched_batch")
        return result

    def active_families(self, outcome_type=None):
        """Evidence families fitted in each inner fold, in canonical order."""
        return tuple(
            f
            for f in FAMILIES
            if (f != "matched_batch_contrast" or self.matched_batch.enabled)
            and (f not in DIAGNOSTIC_FAMILIES or f in self.diagnostic_families)
            and (f != "dina" or outcome_type in {None, "binary"})
        )


def multi_model_config_from_mapping(
    value: Mapping[str, Any] | None,
) -> Stage2MultiModelConfig:
    if value is not None and not isinstance(value, Mapping):
        raise ValueError("stage2.statistical_selection.multi_model must be an object")
    raw = dict(value or {})
    if raw.pop("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ValueError("unsupported multi_model schema_version")
    # public_dict() records the family policy; a saved policy must match it.
    if raw.pop("evidence_family_policy", EVIDENCE_FAMILY_POLICY) != EVIDENCE_FAMILY_POLICY:
        raise ValueError("multi_model.evidence_family_policy does not match this schema")
    unknown = set(raw) - set(Stage2MultiModelConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unsupported multi_model fields: {sorted(unknown)}")
    if "diagnostic_families" in raw:
        names = raw["diagnostic_families"]
        if not isinstance(names, (list, tuple)):
            raise ValueError("multi_model.diagnostic_families must be a list")
        if (
            any(not isinstance(v, str) or v not in DIAGNOSTIC_FAMILIES for v in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError(
                "multi_model.diagnostic_families must list unique values from "
                f"{list(DIAGNOSTIC_FAMILIES)}"
            )
        raw["diagnostic_families"] = tuple(f for f in DIAGNOSTIC_FAMILIES if f in names)
    if "l1_ratios" in raw:
        if not isinstance(raw["l1_ratios"], (list, tuple)):
            raise ValueError("multi_model.l1_ratios must be a list")
        raw["l1_ratios"] = tuple(raw["l1_ratios"])
    if "matched_batch" in raw:
        batch = raw["matched_batch"]
        if not isinstance(batch, Mapping):
            raise ValueError("multi_model.matched_batch must be a configuration object")
        unknown = set(batch) - set(MatchedBatchConfig.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unsupported matched_batch fields: {sorted(unknown)}")
        raw["matched_batch"] = MatchedBatchConfig(**batch)
    if 'stability' in raw:
        stability = raw['stability']
        if not isinstance(stability, Mapping) or set(stability) - set(EvidenceStabilityConfig.__dataclass_fields__):
            raise ValueError('unsupported multi_model.stability configuration')
        raw['stability'] = EvidenceStabilityConfig(**stability)
    if "modifier_count" in raw:
        count = raw["modifier_count"]
        if not isinstance(count, Mapping):
            raise ValueError("multi_model.modifier_count must be an object")
        count = dict(count)
        # The nested count search always runs; accept the former flag only as true.
        if count.pop("enabled", True) is not True:
            raise ValueError(
                "multi_model.modifier_count.enabled=false is no longer supported; "
                "the nested modifier count search always runs"
            )
        unknown = set(count) - set(ModifierCountConfig.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unsupported modifier_count fields: {sorted(unknown)}")
        for name in ("candidate_counts", "estimators"):
            if name in count:
                if not isinstance(count[name], (list, tuple)):
                    raise ValueError(f"modifier_count.{name} must be a list")
                count[name] = tuple(count[name])
        raw["modifier_count"] = ModifierCountConfig(**count)
    result = Stage2MultiModelConfig(**raw)
    result.validate()
    return result
