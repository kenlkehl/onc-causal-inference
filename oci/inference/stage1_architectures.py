"""First-class architecture contract for the research Stage 1 workflow.

This module is intentionally lightweight.  It is imported by configuration,
handoff, Stage 2, and evaluation code, so importing it must not initialize any
model runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

BOW_NUISANCE = "bow_nuisance"
BOW_R_LOSS = "bow_r_loss"
MATCHED_PAIR_UPLIFT = "matched_pair_uplift"
HTR_NEURAL = "htr_neural"
EMBEDDING_WHOLE_COHORT = "embedding_whole_cohort"
EMBEDDING_CLUSTERED = "embedding_clustered"
TFIDF_SEMANTIC_RETRIEVAL = "tfidf_semantic_retrieval_contrasts"
TFIDF_TOPICS = "tfidf_topics"
TFIDF_ORPHAN_NGRAMS = "tfidf_orphan_ngrams"
NEURAL_QUERY_MOMENTS = "neural_query_moments"


@dataclass(frozen=True)
class Stage1ArchitectureSpec:
    """Static orchestration and evaluation metadata for one evidence lane."""

    name: str
    component: str
    support_services: tuple[str, ...]
    native_metric_families: tuple[str, ...]
    description: str


STAGE1_ARCHITECTURE_SPECS = (
    Stage1ArchitectureSpec(
        name=BOW_NUISANCE,
        component="text_models",
        support_services=("split_registry", "bow_views", "bow_nuisance_fits"),
        native_metric_families=("nuisance", "oracle_feature_association"),
        description="Sparse treatment and outcome nuisance evidence.",
    ),
    Stage1ArchitectureSpec(
        name=BOW_R_LOSS,
        component="text_models",
        support_services=(
            "split_registry",
            "bow_views",
            "bow_nuisance_fits",
            "bow_residual_effect",
        ),
        native_metric_families=("r_loss", "oracle_modifier_recovery"),
        description="Sparse residual-effect evidence from R-loss views.",
    ),
    Stage1ArchitectureSpec(
        name=MATCHED_PAIR_UPLIFT,
        component="text_models",
        support_services=(
            "split_registry",
            "bow_nuisance_fits",
            "htr_nuisance_fits",
            "matched_pair_construction",
        ),
        native_metric_families=("matching", "uplift", "oracle_modifier_recovery"),
        description="Text features that separate propensity-matched treatment pairs.",
    ),
    Stage1ArchitectureSpec(
        name=HTR_NEURAL,
        component="text_models",
        support_services=("split_registry", "htr_backbone", "htr_nuisance_fits"),
        native_metric_families=("nuisance", "r_loss", "oracle_attribution"),
        description="Hierarchical-transformer nuisance and heterogeneous-effect evidence.",
    ),
    Stage1ArchitectureSpec(
        name=EMBEDDING_WHOLE_COHORT,
        component="text_models",
        support_services=(
            "split_registry",
            "embedding_cache",
            "whole_cohort_contrasts",
        ),
        native_metric_families=("semantic_recovery", "oracle_feature_association"),
        description="Whole-cohort residualized embedding contrasts.",
    ),
    Stage1ArchitectureSpec(
        name=EMBEDDING_CLUSTERED,
        component="text_models",
        support_services=(
            "split_registry",
            "embedding_cache",
            "whole_cohort_contrasts",
            "cluster_contrasts",
        ),
        native_metric_families=("cluster_stability", "oracle_feature_association"),
        description="Cluster-local residualized embedding contrasts.",
    ),
    Stage1ArchitectureSpec(
        name=TFIDF_SEMANTIC_RETRIEVAL,
        component="text_models",
        support_services=(
            "split_registry",
            "embedding_cache",
            "whole_cohort_contrasts",
            "tfidf_vocabulary",
            "semantic_retrieval",
        ),
        native_metric_families=("retrieval", "oracle_feature_association"),
        description="TF-IDF phrases retrieved from semantic contrast directions.",
    ),
    Stage1ArchitectureSpec(
        name=TFIDF_TOPICS,
        component="tfidf",
        support_services=(
            "split_registry",
            "tfidf_vocabulary",
            "tfidf_nuisance_fits",
            "tfidf_topic_banks",
        ),
        native_metric_families=("topic_stability", "oracle_feature_association"),
        description="Stable TF-IDF/NMF treatment, outcome, and residual-effect topics.",
    ),
    Stage1ArchitectureSpec(
        name=TFIDF_ORPHAN_NGRAMS,
        component="tfidf",
        support_services=(
            "split_registry",
            "tfidf_vocabulary",
            "tfidf_nuisance_fits",
            "tfidf_topic_banks",
            "orphan_ngram_screening",
        ),
        native_metric_families=("ngram_recovery", "oracle_feature_association"),
        description="Stable effect-associated n-grams not represented by selected topics.",
    ),
    Stage1ArchitectureSpec(
        name=NEURAL_QUERY_MOMENTS,
        component="neural_queries",
        support_services=(
            "split_registry",
            "embedding_cache",
            "query_nuisance_fits",
            "neural_query_discovery",
        ),
        native_metric_families=("query_stability", "oracle_feature_association", "r_loss"),
        description="Learned neural cohort-query moments and witness passages.",
    ),
)

STAGE1_ARCHITECTURES = tuple(spec.name for spec in STAGE1_ARCHITECTURE_SPECS)
DEFAULT_STAGE1_ARCHITECTURES = tuple(
    name
    for name in STAGE1_ARCHITECTURES
    if name not in {TFIDF_SEMANTIC_RETRIEVAL, TFIDF_ORPHAN_NGRAMS}
)
STAGE1_ARCHITECTURE_REGISTRY: Mapping[str, Stage1ArchitectureSpec] = {
    spec.name: spec for spec in STAGE1_ARCHITECTURE_SPECS
}


def canonicalize_stage1_architectures(
    value: str | Sequence[str] | None,
    *,
    allow_none: bool = True,
) -> tuple[str, ...] | None:
    """Validate and return a selection in canonical registry order."""

    if value is None:
        if allow_none:
            return None
        return STAGE1_ARCHITECTURES
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() == "all":
            return STAGE1_ARCHITECTURES
        raw = [part.strip() for part in stripped.split(",") if part.strip()]
    else:
        raw = [str(part).strip() for part in value]
    if not raw or any(not part for part in raw):
        raise ValueError("science.stage1_architectures must select at least one architecture")
    duplicates = sorted({name for name in raw if raw.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate Stage 1 architectures: {duplicates}")
    unknown = sorted(set(raw) - set(STAGE1_ARCHITECTURES))
    if unknown:
        raise ValueError(f"unknown Stage 1 architectures: {unknown}")
    selected = set(raw)
    return tuple(name for name in STAGE1_ARCHITECTURES if name in selected)


def legacy_enabled_stage1_architectures(
    applied_config: Any,
    *,
    outcome_type: str,
) -> tuple[str, ...]:
    """Resolve enabled lanes from their implementation settings."""

    mm_config = applied_config.architecture.multi_model_forest
    enabled: set[str] = set()
    bow_enabled = bool(getattr(mm_config, "bow_discovery_enabled", True))
    htr_enabled = bool(getattr(mm_config, "htr_evidence_enabled", True))
    if bow_enabled:
        enabled.update((BOW_NUISANCE, BOW_R_LOSS))
    if htr_enabled:
        enabled.add(HTR_NEURAL)

    matched_enabled = (
        str(outcome_type).lower() != "continuous"
        and bool(getattr(mm_config, "matched_pair_uplift_enabled", True))
        and (
            bow_enabled
            and bool(getattr(mm_config, "matched_pair_bow_enabled", True))
            or htr_enabled
            and bool(getattr(mm_config, "matched_pair_htr_enabled", True))
        )
    )
    if matched_enabled:
        enabled.add(MATCHED_PAIR_UPLIFT)

    embedding = mm_config.embedding_contrast
    if bool(getattr(embedding, "enabled", False)):
        enabled.add(EMBEDDING_WHOLE_COHORT)
        if bool(getattr(embedding, "include_cluster_contrast_vectors", True)):
            enabled.add(EMBEDDING_CLUSTERED)
        if bool(getattr(embedding, "retrieval_tfidf_enabled", False)):
            enabled.add(TFIDF_SEMANTIC_RETRIEVAL)

    enabled.add(TFIDF_TOPICS)
    if bool(getattr(mm_config.tfidf_topic, "orphan_ngram_enabled", False)):
        enabled.add(TFIDF_ORPHAN_NGRAMS)
    enabled.add(NEURAL_QUERY_MOMENTS)
    return tuple(name for name in STAGE1_ARCHITECTURES if name in enabled)


def unavailable_explicit_architectures(
    selected: Iterable[str],
    applied_config: Any,
    *,
    outcome_type: str,
) -> tuple[str, ...]:
    """Return selected lanes unavailable under their prerequisite settings.

    Naming either optional lexical lane explicitly opts into it. Its false
    default does not veto the selector that enables it in the applied mapping;
    embedding availability and orphan score-test prerequisites still apply.
    """

    available = set(
        legacy_enabled_stage1_architectures(
            applied_config,
            outcome_type=outcome_type,
        )
    )
    mm_config = applied_config.architecture.multi_model_forest
    if bool(getattr(mm_config.embedding_contrast, "enabled", False)):
        available.add(TFIDF_SEMANTIC_RETRIEVAL)
    if bool(getattr(mm_config.tfidf_topic, "score_test_enabled", True)):
        available.add(TFIDF_ORPHAN_NGRAMS)
    return tuple(name for name in STAGE1_ARCHITECTURES if name in set(selected) - available)


def resolve_support_services(selected: Iterable[str]) -> tuple[str, ...]:
    """Return the stable first-use ordering of private support services."""

    output: list[str] = []
    for name in canonicalize_stage1_architectures(tuple(selected), allow_none=False) or ():
        for service in STAGE1_ARCHITECTURE_REGISTRY[name].support_services:
            if service not in output:
                output.append(service)
    return tuple(output)


def selected_components(selected: Iterable[str]) -> tuple[str, ...]:
    """Return scientific producer components in workflow execution order."""

    canonical = canonicalize_stage1_architectures(tuple(selected), allow_none=False) or ()
    components = {STAGE1_ARCHITECTURE_REGISTRY[name].component for name in canonical}
    if any(
        "embedding_cache" in STAGE1_ARCHITECTURE_REGISTRY[name].support_services
        for name in canonical
    ):
        components.add("embedding_cache")
    return tuple(
        name
        for name in ("embedding_cache", "tfidf", "text_models", "neural_queries", "handoff")
        if name == "handoff" or name in components
    )


__all__ = [
    "BOW_NUISANCE",
    "BOW_R_LOSS",
    "DEFAULT_STAGE1_ARCHITECTURES",
    "EMBEDDING_CLUSTERED",
    "EMBEDDING_WHOLE_COHORT",
    "HTR_NEURAL",
    "MATCHED_PAIR_UPLIFT",
    "NEURAL_QUERY_MOMENTS",
    "STAGE1_ARCHITECTURES",
    "STAGE1_ARCHITECTURE_REGISTRY",
    "STAGE1_ARCHITECTURE_SPECS",
    "Stage1ArchitectureSpec",
    "TFIDF_ORPHAN_NGRAMS",
    "TFIDF_SEMANTIC_RETRIEVAL",
    "TFIDF_TOPICS",
    "canonicalize_stage1_architectures",
    "legacy_enabled_stage1_architectures",
    "resolve_support_services",
    "selected_components",
    "unavailable_explicit_architectures",
]
