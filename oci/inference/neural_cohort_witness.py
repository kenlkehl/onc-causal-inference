"""Neural soft witnesses for cohort-level treatment-effect contrast moments.

The encoder is deliberately outside this module.  Callers provide frozen chunk
embeddings and this module learns a small bank of semantic queries.  A query's
patient value is a smooth maximum over that patient's chunk similarities.  The
queries optimize a cohort score moment, not a patient-level pseudo-outcome.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class NeuralCohortWitnessConfig:
    """Configuration for a small, constrained soft-witness bank."""

    n_prototypes: int = 16
    initial_pool_size: int = 48
    temperature: float = 0.05
    learning_rate: float = 0.025
    epochs: int = 160
    max_query_drift: float = 0.35
    query_diversity_weight: float = 0.08
    activation_diversity_weight: float = 0.04
    anchor_weight: float = 0.15
    min_activation_sd: float = 0.01
    activation_scale_weight: float = 1.0
    kmeans_iterations: int = 20
    kmeans_sample_chunks: int = 6000
    initialization_max_cosine: float = 0.96
    epsilon: float = 1e-6
    optimizer_beta1: float = 0.9
    optimizer_beta2: float = 0.999
    optimizer_epsilon: float = 1e-8
    optimizer_weight_decay: float = 0.0
    optimizer_amsgrad: bool = False
    optimizer_maximize: bool = False
    optimizer_foreach: bool = False
    optimizer_capturable: bool = False
    optimizer_differentiable: bool = False
    optimizer_fused: bool = False
    gradient_clip_norm: float = 5.0
    consensus_kmeans_init: str = "k-means++"
    consensus_kmeans_n_init: int = 20
    consensus_kmeans_max_iter: int = 300
    consensus_kmeans_tolerance: float = 1e-4
    consensus_kmeans_copy_x: bool = True
    consensus_kmeans_algorithm: str = "lloyd"

    def validate(self) -> None:
        integer_fields = (
            "n_prototypes",
            "initial_pool_size",
            "epochs",
            "kmeans_iterations",
            "kmeans_sample_chunks",
            "consensus_kmeans_n_init",
            "consensus_kmeans_max_iter",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        finite_fields = (
            "temperature",
            "learning_rate",
            "max_query_drift",
            "query_diversity_weight",
            "activation_diversity_weight",
            "anchor_weight",
            "min_activation_sd",
            "activation_scale_weight",
            "initialization_max_cosine",
            "epsilon",
            "optimizer_beta1",
            "optimizer_beta2",
            "optimizer_epsilon",
            "optimizer_weight_decay",
            "gradient_clip_norm",
            "consensus_kmeans_tolerance",
        )
        for name in finite_fields:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be finite")
        boolean_fields = (
            "optimizer_amsgrad",
            "optimizer_maximize",
            "optimizer_foreach",
            "optimizer_capturable",
            "optimizer_differentiable",
            "optimizer_fused",
            "consensus_kmeans_copy_x",
        )
        for name in boolean_fields:
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        if self.initial_pool_size < self.n_prototypes:
            raise ValueError("initial_pool_size must be at least n_prototypes")
        if not 0.0 < self.temperature <= 1.0:
            raise ValueError("temperature must be in (0, 1]")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.max_query_drift <= 2.0:
            raise ValueError("max_query_drift must be in [0, 2]")
        for name in (
            "query_diversity_weight",
            "activation_diversity_weight",
            "anchor_weight",
            "min_activation_sd",
            "activation_scale_weight",
            "optimizer_weight_decay",
            "consensus_kmeans_tolerance",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if not 0.0 <= self.initialization_max_cosine <= 1.0:
            raise ValueError("initialization_max_cosine must be in [0, 1]")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if not 0.0 <= self.optimizer_beta1 < 1.0:
            raise ValueError("optimizer_beta1 must be in [0, 1)")
        if not 0.0 <= self.optimizer_beta2 < 1.0:
            raise ValueError("optimizer_beta2 must be in [0, 1)")
        if self.optimizer_epsilon <= 0.0:
            raise ValueError("optimizer_epsilon must be positive")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        if self.optimizer_foreach and self.optimizer_fused:
            raise ValueError("optimizer_foreach and optimizer_fused cannot both be true")
        if self.consensus_kmeans_init not in {"k-means++", "random"}:
            raise ValueError("consensus_kmeans_init must be k-means++ or random")
        if self.consensus_kmeans_algorithm not in {"lloyd", "elkan"}:
            raise ValueError("consensus_kmeans_algorithm must be lloyd or elkan")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _adamw_parameters(config: NeuralCohortWitnessConfig) -> Dict[str, Any]:
    """Return the complete result-changing AdamW constructor contract."""

    return {
        "lr": float(config.learning_rate),
        "betas": (
            float(config.optimizer_beta1),
            float(config.optimizer_beta2),
        ),
        "eps": float(config.optimizer_epsilon),
        "weight_decay": float(config.optimizer_weight_decay),
        "amsgrad": bool(config.optimizer_amsgrad),
        "maximize": bool(config.optimizer_maximize),
        "foreach": bool(config.optimizer_foreach),
        "capturable": bool(config.optimizer_capturable),
        "differentiable": bool(config.optimizer_differentiable),
        "fused": bool(config.optimizer_fused),
    }


def fit_constant_residual_effect(
    treatment_residual: np.ndarray,
    outcome_residual: np.ndarray,
) -> float:
    """Fit the constant residual effect by least squares through the origin."""

    u = np.asarray(treatment_residual, dtype=float).reshape(-1)
    v = np.asarray(outcome_residual, dtype=float).reshape(-1)
    if u.shape != v.shape:
        raise ValueError("treatment and outcome residuals must have matching shapes")
    denominator = float(np.dot(u, u))
    return 0.0 if denominator <= 0.0 else float(np.dot(u, v) / denominator)


def cohort_contribution(
    treatment_residual: np.ndarray,
    outcome_residual: np.ndarray,
    *,
    constant_effect: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """Return the constant-effect-orthogonalized cohort contribution."""

    u = np.asarray(treatment_residual, dtype=float).reshape(-1)
    v = np.asarray(outcome_residual, dtype=float).reshape(-1)
    if constant_effect is None:
        constant_effect = fit_constant_residual_effect(u, v)
    contribution = u * (v - float(constant_effect) * u)
    return contribution, float(constant_effect)


def direct_target_contribution(
    target: np.ndarray,
    *,
    binary: bool,
) -> np.ndarray:
    """Return a balanced patient-level contrast for a nuisance target.

    For a binary target this is the inverse-prevalence contrast whose sample
    covariance with an activation is exactly the treated/untreated difference
    in mean activation (up to the common sample normalization).  For a
    continuous target it is the centered, scale-normalized target.  Unlike the
    effect witness contribution, neither path constructs a treatment-effect
    pseudo-target.
    """

    values = np.asarray(target, dtype=float).reshape(-1)
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("target must be a non-empty finite vector")
    if binary:
        unique = set(np.unique(values).tolist())
        if not unique.issubset({0.0, 1.0}) or len(unique) != 2:
            raise ValueError("binary target must contain both 0 and 1")
        prevalence = float(np.mean(values))
        if not 0.0 < prevalence < 1.0:
            raise ValueError("binary target prevalence must be in (0, 1)")
        return values / prevalence - (1.0 - values) / (1.0 - prevalence)
    centered = values - float(np.mean(values))
    scale = float(np.std(centered))
    return centered if scale <= 0.0 else centered / scale


def standardized_direct_target_contrasts(
    activations: np.ndarray,
    target: np.ndarray,
    *,
    binary: bool,
) -> Dict[str, np.ndarray]:
    """Score fixed activations against a treatment or outcome target."""

    values = np.asarray(activations, dtype=float)
    contribution = direct_target_contribution(target, binary=binary)
    if values.ndim != 2 or len(values) != len(contribution):
        raise ValueError("Activation and target row counts must match")
    centers = np.mean(values, axis=0)
    row_scores = (values - centers) * contribution[:, None]
    moments = np.mean(row_scores, axis=0)
    scales = np.std(row_scores, axis=0, ddof=1)
    standardized = np.sqrt(len(values)) * moments / np.maximum(scales, 1e-12)
    return {
        "centers": centers,
        "target_contribution": contribution,
        "row_scores": row_scores,
        "moments": moments,
        "scales": scales,
        "standardized_scores": standardized,
        "two_sided_p_values": 2.0 * norm.sf(np.abs(standardized)),
    }


def pad_chunk_embeddings(
    chunk_matrices: Sequence[np.ndarray],
    *,
    dtype: np.dtype = np.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pad variable-length, normalized chunk embeddings for batched retrieval."""

    if not chunk_matrices:
        raise ValueError("At least one patient's chunk embeddings are required")
    dimensions = {
        int(np.asarray(matrix).shape[1])
        for matrix in chunk_matrices
        if np.asarray(matrix).ndim == 2 and np.asarray(matrix).shape[0] > 0
    }
    if len(dimensions) != 1:
        raise ValueError("Chunk matrices must share one non-empty embedding dimension")
    dimension = dimensions.pop()
    max_chunks = max(max(int(np.asarray(matrix).shape[0]), 1) for matrix in chunk_matrices)
    padded = np.zeros((len(chunk_matrices), max_chunks, dimension), dtype=dtype)
    mask = np.zeros((len(chunk_matrices), max_chunks), dtype=bool)
    for index, raw in enumerate(chunk_matrices):
        matrix = np.asarray(raw, dtype=dtype)
        if matrix.ndim != 2 or matrix.shape[1] != dimension:
            raise ValueError("Every chunk matrix must have shape (chunks, embedding_dim)")
        if not len(matrix):
            mask[index, 0] = True
            continue
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.maximum(norms, 1e-12)
        padded[index, : len(matrix)] = matrix
        mask[index, : len(matrix)] = True
    return padded, mask


def soft_retrieval_activations(
    chunk_matrices: Sequence[np.ndarray],
    queries: np.ndarray,
    *,
    temperature: float,
    device: str,
    patient_batch_size: int,
) -> np.ndarray:
    """Compute log-mean-exp patient activations for fixed semantic queries."""

    import torch

    if (
        isinstance(patient_batch_size, bool)
        or not isinstance(patient_batch_size, int)
        or patient_batch_size < 1
    ):
        raise ValueError("patient_batch_size must be a positive integer")
    padded, mask = pad_chunk_embeddings(chunk_matrices)
    query_array = np.asarray(queries, dtype=np.float32)
    if query_array.ndim != 2 or query_array.shape[1] != padded.shape[2]:
        raise ValueError("queries must have shape (queries, embedding_dim)")
    query_array /= np.maximum(np.linalg.norm(query_array, axis=1, keepdims=True), 1e-12)
    query_tensor = torch.as_tensor(query_array, device=device)
    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(padded), int(patient_batch_size)):
            stop = min(len(padded), start + int(patient_batch_size))
            chunks = torch.as_tensor(padded[start:stop], device=device)
            chunk_mask = torch.as_tensor(mask[start:stop], device=device)
            activations = _torch_soft_activations(
                chunks,
                chunk_mask,
                query_tensor,
                temperature=float(temperature),
            )
            outputs.append(activations.detach().cpu().numpy().astype(np.float32))
    return np.vstack(outputs)


def standardized_cohort_moments(
    activations: np.ndarray,
    treatment_residual: np.ndarray,
    outcome_residual: np.ndarray,
    *,
    constant_effect: float,
    center_with_evaluation_treatment: bool = True,
    fixed_centers: Optional[np.ndarray] = None,
    score_contribution=None,
    score_weights=None,
) -> Dict[str, np.ndarray]:
    """Score fixed witness activations using cohort-level moments.

    Centering by squared treatment residual makes each interaction column
    orthogonal to the constant-effect score.  During a held-out score test this
    may use held-out treatment and nuisance predictions, but never outcomes, to
    determine the centering constants.
    """

    values = np.asarray(activations, dtype=float)
    u = np.asarray(treatment_residual, dtype=float).reshape(-1)
    v = np.asarray(outcome_residual, dtype=float).reshape(-1)
    if values.ndim != 2 or len(values) != len(u) or len(u) != len(v):
        raise ValueError("Activation and residual row counts must match")
    if fixed_centers is not None:
        centers = np.asarray(fixed_centers, dtype=float).reshape(-1)
    elif center_with_evaluation_treatment:
        weights = np.square(u) if score_weights is None else np.asarray(score_weights, float)
        denominator = float(np.sum(weights))
        centers = (
            np.mean(values, axis=0)
            if denominator <= 0.0
            else np.sum(weights[:, None] * values, axis=0) / denominator
        )
    else:
        centers = np.mean(values, axis=0)
    if len(centers) != values.shape[1]:
        raise ValueError("One centering constant is required per activation column")
    contribution, _ = cohort_contribution(u, v, constant_effect=constant_effect)
    if score_contribution is not None:
        contribution = np.asarray(score_contribution, float)
    row_scores = (values - centers) * contribution[:, None]
    moments = np.mean(row_scores, axis=0)
    scales = np.std(row_scores, axis=0, ddof=1)
    standardized = np.sqrt(len(values)) * moments / np.maximum(scales, 1e-12)
    return {
        "centers": centers,
        "cohort_contribution": contribution,
        "row_scores": row_scores,
        "moments": moments,
        "scales": scales,
        "standardized_scores": standardized,
        "two_sided_p_values": 2.0 * norm.sf(np.abs(standardized)),
    }


def fit_soft_contrast_queries(
    chunk_matrices: Sequence[np.ndarray],
    contribution: np.ndarray,
    *,
    config: NeuralCohortWitnessConfig,
    seed: int,
    device: str,
    center_weights: Optional[np.ndarray] = None,
    initial_queries: Optional[np.ndarray] = None,
    objective_name: str = "direct_target_contrast",
) -> Dict[str, Any]:
    """Fit semantic queries against any fixed training-row contribution.

    This is the reusable optimizer for the ungated three-bank workflow.  The
    caller must construct ``contribution`` entirely from the supplied training
    rows.  There are deliberately no validation or external-held-out inputs.
    Supplying ``initial_queries`` refits a fixed-size consensus bank on a larger
    training context without rerunning label-aware candidate selection.
    """

    import torch
    import torch.nn.functional as functional

    config.validate()
    torch.manual_seed(int(seed))
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))

    padded, mask = pad_chunk_embeddings(chunk_matrices)
    row_contribution = np.asarray(contribution, dtype=np.float32).reshape(-1)
    if len(padded) != len(row_contribution):
        raise ValueError("contribution must match the number of patients")
    if not np.all(np.isfinite(row_contribution)):
        raise ValueError("contribution must be finite")
    if center_weights is None:
        weights = np.ones(len(row_contribution), dtype=np.float32)
    else:
        weights = np.asarray(center_weights, dtype=np.float32).reshape(-1)
        if len(weights) != len(row_contribution):
            raise ValueError("center_weights must match the number of patients")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("center_weights must be finite and non-negative")
    if float(np.sum(weights)) <= 0.0:
        weights = np.ones(len(row_contribution), dtype=np.float32)

    chunks = torch.as_tensor(padded, device=device)
    chunk_mask = torch.as_tensor(mask, device=device)
    contribution_tensor = torch.as_tensor(row_contribution, device=device)
    weight_tensor = torch.as_tensor(weights, device=device)
    weight_denominator = torch.clamp(
        weight_tensor.sum(), min=float(config.epsilon)
    )

    pool_scores = np.empty(0, dtype=float)
    selected_indices: List[int] = []
    if initial_queries is None:
        flat_chunks = chunks[chunk_mask]
        pool = _torch_spherical_kmeans(
            flat_chunks,
            n_clusters=min(int(config.initial_pool_size), len(flat_chunks)),
            iterations=int(config.kmeans_iterations),
            max_samples=int(config.kmeans_sample_chunks),
            seed=int(seed),
        )
        with torch.no_grad():
            pool_activations = _torch_soft_activations(
                chunks,
                chunk_mask,
                pool,
                temperature=float(config.temperature),
            )
            pool_z = _torch_standardized_moments(
                pool_activations,
                contribution_tensor,
                weight_tensor,
                weight_denominator,
                epsilon=float(config.epsilon),
            )
            selected_indices = _greedy_diverse_indices(
                pool.detach().cpu().numpy(),
                np.abs(pool_z.detach().cpu().numpy()),
                count=int(config.n_prototypes),
                max_cosine=float(config.initialization_max_cosine),
            )
            initial_tensor = pool[selected_indices].detach().clone()
            pool_scores = pool_z.detach().cpu().numpy().astype(float)
    else:
        initial_array = np.asarray(initial_queries, dtype=np.float32)
        if initial_array.ndim != 2 or initial_array.shape != (
            int(config.n_prototypes),
            int(padded.shape[2]),
        ):
            raise ValueError(
                "initial_queries must have shape "
                f"({config.n_prototypes}, {padded.shape[2]})"
            )
        initial_array /= np.maximum(
            np.linalg.norm(initial_array, axis=1, keepdims=True), 1e-12
        )
        initial_tensor = torch.as_tensor(initial_array, device=device).clone()

    queries = torch.nn.Parameter(initial_tensor.clone())
    optimizer = torch.optim.AdamW([queries], **_adamw_parameters(config))
    loss_history: List[float] = []
    for _epoch in range(int(config.epochs)):
        optimizer.zero_grad(set_to_none=True)
        normalized_queries = functional.normalize(queries, dim=1)
        activations = _torch_soft_activations(
            chunks,
            chunk_mask,
            normalized_queries,
            temperature=float(config.temperature),
        )
        z_scores = _torch_standardized_moments(
            activations,
            contribution_tensor,
            weight_tensor,
            weight_denominator,
            epsilon=float(config.epsilon),
        )
        signal = torch.mean(torch.log1p(z_scores.square()))

        query_gram = normalized_queries @ normalized_queries.T
        query_off_diagonal = query_gram - torch.diag_embed(
            torch.diagonal(query_gram)
        )
        query_diversity = query_off_diagonal.square().mean()

        centered_activations = activations - activations.mean(dim=0, keepdim=True)
        activation_norms = torch.sqrt(
            torch.sum(centered_activations.square(), dim=0, keepdim=True)
            + float(config.epsilon)
        )
        normalized_activations = centered_activations / activation_norms
        activation_gram = normalized_activations.T @ normalized_activations
        activation_off_diagonal = activation_gram - torch.diag_embed(
            torch.diagonal(activation_gram)
        )
        activation_diversity = activation_off_diagonal.square().mean()
        anchor = (
            1.0 - torch.sum(normalized_queries * initial_tensor, dim=1)
        ).mean()
        activation_sd = torch.std(activations, dim=0, unbiased=False)
        activation_scale_penalty = torch.relu(
            float(config.min_activation_sd) - activation_sd
        ).square().mean()
        loss = (
            -signal
            + float(config.query_diversity_weight) * query_diversity
            + float(config.activation_diversity_weight) * activation_diversity
            + float(config.anchor_weight) * anchor
            + float(config.activation_scale_weight) * activation_scale_penalty
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [queries],
            max_norm=float(config.gradient_clip_norm),
        )
        optimizer.step()
        with torch.no_grad():
            queries.copy_(functional.normalize(queries, dim=1))
            difference = queries - initial_tensor
            difference_norm = torch.linalg.vector_norm(
                difference, dim=1, keepdim=True
            )
            multiplier = torch.clamp(
                float(config.max_query_drift)
                / torch.clamp(difference_norm, min=float(config.epsilon)),
                max=1.0,
            )
            queries.copy_(
                functional.normalize(initial_tensor + difference * multiplier, dim=1)
            )
        loss_history.append(float(loss.detach().cpu()))

    with torch.no_grad():
        final_queries = functional.normalize(queries, dim=1)
        final_activations = _torch_soft_activations(
            chunks,
            chunk_mask,
            final_queries,
            temperature=float(config.temperature),
        )
        train_z = _torch_standardized_moments(
            final_activations,
            contribution_tensor,
            weight_tensor,
            weight_denominator,
            epsilon=float(config.epsilon),
        )
        drift = torch.linalg.vector_norm(final_queries - initial_tensor, dim=1)

    return {
        "queries": final_queries.detach().cpu().numpy().astype(np.float32),
        "initial_queries": initial_tensor.detach().cpu().numpy().astype(np.float32),
        "train_activations": final_activations.detach().cpu().numpy().astype(np.float32),
        "train_standardized_scores": train_z.detach().cpu().numpy().astype(float),
        "query_drift": drift.detach().cpu().numpy().astype(float),
        "initial_pool_standardized_scores": pool_scores,
        "initial_pool_selected_indices": selected_indices,
        "loss_history": loss_history,
        "objective": str(objective_name),
        "initialized_from_fixed_queries": initial_queries is not None,
        "no_validation_or_heldout_rows_consumed": True,
    }


def fit_soft_target_queries(
    chunk_matrices: Sequence[np.ndarray],
    target: np.ndarray,
    *,
    binary: bool,
    config: NeuralCohortWitnessConfig,
    seed: int,
    device: str,
    initial_queries: Optional[np.ndarray] = None,
    target_name: str = "target",
) -> Dict[str, Any]:
    """Fit treatment/outcome queries using a direct patient-level contrast."""

    contribution = direct_target_contribution(target, binary=binary)
    return fit_soft_contrast_queries(
        chunk_matrices,
        contribution,
        config=config,
        seed=seed,
        device=device,
        center_weights=np.ones(len(contribution), dtype=np.float32),
        initial_queries=initial_queries,
        objective_name=(
            f"maximize standardized direct {target_name} contrast of smooth "
            "semantic activations"
        ),
    )


def build_ungated_consensus_query_bank(
    candidates: Sequence[Dict[str, Any]],
    *,
    candidate_activations: np.ndarray,
    n_queries: int,
    bank: str,
    seed: int,
    config: NeuralCohortWitnessConfig,
) -> Dict[str, Any]:
    """Consolidate every fold-specific query into exactly ``n_queries`` groups.

    There is intentionally no score threshold, sign check, recurrence rule, or
    p-value filter.  K-means operates on centered patient activation patterns;
    every candidate belongs to one group.  The activation medoid is used as the
    sharp semantic initializer for full-context refitting, while every member
    remains attached as agent-visible provenance.
    """

    from sklearn.cluster import KMeans

    if not isinstance(config, NeuralCohortWitnessConfig):
        raise TypeError("config must be NeuralCohortWitnessConfig")
    config.validate()
    if int(n_queries) < 1:
        raise ValueError("n_queries must be positive")
    if int(n_queries) != int(config.n_prototypes):
        raise ValueError("n_queries must equal the configured witness prototype count")
    if len(candidates) < int(n_queries):
        raise ValueError("at least n_queries candidates are required")
    values = np.asarray(candidate_activations, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(candidates):
        raise ValueError("candidate_activations must have one column per candidate")
    centered = values - np.mean(values, axis=0, keepdims=True)
    scales = np.linalg.norm(centered, axis=0, keepdims=True)
    standardized = centered / np.maximum(scales, 1e-12)
    representations = standardized.T
    labels = KMeans(
        n_clusters=int(n_queries),
        init=str(config.consensus_kmeans_init),
        n_init=int(config.consensus_kmeans_n_init),
        max_iter=int(config.consensus_kmeans_max_iter),
        tol=float(config.consensus_kmeans_tolerance),
        random_state=int(seed),
        copy_x=bool(config.consensus_kmeans_copy_x),
        algorithm=str(config.consensus_kmeans_algorithm),
    ).fit_predict(representations)

    groups: List[Dict[str, Any]] = []
    for label in range(int(n_queries)):
        member_indices = np.flatnonzero(labels == label)
        if not len(member_indices):
            raise RuntimeError("ungated consensus unexpectedly produced an empty group")
        centroid = np.mean(representations[member_indices], axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        similarities = representations[member_indices] @ centroid
        medoid_index = int(member_indices[int(np.argmax(similarities))])
        member_scores = [
            abs(float(candidates[int(index)].get("train_standardized_score", 0.0)))
            for index in member_indices
        ]
        groups.append(
            {
                "raw_cluster_label": int(label),
                "medoid_index": medoid_index,
                "medoid_query": _normalize_vector(
                    np.asarray(candidates[medoid_index]["query"], dtype=float)
                ),
                "median_abs_train_score": float(np.median(member_scores)),
                "member_indices": [int(value) for value in member_indices],
            }
        )
    groups.sort(
        key=lambda item: (
            -float(item["median_abs_train_score"]),
            str(candidates[int(item["medoid_index"])].get("candidate_id", "")),
        )
    )

    records: List[Dict[str, Any]] = []
    queries: List[np.ndarray] = []
    assigned: List[int] = []
    for index, group in enumerate(groups, start=1):
        member_indices = group["member_indices"]
        assigned.extend(member_indices)
        queries.append(np.asarray(group["medoid_query"], dtype=np.float32))
        member_subfolds = sorted(
            {
                int(candidates[member_index].get("subfold", 0))
                for member_index in member_indices
            }
        )
        records.append(
            {
                "query_id": f"{bank}_query_{index:03d}",
                "bank": str(bank),
                "medoid_candidate_id": str(
                    candidates[int(group["medoid_index"])].get("candidate_id", "")
                ),
                "member_count": len(member_indices),
                "member_subfolds": member_subfolds,
                "subfold_recurrence": len(member_subfolds),
                "median_abs_train_score": float(group["median_abs_train_score"]),
                "members": [
                    {
                        key: value
                        for key, value in candidates[member_index].items()
                        if key != "query"
                    }
                    for member_index in member_indices
                ],
            }
        )
    if sorted(assigned) != list(range(len(candidates))):
        raise RuntimeError("Every candidate must be assigned exactly once")
    return {
        "queries": np.vstack(queries).astype(np.float32),
        "records": records,
        "bank": str(bank),
        "candidate_count": len(candidates),
        "selected_count": int(n_queries),
        "all_candidates_assigned": True,
        "statistical_gate_applied": False,
        "consolidation": "forced_kmeans_on_centered_patient_activations_with_medoid",
    }


def _torch_soft_activations(chunks, mask, queries, *, temperature: float):
    import torch

    similarities = torch.matmul(chunks, queries.T)
    scaled = similarities / float(temperature)
    scaled = scaled.masked_fill(~mask[:, :, None], -torch.inf)
    counts = torch.clamp(mask.sum(dim=1, keepdim=True), min=1)
    return float(temperature) * (
        torch.logsumexp(scaled, dim=1) - torch.log(counts.to(scaled.dtype))
    )


def _torch_standardized_moments(
    activations,
    contribution,
    treatment_weights,
    weight_denominator,
    *,
    epsilon: float,
):
    import torch

    centers = torch.sum(treatment_weights[:, None] * activations, dim=0) / weight_denominator
    row_scores = (activations - centers) * contribution[:, None]
    means = row_scores.mean(dim=0)
    scales = row_scores.std(dim=0, unbiased=False)
    return np.sqrt(len(activations)) * means / torch.clamp(scales, min=float(epsilon))


def _torch_spherical_kmeans(
    flat_chunks,
    *,
    n_clusters: int,
    iterations: int,
    max_samples: int,
    seed: int,
):
    import torch
    import torch.nn.functional as functional

    generator = torch.Generator(device=flat_chunks.device)
    generator.manual_seed(int(seed))
    if len(flat_chunks) > int(max_samples):
        indices = torch.randperm(
            len(flat_chunks), generator=generator, device=flat_chunks.device
        )[: int(max_samples)]
        samples = flat_chunks[indices]
    else:
        samples = flat_chunks
    samples = functional.normalize(samples.float(), dim=1)
    n_clusters = min(int(n_clusters), len(samples))
    initial_indices = torch.randperm(
        len(samples), generator=generator, device=samples.device
    )[:n_clusters]
    centers = samples[initial_indices].clone()
    for _ in range(max(1, int(iterations))):
        labels = torch.argmax(samples @ centers.T, dim=1)
        sums = torch.zeros_like(centers)
        sums.index_add_(0, labels, samples)
        counts = torch.bincount(labels, minlength=n_clusters)
        empty = counts == 0
        if torch.any(empty):
            replacements = torch.randperm(
                len(samples), generator=generator, device=samples.device
            )[: int(empty.sum())]
            sums[empty] = samples[replacements]
        centers = functional.normalize(sums, dim=1)
    return centers


def _greedy_diverse_indices(
    vectors: np.ndarray,
    scores: np.ndarray,
    *,
    count: int,
    max_cosine: float,
) -> List[int]:
    normalized = np.asarray(vectors, dtype=float)
    normalized /= np.maximum(np.linalg.norm(normalized, axis=1, keepdims=True), 1e-12)
    selected: List[int] = []
    for index in np.argsort(np.asarray(scores, dtype=float))[::-1]:
        if selected and np.max(normalized[selected] @ normalized[int(index)]) > max_cosine:
            continue
        selected.append(int(index))
        if len(selected) >= int(count):
            break
    if len(selected) < int(count):
        for index in np.argsort(np.asarray(scores, dtype=float))[::-1]:
            if int(index) not in selected:
                selected.append(int(index))
            if len(selected) >= int(count):
                break
    return selected


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=float).reshape(-1)
    norm_value = float(np.linalg.norm(array))
    return array if norm_value <= 0.0 else array / norm_value
