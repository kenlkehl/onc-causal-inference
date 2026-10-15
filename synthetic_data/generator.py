# synthetic_data/generator.py
"""Main synthetic data generation pipeline."""

import logging
import json
import random
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import SyntheticDataConfig
from .llm_client import LLMClient
from .prompts import (
    CLINICAL_SYSTEM_PROMPT,
    CONFOUNDER_GENERATION_PROMPT,
    REGRESSION_EQUATION_PROMPT,
    SUMMARY_STATISTICS_PROMPT,
    NOTE_EXPANSION_PROMPT,
    DRUG_PERTURBATION_MAP,
    format_confounder_list,
    format_patient_characteristics,
    build_event_timeline_prompt,
)
from .structured_data import convert_structured_event_to_text


logger = logging.getLogger(__name__)


def generate_synthetic_dataset(
    config: SyntheticDataConfig,
    num_workers: int = 4,
    show_progress: bool = True,
    dgp_metadata_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate a synthetic clinical dataset with known causal structure.
    
    This pipeline:
    1. Uses LLM to generate realistic role-tagged features based on clinical question
    2. Uses LLM to generate treatment and outcome regression equations
    3. Uses LLM to generate summary statistics for confounders
    4. For each patient: samples characteristics, computes logits, generates clinical history
    5. Saves dataset and metadata
    
    Args:
        config: Configuration for generation
        num_workers: Number of parallel workers for patient history generation
        show_progress: Whether to show progress bar
        dgp_metadata_path: Optional saved metadata.json; when given, its features,
            equations and distributions are reused (see ``saved_dgp``)
        
    Returns:
        Tuple of (dataset DataFrame, metadata dictionary)
    """
    if dgp_metadata_path is not None:
        from .saved_dgp import generate_from_saved_dgp
        return generate_from_saved_dgp(config, dgp_metadata_path, num_workers, show_progress)

    config.validate()
    
    # Set random seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    
    # Initialize LLM client
    client = LLMClient(config.llm)
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting synthetic data generation for: {config.clinical_question[:80]}...")

    # Log positivity enforcement settings
    if getattr(config, 'enforce_positivity', False):
        logger.info(f"Positivity enforcement ENABLED: treatment rate per stratum bounded to [{config.min_treatment_rate_per_stratum:.2f}, {config.max_treatment_rate_per_stratum:.2f}]")
    else:
        logger.info("Positivity enforcement disabled (realistic observational data)")

    # Step 1: Generate role-tagged features
    logger.info("Step 1/5: Generating role-tagged features...")
    confounders = _generate_confounders(
        client,
        config.clinical_question,
        num_features=config.num_features,
        num_confounders=config.num_confounders,
        num_effect_modifiers=config.num_effect_modifiers,
    )
    logger.info(f"Generated {len(confounders)} features: {[c['name'] for c in confounders]}")
    
    # Step 2: Generate regression equations
    logger.info("Step 2/5: Generating regression equations...")
    treatment_eq, outcome_eq = _generate_equations(
        client, config.clinical_question, confounders, config.treatment_effect_prob,
        main_coef_scale=config.main_coefficient_scale,
        interaction_coef_scale=config.interaction_coefficient_scale,
    )
    logger.info(f"Treatment equation has {len(treatment_eq['coefficients'])} terms")
    logger.info(f"Outcome equation has {len(outcome_eq['coefficients'])} terms")
    
    # Step 3: Generate summary statistics
    logger.info("Step 3/6: Generating summary statistics...")
    summary_stats = _generate_summary_statistics(client, config.clinical_question, confounders)
    
    # Step 4: Rescale coefficients first to achieve target logit std
    # (Order matters: rescaling changes the linear predictor, so intercepts must be calibrated afterward)
    logger.info("Step 4/7: Rescaling coefficients for target variability...")
    treatment_eq, outcome_eq = _rescale_for_target_logit_std(
        confounders=confounders,
        summary_stats=summary_stats,
        treatment_eq=treatment_eq,
        outcome_eq=outcome_eq,
        target_logit_std=config.target_logit_std,
    )

    # Step 5: Calibrate intercepts to hit target rates (after rescaling)
    logger.info("Step 5/7: Calibrating intercepts to target rates...")
    treatment_eq, outcome_eq = _calibrate_intercepts(
        confounders=confounders,
        summary_stats=summary_stats,
        treatment_eq=treatment_eq,
        outcome_eq=outcome_eq,
        target_treatment_rate=config.target_treatment_rate,
        target_control_outcome_rate=config.target_control_outcome_rate,
    )

    # Step 6: Generate patient data
    logger.info(f"Step 6/7: Generating {config.dataset_size} patients...")
    patient_data = _generate_all_patients(
        client=client,
        config=config,
        confounders=confounders,
        summary_stats=summary_stats,
        treatment_eq=treatment_eq,
        outcome_eq=outcome_eq,
        num_workers=num_workers,
        show_progress=show_progress,
    )
    
    # Step 6: Assemble dataset
    logger.info("Step 7/7: Assembling dataset...")
    df = pd.DataFrame(patient_data)
    
    # Compile metadata
    metadata = {
        "config": asdict(config),
        "features": confounders,
        "confounders": _filter_features_by_role(confounders, "confounder"),
        "effect_modifiers": _filter_features_by_role(confounders, "effect_modifier"),
        "treatment_equation": treatment_eq,
        "outcome_equation": outcome_eq,
        "summary_statistics": summary_stats,
        "dataset_statistics": {
            "n_patients": len(df),
            "treatment_rate": df["treatment_indicator"].mean(),
            "outcome_rate": df["outcome_indicator"].mean(),
            "mean_treatment_prob": df["true_treatment_prob"].mean(),
            "std_treatment_prob": df["true_treatment_prob"].std(),
            "mean_outcome_prob": df["true_outcome_prob"].mean(),
            "std_outcome_prob": df["true_outcome_prob"].std(),
            "mean_ite_prob": df["true_ite_prob"].mean(),
            "std_ite_prob": df["true_ite_prob"].std(),
        }
    }
    
    # Save outputs
    dataset_path = output_dir / "dataset.parquet"
    metadata_path = output_dir / "metadata.json"
    
    df.to_parquet(dataset_path, index=False)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"Dataset saved to {dataset_path}")
    logger.info(f"Metadata saved to {metadata_path}")
    
    return df, metadata


def _resolve_feature_count_request(
    num_features: Optional[int],
    num_confounders: Optional[int],
    num_effect_modifiers: Optional[int],
) -> Optional[int]:
    """Resolve total feature count from total and role-count requests."""
    if num_features is None and (num_confounders is not None or num_effect_modifiers is not None):
        return (num_confounders or 0) + (num_effect_modifiers or 0)
    return num_features


def _build_feature_count_instruction(
    num_features: Optional[int],
    num_confounders: Optional[int],
    num_effect_modifiers: Optional[int],
) -> str:
    """Build the prompt instruction for total features and role counts."""
    resolved_num_features = _resolve_feature_count_request(
        num_features, num_confounders, num_effect_modifiers
    )
    if resolved_num_features is None:
        return "8-12 total features"

    parts = [f"exactly {resolved_num_features} total features"]
    if num_confounders is not None:
        parts.append(f"exactly {num_confounders} features with the confounder role")
    if num_effect_modifiers is not None:
        parts.append(f"exactly {num_effect_modifiers} features with the effect_modifier role")
    if num_confounders is not None or num_effect_modifiers is not None:
        parts.append("features tagged with both roles count toward both role totals")
    return "; ".join(parts)


def _count_features_by_role(features: List[Dict[str, Any]]) -> Dict[str, int]:
    return {
        "total": len(features),
        "confounder": sum("confounder" in feature.get("roles", []) for feature in features),
        "effect_modifier": sum("effect_modifier" in feature.get("roles", []) for feature in features),
    }


def _feature_counts_match_request(
    features: List[Dict[str, Any]],
    num_features: Optional[int],
    num_confounders: Optional[int],
    num_effect_modifiers: Optional[int],
) -> bool:
    counts = _count_features_by_role(features)
    return (
        (num_features is None or counts["total"] == num_features)
        and (num_confounders is None or counts["confounder"] == num_confounders)
        and (num_effect_modifiers is None or counts["effect_modifier"] == num_effect_modifiers)
    )


def _enforce_feature_count_request(
    features: List[Dict[str, Any]],
    num_features: Optional[int],
    num_confounders: Optional[int],
    num_effect_modifiers: Optional[int],
) -> List[Dict[str, Any]]:
    """Enforce requested total and role counts, preserving LLM order when possible."""
    resolved_num_features = _resolve_feature_count_request(
        num_features, num_confounders, num_effect_modifiers
    )
    if resolved_num_features is None and num_confounders is None and num_effect_modifiers is None:
        return features

    if _feature_counts_match_request(
        features, resolved_num_features, num_confounders, num_effect_modifiers
    ):
        return features

    if resolved_num_features is not None and len(features) >= resolved_num_features:
        if num_confounders is None and num_effect_modifiers is None:
            logger.warning(
                "LLM returned %d features but %d requested; truncating to first %d",
                len(features), resolved_num_features, resolved_num_features,
            )
            return features[:resolved_num_features]

        import itertools
        for indices in itertools.combinations(range(len(features)), resolved_num_features):
            subset = [features[i] for i in indices]
            if _feature_counts_match_request(
                subset, resolved_num_features, num_confounders, num_effect_modifiers
            ):
                logger.warning(
                    "LLM feature counts %s did not match request; selected ordered subset "
                    "with requested counts",
                    _count_features_by_role(features),
                )
                return subset

    raise ValueError(
        "LLM-generated feature counts do not satisfy request. "
        f"Requested total={resolved_num_features}, confounders={num_confounders}, "
        f"effect_modifiers={num_effect_modifiers}; got {_count_features_by_role(features)}. "
        "Retry generation, adjust the counts, or specify --num-features to allow overlap."
    )


def _generate_confounders(
    client: LLMClient,
    clinical_question: str,
    num_features: Optional[int] = None,
    num_confounders: Optional[int] = None,
    num_effect_modifiers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Generate role-tagged explicit features using LLM."""
    prompt = CONFOUNDER_GENERATION_PROMPT.format(
        clinical_question=clinical_question,
        feature_count_instruction=_build_feature_count_instruction(
            num_features, num_confounders, num_effect_modifiers
        ),
    )
    
    response = client.generate_json(
        prompt=prompt,
        system_prompt=CLINICAL_SYSTEM_PROMPT,
        temperature=0.7,
    )
    
    confounders = response.get("features", response.get("confounders", []))

    # Validate structure
    confounders = _validate_feature_roles(confounders)

    return _enforce_feature_count_request(
        confounders, num_features, num_confounders, num_effect_modifiers
    )


def _validate_feature_roles(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate feature definitions and normalize causal roles."""
    valid_roles = {"confounder", "effect_modifier"}
    validated = []
    for feature in features:
        if "name" not in feature or "type" not in feature:
            raise ValueError(f"Invalid feature structure: {feature}")
        if feature["type"] == "categorical" and "categories" not in feature:
            raise ValueError(f"Categorical feature missing categories: {feature}")
        roles = feature.get("roles")
        if not roles:
            raise ValueError(
                f"Feature '{feature.get('name')}' is missing roles; expected one or both of "
                "['confounder', 'effect_modifier']"
            )
        invalid = set(roles) - valid_roles
        if invalid:
            raise ValueError(f"Feature '{feature.get('name')}' has invalid roles: {sorted(invalid)}")
        feature = feature.copy()
        feature["roles"] = list(dict.fromkeys(roles))
        validated.append(feature)
    return validated


def _filter_features_by_role(features: List[Dict[str, Any]], role: str) -> List[Dict[str, Any]]:
    """Return features with a given causal role."""
    return [feature for feature in features if role in feature.get("roles", [])]


def _get_valid_coefficient_names(confounders: List[Dict[str, Any]], role: Optional[str] = None) -> set:
    """
    Build a set of valid coefficient names from the confounder definitions.

    For continuous variables: the variable name itself
    For categorical variables: variablename_category for each non-reference category
    """
    if role is not None:
        confounders = _filter_features_by_role(confounders, role)
    valid_names = set()
    for conf in confounders:
        name = conf["name"]
        if conf["type"] == "continuous":
            valid_names.add(name)
        else:
            # Categorical: add dummies for all non-reference categories
            for cat in conf["categories"][1:]:  # Skip reference category (first)
                valid_names.add(f"{name}_{cat}")
    return valid_names


def _validate_equation_coefficients(
    equation: Dict[str, Any],
    valid_names: set,
    equation_name: str,
) -> Dict[str, Any]:
    """
    Filter equation coefficients to only include valid confounder names.

    Removes any coefficients or interaction terms that reference variables
    not in the valid_names set, logging warnings for removed items.
    """
    validated = equation.copy()

    # Ensure required keys exist with defaults
    validated.setdefault("coefficients", {})
    validated.setdefault("intercept", 0.0)
    validated.setdefault("interactions", [])

    # Filter main coefficients
    if validated["coefficients"]:
        original_coefs = validated["coefficients"]
        filtered_coefs = {}
        for name, value in original_coefs.items():
            if name in valid_names:
                filtered_coefs[name] = value
            else:
                logger.warning(
                    f"{equation_name}: Removing invalid coefficient '{name}' "
                    f"(not in confounder list)"
                )
        validated["coefficients"] = filtered_coefs

    # Filter interactions
    if "interactions" in validated:
        original_interactions = validated["interactions"]
        filtered_interactions = []
        for interaction in original_interactions:
            terms = interaction.get("terms", [])
            # Check if all terms are valid
            if all(term in valid_names for term in terms):
                filtered_interactions.append(interaction)
            else:
                invalid_terms = [t for t in terms if t not in valid_names]
                logger.warning(
                    f"{equation_name}: Removing invalid interaction with terms {terms} "
                    f"(invalid terms: {invalid_terms})"
                )
        validated["interactions"] = filtered_interactions

    return validated


def _generate_equations(
    client: LLMClient,
    clinical_question: str,
    confounders: List[Dict[str, Any]],
    treatment_coefficient: float,
    main_coef_scale: float = 0.3,
    interaction_coef_scale: float = 0.1,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate treatment and outcome regression equations with scaled coefficients."""
    confounder_list = format_confounder_list(confounders)

    prompt = REGRESSION_EQUATION_PROMPT.format(
        clinical_question=clinical_question,
        confounder_list=confounder_list,
        treatment_coefficient=treatment_coefficient,
    )

    response = client.generate_json(
        prompt=prompt,
        system_prompt=CLINICAL_SYSTEM_PROMPT,
        temperature=0.5,  # Lower temperature for more consistent equations
    )

    treatment_eq = response.get("treatment_equation", {})
    outcome_eq = response.get("outcome_equation", {})

    # Validate nuisance coefficients against confounder-role variables only.
    confounder_valid_names = _get_valid_coefficient_names(confounders, role="confounder")
    treatment_eq = _validate_equation_coefficients(treatment_eq, confounder_valid_names, "treatment_equation")
    outcome_eq = _validate_equation_coefficients(outcome_eq, confounder_valid_names, "outcome_equation")

    # Scale LLM-generated coefficients to keep logits in reasonable range
    def scale_coefficients(coefficients: Dict[str, float], scale: float) -> Dict[str, float]:
        return {k: v * scale for k, v in coefficients.items()}

    if "coefficients" in treatment_eq:
        treatment_eq["coefficients"] = scale_coefficients(treatment_eq["coefficients"], main_coef_scale)
    if "coefficients" in outcome_eq:
        outcome_eq["coefficients"] = scale_coefficients(outcome_eq["coefficients"], main_coef_scale)

    # Scale any existing interactions from LLM
    for inter in treatment_eq.get("interactions", []):
        inter["coefficient"] = inter.get("coefficient", 0) * interaction_coef_scale
    for inter in outcome_eq.get("interactions", []):
        inter["coefficient"] = inter.get("coefficient", 0) * interaction_coef_scale

    # Add fixed treatment coefficient to outcome equation
    outcome_eq["treatment_coefficient"] = treatment_coefficient

    # Add treatment-feature interactions to outcome equation for heterogeneous treatment effects.
    # Only effect-modifier-role variables are eligible. Features with both roles
    # appear in nuisance equations and in treatment-effect interactions.
    treatment_interactions = []
    for conf in _filter_features_by_role(confounders, "effect_modifier"):
        name = conf["name"]
        if conf["type"] == "continuous":
            # Interaction: treatment * z_scored_confounder
            coef = np.random.uniform(-1.0, 1.0) * interaction_coef_scale
            treatment_interactions.append({
                "term": name,
                "coefficient": coef
            })
        else:
            # For categorical, add interaction with each non-reference category
            for cat in conf["categories"][1:]:  # Skip reference category
                dummy_name = f"{name}_{cat}"
                coef = np.random.uniform(-1.0, 1.0) * interaction_coef_scale
                treatment_interactions.append({
                    "term": dummy_name,
                    "coefficient": coef
                })
    outcome_eq["treatment_interactions"] = treatment_interactions
    logger.info(f"Added {len(treatment_interactions)} treatment-effect-modifier interactions to outcome equation")

    # Add pairwise confounder-role interactions to treatment equation.
    # Get all coefficient names (continuous names + categorical dummies)
    coef_names = []
    for conf in _filter_features_by_role(confounders, "confounder"):
        name = conf["name"]
        if conf["type"] == "continuous":
            coef_names.append(name)
        else:
            for cat in conf["categories"][1:]:  # Skip reference category
                coef_names.append(f"{name}_{cat}")

    # Create all pairwise interactions
    existing_interactions = treatment_eq.get("interactions", [])
    existing_pairs = {tuple(sorted(inter.get("terms", []))) for inter in existing_interactions}

    new_interactions = []
    for i, term1 in enumerate(coef_names):
        for term2 in coef_names[i+1:]:
            pair = tuple(sorted([term1, term2]))
            if pair not in existing_pairs:
                coef = np.random.uniform(-1.0, 1.0) * interaction_coef_scale
                new_interactions.append({
                    "terms": [term1, term2],
                    "coefficient": coef
                })

    treatment_eq["interactions"] = existing_interactions + new_interactions
    logger.info(f"Added {len(new_interactions)} pairwise confounder interactions to treatment equation")

    return treatment_eq, outcome_eq


def _generate_summary_statistics(
    client: LLMClient,
    clinical_question: str,
    confounders: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Generate summary statistics for confounders."""
    confounder_list = format_confounder_list(confounders)
    
    prompt = SUMMARY_STATISTICS_PROMPT.format(
        clinical_question=clinical_question,
        confounder_list=confounder_list,
    )
    
    response = client.generate_json(
        prompt=prompt,
        system_prompt=CLINICAL_SYSTEM_PROMPT,
        temperature=0.5,
    )
    
    return response.get("summary_statistics", {})


def _calibrate_intercepts(
    confounders: List[Dict[str, Any]],
    summary_stats: Dict[str, Any],
    treatment_eq: Dict[str, Any],
    outcome_eq: Dict[str, Any],
    target_treatment_rate: float,
    target_control_outcome_rate: float,
    n_samples: int = 10000,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Calibrate equation intercepts to achieve target marginal rates.
    
    Uses Monte Carlo sampling and binary search to find intercepts that yield
    the desired treatment rate and control group outcome rate.
    
    Args:
        confounders: List of confounder definitions
        summary_stats: Summary statistics for sampling
        treatment_eq: Treatment assignment equation (will be modified)
        outcome_eq: Outcome equation (will be modified)
        target_treatment_rate: Desired proportion receiving treatment=1
        target_control_outcome_rate: Desired outcome rate when treatment=0
        n_samples: Number of Monte Carlo samples for calibration
        
    Returns:
        Tuple of (calibrated_treatment_eq, calibrated_outcome_eq)
    """
    from scipy.optimize import brentq
    
    # Sample characteristics for calibration
    sampled_chars = [
        _sample_patient_characteristics(confounders, summary_stats)
        for _ in range(n_samples)
    ]
    
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def compute_linear_predictor(characteristics, equation, confounders, summary_stats, treatment=None):
        """Compute linear predictor WITHOUT intercept."""
        # Temporarily set intercept to 0
        original_intercept = equation.get("intercept", 0.0)
        equation["intercept"] = 0.0
        logit = _compute_logit(characteristics, confounders, summary_stats, equation, treatment=treatment)
        equation["intercept"] = original_intercept
        return logit
    
    # Compute linear predictors for all samples (without intercept)
    treatment_lps = np.array([
        compute_linear_predictor(chars, treatment_eq, confounders, summary_stats)
        for chars in sampled_chars
    ])
    
    outcome_lps = np.array([
        compute_linear_predictor(chars, outcome_eq, confounders, summary_stats, treatment=0)
        for chars in sampled_chars
    ])
    
    # Calibrate treatment intercept
    def treatment_rate_error(intercept):
        probs = sigmoid(intercept + treatment_lps)
        return probs.mean() - target_treatment_rate
    
    try:
        # Search for intercept in reasonable range
        calibrated_treatment_intercept = brentq(treatment_rate_error, -10, 10)
    except ValueError:
        # If target is outside achievable range, use boundary
        logger.warning(f"Could not achieve target treatment rate {target_treatment_rate}, using closest achievable")
        if treatment_rate_error(-10) > 0:
            calibrated_treatment_intercept = -10
        else:
            calibrated_treatment_intercept = 10
    
    # Calibrate outcome intercept (for control group)
    def outcome_rate_error(intercept):
        probs = sigmoid(intercept + outcome_lps)
        return probs.mean() - target_control_outcome_rate
    
    try:
        calibrated_outcome_intercept = brentq(outcome_rate_error, -10, 10)
    except ValueError:
        logger.warning(f"Could not achieve target control outcome rate {target_control_outcome_rate}, using closest achievable")
        if outcome_rate_error(-10) > 0:
            calibrated_outcome_intercept = -10
        else:
            calibrated_outcome_intercept = 10
    
    # Update equations with calibrated intercepts
    original_treatment_intercept = treatment_eq.get("intercept", 0.0)
    original_outcome_intercept = outcome_eq.get("intercept", 0.0)
    
    treatment_eq["intercept"] = calibrated_treatment_intercept
    treatment_eq["original_intercept"] = original_treatment_intercept
    
    outcome_eq["intercept"] = calibrated_outcome_intercept
    outcome_eq["original_intercept"] = original_outcome_intercept
    
    logger.info(f"Calibrated treatment intercept: {original_treatment_intercept:.3f} -> {calibrated_treatment_intercept:.3f}")
    logger.info(f"Calibrated outcome intercept: {original_outcome_intercept:.3f} -> {calibrated_outcome_intercept:.3f}")
    
    # Verify achieved rates
    achieved_treatment_rate = sigmoid(calibrated_treatment_intercept + treatment_lps).mean()
    achieved_outcome_rate = sigmoid(calibrated_outcome_intercept + outcome_lps).mean()
    logger.info(f"Achieved treatment rate: {achieved_treatment_rate:.3f} (target: {target_treatment_rate:.3f})")
    logger.info(f"Achieved control outcome rate: {achieved_outcome_rate:.3f} (target: {target_control_outcome_rate:.3f})")
    
    return treatment_eq, outcome_eq


def _rescale_for_target_logit_std(
    confounders: List[Dict[str, Any]],
    summary_stats: Dict[str, Any],
    treatment_eq: Dict[str, Any],
    outcome_eq: Dict[str, Any],
    target_logit_std: float = 2.0,
    n_samples: int = 10000,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Rescale all coefficients (except intercept) to achieve target logit std.

    This ensures the final logits have a reasonable range for meaningful probabilities.
    """
    # Sample characteristics
    sampled_chars = [
        _sample_patient_characteristics(confounders, summary_stats)
        for _ in range(n_samples)
    ]

    def compute_logit_without_intercept(characteristics, equation, treatment=None):
        """Compute logit with intercept temporarily set to 0."""
        original_intercept = equation.get("intercept", 0.0)
        equation["intercept"] = 0.0
        logit = _compute_logit(characteristics, confounders, summary_stats, equation, treatment=treatment)
        equation["intercept"] = original_intercept
        return logit

    # Compute current logit stds
    treatment_logits = np.array([
        compute_logit_without_intercept(chars, treatment_eq)
        for chars in sampled_chars
    ])
    outcome_logits_0 = np.array([
        compute_logit_without_intercept(chars, outcome_eq, treatment=0)
        for chars in sampled_chars
    ])
    outcome_logits_1 = np.array([
        compute_logit_without_intercept(chars, outcome_eq, treatment=1)
        for chars in sampled_chars
    ])

    treatment_std = np.std(treatment_logits)
    outcome_std_0 = np.std(outcome_logits_0)
    outcome_std_1 = np.std(outcome_logits_1)
    outcome_std = max(outcome_std_0, outcome_std_1)  # Use the larger one

    logger.info(f"Current logit std - treatment: {treatment_std:.2f}, outcome: {outcome_std:.2f}")

    def scale_equation_coefficients(equation: Dict, scale: float) -> Dict:
        """Scale all coefficients in an equation (except intercept and treatment_coefficient)."""
        eq = equation.copy()

        # Scale main coefficients
        if "coefficients" in eq:
            eq["coefficients"] = {k: v * scale for k, v in eq["coefficients"].items()}

        # Scale interactions
        if "interactions" in eq:
            eq["interactions"] = [
                {**inter, "coefficient": inter.get("coefficient", 0) * scale}
                for inter in eq["interactions"]
            ]

        # Scale treatment interactions (for outcome equation)
        if "treatment_interactions" in eq:
            eq["treatment_interactions"] = [
                {**inter, "coefficient": inter.get("coefficient", 0) * scale}
                for inter in eq["treatment_interactions"]
            ]

        return eq

    # Compute scale factors
    if treatment_std > 0:
        treatment_scale = target_logit_std / treatment_std
        treatment_eq = scale_equation_coefficients(treatment_eq, treatment_scale)
        logger.info(f"Scaled treatment coefficients by {treatment_scale:.3f}")

    if outcome_std > 0:
        outcome_scale = target_logit_std / outcome_std
        outcome_eq = scale_equation_coefficients(outcome_eq, outcome_scale)
        logger.info(f"Scaled outcome coefficients by {outcome_scale:.3f}")

    # Verify new stds
    treatment_logits_new = np.array([
        compute_logit_without_intercept(chars, treatment_eq)
        for chars in sampled_chars
    ])
    outcome_logits_new = np.array([
        compute_logit_without_intercept(chars, outcome_eq, treatment=0)
        for chars in sampled_chars
    ])

    logger.info(f"New logit std - treatment: {np.std(treatment_logits_new):.2f}, outcome: {np.std(outcome_logits_new):.2f}")

    return treatment_eq, outcome_eq


def _sample_patient_characteristics(
    confounders: List[Dict[str, Any]],
    summary_stats: Dict[str, Any],
) -> Dict[str, Any]:
    """Sample patient characteristics from summary statistics."""
    characteristics = {}
    
    for conf in confounders:
        name = conf["name"]
        stats = summary_stats.get(name, {})
        
        if conf["type"] == "continuous":
            # Sample from normal distribution
            mean = stats.get("mean", 0.0)
            std = stats.get("std", 1.0)
            value = np.random.normal(mean, std)
            characteristics[name] = value
        else:
            # Sample from categorical distribution
            categories = conf["categories"]
            proportions = stats.get("proportions", {})
            
            # Default to uniform if no proportions
            if not proportions:
                proportions = {cat: 1.0 / len(categories) for cat in categories}
            
            # Ensure proportions sum to 1
            probs = [proportions.get(cat, 0.0) for cat in categories]
            prob_sum = sum(probs)
            if prob_sum > 0:
                probs = [p / prob_sum for p in probs]
            else:
                probs = [1.0 / len(categories)] * len(categories)
            
            chosen = np.random.choice(categories, p=probs)
            characteristics[name] = chosen

    return characteristics


def _enforce_positivity(
    treatment_prob: float,
    min_rate: float = 0.1,
    max_rate: float = 0.9,
) -> float:
    """
    Enforce positivity bounds on treatment probability.

    Clips the treatment probability to ensure both treatment and control
    groups have adequate representation in each confounder stratum.
    This is essential for valid causal inference - without positivity,
    counterfactual outcomes cannot be estimated for subpopulations that
    never receive treatment.

    Args:
        treatment_prob: Original treatment probability from logistic model
        min_rate: Minimum allowed P(T=1|X) - ensures some patients get treated
        max_rate: Maximum allowed P(T=1|X) - ensures some patients remain control

    Returns:
        Clipped treatment probability within [min_rate, max_rate]
    """
    return np.clip(treatment_prob, min_rate, max_rate)


# ============================================================================
# Two-stage generation helpers
# ============================================================================

def _parse_event_timeline(timeline_text: str) -> List[Dict[str, str]]:
    """
    Parse event timeline text into structured events.

    The primary prompt asks for lines starting with <event_type> tags, but
    some hosted models return JSON arrays with event_type/event_text fields.
    Returns list of dicts with 'event_type' and 'event_text' keys.
    """
    json_events = _parse_event_timeline_json(timeline_text)
    if json_events:
        return json_events

    pattern = re.compile(r'<(\w+)>\s*(.*?)(?=\n\s*<\w+>|\Z)', re.DOTALL)
    events = []
    for event_type, event_text in pattern.findall(timeline_text):
        text = event_text.strip()
        if text:
            events.append({
                "event_type": event_type.strip(),
                "event_text": text,
            })
    return events


def _strip_json_code_fence(text: str) -> str:
    """Strip a Markdown JSON code fence if the model wrapped its response."""
    stripped = text.strip()
    fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    return stripped


def _load_timeline_json(text: str) -> Optional[Any]:
    """Load a JSON timeline from a possibly fenced or prefixed model response."""
    stripped = _strip_json_code_fence(text)
    decoder = json.JSONDecoder()

    candidates = [stripped]
    for delimiter in ("[", "{"):
        idx = stripped.find(delimiter)
        if idx > 0:
            candidates.append(stripped[idx:])

    for candidate in candidates:
        try:
            value, _ = decoder.raw_decode(candidate)
            return value
        except json.JSONDecodeError:
            continue
    return None


def _event_type_from_json(value: Any) -> str:
    event_type = str(value or "").strip()
    event_type = event_type.strip("<>").strip()
    return event_type


def _event_text_from_json(event: Dict[str, Any]) -> str:
    for key in ("event_text", "text", "description", "note", "summary", "findings", "impression"):
        value = event.get(key)
        if value:
            return str(value).strip()

    payload = {
        key: value
        for key, value in event.items()
        if key not in {"event_type", "type"}
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _parse_event_timeline_json(timeline_text: str) -> List[Dict[str, str]]:
    """Parse Gemini-style JSON event arrays into the internal event format."""
    loaded = _load_timeline_json(timeline_text)
    if loaded is None:
        return []

    if isinstance(loaded, dict):
        for key in ("events", "timeline", "event_timeline"):
            if isinstance(loaded.get(key), list):
                loaded = loaded[key]
                break

    if not isinstance(loaded, list):
        return []

    events = []
    for item in loaded:
        if not isinstance(item, dict):
            continue
        event_type = _event_type_from_json(item.get("event_type", item.get("type", "")))
        event_text = _event_text_from_json(item)
        if event_type and event_text:
            events.append({
                "event_type": event_type,
                "event_text": event_text,
            })
    return events


def _create_masked_timeline(events: List[Dict[str, str]], target_idx: int) -> str:
    """
    Create masked version of timeline with one event highlighted.

    The target event is wrapped with BEGIN/END tags so the LLM knows
    which event to expand into a full clinical note.
    """
    lines = []
    for i, event in enumerate(events):
        line = f"<{event['event_type']}> {event['event_text']}"
        if i == target_idx:
            line = (
                f"<BEGIN EVENT CORRESPONDING TO SYNTHETIC NOTE> "
                f"{line} "
                f"<END EVENT CORRESPONDING TO SYNTHETIC NOTE>"
            )
        lines.append(line)
    return "\n".join(lines)


def _event_type_to_note_type(event_type: str) -> str:
    """Map event type tag to human-readable note type for the expansion prompt."""
    mapping = {
        "clinical_note": "clinical progress note",
        "imaging_report": "imaging/radiology report",
        "pathology_report": "pathology report",
        "ngs_report": "next-generation sequencing report",
    }
    return mapping.get(event_type, "clinical document")


# Drug perturbation utilities (adapted from matchminer)
_COMPILED_DRUG_PATTERNS = None


def _get_drug_patterns():
    """Compile regex patterns for drug name replacement (cached)."""
    global _COMPILED_DRUG_PATTERNS
    if _COMPILED_DRUG_PATTERNS is not None:
        return _COMPILED_DRUG_PATTERNS

    patterns = []
    for generic, alternatives in DRUG_PERTURBATION_MAP.items():
        pattern = re.compile(r'\b' + re.escape(generic) + r'\b', re.IGNORECASE)
        patterns.append((pattern, alternatives))
    _COMPILED_DRUG_PATTERNS = patterns
    return patterns


def _apply_drug_perturbation(text: str, rng=None) -> str:
    """
    Replace some generic drug names with brand names or abbreviations.

    For each drug found in the text, randomly decides whether to replace it
    and picks a random alternative from the drug map.
    """
    patterns = _get_drug_patterns()
    for pattern, alternatives in patterns:
        if pattern.search(text):
            replacement = (rng or random).choice(alternatives)
            text = pattern.sub(replacement, text)
    return text


def _compute_logit(
    characteristics: Dict[str, Any],
    confounders: List[Dict[str, Any]],
    summary_stats: Dict[str, Any],
    equation: Dict[str, Any],
    treatment: Optional[int] = None,
) -> float:
    """
    Compute logit from characteristics using regression equation.
    
    For continuous variables: coefficient * (value - mean) / std (z-scored)
    For categorical variables: coefficient of the dummy for selected category
    """
    logit = equation.get("intercept", 0.0)
    coefficients = equation.get("coefficients", {})
    
    # Build z-scored continuous values map
    z_values = {}
    for conf in confounders:
        name = conf["name"]
        if conf["type"] == "continuous":
            stats = summary_stats.get(name, {})
            mean = stats.get("mean", 0.0)
            std = stats.get("std", 1.0)
            if std == 0:
                std = 1.0
            z_values[name] = (characteristics[name] - mean) / std

    def term_value(term: str) -> Optional[float]:
        """Return the numeric value for a coefficient or interaction term."""
        if term in z_values:
            return z_values[term]
        for conf in confounders:
            if conf["type"] != "categorical":
                continue
            name = conf["name"]
            for cat in conf["categories"][1:]:
                if term == f"{name}_{cat}":
                    return 1.0 if characteristics.get(name) == cat else 0.0
        return None
    
    # Apply coefficients
    for coef_name, coef_value in coefficients.items():
        # Check if this is a base continuous variable
        if coef_name in z_values:
            logit += coef_value * z_values[coef_name]
            continue
        
        # Check if this is a categorical dummy (format: varname_category)
        matched = False
        for conf in confounders:
            if conf["type"] == "categorical":
                name = conf["name"]
                for cat in conf["categories"][1:]:  # Skip reference category
                    dummy_name = f"{name}_{cat}"
                    if coef_name == dummy_name:
                        # Add coefficient if this category is selected
                        if characteristics.get(name) == cat:
                            logit += coef_value
                        matched = True
                        break
            if matched:
                break
    
    # Apply interactions
    interactions = equation.get("interactions", [])
    for interaction in interactions:
        terms = interaction.get("terms", [])
        coef = interaction.get("coefficient", 0.0)
        
        # Compute product of continuous z-scores and categorical dummy indicators
        product = 1.0
        for term in terms:
            value = term_value(term)
            if value is None:
                product = 0.0
                break
            product *= value
        
        logit += coef * product
    
    # Add treatment effect if provided
    if treatment is not None:
        treatment_coef = equation.get("treatment_coefficient", 0.0)
        logit += treatment_coef * treatment
        
        # Apply treatment-confounder interactions (for heterogeneous treatment effects)
        treatment_interactions = equation.get("treatment_interactions", [])
        for interaction in treatment_interactions:
            term = interaction.get("term", "")
            coef = interaction.get("coefficient", 0.0)
            value = term_value(term)
            if value is not None:
                logit += coef * treatment * value
    
    return logit


def _enabled_structured_event_types(config: SyntheticDataConfig) -> set:
    structured_config = getattr(config, "structured_data", None)
    enabled_types = set()
    if structured_config is not None and structured_config.enabled:
        if structured_config.include_encounters:
            enabled_types.add("encounter")
        if structured_config.include_hospitalizations:
            enabled_types.add("hospitalization")
        if structured_config.include_labs:
            enabled_types.add("lab_result")
        if structured_config.include_pros:
            enabled_types.add("pro_assessment")
    return enabled_types


def _build_patient_scaffold_record(
    patient_idx: int,
    config: SyntheticDataConfig,
    confounders: List[Dict[str, Any]],
    summary_stats: Dict[str, Any],
    treatment_eq: Dict[str, Any],
    outcome_eq: Dict[str, Any],
) -> Dict[str, Any]:
    """Sample one patient's covariates, treatment and outcome from the DGP.

    This is the single sampler shared by API generation and saved-DGP
    generation; it draws from the global NumPy RNG in a fixed order.
    """
    characteristics = _sample_patient_characteristics(confounders, summary_stats)

    treatment_logit = _compute_logit(
        characteristics, confounders, summary_stats, treatment_eq
    )
    treatment_prob = 1.0 / (1.0 + np.exp(-treatment_logit))

    if getattr(config, "enforce_positivity", False):
        treatment_prob = _enforce_positivity(
            treatment_prob,
            min_rate=getattr(config, "min_treatment_rate_per_stratum", 0.1),
            max_rate=getattr(config, "max_treatment_rate_per_stratum", 0.9),
        )

    treatment = int(np.random.random() < treatment_prob)

    outcome_logit = _compute_logit(
        characteristics, confounders, summary_stats, outcome_eq, treatment=treatment
    )
    outcome_type = getattr(config, "outcome_type", "binary")
    if outcome_type == "continuous":
        noise_std = getattr(config, "outcome_noise_std", 1.0)
        outcome = outcome_logit + np.random.normal(0, noise_std)
    else:
        outcome_prob_for_sample = 1.0 / (1.0 + np.exp(-outcome_logit))
        outcome = int(np.random.random() < outcome_prob_for_sample)

    outcome_logit_0 = _compute_logit(
        characteristics, confounders, summary_stats, outcome_eq, treatment=0
    )
    outcome_logit_1 = _compute_logit(
        characteristics, confounders, summary_stats, outcome_eq, treatment=1
    )
    outcome_prob_0 = 1.0 / (1.0 + np.exp(-outcome_logit_0))
    outcome_prob_1 = 1.0 / (1.0 + np.exp(-outcome_logit_1))
    outcome_prob = 1.0 / (1.0 + np.exp(-outcome_logit))

    patient_prompt = format_patient_characteristics(characteristics, confounders)
    record = {
        "patient_id": patient_idx,
        "patient_prompt": patient_prompt,
        "treatment_indicator": treatment,
        "outcome_indicator": outcome,
        "true_treatment_prob": treatment_prob,
        "true_outcome_prob": outcome_prob,
        "true_y0_prob": outcome_prob_0,
        "true_y1_prob": outcome_prob_1,
        "true_ite_prob": outcome_prob_1 - outcome_prob_0,
    }
    for name, value in characteristics.items():
        record[f"true_{name}"] = value
    return record


def _generate_single_patient(
    patient_idx: int,
    client: LLMClient,
    config: SyntheticDataConfig,
    confounders: List[Dict[str, Any]],
    summary_stats: Dict[str, Any],
    treatment_eq: Dict[str, Any],
    outcome_eq: Dict[str, Any],
) -> Dict[str, Any]:
    """Sample one patient's causal scaffold, then generate its clinical text.

    Text generation is two-stage: an event timeline, then expansion of each
    narrative event into a note (structured events are templated to text).
    """
    record = _build_patient_scaffold_record(
        patient_idx, config, confounders, summary_stats, treatment_eq, outcome_eq
    )

    # Reserve tokens for prompt
    history_max_tokens = max(1000, config.llm.max_tokens - 2000)

    # ---- Stage 1: Generate event timeline ----
    # Use structured-data-aware prompt if enabled
    structured_config = getattr(config, 'structured_data', None)
    event_timeline_prompt_template = build_event_timeline_prompt(structured_config)

    timeline_prompt = event_timeline_prompt_template.format(
        patient_characteristics=record["patient_prompt"],
        clinical_question=config.clinical_question,
        min_events=config.min_events_per_patient,
        max_events=config.max_events_per_patient,
    )

    try:
        timeline_text = client.generate(
            prompt=timeline_prompt,
            system_prompt=CLINICAL_SYSTEM_PROMPT,
            temperature=0.7,
            max_tokens=history_max_tokens,
        )
        if timeline_text is None:
            logger.warning(f"Patient {patient_idx}: LLM returned None for event timeline")
            timeline_text = ""
    except Exception as e:
        logger.error(f"Patient {patient_idx}: Failed to generate event timeline: {e}")
        timeline_text = ""

    events = _parse_event_timeline(timeline_text)

    if not events:
        logger.warning(f"Patient {patient_idx}: No events parsed from timeline")

    # ---- Stage 2: Expand note-worthy events and convert structured events ----
    note_types = set(getattr(config, 'note_types_to_expand', [
        "clinical_note", "imaging_report", "pathology_report", "ngs_report"
    ]))
    note_separator = getattr(config, 'note_separator', "\n\n---\n\n")
    drug_prob = getattr(config, 'drug_perturbation_prob', 0.3)
    enabled_structured_types = _enabled_structured_event_types(config)

    # Collect text blocks with their timeline index for interleaving
    text_blocks = []  # list of (timeline_index, text_content)

    for event_idx, event in enumerate(events):
        etype = event["event_type"]

        if etype in note_types:
            # Narrative note - expand via LLM
            masked_timeline = _create_masked_timeline(events, event_idx)
            note_type = _event_type_to_note_type(etype)

            expansion_prompt = NOTE_EXPANSION_PROMPT.format(
                note_type=note_type,
                clinical_question=config.clinical_question,
                masked_event_timeline=masked_timeline,
            )

            try:
                note_text = client.generate(
                    prompt=expansion_prompt,
                    system_prompt=CLINICAL_SYSTEM_PROMPT,
                    temperature=0.5,
                    max_tokens=history_max_tokens,
                )
                if note_text is None:
                    logger.warning(f"Patient {patient_idx}, event {event_idx}: LLM returned None for note expansion")
                    continue
            except Exception as e:
                logger.error(f"Patient {patient_idx}, event {event_idx}: Failed to expand note: {e}")
                continue

            # Apply drug perturbation
            if drug_prob > 0 and random.random() < drug_prob:
                note_text = _apply_drug_perturbation(note_text)

            text_blocks.append((event_idx, note_text))

        elif etype in enabled_structured_types:
            # Structured event - convert via template
            structured_text = convert_structured_event_to_text(etype, event["event_text"])
            if structured_text:
                text_blocks.append((event_idx, structured_text))

    # Sort by timeline index and join
    text_blocks.sort(key=lambda x: x[0])
    clinical_history = note_separator.join(text for _, text in text_blocks)

    # Validate that text doesn't contain literal category codes
    underscore_patterns = re.findall(r'\b[a-z]+(?:_[a-z0-9]+){2,}\b', clinical_history.lower())
    if underscore_patterns:
        logger.warning(f"Patient {patient_idx}: Clinical text contains underscore patterns (likely category codes): {underscore_patterns[:3]}")

    # Same column layout as saved-DGP generation.
    return dict(
        record,
        event_timeline=timeline_text,
        clinical_text=clinical_history,
        num_notes=len(text_blocks),
    )


def _generate_all_patients(
    client: LLMClient,
    config: SyntheticDataConfig,
    confounders: List[Dict[str, Any]],
    summary_stats: Dict[str, Any],
    treatment_eq: Dict[str, Any],
    outcome_eq: Dict[str, Any],
    num_workers: int = 4,
    show_progress: bool = True,
) -> List[Dict[str, Any]]:
    """Generate data for all patients with parallel LLM calls and checkpoint support."""
    output_dir = Path(config.output_dir)
    checkpoint_path = output_dir / "checkpoint.json"
    
    # Load existing checkpoint if present
    completed_patients = {}
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                completed_patients = {p['patient_id']: p for p in checkpoint_data.get('patients', [])}
            logger.info(f"Resuming from checkpoint: {len(completed_patients)} patients already completed")
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Could not load checkpoint, starting fresh: {e}")
            completed_patients = {}
    
    # Determine which patients still need to be generated
    remaining_indices = [i for i in range(config.dataset_size) if i not in completed_patients]
    logger.info(f"Need to generate {len(remaining_indices)} patients (checkpoint has {len(completed_patients)})")
    
    if not remaining_indices:
        # All patients already generated
        patient_data = list(completed_patients.values())
        patient_data.sort(key=lambda x: x["patient_id"])
        return patient_data
    
    # Collect all patient data (starting with checkpoint data)
    patient_data = list(completed_patients.values())
    checkpoint_interval = 10  # Save checkpoint every N patients
    patients_since_checkpoint = 0
    
    def save_checkpoint():
        """Save current progress to checkpoint file."""
        checkpoint_content = {
            'patients': patient_data,
            'total_expected': config.dataset_size,
        }
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_content, f, ensure_ascii=False)
        logger.debug(f"Checkpoint saved: {len(patient_data)} patients")
    
    if num_workers <= 1:
        # Sequential generation
        iterator = remaining_indices
        if show_progress:
            iterator = tqdm(iterator, desc="Generating patients", initial=len(completed_patients), total=config.dataset_size)
        
        for i in iterator:
            data = _generate_single_patient(
                i, client, config, confounders, summary_stats, treatment_eq, outcome_eq
            )
            patient_data.append(data)
            patients_since_checkpoint += 1
            
            if patients_since_checkpoint >= checkpoint_interval:
                save_checkpoint()
                patients_since_checkpoint = 0
    else:
        # Parallel generation
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    _generate_single_patient,
                    i, client, config, confounders, summary_stats, treatment_eq, outcome_eq
                ): i
                for i in remaining_indices
            }
            
            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(remaining_indices), desc="Generating patients", 
                               initial=len(completed_patients))
            
            for future in iterator:
                data = future.result()
                patient_data.append(data)
                patients_since_checkpoint += 1
                
                if patients_since_checkpoint >= checkpoint_interval:
                    save_checkpoint()
                    patients_since_checkpoint = 0
    
    # Final checkpoint save
    save_checkpoint()
    
    # Sort by patient_id to ensure reproducibility
    patient_data.sort(key=lambda x: x["patient_id"])
    
    # # Clean up checkpoint file on successful completion
    # if checkpoint_path.exists():
    #     checkpoint_path.unlink()
    #     logger.info("Generation complete, checkpoint file removed")
    
    return patient_data


def _load_dgp_metadata(metadata_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load feature definitions, equations, and summary stats from a seed metadata file."""
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    features = _validate_feature_roles(metadata.get("features", []))
    treatment_eq = json.loads(json.dumps(metadata.get("treatment_equation", {})))
    outcome_eq = json.loads(json.dumps(metadata.get("outcome_equation", {})))
    summary_stats = json.loads(json.dumps(metadata.get("summary_statistics", {})))
    source_config = metadata.get("config", {})

    if not features:
        raise ValueError(f"DGP metadata file has no features: {metadata_path}")
    if not treatment_eq or not outcome_eq:
        raise ValueError(f"DGP metadata file is missing treatment/outcome equations: {metadata_path}")
    if not summary_stats:
        raise ValueError(f"DGP metadata file is missing summary_statistics: {metadata_path}")

    return features, treatment_eq, outcome_eq, summary_stats, source_config
