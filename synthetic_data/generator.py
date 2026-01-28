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
    PATIENT_HISTORY_PROMPT,
    format_confounder_list,
    format_patient_characteristics,
    validate_clinical_text,
)

# Type hints for vLLM imports (imported at runtime)
if False:  # TYPE_CHECKING
    from .vllm_batch_client import VLLMBatchClient, VLLMConfig


logger = logging.getLogger(__name__)


def generate_synthetic_dataset(
    config: SyntheticDataConfig,
    num_workers: int = 4,
    show_progress: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate a synthetic clinical dataset with known causal structure.
    
    This pipeline:
    1. Uses LLM to generate realistic confounders based on clinical question
    2. Uses LLM to generate treatment and outcome regression equations
    3. Uses LLM to generate summary statistics for confounders
    4. For each patient: samples characteristics, computes logits, generates clinical history
    5. Saves dataset and metadata
    
    Args:
        config: Configuration for generation
        num_workers: Number of parallel workers for patient history generation
        show_progress: Whether to show progress bar
        
    Returns:
        Tuple of (dataset DataFrame, metadata dictionary)
    """
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

    # Step 1: Generate confounders
    logger.info("Step 1/5: Generating confounders...")
    confounders = _generate_confounders(client, config.clinical_question, num_confounders=config.num_confounders)
    logger.info(f"Generated {len(confounders)} confounders: {[c['name'] for c in confounders]}")
    
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
        "confounders": confounders,
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


def _generate_confounders(client: LLMClient, clinical_question: str, num_confounders: Optional[int] = None) -> List[Dict[str, Any]]:
    """Generate confounders using LLM."""
    # Determine the instruction for number of confounders
    if num_confounders is not None:
        num_confounders_instruction = f"exactly {num_confounders} confounders"
    else:
        num_confounders_instruction = "8-12 confounders"
    
    prompt = CONFOUNDER_GENERATION_PROMPT.format(
        clinical_question=clinical_question,
        num_confounders_instruction=num_confounders_instruction
    )
    
    response = client.generate_json(
        prompt=prompt,
        system_prompt=CLINICAL_SYSTEM_PROMPT,
        temperature=0.7,
    )
    
    confounders = response.get("confounders", [])
    
    # Validate structure
    for conf in confounders:
        if "name" not in conf or "type" not in conf:
            raise ValueError(f"Invalid confounder structure: {conf}")
        if conf["type"] == "categorical" and "categories" not in conf:
            raise ValueError(f"Categorical confounder missing categories: {conf}")
    
    return confounders


def _get_valid_coefficient_names(confounders: List[Dict[str, Any]]) -> set:
    """
    Build a set of valid coefficient names from the confounder definitions.

    For continuous variables: the variable name itself
    For categorical variables: variablename_category for each non-reference category
    """
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

    # Filter main coefficients
    if "coefficients" in validated:
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

    # Validate coefficients - remove any that don't match the confounder list
    valid_names = _get_valid_coefficient_names(confounders)
    treatment_eq = _validate_equation_coefficients(treatment_eq, valid_names, "treatment_equation")
    outcome_eq = _validate_equation_coefficients(outcome_eq, valid_names, "outcome_equation")

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

    # Add treatment-confounder interactions to outcome equation (for heterogeneous treatment effects)
    # Each continuous confounder and each categorical dummy gets a treatment interaction
    treatment_interactions = []
    for conf in confounders:
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
    logger.info(f"Added {len(treatment_interactions)} treatment-confounder interactions to outcome equation")

    # Add pairwise confounder-confounder interactions to treatment equation
    # Get all coefficient names (continuous names + categorical dummies)
    coef_names = []
    for conf in confounders:
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


# ============================================================================
# vLLM-based helper functions for batch generation
# ============================================================================

def _generate_confounders_vllm(client: 'VLLMBatchClient', clinical_question: str, num_confounders: Optional[int] = None) -> List[Dict[str, Any]]:
    """Generate confounders using vLLM batch client."""
    if num_confounders is not None:
        num_confounders_instruction = f"exactly {num_confounders} confounders"
    else:
        num_confounders_instruction = "8-12 confounders"
    
    prompt = CONFOUNDER_GENERATION_PROMPT.format(
        clinical_question=clinical_question,
        num_confounders_instruction=num_confounders_instruction
    )
    
    response = client.generate_json(
        prompt=prompt,
        system_prompt=CLINICAL_SYSTEM_PROMPT,
        temperature=0.7,
    )
    
    confounders = response.get("confounders", [])
    
    for conf in confounders:
        if "name" not in conf or "type" not in conf:
            raise ValueError(f"Invalid confounder structure: {conf}")
        if conf["type"] == "categorical" and "categories" not in conf:
            raise ValueError(f"Categorical confounder missing categories: {conf}")
    
    return confounders


def _generate_equations_vllm(
    client: 'VLLMBatchClient',
    clinical_question: str,
    confounders: List[Dict[str, Any]],
    treatment_coefficient: float,
    main_coef_scale: float = 0.3,
    interaction_coef_scale: float = 0.1,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate treatment and outcome regression equations using vLLM."""
    confounder_list = format_confounder_list(confounders)
    
    prompt = REGRESSION_EQUATION_PROMPT.format(
        clinical_question=clinical_question,
        confounder_list=confounder_list,
        treatment_coefficient=treatment_coefficient,
    )
    
    response = client.generate_json(
        prompt=prompt,
        system_prompt=CLINICAL_SYSTEM_PROMPT,
        temperature=0.5,
    )
    
    treatment_eq = response.get("treatment_equation", {})
    outcome_eq = response.get("outcome_equation", {})

    # Validate coefficients - remove any that don't match the confounder list
    valid_names = _get_valid_coefficient_names(confounders)
    treatment_eq = _validate_equation_coefficients(treatment_eq, valid_names, "treatment_equation")
    outcome_eq = _validate_equation_coefficients(outcome_eq, valid_names, "outcome_equation")

    # Scale coefficients
    if "coefficients" in treatment_eq:
        treatment_eq["coefficients"] = {
            k: v * main_coef_scale for k, v in treatment_eq["coefficients"].items()
        }
    if "interactions" in treatment_eq:
        for interaction in treatment_eq["interactions"]:
            interaction["coefficient"] *= interaction_coef_scale

    if "coefficients" in outcome_eq:
        outcome_eq["coefficients"] = {
            k: v * main_coef_scale for k, v in outcome_eq["coefficients"].items()
        }
    if "interactions" in outcome_eq:
        for interaction in outcome_eq["interactions"]:
            interaction["coefficient"] *= interaction_coef_scale

    # Add fixed treatment coefficient to outcome equation (CRITICAL for ITE)
    outcome_eq["treatment_coefficient"] = treatment_coefficient
    
    # Add treatment-confounder interactions to outcome equation (for heterogeneous treatment effects)
    # Each continuous confounder and each categorical dummy gets a treatment interaction
    treatment_interactions = []
    for conf in confounders:
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
    logger.info(f"Added treatment_coefficient={treatment_coefficient} and {len(treatment_interactions)} treatment-confounder interactions to outcome equation")
    
    return treatment_eq, outcome_eq


def _generate_summary_statistics_vllm(
    client: 'VLLMBatchClient',
    clinical_question: str,
    confounders: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Generate summary statistics using vLLM."""
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
        
        # Compute product of z-scored values
        product = 1.0
        for term in terms:
            if term in z_values:
                product *= z_values[term]
            elif term in characteristics:
                # For categorical in interaction, use indicator
                product *= 1.0
            else:
                product = 0.0
                break
        
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
            
            # Check if it's a continuous variable (in z_values)
            if term in z_values:
                logit += coef * treatment * z_values[term]
            else:
                # Check if it's a categorical dummy
                for conf in confounders:
                    if conf["type"] == "categorical":
                        name = conf["name"]
                        for cat in conf["categories"][1:]:  # Skip reference category
                            dummy_name = f"{name}_{cat}"
                            if term == dummy_name:
                                # Add interaction if this category is selected
                                if characteristics.get(name) == cat:
                                    logit += coef * treatment
                                break
    
    return logit


def _generate_single_patient(
    patient_idx: int,
    client: LLMClient,
    config: SyntheticDataConfig,
    confounders: List[Dict[str, Any]],
    summary_stats: Dict[str, Any],
    treatment_eq: Dict[str, Any],
    outcome_eq: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate data for a single patient.
    
    For binary outcomes: Samples from Bernoulli distribution using sigmoid(logit).
    For continuous outcomes: Uses the logit directly, optionally with Gaussian noise.
    """
    # Sample characteristics
    characteristics = _sample_patient_characteristics(confounders, summary_stats)
    
    # Compute treatment logit and sample treatment (always binary)
    treatment_logit = _compute_logit(
        characteristics, confounders, summary_stats, treatment_eq
    )
    treatment_prob = 1.0 / (1.0 + np.exp(-treatment_logit))

    # Apply positivity enforcement if enabled
    if getattr(config, 'enforce_positivity', False):
        treatment_prob = _enforce_positivity(
            treatment_prob,
            min_rate=getattr(config, 'min_treatment_rate_per_stratum', 0.1),
            max_rate=getattr(config, 'max_treatment_rate_per_stratum', 0.9),
        )

    treatment = int(np.random.random() < treatment_prob)
    
    # Compute outcome logit
    outcome_logit = _compute_logit(
        characteristics, confounders, summary_stats, outcome_eq, treatment=treatment
    )
    
    # Generate outcome based on outcome_type
    outcome_type = getattr(config, 'outcome_type', 'binary')
    if outcome_type == "continuous":
        # Continuous outcome: use logit directly with optional noise
        noise_std = getattr(config, 'outcome_noise_std', 1.0)
        outcome = outcome_logit + np.random.normal(0, noise_std)
    else:
        # Binary outcome: sample from Bernoulli
        outcome_prob = 1.0 / (1.0 + np.exp(-outcome_logit))
        outcome = int(np.random.random() < outcome_prob)
    
    # Compute potential outcome probabilities for causal inference
    outcome_logit_0 = _compute_logit(
        characteristics, confounders, summary_stats, outcome_eq, treatment=0
    )
    outcome_logit_1 = _compute_logit(
        characteristics, confounders, summary_stats, outcome_eq, treatment=1
    )
    outcome_prob_0 = 1.0 / (1.0 + np.exp(-outcome_logit_0))
    outcome_prob_1 = 1.0 / (1.0 + np.exp(-outcome_logit_1))
    true_ite_prob = outcome_prob_1 - outcome_prob_0

    # Compute probability for factual outcome
    outcome_prob = 1.0 / (1.0 + np.exp(-outcome_logit))

    # Format patient characteristics as prompt
    patient_prompt = format_patient_characteristics(characteristics, confounders)

    # Generate clinical history
    history_prompt = PATIENT_HISTORY_PROMPT.format(
        patient_characteristics=patient_prompt,
        clinical_question=config.clinical_question,
    )

    # Reserve tokens for prompt (system prompt + patient history prompt ~1500-2000 tokens)
    history_max_tokens = max(1000, config.llm.max_tokens - 2000)

    try:
        clinical_history = client.generate(
            prompt=history_prompt,
            system_prompt=CLINICAL_SYSTEM_PROMPT,
            temperature=0.4,  # Lower temperature for more faithful confounder representation
            max_tokens=history_max_tokens,
        )
        # Handle None response from LLM
        if clinical_history is None:
            logger.warning(f"Patient {patient_idx}: LLM returned None for clinical_history")
            clinical_history = ""
    except Exception as e:
        logger.error(f"Patient {patient_idx}: Failed to generate clinical_history: {e}")
        clinical_history = ""

    # Validate that text doesn't contain literal category codes
    underscore_patterns = re.findall(r'\b[a-z]+(?:_[a-z0-9]+){2,}\b', clinical_history.lower())
    if underscore_patterns:
        logger.warning(f"Patient {patient_idx}: Clinical text contains underscore patterns (likely category codes): {underscore_patterns[:3]}")

    return {
        "patient_id": patient_idx,
        "patient_prompt": patient_prompt,
        "clinical_text": clinical_history,
        "treatment_indicator": treatment,
        "outcome_indicator": outcome,
        "true_treatment_prob": treatment_prob,
        "true_outcome_prob": outcome_prob,
        "true_y0_prob": outcome_prob_0,
        "true_y1_prob": outcome_prob_1,
        "true_ite_prob": true_ite_prob,
    }


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


def _generate_clinical_texts_worker(
    worker_id: int,
    device_ids: List[int],
    vllm_config_dict: Dict[str, Any],
    patient_indices: List[int],
    history_prompts: List[str],
    reasoning_marker: Optional[str],
    result_queue,
):
    """
    Worker process for multi-GPU clinical text generation.

    Each worker:
    1. Sets CUDA_VISIBLE_DEVICES to its assigned GPUs
    2. Initializes its own VLLMBatchClient
    3. Generates clinical texts for its patient subset
    4. Returns results via multiprocessing Queue

    Args:
        worker_id: Unique identifier for this worker
        device_ids: GPU device IDs for this worker (e.g., [0, 1])
        vllm_config_dict: VLLMConfig parameters as dict (for pickling)
        patient_indices: List of patient indices this worker handles
        history_prompts: List of prompts for clinical history generation
        reasoning_marker: Marker to strip from model outputs
        result_queue: Multiprocessing Queue to return results
    """
    import os

    # Set CUDA_VISIBLE_DEVICES for this worker BEFORE any CUDA initialization
    device_str = ",".join(str(d) for d in device_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = device_str

    # Now import vLLM (which initializes CUDA)
    from .vllm_batch_client import VLLMBatchClient, VLLMConfig

    worker_logger = logging.getLogger(f"{__name__}.worker_{worker_id}")
    worker_logger.info(f"Worker {worker_id}: Starting on GPUs {device_ids} (CUDA_VISIBLE_DEVICES={device_str})")
    worker_logger.info(f"Worker {worker_id}: Processing {len(patient_indices)} patients")

    try:
        # Create VLLMConfig for this worker (device_ids not needed since we set env var)
        worker_config = VLLMConfig(
            model_name=vllm_config_dict["model_name"],
            tensor_parallel_size=vllm_config_dict["tensor_parallel_size"],
            gpu_memory_utilization=vllm_config_dict.get("gpu_memory_utilization", 0.90),
            max_model_len=vllm_config_dict.get("max_model_len"),
            temperature=vllm_config_dict["temperature"],
            max_tokens=vllm_config_dict["max_tokens"],
            download_dir=vllm_config_dict.get("download_dir", "./"),
            reasoning_marker=reasoning_marker,
            device_ids=None,  # Already set via env var
        )

        # Initialize vLLM client
        client = VLLMBatchClient(worker_config)

        # Generate clinical texts in batch
        worker_logger.info(f"Worker {worker_id}: Generating {len(history_prompts)} clinical texts...")
        clinical_texts = client.generate_batch(
            prompts=history_prompts,
            system_prompt=CLINICAL_SYSTEM_PROMPT,
            temperature=0.4,
            max_tokens=worker_config.max_tokens,
        )

        # Strip reasoning prefix from each text
        cleaned_texts = []
        for text in clinical_texts:
            cleaned = VLLMBatchClient.strip_reasoning_prefix(
                text if text else "",
                reasoning_marker
            )
            cleaned_texts.append(cleaned)

        worker_logger.info(f"Worker {worker_id}: Completed {len(cleaned_texts)} clinical texts")

        # Return results via queue
        result_queue.put({
            "worker_id": worker_id,
            "patient_indices": patient_indices,
            "clinical_texts": cleaned_texts,
            "error": None,
        })

    except Exception as e:
        worker_logger.error(f"Worker {worker_id}: Failed with error: {e}")
        result_queue.put({
            "worker_id": worker_id,
            "patient_indices": patient_indices,
            "clinical_texts": None,
            "error": str(e),
        })


def _run_parallel_vllm_workers(
    gpu_device_ids: List[int],
    tensor_parallel_size: int,
    vllm_config_dict: Dict[str, Any],
    patient_indices: List[int],
    history_prompts: List[str],
    reasoning_marker: Optional[str],
) -> List[str]:
    """
    Orchestrate multi-GPU parallel clinical text generation.

    Splits patient records across multiple worker processes, each running
    its own vLLM instance on a subset of GPUs.

    Args:
        gpu_device_ids: All GPU device IDs to use (e.g., [0, 1, 2, 3])
        tensor_parallel_size: GPUs per vLLM instance
        vllm_config_dict: VLLMConfig parameters as dict
        patient_indices: All patient indices to process
        history_prompts: All prompts for clinical history generation
        reasoning_marker: Marker to strip from model outputs

    Returns:
        List of clinical texts in patient_id order
    """
    import multiprocessing as mp

    # Calculate number of workers
    num_workers = len(gpu_device_ids) // tensor_parallel_size
    logger.info(f"Parallel vLLM: {num_workers} workers, {tensor_parallel_size} GPUs each")

    # Split GPUs into groups
    gpu_groups = []
    for i in range(num_workers):
        start_idx = i * tensor_parallel_size
        end_idx = start_idx + tensor_parallel_size
        gpu_groups.append(gpu_device_ids[start_idx:end_idx])

    logger.info(f"GPU groups: {gpu_groups}")

    # Split patients across workers (as evenly as possible)
    patients_per_worker = len(patient_indices) // num_workers
    remainder = len(patient_indices) % num_workers

    patient_splits = []
    prompt_splits = []
    start = 0
    for i in range(num_workers):
        # Workers with lower index get one extra patient if there's remainder
        n = patients_per_worker + (1 if i < remainder else 0)
        end = start + n
        patient_splits.append(patient_indices[start:end])
        prompt_splits.append(history_prompts[start:end])
        start = end

    logger.info(f"Patient distribution: {[len(s) for s in patient_splits]}")

    # Create result queue
    # Use spawn context to avoid CUDA context issues
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    # Spawn worker processes
    processes = []
    for worker_id in range(num_workers):
        p = ctx.Process(
            target=_generate_clinical_texts_worker,
            args=(
                worker_id,
                gpu_groups[worker_id],
                vllm_config_dict,
                patient_splits[worker_id],
                prompt_splits[worker_id],
                reasoning_marker,
                result_queue,
            ),
        )
        processes.append(p)
        p.start()
        logger.info(f"Started worker {worker_id} (PID: {p.pid})")

    # Collect results from all workers
    results_by_worker = {}
    errors = []

    for _ in range(num_workers):
        result = result_queue.get()  # Blocks until result available
        worker_id = result["worker_id"]
        if result["error"]:
            errors.append(f"Worker {worker_id}: {result['error']}")
        else:
            results_by_worker[worker_id] = result
        logger.info(f"Received results from worker {worker_id}")

    # Wait for all processes to finish
    for p in processes:
        p.join()
        logger.info(f"Worker process {p.pid} completed")

    # Check for errors
    if errors:
        raise RuntimeError(f"Multi-GPU generation failed:\n" + "\n".join(errors))

    # Merge results in patient_id order
    # Build mapping from patient_id to clinical_text
    patient_to_text = {}
    for worker_id, result in results_by_worker.items():
        for idx, patient_id in enumerate(result["patient_indices"]):
            patient_to_text[patient_id] = result["clinical_texts"][idx]

    # Return texts in original patient_indices order
    clinical_texts = [patient_to_text[pid] for pid in patient_indices]

    logger.info(f"Merged {len(clinical_texts)} clinical texts from {num_workers} workers")
    return clinical_texts


def generate_synthetic_dataset_batch(
    config: SyntheticDataConfig,
    vllm_config: 'VLLMConfig',
    show_progress: bool = True,
    gpu_device_ids: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate a synthetic clinical dataset using direct vLLM batch inference.

    This is much faster than the HTTP API approach because:
    1. No HTTP overhead
    2. vLLM handles batching optimally
    3. All clinical histories generated in one batch

    Multi-GPU Parallelization:
    When gpu_device_ids is provided with more GPUs than tensor_parallel_size,
    the function spawns multiple parallel vLLM workers:
    - Initialization (confounders, equations, stats) runs on first GPU group only
    - Patient clinical text generation is split across all workers
    - Results are merged into a single output dataset

    Example:
        gpu_device_ids=[0,1,2,3], tensor_parallel_size=2 -> 2 parallel workers
        Worker 0: GPUs 0,1 -> patients 0..N/2
        Worker 1: GPUs 2,3 -> patients N/2..N

    Args:
        config: Configuration for generation
        vllm_config: vLLM configuration (model, tensor_parallel_size, etc.)
        show_progress: Whether to show progress bar
        gpu_device_ids: Optional list of GPU device IDs for multi-GPU parallelization.
            If None, uses single instance (original behavior).
            If provided, len(gpu_device_ids) must be divisible by tensor_parallel_size.

    Returns:
        Tuple of (dataset DataFrame, metadata dictionary)
    """
    from .vllm_batch_client import VLLMBatchClient, VLLMConfig
    
    config.validate()
    
    # Set random seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting batch synthetic data generation for: {config.clinical_question[:80]}...")

    # Log positivity enforcement settings
    if getattr(config, 'enforce_positivity', False):
        logger.info(f"Positivity enforcement ENABLED: treatment rate per stratum bounded to [{config.min_treatment_rate_per_stratum:.2f}, {config.max_treatment_rate_per_stratum:.2f}]")
    else:
        logger.info("Positivity enforcement disabled (realistic observational data)")

    # Determine if multi-GPU mode is enabled
    use_multi_gpu = (
        gpu_device_ids is not None
        and len(gpu_device_ids) > vllm_config.tensor_parallel_size
    )
    num_workers = 1
    if use_multi_gpu:
        num_workers = len(gpu_device_ids) // vllm_config.tensor_parallel_size
        logger.info(f"Multi-GPU mode: {len(gpu_device_ids)} GPUs -> {num_workers} parallel workers")

    # Initialize vLLM client for initialization steps (confounders, equations, stats)
    # In multi-GPU mode, use only the first GPU group for these fast operations
    logger.info("Initializing vLLM for offline batch inference...")
    if use_multi_gpu:
        first_gpu_group = gpu_device_ids[:vllm_config.tensor_parallel_size]
        logger.info(f"Using first GPU group {first_gpu_group} for initialization steps")
        init_config = VLLMConfig(
            model_name=vllm_config.model_name,
            tensor_parallel_size=vllm_config.tensor_parallel_size,
            gpu_memory_utilization=vllm_config.gpu_memory_utilization,
            max_model_len=vllm_config.max_model_len,
            temperature=vllm_config.temperature,
            max_tokens=vllm_config.max_tokens,
            download_dir=vllm_config.download_dir,
            reasoning_marker=vllm_config.reasoning_marker,
            device_ids=first_gpu_group,
        )
        vllm_client = VLLMBatchClient(init_config)
    else:
        vllm_client = VLLMBatchClient(vllm_config)
    
    # Step 1: Generate confounders using vLLM
    logger.info("Step 1/6: Generating confounders...")
    confounders = _generate_confounders_vllm(vllm_client, config.clinical_question, num_confounders=config.num_confounders)
    logger.info(f"Generated {len(confounders)} confounders: {[c['name'] for c in confounders]}")
    
    # Step 2: Generate regression equations
    logger.info("Step 2/6: Generating regression equations...")
    treatment_eq, outcome_eq = _generate_equations_vllm(
        vllm_client, config.clinical_question, confounders, config.treatment_effect_prob,
        main_coef_scale=config.main_coefficient_scale,
        interaction_coef_scale=config.interaction_coefficient_scale,
    )
    logger.info(f"Treatment equation has {len(treatment_eq['coefficients'])} terms")
    logger.info(f"Outcome equation has {len(outcome_eq['coefficients'])} terms")
    
    # Step 3: Generate summary statistics
    logger.info("Step 3/6: Generating summary statistics...")
    summary_stats = _generate_summary_statistics_vllm(vllm_client, config.clinical_question, confounders)
    
    # Step 4: Rescale coefficients first, then calibrate intercepts
    # (Order matters: rescaling changes the linear predictor, so intercepts must be calibrated afterward)
    logger.info("Step 4/6: Rescaling coefficients and calibrating intercepts...")
    treatment_eq, outcome_eq = _rescale_for_target_logit_std(
        confounders=confounders,
        summary_stats=summary_stats,
        treatment_eq=treatment_eq,
        outcome_eq=outcome_eq,
        target_logit_std=config.target_logit_std,
    )
    treatment_eq, outcome_eq = _calibrate_intercepts(
        confounders=confounders,
        summary_stats=summary_stats,
        treatment_eq=treatment_eq,
        outcome_eq=outcome_eq,
        target_treatment_rate=config.target_treatment_rate,
        target_control_outcome_rate=config.target_control_outcome_rate,
    )

    # Step 5: Pre-generate all patient data (without clinical text)
    logger.info(f"Step 5/6: Generating {config.dataset_size} patient records...")
    patient_records = []
    history_prompts = []
    
    iterator = range(config.dataset_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Building patient records")
    
    for patient_idx in iterator:
        # Sample characteristics
        characteristics = _sample_patient_characteristics(confounders, summary_stats)
        
        # Compute treatment logit and sample treatment (always binary)
        treatment_logit = _compute_logit(
            characteristics, confounders, summary_stats, treatment_eq
        )
        treatment_prob = 1.0 / (1.0 + np.exp(-treatment_logit))

        # Apply positivity enforcement if enabled
        if getattr(config, 'enforce_positivity', False):
            treatment_prob = _enforce_positivity(
                treatment_prob,
                min_rate=getattr(config, 'min_treatment_rate_per_stratum', 0.1),
                max_rate=getattr(config, 'max_treatment_rate_per_stratum', 0.9),
            )

        treatment = int(np.random.random() < treatment_prob)

        # Compute outcome logit
        outcome_logit = _compute_logit(
            characteristics, confounders, summary_stats, outcome_eq, treatment=treatment
        )
        
        # Generate outcome based on outcome_type
        outcome_type = getattr(config, 'outcome_type', 'binary')
        if outcome_type == "continuous":
            # Continuous outcome: use logit directly with optional noise
            noise_std = getattr(config, 'outcome_noise_std', 1.0)
            outcome = outcome_logit + np.random.normal(0, noise_std)
        else:
            # Binary outcome: sample from Bernoulli
            outcome_prob = 1.0 / (1.0 + np.exp(-outcome_logit))
            outcome = int(np.random.random() < outcome_prob)
        
        # Compute potential outcome probabilities
        outcome_logit_0 = _compute_logit(
            characteristics, confounders, summary_stats, outcome_eq, treatment=0
        )
        outcome_logit_1 = _compute_logit(
            characteristics, confounders, summary_stats, outcome_eq, treatment=1
        )
        outcome_prob_0 = 1.0 / (1.0 + np.exp(-outcome_logit_0))
        outcome_prob_1 = 1.0 / (1.0 + np.exp(-outcome_logit_1))
        true_ite_prob = outcome_prob_1 - outcome_prob_0

        # Compute probability for factual outcome
        outcome_prob = 1.0 / (1.0 + np.exp(-outcome_logit))

        # Format patient characteristics as prompt
        patient_prompt = format_patient_characteristics(characteristics, confounders)

        # Build clinical history prompt
        history_prompt = PATIENT_HISTORY_PROMPT.format(
            patient_characteristics=patient_prompt,
            clinical_question=config.clinical_question,
        )
        history_prompts.append(history_prompt)

        patient_records.append({
            "patient_id": patient_idx,
            "patient_prompt": patient_prompt,
            "clinical_text": None,  # Will be filled by batch generation
            "treatment_indicator": treatment,
            "outcome_indicator": outcome,
            "true_treatment_prob": treatment_prob,
            "true_outcome_prob": outcome_prob,
            "true_y0_prob": outcome_prob_0,
            "true_y1_prob": outcome_prob_1,
            "true_ite_prob": true_ite_prob,
        })
    
    # Step 6: Batch generate all clinical histories using vLLM
    logger.info(f"Step 6/6: Batch generating {len(history_prompts)} clinical histories with vLLM...")

    if use_multi_gpu:
        # Multi-GPU mode: unload initialization model and spawn parallel workers
        logger.info("Unloading initialization model to free GPU memory for parallel workers...")
        del vllm_client
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        # Build vllm config dict for workers (dataclasses aren't picklable with complex defaults)
        vllm_config_dict = {
            "model_name": vllm_config.model_name,
            "tensor_parallel_size": vllm_config.tensor_parallel_size,
            "gpu_memory_utilization": vllm_config.gpu_memory_utilization,
            "max_model_len": vllm_config.max_model_len,
            "temperature": vllm_config.temperature,
            "max_tokens": vllm_config.max_tokens,
            "download_dir": vllm_config.download_dir,
        }

        # Run parallel workers
        patient_indices = list(range(len(patient_records)))
        clinical_texts = _run_parallel_vllm_workers(
            gpu_device_ids=gpu_device_ids,
            tensor_parallel_size=vllm_config.tensor_parallel_size,
            vllm_config_dict=vllm_config_dict,
            patient_indices=patient_indices,
            history_prompts=history_prompts,
            reasoning_marker=vllm_config.reasoning_marker,
        )
    else:
        # Single-GPU mode: generate all clinical histories in one batch
        clinical_texts = vllm_client.generate_batch(
            prompts=history_prompts,
            system_prompt=CLINICAL_SYSTEM_PROMPT,
            temperature=0.4,  # Lower temperature for more faithful confounder representation
            max_tokens=vllm_config.max_tokens,
        )

        # Strip reasoning prefix in single-GPU mode
        clinical_texts = [
            VLLMBatchClient.strip_reasoning_prefix(
                text if text else "",
                vllm_config.reasoning_marker
            )
            for text in clinical_texts
        ]

    # Merge clinical texts with patient records
    # Also validate that texts don't contain literal category codes
    validation_issues = 0
    for i, cleaned_text in enumerate(clinical_texts):
        patient_records[i]["clinical_text"] = cleaned_text

        # Check for underscore-connected phrases (likely category codes that weren't naturalized)
        underscore_patterns = re.findall(r'\b[a-z]+(?:_[a-z0-9]+){2,}\b', cleaned_text.lower())
        if underscore_patterns:
            validation_issues += 1
            if validation_issues <= 3:  # Only log first few examples
                logger.warning(f"Patient {i}: Clinical text contains underscore patterns: {underscore_patterns[:3]}")

    if validation_issues > 0:
        logger.warning(f"VALIDATION: {validation_issues}/{len(patient_records)} clinical texts contain underscore-connected phrases (likely literal category codes)")
    else:
        logger.info("VALIDATION: All clinical texts passed - no underscore-connected category codes detected")

    logger.info("Batch generation complete!")
    
    # Assemble dataset
    logger.info("Assembling dataset...")
    df = pd.DataFrame(patient_records)
    
    # Compile metadata
    metadata = {
        "config": asdict(config),
        "confounders": confounders,
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
            "clinical_text_stats": {
                "non_empty_count": (df["clinical_text"].str.len() > 0).sum(),
                "mean_length": df["clinical_text"].str.len().mean(),
            }
        }
    }
    
    # Save outputs
    dataset_path = output_dir / "dataset.parquet"
    metadata_path = output_dir / "metadata.json"
    
    df.to_parquet(dataset_path, index=False)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str, ensure_ascii=False)
    
    logger.info(f"Dataset saved to {dataset_path}")
    logger.info(f"Metadata saved to {metadata_path}")
    
    return df, metadata
