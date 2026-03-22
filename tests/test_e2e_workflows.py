# tests/test_e2e_workflows.py
"""End-to-end workflow tests for OCI inference pipelines.

Tests the main applied user workflows:
1. DragonNet + Frozen LLM Pooler + CV
2. RLearner + Frozen LLM Pooler + CV
3. RLearner + Dual Extractors + CV
4. Causal Forest (neural features from R-learner training) + CV
5. TF-IDF Forest + CV
6. Confounders-Only Causal Forest + CV
"""

import gc
import pytest
import pandas as pd
import numpy as np
import torch
from pathlib import Path

from oci.config import (
    AppliedInferenceConfig,
    ModelArchitectureConfig,
    TrainingConfig,
    ExplicitConfounderExtractionConfig,
    ExplicitConfounderSpec,
    CausalForestConfig,
    TfidfForestConfig,
    ConfounderForestConfig,
)
from oci.inference.applied import run_applied_inference


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

SAMPLE_TEXTS = [
    "Patient is a 62-year-old male with stage IIIA non-small cell lung cancer. ECOG performance status 1. Prior chemotherapy with carboplatin.",
    "55-year-old female diagnosed with metastatic breast cancer. HER2-positive. Received trastuzumab. BMI 24.3.",
    "71-year-old male with advanced prostate cancer. Gleason 8. PSA 45.2. Started on enzalutamide.",
    "48-year-old female with stage IV colorectal cancer. KRAS mutant. Received FOLFOX plus bevacizumab.",
    "66-year-old male, history of COPD. Diagnosed with squamous cell carcinoma of lung. ECOG 2.",
    "59-year-old female with triple-negative breast cancer. Received neoadjuvant pembrolizumab plus chemotherapy.",
    "73-year-old male with hepatocellular carcinoma. Child-Pugh A. AFP 312. Started sorafenib.",
    "44-year-old female with ovarian cancer stage IIIC. CA-125 elevated at 890. Debulking surgery performed.",
    "68-year-old male with renal cell carcinoma. Clear cell histology. Started nivolumab plus ipilimumab.",
    "52-year-old female with melanoma. BRAF V600E mutation. Received dabrafenib plus trametinib.",
]


def _create_test_dataset(n: int = 40, seed: int = 42) -> pd.DataFrame:
    """Create a small synthetic dataset for testing."""
    rng = np.random.RandomState(seed)
    texts = [SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)] for i in range(n)]
    treatment = rng.randint(0, 2, size=n)
    outcome = rng.randint(0, 2, size=n)
    true_ite = rng.uniform(-0.2, 0.2, size=n)

    return pd.DataFrame({
        'clinical_text': texts,
        'treatment_indicator': treatment,
        'outcome_indicator': outcome,
        'true_ite_prob': true_ite,
    })


def _add_mock_confounders(df: pd.DataFrame) -> tuple:
    """Add mock confounder columns to DataFrame.

    Returns:
        (specs, confounder_columns) tuple
    """
    rng = np.random.RandomState(123)
    n = len(df)

    specs = [
        ExplicitConfounderSpec(
            name="ecog_status",
            type="categorical",
            categories=["0", "1", "2"],
            description="ECOG performance status"
        ),
        ExplicitConfounderSpec(
            name="age",
            type="continuous",
            description="Patient age at diagnosis"
        ),
    ]

    # Add categorical confounder
    df['explicit_conf_ecog_status'] = rng.choice(["0", "1", "2"], size=n)
    df['explicit_conf_ecog_status_missing'] = False

    # Add continuous confounder
    df['explicit_conf_age'] = rng.uniform(40, 80, size=n)
    df['explicit_conf_age_missing'] = False

    columns = [
        'explicit_conf_ecog_status', 'explicit_conf_ecog_status_missing',
        'explicit_conf_age', 'explicit_conf_age_missing',
    ]

    return specs, columns


def _make_config(
    model_type: str,
    dataset_path: str,
    **overrides
) -> AppliedInferenceConfig:
    """Create a minimal config for testing."""
    arch_kwargs = {
        'model_type': model_type,
        'feature_extractor_type': 'frozen_llm_pooler',
        'flp_model_name': 'Qwen/Qwen3-0.6B-Base',
        'flp_max_length': 64,
        'flp_freeze_llm': True,
        'flp_gated_attention_dim': 32,
        'flp_projection_dim': 32,
        'flp_dropout': 0.0,
        'flp_gradient_checkpointing': False,
        'causal_head_representation_dim': 32,
        'causal_head_hidden_outcome_dim': 16,
        'causal_head_dropout': 0.0,
    }

    # Apply architecture overrides
    for k in list(overrides.keys()):
        if k.startswith('flp_') or k in ('rlearner_dual_extractors',):
            arch_kwargs[k] = overrides.pop(k)

    # Causal forest config with tiny params
    arch_kwargs['causal_forest'] = CausalForestConfig(
        n_estimators=8,
        min_samples_leaf=2,
        honest=False,
        inference=True,
        **overrides.pop('causal_forest_overrides', {})
    )
    arch_kwargs['tfidf_forest'] = TfidfForestConfig(
        max_features=50,
        min_df=1,
        n_estimators=8,
        min_samples_leaf=2,
        honest=False,
        inference=True,
    )
    arch_kwargs['confounder_forest'] = ConfounderForestConfig(
        n_estimators=8,
        min_samples_leaf=2,
        honest=False,
        inference=True,
    )

    training_kwargs = {
        'epochs': 2,
        'batch_size': 8,
        'learning_rate': 1e-3,
        'gradient_clip_norm': 1.0,
    }
    for k in list(overrides.keys()):
        if k in ('epochs', 'batch_size', 'learning_rate'):
            training_kwargs[k] = overrides.pop(k)

    config = AppliedInferenceConfig(
        dataset_path=dataset_path,
        cv_folds=overrides.pop('cv_folds', 2),
        architecture=ModelArchitectureConfig(**arch_kwargs),
        training=TrainingConfig(**training_kwargs),
        **overrides
    )
    return config


def _verify_neural_predictions(results_df: pd.DataFrame, n_expected: int, n_folds: int = 2):
    """Verify predictions from neural model (DragonNet/RLearner)."""
    assert len(results_df) == n_expected, f"Expected {n_expected} rows, got {len(results_df)}"

    required_cols = ['pred_y0_prob', 'pred_y1_prob', 'pred_ite_prob', 'pred_propensity_prob', 'cv_fold']
    for col in required_cols:
        assert col in results_df.columns, f"Missing column: {col}"

    # Check predictions are in valid ranges
    for col in ['pred_y0_prob', 'pred_y1_prob', 'pred_propensity_prob']:
        vals = results_df[col].values
        assert np.all(np.isfinite(vals)), f"{col} has non-finite values"
        assert np.all((vals >= 0) & (vals <= 1)), f"{col} has values outside [0,1]"

    # Check all folds present
    assert set(results_df['cv_fold'].unique()) == set(range(1, n_folds + 1))


def _verify_forest_predictions(results_df: pd.DataFrame, n_expected: int, n_folds: int = 2):
    """Verify predictions from forest model."""
    assert len(results_df) == n_expected, f"Expected {n_expected} rows, got {len(results_df)}"

    required_cols = ['pred_ite_prob', 'pred_y0_prob', 'pred_y1_prob', 'pred_propensity_prob', 'cv_fold']
    for col in required_cols:
        assert col in results_df.columns, f"Missing column: {col}"

    # ITE can be any value (not bounded to [0,1])
    assert np.all(np.isfinite(results_df['pred_ite_prob'].values))

    # CI columns should be present
    assert 'pred_ite_lower' in results_df.columns
    assert 'pred_ite_upper' in results_df.columns

    # Check all folds present
    assert set(results_df['cv_fold'].unique()) == set(range(1, n_folds + 1))


def _cleanup():
    """Free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def test_dataset(tmp_path):
    """Create test dataset and save to parquet."""
    df = _create_test_dataset(n=40)
    path = tmp_path / "dataset.parquet"
    df.to_parquet(path, index=False)
    return df, str(path)


@pytest.fixture
def device():
    return torch.device("cpu")


class TestDragonNet:
    """Test DragonNet + Frozen LLM Pooler + CV."""

    def test_dragonnet_cv(self, test_dataset, tmp_path, device):
        df, dataset_path = test_dataset
        output_path = tmp_path / "applied_inference" / "predictions.parquet"

        config = _make_config("dragonnet", dataset_path)

        run_applied_inference(
            dataset=df,
            config=config,
            output_path=output_path,
            device=device,
        )

        results_df = pd.read_parquet(output_path)
        _verify_neural_predictions(results_df, n_expected=len(df))
        _cleanup()


class TestRLearner:
    """Test RLearner + Frozen LLM Pooler + CV."""

    def test_rlearner_cv(self, test_dataset, tmp_path, device):
        df, dataset_path = test_dataset
        output_path = tmp_path / "applied_inference" / "predictions.parquet"

        config = _make_config("rlearner", dataset_path)

        run_applied_inference(
            dataset=df,
            config=config,
            output_path=output_path,
            device=device,
        )

        results_df = pd.read_parquet(output_path)
        _verify_neural_predictions(results_df, n_expected=len(df))
        _cleanup()

    def test_rlearner_dual_extractors_cv(self, test_dataset, tmp_path, device):
        df, dataset_path = test_dataset
        output_path = tmp_path / "applied_inference" / "predictions.parquet"

        config = _make_config(
            "rlearner", dataset_path,
            rlearner_dual_extractors=True
        )

        run_applied_inference(
            dataset=df,
            config=config,
            output_path=output_path,
            device=device,
        )

        results_df = pd.read_parquet(output_path)
        _verify_neural_predictions(results_df, n_expected=len(df))
        _cleanup()


class TestCausalForest:
    """Test Causal Forest (neural features) + CV."""

    def test_causal_forest_rlearner_cv(self, test_dataset, tmp_path, device):
        df, dataset_path = test_dataset
        output_path = tmp_path / "applied_inference" / "predictions.parquet"

        config = _make_config(
            "causal_forest", dataset_path,
            causal_forest_overrides={
                'use_rlearner_representation': True,
                'gamma_rlearner': 1.0,
            }
        )

        run_applied_inference(
            dataset=df,
            config=config,
            output_path=output_path,
            device=device,
        )

        results_df = pd.read_parquet(output_path)
        _verify_forest_predictions(results_df, n_expected=len(df))
        _cleanup()


class TestTfidfForest:
    """Test TF-IDF Forest + CV (non-neural baseline)."""

    def test_tfidf_forest_cv(self, test_dataset, tmp_path, device):
        df, dataset_path = test_dataset
        output_path = tmp_path / "applied_inference" / "predictions.parquet"

        config = _make_config("tfidf_forest", dataset_path)

        run_applied_inference(
            dataset=df,
            config=config,
            output_path=output_path,
            device=device,
        )

        results_df = pd.read_parquet(output_path)
        _verify_forest_predictions(results_df, n_expected=len(df))


class TestConfounderForest:
    """Test Confounders-Only Causal Forest + CV."""

    def test_confounder_forest_cv(self, test_dataset, tmp_path, device):
        df, dataset_path = test_dataset
        df = df.copy()
        specs, conf_columns = _add_mock_confounders(df)

        output_path = tmp_path / "applied_inference" / "predictions.parquet"

        config = _make_config("confounder_forest", dataset_path)
        config.explicit_confounders = ExplicitConfounderExtractionConfig(
            enabled=True,
            confounders=specs,
        )

        # Call the confounder forest pipeline directly (bypassing extraction)
        from oci.inference.applied_confounder_forest import run_applied_inference_confounder_forest
        run_applied_inference_confounder_forest(
            dataset=df,
            config=config,
            output_path=output_path,
            explicit_confounder_columns=conf_columns,
        )

        results_df = pd.read_parquet(output_path)
        _verify_forest_predictions(results_df, n_expected=len(df))
