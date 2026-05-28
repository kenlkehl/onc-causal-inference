import json

import numpy as np
import pandas as pd
import pytest

from oci.config import (
    AgenticFeatureSearchConfig,
    AppliedInferenceConfig,
    ExplicitFeatureExtractionConfig,
    ExplicitFeatureForestConfig,
    ExplicitFeatureSpec,
    ExperimentConfig,
    ModelArchitectureConfig,
)
from oci.extraction.cache import _compute_config_hash
from oci.inference.agentic_explicit_feature_forest import (
    AgenticFeatureProposal,
    SplitEvaluation,
    apply_proposals,
    compare_candidate_to_baseline,
    run_agentic_explicit_feature_forest,
    validate_agentic_proposals,
)


def _base_specs():
    return [
        ExplicitFeatureSpec(
            name="age",
            type="continuous",
            description="Patient age at treatment initiation",
            roles=["confounder"],
        )
    ]


def test_agentic_proposal_validation_rejects_duplicate_and_leakage():
    search_config = AgenticFeatureSearchConfig(max_additions_per_iter=2)
    raw = [
        {
            "action": "add",
            "name": "Age",
            "type": "continuous",
            "roles": ["confounder"],
            "description": "Patient age",
            "leakage_risk": "low",
        },
        {
            "action": "add",
            "name": "response_category",
            "type": "categorical",
            "categories": ["response", "no_response"],
            "roles": ["effect_modifier"],
            "description": "Response to treatment after therapy",
            "leakage_risk": "medium",
        },
        {
            "action": "add",
            "name": "baseline_nlr",
            "type": "continuous",
            "roles": ["effect_modifier"],
            "description": "Baseline neutrophil to lymphocyte ratio before treatment",
            "leakage_risk": "low",
        },
    ]

    valid, rejected = validate_agentic_proposals(
        raw,
        current_specs=_base_specs(),
        search_config=search_config,
        allow_removals=False,
    )

    assert [proposal.name for proposal in valid] == ["baseline_nlr"]
    assert {item["reason"] for item in rejected} == {
        "duplicate_feature",
        "post_treatment_or_outcome_leakage",
    }


def test_agentic_proposal_validation_allows_baseline_age_response_rationale():
    raw = [
        {
            "action": "add",
            "name": "patient_age",
            "type": "continuous",
            "roles": ["confounder", "effect_modifier"],
            "description": "The age of the patient at the time of baseline diagnosis or presentation.",
            "rationale": (
                "Age influences treatment selection and may modify physiological "
                "response to therapy."
            ),
            "expected_signal": "treatment, outcome",
            "leakage_risk": "low",
        },
        {
            "action": "add",
            "name": "baseline_treatment_response",
            "type": "categorical",
            "categories": ["responder", "non_responder"],
            "roles": ["effect_modifier"],
            "description": "Baseline treatment response category",
            "leakage_risk": "low",
        },
    ]

    valid, rejected = validate_agentic_proposals(
        raw,
        current_specs=[],
        search_config=AgenticFeatureSearchConfig(max_additions_per_iter=2),
        allow_removals=False,
    )

    assert [proposal.name for proposal in valid] == ["patient_age"]
    assert rejected == [
        {
            "proposal": raw[1],
            "reason": "post_treatment_or_outcome_leakage",
        }
    ]


def test_apply_proposals_add_remove_and_update_role():
    specs = _base_specs() + [
        ExplicitFeatureSpec(
            name="pdl1",
            type="categorical",
            categories=["low", "high"],
            roles=["effect_modifier"],
        )
    ]
    proposals = [
        AgenticFeatureProposal(
            action="add",
            name="ecog",
            type="categorical",
            categories=["0", "1", "2"],
            roles=["confounder", "effect_modifier"],
            description="Baseline ECOG performance status",
        ),
        AgenticFeatureProposal(action="remove", name="pdl1"),
        AgenticFeatureProposal(action="update_role", name="age", roles=["confounder", "effect_modifier"]),
    ]

    updated = apply_proposals(specs, proposals)

    assert [spec.name for spec in updated] == ["age", "ecog"]
    assert updated[0].roles == ["confounder", "effect_modifier"]
    assert updated[1].roles == ["confounder", "effect_modifier"]


def test_candidate_acceptance_uses_r_loss_and_auc_guardrails():
    search_config = AgenticFeatureSearchConfig(
        min_r_loss_improvement=0.05,
        max_outcome_auroc_drop=0.002,
        max_treatment_auroc_drop=0.002,
        min_improvement_fold_fraction=1.0,
    )
    baseline = [
        {"inner_fold": 1, "r_loss": 1.0, "outcome_auroc": 0.70, "treatment_auroc": 0.75},
        {"inner_fold": 2, "r_loss": 1.0, "outcome_auroc": 0.70, "treatment_auroc": 0.75},
    ]
    good_candidate = [
        {"inner_fold": 1, "r_loss": 0.90, "outcome_auroc": 0.70, "treatment_auroc": 0.75},
        {"inner_fold": 2, "r_loss": 0.92, "outcome_auroc": 0.70, "treatment_auroc": 0.75},
    ]
    bad_outcome_candidate = [
        {"inner_fold": 1, "r_loss": 0.80, "outcome_auroc": 0.60, "treatment_auroc": 0.75},
        {"inner_fold": 2, "r_loss": 0.82, "outcome_auroc": 0.60, "treatment_auroc": 0.75},
    ]

    assert compare_candidate_to_baseline(baseline, good_candidate, search_config)[
        "passes_acceptance"
    ]
    assert not compare_candidate_to_baseline(baseline, bad_outcome_candidate, search_config)[
        "passes_acceptance"
    ]


def test_extraction_cache_hash_includes_description_and_prompt_settings():
    spec_a = ExplicitFeatureSpec(
        name="age",
        type="continuous",
        description="Age at diagnosis",
        roles=["confounder"],
    )
    spec_b = ExplicitFeatureSpec(
        name="age",
        type="continuous",
        description="Age at treatment initiation",
        roles=["confounder"],
    )
    base = {
        "features": [spec_a],
        "prompt_template_version": "v1",
        "vllm_model_name": "model",
        "extraction_temperature": 0.0,
        "extraction_max_tokens": 128,
        "extraction_max_text_length": 1000,
    }

    desc_hash = _compute_config_hash({**base, "features": [spec_b]})
    prompt_hash = _compute_config_hash({**base, "prompt_template_version": "v2"})

    assert _compute_config_hash(base) != desc_hash
    assert _compute_config_hash(base) != prompt_hash


class FakeAgent:
    def __init__(self):
        self.contexts = []

    def propose(self, context):
        self.contexts.append(context)
        return [
            {
                "action": "add",
                "name": "hidden_modifier",
                "type": "continuous",
                "roles": ["effect_modifier"],
                "description": "Baseline hidden modifier measured before treatment",
                "rationale": "Could explain treatment effect heterogeneity",
                "expected_signal": "tau signal",
                "leakage_risk": "low",
            }
        ]


class FakeExtractionProvider:
    def ensure_features(self, dataset, specs):
        dataset = dataset.copy()
        for spec in specs:
            value_col = f"explicit_feat_{spec.name}"
            missing_col = f"{value_col}_missing"
            if value_col in dataset.columns:
                continue
            if spec.type == "categorical":
                dataset[value_col] = spec.categories[0]
            else:
                dataset[value_col] = np.arange(len(dataset), dtype=float)
            dataset[missing_col] = False
        return dataset


class FakeEvaluator:
    def evaluate_split(self, train_df, test_df, specs, fold_id):
        has_hidden = any(spec.name == "hidden_modifier" for spec in specs)
        r_loss = 0.50 if has_hidden else 1.00
        predictions = test_df.copy()
        predictions["pred_ite_prob"] = 0.10 if has_hidden else 0.0
        predictions["pred_y0_prob"] = 0.40
        predictions["pred_y1_prob"] = 0.50
        predictions["pred_propensity_prob"] = 0.50
        predictions["cv_fold"] = fold_id
        metrics = {
            "fold": fold_id,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "n_explicit_features": len(specs),
            "n_x_features": int(has_hidden),
            "n_w_features": 1,
            "ate_estimate": 0.10 if has_hidden else 0.0,
            "r_loss": r_loss,
            "outcome_auroc": 0.70,
            "treatment_auroc": 0.75,
            "oracle_true_ite_corr": 0.99,
        }
        return SplitEvaluation(predictions=predictions, metrics=metrics)


def test_agentic_runner_accepts_inner_cv_improvement_without_true_ite_leakage(tmp_path):
    df = pd.DataFrame(
        {
            "patient_id": np.arange(12),
            "clinical_text": [f"Patient {i}" for i in range(12)],
            "treatment_indicator": [0, 1] * 6,
            "outcome_indicator": [0, 0, 1, 1] * 3,
            "true_ite_prob": np.linspace(-0.1, 0.1, 12),
        }
    )
    agent = FakeAgent()
    output_path = tmp_path / "predictions.parquet"
    config = AppliedInferenceConfig(
        dataset_path=str(tmp_path / "dataset.parquet"),
        cv_folds=2,
        architecture=ModelArchitectureConfig(
            model_type="agentic_explicit_feature_forest",
            explicit_feature_forest=ExplicitFeatureForestConfig(
                n_estimators=8,
                min_samples_leaf=2,
                honest=False,
                inference=False,
            ),
            agentic_feature_search=AgenticFeatureSearchConfig(
                outer_folds=2,
                inner_folds=2,
                max_iterations=1,
                min_r_loss_improvement=0.01,
                min_improvement_fold_fraction=1.0,
            ),
        ),
        explicit_features=ExplicitFeatureExtractionConfig(
            enabled=True,
            features=_base_specs(),
            cache_enabled=False,
        ),
    )

    run_agentic_explicit_feature_forest(
        dataset=df,
        config=config,
        output_path=output_path,
        proposal_agent=agent,
        extraction_provider=FakeExtractionProvider(),
        evaluator=FakeEvaluator(),
    )

    results = pd.read_parquet(output_path)
    feature_sets = json.loads(
        (tmp_path / "agentic_feature_search" / "feature_sets.json").read_text()
    )
    selected_names = {
        feature["name"]
        for row in feature_sets
        if row["stage"] == "selected"
        for feature in row["features"]
    }

    assert len(results) == len(df)
    assert "hidden_modifier" in selected_names
    assert all("true_ite" not in json.dumps(context) for context in agent.contexts)
    decision_lines = (
        tmp_path / "agentic_feature_search" / "agent_decisions.jsonl"
    ).read_text().splitlines()
    persisted_contexts = [
        json.loads(line)["payload"].get("context", {})
        for line in decision_lines
        if json.loads(line)["event"] == "agent_proposals"
    ]
    assert all(context.get("clinical_text_examples") == [] for context in persisted_contexts)


def test_experiment_config_parses_agentic_search_config(tmp_path):
    dataset_path = tmp_path / "dataset.parquet"
    pd.DataFrame(
        {
            "clinical_text": ["note"],
            "treatment_indicator": [0],
            "outcome_indicator": [0],
        }
    ).to_parquet(dataset_path)
    config = ExperimentConfig.from_dict(
        {
            "applied_inference": {
                "dataset_path": str(dataset_path),
                "architecture": {
                    "model_type": "agentic_explicit_feature_forest",
                    "agentic_feature_search": {"outer_folds": 3, "inner_folds": 2},
                },
                "explicit_features": {
                    "features": [
                        {
                            "name": "age",
                            "type": "continuous",
                            "roles": ["confounder"],
                        }
                    ]
                },
            }
        }
    )

    assert config.applied_inference.architecture.agentic_feature_search.outer_folds == 3
    empty_start = ExperimentConfig.from_dict(
        {
            "applied_inference": {
                "dataset_path": str(dataset_path),
                "architecture": {"model_type": "agentic_explicit_feature_forest"},
                "explicit_features": {"enabled": True, "features": []},
            }
        }
    )
    empty_start.validate()

    with pytest.raises(ValueError, match="requires at least one"):
        ExperimentConfig.from_dict(
            {
                "applied_inference": {
                    "dataset_path": str(dataset_path),
                    "architecture": {"model_type": "explicit_feature_forest"},
                    "explicit_features": {"enabled": True, "features": []},
                }
            }
        ).validate()
