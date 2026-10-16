from __future__ import annotations

import json
import subprocess
import sys
import textwrap

RETIRED_ORCHESTRATORS = {
    "oci.inference.multi_model_agentic_forest",
    "oci.inference.agentic_attention_variable_forest",
    "oci.inference.agentic_explicit_feature_forest",
}

STAGE1_MODULES = (
    "oci.inference.research_all_evidence_workflow",
    "oci.inference.multi_model_forest_stage1",
    "oci.inference.htr_trainer",
    "oci.inference.htr_evidence_provider",
    "oci.inference.multi_model_pair_uplift",
    "oci.inference.stage1_dina",
    "oci.inference.tfidf_topic_stage1",
    "oci.inference.embedding_contrast_discovery",
    "oci.inference.neural_query_discovery_runtime",
    "oci.evaluation.stage1",
)

# A tiny binary-outcome context that exercises every lazy import on the Stage 1
# text-model path: BoW nuisance/effect views, sparse DINA nuisances, the HTR
# trainer (hash sentence encoder), and the matched-pair lane.
_STAGE1_CONTEXT_SCRIPT = textwrap.dedent(
    """
    import json, sys, tempfile
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from oci.config import (
        AgenticAttentionVariableForestConfig,
        AppliedInferenceConfig,
        BoWViewConfig,
        EmbeddingContrastDiscoveryConfig,
        ModelArchitectureConfig,
        MultiModelForestConfig,
        TrainingConfig,
    )
    from oci.inference.multi_model_forest_stage1 import run_multi_model_forest_handoff_context

    rng = np.random.default_rng(0)
    n = 48
    words = ["fatigue", "cough", "stage four", "ecog one", "nausea", "pain", "egfr", "alk"]
    dataset = pd.DataFrame(
        {
            "clinical_text": [" ".join(rng.choice(words, size=12)) for _ in range(n)],
            "treatment_indicator": np.tile([0, 1], n // 2),
            "outcome_indicator": np.tile([0, 0, 1, 1], n // 4),
        }
    )
    config = AppliedInferenceConfig(
        seed=7,
        outcome_type="binary",
        architecture=ModelArchitectureConfig(
            model_type="multi_model_forest",
            feature_extractor_type="hierarchical_transformer",
            htr_sentence_model="hash",
            htr_num_layers=1,
            htr_num_heads=2,
            htr_transformer_dim=16,
            htr_projection_dim=8,
            htr_hash_embedding_dim=16,
            htr_chunk_size_words=8,
            htr_chunk_overlap_words=2,
            htr_max_chunks=4,
            agentic_attention_variable_forest=AgenticAttentionVariableForestConfig(
                nuisance_folds=2,
                effect_folds=2,
                nuisance_epochs=1,
                effect_epochs=1,
            ),
            multi_model_forest=MultiModelForestConfig(
                nuisance_folds=2,
                effect_folds=2,
                top_n_features=5,
                bow_views=[BoWViewConfig(name="tiny", min_df=1, max_df=1.0)],
                embedding_contrast=EmbeddingContrastDiscoveryConfig(
                    enabled=False,
                    disable_reason="runtime import-boundary test",
                ),
            ),
        ),
        training=TrainingConfig(batch_size=8, effect_batch_size=8, epochs=1),
    )
    rows = np.arange(n)
    with tempfile.TemporaryDirectory() as directory:
        row = run_multi_model_forest_handoff_context(
            dataset=dataset,
            config=config,
            context={
                "train_idx": rows[:36],
                "heldout_idx": rows[36:],
                "fold_key": 1,
                "outer_fold": 1,
                "scope": "full_outer_train",
            },
            handoff_dir=Path(directory),
            device="cpu",
            num_workers=1,
        )
    assert {"importance", "htr_evidence"} <= set(row), sorted(row)
    assert set(row["htr_evidence"]) >= {"nuisance", "effect"}
    assert "pair_uplift" not in row["htr_evidence"]
    assert row["importance"]["matched_pair_uplift"]["schema_version"] == (
        "matched_pair_conditional_logit_v1"
    )
    print(json.dumps(sorted(sys.modules)))
    """
)


def _loaded_after(statement: str) -> set[str]:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import json, sys; {statement}; print(json.dumps(sorted(sys.modules)))",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return set(json.loads(completed.stdout.splitlines()[-1]))


def test_importing_package_does_not_initialize_model_runtimes():
    loaded = _loaded_after("import oci")

    assert "torch" not in loaded
    assert "oci.experiments.runner" not in loaded


def test_stage1_modules_do_not_import_retired_orchestrators():
    loaded = _loaded_after("; ".join(f"import {module}" for module in STAGE1_MODULES))

    assert not loaded & RETIRED_ORCHESTRATORS


def test_running_a_stage1_context_does_not_load_retired_orchestrators():
    completed = subprocess.run(
        [sys.executable, "-c", _STAGE1_CONTEXT_SCRIPT],
        check=True,
        capture_output=True,
        text=True,
        env=None,
        timeout=600,
    )
    loaded = set(json.loads(completed.stdout.splitlines()[-1]))

    assert "oci.inference.htr_trainer" in loaded
    assert "oci.inference.stage1_dina" in loaded
    assert not loaded & RETIRED_ORCHESTRATORS
