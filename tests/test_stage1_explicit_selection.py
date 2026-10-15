"""Explicit Stage 1 architecture selections resolve runnable contexts (B1)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oci.inference.embedding_contrast_discovery import EmbeddingContrastEvidenceGenerator
from oci.inference.research_all_evidence_workflow import (
    ResearchAllEvidenceWorkflow,
    compile_config,
)
from oci.inference.stage1_architecture_artifacts import (
    materialize_stage1_architecture_artifacts,
)
from oci.inference.stage1_architectures import (
    STAGE1_ARCHITECTURES,
    canonicalize_stage1_architectures,
)
from oci.models import concept_embedding_cache

EMBEDDING_LANES = {
    "embedding_whole_cohort",
    "embedding_clustered",
    "tfidf_semantic_retrieval_contrasts",
}


def _raw_config(tmp_path: Path, *, embedding_template: dict | None = None) -> dict:
    dataset = tmp_path / "cohort.parquet"
    pd.DataFrame(
        {
            "patient_id": ["a", "b", "c", "d", "e", "f"],
            "clinical_text": [
                "alpha beta gamma",
                "beta gamma delta",
                "gamma delta epsilon",
                "delta epsilon zeta",
                "epsilon zeta eta",
                "zeta eta theta",
            ],
            "treatment_indicator": [0, 1, 0, 1, 0, 1],
            "outcome_indicator": [0, 0, 1, 1, 0, 1],
        }
    ).to_parquet(dataset, index=False)
    multi_model: dict = {"feature_discovery_methods": ["bow", "htr", "embedding_contrast"]}
    if embedding_template is not None:
        multi_model = {"embedding_contrast": embedding_template}
    stage1_template = tmp_path / "stage1.json"
    stage1_template.write_text(
        json.dumps(
            {
                "config": {
                    "architecture": {
                        "model_type": "multi_model_forest",
                        "multi_model_forest": multi_model,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    query_template = tmp_path / "queries.json"
    query_template.write_text(json.dumps({"query_inner_folds": 2}), encoding="utf-8")
    return {
        "dataset": str(dataset),
        "output_dir": str(tmp_path / "output"),
        "stage1_template": str(stage1_template),
        "neural_query_template": str(query_template),
        "science": {"outer_folds": 2, "inner_folds": 2, "seed": 7},
        "run": {"workers": 1},
        "models": {"htr": "test-htr-model", "embeddings": "test-embedding-model"},
    }


SELECTIONS = [
    *[(name,) for name in STAGE1_ARCHITECTURES],
    ("bow_nuisance", "tfidf_topics"),
    ("tfidf_topics", "neural_query_moments"),
    ("embedding_whole_cohort", "neural_query_moments"),
    ("embedding_clustered", "tfidf_semantic_retrieval_contrasts"),
    ("htr_neural", "embedding_whole_cohort"),
    ("matched_pair_uplift", "tfidf_orphan_ngrams"),
    STAGE1_ARCHITECTURES,
]


@pytest.mark.parametrize("selection", SELECTIONS, ids=lambda value: "+".join(value))
def test_every_single_lane_and_combination_resolves_a_runnable_context(
    tmp_path, selection
):
    raw = _raw_config(tmp_path)
    raw["science"]["stage1_architectures"] = list(selection)
    workflow = ResearchAllEvidenceWorkflow(compile_config(raw, config_dir=tmp_path))

    context = workflow._resolved_context()

    canonical = canonicalize_stage1_architectures(selection, allow_none=False)
    assert context.selected_architectures == canonical
    model = context.applied_config.architecture.multi_model_forest
    assert model.tfidf_topic.orphan_ngram_enabled is ("tfidf_orphan_ngrams" in selection)
    if "text_models" in workflow.components:
        # MultiModelForestStage1Runner requires at least one nuisance source.
        assert model.bow_discovery_enabled or model.htr_evidence_enabled
        embedding_selected = bool(EMBEDDING_LANES.intersection(selection))
        assert model.embedding_contrast.enabled is embedding_selected
        if not embedding_selected:
            assert model.embedding_contrast.disable_reason
    if "neural_query_moments" in selection or EMBEDDING_LANES.intersection(selection):
        assert "embedding_cache" in workflow.components
    manifest = json.loads(
        (context.output_dir / "stage1_architectures" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["selected_architectures"] == list(canonical)


def test_embedding_only_selection_runs_bow_nuisance_as_private_support(tmp_path):
    raw = _raw_config(tmp_path)
    raw["science"]["stage1_architectures"] = ["embedding_whole_cohort"]
    context = ResearchAllEvidenceWorkflow(
        compile_config(raw, config_dir=tmp_path)
    )._resolved_context()
    model = context.applied_config.architecture.multi_model_forest

    assert model.bow_discovery_enabled is True
    assert model.htr_evidence_enabled is False
    assert model.matched_pair_uplift_enabled is False
    assert model.feature_discovery_methods == ["bow", "embedding_contrast"]

    # The BoW support evidence is present in the raw text-model row but must
    # not reach the selected-architecture handoff.
    source = context.output_dir / "components" / "text_models" / "evidence.jsonl"
    source.parent.mkdir(parents=True, exist_ok=True)
    raw_row = {
        "source": "text_models",
        "outer_fold": 1,
        "inner_fold": None,
        "scope": "full_outer_train",
        "evidence": {
            "importance": {
                "views": [
                    {
                        "view_name": "linear_1_3",
                        "treatment_positive": [{"feature": "prior chemotherapy", "score": 1.2}],
                        "pseudo_target_positive": [{"feature": "egfr mutation", "score": 0.7}],
                    }
                ],
                "phrase_consensus": [
                    {"feature": "prior chemotherapy", "best_abs_confounder_score": 0.5}
                ],
            },
            "embedding_contrast_evidence": {
                "contrasts": [
                    {
                        "name": "treatment_direction",
                        "contrast_family": "whole_cohort",
                        "role_hint": "confounder",
                        "positive_aligned_chunks": [
                            {"row_id": 1, "chunk_index": 0, "text": "ecog performance status 2"}
                        ],
                    }
                ]
            },
        },
    }
    source.write_text(json.dumps(raw_row) + "\n", encoding="utf-8")

    targeted, manifest = materialize_stage1_architecture_artifacts(
        output_dir=context.output_dir,
        raw_handoff_rows=[raw_row],
        selected_architectures=context.selected_architectures,
        source_artifacts={"text_models": source},
        selection_mode="explicit",
    )

    assert manifest["selected_architectures"] == ["embedding_whole_cohort"]
    assert targeted
    assert {row["evidence"]["architecture"] for row in targeted} == {
        "embedding_whole_cohort"
    }
    assert "prior chemotherapy" not in json.dumps(targeted)


class _FakeSentenceEncoder:
    max_seq_length = None

    def get_sentence_embedding_dimension(self) -> int:
        return 4

    def encode(self, chunks, **_kwargs):
        return np.asarray(
            [[len(chunk), chunk.count(" "), index, 1.0] for index, chunk in enumerate(chunks)],
            dtype=np.float32,
        )


def test_neural_query_selection_prepares_the_cache_with_embedding_lanes_disabled(
    tmp_path, monkeypatch
):
    raw = _raw_config(
        tmp_path,
        embedding_template={
            "enabled": False,
            "disable_reason": "site policy",
            "max_seq_length": None,
        },
    )
    raw["science"]["stage1_architectures"] = ["neural_query_moments"]
    workflow = ResearchAllEvidenceWorkflow(compile_config(raw, config_dir=tmp_path))
    context = workflow._resolved_context()
    assert workflow.components == ("embedding_cache", "neural_queries", "handoff")
    assert context.applied_config.architecture.multi_model_forest.embedding_contrast.enabled is False

    monkeypatch.setattr(
        concept_embedding_cache,
        "load_sentence_transformer",
        lambda *_args, **_kwargs: _FakeSentenceEncoder(),
    )
    generator = EmbeddingContrastEvidenceGenerator(
        config=context.applied_config,
        output_dir=context.component_dir("embedding_cache"),
        precompute_devices=("cpu",),
    )
    generator.prepare(context.dataset)

    matrices = generator.chunk_matrices([0, 3])
    assert len(matrices) == 2
    assert all(matrix.shape[1] == 4 and len(matrix) >= 1 for matrix in matrices)
    assert generator.chunk_texts([0])[0] == ["alpha beta gamma"]
    # Lane evidence stays disabled even though the cache is available.
    assert generator.build_evidence(
        discovery_df=context.dataset,
        y=np.zeros(6),
        t=np.zeros(6),
        pseudo_target=None,
        t_resid=None,
    ) == {}
