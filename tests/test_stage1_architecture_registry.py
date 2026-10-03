from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from oci.inference.neural_query_evidence_contract import query_retrieval_policy

from oci.inference.all_evidence_fusion import PRIMARY_SOURCE_FAMILIES
from oci.inference.plain_handoff_stage2_evidence import (
    SUPPORTED_STAGE2_ARCHITECTURES,
    compile_stage2_handoff_evidence,
)
from oci.inference.plain_handoff_stage2 import (
    PlainHandoffStage2Config,
    plain_stage2_config_from_mapping,
)
from oci.inference.stage1_architecture_artifacts import (
    _score_artifacts,
    iter_stage1_architecture_evidence,
    materialize_stage1_architecture_artifacts,
)
from oci.inference.stage1_architectures import (
    DEFAULT_STAGE1_ARCHITECTURES,
    STAGE1_ARCHITECTURES,
    canonicalize_stage1_architectures,
    resolve_support_services,
    selected_components,
)

EXPECTED_ARCHITECTURES = (
    "bow_nuisance",
    "bow_r_loss",
    "matched_pair_uplift",
    "htr_neural",
    "embedding_whole_cohort",
    "embedding_clustered",
    "tfidf_semantic_retrieval_contrasts",
    "tfidf_topics",
    "tfidf_orphan_ngrams",
    "neural_query_moments",
)


def test_registry_is_the_single_canonical_ten_architecture_contract():
    assert STAGE1_ARCHITECTURES == EXPECTED_ARCHITECTURES
    assert PRIMARY_SOURCE_FAMILIES is STAGE1_ARCHITECTURES
    assert SUPPORTED_STAGE2_ARCHITECTURES is STAGE1_ARCHITECTURES
    assert len(resolve_support_services(STAGE1_ARCHITECTURES)) == len(
        set(resolve_support_services(STAGE1_ARCHITECTURES))
    )


def test_selection_is_validated_and_canonicalized():
    assert canonicalize_stage1_architectures("tfidf_topics,bow_nuisance") == (
        "bow_nuisance",
        "tfidf_topics",
    )
    assert canonicalize_stage1_architectures("all") == STAGE1_ARCHITECTURES
    assert selected_components(("neural_query_moments",)) == (
        "embedding_cache",
        "neural_queries",
        "handoff",
    )
    with pytest.raises(ValueError, match="duplicate"):
        canonicalize_stage1_architectures("bow_nuisance,bow_nuisance")
    with pytest.raises(ValueError, match="unknown"):
        canonicalize_stage1_architectures("retired_agent")


def test_stage2_mapping_accepts_comma_separated_architecture_names():
    config = plain_stage2_config_from_mapping(
        {
            "endpoint": "http://stage2.test/v1",
            "required_architectures": "bow_nuisance,tfidf_topics",
            "included_architectures": "bow_nuisance,tfidf_topics",
        },
        default_workers=1,
    )

    assert config is not None
    assert config.required_architectures == ("bow_nuisance", "tfidf_topics")
    assert config.included_architectures == ("bow_nuisance", "tfidf_topics")


def test_stage2_defaults_require_eight_architectures_and_still_accept_all_ten():
    config = plain_stage2_config_from_mapping(
        {"endpoint": "http://stage2.test/v1"},
        default_workers=1,
    )
    assert config.required_architectures == DEFAULT_STAGE1_ARCHITECTURES
    assert config.included_architectures is None
    assert PlainHandoffStage2Config(endpoint="http://stage2.test/v1").required_architectures == (
        DEFAULT_STAGE1_ARCHITECTURES
    )
    all_architectures = plain_stage2_config_from_mapping(
        {"endpoint": "http://stage2.test/v1", "required_architectures": "all"},
        default_workers=1,
    )
    assert all_architectures.required_architectures == STAGE1_ARCHITECTURES


def test_targeted_artifacts_expose_only_the_selected_architecture(tmp_path: Path):
    source = tmp_path / "components" / "tfidf" / "evidence.jsonl"
    source.parent.mkdir(parents=True)
    primary_scores = source.parent / "predictions.parquet"
    context_scores = (
        source.parent
        / "stage1_tfidf_topics"
        / "contexts"
        / "outer_001_inner_001"
        / "nuisance_predictions.parquet"
    )
    context_scores.parent.mkdir(parents=True)
    primary_scores.touch()
    context_scores.touch()
    raw_rows = [
        {
            "source": "tfidf",
            "outer_fold": 1,
            "inner_fold": None,
            "scope": "full_outer_train",
            "evidence": {
                "discovery": {
                    "topic_banks": {
                        "treatment": {
                            "topics": [
                                {
                                    "topic_id": "topic-1",
                                    "terms": [
                                        {"term": "pretreatment performance status", "loading": 0.9}
                                    ],
                                }
                            ]
                        },
                        "outcome": {"topics": []},
                        "effect": {"topics": []},
                    }
                }
            },
        }
    ]
    source.write_text(json.dumps(raw_rows[0]) + "\n", encoding="utf-8")

    targeted, manifest = materialize_stage1_architecture_artifacts(
        output_dir=tmp_path,
        raw_handoff_rows=raw_rows,
        selected_architectures=("tfidf_topics",),
        source_artifacts={"tfidf": source},
        selection_mode="explicit",
    )

    assert manifest["selected_architectures"] == ["tfidf_topics"]
    assert manifest["architectures"]["tfidf_topics"]["score_artifacts"] == [
        "components/tfidf/predictions.parquet"
    ]
    assert _score_artifacts(tmp_path, "tfidf_orphan_ngrams") == []
    assert {row["evidence"]["architecture"] for row in targeted} == {"tfidf_topics"}
    assert {row["architecture"] for row in iter_stage1_architecture_evidence(tmp_path)} == {
        "tfidf_topics"
    }
    compiled = compile_stage2_handoff_evidence(
        targeted,
        handoff_path=tmp_path / "handoff" / "evidence.jsonl",
        max_cards_per_outer_fold=16,
        max_packet_chars=2_000,
        required_architectures=("tfidf_topics",),
        included_architectures=("tfidf_topics",),
    )
    assert {packet["architecture"] for packet in compiled.packets} == {"tfidf_topics"}
    assert {
        reference["handoff_row"]
        for member in compiled.members_by_outer_fold[1]
        for reference in member["raw_references"]
    } == {1}


def test_neural_query_architecture_artifact_uses_compact_occurrence_records(
    tmp_path: Path,
):
    source = tmp_path / "components" / "neural_queries" / "evidence.jsonl"
    source.parent.mkdir(parents=True)
    raw_rows = [
        {
            "source": "neural_queries",
            "outer_fold": 1,
            "inner_fold": inner_fold,
            "scope": "candidate_consistency_inner_train",
            "evidence": {
                "evidence": [
                    {
                        "retrieval_policy": query_retrieval_policy(1),
                        "query_id": "effect_query",
                        "bank": "effect",
                        "top_contrastive_ngrams": [
                            {
                                "term": "pretreatment performance status",
                                "tfidf_contrast": score,
                            }
                        ],
                    }
                ]
            },
        }
        for inner_fold, score in ((1, 0.4), (2, 0.7))
    ]
    source.write_text(
        "".join(json.dumps(row["evidence"]) + "\n" for row in raw_rows),
        encoding="utf-8",
    )

    targeted, manifest = materialize_stage1_architecture_artifacts(
        output_dir=tmp_path,
        raw_handoff_rows=iter(raw_rows),
        selected_architectures=("neural_query_moments",),
        source_artifacts={"neural_queries": source},
        selection_mode="explicit",
    )

    architecture = manifest["architectures"]["neural_query_moments"]
    assert architecture["occurrences"] == 2
    assert architecture["compact_records"] == 1
    assert len(targeted) == 1
    occurrence = targeted[0]["evidence"]["occurrence"]
    assert occurrence["raw_occurrence_count"] == 2
    assert {row["inner_fold"] for row in occurrence["reference_summaries"]} == {1, 2}
    assert {row["handoff_row"] for row in occurrence["reference_summaries"]} == {1}
    persisted = json.loads(
        (tmp_path / "stage1_architectures" / "neural_query_moments" / "evidence.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert {
        row["handoff_row"]
        for row in persisted["occurrence"]["reference_summaries"]
    } == {1}


def test_stage2_included_architectures_filter_private_support(tmp_path: Path):
    rows = [
        {
            "source": "stage1_architecture",
            "outer_fold": 1,
            "inner_fold": None,
            "scope": "full_outer_train",
            "evidence": {
                "architecture": architecture,
                "occurrence": {
                    "text": f"{architecture} evidence",
                    "evidence_kind": "structured_evidence",
                    "axes": ["semantic"],
                    "polarity": "unsigned",
                    "source_families": [architecture],
                    "architecture": architecture,
                    "reference": {
                        "source": "component",
                        "scope": "full_outer_train",
                        "inner_fold": None,
                        "json_path": "evidence",
                    },
                    "details": {},
                    "scores": {},
                    "patient_row_id": None,
                    "cache_coordinate": None,
                },
            },
        }
        for architecture in ("bow_nuisance", "htr_neural")
    ]
    compiled = compile_stage2_handoff_evidence(
        rows,
        handoff_path=tmp_path / "handoff" / "evidence.jsonl",
        max_cards_per_outer_fold=16,
        max_packet_chars=2_000,
        required_architectures=("bow_nuisance",),
        included_architectures=("bow_nuisance",),
    )
    assert {packet["architecture"] for packet in compiled.packets} == {"bow_nuisance"}
