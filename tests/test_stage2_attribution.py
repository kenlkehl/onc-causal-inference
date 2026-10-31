"""Attribution of Stage 2 variables to Stage 1 architectures (oci-evaluate-attribution).

The fake run directories below use the field names the current Stage 2 writers
produce (cards.jsonl, interpreted_candidates.json, feature_definitions.json,
candidate_missingness_filter.json, selection/, final_definitions.json, and the
held-out extraction files); no LLM or model is involved.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oci.evaluation import attribution
from oci.evaluation.attribution import (
    FrozenArtifactReader,
    collect_stage2_lineage,
    evaluate_attribution,
    load_oracle_data,
)

ROWS = 40
HELDOUT = {1: list(range(0, 20)), 2: list(range(20, 40))}


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _card(card_id: str, architecture: str) -> dict:
    return {
        "card_id": card_id,
        "schema_version": "semantic_cluster_cards_v5",
        "source_architectures": [architecture],
        "evidence_axes": ["outcome"],
    }


def _candidate(candidate_id: str, architecture: str, card: str, name: str) -> dict:
    return {
        "candidate_id": candidate_id,
        "architecture": architecture,
        "name": name,
        "supporting_packet_ids": [card],
        "supporting_architectures": [architecture],
        "evidence_axes": ["outcome"],
    }


def _feature(feature_id: str, name: str, cards, architectures, **extra) -> dict:
    return {
        "feature_id": feature_id,
        "name": name,
        "description": f"Baseline {name.replace('_', ' ')}.",
        "measurement_definition": f"Record the latest pretreatment {name}.",
        "value_type": extra.pop("value_type", "continuous"),
        "roles": [],
        "supporting_packet_ids": list(cards),
        "supporting_architectures": list(architectures),
        **extra,
    }


def _fake_run(root: Path, *, heldout: bool = False) -> Path:
    """Two outer folds; fold 1 exercises merges, splits, latents and explicit features."""

    stage2 = root / "stage2"
    _write_json(root / "run_config.json", {"dataset": str(root / "dataset.parquet")})
    _write_json(
        stage2 / "config.json",
        {"included_architectures": [
            "bow_nuisance", "htr_neural", "tfidf_topics", "neural_query_moments",
        ]},
    )
    # ---- fold 1
    fold = stage2 / "outer_001"
    _write_jsonl(
        stage2 / "evidence_compilation" / "outer_001" / "cards.jsonl",
        [
            _card("c_bow", "bow_nuisance"),
            _card("c_topic", "tfidf_topics"),
            _card("c_htr", "htr_neural"),
            _card("c_htr2", "htr_neural"),
            _card("c_query", "neural_query_moments"),
        ],
    )
    _write_json(
        fold / "interpreted_candidates.json",
        [
            _candidate("candidate_0001", "bow_nuisance", "c_bow", "ecog"),
            _candidate("candidate_0002", "tfidf_topics", "c_topic", "ecog"),
            _candidate("candidate_0003", "htr_neural", "c_htr", "pdl1"),
            _candidate("candidate_0004", "tfidf_topics", "c_topic", "smoking"),
            _candidate("candidate_0005", "neural_query_moments", "c_query", "albumin"),
            _candidate("candidate_0006", "htr_neural", "c_htr2", "weight_loss"),
        ],
    )
    definitions = [
        _feature("outer_001_feature_001", "ecog", ["c_bow", "c_topic"],
                 ["bow_nuisance", "tfidf_topics"], value_type="ordinal"),
        _feature("outer_001_feature_002", "pdl1", ["c_htr"], ["htr_neural"]),
        _feature("outer_001_feature_003", "smoking", ["c_topic"], ["tfidf_topics"]),
        _feature("outer_001_feature_004", "albumin", ["c_query"], ["neural_query_moments"]),
        _feature("outer_001_feature_005", "weight_loss", ["c_htr2"], ["htr_neural"]),
        _feature("outer_001_feature_006", "age", [], [], configured_explicit_feature=True),
    ]
    dispositions = {
        "configured_explicit_feature_0001": {"feature_name": "age", "status": "retained"},
        "candidate_0001": {"feature_name": "ecog", "status": "retained"},
        "candidate_0002": {"feature_name": "ecog", "status": "merged"},
        "candidate_0003": {"feature_name": "pdl1", "status": "retained"},
        "candidate_0004": {"feature_name": "smoking", "status": "retained"},
        "candidate_0005": {"feature_name": "albumin", "status": "retained"},
        "candidate_0006": {"feature_name": "weight_loss", "status": "retained"},
    }
    _write_json(
        fold / "feature_definitions.json",
        {"outer_fold": 1, "features": definitions, "candidate_dispositions": dispositions},
    )
    # The extraction review split pdl1; the child keeps only its parent pointer.
    split_child = {
        "feature_id": "outer_001_feature_002__score",
        "name": "pdl1__score",
        "description": "PD-L1 tumor proportion score.",
        "value_type": "continuous",
        "roles": [],
        "ontology_parent_feature_id": "outer_001_feature_002",
    }
    current = [definitions[0], split_child, definitions[2], definitions[3], definitions[4],
               definitions[5]]
    _write_json(
        fold / "extraction" / "candidate_missingness_filter.json",
        {
            "schema_version": "stage2_candidate_informativeness_filter_v3_documented_unknown",
            "features": [
                {"feature_id": f["feature_id"], "name": f["name"],
                 "action": "drop" if f["name"] == "smoking" else "keep"}
                for f in current
            ],
        },
    )
    kept = [f for f in current if f["name"] != "smoking"]
    _write_json(fold / "selection" / "input.json", {"definitions": kept})
    latent = {
        "feature_id": "latent_001",
        "name": "nutrition_index",
        "description": "Derived nutrition index.",
        "value_type": "continuous",
        "derived_structured_latent": True,
        "source_feature_ids": ["outer_001_feature_004", "outer_001_feature_005"],
        "measurement_dependency_feature_ids": ["outer_001_feature_004", "outer_001_feature_005"],
        "roles": [],
    }
    catalog = [definitions[0], split_child, definitions[5], latent]
    _write_json(fold / "selection" / "nuisance_candidates.json",
                {"features": catalog, "latents": []})
    selected = [
        {**definitions[0], "roles": ["confounder"]},
        {**split_child, "roles": ["effect_modifier"]},
        {**latent, "roles": ["confounder", "effect_modifier"]},
        {**definitions[5], "roles": ["confounder"]},
    ]
    _write_json(
        fold / "final_definitions.json",
        {"features": selected, "measurement_dependencies": [definitions[3], definitions[4]]},
    )
    # ---- fold 2: one sole-source BoW modifier
    fold2 = stage2 / "outer_002"
    _write_jsonl(stage2 / "evidence_compilation" / "outer_002" / "cards.jsonl",
                 [_card("c2_bow", "bow_nuisance")])
    _write_json(fold2 / "interpreted_candidates.json",
                [_candidate("candidate_0001", "bow_nuisance", "c2_bow", "ecog")])
    fold2_feature = _feature("outer_002_feature_001", "ecog_score", ["c2_bow"], ["bow_nuisance"])
    _write_json(
        fold2 / "feature_definitions.json",
        {"outer_fold": 2, "features": [fold2_feature],
         "candidate_dispositions": {"candidate_0001": {"feature_name": "ecog_score"}}},
    )
    _write_json(
        fold2 / "extraction" / "candidate_missingness_filter.json",
        {"features": [{"feature_id": "outer_002_feature_001", "action": "keep"}]},
    )
    _write_json(fold2 / "final_definitions.json",
                {"features": [{**fold2_feature, "roles": ["effect_modifier"]}]})
    if heldout:
        _write_heldout(root)
    return root


def _oracle_frame() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {
            "patient_id": [f"p{i:03d}" for i in range(ROWS)],
            "clinical_text": ["note"] * ROWS,
            "treatment_indicator": [i % 2 for i in range(ROWS)],
            "outcome_indicator": [(i // 2) % 2 for i in range(ROWS)],
            "true_ecog": [f"ECOG {i % 3}" for i in range(ROWS)],
            "true_pdl1": rng.normal(20.0, 8.0, ROWS),
            "true_albumin": rng.normal(3.8, 0.4, ROWS),
            "true_ite_prob": np.linspace(-0.1, 0.1, ROWS),
        }
    )


def _write_heldout(root: Path) -> None:
    truth = _oracle_frame()
    stage2 = root / "stage2"
    rng = np.random.default_rng(11)
    for outer, rows in HELDOUT.items():
        fold = stage2 / f"outer_{outer:03d}"
        part = truth.iloc[rows]
        if outer == 1:
            frame = pd.DataFrame(
                {
                    "_oci_row_id": rows,
                    # Same information as true_ecog under different labels.
                    "ecog": [value.replace("ECOG", "PS") for value in part["true_ecog"]],
                    # Named after the concept, but unrelated to it.
                    "pdl1__score": rng.normal(0.0, 1.0, len(rows)),
                    # Unrelated name, but an exact linear copy of true_pdl1.
                    "nutrition_index": 2.0 * part["true_pdl1"].to_numpy() + 1.0,
                    "age": rng.normal(65.0, 9.0, len(rows)),
                }
            )
            definitions = json.loads(
                (fold / "selection" / "nuisance_candidates.json").read_text()
            )["features"]
        else:
            frame = pd.DataFrame(
                {"_oci_row_id": rows, "ecog_score": rng.normal(0.0, 1.0, len(rows))}
            )
            definitions = json.loads((fold / "final_definitions.json").read_text())["features"]
        (fold / "estimation").mkdir(parents=True, exist_ok=True)
        frame.to_csv(fold / "estimation" / "nuisance_candidate_heldout.csv", index=False)
        _write_json(fold / "estimation" / "nuisance_candidate_definitions.json",
                    {"features": definitions})
        mask = frame[["_oci_row_id"]].copy()
        for column in frame.columns[1:]:
            mask[column] = 0
        mask.to_csv(fold / "extraction" / "failure_mask_heldout.csv", index=False)
    truth.to_parquet(root / "dataset.parquet", index=False)
    _write_json(
        root / "metadata.json",
        {"features": [
            {"name": "ecog", "type": "categorical", "roles": ["confounder"]},
            {"name": "pdl1", "type": "continuous", "roles": ["effect_modifier"]},
            {"name": "albumin", "type": "continuous", "roles": ["confounder"]},
            {"name": "histology", "type": "categorical", "roles": ["effect_modifier"]},
        ]},
    )


def _counts(report, fold, level):
    section = report["per_fold"][fold][level] if fold != "all" else report["totals"][level]
    return {
        architecture: (values["any_source"], values["sole_source"])
        for architecture, values in section["by_architecture"].items()
    }, section["totals"]


def test_attribution_counts_any_and_sole_source_at_every_level(tmp_path: Path):
    run = _fake_run(tmp_path / "run")
    result = evaluate_attribution(run_dir=run)
    output = Path(result["output_dir"])
    assert output == run / "evaluation" / "attribution"
    report = json.loads((output / "attribution.json").read_text())
    assert report["architectures"] == [
        "bow_nuisance", "htr_neural", "tfidf_topics", "neural_query_moments",
    ]

    cards, totals = _counts(report, "1", "cards")
    assert cards == {"bow_nuisance": (1, 1), "htr_neural": (2, 2), "tfidf_topics": (1, 1),
                     "neural_query_moments": (1, 1)}
    candidates, _ = _counts(report, "1", "candidates")
    assert candidates["tfidf_topics"] == (2, 2)

    defined, totals = _counts(report, "1", "variables_defined")
    # ecog merges BoW and topic candidates: credited to both, sole-source to neither.
    assert defined["bow_nuisance"] == (1, 0)
    assert defined["tfidf_topics"] == (2, 1)
    assert defined["htr_neural"] == (2, 2)
    assert totals["investigator_specified"] == 1 and totals["unattributed"] == 0

    passing, totals = _counts(report, "1", "variables_passing_filter")
    # smoking was dropped; the split child of pdl1 inherits HTR through its parent.
    assert passing["tfidf_topics"] == (1, 0)
    assert passing["htr_neural"] == (2, 2)
    assert totals["items"] == 5

    nuisance, totals = _counts(report, "1", "nuisance_set")
    # The latent inherits both measurement dependencies, so it is multi-source.
    assert nuisance == {"bow_nuisance": (1, 0), "htr_neural": (1, 0), "tfidf_topics": (1, 0),
                        "neural_query_moments": (1, 0)}
    assert totals == {"items": 3, "multi_source": 2, "investigator_specified": 1,
                      "unattributed": 0}

    modifiers, _ = _counts(report, "1", "selected_modifiers")
    assert modifiers["htr_neural"] == (2, 1)
    assert modifiers["neural_query_moments"] == (1, 0)
    total_modifiers, _ = _counts(report, "all", "selected_modifiers")
    assert total_modifiers["bow_nuisance"] == (1, 1)
    assert total_modifiers["htr_neural"] == (2, 1)

    counts = pd.read_csv(output / "attribution_counts.csv")
    assert set(counts["outer_fold"].astype(str)) == {"1", "2", "all"}
    row = counts[(counts["outer_fold"].astype(str) == "all")
                 & (counts["level"] == "selected_modifiers")
                 & (counts["architecture"] == "htr_neural")].iloc[0]
    assert (row["any_source"], row["sole_source"]) == (2, 1)

    variables = pd.read_csv(output / "variables.csv", keep_default_na=False)
    child = variables[variables["feature_id"] == "outer_001_feature_002__score"].iloc[0]
    assert child["candidate_ids"] == "candidate_0003"
    assert child["sole_source_architecture"] == "htr_neural"
    latent_row = variables[variables["feature_id"] == "latent_001"].iloc[0]
    assert latent_row["architectures"] == "htr_neural;neural_query_moments"
    assert latent_row["kind"] == "latent"
    age = variables[variables["name"] == "age"].iloc[0]
    assert age["kind"] == "investigator_specified" and age["architectures"] == ""

    manifest = json.loads((output / "attribution_manifest.json").read_text())
    assert manifest["oracle"] == {"requested": False}
    assert "stage2/outer_001/final_definitions.json" in manifest["stage2_artifact_sha256"]
    assert report["oracle_recovery"] == {"status": "not_requested"}


def test_outputs_inside_run_inputs_are_refused(tmp_path: Path):
    run = _fake_run(tmp_path / "run")
    with pytest.raises(ValueError, match="evaluation"):
        evaluate_attribution(run_dir=run, output_dir=run / "stage2" / "attribution")
    evaluate_attribution(run_dir=run, output_dir=tmp_path / "elsewhere")
    assert (tmp_path / "elsewhere" / "attribution.json").is_file()


def test_oracle_data_loads_only_after_every_artifact_is_hashed(tmp_path: Path, monkeypatch):
    run = _fake_run(tmp_path / "run", heldout=True)
    events: list[tuple[str, str]] = []
    original_read = FrozenArtifactReader.read_bytes
    original_load = attribution._read_dataset_columns

    def logged_read(self, path):
        events.append(("read", str(path)))
        return original_read(self, path)

    def logged_load(path, columns):
        events.append(("oracle", str(path)))
        return original_load(path, columns)

    monkeypatch.setattr(FrozenArtifactReader, "read_bytes", logged_read)
    monkeypatch.setattr(attribution, "_read_dataset_columns", logged_load)
    evaluate_attribution(run_dir=run, oracle=True)

    kinds = [kind for kind, _path in events]
    assert kinds.count("oracle") == 1
    oracle_index = kinds.index("oracle")
    assert all(kind == "read" for kind in kinds[:oracle_index])
    assert "read" not in kinds[oracle_index:]
    manifest = json.loads(
        (run / "evaluation" / "attribution" / "attribution_manifest.json").read_text()
    )
    hashed = manifest["stage2_artifact_sha256"]
    assert "stage2/outer_001/estimation/nuisance_candidate_heldout.csv" in hashed
    assert "stage2/outer_001/extraction/failure_mask_heldout.csv" in hashed

    # The reader refuses reads once frozen, and the loader refuses open readers.
    reader = FrozenArtifactReader(run)
    lineage = collect_stage2_lineage(run / "stage2", reader=reader, include_heldout=True)
    with pytest.raises(RuntimeError, match="freeze"):
        load_oracle_data(run / "dataset.parquet", None, readers=[reader])
    reader.freeze()
    with pytest.raises(RuntimeError, match="closed"):
        reader.json(run / "stage2" / "config.json")
    oracle = load_oracle_data(run / "dataset.parquet", run / "metadata.json", readers=[reader])
    assert [concept.name for concept in oracle.concepts] == ["ecog", "pdl1", "albumin"]
    # Without metadata: every true_<variable> column except the outcome-model columns.
    inferred = load_oracle_data(run / "dataset.parquet", None, readers=[reader])
    assert {(c.name, c.value_type) for c in inferred.concepts} == {
        ("ecog", "categorical"), ("pdl1", "continuous"), ("albumin", "continuous")}
    assert lineage.folds["1"].heldout_source == "estimation/nuisance_candidate_heldout.csv"
