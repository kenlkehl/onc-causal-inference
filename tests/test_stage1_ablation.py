"""Leave-one-architecture-out ablation tooling (scripts/ablate_stage1_architectures.py).

The fake runs are built with the current writers where one exists: component
rows in the producers' payload shapes, explicit handoffs from
``materialize_stage1_architecture_artifacts``, the Stage 2 block from a parsed
``PlainHandoffStage2Config``, and estimates from the diagnostics functions.
Nothing calls an LLM or fits a text model.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oci.evaluation import ablation
from oci.evaluation.ablation import compare_ablations, compare_concepts, prepare_ablations
from oci.extraction.llm_routing import redact_api_keys
from oci.inference.plain_handoff_stage2 import (
    _load_stage2_splits,
    _verify_stage2_splits_against_handoff,
    plain_stage2_config_from_mapping,
)
from oci.inference.plain_handoff_stage2_evidence import (
    compile_stage2_handoff_evidence,
    extract_stage1_architecture_occurrences,
)
from oci.inference.stage1_architecture_artifacts import materialize_stage1_architecture_artifacts
from oci.inference.stage2_diagnostics import (
    best_linear_predictor,
    co_primary_estimands,
    pool_blp,
)

ROOT = Path(__file__).resolve().parents[1]
ROWS = 40
HELDOUT = {1: list(range(0, 20)), 2: list(range(20, 40))}
LAUNCHER_KEYS = {
    "dataset", "handoff_path", "split_provenance_path", "output_dir", "report_dir",
    "clinical_question", "unit_id_column", "text_column", "treatment_column",
    "outcome_column", "outcome_type", "inner_folds", "seed", "stage2",
}


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _stage2_mapping(**overrides) -> dict:
    raw = {
        "endpoint": "http://primary.test/v1",
        "model": "primary-model",
        "api_key": "secret-primary",
        "extraction_llm": {"endpoint": "http://extract.test/v1", "model": "small-model",
                           "api_key": "secret-extract"},
        **overrides,
    }
    config = plain_stage2_config_from_mapping(raw, default_workers=8)
    # run_config.json stores the redacted dataclass fields.
    return json.loads(json.dumps(redact_api_keys(asdict(config))))


def _component_rows() -> dict[str, list[dict]]:
    text_models, tfidf = [], []
    for outer, heldout in HELDOUT.items():
        fit = sorted(set(range(ROWS)) - set(heldout))
        # One text-model row carries two architectures: BoW nuisance and R-loss.
        text_models.append(
            {
                "outer_fold": outer,
                "inner_fold": None,
                "scope": "full_outer_train",
                "importance": {"views": [{
                    "view_name": "bow",
                    "outcome_positive": [
                        {"feature": f"ecog two fold {outer}", "coefficient": 0.5},
                        {"feature": "weight loss", "coefficient": 0.3},
                    ],
                    "pseudo_target_positive": [
                        {"feature": "pd-l1 high", "coefficient": 0.4},
                    ],
                }]},
                "metrics": {"handoff_fit_rows": len(fit), "handoff_heldout_rows": len(heldout)},
            }
        )
        tfidf.append(
            {
                "outer_fold": outer,
                "inner_fold": None,
                "scope": "full_outer_train",
                "fit_row_ids": fit,
                "heldout_row_ids": heldout,
                "discovery": {"topic_banks": {"treatment": {"topics": [{
                    "topic_id": f"treatment-ecog-{outer}",
                    "terms": [{"term": "performance status", "loading": 0.9}],
                }]}}},
            }
        )
    return {"text_models": text_models, "tfidf": tfidf}


def _fake_run(
    root: Path,
    *,
    explicit: bool,
    stage2: dict | None = None,
    architectures=("bow_nuisance", "bow_r_loss", "tfidf_topics"),
    sources=("text_models", "tfidf"),
) -> Path:
    components = _component_rows()
    source_paths = {}
    raw_rows = []
    for source in sources:
        path = root / "components" / source / "evidence.jsonl"
        _write_jsonl(path, components[source])
        source_paths[source] = path
        raw_rows.extend(
            {"source": source, "outer_fold": row["outer_fold"], "inner_fold": None,
             "scope": row["scope"], "evidence": row}
            for row in components[source]
        )
    handoff = root / "handoff" / "evidence.jsonl"
    index = {"sources": {name: f"../components/{name}/evidence.jsonl" for name in sources}}
    if explicit:
        rows, _manifest = materialize_stage1_architecture_artifacts(
            output_dir=root,
            raw_handoff_rows=raw_rows,
            selected_architectures=architectures,
            source_artifacts=source_paths,
            selection_mode="explicit",
        )
        index["selected_architectures"] = list(architectures)
    else:
        rows = raw_rows
    _write_jsonl(handoff, rows)
    _write_json(root / "handoff" / "index.json", index)
    _write_json(root / "handoff" / "complete.json", {"status": "complete"})
    split_rows = []
    for outer, heldout in HELDOUT.items():
        fit = sorted(set(range(ROWS)) - set(heldout))
        split_rows.append(
            {"outer_fold": outer, "fit_row_ids": fit, "heldout_row_ids": heldout,
             "inner_splits": [
                 {"inner_fold": 1, "fit_row_ids": fit[:10], "heldout_row_ids": fit[10:]},
                 {"inner_fold": 2, "fit_row_ids": fit[10:], "heldout_row_ids": fit[:10]},
             ]}
        )
    _write_jsonl(root / "components" / "tfidf" / "split_provenance.jsonl", split_rows)
    dataset = root / "cohort.parquet"
    pd.DataFrame(
        {
            "patient_id": [f"p{i:03d}" for i in range(ROWS)],
            "clinical_text": [f"ECOG {i % 3}." for i in range(ROWS)],
            "treatment_indicator": [i % 2 for i in range(ROWS)],
            "outcome_indicator": [(i // 2) % 2 for i in range(ROWS)],
            "true_ecog": [f"ECOG {i % 3}" for i in range(ROWS)],
        }
    ).to_parquet(dataset, index=False)
    _write_json(
        root / "run_config.json",
        {
            "dataset": str(dataset),
            "output_dir": str(root),
            "unit_id_column": "patient_id",
            "text_column": "clinical_text",
            "treatment_column": "treatment_indicator",
            "outcome_column": "outcome_indicator",
            "outcome_type": "binary",
            "clinical_question": "Does vinorelbine outperform gemcitabine?",
            "outer_folds": 2,
            "inner_folds": 2,
            "seed": 7,
            "stage2": _stage2_mapping() if stage2 is None else stage2,
        },
    )
    return root


def _texts_by_architecture(rows) -> dict[str, list[tuple[int, str]]]:
    output: dict[str, list[tuple[int, str]]] = {}
    for fold, occurrences in extract_stage1_architecture_occurrences(rows).items():
        for occurrence in occurrences:
            output.setdefault(occurrence["architecture"], []).append((fold, occurrence["text"]))
    return {name: sorted(values) for name, values in output.items()}


def test_prepare_drops_only_the_ablated_rows_of_an_explicit_handoff(tmp_path: Path, capsys):
    run = _fake_run(tmp_path / "run", explicit=True)
    cache = run / "components" / "embedding_cache"
    cache.mkdir(parents=True)
    out = tmp_path / "ablations"
    index = prepare_ablations(run_dir=run, out_dir=out)
    printed = capsys.readouterr().out

    assert index["run_selection_source"] == "stage1_architectures/manifest.json"
    assert index["handoff_mode"] == "architecture_rows_copied"
    assert [item["name"] for item in index["ablations"]] == [
        "without_bow_nuisance", "without_bow_r_loss", "without_tfidf_topics",
    ]
    source_lines = (run / "handoff" / "evidence.jsonl").read_text().splitlines()
    for item in index["ablations"]:
        removed = item["removed_architecture"]
        directory = Path(item["directory"])
        lines = (directory / "handoff" / "evidence.jsonl").read_text().splitlines()
        # Kept rows are the source lines, byte for byte and in order.
        assert lines == [
            line for line in source_lines
            if json.loads(line)["evidence"]["architecture"] != removed
        ]
        config = json.loads((directory / "stage2_from_artifacts.json").read_text())
        assert LAUNCHER_KEYS <= set(config)
        assert config["dataset"] == str(run / "cohort.parquet")
        assert config["output_dir"] == str(directory / "stage2")
        parsed = plain_stage2_config_from_mapping(config["stage2"], default_workers=8)
        kept = [name for name in ("bow_nuisance", "bow_r_loss", "tfidf_topics") if name != removed]
        assert list(parsed.included_architectures) == kept
        assert list(parsed.required_architectures) == kept
        # Keys stay redacted; the launcher reads them from the environment.
        assert config["stage2"]["api_key"] == "<redacted>"
        assert config["stage2"]["extraction_llm"]["api_key"] == "<redacted>"
        manifest = json.loads((directory / "ablation.json").read_text())
        assert manifest["rows"]["removed_by_architecture"] == {
            removed: len(source_lines) - len(lines)}
        assert manifest["rows"]["removed_by_architecture"][removed] > 0
        for name, path in (
            ("handoff", directory / "handoff" / "evidence.jsonl"),
            ("split_provenance", directory / "components" / "tfidf" / "split_provenance.jsonl"),
            ("launcher_config", directory / "stage2_from_artifacts.json"),
        ):
            assert manifest["sha256"][name] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert manifest["sha256"]["split_provenance"] == manifest["sha256"][
            "source_split_provenance"]
        sources = json.loads((directory / "handoff" / "index.json").read_text())["sources"]
        assert all(Path(path).is_absolute() and Path(path).is_file() for path in sources.values())
        # Stage 2 finds the run's embedding cache beside the ablation handoff.
        link = directory / "components" / "embedding_cache"
        assert link.is_symlink() and link.resolve() == cache.resolve()
        assert manifest["embedding_cache"] == str(cache.resolve())
        assert f"{directory / 'stage2_from_artifacts.json'} --preflight" in printed
    assert "OCI_STAGE2_API_KEY" in printed and "OCI_STAGE2_EXTRACTION_API_KEY" in printed
    assert "compare --run-dir" in printed
    assert json.loads((out / "ablations.json").read_text())["filter_rule"] == ablation.FILTER_RULE


def test_prepare_projects_shared_component_rows_and_stage2_accepts_the_result(tmp_path: Path):
    run = _fake_run(tmp_path / "run", explicit=False)
    out = tmp_path / "ablations"
    index = prepare_ablations(run_dir=run, out_dir=out, architectures="bow_r_loss",
                              include_control=True)
    assert index["run_selection_source"] == "handoff"
    assert index["handoff_mode"] == "projected_to_architecture_rows"
    assert [item["name"] for item in index["ablations"]] == ["without_bow_r_loss", "full_control"]

    original = _texts_by_architecture(_read_jsonl(run / "handoff" / "evidence.jsonl"))
    assert set(original) == {"bow_nuisance", "bow_r_loss", "tfidf_topics"}
    directory = out / "without_bow_r_loss"
    handoff = directory / "handoff" / "evidence.jsonl"
    rows = _read_jsonl(handoff)
    assert all(row["source"] == "stage1_architecture" for row in rows)
    filtered = _texts_by_architecture(rows)
    # The BoW nuisance terms of the shared text-model rows survive the ablation.
    assert filtered == {name: original[name] for name in ("bow_nuisance", "tfidf_topics")}
    control = _texts_by_architecture(_read_jsonl(out / "full_control" / "handoff" /
                                                 "evidence.jsonl"))
    assert control == original

    # Stage 2's own checks accept the ablation inputs without an LLM.
    config = json.loads((directory / "stage2_from_artifacts.json").read_text())
    splits = _load_stage2_splits(provenance_path=Path(config["split_provenance_path"]),
                                 dataset_rows=ROWS, outer_fold_ids=[1, 2])
    _verify_stage2_splits_against_handoff(splits, handoff_path=handoff, outer_fold_ids=[1, 2])
    parsed = plain_stage2_config_from_mapping(config["stage2"], default_workers=8)
    compiled = compile_stage2_handoff_evidence(
        rows, handoff_path=handoff, max_cards_per_outer_fold=16,
        required_architectures=parsed.required_architectures,
        included_architectures=parsed.included_architectures,
    )
    architectures = {name for card in compiled.packets
                     for name in card["content"]["source_architectures"]}
    assert architectures == {"bow_nuisance", "tfidf_topics"}

    # The projected control compiles to exactly the cards of the original handoff.
    def cards(path: Path):
        compiled = compile_stage2_handoff_evidence(
            _read_jsonl(path), handoff_path=path, max_cards_per_outer_fold=16)
        return {
            fold: sorted(
                (card["card_id"], tuple(card["source_architectures"]),
                 json.dumps(card["representative_evidence"], sort_keys=True),
                 json.dumps(card["support"], sort_keys=True))
                for card in values
            )
            for fold, values in compiled.cards_by_outer_fold.items()
        }

    assert cards(out / "full_control" / "handoff" / "evidence.jsonl") == cards(
        run / "handoff" / "evidence.jsonl")


def test_prepare_uses_the_dated_notes_cohort_and_converts_csv(tmp_path: Path):
    run = _fake_run(tmp_path / "run", explicit=True)
    cohort = run / "input" / "analysis_cohort.parquet"
    cohort.parent.mkdir(parents=True)
    pd.read_parquet(run / "cohort.parquet").to_parquet(cohort, index=False)
    prepare_ablations(run_dir=run, out_dir=tmp_path / "notes", architectures=["tfidf_topics"])
    config = json.loads(
        (tmp_path / "notes" / "without_tfidf_topics" / "stage2_from_artifacts.json").read_text()
    )
    assert config["dataset"] == str(cohort.resolve())
    manifest = json.loads(
        (tmp_path / "notes" / "without_tfidf_topics" / "ablation.json").read_text()
    )
    assert manifest["source_dataset"] == str(cohort.resolve())

    csv_run = _fake_run(tmp_path / "csv_run", explicit=True)
    csv_path = csv_run / "cohort.csv"
    pd.read_parquet(csv_run / "cohort.parquet").to_csv(csv_path, index=False)
    run_config = json.loads((csv_run / "run_config.json").read_text())
    _write_json(csv_run / "run_config.json", {**run_config, "dataset": str(csv_path)})
    index = prepare_ablations(run_dir=csv_run, out_dir=tmp_path / "csv",
                              architectures=["tfidf_topics"])
    converted = pd.read_parquet(index["dataset"]["path"])
    # Only the four declared columns, so no oracle column reaches Stage 2.
    assert list(converted.columns) == [
        "patient_id", "clinical_text", "treatment_indicator", "outcome_indicator",
    ]
    assert len(converted) == ROWS


def test_prepare_skips_empty_ablations_and_rejects_foreign_architectures(tmp_path, capsys):
    run = _fake_run(tmp_path / "run", explicit=True, architectures=("tfidf_topics",),
                    sources=("tfidf",))
    index = prepare_ablations(run_dir=run, out_dir=tmp_path / "single")
    assert index["ablations"] == []
    assert index["skipped"][0]["name"] == "without_tfidf_topics"
    assert "empty handoff" in capsys.readouterr().out
    assert not (tmp_path / "single" / "without_tfidf_topics").exists()
    with pytest.raises(ValueError, match="not in this run's selection"):
        prepare_ablations(run_dir=run, out_dir=tmp_path / "other", architectures="htr_neural")
    with pytest.raises(ValueError, match="evaluation"):
        prepare_ablations(run_dir=run, out_dir=run / "stage2" / "ablations")


def test_stage2_template_replaces_managed_pools_with_endpoints(tmp_path: Path):
    managed = json.loads(json.dumps(redact_api_keys(asdict(plain_stage2_config_from_mapping(
        {"model": "primary-model", "vllm": {"gpus": ["cuda:0"]},
         "extraction_llm": {"model": "small-model", "vllm": {"gpus": ["cuda:1"]}}},
        default_workers=8,
    )))))
    run = _fake_run(tmp_path / "run", explicit=True, stage2=managed)
    with pytest.raises(ValueError, match="managed vLLM"):
        prepare_ablations(run_dir=run, out_dir=tmp_path / "no_template")
    template = tmp_path / "template.json"
    _write_json(template, {
        "stage2": {"endpoint": "http://primary.test/v1", "workers": 2,
                   "extraction_llm": {"endpoint": "http://extract.test/v1"}},
        "tokenizer_cache_dir": "/cache/tokenizers",
    })
    prepare_ablations(run_dir=run, out_dir=tmp_path / "templated",
                      architectures="bow_nuisance", stage2_template=template)
    config = json.loads(
        (tmp_path / "templated" / "without_bow_nuisance" / "stage2_from_artifacts.json")
        .read_text()
    )
    assert config["stage2"]["vllm"] is None and config["stage2"]["workers"] == 2
    assert config["stage2"]["extraction_llm"]["vllm"] is None
    assert config["stage2"]["model"] == "primary-model"
    assert config["tokenizer_cache_dir"] == "/cache/tokenizers"
    bad = tmp_path / "bad.json"
    _write_json(bad, {"endpoint": "http://primary.test/v1"})
    with pytest.raises(ValueError, match="under 'stage2'"):
        prepare_ablations(run_dir=run, out_dir=tmp_path / "bad", stage2_template=bad)


# ---------------------------------------------------------------------------
# compare


def _variable(name: str, description: str, roles, feature_id: str) -> dict:
    return {
        "feature_id": feature_id,
        "name": name,
        "description": description,
        "measurement_definition": f"Record the pretreatment {description.lower()}",
        "value_type": "categorical",
        "roles": list(roles),
    }


def _stage2_result(stage2: Path, folds: dict[int, list[dict]], *, shift: float) -> None:
    rng = np.random.default_rng(int(shift * 100) + 5)
    frames = []
    for outer, features in folds.items():
        _write_json(stage2 / f"outer_{outer:03d}" / "final_definitions.json",
                    {"features": features})
        rows = HELDOUT[outer]
        treatment = np.array([i % 2 for i in rows])
        mu0 = np.full(len(rows), 0.4)
        mu1 = mu0 + 0.1 + shift
        cate = rng.normal(0.1 + shift, 0.05, len(rows))
        outcome = rng.binomial(1, np.where(treatment == 1, mu1, mu0))
        frames.append(pd.DataFrame({
            "_oci_row_id": rows, "outer_fold": outer, "treatment": treatment,
            "outcome": outcome, "propensity": 0.5, "mu0": mu0, "mu1": mu1,
            "aipw_score": (mu1 - mu0) + np.where(treatment == 1, (outcome - mu1) / 0.5,
                                                 -(outcome - mu0) / 0.5),
            "estimated_cate": cate,
        }))
    predictions = pd.concat(frames, ignore_index=True)
    co_primary = co_primary_estimands(predictions, propensity_clip=0.01)
    _write_json(stage2 / "causal_estimate.json", {
        "ate": co_primary["ate"]["estimate"], "co_primary_estimands": co_primary,
    })
    per_fold = {
        str(fold): best_linear_predictor(part["aipw_score"], part["estimated_cate"])
        for fold, part in predictions.groupby("outer_fold")
    }
    _write_json(stage2 / "heterogeneity_diagnostics.json",
                {"blp": {"per_fold": per_fold, "pooled": pool_blp(per_fold)}})


FULL_FOLDS = {
    1: [
        _variable("pdl1_expression", "PD-L1 tumor proportion score category",
                  ["effect_modifier"], "f1"),
        _variable("brain_metastases", "Brain metastases at diagnosis", ["effect_modifier"],
                  "f2"),
        _variable("ecog_performance_status", "ECOG performance status", ["confounder"], "f3"),
    ],
    2: [_variable("ecog_performance_status", "ECOG performance status", ["confounder"], "g1")],
}
ABLATED_FOLDS = {
    1: [
        _variable("pd_l1_expression", "PD-L1 tumor proportion score category",
                  ["effect_modifier"], "a1"),
        _variable("egfr_mutation", "EGFR mutation status", ["effect_modifier"], "a2"),
        _variable("ecog_performance_status", "ECOG performance status", ["confounder"], "a3"),
    ],
    2: [_variable("ecog_performance_status", "ECOG performance status", ["confounder"], "b1")],
}


def _compare_fixture(tmp_path: Path) -> tuple[Path, Path]:
    run = tmp_path / "run"
    _stage2_result(run / "stage2", FULL_FOLDS, shift=0.0)
    abl = tmp_path / "ablations"
    entries = []
    for name, removed in (("without_htr_neural", "htr_neural"),
                          ("without_tfidf_topics", "tfidf_topics"),
                          ("without_bow_nuisance", "bow_nuisance")):
        directory = abl / name
        _write_json(directory / "ablation.json", {"removed_architecture": removed})
        entries.append({"name": name, "directory": str(directory),
                        "removed_architecture": removed})
    _stage2_result(abl / "without_htr_neural" / "stage2", ABLATED_FOLDS, shift=0.05)
    _write_json(abl / "without_htr_neural" / "stage2_report" / "status.json",
                {"phase": "complete"})
    _write_json(abl / "without_bow_nuisance" / "stage2_report" / "status.json",
                {"phase": "failed"})
    _write_json(abl / "ablations.json", {"ablations": entries})
    return run, abl


def test_compare_matches_concepts_and_lists_pending_ablations(tmp_path: Path):
    run, abl = _compare_fixture(tmp_path)
    result = compare_ablations(run_dir=run, ablation_dir=abl)
    assert result["complete"] == ["without_htr_neural"]
    assert result["pending"] == ["without_tfidf_topics", "without_bow_nuisance"]
    output = abl / "comparison"
    report = json.loads((output / "comparison.json").read_text())
    by_name = {item["name"]: item for item in report["ablations"]}
    assert by_name["without_tfidf_topics"]["detail"] == "not_started"
    assert by_name["without_bow_nuisance"]["detail"] == "failed"
    fold = by_name["without_htr_neural"]["per_fold"]["1"]
    modifiers = fold["selected_modifiers"]
    assert [(m["full"], m["ablation"]) for m in modifiers["matched"]] == [
        ("pdl1_expression", "pd_l1_expression")]
    assert modifiers["lost"] == ["brain_metastases"]
    assert modifiers["gained"] == ["egfr_mutation"]
    assert modifiers["jaccard"] == pytest.approx(1 / 3)
    assert fold["nuisance_set"]["jaccard"] == pytest.approx(1.0)
    # Fold 2 selected no modifiers on either side: identical empty sets.
    assert by_name["without_htr_neural"]["per_fold"]["2"]["selected_modifiers"]["jaccard"] == 1.0

    full = json.loads((run / "stage2" / "causal_estimate.json").read_text())
    ablated = json.loads(
        (abl / "without_htr_neural" / "stage2" / "causal_estimate.json").read_text())
    differences = by_name["without_htr_neural"]["estimate_differences_ablation_minus_full"]
    assert differences["ate"] == pytest.approx(
        ablated["co_primary_estimands"]["ate"]["estimate"]
        - full["co_primary_estimands"]["ate"]["estimate"])
    estimates = by_name["without_htr_neural"]["estimates"]
    assert estimates["ato"]["ci_low"] == pytest.approx(
        ablated["co_primary_estimands"]["ato"]["confidence_interval_95"][0])
    heterogeneity = json.loads(
        (abl / "without_htr_neural" / "stage2" / "heterogeneity_diagnostics.json").read_text())
    primary = estimates["primary_heterogeneity_test"]
    assert primary["beta2"] == pytest.approx(heterogeneity["blp"]["pooled"]["beta2_heterogeneity"])
    assert primary["p_value_one_sided"] == pytest.approx(
        heterogeneity["blp"]["pooled"]["beta2_p_value_one_sided_greater_than_0"])

    summary = pd.read_csv(output / "summary.csv", keep_default_na=False)
    assert list(summary["ablation"]) == [
        "full_run", "without_htr_neural", "without_tfidf_topics", "without_bow_nuisance"]
    row = summary[summary["ablation"] == "without_htr_neural"].iloc[0]
    assert int(row["modifiers_lost"]) == 1 and int(row["modifiers_gained"]) == 1
    folds = pd.read_csv(output / "folds.csv", keep_default_na=False)
    assert folds.loc[folds["outer_fold"] == 1, "modifiers_lost"].iloc[0] == "brain_metastases"
    markdown = (output / "summary.md").read_text()
    assert "pending (failed)" in markdown and "`without_tfidf_topics`" in markdown


def test_compare_with_oracle_reports_recovery_for_each_side(tmp_path: Path):
    run, abl = _compare_fixture(tmp_path)
    truth = pd.DataFrame({"true_pdl1": [f"TPS {i % 3}" for i in range(ROWS)],
                          "true_brain": [f"B{i % 2}" for i in range(ROWS)]})
    truth.to_parquet(tmp_path / "dataset.parquet", index=False)
    _write_json(run / "run_config.json", {"dataset": str(tmp_path / "dataset.parquet")})
    for root, folds in ((run, FULL_FOLDS), (abl / "without_htr_neural", ABLATED_FOLDS)):
        for outer, features in folds.items():
            rows = HELDOUT[outer]
            frame = pd.DataFrame({"_oci_row_id": rows})
            for feature in features:
                if "pd" in feature["name"]:
                    frame[feature["name"]] = truth["true_pdl1"].iloc[rows].to_numpy()
                elif feature["name"] == "brain_metastases":
                    frame[feature["name"]] = truth["true_brain"].iloc[rows].to_numpy()
                else:
                    frame[feature["name"]] = [f"level {i % 5}" for i in rows]
            estimation = root / "stage2" / f"outer_{outer:03d}" / "estimation"
            estimation.mkdir(parents=True, exist_ok=True)
            frame.to_csv(estimation / "nuisance_candidate_heldout.csv", index=False)
            _write_json(estimation / "nuisance_candidate_definitions.json",
                        {"features": features})
    compare_ablations(run_dir=run, ablation_dir=abl, oracle=True)
    report = json.loads((abl / "comparison" / "comparison.json").read_text())
    assert report["oracle"]["stage2_artifacts_hashed_before_oracle_load"] is True
    assert report["full_run"]["oracle_recovery"]["per_fold"]["1"]["selected_modifiers"] == [
        "pdl1", "brain"]
    item = next(entry for entry in report["ablations"] if entry["name"] == "without_htr_neural")
    assert item["oracle_recovery"]["per_fold"]["1"]["selected_modifiers"] == ["pdl1"]
    summary = pd.read_csv(abl / "comparison" / "summary.csv", keep_default_na=False)
    row = summary[summary["ablation"] == "without_htr_neural"].iloc[0]
    assert row["oracle_lost_selected_modifiers"] == "brain"


def test_compare_concepts_uses_the_cross_fold_matching_rule():
    left = [_variable("pdl1_expression", "PD-L1 tumor proportion score category", [], "x")]
    right = [_variable("pd_l1_expression", "PD-L1 tumor proportion score category", [], "y"),
             _variable("liver_metastases", "Liver metastases at diagnosis", [], "z")]
    result = compare_concepts(left, right)
    assert result["similarity_threshold"] == 0.7
    assert len(result["matched"]) == 1 and result["gained"] == ["liver_metastases"]
    assert compare_concepts([], [])["jaccard"] == 1.0


def test_ablation_script_runs_from_the_repository(tmp_path: Path):
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "ablate_stage1_architectures.py"), "--help"],
        capture_output=True, text=True, check=True, cwd=tmp_path,
    )
    assert "prepare" in completed.stdout and "compare" in completed.stdout
