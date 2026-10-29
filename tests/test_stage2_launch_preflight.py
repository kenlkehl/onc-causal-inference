from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pandas as pd
import pytest

from oci.inference import research_all_evidence_workflow as workflow
from oci.inference import stage2_preflight as preflight
from oci.inference.tfidf_topic_stage1 import tfidf_topic_dataset_fingerprints
from oci.inference.tfidf_topic_discovery import row_set_fingerprint
from tests.test_research_all_evidence_workflow import _inputs, _completed_stage2_reselection_fixture


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, default=str))


def tree(root):
    return {
        str(p.relative_to(root)): (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
        for p in root.rglob("*")
        if p.is_file()
    }


def saved_fixture(tmp_path, managed=False, prepare_raw=None):
    raw, _ = _inputs(tmp_path, components=())
    if prepare_raw is not None:
        prepare_raw(raw, tmp_path)
    raw["run"]["mode"] = "stage2"
    raw["stage2"] = {
        "endpoint": "http://primary.test/v1",
        "model": "preserved-primary",
        "extraction_llm": {"endpoint": "http://extract.test/v1", "model": "preserved-extractor"},
    }
    if managed:
        raw["stage2"]["endpoint"] = ""
        raw["stage2"]["vllm"] = {"gpus": ["cuda:1"], "server_count": 1}
    config = workflow.compile_config(raw, config_dir=tmp_path)
    root = config.output_dir
    write(root / "run_config.json", config.as_dict())
    applied = workflow._load_stage1_template(config)
    write(root / "resolved_stage1_model_config.json", applied)
    data = workflow._load_analysis_dataset(config, materialize=True)
    identity = tfidf_topic_dataset_fingerprints(
        data, workflow.ExperimentConfig.from_dict({"applied_inference": applied}).applied_inference
    )
    write(
        root / "components/tfidf/manifest.json",
        {
            "dataset_" + key: identity[key]
            for key in ["content_fingerprint", "ordered_row_fingerprint"]
        },
    )
    splits, evidence = [], []
    for fold, heldout in enumerate(([0, 1, 2], [3, 4, 5]), 1):
        fit = sorted(set(range(6)) - set(heldout))
        inner = []
        for i, test in enumerate((fit[:1], fit[1:]), 1):
            inner.append(
                {
                    "inner_fold": i,
                    "fit_row_ids": sorted(set(fit) - set(test)),
                    "heldout_row_ids": test,
                }
            )
        splits.append(
            {
                "outer_fold": fold,
                "fit_row_ids": fit,
                "heldout_row_ids": heldout,
                "inner_splits": inner,
            }
        )
        for i, item in enumerate([splits[-1], *inner]):
            row = {
                "outer_fold": fold,
                "fold_key": fold * 10 + i,
                "inner_fold": i or None,
                "scope": "full_outer_train" if not i else "candidate_selection_inner_fit",
            }
            for side in ["fit", "heldout"]:
                row[side + "_row_ids"] = item[side + "_row_ids"]
                row[side + "_row_fingerprint"] = row_set_fingerprint(item[side + "_row_ids"])
            row["discovery"] = dict(row)
            evidence.append(row)
    for name, rows in [("split_provenance.jsonl", splits), ("evidence.jsonl", evidence)]:
        (root / "components/tfidf" / name).write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    write(root / "components/tfidf/complete.json", {"status": "complete", "artifacts": []})
    write(root / "handoff/complete.json", {"status": "complete"})
    write(
        root / "handoff/index.json",
        {
            "dataset": str(config.analysis_dataset),
            "columns": {
                "unit_id": config.unit_id_column,
                "text": config.text_column,
                "treatment": config.treatment_column,
                "outcome": config.outcome_column,
            },
            "sources": {"tfidf": "../components/tfidf/evidence.jsonl"},
        },
    )
    (root / "handoff/evidence.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "source": "tfidf",
                    "outer_fold": r["outer_fold"],
                    "inner_fold": r["inner_fold"],
                    "scope": r["scope"],
                    "evidence": r,
                }
            )
            + "\n"
            for r in evidence
        )
    )
    return config


def _saved_stage2_config(config, *extra):
    """Compile the documented saved-run Stage 2-only command line."""
    args = workflow.build_parser().parse_args(
        ["--config", str(config.output_dir / "run_config.json"), "--stage2-only", *extra]
    )
    raw, directory = workflow._raw_config_from_args(args)
    return workflow.compile_config(raw, config_dir=directory)


@pytest.mark.parametrize("managed", [False])
def test_preflight_and_saved_command_preserve_science_and_do_not_run_stage1(
    tmp_path, monkeypatch, managed
):
    config = saved_fixture(tmp_path, managed)
    before = tree(tmp_path)
    resolved = _saved_stage2_config(config)
    assert resolved.seed == 7 and resolved.outer_folds == 2 and resolved.inner_folds == 2
    assert resolved.stage2.model == "preserved-primary"
    assert resolved.components == ("stage2",)
    assert resolved.stage2 == config.stage2
    result = preflight.preflight_stage2(resolved)
    assert result["primary_serving"] == ("managed" if managed else "external")
    assert tree(tmp_path) == before
    for name in workflow.STAGE1_COMPONENT_ORDER:
        monkeypatch.setitem(
            workflow.DEFAULT_COMPONENT_RUNNERS,
            name,
            lambda *a, **k: pytest.fail("Stage 1 must never run"),
        )
    monkeypatch.setattr(
        workflow.ResearchAllEvidenceWorkflow,
        "_resolved_context",
        lambda *a: pytest.fail("must not rebuild Stage 1 context"),
    )
    monkeypatch.setitem(workflow.DEFAULT_COMPONENT_RUNNERS, "stage2", lambda *a: {"phase": "test"})
    workflow.ResearchAllEvidenceWorkflow(resolved).run()
    assert not (config.output_dir / "resolved_neural_query_config.json").exists()


@pytest.mark.parametrize(
    "damage,match",
    [("split", "split"), ("model", "model ID")],
)
def test_preflight_refuses_incompatible_inputs_without_writes(tmp_path, damage, match):
    config = saved_fixture(tmp_path)
    root = config.output_dir
    if damage == "handoff":
        (root / "handoff/complete.json").unlink()
    if damage == "split":
        (root / "components/tfidf/split_provenance.jsonl").write_text("")
    if damage == "rows":
        pd.read_parquet(config.dataset).iloc[::-1].to_parquet(config.dataset, index=False)
    if damage == "model":
        write(root / "stage2/model_identity.json", {"primary": {"selected_model": "other"}})
    if damage == "seed":
        config = replace(config, seed=99)
    if damage == "source":
        (root / "components/tfidf/evidence.jsonl").unlink()
    if damage == "columns":
        index = json.loads((root / "handoff/index.json").read_text())
        index["columns"]["text"] = "wrong"
        write(root / "handoff/index.json", index)
    before = tree(tmp_path)
    with pytest.raises((RuntimeError, FileNotFoundError), match=match):
        preflight.preflight_stage2(config)
    assert tree(tmp_path) == before


def test_guarded_dry_run_and_migration_preserve_measurements(tmp_path):
    config = _completed_stage2_reselection_fixture(tmp_path)
    before = tree(tmp_path)
    planned = workflow.prepare_stage2_reselection(config=config, dry_run=True)
    assert planned["status"] == "validated" and tree(tmp_path) == before
    matrix = config.output_dir / "stage2/outer_001/extraction/all_candidates_fit/extracted.csv"
    frozen = matrix.read_bytes()
    state = workflow.prepare_stage2_reselection(config=config)
    assert matrix.read_bytes() == frozen
    before = tree(tmp_path)
    workflow.prepare_stage2_reselection(config=config, dry_run=True)
    assert tree(tmp_path) == before
    assert (
        config.output_dir / "stage2" / state["archive_path"] / "artifacts/causal_estimate.json"
    ).is_file()


def test_core_cli_preflight_returns_before_any_logging_or_execution(tmp_path, monkeypatch):
    config = saved_fixture(tmp_path)
    before = tree(tmp_path)
    monkeypatch.setattr(
        workflow, "_configure_logging", lambda *a: pytest.fail("preflight logging writes")
    )
    monkeypatch.setattr(
        workflow.ResearchAllEvidenceWorkflow, "run", lambda *a: pytest.fail("preflight ran models")
    )
    assert (
        workflow.main(
            [
                "--config",
                str(config.output_dir / "run_config.json"),
                "--stage2-only",
                "--preflight-only",
            ]
        )
        == 0
    )
    assert tree(tmp_path) == before
