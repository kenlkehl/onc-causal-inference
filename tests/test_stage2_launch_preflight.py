from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from dataclasses import replace
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from oci.inference import research_all_evidence_workflow as workflow
from oci.inference import stage2_preflight as preflight
from oci.inference.tfidf_topic_stage1 import tfidf_topic_dataset_fingerprints
from oci.inference.tfidf_topic_discovery import row_set_fingerprint
from tests.test_research_all_evidence_workflow import _inputs, _completed_stage2_reselection_fixture

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "saved_launcher", ROOT / "scripts/launch_saved_stage2.py"
)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, default=str))


def tree(root):
    return {
        str(p.relative_to(root)): (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
        for p in root.rglob("*")
        if p.is_file()
    }


def saved_fixture(tmp_path, managed=False):
    raw, _ = _inputs(tmp_path, components=())
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
    data = pd.read_parquet(config.dataset)
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
            "dataset": str(config.dataset),
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


@pytest.mark.parametrize("managed", [False, True])
def test_preflight_and_saved_command_preserve_science_and_do_not_run_stage1(
    tmp_path, monkeypatch, managed
):
    config = saved_fixture(tmp_path, managed)
    before = tree(tmp_path)
    args = launcher.command(
        config.dataset,
        str(config.output_dir),
        {
            "OCI_RUN_CONFIG": str(config.output_dir / "run_config.json"),
            "STAGE2_SELECTION_MODE": "independent_tasks",
            "STAGE2_ONLY": "1",
        },
    )
    parsed = workflow.build_parser().parse_args(args)
    raw, directory = workflow._raw_config_from_args(parsed)
    resolved = workflow.compile_config(raw, config_dir=directory)
    assert resolved.seed == 7 and resolved.outer_folds == 2 and resolved.inner_folds == 2
    assert resolved.stage2.model == "preserved-primary"
    assert resolved.stage2.statistical_selection.selection_mode == "independent_tasks"
    assert resolved.components == ("stage2",)
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
    [
        ("handoff", "handoff"),
        ("split", "split"),
        ("rows", "fingerprint"),
        ("model", "model ID"),
        ("seed", "seed"),
        ("source", "source"),
        ("columns", "columns"),
    ],
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


def test_saved_command_rejects_cohort_and_output_mismatch(tmp_path):
    config = saved_fixture(tmp_path)
    env = {"OCI_RUN_CONFIG": str(config.output_dir / "run_config.json")}
    with pytest.raises(ValueError, match="cohort"):
        launcher.command(tmp_path / "wrong.parquet", "", env)
    with pytest.raises(ValueError, match="output"):
        launcher.command(config.dataset, str(tmp_path / "wrong"), env)
    with pytest.raises(ValueError, match="runtime override"):
        launcher.command(config.dataset, "", {**env, "STAGE2_CONSOLIDATION_MAX_ROUNDS": "1"})


@pytest.mark.parametrize("wrapper", ["run_one_conf_one_mod.sh", "run_five_conf_five_mod.sh"])
def test_wrappers_delegate_saved_config_without_injecting_defaults(tmp_path, wrapper):
    fake = tmp_path / "python"
    capture = tmp_path / "capture.json"
    fake.write_text(
        "#!/usr/bin/env python3\nimport os,json,sys\n"
        'open(os.environ["CAPTURE"],"w").write(json.dumps({"args":sys.argv[1:],'
        '"env":{k:v for k,v in os.environ.items() if k.startswith("STAGE2_")}}))\n'
    )
    fake.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("STAGE2_", "OCI_"))}
    env.update(
        OCI_PYTHON=str(fake),
        OCI_RUN_CONFIG=str(tmp_path / "saved.json"),
        STAGE2_ONLY="1",
        STAGE2_SELECTION_MODE="independent_tasks",
        OCI_PREFLIGHT_ONLY="1",
        CAPTURE=str(capture),
    )
    subprocess.run(["bash", str(ROOT / wrapper)], env=env, check=True, capture_output=True)
    result = json.loads(capture.read_text())
    assert result["args"][0].endswith("launch_saved_stage2.py")
    assert wrapper.split("_conf_")[0].removeprefix("run_") in result["args"][1]
    assert set(result["env"]) == {"STAGE2_ONLY", "STAGE2_SELECTION_MODE"}


@pytest.mark.parametrize(
    "env",
    [
        {"STAGE2_ONLY": "true"},
        {"STAGE2_RESELECT": "yes"},
        {"OCI_PREFLIGHT_ONLY": "2"},
        {"STAGE2_SELECTION_MODE": "typo"},
        {"STAGE2_RESELECT": "1"},
        {"STAGE2_ONLY": "1", "STAGE2_ENDPOINT": ""},
        {"STAGE2_SELECTION_MODE": "independent_tasks", "STAGE2_ENDPOINT": ""},
        {"OCI_PREFLIGHT_ONLY": "1"},
    ],
)
def test_invalid_controls_fail_before_python(tmp_path, env):
    process = subprocess.run(
        ["bash", str(ROOT / "run_one_conf_one_mod.sh"), str(tmp_path)],
        env={**os.environ, "OCI_PYTHON": "/nonexistent", **env},
        text=True,
        capture_output=True,
    )
    assert process.returncode != 0
    assert "No such file" not in process.stderr


def test_mode_switch_cannot_resume_legacy_completion(tmp_path):
    config = saved_fixture(tmp_path)
    write(config.output_dir / "stage2/config.json", {"statistical_selection": {}})
    write(config.output_dir / "stage2/complete.json", {"status": "complete"})
    preflight.validate_selection_resume(config)
    new = replace(
        config,
        stage2=replace(
            config.stage2,
            statistical_selection=replace(
                config.stage2.statistical_selection, selection_mode="independent_tasks"
            ),
        ),
    )
    before = tree(tmp_path)
    with pytest.raises(RuntimeError, match="selection mode changed"):
        workflow.ResearchAllEvidenceWorkflow(new).run()
    assert tree(tmp_path) == before


def test_guarded_dry_run_and_independent_migration_preserve_measurements(tmp_path):
    config = _completed_stage2_reselection_fixture(tmp_path)
    config = replace(
        config,
        stage2=replace(
            config.stage2,
            statistical_selection=replace(
                config.stage2.statistical_selection, selection_mode="independent_tasks"
            ),
        ),
    )
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


def test_interrupted_reselection_resumes_only_with_migration_flag(tmp_path, monkeypatch):
    config = _completed_stage2_reselection_fixture(tmp_path)
    original = workflow._resume_reselection_archive_moves

    def interrupted(*, stage2_dir, state):
        name = state["planned_artifacts"][0]
        target = stage2_dir / state["archive_path"] / "artifacts" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        (stage2_dir / name).rename(target)
        raise InterruptedError("test interruption")

    monkeypatch.setattr(workflow, "_resume_reselection_archive_moves", interrupted)
    with pytest.raises(InterruptedError):
        workflow.prepare_stage2_reselection(config=config)
    before = tree(tmp_path)
    assert workflow.prepare_stage2_reselection(config=config, dry_run=True)["status"] == "preparing"
    assert tree(tmp_path) == before
    with pytest.raises(RuntimeError, match="interrupted"):
        preflight.validate_selection_resume(config)
    monkeypatch.setattr(workflow, "_resume_reselection_archive_moves", original)
    state = workflow.prepare_stage2_reselection(config=config)
    assert state["status"] == "prepared"
    assert len(list((config.output_dir / "stage2/reselection_archives").iterdir())) == 1


def test_archive_restore_copies_and_refuses_existing_destination(tmp_path, monkeypatch):
    config = _completed_stage2_reselection_fixture(tmp_path)
    source = config.output_dir / "stage2_pre_roles_refactor"
    (config.output_dir / "stage2").rename(source)
    monkeypatch.setattr(preflight, "preflight_stage2", lambda *a, **k: None)
    before = tree(source)
    destination = preflight.restore_stage2_archive(config, source)
    assert tree(source) == before
    src = source / "config.json"
    dst = destination / "config.json"
    assert src.stat().st_ino != dst.stat().st_ino
    dst.write_text("changed")
    assert tree(source) == before
    with pytest.raises(RuntimeError, match="already exists"):
        preflight.restore_stage2_archive(config, source)


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


def test_normal_resume_finalizes_existing_migration_without_rearchival(tmp_path, monkeypatch):
    config = _completed_stage2_reselection_fixture(tmp_path)
    state = workflow.prepare_stage2_reselection(config=config)
    config_path = tmp_path / "resume.json"
    write(config_path, config.as_dict())

    def completed_run(self):
        write(
            config.output_dir / "stage2/complete.json",
            {"status": "complete", "phase": "causal_estimation"},
        )
        return {"status": "complete"}

    monkeypatch.setattr(workflow.ResearchAllEvidenceWorkflow, "run", completed_run)
    monkeypatch.setattr(workflow, "_configure_logging", lambda *a: None)
    for _ in range(2):
        assert workflow.main(["--config", str(config_path), "--stage2-only"]) == 0
    persisted = json.loads((config.output_dir / "stage2/reselection_state.json").read_text())
    assert persisted["status"] == "complete"
    assert persisted["reselection_id"] == state["reselection_id"]
    assert len(list((config.output_dir / "stage2/reselection_archives").iterdir())) == 1


def test_interrupted_archive_copy_does_not_publish_partial_stage2(tmp_path, monkeypatch):
    config = _completed_stage2_reselection_fixture(tmp_path)
    source = config.output_dir / "stage2_pre_roles_refactor"
    (config.output_dir / "stage2").rename(source)
    monkeypatch.setattr(preflight, "preflight_stage2", lambda *a, **k: None)
    original = preflight.shutil.copytree

    def interrupted(src, dst):
        dst.mkdir()
        (dst / "partial").write_text("incomplete")
        raise InterruptedError("copy interrupted")

    monkeypatch.setattr(preflight.shutil, "copytree", interrupted)
    before = tree(source)
    with pytest.raises(InterruptedError):
        preflight.restore_stage2_archive(config, source)
    assert not (config.output_dir / "stage2").exists()
    assert tree(source) == before
    monkeypatch.setattr(preflight.shutil, "copytree", original)
    restored = preflight.restore_stage2_archive(config, source)
    assert (restored / "complete.json").is_file()
    assert not (restored / "partial").exists()
    assert tree(source) == before
