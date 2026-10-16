"""Benchmark harness (scripts/benchmark.py, oci/evaluation/benchmark.py).

The generator's re-equation module is mocked: the tests program against its
contract (``reequation_cohort(source_dir, scenario, *, size, seed, output_dir)``
writing ``dataset.parquet``, ``metadata.json``, ``reequation_manifest.json`` and
``generation_config.json``). Fake run directories are written with the current
Stage 2 writers (``co_primary_estimands``, ``build_heterogeneity_diagnostics``)
and the field names of the attribution report's fake runs; no LLM is involved.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oci.evaluation import benchmark
from oci.evaluation.attribution import FrozenArtifactReader, OracleConcept, file_sha256
from oci.evaluation.benchmark import (
    SeedGuardError,
    ate_metrics,
    ato_metrics,
    cate_metrics,
    gates_coverage,
    leak_rows,
    load_registry,
    load_seed_registry,
    resolve_seeds,
    run_status,
    scenario_rows,
    selection_metrics,
    short_nuisances,
)

REPO = Path(__file__).resolve().parents[1]
DEV, CONF = [11, 12], [91, 92]
FOLDS = (1, 2)
SEED_BASE = 50_000


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _expit(x):
    return 1.0 / (1.0 + np.exp(-x))


BASE_DGP = {
    "name": "baseline",
    "roles": {"confounder": 2, "effect_modifier": 1},
    "role_assignment": "random",
    "modifier_main_effects": False,
    "nonlinearity": "none",
    "unmeasured_confounders": 0,
}


def _registry(tmp_path: Path) -> Path:
    path = tmp_path / "scenarios.json"
    _write_json(path, {
        "schema_version": "oci_benchmark_registry_v1",
        "scenarios": [
            {
                "id": "S0", "family": "S0", "label": "Baseline", "factor": "baseline",
                "materialization": "reequation", "sizes": [40, 80],
                "source": {"kind": "reequation_source", "minimum_patients": 80},
                "dgp": BASE_DGP,
                "run_variants": [{"name": "default"}],
            },
            {
                "id": "S4a", "family": "S4", "label": "U", "factor": "U",
                "materialization": "reequation", "sizes": [40],
                "source": {"kind": "reequation_source", "minimum_patients": 40},
                "dgp": {**BASE_DGP, "name": "u", "unmeasured_confounders": 1,
                        "unmeasured_strength": 0.5},
                "run_variants": [{"name": "default"}],
            },
            {
                "id": "S6", "family": "S6", "label": "Leak", "factor": "leak",
                "materialization": "subsample_of_generated", "sizes": [30, 40],
                "source": {"kind": "generated_corpus", "minimum_patients": 40},
                "generation": {"dataset_size": 40, "post_treatment_notes": {"rate": 1.0}},
                "dgp": None,
                "run_variants": [
                    {"name": "text_column",
                     "columns": {"text": "clinical_text_with_post_treatment"}},
                    {"name": "dated_notes", "columns": {"text": "dated_notes_text"},
                     "notes": {"path": "notes.parquet",
                               "columns": {"unit_id": "patient_id", "date": "note_date",
                                           "text": "note_text", "note_type": "note_type"},
                               "index_date_column": "index_date"}},
                ],
            },
        ],
    })
    return path


def _seeds(tmp_path: Path) -> Path:
    path = tmp_path / "seeds.json"
    _write_json(path, {"schema_version": "oci_benchmark_seeds_v1", "rule": "tune on development",
                       "development": DEV, "confirmatory": CONF})
    return path


METADATA_FEATURES = [
    {"name": "age", "type": "continuous", "roles": ["confounder"]},
    {"name": "ecog", "type": "categorical", "categories": ["0", "1", "2"], "roles": ["confounder"]},
    {"name": "pdl1", "type": "categorical", "categories": ["low", "high"],
     "roles": ["effect_modifier"]},
    {"name": "smoker", "type": "categorical", "categories": ["never", "former", "current"],
     "roles": ["instrument"]},
    {"name": "albumin", "type": "continuous", "roles": ["prognostic"]},
    {"name": "noise_marker", "type": "continuous", "roles": []},
]
GAMMA_T, GAMMA_Y = 0.9, 0.7


def _cohort(n: int, seed: int, *, unmeasured: bool = False, leak: bool = False):
    rng = np.random.default_rng(seed)
    age = rng.normal(65, 10, n)
    ecog = rng.choice(["0", "1", "2"], n)
    pdl1 = rng.choice(["low", "high"], n)
    smoker = rng.choice(["never", "former", "current"], n)
    albumin = rng.normal(3.8, 0.4, n)
    u = rng.normal(size=n) if unmeasured else np.zeros(n)
    lp_t = 0.04 * (age - 65) + 0.6 * (ecog == "2") + 0.5 * (smoker == "current") + GAMMA_T * u
    e = _expit(lp_t)
    t = rng.binomial(1, e)
    lp0 = -0.03 * (age - 65) - 0.5 * (ecog == "1") + 0.8 * (albumin - 3.8) + GAMMA_Y * u
    lp1 = lp0 + 0.3 + 0.9 * (pdl1 == "high")
    y0, y1 = _expit(lp0), _expit(lp1)
    y = rng.binomial(1, np.where(t == 1, y1, y0))
    frame = pd.DataFrame({
        "patient_id": np.arange(n), "clinical_text": ["pretreatment note"] * n,
        "treatment_indicator": t, "outcome_indicator": y,
        "true_treatment_prob": e, "true_outcome_prob": np.where(t == 1, y1, y0),
        "true_y0_prob": y0, "true_y1_prob": y1, "true_ite_prob": y1 - y0,
        "true_y0": y0, "true_y1": y1, "true_ite": y1 - y0,
        "true_age": age, "true_ecog": ecog, "true_pdl1": pdl1, "true_smoker": smoker,
        "true_albumin": albumin, "true_noise_marker": rng.normal(size=n),
    })
    metadata = {"config": {"outcome_type": "binary"}, "features": METADATA_FEATURES,
                "scenario": None, "unmeasured_confounders": []}
    if unmeasured:
        frame["true_u_1"] = u
        metadata["unmeasured_confounders"] = [
            {"name": "u_1", "coefficients": {"treatment": GAMMA_T, "outcome": GAMMA_Y}}
        ]
    if leak:
        frame["true_post_treatment_toxicity"] = rng.binomial(1, _expit(-0.5 + 1.5 * t))
        frame["clinical_text_with_post_treatment"] = [
            f"pretreatment note\n\npost note {value}" for value in t
        ]
        frame["index_date"] = "2024-03-01"
        metadata["post_treatment"] = {"events": ["toxicity"]}
    return frame, metadata


def _write_cohort(directory: Path, n: int, seed: int, **kwargs) -> Path:
    frame, metadata = _cohort(n, seed, **kwargs)
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(directory / "dataset.parquet", index=False)
    _write_json(directory / "metadata.json", metadata)
    _write_json(directory / "generation_config.json", {"outcome_type": "binary", "dataset_size": n})
    if kwargs.get("leak"):
        notes = pd.DataFrame({
            "patient_id": np.repeat(frame["patient_id"].to_numpy(), 2),
            "note_date": ["2024-01-15", "2024-04-01"] * n,
            "note_text": ["pretreatment note", "post note"] * n,
            "note_type": ["progress", "progress"] * n,
        })
        notes.to_parquet(directory / "notes.parquet", index=False)
    return directory


@pytest.fixture
def fake_reequation(monkeypatch):
    """The generator contract, recording its calls (Package G is not in this checkout)."""

    calls = []

    def reequation_cohort(source_dir, scenario, *, size, seed, output_dir):
        calls.append({"source_dir": Path(source_dir), "scenario": dict(scenario), "size": size,
                      "seed": seed, "output_dir": Path(output_dir)})
        source = pd.read_parquet(Path(source_dir) / "dataset.parquet")
        if size is not None and size > len(source):
            raise ValueError("size larger than the source")
        unmeasured = int(scenario.get("unmeasured_confounders") or 0) > 0
        frame, metadata = _cohort(len(source), seed, unmeasured=unmeasured)
        rows = np.sort(np.random.default_rng(seed).choice(len(source), size=size, replace=False))
        frame = frame.iloc[rows].reset_index(drop=True)
        frame["clinical_text"] = source["clinical_text"].iloc[rows].to_numpy()
        frame["patient_id"] = source["patient_id"].iloc[rows].to_numpy()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
        frame.to_parquet(output_dir / "dataset.parquet", index=False)
        _write_json(output_dir / "metadata.json", {**metadata, "scenario": dict(scenario)})
        _write_json(output_dir / "generation_config.json", {"outcome_type": "binary"})
        _write_json(output_dir / "reequation_manifest.json",
                    {"source": str(source_dir), "scenario": dict(scenario), "size": size,
                     "seed": seed, "code_version": "test"})
        return output_dir

    module = types.ModuleType("synthetic_data.reequation")
    module.reequation_cohort = reequation_cohort
    monkeypatch.setitem(sys.modules, "synthetic_data.reequation", module)
    return calls


# ---------------------------------------------------------------------------
# Registry and seeds


def test_bundled_registry_and_seeds_follow_the_design():
    registry = load_registry()
    ids = list(registry.scenarios)
    assert ids == ["S0", "S1", "S2", "S3", "S4a", "S4b", "S5", "S6"]
    for scenario in registry.scenarios.values():
        expected = {1000, 5000} | ({10000} if scenario.family in {"S0", "S4"} else set())
        assert set(scenario.sizes) == expected
        assert scenario.minimum_source_patients == max(scenario.sizes)
    assert [s.id for s in registry.select("S4")] == ["S4a", "S4b"]
    assert {s.id for s in registry.scenarios.values() if s.needs_generated_corpus} == {"S5", "S6"}
    base = registry.scenarios["S0"].dgp
    changed = {
        scenario.id: sorted(
            key for key in set(base) | set(scenario.dgp or {})
            if key != "name" and (scenario.dgp or {}).get(key) != base.get(key)
        )
        for scenario in registry.scenarios.values() if scenario.dgp is not None
    }
    assert changed == {
        "S0": [], "S1": ["roles"], "S2": ["modifier_main_effects", "roles"],
        "S3": ["nonlinearity"], "S4a": ["unmeasured_confounders", "unmeasured_strength"],
        "S4b": ["unmeasured_confounders"], "S5": [],
    }
    assert registry.scenarios["S1"].dgp["roles"] == {
        "confounder": 3, "effect_modifier": 3, "instrument": 2, "prognostic": 2}
    assert registry.scenarios["S4a"].unmeasured_strength == 0.5
    assert registry.scenarios["S4b"].unmeasured_strength == 1.0
    assert registry.scenarios["S5"].generation["documentation_mechanism"] == "mnar"
    s6 = registry.scenarios["S6"]
    assert s6.dgp is None and s6.generation["post_treatment_notes"]
    variants = {variant.name: variant for variant in s6.variants}
    assert variants["text_column"].columns == {"text": "clinical_text_with_post_treatment"}
    assert variants["dated_notes"].notes["index_date_column"] == "index_date"
    seeds = load_seed_registry()
    assert len(seeds.development) == 3 and len(seeds.confirmatory) == 3
    assert not set(seeds.development) & set(seeds.confirmatory)
    assert "Tune only on development seeds" in seeds.rule


def test_registry_rejects_inconsistent_scenarios(tmp_path: Path):
    path = _registry(tmp_path)
    payload = json.loads(path.read_text())
    payload["scenarios"][0]["source"]["minimum_patients"] = 10
    _write_json(path, payload)
    with pytest.raises(ValueError, match="minimum_patients"):
        load_registry(path)
    with pytest.raises(ValueError, match="unknown scenario"):
        load_registry(_registry(tmp_path)).select("S9")


def test_seed_guard_refuses_confirmatory_seeds_without_the_flag(tmp_path: Path):
    seeds = load_seed_registry(_seeds(tmp_path))
    assert resolve_seeds(seeds) == [(11, "development"), (12, "development")]
    assert resolve_seeds(seeds, "12,13") == [(12, "development"), (13, "unregistered")]
    for request in ("91", "11,92", "confirmatory"):
        with pytest.raises(SeedGuardError, match="confirmatory"):
            resolve_seeds(seeds, request)
    assert resolve_seeds(seeds, None, confirmatory=True) == [(91, "confirmatory"), (92, "confirmatory")]
    assert resolve_seeds(seeds, None, confirmatory=True, include_development=True) == [
        (11, "development"), (12, "development"), (91, "confirmatory"), (92, "confirmatory")]
    # The CLI reports the refusal instead of materializing anything.
    source = _write_cohort(tmp_path / "source", 80, 0)
    code = benchmark.main([
        "materialize", "--scenarios", "S0", "--source", str(source), "--out", str(tmp_path / "b"),
        "--seeds", "91", "--registry", str(_registry(tmp_path)), "--seeds-file", str(_seeds(tmp_path)),
    ])
    assert code == 2 and not (tmp_path / "b" / "S0").exists()


# ---------------------------------------------------------------------------
# materialize


def test_materialize_writes_the_cell_layout_through_the_contract(tmp_path: Path, fake_reequation, capsys):
    registry, seeds = _registry(tmp_path), _seeds(tmp_path)
    source = _write_cohort(tmp_path / "source", 60, 0)
    bench = tmp_path / "bench"
    result = benchmark.materialize(out=bench, scenarios="S0", source=source,
                                   registry_path=registry, seeds_path=seeds)
    statuses = {item["cell"]: item["status"] for item in result["cells"]}
    assert statuses == {
        "S0/n40/seed11": "materialized", "S0/n40/seed12": "materialized",
        # The 60-patient source cannot provide 80 patients.
        "S0/n80/seed11": "skipped_source_too_small", "S0/n80/seed12": "skipped_source_too_small",
    }
    assert [(call["size"], call["seed"]) for call in fake_reequation] == [(40, 11), (40, 12)]
    call = fake_reequation[0]
    assert call["source_dir"] == source.resolve()
    assert call["scenario"] == BASE_DGP
    assert call["output_dir"] == bench.resolve() / "S0" / "n40" / "seed11" / "data"
    data = bench / "S0" / "n40" / "seed11" / "data"
    assert sorted(path.name for path in data.iterdir()) == [
        "dataset.parquet", "generation_config.json", "metadata.json", "reequation_manifest.json"]
    record = json.loads((bench / "S0" / "n40" / "seed11" / "materialize.json").read_text())
    assert record["identity"]["source"]["sha256"]["dataset.parquet"] == file_sha256(
        source / "dataset.parquet")
    assert record["outputs_sha256"]["dataset.parquet"] == file_sha256(data / "dataset.parquet")
    assert "note:" in capsys.readouterr().out
    # Rerunning is a no-op; a modified cohort is a conflict.
    again = benchmark.materialize(out=bench, scenarios="S0", source=source, sizes="40",
                                  registry_path=registry, seeds_path=seeds)
    assert {item["status"] for item in again["cells"]} == {"already_materialized"}
    assert len(fake_reequation) == 2
    (data / "metadata.json").write_text("{}")
    with pytest.raises(FileExistsError, match="different or modified"):
        benchmark.materialize(out=bench, scenarios="S0", source=source, sizes="40",
                              registry_path=registry, seeds_path=seeds)
    with pytest.raises(ValueError, match="--source"):
        benchmark.materialize(out=bench, scenarios="S0", registry_path=registry, seeds_path=seeds)


def test_generated_scenarios_print_generator_commands_then_subsample(tmp_path: Path, capsys):
    registry, seeds = _registry(tmp_path), _seeds(tmp_path)
    bench = tmp_path / "bench"
    result = benchmark.materialize(out=bench, scenarios="S6", registry_path=registry, seeds_path=seeds)
    assert result["cells"] == []
    instruction = result["generator_instructions"][0]
    config = json.loads(Path(instruction["generator_config"]).read_text())
    assert config == {"dataset_size": 40, "post_treatment_notes": {"rate": 1.0}}
    printed = capsys.readouterr().out
    assert f"python -m synthetic_data.cli --config {instruction['generator_config']}" in printed
    assert "--generated" in printed
    corpus = _write_cohort(tmp_path / "corpus", 40, 5, leak=True)
    result = benchmark.materialize(out=bench, scenarios="S6", generated=corpus,
                                   registry_path=registry, seeds_path=seeds)
    assert {item["cell"]: item["status"] for item in result["cells"]} == {
        "S6/n30/seed11": "materialized", "S6/n30/seed12": "materialized",
        "S6/n40/seed11": "materialized", "S6/n40/seed12": "materialized",
    }
    source = pd.read_parquet(corpus / "dataset.parquet")
    small = bench / "S6" / "n30" / "seed11" / "data"
    subset = pd.read_parquet(small / "dataset.parquet")
    expected_rows = np.sort(np.random.default_rng(11).choice(40, size=30, replace=False))
    assert subset["patient_id"].tolist() == source["patient_id"].iloc[expected_rows].tolist()
    notes = pd.read_parquet(small / "notes.parquet")
    assert set(notes["patient_id"]) == set(subset["patient_id"]) and len(notes) == 60
    manifest = json.loads((small / "subsample_manifest.json").read_text())
    assert manifest["rows_identical_to_source"] is False
    full = json.loads((bench / "S6" / "n40" / "seed12" / "data" / "subsample_manifest.json").read_text())
    assert full["rows_identical_to_source"] is True
    other = pd.read_parquet(bench / "S6" / "n30" / "seed12" / "data" / "dataset.parquet")
    assert other["patient_id"].tolist() != subset["patient_id"].tolist()


def test_generated_corpus_must_belong_to_one_scenario(tmp_path: Path):
    registry, seeds = _registry(tmp_path), _seeds(tmp_path)
    payload = json.loads(registry.read_text())
    second = json.loads(json.dumps(payload["scenarios"][2]))
    second["id"] = second["family"] = "S7"
    payload["scenarios"].append(second)
    _write_json(registry, payload)
    corpus = _write_cohort(tmp_path / "corpus", 40, 5, leak=True)
    with pytest.raises(ValueError, match="exactly one"):
        benchmark.materialize(out=tmp_path / "bench", scenarios="S6,S7", generated=corpus,
                              registry_path=registry, seeds_path=seeds)


# ---------------------------------------------------------------------------
# configs and the confirmatory ledger


def _base_config(tmp_path: Path) -> Path:
    base_dir = tmp_path / "base"
    (base_dir / "templates").mkdir(parents=True)
    (base_dir / "templates" / "stage1.json").write_text("{}")
    path = base_dir / "base.json"
    _write_json(path, {
        "dataset": "placeholder.parquet",
        "output_dir": "placeholder",
        "stage1_template": "templates/stage1.json",
        "columns": {"unit_id": "patient_id", "text": "clinical_text",
                    "treatment": "treatment_indicator", "outcome": "outcome_indicator"},
        "science": {"outcome_type": "continuous", "outer_folds": 2, "inner_folds": 2, "seed": 1},
        "stage2": {"endpoint": "http://127.0.0.1:9/v1", "model": "primary",
                   "extraction_llm": {"endpoint": "http://127.0.0.1:9/v1", "model": "extractor"}},
        "notes": {"path": "unrelated.parquet", "index_date_column": "index_date"},
    })
    return path


def _materialize_s6(tmp_path: Path, bench: Path, seeds_path: Path, *, seeds: str | None = None,
                    confirmatory: bool = False) -> None:
    corpus = tmp_path / "corpus"
    if not corpus.exists():
        _write_cohort(corpus, 40, 5, leak=True)
    benchmark.materialize(out=bench, scenarios="S6", generated=corpus, sizes="30",
                          registry_path=_registry(tmp_path), seeds_path=seeds_path,
                          seeds=seeds, confirmatory=confirmatory)


def test_s6_configs_hold_the_text_column_and_dated_notes_variants(tmp_path: Path, capsys, monkeypatch):
    from oci.inference.research_all_evidence_workflow import compile_config

    def no_subprocess(*args, **kwargs):
        raise AssertionError("the harness must not start processes")

    monkeypatch.setattr(subprocess, "run", no_subprocess)
    monkeypatch.setattr(subprocess, "Popen", no_subprocess)
    seeds = _seeds(tmp_path)
    bench = tmp_path / "bench"
    _materialize_s6(tmp_path, bench, seeds)
    base = _base_config(tmp_path)
    result = benchmark.write_configs(base=base, out=bench, scenarios="S6", registry_path=_registry(tmp_path),
                                     seeds_path=seeds, ledger_path=tmp_path / "ledger.jsonl")
    assert result["not_materialized"] == ["S6/n40/seed11", "S6/n40/seed12"]
    runs = {(item["cell"], item["variant"]): item for item in result["runs"]}
    assert set(runs) == {(cell, variant) for cell in ("S6/n30/seed11", "S6/n30/seed12")
                         for variant in ("text_column", "dated_notes")}
    cell = bench.resolve() / "S6" / "n30" / "seed11"
    text = json.loads((cell / "runs" / "text_column" / "config.json").read_text())
    notes = json.loads((cell / "runs" / "dated_notes" / "config.json").read_text())
    for config, variant in ((text, "text_column"), (notes, "dated_notes")):
        assert config["dataset"] == str(cell / "data" / "dataset.parquet")
        assert config["output_dir"] == str(cell / "runs" / variant / "output")
        assert config["science"]["seed"] == 11
        # The cohort's outcome type overrides the base configuration's.
        assert config["science"]["outcome_type"] == "binary"
        assert config["stage1_template"] == str((base.parent / "templates" / "stage1.json").resolve())
    assert text["columns"]["text"] == "clinical_text_with_post_treatment" and "notes" not in text
    assert notes["columns"]["text"] == "dated_notes_text"
    assert notes["notes"]["path"] == str(cell / "data" / "notes.parquet")
    assert notes["notes"]["index_date_column"] == "index_date"
    for config, variant in ((text, "text_column"), (notes, "dated_notes")):
        compiled = compile_config(config, config_dir=cell / "runs" / variant)
        assert compiled.seed == 11 and compiled.dataset == cell / "data" / "dataset.parquet"
    assert compile_config(notes, config_dir=cell).notes.path == cell / "data" / "notes.parquet"
    assert compile_config(text, config_dir=cell).notes is None
    printed = capsys.readouterr().out
    assert "no run was launched" in printed
    assert f"run_all_evidence.py --config {cell / 'runs' / 'dated_notes' / 'config.json'}" in printed
    index = json.loads((bench / "configs.json").read_text())
    assert len(index["runs"]) == 4
    assert not (tmp_path / "ledger.jsonl").exists()


def test_confirmatory_configs_append_the_ledger_and_refuse_reuse(tmp_path: Path, monkeypatch):
    seeds = _seeds(tmp_path)
    bench = tmp_path / "bench"
    _materialize_s6(tmp_path, bench, seeds, confirmatory=True)
    base, ledger = _base_config(tmp_path), tmp_path / "ledger.jsonl"
    commit = {"commit": "a" * 40, "dirty": False}
    monkeypatch.setattr(benchmark, "git_state", lambda root=None: dict(commit))
    arguments = dict(base=base, out=bench, scenarios="S6", registry_path=_registry(tmp_path),
                     seeds_path=seeds, ledger_path=ledger)
    with pytest.raises(SeedGuardError):
        benchmark.write_configs(seeds="91", **arguments)
    benchmark.write_configs(confirmatory=True, **arguments)
    entries = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["scenario"] == "S6" and entry["git_commit"] == "a" * 40
    assert entry["seeds"] == [91, 92] and entry["forced"] is False
    assert entry["time"] and len(entry["config_hash"]) == 64
    config_bytes = sorted(
        ([f"S6/n30/seed{seed}", variant,
          file_sha256(bench / "S6" / "n30" / f"seed{seed}" / "runs" / variant / "config.json")]
         for seed in (91, 92) for variant in ("dated_notes", "text_column")),
    )
    assert entry["config_hash"] == benchmark._sha256_json(config_bytes)
    with pytest.raises(SeedGuardError, match="already used"):
        benchmark.write_configs(confirmatory=True, **arguments)
    benchmark.write_configs(confirmatory=True, force=True, **arguments)
    assert [json.loads(line)["forced"] for line in ledger.read_text().splitlines()] == [False, True]
    # Another commit may use the confirmatory seeds once.
    commit["commit"] = "b" * 40
    benchmark.write_configs(confirmatory=True, **arguments)
    assert len(ledger.read_text().splitlines()) == 3
    # Development seeds never touch the ledger.
    _materialize_s6(tmp_path, bench, seeds)
    benchmark.write_configs(**arguments)
    assert len(ledger.read_text().splitlines()) == 3


# ---------------------------------------------------------------------------
# Metrics against hand-computed values


def _predictions(rows: int = 10) -> pd.DataFrame:
    return pd.DataFrame({
        "_oci_row_id": np.arange(rows),
        "outer_fold": [1] * (rows // 2) + [2] * (rows - rows // 2),
        "treatment": [0, 1] * (rows // 2),
        "outcome": [0, 1, 1, 0, 1, 1, 0, 0, 1, 0][:rows],
        "propensity": np.linspace(0.02, 0.98, rows),
        "mu0": np.full(rows, 0.4),
        "mu1": np.full(rows, 0.6),
        "aipw_score": [0.1, 0.3, np.nan, 0.5, 0.2, 0.4, 0.0, 0.6, 0.2, 0.1][:rows],
        "estimated_cate": np.linspace(-0.2, 0.4, rows),
    })


def test_ate_ato_and_cate_metrics_match_hand_computation():
    predictions = _predictions()
    truth = np.array([0.1, 0.2, 0.9, 0.3, 0.0, 0.2, 0.1, 0.4, 0.2, 0.3])
    ate = ate_metrics(predictions, truth, {"estimate": 0.25, "standard_error": 0.05,
                                           "confidence_interval_95": [0.152, 0.348]})
    eligible = [0, 1, 3, 4, 5, 6, 7, 8, 9]  # row 2 has no AIPW score
    true_ate = sum(truth[eligible]) / 9
    assert ate["rows"] == 9 and ate["true"] == pytest.approx(true_ate)
    assert ate["bias"] == pytest.approx(0.25 - true_ate)
    assert ate["covered"] is True and ate["ci_width"] == pytest.approx(0.196)
    assert ate_metrics(predictions, truth, {"estimate": 0.5, "confidence_interval_95": [0.45, 0.55]})[
        "covered"] is False

    e_true = np.full(10, 0.5)
    e_true[0] = 0.1
    ato = ato_metrics(predictions, truth, e_true, {"estimate": 0.2,
                                                   "confidence_interval_95": [0.1, 0.3]})
    weights = e_true * (1 - e_true)
    assert ato["true"] == pytest.approx(float(np.sum(weights * truth) / np.sum(weights)))
    e_hat = predictions["propensity"].to_numpy()
    assert ato["true_estimated_weights"] == pytest.approx(
        float(np.sum(e_hat * (1 - e_hat) * truth) / np.sum(e_hat * (1 - e_hat))))
    assert ato["bias"] == pytest.approx(0.2 - ato["true"])
    assert ato["covered"] is True and ato["rows"] == 10

    cate = cate_metrics(predictions, truth, 0.25)
    estimate = predictions["estimated_cate"].to_numpy()
    rmse = math.sqrt(float(np.mean((estimate - truth) ** 2)))
    baseline = math.sqrt(float(np.mean((0.25 - truth) ** 2)))
    assert cate["rmse"] == pytest.approx(rmse)
    assert cate["constant_ate_rmse"] == pytest.approx(baseline)
    assert cate["relative_improvement"] == pytest.approx(1 - rmse / baseline)
    assert cate["pearson"] == pytest.approx(float(np.corrcoef(estimate, truth)[0, 1]))
    assert cate["true_ite_sd"] == pytest.approx(float(np.std(truth)))


def test_gates_coverage_uses_the_diagnostics_groups():
    from oci.inference.stage2_diagnostics import build_heterogeneity_diagnostics

    rng = np.random.default_rng(4)
    n = 100
    predictions = pd.DataFrame({
        "_oci_row_id": np.arange(n), "outer_fold": np.repeat([1, 2], n // 2),
        "treatment": rng.integers(0, 2, n), "outcome": rng.integers(0, 2, n),
        "propensity": rng.uniform(0.2, 0.8, n), "mu0": rng.uniform(0.3, 0.6, n),
        "mu1": rng.uniform(0.3, 0.6, n), "aipw_score": rng.normal(0.1, 0.5, n),
        # Distinct values, so groups follow the ranks and do not depend on tie-breaking.
        "estimated_cate": rng.permutation(np.linspace(-0.3, 0.5, n)),
    })
    diagnostics = build_heterogeneity_diagnostics(predictions, propensity_clip=0.02, seed=7,
                                                  n_bootstrap=2, outcome_type="binary")
    truth = predictions["estimated_cate"].to_numpy() * 0.5 + 0.05
    result = gates_coverage(predictions, truth, diagnostics)
    assert result["status"] == "ok"
    hits = []
    pooled_truth = {g: [] for g in range(1, 6)}
    for fold in (1, 2):
        part = predictions[predictions["outer_fold"] == fold].reset_index(drop=True)
        ranks = part["estimated_cate"].rank(method="first").astype(int).to_numpy() - 1
        groups = ranks * 5 // len(part) + 1
        stored = diagnostics["gates"]["per_fold"][str(fold)]["groups"]
        for group in range(1, 6):
            members = part.index[groups == group]
            true_mean = truth[part.loc[members, "_oci_row_id"].to_numpy()].mean()
            pooled_truth[group].extend(truth[part.loc[members, "_oci_row_id"].to_numpy()])
            low, high = stored[group - 1]["confidence_interval_95"]
            hits.append(low <= true_mean <= high)
            computed = result["per_fold"][str(fold)]["groups"][group - 1]
            assert computed["true_mean_ite"] == pytest.approx(true_mean)
            assert computed["covered"] is bool(low <= true_mean <= high)
    assert result["fold_intervals"] == 10
    assert result["fold_coverage"] == pytest.approx(np.mean(hits))
    pooled = diagnostics["gates"]["pooled"]["groups"]
    expected = np.mean([
        pooled[g - 1]["confidence_interval_95"][0] <= np.mean(pooled_truth[g])
        <= pooled[g - 1]["confidence_interval_95"][1] for g in range(1, 6)
    ])
    assert result["pooled"]["coverage"] == pytest.approx(expected)
    tampered = json.loads(json.dumps(diagnostics))
    tampered["gates"]["per_fold"]["1"]["groups"][0]["estimate"] += 0.5
    assert gates_coverage(predictions, truth, tampered)["per_fold"]["1"]["status"] == "unavailable"


def _r2_row(fold, feature, concept, r2, *, nuisance=False, modifier=False, threshold=0.5):
    return {"outer_fold": fold, "feature_id": feature, "name": feature, "concept": concept,
            "r_squared": r2, "recovered": r2 is not None and r2 >= threshold,
            "nuisance_set": nuisance, "selected_modifier": modifier}


def test_selection_metrics_match_hand_counts():
    concepts = [
        OracleConcept("age", "true_age", "continuous", ("confounder",)),
        OracleConcept("ecog", "true_ecog", "categorical", ("confounder", "effect_modifier")),
        OracleConcept("pdl1", "true_pdl1", "categorical", ("effect_modifier",)),
        OracleConcept("smoker", "true_smoker", "categorical", ("instrument",)),
        OracleConcept("albumin", "true_albumin", "continuous", ("prognostic",)),
        OracleConcept("noise", "true_noise", "continuous", ()),
        OracleConcept("u_1", "true_u_1", "continuous", ("unmeasured_confounder",)),
        OracleConcept("post_treatment_toxicity", "true_post_treatment_toxicity", "categorical",
                      ("post_treatment",)),
    ]
    rows = [
        # Fold 1: modifiers m1 (recovers pdl1) and m2 (recovers only the noise variable);
        # nuisance set n1 (ecog), n2 (smoker, an instrument), n3 (the post-treatment event).
        _r2_row("1", "m1", "pdl1", 0.9, modifier=True),
        _r2_row("1", "m2", "noise", 0.8, modifier=True),
        _r2_row("1", "n1", "ecog", 0.7, nuisance=True),
        _r2_row("1", "n2", "smoker", 0.6, nuisance=True),
        _r2_row("1", "n3", "post_treatment_toxicity", 0.95, nuisance=True),
        _r2_row("1", "n1", "u_1", 0.1, nuisance=True),
        _r2_row("1", "x", "age", 0.9),  # recovers age but was not selected
        # Fold 2: one modifier recovering ecog (dual role) and albumin in the nuisance set.
        _r2_row("2", "m3", "ecog", 0.55, modifier=True, nuisance=True),
        _r2_row("2", "n4", "albumin", 0.7, nuisance=True),
        _r2_row("2", "m3", "u_1", 0.3, modifier=True, nuisance=True),
    ]
    recovery = {"status": "ok", "folds": ["1", "2"], "threshold_r_squared": 0.5,
                "r_squared": rows, "per_fold": {"1": {}, "2": {}}}
    result = selection_metrics(recovery, concepts)
    # True modifiers: ecog and pdl1. Fold 1 recovers pdl1, fold 2 recovers ecog.
    assert result["modifier_recall"] == pytest.approx(2 / 4)
    # Selected modifiers: m1, m2 (fold 1) and m3 (fold 2); m1 and m3 recover a modifier.
    assert result["modifier_precision"] == pytest.approx(2 / 3)
    assert result["selected_modifiers"] == 3
    # True confounders: age and ecog. Fold 1 recovers ecog; fold 2 recovers ecog through m3.
    assert result["confounder_recall"] == pytest.approx(2 / 4)
    assert result["instruments_in_nuisance_set"] == 1
    assert result["instruments_among_modifiers"] == 0
    assert result["prognostic_in_nuisance_set"] == 1
    assert result["unmeasured_max_r_squared_selected"] == {"u_1": 0.3}
    assert result["unmeasured_recovered_by_selected"] is False
    assert result["post_treatment_recovered"] is True
    assert result["post_treatment_recoveries"][0]["feature_id"] == "n3"
    assert result["per_fold"]["1"]["true_modifiers_recovered"] == ["pdl1"]
    assert selection_metrics({"status": "unavailable"}, concepts)["status"] == "unavailable"


def test_short_nuisances_integrate_the_unmeasured_confounder_out():
    rng = np.random.default_rng(1)
    n = 6
    eta_t, eta_0 = rng.normal(size=n), rng.normal(size=n)
    u = rng.normal(size=(n, 1))
    e = _expit(eta_t + GAMMA_T * u[:, 0])
    mu0 = _expit(eta_0 + GAMMA_Y * u[:, 0])
    mu1 = _expit(eta_0 + 0.5 + GAMMA_Y * u[:, 0])
    e_short, g0, g1 = short_nuisances(
        propensity=e, mu0=mu0, mu1=mu1, u=u, treatment_coefficients=np.array([GAMMA_T]),
        outcome_coefficients=np.array([GAMMA_Y]), binary=True, nodes=40)
    z = np.random.default_rng(2).normal(size=400_000)
    for i in range(n):
        e_z = _expit(eta_t[i] + GAMMA_T * z)
        m0_z, m1_z = _expit(eta_0[i] + GAMMA_Y * z), _expit(eta_0[i] + 0.5 + GAMMA_Y * z)
        assert e_short[i] == pytest.approx(e_z.mean(), abs=3e-3)
        assert g1[i] == pytest.approx((m1_z * e_z).mean() / e_z.mean(), abs=3e-3)
        assert g0[i] == pytest.approx((m0_z * (1 - e_z)).mean() / (1 - e_z).mean(), abs=3e-3)
    # Without U the short nuisances are the long ones.
    same = short_nuisances(propensity=e, mu0=mu0, mu1=mu1, u=u,
                           treatment_coefficients=np.zeros(1), outcome_coefficients=np.zeros(1),
                           binary=True)
    assert np.allclose(same[0], e) and np.allclose(same[1], mu0) and np.allclose(same[2], mu1)
    # Continuous outcomes: E[Y | D, X] shifts with the propensity-weighted mean of U.
    cont = short_nuisances(propensity=e, mu0=eta_0 + GAMMA_Y * u[:, 0], mu1=eta_0 + 1 + GAMMA_Y * u[:, 0],
                           u=u, treatment_coefficients=np.array([GAMMA_T]),
                           outcome_coefficients=np.array([GAMMA_Y]), binary=False, nodes=40)
    i = 0
    e_z = _expit(eta_t[i] + GAMMA_T * z)
    assert cont[2][i] == pytest.approx(eta_0[i] + 1 + GAMMA_Y * (z * e_z).mean() / e_z.mean(), abs=3e-3)


def test_sensitivity_metrics_follow_the_benchmark_formulas():
    from oci.evaluation.attribution import OracleData

    frame, metadata = _cohort(3000, 8, unmeasured=True)
    frame["_oci_row_id"] = np.arange(len(frame))
    clip = 0.02
    # The run's nuisances omit U: the true short nuisances plus noise.
    u = frame[["true_u_1"]].to_numpy()
    e, y0, y1 = (frame[c].to_numpy() for c in ("true_treatment_prob", "true_y0", "true_y1"))
    e_s, g0, g1 = short_nuisances(propensity=e, mu0=y0, mu1=y1, u=u,
                                  treatment_coefficients=np.array([GAMMA_T]),
                                  outcome_coefficients=np.array([GAMMA_Y]), binary=True)
    t, y = frame["treatment_indicator"].to_numpy(float), frame["outcome_indicator"].to_numpy(float)
    rng = np.random.default_rng(9)
    e_hat = np.clip(e_s + rng.normal(0, 0.02, len(t)), 0.03, 0.97)
    m0_hat, m1_hat = np.clip(g0, 0.02, 0.98), np.clip(g1, 0.02, 0.98)
    clipped = np.clip(e_hat, clip, 1 - clip)
    psi = m1_hat - m0_hat + t * (y - m1_hat) / clipped - (1 - t) * (y - m0_hat) / (1 - clipped)
    predictions = pd.DataFrame({"_oci_row_id": frame["_oci_row_id"], "treatment": t, "outcome": y,
                                "propensity": e_hat, "mu0": m0_hat, "mu1": m1_hat,
                                "aipw_score": psi})
    concepts = [OracleConcept("u_1", "true_u_1", "continuous", ("unmeasured_confounder",))]
    oracle = benchmark.BenchmarkOracle(
        data=OracleData(dataset=Path("x"), dataset_sha256="", metadata=None, metadata_sha256=None,
                        concepts=concepts, frame=frame[["_oci_row_id", "true_u_1"]],
                        absent_concepts=[]),
        values=frame, metadata=metadata, ite_column="true_ite",
        potential_outcome_columns=("true_y0", "true_y1"),
        unmeasured=metadata["unmeasured_confounders"], post_treatment_concepts=[],
    )
    heterogeneity = {"unmeasured_confounding_sensitivity": {
        "evalues": {"ate": {"pooled": {"evalue_point": 1.8, "evalue_confidence_limit": 1.2}}},
        "omitted_variable_bias": {"ate": {"pooled": {"robustness_value": 0.07,
                                                     "robustness_value_confidence": 0.02}}},
    }}
    true_ate = float(frame["true_ite"].mean())
    result = benchmark.sensitivity_metrics(predictions, oracle, heterogeneity, outcome_type="binary",
                                           propensity_clip=clip, true_ate=true_ate,
                                           unmeasured_strength=1.0)
    assert result["status"] == "ok"
    assert (result["evalue_point"], result["robustness_value_confidence"]) == (1.8, 0.02)

    def elements(e_value, mu0, mu1):
        ec = np.clip(e_value, clip, 1 - clip)
        theta = np.mean(mu1 - mu0 + t * (y - mu1) / ec - (1 - t) * (y - mu0) / (1 - ec))
        sigma2 = np.mean((y - (t * mu1 + (1 - t) * mu0)) ** 2)
        alpha = t / ec - (1 - t) / (1 - ec)
        nu2 = np.mean(2 * (1 / ec + 1 / (1 - ec)) - alpha ** 2)
        return theta, sigma2, nu2

    _theta_l, sigma2_l, nu2_l = elements(e, y0, y1)
    _theta_s, sigma2_s, nu2_s = elements(e_s, g0, g1)
    variance = np.var(y)
    r2_long, r2_short = 1 - sigma2_l / variance, 1 - sigma2_s / variance
    cf_y = (r2_long - r2_short) / (1 - r2_long)
    cf_d = (1 - nu2_s / nu2_l) / (nu2_s / nu2_l)
    strength = result["true_strength"]
    assert strength["cf_y"] == pytest.approx(cf_y)
    assert strength["cf_d"] == pytest.approx(cf_d)
    assert 0 < cf_y < 1 and 0 < cf_d < 1
    theta_run, sigma2_run, nu2_run = elements(e_hat, m0_hat, m1_hat)
    bias = math.sqrt(sigma2_run * nu2_run) * math.sqrt(cf_y * cf_d / (1 - cf_d))
    bound = result["run_bound_at_true_strength"]["rho_1"]
    assert bound["lower"] == pytest.approx(theta_run - bias)
    assert bound["upper"] == pytest.approx(theta_run + bias)
    assert bound["covers_true_ate"] is bool(theta_run - bias <= true_ate <= theta_run + bias)
    # The run omits U, and the bound at U's own strength contains the true ATE.
    assert bound["covers_true_ate"] is True
    no_u = benchmark.sensitivity_metrics(
        predictions, benchmark.BenchmarkOracle(**{**oracle.__dict__, "unmeasured": []}),
        heterogeneity, outcome_type="binary", propensity_clip=clip, true_ate=true_ate)
    assert no_u["status"] == "not_applicable" and no_u["robustness_value"] == 0.07


def test_unmeasured_coefficients_from_metadata_or_least_squares():
    frame, metadata = _cohort(4000, 3, unmeasured=True)
    frame["_oci_row_id"] = np.arange(len(frame))
    concepts = [OracleConcept(f["name"], f"true_{f['name']}",
                              "continuous" if f["type"] == "continuous" else "categorical",
                              tuple(f["roles"])) for f in METADATA_FEATURES]
    arguments = dict(values=frame, concepts=concepts,
                     treatment_logit=np.log(frame["true_treatment_prob"] / (1 - frame["true_treatment_prob"])).to_numpy(),
                     outcome_scale_y0=np.log(frame["true_y0"] / (1 - frame["true_y0"])).to_numpy())
    stated = benchmark.unmeasured_coefficients(metadata["unmeasured_confounders"], **arguments)[0]
    assert stated["source"] == "metadata"
    assert (stated["treatment_coefficient"], stated["outcome_coefficient"]) == (GAMMA_T, GAMMA_Y)
    fitted = benchmark.unmeasured_coefficients([{"name": "u_1", "coefficients": {}}], **arguments)[0]
    assert fitted["source"] == "least_squares_on_oracle_columns"
    assert fitted["treatment_coefficient"] == pytest.approx(GAMMA_T, abs=1e-6)
    assert fitted["outcome_coefficient"] == pytest.approx(GAMMA_Y, abs=1e-6)


def test_scenario_and_leak_tables():
    def row(seed, variant, bias, covered, overlap, leaked, status="complete"):
        return {"scenario": "S6", "family": "S6", "variant": variant, "size": 30, "seed": seed,
                "seed_set": "development", "status": status, "ate_bias": bias,
                "ate_covered": covered, "outside_overlap_fraction": overlap,
                "post_treatment_recovered": leaked}

    rows = [
        row(11, "text_column", 0.3, False, 0.4, True), row(11, "dated_notes", 0.1, True, 0.1, False),
        row(12, "text_column", -0.1, True, 0.2, True), row(12, "dated_notes", 0.0, True, 0.0, False),
        row(13, "text_column", None, None, None, None, status="pending"),
        row(14, "text_column", None, None, None, None, status="error"),
    ]
    table = {item["variant"]: item for item in scenario_rows(rows)}
    text = table["text_column"]
    assert (text["cells_expected"], text["cells_complete"], text["cells_pending"],
            text["cells_error"]) == (4, 2, 1, 1)
    assert text["ate_mean_bias"] == pytest.approx(0.1)
    assert text["ate_rmse"] == pytest.approx(math.sqrt((0.09 + 0.01) / 2))
    assert text["ate_coverage"] == pytest.approx(0.5)
    assert text["post_treatment_recovered_rate"] == pytest.approx(1.0)
    leaks = leak_rows(rows)
    assert [item["seed"] for item in leaks] == [11, 12]
    assert leaks[0]["ate_bias_text_minus_notes"] == pytest.approx(0.2)
    assert leaks[0]["outside_overlap_fraction_text_minus_notes"] == pytest.approx(0.3)
    assert leaks[1]["text_column_post_treatment_recovered"] is True
    assert leaks[1]["dated_notes_post_treatment_recovered"] is False


def test_overlap_metrics_count_rows_outside_the_flag_bounds():
    predictions = pd.DataFrame({"propensity": [0.01, 0.05, 0.5, 0.95, 0.97, np.nan]})
    result = benchmark.overlap_metrics(predictions)
    assert (result["rows"], result["outside_overlap"]) == (5, 2)
    assert result["outside_overlap_fraction"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Fake runs and aggregate


STAGE2_FINAL_FEATURES = [
    {"feature_id": "f_ecog", "name": "performance_status", "value_type": "categorical",
     "roles": ["confounder"]},
    {"feature_id": "f_smoking", "name": "smoking_history", "value_type": "categorical",
     "roles": ["confounder"]},
    {"feature_id": "f_pdl1", "name": "pdl1_expression", "value_type": "categorical",
     "roles": ["effect_modifier"]},
    {"feature_id": "f_marker", "name": "random_marker", "value_type": "continuous",
     "roles": ["effect_modifier"]},
]
LEAK_FEATURE = {"feature_id": "f_toxicity", "name": "documented_toxicity",
                "value_type": "categorical", "roles": ["confounder"]}


def _splits(n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    rows = []
    for fold, held in zip(FOLDS, (np.sort(order[: n // 2]), np.sort(order[n // 2:]))):
        fit = np.setdiff1d(np.arange(n), held)
        inner_order = rng.permutation(fit)
        inner = []
        for k, valid in enumerate((np.sort(inner_order[::2]), np.sort(inner_order[1::2])), start=1):
            inner.append({"inner_fold": k, "fit_row_ids": np.setdiff1d(fit, valid).tolist(),
                          "heldout_row_ids": valid.tolist()})
        rows.append({"outer_fold": fold, "fit_row_ids": fit.tolist(),
                     "heldout_row_ids": held.tolist(), "inner_splits": inner})
    return rows


def _fake_run(run_dir: Path, data_dir: Path, *, seed: int, leak: bool = False,
              dated_notes: bool = False) -> None:
    """A finished workflow output written with the current Stage 2 writers."""

    from oci.inference.stage2_diagnostics import (
        build_heterogeneity_diagnostics,
        co_primary_estimands,
        dr_scores,
    )
    from oci.inference.stage2_elastic_net_selection import (
        statistical_selection_config_from_mapping,
    )
    from oci.models.super_learner_nuisance import SuperLearnerConfig

    dataset = pd.read_parquet(data_dir / "dataset.parquet")
    n = len(dataset)
    rng = np.random.default_rng(seed + 1)
    run_dir.mkdir(parents=True)
    _write_json(run_dir / "run_config.json", {
        "dataset": str(data_dir / "dataset.parquet"), "output_dir": str(run_dir),
        "unit_id_column": "patient_id", "text_column": "clinical_text",
        "treatment_column": "treatment_indicator", "outcome_column": "outcome_indicator",
        "outcome_type": "binary", "seed": seed, "inner_folds": 2, "outer_folds": 2,
        "stage2": {"api_key": "<redacted>"},
    })
    if dated_notes:
        cohort = dataset.copy()
        cohort["dated_notes_text"] = "pretreatment note"
        (run_dir / "input").mkdir()
        cohort.to_parquet(run_dir / "input" / "analysis_cohort.parquet", index=False)
    stage2 = run_dir / "stage2"
    _write_json(stage2 / "config.json", {
        "statistical_selection": statistical_selection_config_from_mapping({}).public_dict(),
        "nuisance_learner": "elastic_net", "super_learner": SuperLearnerConfig().public_dict(),
        "propensity_clip": 0.02, "estimation_trees": 8, "min_propensity": None,
        "max_propensity": None, "lead_estimand": "ate", "overlap_flag_bounds": [0.05, 0.95],
    })
    splits = _splits(n, seed)
    split_path = run_dir / "components" / "tfidf" / "split_provenance.jsonl"
    split_path.parent.mkdir(parents=True)
    split_path.write_text("".join(json.dumps(row) + "\n" for row in splits))
    frames = []
    for split in splits:
        held = np.array(split["heldout_row_ids"])
        part = dataset.iloc[held]
        e = np.clip(part["true_treatment_prob"].to_numpy() + rng.normal(0, 0.05, len(held)), 0.03, 0.97)
        if leak:
            e = np.where(part["treatment_indicator"].to_numpy() == 1, 0.97, 0.03)
        mu0 = np.clip(part["true_y0"].to_numpy() + rng.normal(0, 0.05, len(held)), 0.02, 0.98)
        mu1 = np.clip(part["true_y1"].to_numpy() + rng.normal(0, 0.05, len(held)), 0.02, 0.98)
        t, y = part["treatment_indicator"].to_numpy(), part["outcome_indicator"].to_numpy()
        frames.append(pd.DataFrame({
            "_oci_row_id": held, "outer_fold": split["outer_fold"],
            "patient_id": part["patient_id"].to_numpy(), "treatment": t, "outcome": y,
            "propensity": e, "mu0": mu0, "mu1": mu1, "effect_eligible": True,
            "aipw_score": dr_scores(y, t, e, mu0, mu1, clip=0.02),
            "estimated_cate": part["true_ite"].to_numpy() + rng.normal(0, 0.05, len(held)),
        }))
        fold_dir = stage2 / f"outer_{split['outer_fold']:03d}"
        features = [*STAGE2_FINAL_FEATURES, *([LEAK_FEATURE] if leak else [])]
        _write_json(fold_dir / "final_definitions.json",
                    {"features": features, "final_estimator": "ensemble"})
        _write_json(fold_dir / "estimation" / "nuisance_candidate_definitions.json",
                    {"features": features})
        heldout = pd.DataFrame({
            "_oci_row_id": held,
            "performance_status": [f"PS {value}" for value in part["true_ecog"]],
            "smoking_history": part["true_smoker"].to_numpy(),
            "pdl1_expression": part["true_pdl1"].to_numpy(),
            "random_marker": rng.normal(size=len(held)),
        })
        if leak:
            heldout["documented_toxicity"] = part["true_post_treatment_toxicity"].astype(str).to_numpy()
        heldout.to_csv(fold_dir / "estimation" / "nuisance_candidate_heldout.csv", index=False)
    predictions = pd.concat(frames, ignore_index=True).sort_values("_oci_row_id").reset_index(drop=True)
    predictions.to_csv(stage2 / "cross_fitted_predictions.csv", index=False)
    _write_json(stage2 / "causal_estimate.json",
                {"co_primary_estimands": co_primary_estimands(predictions, propensity_clip=0.02)})
    _write_json(stage2 / "heterogeneity_diagnostics.json", build_heterogeneity_diagnostics(
        predictions, propensity_clip=0.02, seed=seed, n_bootstrap=5, outcome_type="binary"))
    _write_json(stage2 / "complete.json", {"status": "complete", "phase": "causal_estimation"})
    _write_json(run_dir / "progress.json", {"status": "complete"})


def _bench_with_runs(tmp_path: Path, fake_reequation) -> tuple[Path, dict]:
    registry, seeds = _registry(tmp_path), _seeds(tmp_path)
    bench = tmp_path / "bench"
    source = _write_cohort(tmp_path / "source", 120, 0)
    corpus = _write_cohort(tmp_path / "corpus", 120, 5, leak=True)
    payload = json.loads(registry.read_text())
    for scenario in payload["scenarios"]:
        scenario["sizes"] = [120]
        scenario["source"]["minimum_patients"] = 120
        if scenario.get("generation"):
            scenario["generation"]["dataset_size"] = 120
    _write_json(registry, payload)
    common = dict(out=bench, registry_path=registry, seeds_path=seeds)
    benchmark.materialize(scenarios="S0,S4", source=source, **common)
    benchmark.materialize(scenarios="S6", generated=corpus, **common)
    benchmark.write_configs(base=_base_config(tmp_path), seeds="11", ledger_path=tmp_path / "l.jsonl",
                            **common)
    def cell(scenario: str) -> Path:
        return bench.resolve() / scenario / "n120" / "seed11"

    _fake_run(cell("S0") / "runs" / "default" / "output", cell("S0") / "data", seed=11)
    _fake_run(cell("S4a") / "runs" / "default" / "output", cell("S4a") / "data", seed=11)
    _fake_run(cell("S6") / "runs" / "text_column" / "output", cell("S6") / "data", seed=11, leak=True)
    _fake_run(cell("S6") / "runs" / "dated_notes" / "output", cell("S6") / "data", seed=11,
              dated_notes=True)
    # Seed 12 is materialized for S0 but has no configuration; S4a's seed-12 run is running.
    running = cell("S4a").parent / "seed12" / "runs" / "default" / "output"
    benchmark.write_configs(base=_base_config(tmp_path / "again"), seeds="12", scenarios="S4",
                            ledger_path=tmp_path / "l.jsonl", **common)
    running.mkdir(parents=True)
    _write_json(running / "progress.json", {"status": "running", "current_component": "text_models"})
    return bench, common


def _tree_state(root: Path) -> dict[str, tuple[int, int]]:
    return {str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file()}


def test_aggregate_reads_finished_runs_and_lists_pending_ones(tmp_path: Path, fake_reequation, monkeypatch):
    from oci.inference import plain_handoff_stage2_analysis as analysis

    bench, common = _bench_with_runs(tmp_path, fake_reequation)
    s0_run = bench.resolve() / "S0" / "n120" / "seed11" / "runs" / "default" / "output"
    run_states = {path: _tree_state(path) for path in bench.resolve().glob("*/n120/seed*/runs/*/output")}
    frozen_at_load: list[tuple[bool, set]] = []
    original_load = benchmark.load_oracle_data

    def spy_load(dataset, metadata, *, readers):
        frozen_at_load.append((all(reader.frozen for reader in readers),
                               {key for reader in readers for key in reader.hashes}))
        return original_load(dataset, metadata, readers=readers)

    monkeypatch.setattr(benchmark, "load_oracle_data", spy_load)
    calls = []
    original_estimate = analysis.estimate_outer_fold

    def spy_estimate(**kwargs):
        calls.append(kwargs)
        return original_estimate(**kwargs)

    monkeypatch.setattr(analysis, "estimate_outer_fold", spy_estimate)
    result = benchmark.aggregate(seeds="11,12", **common)
    assert result["errors"] == []
    output = bench.resolve() / "aggregate"
    assert sorted(path.name for path in output.iterdir()) == [
        "cells.csv", "scenarios.csv", "summary.json", "summary.md"]
    cells = pd.read_csv(output / "cells.csv")
    status = {(row.cell, row.variant): (row.status, row.detail) for row in cells.itertuples()}
    assert status[("S0/n120/seed11", "default")] == ("complete", "complete")
    assert status[("S0/n120/seed12", "default")] == ("pending", "no_config")
    assert status[("S4a/n120/seed12", "default")] == ("pending", "running_text_models")
    assert status[("S6/n120/seed12", "text_column")] == ("pending", "no_config")
    assert status[("S6/n120/seed11", "dated_notes")] == ("complete", "complete")
    summary = json.loads((output / "summary.json").read_text())
    assert {(item["cell"], item["variant"]) for item in summary["pending"]} == {
        key for key, value in status.items() if value[0] == "pending"}
    assert "S4a/n120/seed12" in (output / "summary.md").read_text()

    # Every oracle load happened after the run artifacts were hashed and frozen.
    assert len(frozen_at_load) == 4 and all(frozen for frozen, _keys in frozen_at_load)
    assert {"stage2/cross_fitted_predictions.csv", "stage2/causal_estimate.json",
            "stage2/heterogeneity_diagnostics.json", "components/tfidf/split_provenance.jsonl",
            "run_config.json"} <= frozen_at_load[0][1]
    s0 = next(item for item in summary["cells"] if item["cell"] == "S0/n120/seed11")
    metrics = s0["metrics"]
    assert metrics["run_artifacts_hashed_before_oracle_load"] is True
    assert metrics["run_artifact_sha256"]["stage2/cross_fitted_predictions.csv"] == file_sha256(
        s0_run / "stage2" / "cross_fitted_predictions.csv")
    # Nothing was written inside any run directory.
    for path, state in run_states.items():
        assert _tree_state(path) == state

    # The ATE row matches a direct computation from the files.
    predictions = pd.read_csv(s0_run / "stage2" / "cross_fitted_predictions.csv")
    dataset = pd.read_parquet(bench.resolve() / "S0" / "n120" / "seed11" / "data" / "dataset.parquet")
    truth = dataset["true_ite"].to_numpy()[predictions["_oci_row_id"].to_numpy()]
    estimate = json.loads((s0_run / "stage2" / "causal_estimate.json").read_text())[
        "co_primary_estimands"]["ate"]
    row = cells[(cells["cell"] == "S0/n120/seed11")].iloc[0]
    assert row["ate_true"] == pytest.approx(truth.mean())
    assert row["ate_bias"] == pytest.approx(estimate["estimate"] - truth.mean())
    low, high = estimate["confidence_interval_95"]
    assert bool(row["ate_covered"]) is bool(low <= truth.mean() <= high)
    cate = predictions["estimated_cate"].to_numpy()
    assert row["cate_rmse"] == pytest.approx(math.sqrt(np.mean((cate - truth) ** 2)))
    assert row["constant_ate_rmse"] == pytest.approx(
        math.sqrt(np.mean((estimate["estimate"] - truth) ** 2)))

    # Role-wise selection by held-out R^2: pdl1 recovered by pdl1_expression (precision 1/2
    # because random_marker recovers nothing); ecog but not age in the nuisance set; the
    # smoking instrument in the nuisance set in both folds.
    assert row["modifier_recall"] == pytest.approx(1.0)
    assert row["modifier_precision"] == pytest.approx(0.5)
    assert row["confounder_recall"] == pytest.approx(0.5)
    assert row["instruments_in_nuisance_set"] == 2
    assert row["prognostic_in_nuisance_set"] == 0

    # The ceiling ran Stage 2's estimate_outer_fold on the true covariates, per fold,
    # with the run's folds, seeds, learner and estimator; its outputs are outside the run.
    s0_calls = [call for call in calls if "S0" in str(call["output_dir"])]
    assert [call["seed"] for call in s0_calls] == [11 + 100_000, 11 + 200_000]
    splits = [json.loads(line) for line in
              (s0_run / "components" / "tfidf" / "split_provenance.jsonl").read_text().splitlines()]
    assert [call["split"]["heldout_row_ids"] for call in s0_calls] == [
        split["heldout_row_ids"] for split in splits]
    first = s0_calls[0]
    assert first["estimator"] == "ensemble" and first["nuisance_learner"] == {"name": "elastic_net"}
    assert first["estimation_trees"] == 8 and first["inner_folds"] == 2
    assert first["nuisance_model_store"] is None
    roles = {item["name"]: item["roles"] for item in first["definitions"]}
    assert roles == {"age": ["confounder"], "ecog": ["confounder"],
                     "pdl1": ["effect_modifier"], "albumin": ["confounder"]}
    assert [item["name"] for item in first["nuisance_candidate_definitions"]] == [
        "age", "ecog", "pdl1", "smoker", "albumin"]
    fit_frame = first["nuisance_candidate_fit"]
    assert np.allclose(fit_frame["age"], dataset["true_age"].to_numpy()[fit_frame["_oci_row_id"]])
    ceiling = metrics["oracle_ceiling"]
    assert ceiling["status"] == "ok"
    assert Path(ceiling["output_dir"]).parent == bench.resolve() / "S0" / "n120" / "seed11" / "oracle_ceiling"
    assert ceiling["ate"]["rows"] == 120 and ceiling["cate"]["status"] == "ok"
    assert row["ceiling_status"] == "ok" and not math.isnan(row["ceiling_cate_rmse"])

    # S4: U's strength from the oracle columns, next to the run's E-value and RV.
    s4 = cells[cells["cell"] == "S4a/n120/seed11"].iloc[0]
    assert 0.0 < s4["u_cf_y"] <= 1.0 and 0.0 < s4["u_cf_d"] <= 1.0
    assert s4["unmeasured_strength"] == pytest.approx(0.5)
    assert not math.isnan(s4["robustness_value"]) and not math.isnan(s4["evalue_point"])
    s4_metrics = next(item for item in summary["cells"] if item["cell"] == "S4a/n120/seed11")["metrics"]
    bound = s4_metrics["sensitivity"]["run_bound_at_true_strength"]["rho_1"]
    true_ate = s4_metrics["ate"]["true"]
    assert bound["covers_true_ate"] is bool(bound["lower"] <= true_ate <= bound["upper"])
    assert s4_metrics["sensitivity"]["unmeasured_confounders"][0]["source"] == "metadata"
    assert s4["unmeasured_max_r_squared_selected"] < 0.5

    # S6: the leaked variable recovers the post-treatment event in the text-column run only.
    text = cells[(cells["cell"] == "S6/n120/seed11") & (cells["variant"] == "text_column")].iloc[0]
    notes = cells[(cells["cell"] == "S6/n120/seed11") & (cells["variant"] == "dated_notes")].iloc[0]
    assert bool(text["post_treatment_recovered"]) is True
    assert bool(notes["post_treatment_recovered"]) is False
    assert text["outside_overlap_fraction"] == pytest.approx(1.0)
    assert notes["outside_overlap_fraction"] < 1.0
    leak = summary["leak_comparison"][0]
    assert leak["ate_bias_text_minus_notes"] == pytest.approx(text["ate_bias"] - notes["ate_bias"])
    s6_notes = next(item for item in summary["cells"]
                    if item["cell"] == "S6/n120/seed11" and item["variant"] == "dated_notes")
    assert s6_notes["metrics"]["oracle_dataset"].endswith("input/analysis_cohort.parquet")
    scenarios = pd.read_csv(output / "scenarios.csv")
    s0_summary = scenarios[(scenarios["scenario"] == "S0")].iloc[0]
    assert (s0_summary["cells_expected"], s0_summary["cells_complete"]) == (2, 1)
    assert s0_summary["ate_rmse"] == pytest.approx(abs(row["ate_bias"]))

    # A second aggregate reuses the ceiling's estimation checkpoints (same values).
    first_ceiling = row["ceiling_cate_rmse"]
    calls.clear()
    benchmark.aggregate(seeds="11", scenarios="S0", **common)
    assert len(calls) == 2
    again = pd.read_csv(output / "cells.csv")
    assert again.iloc[0]["ceiling_cate_rmse"] == pytest.approx(first_ceiling)
    estimation = Path(ceiling["output_dir"]) / "outer_001" / "estimation"
    assert (estimation / "complete.json").is_file() and (estimation / "predictions.csv").is_file()


def test_aggregate_refuses_confirmatory_cells_without_the_flag(tmp_path: Path):
    common = dict(out=tmp_path / "bench", registry_path=_registry(tmp_path), seeds_path=_seeds(tmp_path))
    with pytest.raises(SeedGuardError):
        benchmark.aggregate(seeds="91", ceiling=False, **common)
    result = benchmark.aggregate(ceiling=False, confirmatory=True, scenarios="S0", **common)
    cells = pd.read_csv(Path(result["output_dir"]) / "cells.csv")
    assert set(cells["seed_set"]) == {"development", "confirmatory"}
    assert set(cells["detail"]) == {"not_materialized"}


def test_oracle_loading_requires_frozen_readers(tmp_path: Path):
    data = _write_cohort(tmp_path / "data", 20, 0)
    reader = FrozenArtifactReader(tmp_path)
    with pytest.raises(RuntimeError, match="hash and freeze"):
        benchmark.load_benchmark_oracle(data / "dataset.parquet", data / "metadata.json",
                                        readers=[reader], outcome_type="binary", label_columns=[])
    reader.freeze()
    oracle = benchmark.load_benchmark_oracle(data / "dataset.parquet", data / "metadata.json",
                                             readers=[reader], outcome_type="binary",
                                             label_columns=["treatment_indicator"])
    assert oracle.ite_column == "true_ite"
    assert [concept.name for concept in oracle.data.concepts] == [f["name"] for f in METADATA_FEATURES]


def test_run_status_reports_pending_details(tmp_path: Path):
    assert run_status(tmp_path / "missing") == ("pending", "not_started")
    run = tmp_path / "run"
    run.mkdir()
    assert run_status(run) == ("pending", "incomplete")
    _write_json(run / "progress.json", {"status": "failed", "current_component": "stage2"})
    assert run_status(run) == ("pending", "failed_stage2")
    _write_json(run / "progress.json", {"status": "complete"})
    assert run_status(run) == ("pending", "stage2_not_run")
    _write_json(run / "stage2" / "complete.json", {"status": "complete", "phase": "feature_definitions"})
    assert run_status(run) == ("pending", "stage2_phase_feature_definitions")
    _write_json(run / "stage2" / "complete.json", {"status": "complete", "phase": "causal_estimation"})
    assert run_status(run) == ("pending", "stage2_outputs_missing")
    for name in ("cross_fitted_predictions.csv", "causal_estimate.json", "heterogeneity_diagnostics.json"):
        (run / "stage2" / name).write_text("{}")
    assert run_status(run) == ("complete", "complete")


def test_benchmark_import_stays_lightweight():
    script = (
        "import sys, oci.evaluation.benchmark; "
        "heavy = [m for m in ('synthetic_data.reequation', 'torch', "
        "'oci.inference.plain_handoff_stage2_analysis', 'oci.evaluation.benchmark_ceiling') "
        "if m in sys.modules]; print(heavy)"
    )
    result = subprocess.run([sys.executable, "-c", script], cwd=REPO, capture_output=True,
                            text=True, check=True)
    assert result.stdout.strip() == "[]"
