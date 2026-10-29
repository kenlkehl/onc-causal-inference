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
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oci.evaluation import benchmark
from oci.evaluation.attribution import FrozenArtifactReader, file_sha256
from oci.evaluation.benchmark import (
    SeedGuardError,
    load_seed_registry,
    resolve_seeds,
)

DEV, CONF = [11, 12], [91, 92]
FOLDS = (1, 2)


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
