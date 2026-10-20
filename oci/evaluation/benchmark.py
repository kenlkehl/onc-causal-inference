"""Benchmark harness: scenario registry, seed guard, materialization, run configs, aggregation.

``scripts/benchmark.py`` is the command-line entry point; everything it does is
implemented here so that it can be tested.

* ``materialize`` writes one cohort per cell, ``BENCH/<scenario>/n<size>/seed<k>/data/``.
  Re-equation scenarios call ``synthetic_data.reequation.reequation_cohort`` (the
  generator's re-equation contract, imported only here). Scenarios that need
  LLM text (S5, S6) print the generator command instead; with ``--generated DIR``
  the generated corpus is re-equated (S5) or subsampled (S6).
* ``configs`` writes one ``run_all_evidence`` configuration per cell and run
  variant and prints the commands. It never launches a run or an LLM request.
* ``aggregate`` reads the finished runs, lists the others as pending, and
  compares the frozen outputs with the oracle columns.

Seeds come from ``benchmarks/seeds.json``. Confirmatory seeds are refused
without ``--confirmatory``; ``configs --confirmatory`` appends one entry per
scenario to ``benchmarks/confirmatory_ledger.jsonl`` and refuses a second use for
the same git commit and scenario unless ``--force`` is given.

Oracle use in ``aggregate`` is post hoc and follows the attribution report's
rule: every run artifact the aggregate reads is read once through a
:class:`~oci.evaluation.attribution.FrozenArtifactReader`, hashed, and the
reader is frozen before any oracle column is loaded. Recovery of oracle concepts
reuses :func:`~oci.evaluation.attribution.oracle_recovery` (held-out R^2). The
oracle-covariate ceiling (:mod:`oci.evaluation.benchmark_ceiling`) refits Stage
2's final estimation on the true covariates after the reader is frozen.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .attribution import (
    DEFAULT_RECOVERY_R2,
    FrozenArtifactReader,
    OracleConcept,
    OracleData,
    assert_output_outside_run_inputs,
    collect_stage2_lineage,
    file_sha256,
    load_oracle_data,
    oracle_recovery,
    resolve_oracle_inputs,
    write_csv,
    write_json,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPOSITORY_ROOT / "benchmarks"
DEFAULT_REGISTRY = BENCHMARK_DIR / "scenarios.json"
DEFAULT_SEEDS = BENCHMARK_DIR / "seeds.json"
DEFAULT_LEDGER = BENCHMARK_DIR / "confirmatory_ledger.jsonl"
REGISTRY_SCHEMA_VERSION = "oci_benchmark_registry_v1"
SEEDS_SCHEMA_VERSION = "oci_benchmark_seeds_v1"
LEDGER_SCHEMA_VERSION = "oci_benchmark_confirmatory_ledger_v1"
MATERIALIZATION_SCHEMA_VERSION = "oci_benchmark_materialization_v1"
CONFIGS_SCHEMA_VERSION = "oci_benchmark_configs_v1"
AGGREGATE_SCHEMA_VERSION = "oci_benchmark_aggregate_v1"
MATERIALIZATIONS = ("reequation", "reequation_of_generated", "subsample_of_generated")
SEED_SETS = ("development", "confirmatory")
DEFAULT_VARIANT = "default"
OVERLAP_FLAG_BOUNDS = (0.05, 0.95)
ROW_ID = "_oci_row_id"
DATA_FILES = ("dataset.parquet", "metadata.json", "generation_config.json")
REEQUATION_OUTPUTS = (*DATA_FILES, "reequation_manifest.json")
RUN_COMMAND_ENV = "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1"
GAUSS_HERMITE_NODES = 24
POST_TREATMENT_PREFIX = "true_post_treatment_"

METRIC_DEFINITIONS = {
    "ate": (
        "The run's co-primary AIPW ATE and 95% CI (stage2/causal_estimate.json) against the "
        "true ATE over the same rows: the mean true ITE over the effect-eligible held-out "
        "rows (finite aipw_score). bias = estimate - truth; covered = the CI contains the "
        "truth. Across seeds: mean bias, RMSE = sqrt(mean(bias^2)) and coverage rate."
    ),
    "ato": (
        "The run's augmented overlap-weighted ATO and 95% CI against the true overlap-"
        "weighted effect sum(e(1-e) tau) / sum(e(1-e)) over the held-out rows the ATO uses, "
        "with e the true propensity (true_treatment_prob). OCI weights by its estimated "
        "propensity, so its target differs from this one; the same effect weighted by the "
        "estimated propensity is reported beside it (ato_true_estimated_weights)."
    ),
    "cate": (
        "RMSE and Pearson and Spearman correlations of estimated_cate with the true ITE over "
        "held-out rows with an estimate; constant-ATE baseline: the RMSE of predicting the "
        "run's ATE estimate for every patient; relative improvement = 1 - RMSE / baseline "
        "RMSE (positive means the CATEs beat the constant)."
    ),
    "gates": (
        "The run's GATES groups (within-fold quintiles of estimated_cate) are rebuilt with "
        "the diagnostics' own grouping function and seed and checked against the stored "
        "group means; coverage is the share of GATES 95% intervals that contain the true "
        "mean ITE of the group's patients, per fold and pooled."
    ),
    "oracle_ceiling": (
        "Stage 2's final estimation (estimate_outer_fold) refitted on the true covariates "
        "with the run's folds, seeds, nuisance learner, policy and estimators; the same ATE, "
        "ATO, CATE and GATES metrics."
    ),
    "selection": (
        "From the attribution report's held-out R^2 recovery: modifier recall is the share "
        "of true effect modifiers (dual-role included) that some selected modifier recovers; "
        "modifier precision is the share of selected modifiers that recover a true effect "
        "modifier; confounder recall is the share of true confounders recovered by the "
        "nuisance set; instruments and prognostic-only variables are counted in the nuisance "
        "set and among the modifiers; U proxies report the largest held-out R^2 of each "
        "unmeasured confounder on a selected variable. Pooled over folds as sums of "
        "numerators over sums of denominators."
    ),
    "sensitivity": (
        "Unmeasured confounder U's strength from the oracle columns with the DoubleML "
        "benchmark formulas of stage2_sensitivity.benchmark_gains: the long model is the true "
        "propensity and potential outcomes (with U), the short model integrates U out "
        "(Gauss-Hermite) given the measured covariates; cf_y, cf_d and rho are its partial "
        "R^2 with the outcome, with the Riesz representer, and its bias ratio. The run's "
        "omitted-variable-bias bound at (cf_y, cf_d) with rho = 1 is checked for containing "
        "the true ATE, next to the run's E-values, RV and RVa."
    ),
    "leak": (
        "S6 text-column versus dated-notes variants: ATE bias, the share of held-out patients "
        "with estimated propensity outside [0.05, 0.95], and whether any selected variable "
        "recovers a true_post_treatment_* column (held-out R^2 at the recovery threshold)."
    ),
}
CAVEATS = (
    (
        "The true ATE and group means are sample quantities over the held-out rows; the AIPW "
        "intervals target the population ATE, so coverage of the sample truth is approximate."
    ),
    (
        "OCI's ATO weights patients by its estimated propensity; the ATO truth uses the true "
        "propensity, so that comparison is approximate."
    ),
    (
        "The oracle-covariate ceiling assumes perfect documentation, extraction and selection; "
        "it is not an upper bound in every sample."
    ),
    (
        "Recovery is measured by held-out R^2 of the oracle column on a variable's encoded "
        "columns, at the recovery threshold; a variable can recover a concept without naming it."
    ),
    (
        "U's strength is computed from the data-generating equations; the run's bound also "
        "absorbs measurement error in the text-derived confounders, which U's strength does not "
        "include."
    ),
    (
        "S6 cells at the generated corpus size are the whole corpus for every seed, so their "
        "replicates share one data draw and differ only in folds and model seeds."
    ),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha256_json(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_text_if_changed(path: Path, text: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    if path.is_file() and path.read_bytes() == data:
        return False
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)
    return True


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _parquet_rows(path: Path) -> int:
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path).metadata.num_rows)


# ---------------------------------------------------------------------------
# Registry


@dataclass(frozen=True)
class RunVariant:
    """One workflow run per cell: column overrides and an optional ``notes`` block."""

    name: str
    description: str = ""
    columns: Mapping[str, str] = field(default_factory=dict)
    notes: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class Scenario:
    id: str
    family: str
    label: str
    factor: str
    materialization: str
    sizes: tuple[int, ...]
    source: Mapping[str, Any]
    dgp: Mapping[str, Any] | None
    generation: Mapping[str, Any] | None
    variants: tuple[RunVariant, ...]

    @property
    def minimum_source_patients(self) -> int:
        return int(self.source["minimum_patients"])

    @property
    def needs_generated_corpus(self) -> bool:
        return self.materialization != "reequation"

    @property
    def unmeasured_strength(self) -> float | None:
        if not self.dgp or not int(self.dgp.get("unmeasured_confounders") or 0):
            return None
        return _float(self.dgp.get("unmeasured_strength", 1.0))

    def role_feature_count(self) -> int:
        """Source features the DGP's roles require."""

        roles = (self.dgp or {}).get("roles") or {}
        return sum(
            len(value) if isinstance(value, (list, tuple)) else int(value)
            for value in roles.values()
        )


@dataclass(frozen=True)
class Registry:
    path: Path
    sha256: str
    scenarios: dict[str, Scenario]

    def select(self, spec: str | Sequence[str] | None) -> list[Scenario]:
        """Scenarios named by id or family (``S4`` selects S4a and S4b); all when omitted."""

        if spec is None or (isinstance(spec, str) and spec.strip().lower() in {"", "all"}):
            return list(self.scenarios.values())
        names = (
            [part.strip() for part in spec.split(",") if part.strip()]
            if isinstance(spec, str)
            else [str(part) for part in spec]
        )
        selected: list[str] = []
        for name in names:
            matches = [
                scenario.id
                for scenario in self.scenarios.values()
                if scenario.id == name or scenario.family == name
            ]
            if not matches:
                raise ValueError(
                    f"unknown scenario {name!r}; registry scenarios: {sorted(self.scenarios)}"
                )
            selected.extend(match for match in matches if match not in selected)
        return [self.scenarios[name] for name in self.scenarios if name in selected]


def load_registry(path: Path | None = None) -> Registry:
    path = Path(path or DEFAULT_REGISTRY).expanduser().resolve()
    payload = _read_json(path)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise ValueError(f"{path} is not an {REGISTRY_SCHEMA_VERSION} scenario registry")
    scenarios: dict[str, Scenario] = {}
    for raw in payload.get("scenarios") or []:
        scenario_id = str(raw["id"])
        if scenario_id in scenarios:
            raise ValueError(f"duplicate scenario id {scenario_id!r}")
        materialization = str(raw["materialization"])
        if materialization not in MATERIALIZATIONS:
            raise ValueError(f"scenario {scenario_id}: materialization must be one of {MATERIALIZATIONS}")
        sizes = tuple(int(size) for size in raw["sizes"])
        if not sizes or len(set(sizes)) != len(sizes) or min(sizes) < 1:
            raise ValueError(f"scenario {scenario_id}: sizes must be distinct positive integers")
        source = dict(raw.get("source") or {})
        if int(source.get("minimum_patients") or 0) < max(sizes):
            raise ValueError(
                f"scenario {scenario_id}: source.minimum_patients must cover the largest size"
            )
        dgp = raw.get("dgp")
        if dgp is not None and not str((dgp or {}).get("name") or "").strip():
            raise ValueError(f"scenario {scenario_id}: dgp requires a name")
        if materialization != "subsample_of_generated" and dgp is None:
            raise ValueError(f"scenario {scenario_id}: re-equation requires a dgp block")
        generation = raw.get("generation")
        if materialization != "reequation":
            if not isinstance(generation, Mapping):
                raise ValueError(f"scenario {scenario_id}: a generated corpus needs a generation block")
            if int(generation.get("dataset_size") or 0) < int(source["minimum_patients"]):
                raise ValueError(
                    f"scenario {scenario_id}: generation.dataset_size is below the minimum source size"
                )
            # Validate every generated corpus before an earlier scenario spends
            # time writing notes. Use the generator's own rules, and validate a
            # copy because validation can normalize nested factor settings.
            from synthetic_data.config import SyntheticDataConfig

            try:
                SyntheticDataConfig.from_dict(copy.deepcopy(dict(generation))).validate()
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"scenario {scenario_id}: invalid generation settings: {error}"
                ) from error
        variants = tuple(
            RunVariant(
                name=str(item["name"]),
                description=str(item.get("description") or ""),
                columns=dict(item.get("columns") or {}),
                notes=dict(item["notes"]) if item.get("notes") else None,
            )
            for item in raw.get("run_variants") or [{"name": DEFAULT_VARIANT}]
        )
        if len({variant.name for variant in variants}) != len(variants):
            raise ValueError(f"scenario {scenario_id}: run variant names must be distinct")
        scenarios[scenario_id] = Scenario(
            id=scenario_id,
            family=str(raw.get("family") or scenario_id),
            label=str(raw.get("label") or scenario_id),
            factor=str(raw.get("factor") or ""),
            materialization=materialization,
            sizes=sizes,
            source=source,
            dgp=copy.deepcopy(dict(dgp)) if dgp is not None else None,
            generation=copy.deepcopy(dict(generation)) if generation is not None else None,
            variants=variants,
        )
    if not scenarios:
        raise ValueError(f"{path} lists no scenarios")
    return Registry(path=path, sha256=file_sha256(path), scenarios=scenarios)


# ---------------------------------------------------------------------------
# Seeds, the confirmatory guard and the ledger


class SeedGuardError(RuntimeError):
    """A confirmatory seed was requested without authorization, or reused."""


@dataclass(frozen=True)
class SeedRegistry:
    path: Path
    sha256: str
    development: tuple[int, ...]
    confirmatory: tuple[int, ...]
    rule: str

    def seed_set(self, seed: int) -> str:
        if seed in self.confirmatory:
            return "confirmatory"
        if seed in self.development:
            return "development"
        return "unregistered"


def load_seed_registry(path: Path | None = None) -> SeedRegistry:
    path = Path(path or DEFAULT_SEEDS).expanduser().resolve()
    payload = _read_json(path)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != SEEDS_SCHEMA_VERSION:
        raise ValueError(f"{path} is not an {SEEDS_SCHEMA_VERSION} seed file")
    development = tuple(int(seed) for seed in payload.get("development") or [])
    confirmatory = tuple(int(seed) for seed in payload.get("confirmatory") or [])
    if not development or not confirmatory:
        raise ValueError(f"{path} must list development and confirmatory seeds")
    if set(development) & set(confirmatory):
        raise ValueError(f"{path}: development and confirmatory seeds must be disjoint")
    return SeedRegistry(
        path=path,
        sha256=file_sha256(path),
        development=development,
        confirmatory=confirmatory,
        rule=str(payload.get("rule") or ""),
    )


def resolve_seeds(
    seeds: SeedRegistry,
    requested: str | Sequence[int] | None = None,
    *,
    confirmatory: bool = False,
    include_development: bool = False,
) -> list[tuple[int, str]]:
    """(seed, seed set) pairs; confirmatory seeds require ``confirmatory``.

    Without ``requested`` the development seeds are used, or the confirmatory
    seeds when ``confirmatory`` is set (``include_development`` keeps the
    development seeds as well, as ``aggregate`` does).
    """

    if requested is None or (isinstance(requested, str) and not requested.strip()):
        chosen = list(seeds.confirmatory) if confirmatory else list(seeds.development)
        if confirmatory and include_development:
            chosen = [*seeds.development, *seeds.confirmatory]
    else:
        text = requested if isinstance(requested, str) else ",".join(map(str, requested))
        if text.strip().lower() in SEED_SETS:
            name = text.strip().lower()
            if name == "confirmatory" and not confirmatory:
                raise SeedGuardError(
                    "confirmatory seeds require --confirmatory; tune only on development seeds"
                )
            chosen = list(getattr(seeds, name))
        else:
            chosen = [int(part) for part in text.split(",") if part.strip()]
    pairs = []
    for seed in dict.fromkeys(chosen):
        kind = seeds.seed_set(int(seed))
        if kind == "confirmatory" and not confirmatory:
            raise SeedGuardError(
                f"seed {seed} is a confirmatory seed ({seeds.path.name}); pass --confirmatory to "
                "use it. Tune only on development seeds."
            )
        pairs.append((int(seed), kind))
    if not pairs:
        raise ValueError("no seeds selected")
    return pairs


def git_state(root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    """The checkout's commit and whether tracked files differ from it."""

    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *args], capture_output=True, text=True, check=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip()

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit or "unknown", "dirty": None if status is None else bool(status)}


def read_ledger(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_ledger(
    path: Path, *, commit: str, scenarios: Iterable[str], force: bool
) -> list[dict[str, Any]]:
    """Earlier confirmatory uses of these scenarios at this commit; raises unless ``force``."""

    wanted = set(scenarios)
    previous = [
        entry for entry in read_ledger(path)
        if entry.get("git_commit") == commit and entry.get("scenario") in wanted
    ]
    if previous and not force:
        used = sorted({str(entry["scenario"]) for entry in previous})
        raise SeedGuardError(
            f"confirmatory seeds were already used for {used} at commit {commit} "
            f"({path}); pass --force to use them again"
        )
    return previous


def append_ledger(path: Path, entries: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(dict(entry), sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Cells and layout


@dataclass(frozen=True)
class Cell:
    scenario: Scenario
    size: int
    seed: int
    seed_set: str

    @property
    def id(self) -> str:
        return f"{self.scenario.id}/n{self.size}/seed{self.seed}"

    def directory(self, bench: Path) -> Path:
        return Path(bench) / self.scenario.id / f"n{self.size}" / f"seed{self.seed}"

    def data_dir(self, bench: Path) -> Path:
        return self.directory(bench) / "data"

    def config_path(self, bench: Path, variant: str) -> Path:
        return self.directory(bench) / "runs" / variant / "config.json"

    def run_dir(self, bench: Path, variant: str) -> Path:
        return self.directory(bench) / "runs" / variant / "output"


def _parse_sizes(value: str | Sequence[int] | None) -> set[int] | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, str):
        return {int(part) for part in value.split(",") if part.strip()}
    return {int(part) for part in value}


def expand_cells(
    scenarios: Sequence[Scenario],
    seeds: Sequence[tuple[int, str]],
    sizes: set[int] | None = None,
) -> list[Cell]:
    return [
        Cell(scenario, size, seed, kind)
        for scenario in scenarios
        for size in scenario.sizes
        if sizes is None or size in sizes
        for seed, kind in seeds
    ]


# ---------------------------------------------------------------------------
# materialize


def _source_identity(directory: Path) -> dict[str, Any]:
    directory = Path(directory).expanduser().resolve(strict=True)
    missing = [name for name in DATA_FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{directory} lacks {missing}")
    return {
        "directory": str(directory),
        "patients": _parquet_rows(directory / "dataset.parquet"),
        "sha256": {name: file_sha256(directory / name) for name in DATA_FILES},
    }


def _materialization_record(cell: Cell, bench: Path) -> Path:
    return cell.directory(bench) / "materialize.json"


def _existing_materialization(cell: Cell, bench: Path, identity: Mapping[str, Any]) -> bool:
    """True when the cell already holds exactly this materialization; raises on conflicts."""

    record_path = _materialization_record(cell, bench)
    data_dir = cell.data_dir(bench)
    if record_path.is_file():
        record = _read_json(record_path)
        outputs = record.get("outputs_sha256") or {}
        intact = all(
            (data_dir / name).is_file() and file_sha256(data_dir / name) == digest
            for name, digest in outputs.items()
        )
        if record.get("identity") == dict(identity) and outputs and intact:
            return True
        raise FileExistsError(
            f"{cell.directory(bench)} already holds a different or modified cohort "
            f"({record_path.name}); use another --out directory or remove the cell"
        )
    if data_dir.exists() and any(data_dir.iterdir()):
        raise FileExistsError(
            f"{data_dir} holds files that this harness did not record; remove them or use "
            "another --out directory"
        )
    return False


def _write_materialization(cell: Cell, bench: Path, identity: Mapping[str, Any], files: Sequence[str]) -> dict[str, Any]:
    data_dir = cell.data_dir(bench)
    missing = [name for name in files if not (data_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"materialization of {cell.id} did not write {missing}")
    record = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "created_at": _now(),
        "cell": cell.id,
        "seed_set": cell.seed_set,
        "identity": dict(identity),
        "outputs_sha256": {name: file_sha256(data_dir / name) for name in files},
    }
    write_json(_materialization_record(cell, bench), record)
    return record


def _dgp_identity(scenario: Scenario) -> dict[str, Any]:
    return {
        "dgp": scenario.dgp,
        "dgp_sha256": _sha256_json(scenario.dgp) if scenario.dgp is not None else None,
    }


def materialize_reequation(
    cell: Cell, bench: Path, source: Mapping[str, Any], *, method: str
) -> str:
    """One re-equation cell through the generator contract; returns its status."""

    identity = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "scenario": cell.scenario.id,
        "method": method,
        **_dgp_identity(cell.scenario),
        "size": cell.size,
        "seed": cell.seed,
        "source": dict(source),
    }
    if _existing_materialization(cell, bench, identity):
        return "already_materialized"
    try:
        from synthetic_data.reequation import reequation_cohort
    except ImportError as error:  # pragma: no cover - depends on the generator version
        raise RuntimeError(
            "synthetic_data.reequation is not available in this checkout; the re-equation "
            "scenarios need the generator's re-equation module"
        ) from error
    data_dir = cell.data_dir(bench)
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        reequation_cohort(
            Path(source["directory"]),
            copy.deepcopy(dict(cell.scenario.dgp or {})),
            size=cell.size,
            seed=cell.seed,
            output_dir=data_dir,
        )
    except Exception as error:
        raise RuntimeError(
            f"re-equation of {cell.id} from {source['directory']} failed: "
            f"{type(error).__name__}: {error}; remove {data_dir} before retrying"
        ) from error
    _write_materialization(cell, bench, identity, REEQUATION_OUTPUTS)
    return "materialized"


def _variant_requirements(scenario: Scenario) -> tuple[list[str], list[str]]:
    """(dataset columns, data files) the scenario's run variants read."""

    columns, files = [], []
    for variant in scenario.variants:
        text = variant.columns.get("text")
        if text and not variant.notes:
            columns.append(str(text))
        if variant.notes:
            files.append(str(variant.notes["path"]))
            columns.append(str(variant.notes.get("index_date_column") or "index_date"))
    return sorted(set(columns)), sorted(set(files))


def materialize_subsample(cell: Cell, bench: Path, source: Mapping[str, Any]) -> str:
    """A seeded subsample (without replacement) of a generated corpus and its notes table."""

    columns, files = _variant_requirements(cell.scenario)
    identity = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "scenario": cell.scenario.id,
        "method": "subsample_of_generated",
        "size": cell.size,
        "seed": cell.seed,
        "source": dict(source),
        "extra_files_sha256": {
            name: file_sha256(Path(source["directory"]) / name) for name in files
        },
    }
    if _existing_materialization(cell, bench, identity):
        return "already_materialized"
    directory = Path(source["directory"])
    dataset = pd.read_parquet(directory / "dataset.parquet")
    missing = [column for column in columns if column not in dataset.columns]
    if missing:
        raise ValueError(f"the generated corpus {directory} lacks columns {missing}")
    rows = len(dataset)
    if cell.size > rows:
        raise ValueError(f"{cell.id}: the generated corpus has only {rows} patients")
    chosen = np.sort(np.random.default_rng(cell.seed).choice(rows, size=cell.size, replace=False))
    data_dir = cell.data_dir(bench)
    data_dir.mkdir(parents=True, exist_ok=True)
    subset = dataset.iloc[chosen].reset_index(drop=True)
    subset.to_parquet(data_dir / "dataset.parquet", index=False)
    written = list(DATA_FILES)
    for name in ("metadata.json", "generation_config.json"):
        shutil.copy2(directory / name, data_dir / name)
    for variant in cell.scenario.variants:
        if not variant.notes:
            continue
        notes_name = str(variant.notes["path"])
        unit = str((variant.notes.get("columns") or {}).get("unit_id") or "patient_id")
        notes = pd.read_parquet(directory / notes_name)
        kept = notes[notes[unit].isin(set(subset[unit].tolist()))].reset_index(drop=True)
        kept.to_parquet(data_dir / notes_name, index=False)
        written.append(notes_name)
    manifest = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "method": "subsample_of_generated",
        "source": dict(source),
        "size": cell.size,
        "seed": cell.seed,
        "source_row_positions_sha256": hashlib.sha256(chosen.astype(np.int64).tobytes()).hexdigest(),
        "rows_identical_to_source": bool(cell.size == rows),
        "treatment_and_outcome": "as generated (re-equation refuses corpora with post-treatment notes)",
    }
    write_json(data_dir / "subsample_manifest.json", manifest)
    written.append("subsample_manifest.json")
    _write_materialization(cell, bench, identity, written)
    return "materialized"


def generator_instructions(scenario: Scenario, bench: Path) -> dict[str, Any]:
    """Write the generator configuration of an S5/S6-type scenario and return the commands."""

    root = Path(bench) / scenario.id / "generation"
    config_path = root / "generator_config.json"
    corpus = root / "corpus"
    _write_text_if_changed(
        config_path, json.dumps(scenario.generation, indent=2, sort_keys=True) + "\n"
    )
    generate = (
        f"python -m synthetic_data.cli --config {config_path} --api-url URL --model MODEL "
        f"--output-dir {corpus}"
    )
    use = (
        f"python {_script('benchmark.py')} materialize --scenarios {scenario.id} "
        f"--generated {corpus} --out {bench}"
    )
    return {
        "scenario": scenario.id,
        "generator_config": str(config_path),
        "corpus": str(corpus),
        "commands": [generate, use],
        "minimum_patients": scenario.minimum_source_patients,
    }


def _script(name: str) -> str:
    path = REPOSITORY_ROOT / "scripts" / name
    return str(path) if path.is_file() else f"scripts/{name}"


def materialize(
    *,
    out: Path,
    scenarios: str | None = None,
    source: Path | None = None,
    generated: Path | None = None,
    sizes: str | None = None,
    seeds: str | None = None,
    confirmatory: bool = False,
    registry_path: Path | None = None,
    seeds_path: Path | None = None,
) -> dict[str, Any]:
    registry = load_registry(registry_path)
    seed_registry = load_seed_registry(seeds_path)
    selected = registry.select(scenarios)
    seed_pairs = resolve_seeds(seed_registry, seeds, confirmatory=confirmatory)
    size_filter = _parse_sizes(sizes)
    bench = Path(out).expanduser().resolve()
    generated_scenarios = [scenario for scenario in selected if scenario.needs_generated_corpus]
    if generated is not None and len(generated_scenarios) != 1:
        raise ValueError(
            "--generated applies to exactly one generated-corpus scenario; select it with "
            f"--scenarios (generated-corpus scenarios selected: {[s.id for s in generated_scenarios]})"
        )
    reequation = [scenario for scenario in selected if not scenario.needs_generated_corpus]
    if reequation and source is None:
        raise ValueError(
            f"scenarios {[s.id for s in reequation]} need --source (a re-equation source cohort)"
        )
    results: list[dict[str, Any]] = []
    instructions: list[dict[str, Any]] = []
    source_cache: dict[str, dict[str, Any]] = {}

    def identity_of(directory: Path) -> dict[str, Any]:
        key = str(Path(directory).expanduser().resolve())
        if key not in source_cache:
            source_cache[key] = _source_identity(Path(directory))
        return source_cache[key]

    for scenario in selected:
        if scenario.needs_generated_corpus and generated is None:
            instruction = generator_instructions(scenario, bench)
            instructions.append(instruction)
            print(f"\n# {scenario.id} ({scenario.label}) needs LLM text: generate a corpus of "
                  f"at least {scenario.minimum_source_patients} patients, then materialize it")
            for command in instruction["commands"]:
                print(command)
            continue
        origin = identity_of(generated if scenario.needs_generated_corpus else source)
        for cell in expand_cells([scenario], seed_pairs, size_filter):
            if cell.size > origin["patients"]:
                status = "skipped_source_too_small"
            elif scenario.materialization == "subsample_of_generated":
                status = materialize_subsample(cell, bench, origin)
            else:
                status = materialize_reequation(cell, bench, origin, method=scenario.materialization)
            results.append({
                "cell": cell.id, "seed_set": cell.seed_set, "status": status,
                "data_dir": str(cell.data_dir(bench)),
            })
            print(f"{cell.id}: {status}")
        if origin["patients"] < scenario.minimum_source_patients:
            print(
                f"note: {scenario.id} needs a source of at least "
                f"{scenario.minimum_source_patients} patients for every size; "
                f"{origin['directory']} has {origin['patients']}"
            )
    summary = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "bench": str(bench),
        "registry_sha256": registry.sha256,
        "cells": results,
        "generator_instructions": instructions,
    }
    return summary


# ---------------------------------------------------------------------------
# configs


_PATH_KEYS = ("stage1_template", "neural_query_template")
_MODEL_KEYS = (("models", "htr"), ("models", "embeddings"), (None, "htr_model"), (None, "embedding_model"))
_FLAT_COLUMNS = {
    "unit_id_column": "unit_id", "text_column": "text",
    "treatment_column": "treatment", "outcome_column": "outcome",
}
_FLAT_SCIENCE = ("outcome_type", "clinical_question", "outer_folds", "inner_folds", "seed")


def load_base_config(path: Path) -> tuple[dict[str, Any], Path]:
    path = Path(path).expanduser().resolve(strict=True)
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml  # optional, as for the workflow

        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        value = _read_json(path)
    if not isinstance(value, Mapping):
        raise ValueError("--base must be a JSON or YAML object")
    return dict(value), path.parent


def _rebased(value: Any, base: Path, *, model: bool) -> Any:
    text = str(value)
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return text
    if not model or text.startswith(("./", "../")) or (base / candidate).exists():
        return str((base / candidate).resolve())
    return text


def build_run_config(
    base: Mapping[str, Any],
    base_dir: Path,
    *,
    cell: Cell,
    variant: RunVariant,
    bench: Path,
) -> dict[str, Any]:
    """One workflow configuration: the base with this cell's dataset, output, seed and text input."""

    config = copy.deepcopy(dict(base))
    for key in _PATH_KEYS:
        if config.get(key):
            config[key] = _rebased(config[key], base_dir, model=False)
    for section, key in _MODEL_KEYS:
        holder = config.get(section) if section else config
        if isinstance(holder, dict) and holder.get(key):
            holder[key] = _rebased(holder[key], base_dir, model=True)
    columns = dict(config.get("columns") or {})
    science = dict(config.get("science") or {})
    for flat, key in _FLAT_COLUMNS.items():
        if flat in config:
            columns.setdefault(key, config.pop(flat))
    for key in _FLAT_SCIENCE:
        if key in config:
            science.setdefault(key, config.pop(key))
    columns.update({str(key): str(value) for key, value in variant.columns.items()})
    data_dir = cell.data_dir(bench)
    science["seed"] = int(cell.seed)
    outcome_type = _cohort_outcome_type(data_dir)
    if outcome_type:
        science["outcome_type"] = outcome_type
    config.pop("notes", None)
    if variant.notes:
        notes = copy.deepcopy(dict(variant.notes))
        notes["path"] = str(data_dir / str(notes["path"]))
        config["notes"] = notes
    config["dataset"] = str(data_dir / "dataset.parquet")
    config["output_dir"] = str(cell.run_dir(bench, variant.name))
    config["columns"] = columns
    config["science"] = science
    return config


def _cohort_outcome_type(data_dir: Path) -> str | None:
    for name, keys in (("metadata.json", ("config", "outcome_type")), ("generation_config.json", ("outcome_type",))):
        path = Path(data_dir) / name
        if not path.is_file():
            continue
        value: Any = _read_json(path)
        for key in keys:
            value = value.get(key) if isinstance(value, Mapping) else None
        if value in {"binary", "continuous"}:
            return str(value)
    return None


def write_configs(
    *,
    base: Path,
    out: Path,
    scenarios: str | None = None,
    sizes: str | None = None,
    seeds: str | None = None,
    confirmatory: bool = False,
    force: bool = False,
    registry_path: Path | None = None,
    seeds_path: Path | None = None,
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    registry = load_registry(registry_path)
    seed_registry = load_seed_registry(seeds_path)
    selected = registry.select(scenarios)
    seed_pairs = resolve_seeds(seed_registry, seeds, confirmatory=confirmatory)
    bench = Path(out).expanduser().resolve()
    base_config, base_dir = load_base_config(base)
    planned: list[dict[str, Any]] = []
    missing: list[str] = []
    for cell in expand_cells(selected, seed_pairs, _parse_sizes(sizes)):
        if not (cell.data_dir(bench) / "dataset.parquet").is_file():
            missing.append(cell.id)
            continue
        for variant in cell.scenario.variants:
            config = build_run_config(base_config, base_dir, cell=cell, variant=variant, bench=bench)
            path = cell.config_path(bench, variant.name)
            text = json.dumps(config, indent=2, sort_keys=True) + "\n"
            run_dir = cell.run_dir(bench, variant.name)
            if (
                path.is_file() and path.read_text(encoding="utf-8") != text
                and (run_dir / "run_config.json").exists()
            ):
                raise FileExistsError(
                    f"{run_dir} holds a run started from a different configuration; use another "
                    "--out directory"
                )
            planned.append({
                "cell": cell, "variant": variant.name, "path": path, "text": text,
                "config_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "run_dir": run_dir,
            })
    ledger = Path(ledger_path or DEFAULT_LEDGER)
    confirmatory_scenarios = sorted({
        item["cell"].scenario.id for item in planned if item["cell"].seed_set == "confirmatory"
    })
    git = git_state() if confirmatory_scenarios else {"commit": None, "dirty": None}
    if confirmatory_scenarios:
        check_ledger(ledger, commit=git["commit"], scenarios=confirmatory_scenarios, force=force)
    entries = []
    for item in planned:
        _write_text_if_changed(item["path"], item["text"])
        command = f"{RUN_COMMAND_ENV} python {_script('run_all_evidence.py')} --config {item['path']}"
        entries.append({
            "cell": item["cell"].id,
            "scenario": item["cell"].scenario.id,
            "size": item["cell"].size,
            "seed": item["cell"].seed,
            "seed_set": item["cell"].seed_set,
            "variant": item["variant"],
            "config": str(item["path"]),
            "config_sha256": item["config_sha256"],
            "output_dir": str(item["run_dir"]),
            "command": command,
        })
    if confirmatory_scenarios:
        ledger_entries = []
        for scenario in confirmatory_scenarios:
            mine = [entry for entry in entries
                    if entry["scenario"] == scenario and entry["seed_set"] == "confirmatory"]
            ledger_entries.append({
                "schema_version": LEDGER_SCHEMA_VERSION,
                "time": _now(),
                "scenario": scenario,
                "git_commit": git["commit"],
                "git_dirty": git["dirty"],
                "config_hash": _sha256_json(sorted(
                    [entry["cell"], entry["variant"], entry["config_sha256"]] for entry in mine
                )),
                "seeds": sorted({entry["seed"] for entry in mine}),
                "cells": sorted({entry["cell"] for entry in mine}),
                "base_config": str(Path(base).expanduser().resolve()),
                "base_config_sha256": file_sha256(Path(base).expanduser().resolve()),
                "bench": str(bench),
                "forced": bool(force),
            })
        append_ledger(ledger, ledger_entries)
    index_path = bench / "configs.json"
    existing = _read_json(index_path).get("runs", []) if index_path.is_file() else []
    keyed = {(item["cell"], item["variant"]): item for item in existing}
    keyed.update({(item["cell"], item["variant"]): item for item in entries})
    index = {
        "schema_version": CONFIGS_SCHEMA_VERSION,
        "updated_at": _now(),
        "registry_sha256": registry.sha256,
        "seeds_sha256": seed_registry.sha256,
        "base_config": str(Path(base).expanduser().resolve()),
        "runs": [keyed[key] for key in sorted(keyed)],
    }
    write_json(index_path, index)
    for cell_id in missing:
        print(f"skip {cell_id}: not materialized")
    print(f"wrote {len(entries)} run configuration(s); no run was launched")
    for entry in entries:
        print(entry["command"])
    return {"runs": entries, "not_materialized": missing, "index": str(index_path)}


# ---------------------------------------------------------------------------
# aggregate: run status and frozen run artifacts


def run_status(run_dir: Path) -> tuple[str, str]:
    """("complete" | "pending", detail) of one workflow output directory."""

    run_dir = Path(run_dir)
    stage2 = run_dir / "stage2"
    completion: Mapping[str, Any] = {}
    if (stage2 / "complete.json").is_file():
        try:
            completion = _read_json(stage2 / "complete.json")
        except (OSError, ValueError):
            completion = {}
    files = all(
        (stage2 / name).is_file()
        for name in ("cross_fitted_predictions.csv", "causal_estimate.json",
                     "heterogeneity_diagnostics.json")
    )
    if (
        completion.get("status") == "complete"
        and completion.get("phase", "causal_estimation") == "causal_estimation"
        and files
    ):
        return "complete", "complete"
    if not run_dir.exists():
        return "pending", "not_started"
    if completion.get("status") == "complete":
        phase = completion.get("phase", "causal_estimation")
        if phase == "causal_estimation":
            return "pending", "stage2_outputs_missing"
        return "pending", f"stage2_phase_{phase}"
    progress_path = run_dir / "progress.json"
    if progress_path.is_file():
        try:
            progress = _read_json(progress_path)
        except (OSError, ValueError):
            return "pending", "progress_unreadable"
        status = str(progress.get("status") or "unknown")
        component = progress.get("current_component")
        if status == "complete":
            return "pending", "stage2_not_run"
        return "pending", f"{status}{f'_{component}' if component else ''}"
    return "pending", "incomplete"


@dataclass
class RunArtifacts:
    run_dir: Path
    reader: FrozenArtifactReader
    run_config: dict[str, Any]
    stage2_config: dict[str, Any]
    predictions: pd.DataFrame
    causal_estimate: dict[str, Any]
    heterogeneity: dict[str, Any]
    splits: dict[int, dict[str, Any]]
    final_estimators: dict[int, str | None]
    lineage: Any
    hashes: dict[str, str] = field(default_factory=dict)


def _parse_splits(rows: Sequence[Mapping[str, Any]] | None) -> dict[int, dict[str, Any]]:
    """The frozen Stage 1 split file in the form Stage 2 passes to each fold."""

    splits: dict[int, dict[str, Any]] = {}
    for raw in rows or []:
        fold = int(raw["outer_fold"])
        if fold in splits:
            raise ValueError(f"duplicate split provenance for outer fold {fold}")
        fit_ids = [int(value) for value in raw["fit_row_ids"]]
        heldout_ids = [int(value) for value in raw["heldout_row_ids"]]
        if set(fit_ids) & set(heldout_ids):
            raise ValueError(f"outer fold {fold} has overlapping fit and held-out rows")
        splits[fold] = {
            "outer_fold": fold,
            "fit_row_ids": fit_ids,
            "heldout_row_ids": heldout_ids,
            "inner_splits": [
                {
                    "inner_fold": int(inner.get("inner_fold", index)),
                    "fit_row_ids": [int(value) for value in inner["fit_row_ids"]],
                    "heldout_row_ids": [int(value) for value in inner["heldout_row_ids"]],
                }
                for index, inner in enumerate(raw.get("inner_splits") or [], start=1)
            ],
        }
    return splits


def read_run_artifacts(run_dir: Path) -> RunArtifacts:
    """Read and hash every artifact the aggregate uses, then freeze the reader."""

    run_dir = Path(run_dir).expanduser().resolve(strict=True)
    reader = FrozenArtifactReader(run_dir)
    stage2 = run_dir / "stage2"
    run_config = reader.json(run_dir / "run_config.json") or {}
    stage2_config = reader.json(stage2 / "config.json")
    if not isinstance(stage2_config, Mapping):
        stage2_config = run_config.get("stage2") if isinstance(run_config.get("stage2"), Mapping) else {}
    predictions_bytes = reader.read_bytes(stage2 / "cross_fitted_predictions.csv")
    if predictions_bytes is None:
        raise FileNotFoundError(f"{stage2 / 'cross_fitted_predictions.csv'} is missing")
    predictions = pd.read_csv(io.BytesIO(predictions_bytes))
    if any(str(column).startswith("true_") for column in predictions.columns):
        raise RuntimeError("frozen Stage 2 predictions contain an oracle column")
    predictions = predictions.sort_values(ROW_ID, kind="mergesort").reset_index(drop=True)
    causal = reader.json(stage2 / "causal_estimate.json") or {}
    heterogeneity = reader.json(stage2 / "heterogeneity_diagnostics.json") or {}
    splits = _parse_splits(reader.jsonl(run_dir / "components" / "tfidf" / "split_provenance.jsonl"))
    final_estimators: dict[int, str | None] = {}
    for path in sorted(stage2.glob("outer_[0-9][0-9][0-9]/final_definitions.json")):
        payload = reader.json(path) or {}
        final_estimators[int(path.parent.name.removeprefix("outer_"))] = payload.get(
            "final_estimator"
        )
    lineage = collect_stage2_lineage(stage2, reader=reader, include_heldout=True)
    artifacts = RunArtifacts(
        run_dir=run_dir,
        reader=reader,
        run_config=dict(run_config),
        stage2_config=dict(stage2_config or {}),
        predictions=predictions,
        causal_estimate=dict(causal),
        heterogeneity=dict(heterogeneity),
        splits=splits,
        final_estimators=final_estimators,
        lineage=lineage,
    )
    # Every run artifact is now read and hashed; only then may oracle values load.
    artifacts.hashes = reader.freeze()
    return artifacts


# ---------------------------------------------------------------------------
# aggregate: oracle data


@dataclass
class BenchmarkOracle:
    data: OracleData
    values: pd.DataFrame
    metadata: dict[str, Any]
    ite_column: str | None
    potential_outcome_columns: tuple[str, str] | None
    unmeasured: list[dict[str, Any]]
    post_treatment_concepts: list[str]


def oracle_ite_column(columns: Iterable[Any], outcome_type: str) -> str | None:
    """``true_ite`` (outcome scale), or ``true_ite_prob`` for binary outcomes."""

    names = {str(column) for column in columns}
    if "true_ite" in names:
        return "true_ite"
    if "true_ite_prob" in names and str(outcome_type) == "binary":
        return "true_ite_prob"
    return None


def load_benchmark_oracle(
    dataset: Path,
    metadata: Path | None,
    *,
    readers: Sequence[FrozenArtifactReader],
    outcome_type: str,
    label_columns: Sequence[str],
) -> BenchmarkOracle:
    """Oracle concepts and effect columns; every reader must already be frozen.

    Concepts are the metadata features (as for the attribution report), each
    unmeasured confounder ``true_<u>`` and each ``true_post_treatment_*`` column.
    """

    base = load_oracle_data(dataset, metadata, readers=readers)
    from .attribution import _dataset_columns, _read_dataset_columns

    available = set(_dataset_columns(base.dataset))
    meta: dict[str, Any] = (
        json.loads(base.metadata.read_text(encoding="utf-8")) if base.metadata is not None else {}
    )
    known = {concept.column for concept in base.concepts}
    unmeasured = [
        dict(item) for item in meta.get("unmeasured_confounders") or []
        if isinstance(item, Mapping) and f"true_{item.get('name')}" in available
    ]
    extra = [
        OracleConcept(str(item["name"]), f"true_{item['name']}", "continuous",
                      ("unmeasured_confounder",))
        for item in unmeasured if f"true_{item['name']}" not in known
    ]
    post = sorted(column for column in available if column.startswith(POST_TREATMENT_PREFIX))
    extra.extend(
        OracleConcept(column[len("true_"):], column, "categorical", ("post_treatment",))
        for column in post if column not in known
    )
    ite = oracle_ite_column(available, outcome_type)
    if outcome_type == "binary" and {"true_y0_prob", "true_y1_prob"} <= available and not {
        "true_y0", "true_y1"
    } <= available:
        potential: tuple[str, str] | None = ("true_y0_prob", "true_y1_prob")
    elif {"true_y0", "true_y1"} <= available:
        potential = ("true_y0", "true_y1")
    else:
        potential = None
    wanted = [
        *(concept.column for concept in extra),
        *([ite] if ite else []),
        *(["true_treatment_prob"] if "true_treatment_prob" in available else []),
        *(potential or ()),
        *(column for column in label_columns if column in available),
        *(concept.column for concept in base.concepts),
    ]
    values = _read_dataset_columns(base.dataset, list(dict.fromkeys(wanted)))
    frame = base.frame
    if extra:
        frame = frame.merge(values[[ROW_ID, *(c.column for c in extra)]], on=ROW_ID, how="left")
    data = OracleData(
        dataset=base.dataset,
        dataset_sha256=base.dataset_sha256,
        metadata=base.metadata,
        metadata_sha256=base.metadata_sha256,
        concepts=[*base.concepts, *extra],
        frame=frame,
        absent_concepts=base.absent_concepts,
    )
    return BenchmarkOracle(
        data=data,
        values=values,
        metadata=meta,
        ite_column=ite,
        potential_outcome_columns=potential,
        unmeasured=unmeasured,
        post_treatment_concepts=[column[len("true_"):] for column in post],
    )


def _aligned(values: pd.DataFrame, row_ids: Sequence[int], column: str) -> np.ndarray:
    indexed = values.set_index(ROW_ID)
    ids = [int(row) for row in row_ids]
    missing = sorted(set(ids) - set(indexed.index.tolist()))
    if missing:
        raise ValueError(
            f"prediction rows {missing[:5]} are outside the oracle dataset; evaluate with the "
            "dataset the run used"
        )
    return pd.to_numeric(indexed.loc[ids, column], errors="coerce").to_numpy(dtype=float)


def check_alignment(predictions: pd.DataFrame, oracle: BenchmarkOracle, run_config: Mapping[str, Any]) -> None:
    """The oracle dataset holds the run's rows: treatment and outcome agree for every row."""

    for prediction_column, key in (("treatment", "treatment_column"), ("outcome", "outcome_column")):
        dataset_column = str(run_config.get(key) or "")
        if dataset_column not in oracle.values.columns or prediction_column not in predictions:
            continue
        expected = _aligned(oracle.values, predictions[ROW_ID], dataset_column)
        observed = pd.to_numeric(predictions[prediction_column], errors="coerce").to_numpy(float)
        if not np.allclose(expected, observed, equal_nan=True):
            raise ValueError(
                f"the oracle dataset's {dataset_column} differs from the run's predictions; "
                "evaluate with the dataset the run used"
            )


# ---------------------------------------------------------------------------
# aggregate: metrics (pure functions of frozen predictions and oracle values)


def _estimate_section(section: Any) -> dict[str, Any]:
    section = section if isinstance(section, Mapping) else {}
    interval = section.get("confidence_interval_95")
    interval = list(interval) if isinstance(interval, (list, tuple)) and len(interval) == 2 else [None, None]
    return {
        "estimate": _float(section.get("estimate")),
        "standard_error": _float(section.get("standard_error")),
        "ci_low": _float(interval[0]),
        "ci_high": _float(interval[1]),
    }


def _covers(low: float | None, high: float | None, truth: float | None) -> bool | None:
    if low is None or high is None or truth is None:
        return None
    return bool(low <= truth <= high)


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        return np.full(len(frame), np.nan)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def ate_metrics(predictions: pd.DataFrame, truth: np.ndarray, section: Any) -> dict[str, Any]:
    """The ATE against the mean true ITE over the effect-eligible held-out rows."""

    estimate = _estimate_section(section)
    eligible = np.isfinite(_numeric(predictions, "aipw_score")) & np.isfinite(truth)
    true_ate = float(np.mean(truth[eligible])) if eligible.any() else None
    bias = (
        estimate["estimate"] - true_ate
        if estimate["estimate"] is not None and true_ate is not None else None
    )
    width = (
        estimate["ci_high"] - estimate["ci_low"]
        if estimate["ci_low"] is not None and estimate["ci_high"] is not None else None
    )
    return {
        "rows": int(eligible.sum()),
        "true": true_ate,
        **estimate,
        "bias": bias,
        "ci_width": width,
        "covered": _covers(estimate["ci_low"], estimate["ci_high"], true_ate),
    }


def ato_metrics(
    predictions: pd.DataFrame, truth: np.ndarray, true_propensity: np.ndarray | None, section: Any
) -> dict[str, Any]:
    """The ATO against sum(e(1-e) tau) / sum(e(1-e)) with the true propensity e."""

    estimate = _estimate_section(section)
    finite = np.logical_and.reduce([
        np.isfinite(_numeric(predictions, column))
        for column in ("outcome", "treatment", "propensity", "mu0", "mu1")
    ]) & np.isfinite(truth)
    result: dict[str, Any] = {"rows": int(finite.sum()), **estimate}

    def weighted(propensity: np.ndarray) -> float | None:
        usable = finite & np.isfinite(propensity)
        weights = propensity[usable] * (1.0 - propensity[usable])
        return float(np.sum(weights * truth[usable]) / np.sum(weights)) if weights.sum() > 0 else None

    true_ato = weighted(true_propensity) if true_propensity is not None else None
    at_estimated = weighted(_numeric(predictions, "propensity"))
    result.update(
        true=true_ato,
        true_estimated_weights=at_estimated,
        bias=(
            estimate["estimate"] - true_ato
            if estimate["estimate"] is not None and true_ato is not None else None
        ),
        covered=_covers(estimate["ci_low"], estimate["ci_high"], true_ato),
        covered_estimated_weights=_covers(estimate["ci_low"], estimate["ci_high"], at_estimated),
    )
    return result


def _correlation(left: np.ndarray, right: np.ndarray, *, rank: bool) -> float | None:
    if len(left) < 3:
        return None
    if rank:
        left = pd.Series(left).rank(method="average").to_numpy(float)
        right = pd.Series(right).rank(method="average").to_numpy(float)
    if np.std(left) <= 0 or np.std(right) <= 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def cate_metrics(
    predictions: pd.DataFrame, truth: np.ndarray, ate_estimate: float | None
) -> dict[str, Any]:
    """RMSE and correlation of estimated_cate with the true ITE, and the constant-ATE baseline."""

    cate = _numeric(predictions, "estimated_cate")
    rows = np.isfinite(cate) & np.isfinite(truth)
    if not rows.any():
        return {"rows": 0, "status": "unavailable", "reason": "no held-out row has an estimate"}
    error = cate[rows] - truth[rows]
    rmse = float(np.sqrt(np.mean(error ** 2)))
    baseline = (
        float(np.sqrt(np.mean((ate_estimate - truth[rows]) ** 2)))
        if ate_estimate is not None else None
    )
    return {
        "status": "ok",
        "rows": int(rows.sum()),
        "rmse": rmse,
        "pearson": _correlation(cate[rows], truth[rows], rank=False),
        "spearman": _correlation(cate[rows], truth[rows], rank=True),
        "constant_ate_rmse": baseline,
        "relative_improvement": (1.0 - rmse / baseline) if baseline else None,
        "true_ite_sd": float(np.std(truth[rows])),
        "estimated_cate_sd": float(np.std(cate[rows])),
    }


def gates_coverage(
    predictions: pd.DataFrame, truth: np.ndarray, heterogeneity: Mapping[str, Any]
) -> dict[str, Any]:
    """Coverage of the true group-mean ITE by each GATES 95% interval, per fold and pooled."""

    from ..inference.stage2_diagnostics import cate_groups

    section = heterogeneity.get("gates") if isinstance(heterogeneity, Mapping) else None
    inputs = heterogeneity.get("inputs") if isinstance(heterogeneity, Mapping) else None
    if not isinstance(section, Mapping) or not isinstance(inputs, Mapping):
        return {"status": "unavailable", "reason": "heterogeneity diagnostics have no GATES"}
    groups = int(inputs.get("cate_groups") or 5)
    seed_base = int(inputs.get("bootstrap_seed_base") or 0)
    frame = pd.DataFrame({
        ROW_ID: predictions[ROW_ID].astype(int).to_numpy(),
        "outer_fold": predictions["outer_fold"].astype(int).to_numpy(),
        "psi": _numeric(predictions, "aipw_score"),
        "cate": _numeric(predictions, "estimated_cate"),
        "truth": truth,
    }).sort_values(ROW_ID, kind="mergesort")
    ranked = []
    per_fold: dict[str, Any] = {}
    stored = section.get("per_fold") or {}
    for fold, part in frame.groupby("outer_fold", sort=True):
        rows = part[np.isfinite(part["psi"]) & np.isfinite(part["cate"])].reset_index(drop=True)
        record = stored.get(str(int(fold))) or {}
        if record.get("status") != "ok" or len(rows) < max(groups, 4):
            per_fold[str(int(fold))] = {"status": "unavailable",
                                        "reason": record.get("reason") or "no GATES for this fold"}
            continue
        rows["group"] = cate_groups(rows["cate"], groups=groups, seed=seed_base + int(fold))
        per_fold[str(int(fold))] = _group_coverage(rows, record, groups)
        ranked.append(rows)
    pooled_record = section.get("pooled") or {}
    if ranked and pooled_record.get("status") == "ok":
        pooled = _group_coverage(pd.concat(ranked, ignore_index=True), pooled_record, groups)
    else:
        pooled = {"status": "unavailable", "reason": pooled_record.get("reason") or "no pooled GATES"}
    fold_hits = [g["covered"] for f in per_fold.values() if f.get("status") == "ok"
                 for g in f["groups"] if g["covered"] is not None]
    return {
        "status": "ok" if pooled.get("status") == "ok" or fold_hits else "unavailable",
        "groups": groups,
        "per_fold": per_fold,
        "pooled": pooled,
        "fold_intervals": len(fold_hits),
        "fold_coverage": float(np.mean(fold_hits)) if fold_hits else None,
    }


def _group_coverage(rows: pd.DataFrame, record: Mapping[str, Any], groups: int) -> dict[str, Any]:
    stored = {int(item["group"]): item for item in record.get("groups") or []}
    output = []
    for label in range(1, groups + 1):
        members = rows[rows["group"] == label]
        item = stored.get(label) or {}
        interval = item.get("confidence_interval_95")
        estimate = _float(item.get("estimate"))
        recomputed = float(members["psi"].mean()) if len(members) else None
        if estimate is not None and recomputed is not None and not math.isclose(
            estimate, recomputed, rel_tol=1e-6, abs_tol=1e-9
        ):
            return {"status": "unavailable",
                    "reason": "rebuilt GATES groups do not match the stored group means"}
        truth = float(members["truth"].mean()) if len(members) else None
        low, high = (interval if isinstance(interval, list) and len(interval) == 2 else (None, None))
        output.append({
            "group": label,
            "rows": len(members),
            "estimate": estimate,
            "ci_low": _float(low),
            "ci_high": _float(high),
            "true_mean_ite": truth,
            "covered": _covers(_float(low), _float(high), truth),
        })
    hits = [item["covered"] for item in output if item["covered"] is not None]
    return {
        "status": "ok",
        "groups": output,
        "intervals": len(hits),
        "coverage": float(np.mean(hits)) if hits else None,
    }


def selection_metrics(recovery: Mapping[str, Any], concepts: Sequence[OracleConcept]) -> dict[str, Any]:
    """Role-wise recovery of the oracle concepts by the selected Stage 2 variables."""

    if recovery.get("status") != "ok":
        return {"status": "unavailable", "reason": "no fold has a frozen held-out extraction"}
    roles = {concept.name: set(concept.roles) for concept in concepts}
    true_modifiers = [name for name, value in roles.items() if "effect_modifier" in value]
    true_confounders = [name for name, value in roles.items() if "confounder" in value]
    instruments = [name for name, value in roles.items() if "instrument" in value]
    prognostic = [
        name for name, value in roles.items()
        if "prognostic" in value and not value & {"confounder", "effect_modifier"}
    ]
    unmeasured = [name for name, value in roles.items() if "unmeasured_confounder" in value]
    post_treatment = [name for name, value in roles.items() if "post_treatment" in value]
    rows = list(recovery.get("r_squared") or [])
    totals = {key: [0, 0] for key in ("modifier_recall", "modifier_precision", "confounder_recall")}
    counts = {key: 0 for key in (
        "instruments_in_nuisance_set", "instruments_among_modifiers",
        "prognostic_in_nuisance_set", "prognostic_among_modifiers",
    )}
    per_fold: dict[str, Any] = {}
    unmeasured_best: dict[str, float | None] = {name: None for name in unmeasured}
    post_hits: list[dict[str, Any]] = []
    for fold in recovery.get("folds") or []:
        fold_rows = [row for row in rows if str(row["outer_fold"]) == str(fold)]
        modifiers = sorted({row["feature_id"] for row in fold_rows if row["selected_modifier"]})
        nuisance = sorted({row["feature_id"] for row in fold_rows if row["nuisance_set"]})

        def recovered(
            feature_ids: Sequence[str], concept: str, fold_rows: Sequence[Mapping] = fold_rows
        ) -> bool:
            ids = set(feature_ids)
            return any(row["recovered"] for row in fold_rows
                       if row["feature_id"] in ids and row["concept"] == concept)

        recovered_modifiers = [name for name in true_modifiers if recovered(modifiers, name)]
        precise = [
            feature for feature in modifiers
            if any(recovered([feature], name) for name in true_modifiers)
        ]
        recovered_confounders = [name for name in true_confounders if recovered(nuisance, name)]
        fold_counts = {
            "instruments_in_nuisance_set": [n for n in instruments if recovered(nuisance, n)],
            "instruments_among_modifiers": [n for n in instruments if recovered(modifiers, n)],
            "prognostic_in_nuisance_set": [n for n in prognostic if recovered(nuisance, n)],
            "prognostic_among_modifiers": [n for n in prognostic if recovered(modifiers, n)],
        }
        for key, names in fold_counts.items():
            counts[key] += len(names)
        for key, (hit, total) in (
            ("modifier_recall", (len(recovered_modifiers), len(true_modifiers))),
            ("modifier_precision", (len(precise), len(modifiers))),
            ("confounder_recall", (len(recovered_confounders), len(true_confounders))),
        ):
            totals[key][0] += hit
            totals[key][1] += total
        selected = set(modifiers) | set(nuisance)
        fold_unmeasured = {}
        for name in unmeasured:
            values = [row["r_squared"] for row in fold_rows
                      if row["concept"] == name and row["feature_id"] in selected
                      and row["r_squared"] is not None]
            best = max(values) if values else None
            fold_unmeasured[name] = best
            if best is not None and (unmeasured_best[name] is None or best > unmeasured_best[name]):
                unmeasured_best[name] = best
        for row in fold_rows:
            if row["concept"] in post_treatment and row["feature_id"] in selected and row["recovered"]:
                post_hits.append({
                    "outer_fold": str(fold), "feature_id": row["feature_id"], "name": row["name"],
                    "concept": row["concept"], "r_squared": row["r_squared"],
                })
        per_fold[str(fold)] = {
            "selected_modifiers": len(modifiers),
            "nuisance_set": len(nuisance),
            "true_modifiers_recovered": recovered_modifiers,
            "modifiers_recovering_a_true_modifier": len(precise),
            "true_confounders_recovered": recovered_confounders,
            **fold_counts,
            "unmeasured_max_r_squared_selected": fold_unmeasured,
            "selected_variables_missing_from_heldout_extraction": (
                (recovery.get("per_fold") or {}).get(str(fold), {})
                .get("selected_variables_missing_from_heldout_extraction", [])
            ),
        }
    threshold = float(recovery.get("threshold_r_squared") or DEFAULT_RECOVERY_R2)

    def ratio(key: str) -> float | None:
        hit, total = totals[key]
        return hit / total if total else None

    return {
        "status": "ok",
        "folds": list(recovery.get("folds") or []),
        "threshold_r_squared": threshold,
        "true_concepts": {
            "effect_modifiers": true_modifiers, "confounders": true_confounders,
            "instruments": instruments, "prognostic_only": prognostic,
            "unmeasured_confounders": unmeasured, "post_treatment": post_treatment,
        },
        "modifier_recall": ratio("modifier_recall"),
        "modifier_precision": ratio("modifier_precision"),
        "confounder_recall": ratio("confounder_recall"),
        "selected_modifiers": totals["modifier_precision"][1],
        **counts,
        "unmeasured_max_r_squared_selected": unmeasured_best,
        "unmeasured_recovered_by_selected": (
            any(value is not None and value >= threshold for value in unmeasured_best.values())
            if unmeasured else None
        ),
        "post_treatment_recovered": bool(post_hits) if post_treatment else None,
        "post_treatment_recoveries": post_hits,
        "per_fold": per_fold,
    }


def overlap_metrics(predictions: pd.DataFrame, bounds: Sequence[float] = OVERLAP_FLAG_BOUNDS) -> dict[str, Any]:
    from ..inference.stage2_diagnostics import outside_overlap_flags

    propensity = _numeric(predictions, "propensity")
    flags = outside_overlap_flags(propensity, bounds)
    finite = np.isfinite(propensity)
    return {
        "bounds": [float(bounds[0]), float(bounds[1])],
        "rows": int(finite.sum()),
        "outside_overlap": int(flags.sum()),
        "outside_overlap_fraction": float(flags.sum() / finite.sum()) if finite.any() else None,
    }


def _expit(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def _logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-12, 1.0 - 1e-12)
    return np.log(probability / (1.0 - probability))


def _coefficient(mapping: Any, words: Sequence[str]) -> float | None:
    if not isinstance(mapping, Mapping):
        return None
    for key, value in mapping.items():
        if any(word in str(key).lower() for word in words):
            number = _float(value.get("coefficient") if isinstance(value, Mapping) else value)
            if number is not None:
                return number
    return None


def unmeasured_coefficients(
    entries: Sequence[Mapping[str, Any]], values: pd.DataFrame, concepts: Sequence[OracleConcept],
    *, treatment_logit: np.ndarray, outcome_scale_y0: np.ndarray,
) -> list[dict[str, Any]]:
    """Each U's treatment and outcome coefficients: from the metadata when it states them,
    otherwise by least squares on the oracle columns (the U's are independent of X)."""

    design = _oracle_design(values, concepts)
    outputs = []
    for entry in entries:
        name = str(entry["name"])
        u = pd.to_numeric(values[f"true_{name}"], errors="coerce").to_numpy(float)
        coefficients = entry.get("coefficients")
        if isinstance(coefficients, (list, tuple)) and len(coefficients) == 2:
            stated = (_float(coefficients[0]), _float(coefficients[1]))
        else:
            stated = (
                _coefficient(coefficients, ("treat", "propensity")),
                _coefficient(coefficients, ("outcome",)),
            )
        fitted = []
        for target in (treatment_logit, outcome_scale_y0):
            x = np.column_stack([np.ones(len(u)), u, design])
            usable = np.isfinite(target) & np.isfinite(x).all(axis=1)
            beta = np.linalg.lstsq(x[usable], target[usable], rcond=None)[0]
            fitted.append(float(beta[1]))
        source = "metadata" if None not in stated else "least_squares_on_oracle_columns"
        treatment, outcome = stated if None not in stated else fitted
        outputs.append({
            "name": name,
            "treatment_coefficient": float(treatment),
            "outcome_coefficient": float(outcome),
            "source": source,
            "least_squares_check": {"treatment": fitted[0], "outcome": fitted[1]},
        })
    return outputs


def _oracle_design(values: pd.DataFrame, concepts: Sequence[OracleConcept]) -> np.ndarray:
    parts = []
    for concept in concepts:
        if concept.roles and set(concept.roles) & {"unmeasured_confounder", "post_treatment"}:
            continue
        if concept.column not in values:
            continue
        column = values[concept.column]
        if concept.value_type == "continuous":
            numeric = pd.to_numeric(column, errors="coerce").to_numpy(float)
            sd = np.nanstd(numeric)
            parts.append(np.nan_to_num((numeric - np.nanmean(numeric)) / (sd or 1.0)).reshape(-1, 1))
        else:
            dummies = pd.get_dummies(column.astype(str), drop_first=True).to_numpy(float)
            if dummies.size:
                parts.append(dummies)
    return np.column_stack(parts) if parts else np.empty((len(values), 0))


def short_nuisances(
    *, propensity: np.ndarray, mu0: np.ndarray, mu1: np.ndarray, u: np.ndarray,
    treatment_coefficients: np.ndarray, outcome_coefficients: np.ndarray, binary: bool,
    nodes: int = GAUSS_HERMITE_NODES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """P(D=1|X), E[Y|D=0,X], E[Y|D=1,X] with the unmeasured confounders integrated out.

    ``u`` is the (rows, k) matrix of true U values, independent standard normals
    entering logit(e) and the outcome predictor linearly. Their combined
    treatment and outcome scores are bivariate normal; the expectation over
    them uses a tensor Gauss-Hermite rule.
    """

    gamma_t = np.asarray(treatment_coefficients, dtype=float)
    gamma_y = np.asarray(outcome_coefficients, dtype=float)
    u = np.asarray(u, dtype=float).reshape(len(propensity), -1)
    eta_t = _logit(propensity) - u @ gamma_t
    shift_y = u @ gamma_y
    sd_t = float(np.linalg.norm(gamma_t))
    covariance = float(gamma_t @ gamma_y)
    sd_y = float(np.linalg.norm(gamma_y))
    # s_t = a Z1 and s_y = b Z1 + c Z2 reproduce the scores' covariance.
    a = sd_t
    b = covariance / a if a > 0 else 0.0
    c = math.sqrt(max(sd_y ** 2 - b ** 2, 0.0))
    points, weights = np.polynomial.hermite_e.hermegauss(int(nodes))
    weights = weights / weights.sum()
    if c <= 1e-12 * max(1.0, sd_y) or a == 0.0:
        # One latent direction (one U, or collinear coefficients): a 1-D rule suffices.
        w = weights
        s_t = a * points
        s_y = (b if a > 0 else sd_y) * points
    else:
        z1, z2 = np.meshgrid(points, points, indexing="ij")
        w = np.outer(weights, weights).reshape(-1)
        s_t = (a * z1).reshape(-1)
        s_y = (b * z1 + c * z2).reshape(-1)
    e_draw = _expit(eta_t[:, None] + s_t[None, :])
    if binary:
        mu0_draw = _expit(_logit(mu0)[:, None] - shift_y[:, None] + s_y[None, :])
        mu1_draw = _expit(_logit(mu1)[:, None] - shift_y[:, None] + s_y[None, :])
    else:
        mu0_draw = (mu0 - shift_y)[:, None] + s_y[None, :]
        mu1_draw = (mu1 - shift_y)[:, None] + s_y[None, :]
    e_short = e_draw @ w
    g1 = (mu1_draw * e_draw) @ w / e_short
    g0 = (mu0_draw * (1.0 - e_draw)) @ w / (1.0 - e_short)
    return e_short, g0, g1


def sensitivity_metrics(
    predictions: pd.DataFrame,
    oracle: BenchmarkOracle,
    heterogeneity: Mapping[str, Any],
    *,
    outcome_type: str,
    propensity_clip: float,
    true_ate: float | None,
    unmeasured_strength: float | None = None,
    ceiling_predictions: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """U's true strength (cf_y, cf_d, rho) next to the run's E-values, RV and RVa."""

    from ..inference.stage2_sensitivity import (
        ate_elements,
        benchmark_gains,
        ovb_bounds,
    )

    section = heterogeneity.get("unmeasured_confounding_sensitivity") or {}
    evalues = ((section.get("evalues") or {}).get("ate") or {}).get("pooled") or {}
    bounds = ((section.get("omitted_variable_bias") or {}).get("ate") or {}).get("pooled") or {}
    reported = {
        "evalue_point": _float(evalues.get("evalue_point")),
        "evalue_confidence_limit": _float(evalues.get("evalue_confidence_limit")),
        "robustness_value": _float(bounds.get("robustness_value")),
        "robustness_value_confidence": _float(bounds.get("robustness_value_confidence")),
    }
    if not oracle.unmeasured:
        return {"status": "not_applicable", "reason": "the cohort has no unmeasured confounder",
                **reported}
    if oracle.potential_outcome_columns is None or "true_treatment_prob" not in oracle.values:
        return {"status": "unavailable", "reason": "true propensity or potential outcomes missing",
                **reported}
    eligible = np.isfinite(_numeric(predictions, "aipw_score"))
    rows = predictions.loc[eligible].reset_index(drop=True)
    ids = rows[ROW_ID].astype(int).tolist()
    y, t = _numeric(rows, "outcome"), _numeric(rows, "treatment")
    e_long = _aligned(oracle.values, ids, "true_treatment_prob")
    y0_name, y1_name = oracle.potential_outcome_columns
    mu0_long = _aligned(oracle.values, ids, y0_name)
    mu1_long = _aligned(oracle.values, ids, y1_name)
    binary = outcome_type == "binary"
    all_ids = oracle.values[ROW_ID].astype(int).tolist()
    coefficients = unmeasured_coefficients(
        oracle.unmeasured, oracle.values, oracle.data.concepts,
        treatment_logit=_logit(_aligned(oracle.values, all_ids, "true_treatment_prob")),
        outcome_scale_y0=(
            _logit(_aligned(oracle.values, all_ids, y0_name)) if binary
            else _aligned(oracle.values, all_ids, y0_name)
        ),
    )
    u = np.column_stack([_aligned(oracle.values, ids, f"true_{item['name']}") for item in coefficients])
    e_short, mu0_short, mu1_short = short_nuisances(
        propensity=e_long, mu0=mu0_long, mu1=mu1_long, u=u,
        treatment_coefficients=np.array([item["treatment_coefficient"] for item in coefficients]),
        outcome_coefficients=np.array([item["outcome_coefficient"] for item in coefficients]),
        binary=binary,
    )
    long = ate_elements(y, t, e_long, mu0_long, mu1_long, clip=propensity_clip)
    short = ate_elements(y, t, e_short, mu0_short, mu1_short, clip=propensity_clip)
    gains = benchmark_gains(long, short, outcome_variance=float(np.var(y)))
    result: dict[str, Any] = {
        "status": "ok",
        **reported,
        "unmeasured_strength": unmeasured_strength,
        "unmeasured_confounders": coefficients,
        "true_strength": gains,
        "true_ate": true_ate,
    }
    if gains.get("status") != "ok":
        return {**result, "status": "unavailable", "reason": gains.get("reason")}

    def bound_check(frame: pd.DataFrame) -> dict[str, Any]:
        part = frame[np.isfinite(_numeric(frame, "aipw_score"))]
        elements = ate_elements(
            _numeric(part, "outcome"), _numeric(part, "treatment"), _numeric(part, "propensity"),
            _numeric(part, "mu0"), _numeric(part, "mu1"), clip=propensity_clip,
        )
        if gains["cf_d"] >= 1.0:
            return {"status": "unbounded", "reason": "cf_d = 1 makes the bound infinite"}
        output: dict[str, Any] = {"status": "ok", "estimate": elements.theta}
        for label, rho in (("rho_1", 1.0), ("true_rho", gains.get("rho"))):
            if rho is None:
                continue
            bound = ovb_bounds(elements, gains["cf_y"], gains["cf_d"], rho=float(rho))
            output[label] = {
                **{key: bound[key] for key in ("rho", "max_bias", "lower", "upper",
                                               "lower_confidence_bound", "upper_confidence_bound")},
                "covers_true_ate": _covers(bound["lower"], bound["upper"], true_ate),
                "confidence_bounds_cover_true_ate": _covers(
                    bound["lower_confidence_bound"], bound["upper_confidence_bound"], true_ate
                ),
                "includes_zero": bool(bound["lower"] <= 0.0 <= bound["upper"]),
            }
        return output

    result["run_bound_at_true_strength"] = bound_check(predictions)
    if ceiling_predictions is not None:
        result["ceiling_bound_at_true_strength"] = bound_check(ceiling_predictions)
    return result


# ---------------------------------------------------------------------------
# aggregate: per cell


def _truth(predictions: pd.DataFrame, oracle: BenchmarkOracle) -> np.ndarray:
    if oracle.ite_column is None:
        raise ValueError("the dataset has no oracle ITE column (true_ite or true_ite_prob)")
    return _aligned(oracle.values, predictions[ROW_ID], oracle.ite_column)


def effect_metrics(
    predictions: pd.DataFrame, co_primary: Mapping[str, Any], heterogeneity: Mapping[str, Any],
    oracle: BenchmarkOracle,
) -> dict[str, Any]:
    truth = _truth(predictions, oracle)
    propensity = (
        _aligned(oracle.values, predictions[ROW_ID], "true_treatment_prob")
        if "true_treatment_prob" in oracle.values else None
    )
    ate = ate_metrics(predictions, truth, co_primary.get("ate"))
    return {
        "ate": ate,
        "ato": ato_metrics(predictions, truth, propensity, co_primary.get("ato")),
        "cate": cate_metrics(predictions, truth, ate["estimate"]),
        "gates": gates_coverage(predictions, truth, heterogeneity),
        "overlap": overlap_metrics(predictions),
    }


def _metadata_features(oracle: BenchmarkOracle) -> list[dict[str, Any]]:
    from .stage1 import _metadata_features as features

    return features(oracle.metadata) if oracle.metadata else []


def evaluate_cell(
    *,
    run_dir: Path,
    cell: Cell,
    bench: Path,
    recovery_r2: float,
    ceiling: bool,
    ceiling_workers: int = 1,
) -> dict[str, Any]:
    """Every metric for one finished run; run artifacts are hashed before oracle data loads."""

    dataset_path, metadata_path = resolve_oracle_inputs(run_dir)
    artifacts = read_run_artifacts(run_dir)
    run_config = artifacts.run_config
    outcome_type = str(run_config.get("outcome_type") or "binary")
    oracle = load_benchmark_oracle(
        dataset_path, metadata_path, readers=[artifacts.reader], outcome_type=outcome_type,
        label_columns=[str(run_config.get(key)) for key in
                       ("unit_id_column", "treatment_column", "outcome_column")
                       if run_config.get(key)],
    )
    predictions = artifacts.predictions
    check_alignment(predictions, oracle, run_config)
    for fold, split in artifacts.splits.items():
        held = predictions.loc[predictions["outer_fold"].astype(int) == fold, ROW_ID].astype(int)
        if sorted(held.tolist()) != sorted(split["heldout_row_ids"]):
            raise ValueError(f"the split file's outer fold {fold} does not match the predictions")
    co_primary = artifacts.causal_estimate.get("co_primary_estimands") or {
        "ate": {
            "estimate": artifacts.causal_estimate.get("ate"),
            "standard_error": artifacts.causal_estimate.get("standard_error"),
            "confidence_interval_95": artifacts.causal_estimate.get("confidence_interval_95"),
        }
    }
    result: dict[str, Any] = {
        "run_dir": str(artifacts.run_dir),
        "run_artifact_sha256": artifacts.hashes,
        "run_artifacts_hashed_before_oracle_load": True,
        "oracle_dataset": str(oracle.data.dataset),
        "oracle_dataset_sha256": oracle.data.dataset_sha256,
        "oracle_metadata_sha256": oracle.data.metadata_sha256,
        "oracle_ite_column": oracle.ite_column,
        **effect_metrics(predictions, co_primary, artifacts.heterogeneity, oracle),
    }
    recovery = oracle_recovery(artifacts.lineage, oracle.data, threshold=recovery_r2)
    result["selection"] = selection_metrics(recovery, oracle.data.concepts)
    # The clip the run's own diagnostics used, else its configured clip.
    propensity_clip = float(
        (artifacts.heterogeneity.get("inputs") or {}).get("propensity_clip")
        or artifacts.stage2_config.get("propensity_clip") or 0.02
    )
    ceiling_result: dict[str, Any] = {"status": "not_requested"}
    ceiling_predictions = None
    if ceiling:
        try:
            ceiling_result, ceiling_predictions = _ceiling(
                artifacts, oracle, cell=cell, bench=bench, workers=ceiling_workers
            )
        except Exception as error:  # one failed refit must not hide the run's metrics
            ceiling_result = {"status": "error", "reason": f"{type(error).__name__}: {error}"}
    result["oracle_ceiling"] = ceiling_result
    try:
        result["sensitivity"] = sensitivity_metrics(
            predictions, oracle, artifacts.heterogeneity, outcome_type=outcome_type,
            propensity_clip=propensity_clip, true_ate=result["ate"]["true"],
            unmeasured_strength=cell.scenario.unmeasured_strength,
            ceiling_predictions=ceiling_predictions,
        )
    except Exception as error:  # reporting only, like the run's own sensitivity section
        result["sensitivity"] = {"status": "error", "reason": f"{type(error).__name__}: {error}"}
    return result


def _ceiling(
    artifacts: RunArtifacts, oracle: BenchmarkOracle, *, cell: Cell, bench: Path, workers: int
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    from .benchmark_ceiling import ceiling_settings, run_oracle_ceiling

    if not artifacts.splits:
        return {"status": "unavailable", "reason": "the run has no split file"}, None
    features = _metadata_features(oracle)
    if not features:
        return {"status": "unavailable", "reason": "no generator metadata features"}, None
    from .attribution import _read_dataset_columns

    settings = ceiling_settings(artifacts.stage2_config, artifacts.run_config)
    columns = [settings["unit_id_column"], settings["treatment_column"], settings["outcome_column"]]
    truth_columns = [f"true_{feature['name']}" for feature in features]
    from .attribution import _dataset_columns

    available = set(_dataset_columns(oracle.data.dataset))
    dataset = _read_dataset_columns(
        oracle.data.dataset, [c for c in [*columns, *truth_columns] if c in available]
    )
    output_root = cell.directory(bench) / "oracle_ceiling"
    assert_output_outside_run_inputs(output_root, artifacts.run_dir)
    fitted = run_oracle_ceiling(
        dataset=dataset,
        features=features,
        splits=artifacts.splits,
        settings=settings,
        final_estimators=artifacts.final_estimators,
        output_root=output_root,
        identity_inputs={
            "dataset_sha256": oracle.data.dataset_sha256,
            "split_provenance_sha256": artifacts.hashes.get(
                os.path.join("components", "tfidf", "split_provenance.jsonl")
            ),
        },
        workers=workers,
    )
    if fitted.get("status") != "ok":
        return fitted, None
    predictions = fitted["predictions"]
    metrics = effect_metrics(predictions, fitted["co_primary_estimands"], fitted["heterogeneity"], oracle)
    return {
        "status": "ok",
        "output_dir": fitted["output_dir"],
        "fingerprint": fitted["fingerprint"],
        "estimators": fitted["estimators"],
        "design_rule": fitted["design_rule"],
        "selected_features": fitted["selected_features"],
        "nuisance_candidates": fitted["nuisance_candidates"],
        **metrics,
    }, predictions


# ---------------------------------------------------------------------------
# aggregate: tables


CELL_COLUMNS = (
    "scenario", "family", "variant", "size", "seed", "seed_set", "cell", "status", "detail",
    "run_dir",
    "ate_true", "ate_estimate", "ate_ci_low", "ate_ci_high", "ate_bias", "ate_covered", "ate_rows",
    "ato_true", "ato_true_estimated_weights", "ato_estimate", "ato_ci_low", "ato_ci_high",
    "ato_bias", "ato_covered", "ato_covered_estimated_weights",
    "cate_rows", "cate_rmse", "cate_pearson", "cate_spearman", "constant_ate_rmse",
    "cate_relative_improvement", "true_ite_sd",
    "gates_pooled_coverage", "gates_pooled_intervals", "gates_fold_coverage",
    "gates_fold_intervals",
    "outside_overlap_fraction",
    "ceiling_status", "ceiling_ate_estimate", "ceiling_ate_bias", "ceiling_ate_covered",
    "ceiling_ato_bias", "ceiling_ato_covered", "ceiling_cate_rmse", "ceiling_cate_pearson",
    "ceiling_constant_ate_rmse", "ceiling_cate_relative_improvement",
    "ceiling_gates_pooled_coverage",
    "modifier_recall", "modifier_precision", "selected_modifiers", "confounder_recall",
    "instruments_in_nuisance_set", "instruments_among_modifiers", "prognostic_in_nuisance_set",
    "prognostic_among_modifiers", "unmeasured_max_r_squared_selected",
    "unmeasured_recovered_by_selected", "post_treatment_recovered",
    "unmeasured_strength", "u_cf_y", "u_cf_d", "u_rho", "evalue_point",
    "evalue_confidence_limit", "robustness_value", "robustness_value_confidence",
    "ovb_lower_at_u", "ovb_upper_at_u", "ovb_covers_true_ate",
    "ovb_confidence_covers_true_ate", "ovb_includes_zero_at_u", "ceiling_ovb_covers_true_ate",
)


def _get(mapping: Any, *keys: str) -> Any:
    for key in keys:
        mapping = mapping.get(key) if isinstance(mapping, Mapping) else None
    return mapping


def cell_row(entry: Mapping[str, Any]) -> dict[str, Any]:
    cell: Cell = entry["cell"]
    row: dict[str, Any] = {
        "scenario": cell.scenario.id, "family": cell.scenario.family, "variant": entry["variant"],
        "size": cell.size, "seed": cell.seed, "seed_set": cell.seed_set, "cell": cell.id,
        "status": entry["status"], "detail": entry["detail"], "run_dir": entry["run_dir"],
    }
    metrics = entry.get("metrics")
    if not metrics:
        return row
    for prefix, source in (("", metrics), ("ceiling_", metrics.get("oracle_ceiling") or {})):
        for key in ("estimate", "bias", "covered"):
            row[f"{prefix}ate_{key}"] = _get(source, "ate", key)
        for key in ("bias", "covered"):
            row[f"{prefix}ato_{key}"] = _get(source, "ato", key)
        row[f"{prefix}cate_rmse"] = _get(source, "cate", "rmse")
        row[f"{prefix}cate_pearson"] = _get(source, "cate", "pearson")
        row[f"{prefix}constant_ate_rmse"] = _get(source, "cate", "constant_ate_rmse")
        row[f"{prefix}cate_relative_improvement"] = _get(source, "cate", "relative_improvement")
        row[f"{prefix}gates_pooled_coverage"] = _get(source, "gates", "pooled", "coverage")
    row["ceiling_status"] = _get(metrics, "oracle_ceiling", "status")
    row.update({
        "ate_true": _get(metrics, "ate", "true"),
        "ate_ci_low": _get(metrics, "ate", "ci_low"),
        "ate_ci_high": _get(metrics, "ate", "ci_high"),
        "ate_rows": _get(metrics, "ate", "rows"),
        "ato_true": _get(metrics, "ato", "true"),
        "ato_true_estimated_weights": _get(metrics, "ato", "true_estimated_weights"),
        "ato_estimate": _get(metrics, "ato", "estimate"),
        "ato_ci_low": _get(metrics, "ato", "ci_low"),
        "ato_ci_high": _get(metrics, "ato", "ci_high"),
        "ato_covered_estimated_weights": _get(metrics, "ato", "covered_estimated_weights"),
        "cate_rows": _get(metrics, "cate", "rows"),
        "cate_spearman": _get(metrics, "cate", "spearman"),
        "true_ite_sd": _get(metrics, "cate", "true_ite_sd"),
        "gates_pooled_intervals": _get(metrics, "gates", "pooled", "intervals"),
        "gates_fold_coverage": _get(metrics, "gates", "fold_coverage"),
        "gates_fold_intervals": _get(metrics, "gates", "fold_intervals"),
        "outside_overlap_fraction": _get(metrics, "overlap", "outside_overlap_fraction"),
    })
    selection = metrics.get("selection") or {}
    for key in ("modifier_recall", "modifier_precision", "selected_modifiers", "confounder_recall",
                "instruments_in_nuisance_set", "instruments_among_modifiers",
                "prognostic_in_nuisance_set", "prognostic_among_modifiers",
                "unmeasured_recovered_by_selected", "post_treatment_recovered"):
        row[key] = selection.get(key)
    best = [value for value in (selection.get("unmeasured_max_r_squared_selected") or {}).values()
            if value is not None]
    row["unmeasured_max_r_squared_selected"] = max(best) if best else None
    sensitivity = metrics.get("sensitivity") or {}
    for key in ("evalue_point", "evalue_confidence_limit", "robustness_value",
                "robustness_value_confidence", "unmeasured_strength"):
        row[key] = sensitivity.get(key)
    row["u_cf_y"] = _get(sensitivity, "true_strength", "cf_y")
    row["u_cf_d"] = _get(sensitivity, "true_strength", "cf_d")
    row["u_rho"] = _get(sensitivity, "true_strength", "rho")
    bound = _get(sensitivity, "run_bound_at_true_strength", "rho_1") or {}
    row["ovb_lower_at_u"] = bound.get("lower")
    row["ovb_upper_at_u"] = bound.get("upper")
    row["ovb_covers_true_ate"] = bound.get("covers_true_ate")
    row["ovb_confidence_covers_true_ate"] = bound.get("confidence_bounds_cover_true_ate")
    row["ovb_includes_zero_at_u"] = bound.get("includes_zero")
    row["ceiling_ovb_covers_true_ate"] = _get(
        sensitivity, "ceiling_bound_at_true_strength", "rho_1", "covers_true_ate"
    )
    return row


def _values(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    output = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (bool, np.bool_)) or _float(value) is not None:
            output.append(float(value))
    return output


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = _values(rows, key)
    return float(np.mean(values)) if values else None


def _rmse(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = _values(rows, key)
    return float(np.sqrt(np.mean(np.square(values)))) if values else None


SCENARIO_SUMMARY = (
    ("ate_mean_true", _mean, "ate_true"),
    ("ate_mean_estimate", _mean, "ate_estimate"),
    ("ate_mean_bias", _mean, "ate_bias"),
    ("ate_rmse", _rmse, "ate_bias"),
    ("ate_coverage", _mean, "ate_covered"),
    ("ato_mean_bias", _mean, "ato_bias"),
    ("ato_rmse", _rmse, "ato_bias"),
    ("ato_coverage", _mean, "ato_covered"),
    ("ato_coverage_estimated_weights", _mean, "ato_covered_estimated_weights"),
    ("cate_mean_rmse", _mean, "cate_rmse"),
    ("constant_ate_mean_rmse", _mean, "constant_ate_rmse"),
    ("cate_mean_relative_improvement", _mean, "cate_relative_improvement"),
    ("cate_mean_pearson", _mean, "cate_pearson"),
    ("cate_mean_spearman", _mean, "cate_spearman"),
    ("gates_pooled_coverage", _mean, "gates_pooled_coverage"),
    ("gates_fold_coverage", _mean, "gates_fold_coverage"),
    ("outside_overlap_fraction", _mean, "outside_overlap_fraction"),
    ("ceiling_ate_mean_bias", _mean, "ceiling_ate_bias"),
    ("ceiling_ate_rmse", _rmse, "ceiling_ate_bias"),
    ("ceiling_ate_coverage", _mean, "ceiling_ate_covered"),
    ("ceiling_ato_coverage", _mean, "ceiling_ato_covered"),
    ("ceiling_cate_mean_rmse", _mean, "ceiling_cate_rmse"),
    ("ceiling_cate_mean_relative_improvement", _mean, "ceiling_cate_relative_improvement"),
    ("ceiling_gates_pooled_coverage", _mean, "ceiling_gates_pooled_coverage"),
    ("modifier_recall", _mean, "modifier_recall"),
    ("modifier_precision", _mean, "modifier_precision"),
    ("confounder_recall", _mean, "confounder_recall"),
    ("instruments_in_nuisance_set", _mean, "instruments_in_nuisance_set"),
    ("instruments_among_modifiers", _mean, "instruments_among_modifiers"),
    ("prognostic_in_nuisance_set", _mean, "prognostic_in_nuisance_set"),
    ("prognostic_among_modifiers", _mean, "prognostic_among_modifiers"),
    ("unmeasured_max_r_squared_selected", _mean, "unmeasured_max_r_squared_selected"),
    ("post_treatment_recovered_rate", _mean, "post_treatment_recovered"),
    ("u_cf_y", _mean, "u_cf_y"),
    ("u_cf_d", _mean, "u_cf_d"),
    ("evalue_point", _mean, "evalue_point"),
    ("robustness_value", _mean, "robustness_value"),
    ("robustness_value_confidence", _mean, "robustness_value_confidence"),
    ("ovb_coverage_rate", _mean, "ovb_covers_true_ate"),
    ("ceiling_ovb_coverage_rate", _mean, "ceiling_ovb_covers_true_ate"),
)
SCENARIO_KEYS = ("scenario", "family", "variant", "size", "seed_set")
SCENARIO_COLUMNS = (
    *SCENARIO_KEYS, "cells_expected", "cells_complete", "cells_pending", "cells_error",
    *(name for name, _function, _key in SCENARIO_SUMMARY),
)


def scenario_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in SCENARIO_KEYS), []).append(row)
    output = []
    for key, members in groups.items():
        complete = [row for row in members if row["status"] == "complete"]
        entry = dict(zip(SCENARIO_KEYS, key))
        entry.update(
            cells_expected=len(members),
            cells_complete=len(complete),
            cells_pending=sum(row["status"] == "pending" for row in members),
            cells_error=sum(row["status"] == "error" for row in members),
        )
        for name, function, source in SCENARIO_SUMMARY:
            entry[name] = function(complete, source)
        output.append(entry)
    return output


def leak_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """S6-type pairs: the text-column variant against the dated-notes variant per size and seed."""

    pairs: dict[tuple, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        if row["status"] != "complete" or row["variant"] not in {"text_column", "dated_notes"}:
            continue
        pairs.setdefault((row["scenario"], row["size"], row["seed"], row["seed_set"]), {})[
            row["variant"]
        ] = row
    output = []
    for (scenario, size, seed, seed_set), variants in sorted(pairs.items()):
        text, notes = variants.get("text_column"), variants.get("dated_notes")
        entry: dict[str, Any] = {"scenario": scenario, "size": size, "seed": seed,
                                 "seed_set": seed_set}
        for name, row in (("text_column", text), ("dated_notes", notes)):
            for key in ("ate_bias", "ate_covered", "outside_overlap_fraction",
                        "post_treatment_recovered"):
                entry[f"{name}_{key}"] = None if row is None else row.get(key)
        for key in ("ate_bias", "outside_overlap_fraction"):
            left, right = entry[f"text_column_{key}"], entry[f"dated_notes_{key}"]
            entry[f"{key}_text_minus_notes"] = (
                float(left) - float(right) if left is not None and right is not None else None
            )
        output.append(entry)
    return output


def _format(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    number = _float(value)
    return "n/a" if number is None else f"{number:.{digits}f}"


def _markdown(summary: Mapping[str, Any], scenarios: Sequence[Mapping[str, Any]],
              rows: Sequence[Mapping[str, Any]], leaks: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# OCI benchmark aggregate", "",
        (
            f"Benchmark directory: `{summary['bench']}`. Seed sets: "
            f"{', '.join(summary['seed_sets'])}. Recovery threshold R^2 = "
            f"{summary['recovery_threshold_r_squared']}."
        ),
        "",
        "## Scenarios", "",
        (
            "| Scenario | Variant | n | Seeds | Complete | ATE bias | ATE RMSE | ATE coverage | "
            "ATO coverage | CATE RMSE (constant) | Relative improvement | GATES coverage | "
            "Ceiling ATE RMSE | Ceiling CATE RMSE | Modifier recall | Modifier precision | "
            "Confounder recall |"
        ),
        "|" + "---|" * 17,
    ]
    for row in scenarios:
        lines.append("| " + " | ".join([
            str(row["scenario"]), str(row["variant"]), str(row["size"]), str(row["seed_set"]),
            f"{row['cells_complete']}/{row['cells_expected']}",
            _format(row["ate_mean_bias"]), _format(row["ate_rmse"]), _format(row["ate_coverage"], 2),
            _format(row["ato_coverage"], 2),
            f"{_format(row['cate_mean_rmse'])} ({_format(row['constant_ate_mean_rmse'])})",
            _format(row["cate_mean_relative_improvement"]),
            _format(row["gates_pooled_coverage"], 2), _format(row["ceiling_ate_rmse"]),
            _format(row["ceiling_cate_mean_rmse"]), _format(row["modifier_recall"], 2),
            _format(row["modifier_precision"], 2), _format(row["confounder_recall"], 2),
        ]) + " |")
    sensitivity = [row for row in rows if row["status"] == "complete" and row.get("u_cf_y") is not None]
    if sensitivity:
        lines += [
            "", "## Unmeasured confounding (S4)", "",
            "| Cell | U strength | cf_y | cf_d | E-value | RV | RVa | Bound at U | Covers true ATE |",
            "|" + "---|" * 9,
        ]
        for row in sensitivity:
            lines.append("| " + " | ".join([
                str(row["cell"]), _format(row["unmeasured_strength"], 2), _format(row["u_cf_y"]),
                _format(row["u_cf_d"]), _format(row["evalue_point"], 2),
                _format(row["robustness_value"]), _format(row["robustness_value_confidence"]),
                f"[{_format(row['ovb_lower_at_u'])}, {_format(row['ovb_upper_at_u'])}]",
                _format(row["ovb_covers_true_ate"]),
            ]) + " |")
    if leaks:
        lines += [
            "", "## Post-treatment leak (S6)", "",
            (
                "| Scenario | n | Seed | ATE bias text / notes | Outside overlap text / notes | "
                "Post-treatment recovered text / notes |"
            ),
            "|" + "---|" * 6,
        ]
        for row in leaks:
            lines.append("| " + " | ".join([
                str(row["scenario"]), str(row["size"]), str(row["seed"]),
                f"{_format(row['text_column_ate_bias'])} / {_format(row['dated_notes_ate_bias'])}",
                (
                    f"{_format(row['text_column_outside_overlap_fraction'], 2)} / "
                    f"{_format(row['dated_notes_outside_overlap_fraction'], 2)}"
                ),
                (
                    f"{_format(row['text_column_post_treatment_recovered'])} / "
                    f"{_format(row['dated_notes_post_treatment_recovered'])}"
                ),
            ]) + " |")
    pending = summary.get("pending") or []
    if pending:
        lines += ["", "## Pending", ""]
        lines += [f"- `{item['cell']}` ({item['variant']}): {item['detail']}" for item in pending]
    errors = summary.get("errors") or []
    if errors:
        lines += ["", "## Errors", ""]
        lines += [f"- `{item['cell']}` ({item['variant']}): {item['error']}" for item in errors]
    lines += ["", "## Caveats", ""] + [f"- {text}" for text in CAVEATS] + [""]
    return "\n".join(lines)


def aggregate(
    *,
    out: Path,
    scenarios: str | None = None,
    sizes: str | None = None,
    seeds: str | None = None,
    confirmatory: bool = False,
    recovery_r2: float = DEFAULT_RECOVERY_R2,
    ceiling: bool = True,
    ceiling_workers: int = 1,
    registry_path: Path | None = None,
    seeds_path: Path | None = None,
) -> dict[str, Any]:
    if not 0.0 <= float(recovery_r2) <= 1.0:
        raise ValueError("--recovery-r2 must be between 0 and 1")
    registry = load_registry(registry_path)
    seed_registry = load_seed_registry(seeds_path)
    selected = registry.select(scenarios)
    seed_pairs = resolve_seeds(
        seed_registry, seeds, confirmatory=confirmatory, include_development=True
    )
    bench = Path(out).expanduser().resolve()
    output_dir = bench / "aggregate"
    index_path = bench / "configs.json"
    index = {
        (item["cell"], item["variant"]): item
        for item in (_read_json(index_path).get("runs", []) if index_path.is_file() else [])
    }
    entries: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for cell in expand_cells(selected, seed_pairs, _parse_sizes(sizes)):
        for variant in cell.scenario.variants:
            indexed = index.get((cell.id, variant.name))
            run_dir = Path(indexed["output_dir"]) if indexed else cell.run_dir(bench, variant.name)
            entry: dict[str, Any] = {"cell": cell, "variant": variant.name, "run_dir": str(run_dir)}
            if not (cell.data_dir(bench) / "dataset.parquet").is_file():
                entry.update(status="pending", detail="not_materialized")
            elif not cell.config_path(bench, variant.name).is_file() and indexed is None:
                entry.update(status="pending", detail="no_config")
            else:
                status, detail = run_status(run_dir)
                entry.update(status=status, detail=detail)
            if entry["status"] == "complete":
                assert_output_outside_run_inputs(output_dir, run_dir)
                try:
                    entry["metrics"] = evaluate_cell(
                        run_dir=run_dir, cell=cell, bench=bench, recovery_r2=recovery_r2,
                        ceiling=ceiling, ceiling_workers=ceiling_workers,
                    )
                except Exception as error:  # one unreadable run must not stop the aggregate
                    entry.update(status="error", detail=f"{type(error).__name__}: {error}")
                    errors.append({"cell": cell.id, "variant": variant.name,
                                   "error": entry["detail"]})
            record = _read_json(_materialization_record(cell, bench)) if _materialization_record(
                cell, bench).is_file() else {}
            entry["dgp_matches_registry"] = (
                (record.get("identity") or {}).get("dgp_sha256") == _dgp_identity(cell.scenario)["dgp_sha256"]
                if record else None
            )
            entries.append(entry)
    rows = [cell_row(entry) for entry in entries]
    scenario_table = scenario_rows(rows)
    leaks = leak_rows(rows)
    pending = [
        {"cell": entry["cell"].id, "variant": entry["variant"], "detail": entry["detail"]}
        for entry in entries if entry["status"] == "pending"
    ]
    summary = {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "created_at": _now(),
        "bench": str(bench),
        "registry": str(registry.path),
        "registry_sha256": registry.sha256,
        "seeds_file": str(seed_registry.path),
        "seeds_sha256": seed_registry.sha256,
        "seed_rule": seed_registry.rule,
        "seed_sets": sorted({kind for _seed, kind in seed_pairs}),
        "recovery_threshold_r_squared": float(recovery_r2),
        "oracle_ceiling_requested": bool(ceiling),
        "metric_definitions": METRIC_DEFINITIONS,
        "caveats": list(CAVEATS),
        "cells": [
            {
                "cell": entry["cell"].id, "scenario": entry["cell"].scenario.id,
                "variant": entry["variant"], "size": entry["cell"].size,
                "seed": entry["cell"].seed, "seed_set": entry["cell"].seed_set,
                "status": entry["status"], "detail": entry["detail"],
                "run_dir": entry["run_dir"],
                "dgp_matches_registry": entry.get("dgp_matches_registry"),
                **({"metrics": entry["metrics"]} if entry.get("metrics") else {}),
            }
            for entry in entries
        ],
        "scenarios": scenario_table,
        "leak_comparison": leaks,
        "pending": pending,
        "errors": errors,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "cells.csv", rows, CELL_COLUMNS)
    write_csv(output_dir / "scenarios.csv", scenario_table, SCENARIO_COLUMNS)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(
        _markdown(summary, scenario_table, rows, leaks), encoding="utf-8"
    )
    complete = sum(entry["status"] == "complete" for entry in entries)
    print(f"aggregated {complete} finished run(s); pending: {len(pending)}; errors: {len(errors)}")
    print(f"wrote {output_dir}")
    return {"output_dir": str(output_dir), "complete": complete, "pending": pending,
            "errors": errors}


# ---------------------------------------------------------------------------
# CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "OCI benchmark harness: materialize scenario cohorts, write run configurations, "
            "and aggregate finished runs against the oracle columns. Never launches a run or "
            "an LLM request."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--out", required=True, type=Path, help="benchmark directory BENCH")
        sub.add_argument("--scenarios",
                         help="comma-separated scenario ids or families (default: all)")
        sub.add_argument("--sizes", help="comma-separated sizes to include (default: all)")
        sub.add_argument("--seeds", help="comma-separated seeds, or 'development' / "
                         "'confirmatory' (default: development)")
        sub.add_argument("--confirmatory", action="store_true",
                         help="allow confirmatory seeds (see benchmarks/seeds.json)")
        sub.add_argument("--registry", type=Path, help=f"scenario registry (default {DEFAULT_REGISTRY.name})")
        sub.add_argument("--seeds-file", type=Path, help=f"seed file (default {DEFAULT_SEEDS.name})")

    materialize_parser = commands.add_parser("materialize", help="write one cohort per cell")
    common(materialize_parser)
    materialize_parser.add_argument("--source", type=Path,
                                    help="re-equation source cohort directory")
    materialize_parser.add_argument("--generated", type=Path,
                                    help="generated corpus for one S5- or S6-type scenario")
    configs_parser = commands.add_parser("configs", help="write one run configuration per cell")
    common(configs_parser)
    configs_parser.add_argument("--base", required=True, type=Path,
                                help="base run_all_evidence configuration (JSON or YAML)")
    configs_parser.add_argument("--force", action="store_true",
                                help="use confirmatory seeds again at the same commit")
    configs_parser.add_argument("--ledger", type=Path,
                                help=f"confirmatory ledger (default {DEFAULT_LEDGER.name})")
    aggregate_parser = commands.add_parser("aggregate", help="aggregate finished runs")
    common(aggregate_parser)
    aggregate_parser.add_argument("--recovery-r2", type=float, default=DEFAULT_RECOVERY_R2,
                                  help="held-out R^2 at which a variable recovers an oracle concept")
    aggregate_parser.add_argument("--no-ceiling", action="store_true",
                                  help="skip the oracle-covariate ceiling refits")
    aggregate_parser.add_argument("--ceiling-workers", type=int, default=1,
                                  help="CPU processes for the ceiling refits")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    shared = {
        "out": args.out, "scenarios": args.scenarios, "sizes": args.sizes, "seeds": args.seeds,
        "confirmatory": args.confirmatory, "registry_path": args.registry,
        "seeds_path": args.seeds_file,
    }
    try:
        if args.command == "materialize":
            materialize(source=args.source, generated=args.generated, **shared)
        elif args.command == "configs":
            write_configs(base=args.base, force=args.force, ledger_path=args.ledger, **shared)
        else:
            aggregate(recovery_r2=args.recovery_r2, ceiling=not args.no_ceiling,
                      ceiling_workers=args.ceiling_workers, **shared)
    except SeedGuardError as error:
        print(f"refused: {error}")
        return 2
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}")
        return 1
    return 0


__all__ = [
    "AGGREGATE_SCHEMA_VERSION",
    "CAVEATS",
    "METRIC_DEFINITIONS",
    "BenchmarkOracle",
    "Cell",
    "Registry",
    "RunVariant",
    "Scenario",
    "SeedGuardError",
    "SeedRegistry",
    "aggregate",
    "append_ledger",
    "ate_metrics",
    "ato_metrics",
    "build_parser",
    "build_run_config",
    "cate_metrics",
    "cell_row",
    "check_ledger",
    "evaluate_cell",
    "expand_cells",
    "gates_coverage",
    "git_state",
    "leak_rows",
    "load_benchmark_oracle",
    "load_registry",
    "load_seed_registry",
    "main",
    "materialize",
    "oracle_ite_column",
    "overlap_metrics",
    "read_ledger",
    "read_run_artifacts",
    "resolve_seeds",
    "run_status",
    "scenario_rows",
    "selection_metrics",
    "sensitivity_metrics",
    "short_nuisances",
    "unmeasured_coefficients",
    "write_configs",
]
