"""Leave-one-architecture-out Stage 2 ablations: prepare inputs, then compare results.

``prepare`` writes, for each Stage 1 architecture A of a run, a directory
``without_<A>/`` that ``scripts/run_stage2_from_artifacts.py`` can run: the
run's handoff without A's evidence, a copy of the split file, a hash manifest,
and a launcher configuration. It never starts an LLM request.

Handoff filtering works on architecture rows. A handoff written for an explicit
architecture selection already has one architecture per row
(``source == "stage1_architecture"``); rows whose ``evidence.architecture`` is A
are dropped and every other line is copied unchanged. A handoff written without
a selection streams component rows, and one ``text_models`` row can hold the
evidence of several architectures (BoW nuisance and R-loss views, matched pairs,
embedding contrasts, HTR). Such handoffs are first projected into one row per
architecture occurrence with the same extractor Stage 2 uses
(``materialize_stage1_architecture_artifacts``), so removing A drops exactly
the occurrences Stage 2 would attribute to A and keeps the other architectures'
evidence from shared rows. Each ablation's ``stage2.included_architectures``
lists the remaining architectures.

``compare`` reads each finished ablation and the full run fold by fold.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..inference.stage1_architectures import canonicalize_stage1_architectures
from .attribution import (
    ANALYSIS_COHORT,
    DEFAULT_RECOVERY_R2,
    FrozenArtifactReader,
    assert_output_outside_run_inputs,
    canonical_architectures,
    collect_stage2_lineage,
    file_sha256,
    load_oracle_data,
    oracle_recovery,
    resolve_oracle_inputs,
    write_csv,
    write_json,
)

ABLATION_SCHEMA_VERSION = "stage1_architecture_ablation_v1"
ABLATION_INDEX_SCHEMA_VERSION = "stage1_architecture_ablation_index_v1"
ABLATION_HANDOFF_INDEX_SCHEMA_VERSION = "stage1_architecture_ablation_handoff_v1"
COMPARISON_SCHEMA_VERSION = "stage1_architecture_ablation_comparison_v1"
CONTROL_NAME = "full_control"
LAUNCHER_CONFIG_NAME = "stage2_from_artifacts.json"
INDEX_NAME = "ablations.json"
TEMPLATE_KEYS = frozenset({"stage2", "concurrency_control_path", "tokenizer_cache_dir"})
API_KEY_ENVIRONMENT = {
    "stage2.api_key": "OCI_STAGE2_API_KEY",
    "stage2.extraction_llm.api_key": "OCI_STAGE2_EXTRACTION_API_KEY",
    "stage2.extraction_llm.runtime_api_key": "OCI_STAGE2_EXTRACTION_RUNTIME_API_KEY",
}
FILTER_RULE = (
    "Handoff rows are architecture rows. Rows of an explicit-selection handoff "
    "(source stage1_architecture) whose evidence.architecture is the ablated architecture are "
    "dropped and all other lines are copied byte for byte. A handoff without a selection is "
    "first projected into one row per architecture occurrence with Stage 2's own extractor "
    "(materialize_stage1_architecture_artifacts), so shared text_models rows keep the other "
    "architectures' occurrences; the ablated architecture's rows are then dropped."
)
ROLE_SETS = {"selected_modifiers": "effect_modifier", "nuisance_set": "confounder"}
REPO_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _script(name: str) -> str:
    path = REPO_SCRIPTS / name
    return str(path) if path.is_file() else f"scripts/{name}"


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _row_architecture(row: Mapping[str, Any]) -> str | None:
    if str(row.get("source") or "") != "stage1_architecture":
        return None
    evidence = row.get("evidence")
    return str(evidence.get("architecture") or "") if isinstance(evidence, Mapping) else None


# ---------------------------------------------------------------------------
# prepare


def _run_selection(
    run_dir: Path, present: Sequence[str]
) -> tuple[list[str], str]:
    """The run's architecture selection, restricted to architectures with evidence."""

    manifest_path = run_dir / "stage1_architectures" / "manifest.json"
    if manifest_path.is_file():
        manifest = _read_json(manifest_path)
        if str(manifest.get("selection_mode") or "") == "explicit":
            selected = canonicalize_stage1_architectures(
                manifest.get("selected_architectures"), allow_none=False
            )
            return [name for name in selected or () if name in present], (
                "stage1_architectures/manifest.json"
            )
    summary_path = run_dir / "stage2" / "evidence_compilation" / "summary.json"
    if summary_path.is_file():
        signature = _read_json(summary_path).get("compiler_signature") or {}
        included = signature.get("included_architectures")
        if included:
            return (
                [name for name in canonical_architectures(included) if name in present],
                "stage2/evidence_compilation/summary.json",
            )
    return list(present), "handoff"


def _component_sources(run_dir: Path) -> dict[str, str]:
    """Absolute paths of the component evidence that records each context's rows.

    Stage 2 reads them through ``handoff/index.json`` to check the split file
    against every Stage 1 context of an architecture-row handoff.
    """

    handoff_dir = run_dir / "handoff"
    sources: dict[str, str] = {}
    index_path = handoff_dir / "index.json"
    if index_path.is_file():
        for name, relative in (_read_json(index_path).get("sources") or {}).items():
            path = Path(str(relative)).expanduser()
            path = (path if path.is_absolute() else handoff_dir / path).resolve()
            if path.is_file():
                sources[str(name)] = str(path)
    if not sources:
        for name in ("text_models", "tfidf", "neural_queries"):
            path = run_dir / "components" / name / "evidence.jsonl"
            if path.is_file() and path.stat().st_size:
                sources[name] = str(path.resolve())
    if not sources:
        raise FileNotFoundError(
            "cannot find the Stage 1 component evidence behind the handoff "
            f"({index_path} lists no existing sources); Stage 2 needs it to verify folds"
        )
    return sources


def _architecture_rows(
    run_dir: Path, out_dir: Path, sources: Mapping[str, str]
) -> tuple[list[tuple[str, str]], str, dict[str, Any]]:
    """(architecture, JSONL line) pairs for the full handoff, and how they were made."""

    handoff = run_dir / "handoff" / "evidence.jsonl"
    with handoff.open(encoding="utf-8") as stream:
        lines = [line.rstrip("\n") for line in stream if line.strip()]
    rows = [json.loads(line) for line in lines]
    architectures = [_row_architecture(row) for row in rows]
    if rows and all(architectures):
        return [(str(a), line) for a, line in zip(architectures, lines, strict=True)], (
            "architecture_rows_copied"
        ), {}
    from ..inference.plain_handoff_stage2_evidence import (
        extract_stage1_architecture_occurrences,
    )
    from ..inference.stage1_architecture_artifacts import (
        materialize_stage1_architecture_artifacts,
    )

    present = sorted(
        {
            str(occurrence["architecture"])
            for occurrences in extract_stage1_architecture_occurrences(rows).values()
            for occurrence in occurrences
        }
    )
    projection_dir = out_dir / "projection"
    canonical, manifest = materialize_stage1_architecture_artifacts(
        output_dir=projection_dir,
        raw_handoff_rows=rows,
        selected_architectures=canonical_architectures(present),
        source_artifacts={name: Path(path) for name, path in sources.items()},
        selection_mode="explicit",
    )
    pairs = [
        (
            str(row["evidence"]["architecture"]),
            json.dumps(dict(row), sort_keys=True, allow_nan=False),
        )
        for row in canonical
    ]
    return pairs, "projected_to_architecture_rows", {
        "projection_dir": str(projection_dir),
        "projection_manifest": str(projection_dir / "stage1_architectures" / "manifest.json"),
        "projected_rows": len(pairs),
        "source_rows": len(rows),
        "architecture_occurrences": {
            name: entry.get("occurrences")
            for name, entry in (manifest.get("architectures") or {}).items()
        },
    }


def _occurrence_folds(lines: Iterable[str]) -> dict[str, set[int]]:
    folds: dict[str, set[int]] = {}
    for line in lines:
        row = json.loads(line)
        folds.setdefault(str(row["evidence"]["architecture"]), set()).add(int(row["outer_fold"]))
    return folds


def _stage2_dataset(run_dir: Path, run_config: Mapping[str, Any], out_dir: Path) -> dict[str, Any]:
    """The cohort Stage 2 reads: the dated-notes cohort when present, else the run's dataset."""

    cohort = run_dir / ANALYSIS_COHORT
    if cohort.is_file():
        source, origin = cohort.resolve(), "dated_notes_analysis_cohort"
    else:
        value = run_config.get("dataset")
        if not value:
            raise ValueError("run_config.json records no dataset")
        source = Path(str(value)).expanduser()
        source = (source if source.is_absolute() else run_dir / source).resolve(strict=True)
        origin = "run_config.dataset"
    info: dict[str, Any] = {
        "source": str(source),
        "source_origin": origin,
        "source_sha256": file_sha256(source),
    }
    if source.suffix.lower() in {".parquet", ".pq"}:
        info["path"] = str(source)
        info["sha256"] = info["source_sha256"]
        return info
    if source.suffix.lower() != ".csv":
        raise ValueError("the run dataset must be Parquet or CSV")
    # The launcher reads Parquet only. Keep just the four declared columns, in
    # the original row order, so no oracle column is copied.
    columns = [
        str(run_config[key])
        for key in ("unit_id_column", "text_column", "treatment_column", "outcome_column")
    ]
    target = out_dir / "shared" / "stage2_cohort.parquet"
    frame = pd.read_csv(source)[columns].reset_index(drop=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    if target.is_file() and file_sha256(target) == file_sha256(temporary):
        temporary.unlink()
    else:
        os.replace(temporary, target)
    info["path"] = str(target)
    info["sha256"] = file_sha256(target)
    info["converted_from_csv_columns"] = columns
    return info


def _load_template(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    value = _read_json(Path(path).expanduser())
    if not isinstance(value, Mapping):
        raise ValueError("--stage2-config must be a JSON object")
    unknown = sorted(set(value) - TEMPLATE_KEYS)
    if unknown:
        raise ValueError(
            f"--stage2-config has unsupported top-level keys {unknown}; put Stage 2 "
            f"settings under 'stage2' (allowed keys: {sorted(TEMPLATE_KEYS)})"
        )
    if "stage2" in value and not isinstance(value["stage2"], Mapping):
        raise ValueError("--stage2-config 'stage2' must be an object")
    return dict(value)


def stage2_block(run_config: Mapping[str, Any], template: Mapping[str, Any]) -> dict[str, Any]:
    """The run's Stage 2 settings with the template's overrides, checked for the launcher."""

    from ..inference.plain_handoff_stage2 import plain_stage2_config_from_mapping

    base = run_config.get("stage2") if isinstance(run_config.get("stage2"), Mapping) else {}
    merged = _deep_merge(base or {}, template.get("stage2") or {})
    if not merged:
        raise ValueError(
            "the run records no stage2 settings; pass --stage2-config with a 'stage2' block"
        )
    # External endpoints replace managed pools: the launcher never starts vLLM.
    if str(merged.get("endpoint") or "").strip():
        merged["vllm"] = None
    extraction = merged.get("extraction_llm")
    if isinstance(extraction, Mapping) and (
        str(extraction.get("endpoint") or "").strip() or extraction.get("endpoints")
    ):
        merged["extraction_llm"] = {**extraction, "vllm": None}
    merged.pop("required_architectures", None)
    merged.pop("included_architectures", None)
    config = plain_stage2_config_from_mapping(merged, default_workers=8)
    if config is None:
        raise ValueError("stage2 settings configure no endpoint; pass --stage2-config")
    if config.vllm is not None or (
        config.extraction_llm is not None and config.extraction_llm.vllm is not None
    ):
        raise ValueError(
            "run_stage2_from_artifacts.py does not start managed vLLM pools; pass "
            "--stage2-config with stage2.endpoint and stage2.extraction_llm.endpoint "
            "(or endpoints) for running servers"
        )
    if config.extraction_llm is None:
        raise ValueError("Stage 2 with a dataset requires stage2.extraction_llm")
    return merged


def _key_sources(block: Mapping[str, Any]) -> dict[str, str]:
    """Where each API key will come from when the launcher starts."""

    from ..extraction.llm_routing import REDACTED_API_KEY

    def value(path: str) -> Any:
        current: Any = {"stage2": block}
        for part in path.split("."):
            current = current.get(part) if isinstance(current, Mapping) else None
        return current

    extraction = block.get("extraction_llm")
    output = {}
    for path, variable in API_KEY_ENVIRONMENT.items():
        if path.startswith("stage2.extraction_llm") and not isinstance(extraction, Mapping):
            continue
        if path.endswith("runtime_api_key") and not str(
            (extraction or {}).get("runtime_endpoint") or ""
        ).strip():
            continue
        key = value(path)
        output[path] = (
            f"environment variable {variable}"
            if not key or str(key) in {REDACTED_API_KEY, "EMPTY"}
            else "written in the configuration"
        )
    return output


def _write_text_if_changed(path: Path, text: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    if path.is_file() and path.read_bytes() == data:
        return False
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)
    return True


def _link_embedding_cache(run_dir: Path, directory: Path) -> str | None:
    """Link the run's embedding cache so Stage 2 groups cards exactly as in the run.

    Stage 2 looks for ``components/embedding_cache`` two levels above the
    handoff; without it, card grouping falls back to lexical hashing.
    """

    source = run_dir / "components" / "embedding_cache"
    link = directory / "components" / "embedding_cache"
    if not source.is_dir():
        return None
    target = source.resolve()
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() == target:
            return str(target)
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"{link} exists and is not a link to the run's embedding cache")
    link.symlink_to(target, target_is_directory=True)
    return str(target)


def prepare_ablations(
    *,
    run_dir: Path,
    out_dir: Path,
    architectures: str | Sequence[str] | None = None,
    stage2_template: Path | None = None,
    include_control: bool = False,
) -> dict[str, Any]:
    """Write one leave-one-architecture-out input directory per architecture."""

    run_dir = Path(run_dir).expanduser().resolve(strict=True)
    out_dir = Path(out_dir).expanduser().resolve()
    assert_output_outside_run_inputs(out_dir, run_dir)
    run_config_path = run_dir / "run_config.json"
    if not run_config_path.is_file():
        raise FileNotFoundError(f"run has no run_config.json: {run_config_path}")
    run_config = _read_json(run_config_path)
    handoff = run_dir / "handoff" / "evidence.jsonl"
    if not handoff.is_file() or not (handoff.parent / "complete.json").is_file():
        raise FileNotFoundError(f"run has no completed handoff: {handoff}")
    split = run_dir / "components" / "tfidf" / "split_provenance.jsonl"
    if not split.is_file():
        raise FileNotFoundError(f"run has no split file: {split}")
    template = _load_template(stage2_template)
    block = stage2_block(run_config, template)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = _component_sources(run_dir)
    pairs, handoff_mode, projection = _architecture_rows(run_dir, out_dir, sources)
    present = canonical_architectures(architecture for architecture, _line in pairs)
    selected, selection_source = _run_selection(run_dir, present)
    requested = (
        list(selected)
        if architectures is None
        else list(canonicalize_stage1_architectures(architectures, allow_none=False) or ())
    )
    outside = sorted(set(requested) - set(selected))
    if outside:
        raise ValueError(
            f"architectures {outside} are not in this run's selection {selected}"
        )
    dataset = _stage2_dataset(run_dir, run_config, out_dir)
    handoff_sha256 = file_sha256(handoff)
    split_sha256 = file_sha256(split)
    outer_folds = sorted({fold for folds in _occurrence_folds(
        line for _architecture, line in pairs).values() for fold in folds})
    targets: list[tuple[str, str | None]] = [(f"without_{name}", name) for name in requested]
    if include_control:
        targets.append((CONTROL_NAME, None))
    prepared: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    key_sources = _key_sources(block)
    for name, removed in targets:
        kept = [architecture for architecture in selected if architecture != removed]
        lines = [line for architecture, line in pairs if architecture in set(kept)]
        if not kept or not lines:
            reason = (
                f"removing {removed} leaves an empty handoff"
                if removed
                else "the handoff has no evidence of the selected architectures"
            )
            skipped.append({"name": name, "removed_architecture": removed, "reason": reason})
            print(f"skip {name}: {reason}")
            continue
        directory = out_dir / name
        evidence_path = directory / "handoff" / "evidence.jsonl"
        text = "".join(f"{line}\n" for line in lines)
        new_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if (
            evidence_path.is_file()
            and file_sha256(evidence_path) != new_sha256
            and (directory / "stage2").exists()
        ):
            raise RuntimeError(
                f"{directory} holds a Stage 2 run on a different handoff; "
                "use another --out directory"
            )
        _write_text_if_changed(evidence_path, text)
        folds_by_architecture = _occurrence_folds(lines)
        # Stage 2 requires each listed architecture in every outer fold.
        required = [
            architecture
            for architecture in kept
            if folds_by_architecture.get(architecture, set()) >= set(outer_folds)
        ]
        handoff_index = {
            "schema_version": ABLATION_HANDOFF_INDEX_SCHEMA_VERSION,
            "combined_evidence": "evidence.jsonl",
            "rows": len(lines),
            "selected_architectures": kept,
            "removed_architecture": removed,
            "sources": dict(sources),
            "source_storage": "referenced_without_copy",
            "source_handoff": {"path": str(handoff), "sha256": handoff_sha256},
            "dataset": dataset["path"],
            "columns": {
                key: run_config.get(f"{key}_column")
                for key in ("unit_id", "text", "treatment", "outcome")
            },
        }
        _write_text_if_changed(
            directory / "handoff" / "index.json",
            json.dumps(handoff_index, indent=2, sort_keys=True) + "\n",
        )
        split_copy = directory / "components" / "tfidf" / "split_provenance.jsonl"
        split_copy.parent.mkdir(parents=True, exist_ok=True)
        if not split_copy.is_file() or file_sha256(split_copy) != split_sha256:
            shutil.copy2(split, split_copy)
        embedding_cache = _link_embedding_cache(run_dir, directory)
        stage2 = copy.deepcopy(block)
        stage2["required_architectures"] = required
        stage2["included_architectures"] = kept
        config = {
            "dataset": dataset["path"],
            "handoff_path": str(evidence_path),
            "split_provenance_path": str(split_copy),
            "output_dir": str(directory / "stage2"),
            "report_dir": str(directory / "stage2_report"),
            "clinical_question": run_config["clinical_question"],
            "unit_id_column": run_config["unit_id_column"],
            "text_column": run_config["text_column"],
            "treatment_column": run_config["treatment_column"],
            "outcome_column": run_config["outcome_column"],
            "outcome_type": run_config["outcome_type"],
            "inner_folds": int(run_config["inner_folds"]),
            "seed": int(run_config["seed"]),
            "stage2": stage2,
            **{
                key: template[key]
                for key in ("concurrency_control_path", "tokenizer_cache_dir")
                if template.get(key)
            },
        }
        config_path = directory / LAUNCHER_CONFIG_NAME
        _write_text_if_changed(config_path, json.dumps(config, indent=2, sort_keys=True) + "\n")
        removed_rows = {
            architecture: sum(1 for value, _line in pairs if value == architecture)
            for architecture in present
            if architecture not in kept
        }
        command = f"python {_script('run_stage2_from_artifacts.py')} {config_path}"
        manifest = {
            "schema_version": ABLATION_SCHEMA_VERSION,
            "created_at": _now(),
            "name": name,
            "removed_architecture": removed,
            "kept_architectures": kept,
            "required_architectures": required,
            "run_selected_architectures": selected,
            "run_selection_source": selection_source,
            "source_run_dir": str(run_dir),
            "source_dataset": dataset["source"],
            "filter_rule": FILTER_RULE,
            "handoff_mode": handoff_mode,
            "rows": {
                "kept": len(lines),
                "removed_by_architecture": removed_rows,
                "kept_by_architecture": {
                    architecture: sum(1 for value, _line in pairs if value == architecture)
                    for architecture in kept
                },
            },
            "embedding_cache": embedding_cache,
            "api_keys": key_sources,
            "sha256": {
                "source_handoff": handoff_sha256,
                "handoff": file_sha256(evidence_path),
                "handoff_index": file_sha256(directory / "handoff" / "index.json"),
                "source_split_provenance": split_sha256,
                "split_provenance": file_sha256(split_copy),
                "dataset": dataset["sha256"],
                "source_dataset": dataset["source_sha256"],
                "launcher_config": file_sha256(config_path),
                "component_sources": {
                    source: file_sha256(Path(path)) for source, path in sorted(sources.items())
                },
            },
            "commands": {
                "preflight": f"{command} --preflight",
                "run": (
                    "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
                    f"NUMEXPR_NUM_THREADS=1 {command}"
                ),
            },
        }
        write_json(directory / "ablation.json", manifest)
        prepared.append(
            {
                "name": name,
                "directory": str(directory),
                "removed_architecture": removed,
                "kept_architectures": kept,
                "launcher_config": str(config_path),
                "commands": manifest["commands"],
            }
        )
    compare_command = (
        f"python {_script('ablate_stage1_architectures.py')} compare "
        f"--run-dir {run_dir} --ablation-dir {out_dir}"
    )
    index = {
        "schema_version": ABLATION_INDEX_SCHEMA_VERSION,
        "created_at": _now(),
        "source_run_dir": str(run_dir),
        "run_selected_architectures": selected,
        "run_selection_source": selection_source,
        "handoff_mode": handoff_mode,
        "projection": projection,
        "filter_rule": FILTER_RULE,
        "stage2_template": None if stage2_template is None else str(Path(stage2_template)),
        "dataset": dataset,
        "outer_folds": outer_folds,
        "ablations": prepared,
        "skipped": skipped,
        "compare_command": compare_command,
    }
    write_json(out_dir / INDEX_NAME, index)
    print(f"prepared {len(prepared)} ablation(s) in {out_dir}")
    environment = sorted(
        {
            source.removeprefix("environment variable ")
            for source in key_sources.values()
            if source.startswith("environment variable")
        }
    )
    if environment:
        print("API keys are read when each launch starts from: " + ", ".join(environment))
    if any(source == "written in the configuration" for source in key_sources.values()):
        print("warning: the --stage2-config API keys are written into every launcher config")
    for item in prepared:
        print(f"\n# {item['name']}")
        print(item["commands"]["preflight"])
        print(item["commands"]["run"])
    print(f"\n# after the ablations finish\n{compare_command}")
    return index


# ---------------------------------------------------------------------------
# compare


def _ablation_entries(ablation_dir: Path) -> list[dict[str, Any]]:
    index_path = ablation_dir / INDEX_NAME
    if index_path.is_file():
        entries = list(_read_json(index_path).get("ablations") or [])
    else:
        entries = [
            {"name": path.parent.name, "directory": str(path.parent)}
            for path in sorted(ablation_dir.glob("*/ablation.json"))
        ]
    output = []
    for entry in entries:
        directory = Path(str(entry["directory"]))
        if not directory.is_dir() and (ablation_dir / str(entry.get("name") or "")).is_dir():
            # The ablation directory was moved after prepare.
            directory = ablation_dir / str(entry["name"])
        manifest_path = directory / "ablation.json"
        manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
        output.append(
            {
                "name": str(entry.get("name") or directory.name),
                "directory": directory,
                "removed_architecture": manifest.get(
                    "removed_architecture", entry.get("removed_architecture")
                ),
            }
        )
    if not output:
        raise FileNotFoundError(f"no prepared ablations under {ablation_dir}")
    return output


def _ablation_status(directory: Path) -> tuple[str, str]:
    stage2 = directory / "stage2"
    status_path = directory / "stage2_report" / "status.json"
    phase = None
    if status_path.is_file():
        try:
            phase = str(_read_json(status_path).get("phase") or "") or None
        except (OSError, ValueError):
            phase = None
    finished = (stage2 / "causal_estimate.json").is_file() and (
        stage2 / "heterogeneity_diagnostics.json"
    ).is_file()
    if finished and phase in {None, "complete"}:
        return "complete", "complete"
    if phase:
        return "pending", phase
    return "pending", "not_started" if not stage2.exists() else "incomplete"


def _load_result(stage2_dir: Path, reader: FrozenArtifactReader) -> dict[str, Any]:
    folds: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(stage2_dir.glob("outer_*/final_definitions.json")):
        suffix = path.parent.name[len("outer_"):]
        if not suffix.isdigit():
            continue
        payload = reader.json(path)
        folds[str(int(suffix))] = [
            dict(feature) for feature in (payload or {}).get("features") or []
        ]
    return {
        "folds": folds,
        "causal_estimate": reader.json(stage2_dir / "causal_estimate.json") or {},
        "heterogeneity": reader.json(stage2_dir / "heterogeneity_diagnostics.json") or {},
    }


def _estimate(section: Any) -> dict[str, Any]:
    section = section if isinstance(section, Mapping) else {}
    interval = section.get("confidence_interval_95")
    interval = list(interval) if isinstance(interval, (list, tuple)) and len(interval) == 2 else None
    return {
        "estimate": section.get("estimate"),
        "standard_error": section.get("standard_error"),
        "ci_low": None if interval is None else interval[0],
        "ci_high": None if interval is None else interval[1],
    }


def _estimates(result: Mapping[str, Any]) -> dict[str, Any]:
    from ..inference.stage2_diagnostics import primary_heterogeneity_test

    causal = result["causal_estimate"]
    co_primary = causal.get("co_primary_estimands") or {}
    ate_section = co_primary.get("ate") or {
        "estimate": causal.get("ate"),
        "standard_error": causal.get("standard_error"),
        "confidence_interval_95": causal.get("confidence_interval_95"),
    }
    ato_section = co_primary.get("ato") or {}
    heterogeneity = result["heterogeneity"]
    primary = primary_heterogeneity_test(heterogeneity) if heterogeneity else {}
    blp_folds = (heterogeneity.get("blp") or {}).get("per_fold") or {}
    per_fold = {}
    for fold in sorted(
        set(result["folds"]) | set(ate_section.get("per_fold") or {}),
        key=lambda value: (int(value), value) if str(value).isdigit() else (10**9, str(value)),
    ):
        blp = blp_folds.get(fold) or {}
        per_fold[fold] = {
            "ate": _estimate((ate_section.get("per_fold") or {}).get(fold)),
            "ato": _estimate((ato_section.get("per_fold") or {}).get(fold)),
            "blp_beta2": blp.get("beta2_heterogeneity"),
            "blp_beta2_p_value_one_sided": blp.get("beta2_p_value_one_sided_greater_than_0"),
        }
    interval = primary.get("confidence_interval_95")
    return {
        "ate": _estimate(ate_section),
        "ato": _estimate(ato_section),
        "primary_heterogeneity_test": {
            "status": primary.get("status"),
            "beta2": primary.get("beta2"),
            "standard_error": primary.get("standard_error"),
            "ci_low": interval[0] if isinstance(interval, list) and len(interval) == 2 else None,
            "ci_high": interval[1] if isinstance(interval, list) and len(interval) == 2 else None,
            "p_value_one_sided": primary.get("p_value_one_sided"),
            "reason": primary.get("reason"),
        },
        "per_fold": per_fold,
    }


def compare_concepts(
    full: Sequence[Mapping[str, Any]],
    ablation: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Match two variable sets by concept with the cross-fold matching rule."""

    from ..inference.stage2_diagnostics import (
        CONCEPT_SIMILARITY_THRESHOLD,
        _definition_text,
        match_concepts,
    )

    entries = [
        {
            "fold": side,
            "name": str(feature.get("name") or feature.get("feature_id")),
            "feature_id": str(feature.get("feature_id") or ""),
            "definition_text": _definition_text(feature),
        }
        for side, features in (("full", full), ("ablation", ablation))
        for feature in features
    ]
    matched, lost, gained = [], [], []
    for concept in match_concepts(entries, threshold=CONCEPT_SIMILARITY_THRESHOLD):
        members = {entries[index]["fold"]: entries[index] for index in concept["member_indices"]}
        if "full" in members and "ablation" in members:
            matched.append(
                {
                    "full": members["full"]["name"],
                    "ablation": members["ablation"]["name"],
                    "similarity": concept["minimum_similarity"],
                }
            )
        elif "full" in members:
            lost.append(members["full"]["name"])
        else:
            gained.append(members["ablation"]["name"])
    union = len(matched) + len(lost) + len(gained)
    return {
        "full": len(full),
        "ablation": len(ablation),
        "matched": matched,
        "lost": sorted(lost),
        "gained": sorted(gained),
        # Two empty sets are identical.
        "jaccard": (len(matched) / union) if union else 1.0,
        "similarity_threshold": CONCEPT_SIMILARITY_THRESHOLD,
    }


def _role_features(features: Sequence[Mapping[str, Any]], role: str) -> list[Mapping[str, Any]]:
    return [feature for feature in features if role in {str(r) for r in feature.get("roles") or []}]


def _delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(right) - float(left)


def _recovery_summary(recovery: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {"per_fold": {}, "totals": {}}
    for fold, section in (recovery.get("per_fold") or {}).items():
        if section.get("status") != "ok":
            output["per_fold"][fold] = {"status": section.get("status")}
            continue
        output["per_fold"][fold] = {
            level: list(entry["all_variables"]) for level, entry in section["levels"].items()
        }
    for level, entry in (recovery.get("totals") or {}).items():
        output["totals"][level] = {
            "fold_concept_count": entry["all_variables"]["fold_concept_count"],
            "concepts": list(entry["all_variables"]["concepts"]),
        }
    return output


def _format_estimate(value: Mapping[str, Any]) -> str:
    if value.get("estimate") is None:
        return "n/a"
    text = f"{float(value['estimate']):.3f}"
    if value.get("ci_low") is not None and value.get("ci_high") is not None:
        text += f" ({float(value['ci_low']):.3f}, {float(value['ci_high']):.3f})"
    return text


def compare_ablations(
    *,
    run_dir: Path,
    ablation_dir: Path,
    oracle: bool = False,
    output_dir: Path | None = None,
    recovery_r2: float = DEFAULT_RECOVERY_R2,
    dataset_path: Path | None = None,
    metadata_path: Path | None = None,
) -> dict[str, Any]:
    """Compare finished ablations with the full run, fold by fold."""

    run_dir = Path(run_dir).expanduser().resolve(strict=True)
    ablation_dir = Path(ablation_dir).expanduser().resolve(strict=True)
    output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else ablation_dir / "comparison"
    )
    assert_output_outside_run_inputs(output_dir, run_dir)
    entries = _ablation_entries(ablation_dir)
    for entry in entries:
        for protected in ("handoff", "components", "stage2", "stage2_report"):
            root = (entry["directory"] / protected).resolve()
            if output_dir == root or root in output_dir.parents:
                raise ValueError(f"refusing to write comparison outputs inside {root}")
    if oracle:
        dataset, metadata = resolve_oracle_inputs(
            run_dir, dataset=dataset_path, metadata=metadata_path
        )
    full_stage2 = run_dir / "stage2"
    if not (full_stage2 / "causal_estimate.json").is_file():
        raise FileNotFoundError(f"the full run has no completed Stage 2: {full_stage2}")
    readers = {"full_run": FrozenArtifactReader(run_dir)}
    results = {"full_run": _load_result(full_stage2, readers["full_run"])}
    lineages = {}
    if oracle:
        lineages["full_run"] = collect_stage2_lineage(
            full_stage2, reader=readers["full_run"], include_heldout=True
        )
    statuses = {}
    for entry in entries:
        status, detail = _ablation_status(entry["directory"])
        statuses[entry["name"]] = (status, detail)
        if status != "complete":
            continue
        reader = FrozenArtifactReader(entry["directory"])
        readers[entry["name"]] = reader
        results[entry["name"]] = _load_result(entry["directory"] / "stage2", reader)
        if oracle:
            lineages[entry["name"]] = collect_stage2_lineage(
                entry["directory"] / "stage2", reader=reader, include_heldout=True
            )
    # Every Stage 2 artifact of the full run and the finished ablations is
    # now hashed; only then may oracle values load.
    hashes = {name: reader.freeze() for name, reader in readers.items()}
    recoveries: dict[str, Any] = {}
    oracle_manifest: dict[str, Any] = {"requested": False}
    if oracle:
        oracle_data = load_oracle_data(dataset, metadata, readers=list(readers.values()))
        recoveries = {
            name: _recovery_summary(
                oracle_recovery(lineage, oracle_data, threshold=recovery_r2)
            )
            for name, lineage in lineages.items()
        }
        oracle_manifest = {
            "requested": True,
            "dataset": str(oracle_data.dataset),
            "dataset_sha256": oracle_data.dataset_sha256,
            "metadata": None if oracle_data.metadata is None else str(oracle_data.metadata),
            "oracle_columns": [concept.column for concept in oracle_data.concepts],
            "threshold_r_squared": float(recovery_r2),
            "stage2_artifacts_hashed_before_oracle_load": True,
        }
    full = results["full_run"]
    full_estimates = _estimates(full)
    comparisons: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = [
        {
            "ablation": "full_run",
            "removed_architecture": "",
            "status": "complete",
            "detail": "reference",
            **_summary_estimates(full_estimates, full_estimates),
            **(
                _summary_recovery(recoveries["full_run"], recoveries["full_run"])
                if oracle
                else {}
            ),
        }
    ]
    for entry in entries:
        status, detail = statuses[entry["name"]]
        item: dict[str, Any] = {
            "name": entry["name"],
            "directory": str(entry["directory"]),
            "removed_architecture": entry["removed_architecture"],
            "status": status,
            "detail": detail,
        }
        if status != "complete":
            comparisons.append(item)
            summary_rows.append(
                {
                    "ablation": entry["name"],
                    "removed_architecture": entry["removed_architecture"] or "",
                    "status": status,
                    "detail": detail,
                }
            )
            continue
        result = results[entry["name"]]
        estimates = _estimates(result)
        folds = sorted(
            set(full["folds"]) | set(result["folds"]),
            key=lambda value: (int(value), value) if value.isdigit() else (10**9, value),
        )
        per_fold = {}
        for fold in folds:
            fold_result: dict[str, Any] = {}
            for role_set, role in ROLE_SETS.items():
                fold_result[role_set] = compare_concepts(
                    _role_features(full["folds"].get(fold, []), role),
                    _role_features(result["folds"].get(fold, []), role),
                )
            fold_result["full_estimates"] = full_estimates["per_fold"].get(fold, {})
            fold_result["ablation_estimates"] = estimates["per_fold"].get(fold, {})
            if oracle:
                fold_result["oracle_recovery"] = {
                    "full_run": recoveries["full_run"]["per_fold"].get(fold),
                    "ablation": recoveries[entry["name"]]["per_fold"].get(fold),
                }
            per_fold[fold] = fold_result
            fold_rows.append(_fold_row(entry, fold, fold_result, oracle))
        item.update(
            {
                "estimates": estimates,
                "estimate_differences_ablation_minus_full": {
                    "ate": _delta(full_estimates["ate"]["estimate"], estimates["ate"]["estimate"]),
                    "ato": _delta(full_estimates["ato"]["estimate"], estimates["ato"]["estimate"]),
                    "blp_beta2": _delta(
                        full_estimates["primary_heterogeneity_test"]["beta2"],
                        estimates["primary_heterogeneity_test"]["beta2"],
                    ),
                },
                "per_fold": per_fold,
                **({"oracle_recovery": recoveries[entry["name"]]} if oracle else {}),
            }
        )
        comparisons.append(item)
        jaccards = {
            role_set: [per_fold[fold][role_set]["jaccard"] for fold in folds]
            for role_set in ROLE_SETS
        }
        summary_rows.append(
            {
                "ablation": entry["name"],
                "removed_architecture": entry["removed_architecture"] or "",
                "status": status,
                "detail": detail,
                "folds_compared": len(folds),
                **{
                    f"{prefix}_{key}": value
                    for role_set, prefix in (
                        ("selected_modifiers", "modifiers"),
                        ("nuisance_set", "nuisance"),
                    )
                    for key, value in (
                        ("mean_jaccard", float(np.mean(jaccards[role_set]))),
                        ("lost", sum(len(per_fold[f][role_set]["lost"]) for f in folds)),
                        ("gained", sum(len(per_fold[f][role_set]["gained"]) for f in folds)),
                    )
                },
                **_summary_estimates(estimates, full_estimates),
                **(
                    _summary_recovery(recoveries[entry["name"]], recoveries["full_run"])
                    if oracle
                    else {}
                ),
            }
        )
    pending = [item["name"] for item in comparisons if item["status"] != "complete"]
    report = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "created_at": _now(),
        "run_dir": str(run_dir),
        "ablation_dir": str(ablation_dir),
        "full_run": {"estimates": full_estimates, **(
            {"oracle_recovery": recoveries["full_run"]} if oracle else {}
        )},
        "concept_matching": (
            "stage2_diagnostics.match_concepts on name, description and measurement "
            "definition (TF-IDF word and character n-grams, complete linkage, threshold 0.7), "
            "with the full run and the ablation as the two sides; Jaccard is matched concepts "
            "over all concepts, and 1 when both sets are empty"
        ),
        "ablations": comparisons,
        "pending": pending,
        "stage2_artifact_sha256": hashes,
        "oracle": oracle_manifest,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "comparison.json", report)
    leading = [
        "ablation", "removed_architecture", "status", "detail", "folds_compared",
        *(f"{prefix}_{key}" for prefix in ("modifiers", "nuisance")
          for key in ("mean_jaccard", "lost", "gained")),
    ]
    summary_columns = list(
        dict.fromkeys([*leading, *(key for row in summary_rows for key in row)])
    )
    write_csv(output_dir / "summary.csv", summary_rows, summary_columns)
    fold_columns = list(dict.fromkeys(key for row in fold_rows for key in row)) or [
        "ablation", "removed_architecture", "outer_fold",
    ]
    write_csv(output_dir / "folds.csv", fold_rows, fold_columns)
    (output_dir / "summary.md").write_text(
        _markdown(report, summary_rows, oracle), encoding="utf-8"
    )
    print(f"compared {len(comparisons) - len(pending)} ablation(s); pending: {pending or 'none'}")
    print(f"wrote {output_dir}")
    return {
        "output_dir": str(output_dir),
        "complete": [item["name"] for item in comparisons if item["status"] == "complete"],
        "pending": pending,
    }


def _summary_estimates(estimates: Mapping[str, Any], full: Mapping[str, Any]) -> dict[str, Any]:
    primary = estimates["primary_heterogeneity_test"]
    return {
        "ate": estimates["ate"]["estimate"],
        "ate_ci_low": estimates["ate"]["ci_low"],
        "ate_ci_high": estimates["ate"]["ci_high"],
        "ate_minus_full": _delta(full["ate"]["estimate"], estimates["ate"]["estimate"]),
        "ato": estimates["ato"]["estimate"],
        "ato_ci_low": estimates["ato"]["ci_low"],
        "ato_ci_high": estimates["ato"]["ci_high"],
        "ato_minus_full": _delta(full["ato"]["estimate"], estimates["ato"]["estimate"]),
        "blp_beta2": primary["beta2"],
        "blp_beta2_ci_low": primary["ci_low"],
        "blp_beta2_ci_high": primary["ci_high"],
        "primary_test_p_value": primary["p_value_one_sided"],
    }


def _summary_recovery(recovery: Mapping[str, Any], full: Mapping[str, Any]) -> dict[str, Any]:
    output = {}
    for level in ("selected_variables", "selected_modifiers", "nuisance_set"):
        mine = recovery["totals"].get(level) or {}
        reference = full["totals"].get(level) or {}
        concepts = set(mine.get("concepts") or [])
        reference_concepts = set(reference.get("concepts") or [])
        output[f"oracle_recovered_{level}"] = mine.get("fold_concept_count")
        output[f"oracle_lost_{level}"] = ";".join(sorted(reference_concepts - concepts))
        output[f"oracle_gained_{level}"] = ";".join(sorted(concepts - reference_concepts))
    return output


def _fold_row(
    entry: Mapping[str, Any], fold: str, result: Mapping[str, Any], oracle: bool
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "ablation": entry["name"],
        "removed_architecture": entry["removed_architecture"] or "",
        "outer_fold": fold,
    }
    for role_set, prefix in (("selected_modifiers", "modifiers"), ("nuisance_set", "nuisance")):
        concepts = result[role_set]
        row.update(
            {
                f"{prefix}_full": concepts["full"],
                f"{prefix}_ablation": concepts["ablation"],
                f"{prefix}_matched": len(concepts["matched"]),
                f"{prefix}_jaccard": concepts["jaccard"],
                f"{prefix}_lost": ";".join(concepts["lost"]),
                f"{prefix}_gained": ";".join(concepts["gained"]),
            }
        )
    for side in ("full", "ablation"):
        estimates = result[f"{side}_estimates"] or {}
        for estimand in ("ate", "ato"):
            value = estimates.get(estimand) or {}
            row[f"{estimand}_{side}"] = value.get("estimate")
            row[f"{estimand}_{side}_ci_low"] = value.get("ci_low")
            row[f"{estimand}_{side}_ci_high"] = value.get("ci_high")
        row[f"blp_beta2_{side}"] = estimates.get("blp_beta2")
        row[f"blp_beta2_p_{side}"] = estimates.get("blp_beta2_p_value_one_sided")
    if oracle:
        for side in ("full_run", "ablation"):
            section = result["oracle_recovery"].get(side) or {}
            for level in ("selected_variables", "selected_modifiers", "nuisance_set"):
                values = section.get(level)
                row[f"oracle_{level}_{side}"] = (
                    ";".join(values) if isinstance(values, list) else ""
                )
    return row


def _markdown(report: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], oracle: bool) -> str:
    def number(value: Any, digits: int = 3) -> str:
        return "n/a" if value is None else f"{float(value):.{digits}f}"

    lines = [
        "# Stage 1 architecture ablations",
        "",
        f"Full run: `{report['run_dir']}`",
        "",
        (
            "Each ablation reruns Stage 2 without one architecture's evidence, so its "
            "differences combine the ablation with run-to-run LLM variation; compare them "
            f"with the `{CONTROL_NAME}` rerun when one was prepared."
        ),
        "",
    ]
    header = ["Ablation", "Status", "ATE (95% CI)", "ATO (95% CI)", "BLP beta2", "Primary p",
              "Modifier Jaccard", "Modifiers lost/gained"]
    if oracle:
        header.append("Oracle concepts recovered (fold total)")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for row in rows:
        if row["status"] != "complete":
            cells = [str(row["ablation"]), f"pending ({row['detail']})", *[""] * (len(header) - 2)]
            lines.append("| " + " | ".join(cells) + " |")
            continue
        cells = [
            str(row["ablation"]),
            str(row["detail"]) if row["ablation"] == "full_run" else "complete",
            _format_estimate({"estimate": row["ate"], "ci_low": row["ate_ci_low"],
                              "ci_high": row["ate_ci_high"]}),
            _format_estimate({"estimate": row["ato"], "ci_low": row["ato_ci_low"],
                              "ci_high": row["ato_ci_high"]}),
            number(row["blp_beta2"]),
            number(row["primary_test_p_value"], 4),
            number(row.get("modifiers_mean_jaccard"), 2) if "modifiers_mean_jaccard" in row else "",
            (f"{row['modifiers_lost']}/{row['modifiers_gained']}"
             if "modifiers_lost" in row else ""),
        ]
        if oracle:
            cells.append(str(row.get("oracle_recovered_selected_variables", "")))
        lines.append("| " + " | ".join(cells) + " |")
    if report["pending"]:
        lines.extend(["", "Pending: " + ", ".join(f"`{name}`" for name in report["pending"])])
    lines.extend([
        "",
        "Fold-level concept matches, losses, and gains are in `folds.csv` and `comparison.json`.",
        "",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare leave-one-architecture-out Stage 2 inputs from a finished run, and "
            "compare the finished ablations with the full run. Never starts LLM requests."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="write one ablation directory per architecture")
    prepare.add_argument("--run-dir", required=True, type=Path, help="workflow output directory")
    prepare.add_argument("--out", required=True, type=Path, help="ablation directory to write")
    prepare.add_argument("--architectures",
                         help="comma-separated architectures to ablate (default: all in the run)")
    prepare.add_argument("--stage2-config", type=Path,
                         help="JSON with a 'stage2' block (endpoints, keys) merged over the "
                         "run's Stage 2 settings, plus optional concurrency_control_path and "
                         "tokenizer_cache_dir")
    prepare.add_argument("--include-control", action="store_true",
                         help=f"also write {CONTROL_NAME}/, a rerun with every architecture")
    compare = commands.add_parser("compare", help="compare finished ablations with the full run")
    compare.add_argument("--run-dir", required=True, type=Path, help="workflow output directory")
    compare.add_argument("--ablation-dir", required=True, type=Path,
                         help="directory written by prepare")
    compare.add_argument("--oracle", action="store_true",
                         help="add oracle recovery (synthetic cohorts with true_* columns)")
    compare.add_argument("--out", type=Path,
                         help="output directory (default ABLATION_DIR/comparison)")
    compare.add_argument("--recovery-r2", type=float, default=DEFAULT_RECOVERY_R2,
                         help="held-out R^2 at which a variable recovers an oracle concept")
    compare.add_argument("--dataset", type=Path, help="oracle-bearing dataset (with --oracle)")
    compare.add_argument("--metadata", type=Path, help="generator metadata.json (with --oracle)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        prepare_ablations(
            run_dir=args.run_dir,
            out_dir=args.out,
            architectures=args.architectures,
            stage2_template=args.stage2_config,
            include_control=args.include_control,
        )
    else:
        compare_ablations(
            run_dir=args.run_dir,
            ablation_dir=args.ablation_dir,
            oracle=args.oracle,
            output_dir=args.out,
            recovery_r2=args.recovery_r2,
            dataset_path=args.dataset,
            metadata_path=args.metadata,
        )
    return 0


__all__ = [
    "ABLATION_INDEX_SCHEMA_VERSION",
    "ABLATION_SCHEMA_VERSION",
    "COMPARISON_SCHEMA_VERSION",
    "CONTROL_NAME",
    "FILTER_RULE",
    "build_parser",
    "compare_ablations",
    "compare_concepts",
    "main",
    "prepare_ablations",
    "stage2_block",
]
