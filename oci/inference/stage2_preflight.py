"""Read-only validation of an existing Stage 1 handoff and Stage 2 reuse plan."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from . import research_all_evidence_workflow as workflow


def _object(path: Path) -> dict[str, Any]:
    return workflow._read_json_object(path, description=str(path))


def validate_selection_resume(config: workflow.ResearchStage1Config) -> None:
    """Do not let the workflow's completion shortcut hide a policy change."""
    if config.stage2 is None:
        return
    root = config.output_dir / "stage2"
    state_path = root / "reselection_state.json"
    state = _object(state_path) if state_path.exists() else {}
    if state.get("status") in {"preparing", "prepared"}:
        if state.get("policy_fingerprint") != workflow._stage2_reselection_policy_fingerprint(
            config
        ):
            raise RuntimeError("unfinished Stage 2 reselection uses a different scientific policy")
        if state["status"] == "preparing":
            raise RuntimeError(
                "reselection preparation was interrupted; resume with --stage2-reselect"
            )
    saved_path = root / "config.json"
    if not saved_path.exists():
        return
    saved = _object(saved_path)
    for name in ("min_propensity", "max_propensity"):
        if saved.get(name) != getattr(config.stage2, name):
            raise RuntimeError("Stage 2 propensity bounds changed; use guarded --stage2-reselect")
    from .stage2_elastic_net_selection import statistical_selection_config_from_mapping
    from .stage2_multi_model_config import SCHEMA_VERSION

    saved_multi = (saved.get("statistical_selection") or {}).get("multi_model") or {}
    if saved_multi.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(
            "Stage 2 multi-model selection schema changed; use guarded --stage2-reselect"
        )
    saved_policy = statistical_selection_config_from_mapping(saved.get("statistical_selection"))
    if saved_policy.public_dict() != config.stage2.statistical_selection.public_dict():
        raise RuntimeError("Stage 2 multi-model policy changed; use guarded --stage2-reselect")
    orthogonal = (
        config.stage2.statistical_selection.multi_model.modifier_selection.rule
        == "orthogonal_tests"
    )
    # Only the balanced_evidence count search scores forests of estimation_trees;
    # under orthogonal_tests a change refits final estimation only.
    if not orthogonal and saved.get("estimation_trees") != config.stage2.estimation_trees:
        raise RuntimeError(
            "Stage 2 modifier-count estimation_trees changed; use guarded --stage2-reselect"
        )
    if (
        orthogonal
        and "propensity_clip" in saved
        and saved["propensity_clip"] != config.stage2.propensity_clip
    ):
        raise RuntimeError(
            "Stage 2 propensity_clip changed and the orthogonal_tests modifier rule uses it; "
            "use guarded --stage2-reselect"
        )
    # The modifier tests use the final nuisance learner.
    if orthogonal and "nuisance_learner" in saved:
        from ..models.super_learner_nuisance import learner_spec, super_learner_config_from_mapping

        saved_learner = learner_spec(
            saved["nuisance_learner"], super_learner_config_from_mapping(saved.get("super_learner"))
        )
        if saved_learner != config.stage2.nuisance_learner_spec():
            raise RuntimeError(
                "Stage 2 nuisance_learner or super_learner changed and the orthogonal_tests "
                "modifier rule uses it; use guarded --stage2-reselect"
            )


def _validate_partition(fit: list[int], heldout: list[int], universe: set[int]) -> None:
    if (
        not fit
        or not heldout
        or len(fit) != len(set(fit))
        or len(heldout) != len(set(heldout))
        or set(fit) & set(heldout)
        or set(fit) | set(heldout) != universe
    ):
        raise RuntimeError("saved splits do not partition their parent rows exactly")


def preflight_stage2(
    config: workflow.ResearchStage1Config,
    *,
    reselect: bool = False,
) -> dict[str, Any]:
    """Validate without directory creation, fold resampling, fitting, or network calls."""
    if config.components != ("stage2",) or config.stage2 is None:
        raise ValueError("preflight requires Stage 2-only mode and enabled Stage 2")
    root = config.output_dir
    if not root.is_dir() or not os.access(root, os.W_OK):
        raise RuntimeError(f"existing output directory is missing or not writable: {root}")
    # Constructor checks the saved architecture contract and does not write.
    workflow.ResearchAllEvidenceWorkflow(config)
    baseline = _object(root / "run_config.json")
    baseline["run"] = {**baseline.get("run", {}), "mode": "stage2", "components": []}
    saved = workflow.compile_config(baseline, config_dir=root)
    for name in (
        "dataset",
        "output_dir",
        "unit_id_column",
        "text_column",
        "treatment_column",
        "outcome_column",
        "outcome_type",
        "clinical_question",
        "outer_folds",
        "inner_folds",
        "seed",
        "stage1_architectures",
        "htr_enabled",
        "htr_model",
        "embedding_model",
        "stage1_overrides",
        "neural_query_overrides",
    ):
        if getattr(saved, name) != getattr(config, name):
            raise RuntimeError(f"saved Stage 1 configuration mismatch: {name}")
    handoff = root / "handoff"
    marker = _object(handoff / "complete.json")
    if marker.get("status") != "complete":
        raise RuntimeError("Stage 1 handoff is not complete")
    index = _object(handoff / "index.json")
    if Path(index.get("dataset", "")).resolve() != config.dataset:
        raise RuntimeError("handoff dataset does not match saved configuration")
    columns = {
        "unit_id": config.unit_id_column,
        "text": config.text_column,
        "treatment": config.treatment_column,
        "outcome": config.outcome_column,
    }
    if index.get("columns") != columns:
        raise RuntimeError("handoff columns do not match saved configuration")
    references = [handoff / "evidence.jsonl"]
    references += [handoff / value for value in index.get("sources", {}).values()]
    for path in references:
        if not path.is_file():
            raise FileNotFoundError(f"handoff source is missing: {path}")
    if not index.get("schema_version"):
        # Without an architecture selector the handoff is the raw format: each
        # row is an envelope around one exact row of a component's evidence.
        iterators = {
            name: iter(workflow._iter_jsonl(handoff / ref))
            for name, ref in index.get("sources", {}).items()
        }
        count = 0
        for row in workflow._iter_jsonl(handoff / "evidence.jsonl"):
            source = row.get("source")
            expected = next(iterators[source], None) if source in iterators else None
            if expected is None or row.get("evidence") != expected:
                raise RuntimeError("combined handoff differs from frozen component evidence")
            for key in ("outer_fold", "inner_fold", "scope"):
                if row.get(key) != expected.get(key):
                    raise RuntimeError(f"combined handoff {key} differs from component evidence")
            count += 1
        if not count or any(next(rows, None) is not None for rows in iterators.values()):
            raise RuntimeError("combined handoff is missing component evidence")
        if index.get("rows", count) != count or marker.get("rows", count) != count:
            raise RuntimeError("handoff row count does not match its completion metadata")
    component_names = set(index.get("sources", {}))
    if (root / "components" / "embedding_cache").exists():
        component_names.add("embedding_cache")
    for name in component_names:
        directory = root / "components" / name
        completion = _object(directory / "complete.json")
        if completion.get("status") != "complete":
            raise RuntimeError(f"Stage 1 component is incomplete: {directory}")
        for ref in completion.get("artifacts", []):
            path = Path(ref)
            if not path.is_absolute():
                path = directory / path
            if not path.exists():
                raise FileNotFoundError(f"Stage 1 component artifact is missing: {path}")
    data = workflow._load_reselection_dataset(config)
    if data[config.unit_id_column].isna().any() or data[config.unit_id_column].duplicated().any():
        raise RuntimeError("dataset unit IDs must be nonmissing and unique")
    from .tfidf_topic_stage1 import tfidf_topic_dataset_fingerprints
    from .tfidf_topic_split_registry import validate_handoff_rows_against_split_registry
    from .plain_handoff_stage2 import _has_extraction_dependent_checkpoints, _load_stage2_splits
    from .plain_handoff_stage2_evidence import stage1_embedding_cache_dependency_identity

    applied = workflow.ExperimentConfig.from_dict(
        {"applied_inference": _object(root / "resolved_stage1_model_config.json")}
    ).applied_inference
    identity = tfidf_topic_dataset_fingerprints(data, applied)
    manifest = _object(root / "components" / "tfidf" / "manifest.json")
    for key in ("content_fingerprint", "ordered_row_fingerprint"):
        if manifest.get("dataset_" + key) != identity[key]:
            raise RuntimeError(f"dataset {key} differs from frozen Stage 1")
    provenance = root / "components" / "tfidf" / "split_provenance.jsonl"
    if not provenance.is_file() or not provenance.stat().st_size:
        raise FileNotFoundError(
            f"saved split provenance required; resampling is forbidden: {provenance}"
        )
    splits = _load_stage2_splits(
        provenance_path=provenance,
        dataset_rows=len(data),
        outer_fold_ids=range(1, config.outer_folds + 1),
        inner_folds=config.inner_folds,
        seed=config.seed,
    )
    if set(splits) != set(range(1, config.outer_folds + 1)):
        raise RuntimeError("saved outer-fold count differs from configuration")
    all_heldout = []
    for split in splits.values():
        _validate_partition(split["fit_row_ids"], split["heldout_row_ids"], set(range(len(data))))
        inner = split["inner_splits"]
        if len(inner) != config.inner_folds or {s["inner_fold"] for s in inner} != set(
            range(1, config.inner_folds + 1)
        ):
            raise RuntimeError("saved inner-fold count differs from configuration")
        inner_heldout = []
        for item in inner:
            _validate_partition(
                item["fit_row_ids"], item["heldout_row_ids"], set(split["fit_row_ids"])
            )
            inner_heldout.extend(item["heldout_row_ids"])
        if sorted(inner_heldout) != sorted(split["fit_row_ids"]):
            raise RuntimeError("saved inner held-out rows do not cover training rows exactly once")
        all_heldout.extend(split["heldout_row_ids"])
    if sorted(all_heldout) != list(range(len(data))):
        raise RuntimeError("saved outer held-out rows do not cover dataset exactly once")
    evidence = workflow._read_jsonl_objects(
        root / "components" / "tfidf" / "evidence.jsonl", description="frozen TF-IDF evidence"
    )
    registry = {"outer_folds": [{**s, "inner_folds": s["inner_splits"]} for s in splits.values()]}
    validate_handoff_rows_against_split_registry(evidence, registry)
    cache = stage1_embedding_cache_dependency_identity(handoff / "evidence.jsonl")
    if "embedding_cache" in component_names and cache is None:
        raise RuntimeError("completed Stage 1 embedding cache is missing")
    if cache and (
        cache["semantic_metadata"].get("num_samples") != len(data)
        or Path(cache["semantic_metadata"]["dataset_path"]).resolve() != config.dataset
    ):
        raise RuntimeError("Stage 1 embedding cache dataset identity mismatch")
    stage2 = root / "stage2"
    for fold, split in splits.items():
        outer = stage2 / f"outer_{fold:03d}"
        selection_path = outer / "selection" / "input.json"
        if selection_path.exists():
            selection = _object(selection_path)
            if selection.get("inner_splits") != split["inner_splits"]:
                raise RuntimeError(f"outer fold {fold} selection splits differ from frozen Stage 1")
        snapshot_path = outer / "preselection" / "input.json"
        if snapshot_path.exists():
            from .plain_handoff_stage2_analysis import _load_frozen_preselection_snapshot

            state_path = stage2 / "reselection_state.json"
            state = _object(state_path) if state_path.exists() else {}
            if state.get("status") != "preparing":
                _load_frozen_preselection_snapshot(
                    output_dir=outer,
                    dataset=data,
                    definitions=_object(outer / "feature_definitions.json")["features"],
                    fit_ids=split["fit_row_ids"],
                    heldout_ids=split["heldout_row_ids"],
                    inner_splits=split["inner_splits"],
                    unit_id_column=config.unit_id_column,
                    text_column=config.text_column,
                    treatment_column=config.treatment_column,
                    outcome_column=config.outcome_column,
                    outcome_type=config.outcome_type,
                    stage1_packets=workflow._read_jsonl_objects(
                        outer / "input_packets.jsonl", description="compiled Stage 1 packets"
                    ),
                    config=config.stage2,
                )
    migration = None
    if reselect:
        migration = workflow.prepare_stage2_reselection(config=config, dry_run=True)
    else:
        validate_selection_resume(config)
    extraction = config.stage2.extraction_llm
    if not config.stage2.model or extraction is None or not extraction.model:
        raise RuntimeError(
            "saved Stage 2 configuration must specify primary and extraction model IDs"
        )
    if (stage2 / "model_identity.json").exists():
        models = _object(stage2 / "model_identity.json")
        for role, model in (("primary", config.stage2.model), ("extraction", extraction.model)):
            if (models.get(role) or {}).get("selected_model") != model:
                # Match the runner's resume contract: changing the extractor
                # is allowed after its measurements and downstream outputs
                # have been removed, while discovery remains frozen.
                if role == "extraction" and not _has_extraction_dependent_checkpoints(stage2):
                    continue
                raise RuntimeError(f"saved {role} model ID differs from requested Stage 2 model")
    return {
        "status": "validated",
        "read_only": True,
        "dataset": str(config.dataset),
        "output_dir": str(root),
        "rows": len(data),
        "columns": columns,
        "seed": config.seed,
        "outer_folds": config.outer_folds,
        "inner_folds": config.inner_folds,
        "dataset_identity": identity,
        "propensity_bounds": {"minimum": config.stage2.min_propensity,
                              "maximum": config.stage2.max_propensity,
                              "inclusive": True},
        "primary_model": config.stage2.model,
        "extraction_model": extraction.model,
        "primary_serving": "managed" if config.stage2.vllm else "external",
        "extraction_serving": "managed" if extraction.vllm else "external",
        "stage1": "reuse frozen handoff, splits and components; no Stage 1 execution",
        "stage2_action": (
            "guarded reselection"
            if reselect
            else ("resume" if stage2.exists() else "fresh Stage 2 discovery and extraction")
        ),
        "migration": migration,
        "live_endpoints_checked": False,
    }
