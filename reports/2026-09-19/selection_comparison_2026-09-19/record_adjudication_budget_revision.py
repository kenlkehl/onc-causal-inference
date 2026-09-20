"""Freeze the comparison-only prompt correction and its checkpoint provenance."""
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from compare import HERE, immutable_json, now, read_json, sha256, validate_run_inputs

OUT = HERE / "results"
PRIOR = OUT / "revisions/structural_reasoning_v3"
REVISION = OUT / "revisions/adjudication_budget_v4"


def main():
    manifest_path = REVISION / "manifest.json"
    if manifest_path.exists():
        raise RuntimeError("Revision is frozen; validate it instead of rewriting it")
    prior_path = PRIOR / "manifest.json"
    prior = read_json(prior_path)
    diagnosis = read_json(REVISION / "failure_diagnosis.json")
    if diagnosis["prior_manifest_sha256"] != sha256(prior_path):
        raise ValueError("Prior revision changed after the failure was recorded")
    if diagnosis["prior_runner_state"]["phase"] != "failed" or diagnosis["prior_runner_state"]["returncode"] != 1:
        raise ValueError("Expected the documented prompt-limit failure")
    before = read_json(REVISION / "before_sources.json")
    source = str(HERE / "compare.py")
    if sha256(before["files"][source]["archive"]) != prior["source_overrides"][source]["after_sha256"]:
        raise ValueError("Archived comparison driver differs from the prior revision")
    if sha256(source) == before["files"][source]["sha256"]:
        raise ValueError("The comparison driver has not been corrected")

    def completed_patient(path):
        marker = path / "complete.json"
        return (str(marker), {"sha256": sha256(marker)}) if marker.is_file() else None

    paths = set()
    for pattern in (
        "outer_*/ontology_supervision/round_*/extraction/batches",
        "outer_*/ontology_supervision/round_*/extraction/changed_features/batches",
        "outer_*/ontology_supervision/round_*/failure_ontology_refinement/round_*/extraction/changed_features/batches",
        "outer_*/extraction/*/batches", "outer_*/extraction/*/changed_features/batches",
    ):
        paths.update((OUT / "refresh").glob(pattern))
    markers = {}
    scopes = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        for path in sorted(paths):
            patients = list(path.glob("batch_*"))
            complete = [item for item in pool.map(completed_patient, patients) if item is not None]
            markers.update(complete)
            scopes.append({"path": str(path), "patients_started": len(patients),
                           "patients_complete": len(complete)})
    aggregate_files = {}
    for pattern in (
        "outer_*/comparison_measurements/fit.*", "outer_*/selection/*.json",
        "outer_*/selection/nuisance_predictions.csv", "outer_*/extraction/*/extracted.csv",
        "outer_*/estimation/complete.json", "outer_*/complete.json",
    ):
        for path in (OUT / "refresh").glob(pattern):
            aggregate_files[str(path)] = {"sha256": sha256(path)}
    inventory = REVISION / "checkpoints_before.json"
    immutable_json(inventory, {"recorded_at": now(), "scopes": scopes,
        "patient_markers": markers, "aggregate_files": aggregate_files,
        "scope": "Completed patient markers and aggregate checkpoints; no patient values were opened."})

    config_path = REVISION / "refresh_config.json"
    immutable_json(config_path, read_json(prior["config"]["path"]))
    if sha256(config_path) != prior["config"]["sha256"]:
        raise ValueError("The runtime configuration must remain byte-identical")
    revision = copy.deepcopy(prior)
    revision.update(revision="adjudication_budget_v4", recorded_at=now(),
        authorization="Checkpoint-compatible implementation correction within the authorized comparison and recovery scope.",
        parent_revision={"path": str(prior_path), "sha256": sha256(prior_path)},
        prior_run_failure=diagnosis,
        policy_change={"scope": "Comparison driver only; production source and frozen configuration unchanged.",
            "before": "Binding LLM batches use the configured candidate count without a rendered-size preflight.",
            "after": "Choose the largest uniform candidate count no greater than the configured maximum for which all full rendered prompts fit the existing character limit.",
            "candidate_evidence_changed": False, "admission_rules_changed": False,
            "llm_batch_context_changed": True, "oracle_values_opened": False},
        config={"path": str(config_path), "sha256": sha256(config_path)})
    revision["source_overrides"][source]["after_sha256"] = sha256(source)
    frozen = [prior_path, Path(prior["config"]["path"]), inventory,
              REVISION / "before_sources.json", REVISION / "failure_diagnosis.json"]
    frozen.extend(Path(record["archive"]) for record in before["files"].values())
    frozen.extend(path for path in (REVISION / "failed_adjudication").rglob("*") if path.is_file())
    frozen.extend(path for path in (REVISION / "validation").rglob("*") if path.is_file())
    frozen.extend(REVISION.glob("prior_*"))
    frozen.extend([REVISION / "focused_tests.log", REVISION / "full_tests.log"])
    revision["frozen_inputs"].update({str(path): {"sha256": sha256(path)} for path in frozen})
    additional = [Path(__file__).resolve(), HERE / "resume_adjudication_budget.py",
                  HERE / "test_compare.py", HERE / "ADJUDICATION_BUDGET_RECOVERY_2026-09-20.md"]
    revision["additional_sources"].update({str(path): {"sha256": sha256(path)} for path in additional})
    revision["checkpoint_compatibility"] = {
        "patient_and_feature_results": "Extraction implementation and configuration unchanged; existing fingerprints remain enforced.",
        "statistical_selection": "Evidence, exact training frame, splits, and model configuration unchanged.",
        "role_decisions": "No role response existed before failure; effective batch policy is included in the existing adjudication fingerprint.",
        "completed_fallback_results": "Retained with their existing failure audits.",
        "checkpoint_inventory": str(inventory), "oracle_values_opened": False,
        "comparison_admission_rules_changed": False,
    }
    immutable_json(manifest_path, revision)
    validate_run_inputs(OUT, manifest_path)
    print({"manifest": str(manifest_path), "sha256": sha256(manifest_path),
           "patient_markers": len(markers), "aggregate_files": len(aggregate_files)}, flush=True)


if __name__ == "__main__":
    main()
