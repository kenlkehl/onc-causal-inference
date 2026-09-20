"""Freeze the structural-repair policy change without rewriting earlier revisions."""
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from compare import HERE, ROOT, immutable_json, now, read_json, sha256, validate_run_inputs

OUT = HERE / "results"
PRIOR = OUT / "revisions/reliability_v2"
REVISION = OUT / "revisions/structural_reasoning_v3"


def main():
    manifest_path = REVISION / "manifest.json"
    if manifest_path.exists():
        raise RuntimeError("Revision already frozen; validate it rather than rewriting it")
    prior_path = PRIOR / "manifest.json"
    prior = read_json(prior_path)
    stop = read_json(REVISION / "stop_prior.json")
    if stop["prior_manifest_sha256"] != sha256(prior_path):
        raise ValueError("Prior revision manifest changed after stopping its worker")
    if stop["prior_runner_final_state"]["returncode"] != -15:
        raise ValueError("Prior worker did not record the deliberate stop")
    before = read_json(REVISION / "before_sources.json")
    source = str(ROOT / "oci/inference/plain_handoff_stage2.py")
    old_source = before["files"][source]
    if sha256(old_source["archive"]) != prior["source_overrides"][source]["after_sha256"]:
        raise ValueError("Prior extraction source archive does not match its frozen revision")
    if sha256(source) == old_source["sha256"]:
        raise ValueError("Structural-repair policy source has not changed")

    def completed_patient(path):
        marker = path / "complete.json"
        return (str(marker), {"sha256": sha256(marker)}) if marker.is_file() else None

    checkpoint_files = {}
    rounds = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        for batches in sorted((OUT / "refresh").glob(
            "outer_*/ontology_supervision/round_*/extraction/batches"
        )):
            patients = list(batches.glob("batch_*"))
            completed = [item for item in pool.map(completed_patient, patients) if item is not None]
            checkpoint_files.update(completed)
            rounds.append({"path": str(batches), "patients_started": len(patients),
                           "patients_complete": len(completed)})
    inventory = REVISION / "patient_checkpoints_before.json"
    immutable_json(inventory, {"recorded_at": now(), "rounds": rounds, "files": checkpoint_files,
                               "scope": "Completed patient markers; compatible feature/chunk checkpoints also remain reusable."})

    config_path = REVISION / "refresh_config.json"
    immutable_json(config_path, read_json(prior["config"]["path"]))
    if sha256(config_path) != prior["config"]["sha256"]:
        raise ValueError("Runtime configuration must remain byte-identical to reliability_v2")
    revision = copy.deepcopy(prior)
    revision.update(
        revision="structural_reasoning_v3", recorded_at=now(),
        authorization="User requested reasoning for structural extraction errors too.",
        prior_run_stop=stop,
        parent_revision={"path": str(prior_path), "sha256": sha256(prior_path)},
        policy_change={"before": "Three structural extraction errors exempted from reasoning escalation.",
                       "after": "All validation errors enable at least high reasoning after five repair attempts.",
                       "reasoning_output_ceiling": 32768, "max_response_repairs": 15,
                       "logical_deadline_seconds": 7200},
        config={"path": str(config_path), "sha256": sha256(config_path)},
    )
    revision["source_overrides"][source]["after_sha256"] = sha256(source)
    # Keep the original-to-current guard and authenticate the intervening revision
    # and its archived source through the existing frozen-input validation.
    frozen = [prior_path, Path(prior["config"]["path"]), inventory,
              REVISION / "before_sources.json", REVISION / "stop_prior.json"]
    frozen.extend(Path(record["archive"]) for record in before["files"].values())
    revision["frozen_inputs"].update({str(path): {"sha256": sha256(path)} for path in frozen})
    additional = [Path(__file__).resolve(), HERE / "resume_structural_reasoning.py"]
    revision["additional_sources"].update({str(path): {"sha256": sha256(path)} for path in additional})
    revision["checkpoint_compatibility"] = {
        "patient_and_feature_results": "Normal model/text/definition fingerprint validation remains enforced.",
        "incomplete_serial_chunks": "Configuration and output reservations are unchanged; compatible chunk plans remain reusable.",
        "completed_fallback_results": "Retained with their existing failure audits; no retrospective measurement replacement.",
        "patient_checkpoint_inventory": str(inventory),
        "source_scientific_artifacts_modified": False,
        "oracle_values_opened": False,
        "comparison_policy_changed": False,
    }
    immutable_json(manifest_path, revision)
    validate_run_inputs(OUT, manifest_path)
    print({"manifest": str(manifest_path), "sha256": sha256(manifest_path),
           "completed_patient_checkpoints": len(checkpoint_files), "rounds": rounds}, flush=True)


if __name__ == "__main__":
    main()
