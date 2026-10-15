"""Resumable API generation using an existing dataset's frozen causal process."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict, replace
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import logging
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd

from .llm_client import LLMClient
from . import generator as g
from .prompts import CLINICAL_SYSTEM_PROMPT, NOTE_EXPANSION_PROMPT, build_event_timeline_prompt
from .structured_data import convert_structured_event_to_text

LOGGER = logging.getLogger(__name__)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def _read_checkpoint(path, identity):
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        if payload["identity"] == identity and payload["sha256"] == _digest(payload["value"]):
            return payload["value"]
    except (ValueError, KeyError, TypeError):
        pass
    raise ValueError(f"Invalid or incompatible generation checkpoint: {path}")


def _checkpoint(path, identity, value):
    _write_json(path, {"identity": identity, "value": value, "sha256": _digest(value)})


NOTE_OUTPUT_FORMAT = (
    "\nFor this clinical-document request, return only the document as readable text, "
    "with clinical section headings where appropriate. Do not return JSON, dictionaries, "
    "XML wrappers, or fenced code blocks."
)


def _request(client, path, identity, prompt, temperature, max_tokens, validate,
             system_prompt=CLINICAL_SYSTEM_PROMPT):
    request = dict(prompt=prompt, system_prompt=system_prompt,
                   temperature=temperature, max_tokens=max_tokens)
    request_id = _digest({"generation": identity, "request": request})
    cached = _read_checkpoint(path, request_id)
    if cached is not None:
        validate(cached)
        return cached
    for attempt in range(3):
        try:
            text = client.generate(**request)
            validate(text)
            _checkpoint(path, request_id, text)
            return text
        except Exception:
            if attempt == 2:
                raise
            LOGGER.warning("Retrying %s (attempt %d)", path, attempt + 2, exc_info=True)
            time.sleep(2 ** attempt)


def _nonempty(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Model returned empty text")


def _clinical_document(text):
    _nonempty(text)
    # Fail visibly rather than publish JSON escapes as a clinical narrative.
    if text.lstrip().startswith(("{", "[", "```")):
        raise ValueError("Expected a clinical document, received a structured/code wrapper")


def _patient(base, config, client, output, identity):
    patient_id = int(base["patient_id"])
    patient_path = output / "patients" / f"patient_{patient_id:06d}.json"
    patient_identity = _digest({"generation": identity, "scaffold": base})
    cached = _read_checkpoint(patient_path, patient_identity)
    if cached is not None:
        return cached

    requests = output / "requests" / f"patient_{patient_id:06d}"
    note_types = set(config.note_types_to_expand)
    structured_types = g._enabled_structured_event_types(config)

    def validate_timeline(text):
        _nonempty(text)
        events = g._parse_event_timeline(text)
        if not events or not any(event["event_type"] in note_types for event in events):
            raise ValueError("Timeline has no parseable narrative events")

    timeline_prompt = build_event_timeline_prompt(config.structured_data).format(
        patient_characteristics=base["patient_prompt"], clinical_question=config.clinical_question,
        min_events=config.min_events_per_patient, max_events=config.max_events_per_patient,
    )
    max_tokens = max(1000, config.llm.max_tokens - 2000)
    timeline = _request(client, requests / "timeline.json", identity, timeline_prompt,
                        0.7, max_tokens, validate_timeline)
    events = g._parse_event_timeline(timeline)
    blocks = []
    for index, event in enumerate(events):
        event_type = event["event_type"]
        if event_type in note_types:
            prompt = NOTE_EXPANSION_PROMPT.format(
                note_type=g._event_type_to_note_type(event_type),
                clinical_question=config.clinical_question,
                masked_event_timeline=g._create_masked_timeline(events, index),
            )
            text = _request(client, requests / f"note_{index:03d}.json", identity,
                            prompt, 0.5, max_tokens, _clinical_document,
                            system_prompt=CLINICAL_SYSTEM_PROMPT + NOTE_OUTPUT_FORMAT)
            # Independent RNG makes text assembly invariant to worker count/resume.
            rng = random.Random(f"{config.seed}:{patient_id}:{index}")
            if rng.random() < config.drug_perturbation_prob:
                text = g._apply_drug_perturbation(text, rng=rng)
            blocks.append(text)
        elif event_type in structured_types:
            text = convert_structured_event_to_text(event_type, event["event_text"])
            if text:
                blocks.append(text)
    row = dict(base, event_timeline=timeline, clinical_text=config.note_separator.join(blocks),
               num_notes=len(blocks))
    _nonempty(row["clinical_text"])
    _checkpoint(patient_path, patient_identity, row)
    return row


def generate_from_saved_dgp(config, metadata_path, num_workers=4, show_progress=True):
    """Keep saved equations/distributions, sample fresh patients, then generate text.

    Each successful timeline, note and completed patient is checkpointed. Failed
    or truncated responses cannot silently become empty/partial patient records.
    Worker count and endpoint placement can change on resume; science/model/prompt
    changes require a different output directory.
    """
    config.validate()
    if num_workers < 1:
        raise ValueError("num_workers must be positive")
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".generation.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _generate_locked(config, Path(metadata_path), num_workers, output)


def _generate_locked(config, metadata_path, num_workers, output):
    features, treatment_eq, outcome_eq, statistics, source_config = g._load_dgp_metadata(str(metadata_path))
    source = json.loads(metadata_path.read_text())
    public_config = asdict(config)
    public_config["llm"]["api_key"] = ""
    # Source configs saved before continuous_covariate_distribution existed
    # were generated with Normal draws, so an absent value means "normal".
    source_config = {"continuous_covariate_distribution": "normal", **source_config}
    # These are the intended changes when expanding a saved dataset.
    for field, value in source_config.items():
        if field not in {"dataset_size", "output_dir", "llm", "seed"} and public_config.get(field) != value:
            raise ValueError(f"Saved-DGP configuration changed {field!r}; retain the source generation settings")
    scientific_config = {key: value for key, value in public_config.items() if key not in {"output_dir", "llm"}}
    scientific_config["llm"] = {key: value for key, value in public_config["llm"].items()
                                 if key not in {"api_key", "api_base_url", "api_base_urls", "timeout", "reject_incomplete"}}
    source_hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                     for name in ("saved_dgp.py", "generator.py", "prompts.py", "structured_data.py", "llm_client.py")}
    identity = _digest({"config": scientific_config, "source": source, "code": source_hashes})
    manifest_path = output / "generation_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["identity"] != identity:
        raise ValueError("Output directory belongs to a different generation configuration/DGP/code")
    _write_json(manifest_path, {"identity": identity, "source_metadata": str(metadata_path.resolve()),
                               "source_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                               "source_hashes": source_hashes, "config": public_config,
                               "num_workers": num_workers})
    random.seed(config.seed)
    np.random.seed(config.seed)
    scaffold_path = output / "patient_scaffold.parquet"
    digest_path = output / "scaffold_sha256.json"
    if scaffold_path.exists():
        digest = hashlib.sha256(scaffold_path.read_bytes()).hexdigest()
        if not digest_path.exists() or json.loads(digest_path.read_text()) != digest:
            raise ValueError("Patient scaffold checksum mismatch")
        scaffold = pd.read_parquet(scaffold_path)
    else:
        scaffold = pd.DataFrame([
            g._build_patient_scaffold_record(i, config, features, statistics, treatment_eq, outcome_eq)
            for i in range(config.dataset_size)
        ])
        temporary = scaffold_path.with_suffix(".tmp.parquet")
        scaffold.to_parquet(temporary, index=False)
        _write_json(digest_path, hashlib.sha256(temporary.read_bytes()).hexdigest())
        temporary.replace(scaffold_path)
    if scaffold["patient_id"].tolist() != list(range(config.dataset_size)):
        raise ValueError("Unexpected patient IDs in scaffold")
    metadata = {"config": public_config, "features": features,
                "confounders": g._filter_features_by_role(features, "confounder"),
                "effect_modifiers": g._filter_features_by_role(features, "effect_modifier"),
                "treatment_equation": treatment_eq, "outcome_equation": outcome_eq,
                "summary_statistics": statistics,
                "dgp_source": {"path": str(metadata_path.resolve()), "sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest()},
                "generation_status": "running"}
    _write_json(output / "metadata.json", metadata)
    _write_json(output / "generation_config.json", public_config)
    client = LLMClient(replace(config.llm, reject_incomplete=True))
    rows, errors = [], {}
    start = time.monotonic()
    last_report = 0.0

    def report(status):
        _write_json(output / "progress.json", {
            "status": status, "completed_patients": len(rows), "total_patients": config.dataset_size,
            "failed_patients": errors, "elapsed_seconds": round(time.monotonic() - start, 1),
            "updated_at": datetime.now(timezone.utc).isoformat(), "num_workers": num_workers,
        })

    try:
        report("running")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            pending = {executor.submit(_patient, base, config, client, output, identity): base["patient_id"]
                       for base in scaffold.to_dict("records")}
            while pending:
                done, _ = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                for future in done:
                    patient_id = pending.pop(future)
                    try:
                        rows.append(future.result())
                    except Exception as exc:
                        errors[str(patient_id)] = f"{type(exc).__name__}: {exc}"
                        LOGGER.exception("Patient %s failed; successful requests retained for resume", patient_id)
                if time.monotonic() - last_report >= 30 or not pending:
                    report("running")
                    LOGGER.info("Completed %d/%d patients; %d failures", len(rows), config.dataset_size, len(errors))
                    last_report = time.monotonic()
                if len(errors) >= 10:
                    for future in pending:
                        future.cancel()
                    raise RuntimeError("Stopped after ten patient failures; inspect progress.json and resume")
        if errors:
            raise RuntimeError(f"{len(errors)} patients incomplete; resume to retry failed requests")
        df = pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)
        if len(df) != config.dataset_size or not df["clinical_text"].str.strip().ne("").all():
            raise ValueError("Final dataset has missing patients/text")
        metadata["generation_status"] = "complete"
        metadata["dataset_statistics"] = g.oracle_dataset_statistics(df)
        dataset_stats = metadata["dataset_statistics"]
        dataset_stats["generation_mode"] = config.generation_mode
        dataset_stats["clinical_text_stats"] = {"non_empty_count": len(df), "mean_length": float(df.clinical_text.str.len().mean())}
        dataset_stats["two_stage_stats"] = {"mean_notes_per_patient": float(df.num_notes.mean()),
                                           "min_notes_per_patient": int(df.num_notes.min()),
                                           "max_notes_per_patient": int(df.num_notes.max())}
        counts = {kind: [] for kind in g._enabled_structured_event_types(config)}
        for timeline in df.event_timeline:
            events = g._parse_event_timeline(timeline)
            for kind, values in counts.items():
                values.append(sum(event["event_type"] == kind for event in events))
        dataset_stats["structured_data_stats"] = {
            kind: {"total": sum(values), "mean_per_patient": sum(values) / len(df),
                   "min_per_patient": min(values), "max_per_patient": max(values)}
            for kind, values in counts.items()
        }
        temporary = output / "dataset.tmp.parquet"
        df.to_parquet(temporary, index=False)
        temporary.replace(output / "dataset.parquet")
        _write_json(output / "metadata.json", metadata)
        report("complete")
        return df, metadata
    except BaseException:
        report("failed")
        raise
    finally:
        client.close()
