"""Helpers for the frozen-evidence cross-fold concept experiment."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import threading

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
PRIOR = ROOT / "reports/2026-09-22/fold_1_overlap_architecture_search_2026-09-22"
INPUTS = ROOT / "reports/2026-09-21/fold_1_extracted_logistic_interactions_2026-09-21/inputs"
BASE = ROOT / "reports/2026-09-21/fold_1_multi_model_selection_2026-09-21"
SOURCE_CONFIG = ROOT / "reports/2026-09-19/selection_comparison_2026-09-19/results/inputs/refresh_config.json"
DATE = "2026-09-23"
N = 100
MODEL = "gemma4-31b"
ENDPOINT = "http://sn4622130540:8000/v1"


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(files):
    for path, expected in files.items():
        if sha(path) != expected:
            raise ValueError(f"Frozen file changed: {path}")


def runtime():
    from oci.inference.plain_handoff_stage2 import PlainHandoffStage2, plain_stage2_config_from_mapping

    config = read(SOURCE_CONFIG)
    config.update(endpoint=ENDPOINT, model=MODEL, api_key="EMPTY", extraction_llm=None,
                  vllm=None, workers=15, max_tokens=100000,
                  interpretation_reasoning_effort="high", max_prompt_chars=900000)
    config["selection_consolidation"]["enabled"] = False
    service = PlainHandoffStage2(
        config=plain_stage2_config_from_mapping(config, default_workers=15), clinical_question="")
    service._check_and_record_model_identity(HERE / "llm_runtime")
    write(HERE / "llm_runtime/config.json", {"model": service.model_identity,
                                           "config": service.config.public_dict()})
    return service


def request(service, *, messages, validate, audit_dir):
    from oci.inference.plain_handoff_stage2 import _request_json
    from oci.inference import stage2_request_audit

    with stage2_request_audit.context(_audit_path=str(Path(audit_dir) / "request_events.jsonl"),
                                     experiment="cross_fold_modifier_concepts"):
        return _request_json(messages=messages, config=service.config,
                             completion=service.completion, validate=validate,
                             request_kind="interpretation")
