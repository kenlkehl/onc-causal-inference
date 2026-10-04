#!/usr/bin/env python3
"""Exercise the real Stage 2 decision backend; gold values never enter prompts.

Requires a Plumb vLLM classifier and the existing continuous feasibility fixture.
Uses the actual ColBERT retriever on CPU by default, leaving GPU allocation to
the separately managed vLLM server. Results include acceptance and error rates.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8134/v1")
    parser.add_argument("--model", default="plumb-4b")
    parser.add_argument("--tokenizer", default=str(ROOT / "artifacts/plumb_feasibility/model"))
    parser.add_argument("--colbert-model", default="lightonai/GTE-ModernColBERT-v1")
    parser.add_argument("--questions", type=Path, default=ROOT / "experiments/plumb_feasibility/continuous_questions.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/plumb_feasibility/stage2_decision_results")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--verification-min-probability", type=float, default=0.8)
    args = parser.parse_args()
    import torch
    import pandas as pd
    from oci.colbert_config import ColBERTConfig
    from oci.inference.plain_handoff_stage2_analysis import extract_rows
    from oci.inference.stage2_decision_client import VLLMDecisionClient
    from oci.inference.stage2_decision_config import DecisionExtractionConfig
    from oci.inference.stage2_endpoint_pool import ExtractionEndpoint

    torch.set_num_threads(4)
    policy = DecisionExtractionConfig(enabled=True, tokenizer_name=args.tokenizer, tokenizer_revision="",
                                     verification_min_probability=args.verification_min_probability)
    client = VLLMDecisionClient(policy=policy, model=args.model,
        endpoints=(ExtractionEndpoint(args.endpoint, args.workers),), workers=args.workers)
    retrieval = ColBERTConfig(devices=("cpu",), model_name=args.colbert_model)
    fixture = json.loads(args.questions.read_text())
    # The measurement dataset contains text only; gold stays in the scoring code.
    dataset = pd.DataFrame({"text": [case["state"] for case in fixture["cases"]]})
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("Decision extraction attempted a generative LLM request")

    for field_name, field in fixture["fields"].items():
        definition = {"name": field_name, "feature_id": field_name, "description": field["target"],
            "value_type": "continuous", "categories_or_unit": [field["unit"]],
            "measurement_definition": "Extract " + field["target"] + ". Use the stated time scope; convert units if needed.",
            "missing_value_rule": "Missing if not documented, ambiguous, or only a threshold is stated.",
            "decision_ontology": {"minimum": float(field["low"]), "maximum": float(field["high"])}}
        row_ids = [i for i, case in enumerate(fixture["cases"]) if case["field"] == field_name]
        extract_rows(dataset=dataset, row_ids=row_ids, text_column="text", definitions=[definition],
            output_dir=args.output / field_name, request_json=forbidden, workers=args.workers,
            max_prompt_chars=100000, context_strategy="colbert", colbert=retrieval,
            decision_extraction=policy, decision_client=client,
            request_identity={"model": args.model})
        print(f"Completed {field_name}: {len(row_ids)} cases", flush=True)
    scored = []
    for path in sorted(args.output.glob("*/decisions/row_*/*/result.json")):
        result = json.loads(path.read_text())
        case = fixture["cases"][result["row_id"]]
        gold = float(case["gold"]) if case.get("gold") is not None else None
        candidate = result.get("candidate")
        within = (abs(candidate-gold) <= policy.verification_relative_tolerance*abs(gold)
                  if gold is not None and candidate is not None else None)
        scored.append({"id": case["id"], "group": case["group"], "expected_status": case["expected_status"],
            "gold": gold, "status": result["status"], "value": result["value"], "candidate": candidate,
            "candidate_within_5_percent": within, "max_prompt_tokens": max((c["prompt_tokens"] for c in result["calls"]), default=0),
            "verification_probability": (result["calls"][-1]["probabilities"].get("true") if candidate is not None else None),
            "artifact": str(path)})
    accepted = [r for r in scored if r["status"] == "accepted"]
    checked = [r for r in scored if r["candidate"] is not None]
    summary = {"cases": len(scored), "status_counts": dict(Counter(r["status"] for r in scored)),
        "numeric_candidates_verified": len(checked), "accepted": len(accepted),
        "accepted_within_5_percent": sum(r["candidate_within_5_percent"] is True for r in accepted),
        "false_acceptances": sum(r["candidate_within_5_percent"] is not True for r in accepted),
        "false_rejections": sum(r["candidate_within_5_percent"] is True and r["status"] != "accepted" for r in checked),
        "max_prompt_tokens": max(r["max_prompt_tokens"] for r in scored),
        "elapsed_seconds": time.monotonic()-started, "decision_policy": policy.public_dict(),
        "colbert": retrieval.measurement_identity(), "model": args.model,
        "unit_conversion_cases": [r for r in scored if r["id"].startswith("conversion_")]}
    (args.output / "scored.json").write_text(json.dumps(scored, indent=2, allow_nan=False)+"\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
