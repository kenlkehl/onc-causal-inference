#!/usr/bin/env python3
"""Evaluate continuous extraction by adaptive, single-pass Plumb choice questions.

This is an isolated experiment, not a Stage 2 backend. Numeric ranges are fixed
by field, independent of each case's gold value. Gold values are used only after
the model has chosen an interval, never to construct its adaptive search path.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, localcontext
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "experiments/plumb_feasibility/continuous_questions.json"
SEED = 20261004
FIELDS = {
    "age": {"unit": "years", "low": "0", "high": "128", "target": "the patient's age in years"},
    "weight": {"unit": "kg", "low": "0", "high": "256", "target": "the patient's pretreatment weight in kg"},
    "tumor": {"unit": "cm", "low": "0", "high": "16", "target": "the tumor diameter in cm"},
    "creatinine": {"unit": "mg/dL", "low": "0", "high": "8", "target": "the documented creatinine value in mg/dL"},
    "temperature": {"unit": "degrees C", "low": "-40", "high": "80", "target": "the sensor temperature in degrees C"},
    "amount": {"unit": "dollars", "low": "0", "high": "1024", "target": "the invoice total in dollars"},
    "percentage": {"unit": "percent", "low": "0", "high": "100", "target": "the documented tumor proportion score in percent"},
    "signed": {"unit": "units", "low": "-64", "high": "64", "target": "the signed instrument reading in units"},
}


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def number(value):
    return format(Decimal(value).normalize(), "f")


def make_fixture():
    rng = random.Random(SEED)
    cases = []

    def add(case_id, field, evidence, gold=None, group="challenge", expected="numeric", target=None):
        cases.append({"id": case_id, "field": field, "group": group, "state": evidence,
                      "target": target or FIELDS[field]["target"],
                      "gold": str(gold) if gold is not None else None,
                      "expected_status": expected})

    templates = {
        "age": (1, 100, 1, "The patient's age is {value} years."),
        "weight": (200, 2500, 10, "Pretreatment weight: {value} kg."),
        "tumor": (1, 1599, 100, "The tumor diameter is {value} cm."),
        "creatinine": (10, 790, 100, "Creatinine: {value} mg/dL."),
        "temperature": (-399, 799, 10, "The sensor temperature is {value} degrees C."),
        "amount": (1, 102399, 100, "The invoice total is {value} dollars."),
        "percentage": (0, 1000, 10, "Tumor proportion score: {value} percent."),
        "signed": (-6399, 6399, 100, "The signed instrument reading is {value} units."),
    }
    for field, (low, high, scale, template) in templates.items():
        for index in range(6):
            value = number(Decimal(rng.randint(low, high)) / scale)
            add(f"random_{field}_{index + 1}", field, template.format(value=value), value, "seeded_direct")
    for age in (16, 64, 128):
        add(f"boundary_age_{age}", "age", f"The patient's recorded age is {age} years.", age)
    for weight in ("64", "72", "72.6"):
        add(f"boundary_weight_{weight.replace('.', '_')}", "weight", f"Pretreatment weight: {weight} kg.", weight)
    for value in ("1.375", "1.24"):
        add(f"decimal_creatinine_{value.replace('.', '_')}", "creatinine", f"Creatinine: {value} mg/dL.", value)
    for value in ("0", "1.99", "2"):
        add(f"boundary_tumor_{value.replace('.', '_')}", "tumor", f"The tumor diameter is explicitly recorded as {value} cm.", value)
    add("conversion_mm_to_cm", "tumor", "The tumor diameter is 27 mm.", "2.7",
        target="the tumor diameter in cm; use exactly 10 mm = 1 cm")
    add("conversion_g_to_kg", "weight", "Pretreatment weight: 72500 grams.", "72.5",
        target="the pretreatment weight in kg; use exactly 1000 grams = 1 kg")
    add("conversion_lb_to_kg", "weight", "Pretreatment weight: 180 pounds.", "81.6466266",
        target="the pretreatment weight in kg; use exactly 1 pound = 0.45359237 kg")
    add("pretreatment_not_followup", "weight", "Before treatment the patient weighed 72 kg. After treatment the weight was 68 kg.", "72")
    add("distractor_height", "weight", "Pretreatment height: 168 cm. Pretreatment weight: 72.4 kg.", "72.4")
    add("corrected_tumor", "tumor", "Correction: the tumor diameter is 2.4 cm, not 3.2 cm. The 3.2 cm entry was a transcription error.", "2.4")
    add("patient_not_father", "age", "The patient's father is 82 years old. The patient is 61 years old.", "61")
    add("spelled_decimal", "creatinine", "Creatinine was recorded as one point two five mg/dL.", "1.25")
    add("negative_decimal", "signed", "The signed instrument reading is -0.125 units.", "-0.125")
    add("missing_weight", "weight", "Pretreatment weight was not recorded.", expected="unresolved")
    add("family_only_weight", "weight", "The patient's father weighs 82 kg. The patient's own weight is not documented.", expected="unresolved")
    add("inequality_tumor", "tumor", "Tumor diameter is less than 2 cm; no exact measurement is provided.", expected="unresolved")
    add("interval_weight", "weight", "Pretreatment weight is reported only as a range of 70 to 74 kg; no exact value is available.", expected="unresolved")
    add("conflicting_weight", "weight", "Two equally authoritative entries for the same pretreatment measurement give 71 kg and 75 kg. Neither is corrected or preferred.", expected="unresolved")
    add("qualitative_creatinine", "creatinine", "Creatinine is described as normal. No numerical value is documented.", expected="unresolved")
    add("outside_creatinine", "creatinine", "Creatinine: 12.4 mg/dL.", "12.4", expected="outside")
    add("outside_weight", "weight", "Pretreatment weight: 300 kg.", "300", expected="outside")
    return {"seed": SEED, "fields": FIELDS, "cases": cases}


def intervals(low, high, count, upper_inclusive):
    """Use exactly the displayed decimal edges for both inference and evaluation."""
    with localcontext() as context:
        context.prec = 40
        edges = [low] + [
            (low + (high - low) * index / count).quantize(Decimal("0.00000001"))
            for index in range(1, count)
        ] + [high]
    if any(a >= b for a, b in zip(edges, edges[1:])):
        raise ValueError("Requested intervals exceed the displayed numeric precision")
    return [{"low": number(edges[i]), "high": number(edges[i + 1]),
             "upper_inclusive": upper_inclusive and i == count - 1}
            for i in range(count)]


def contains(interval, value):
    if value is None:
        return False
    x, low, high = Decimal(value), Decimal(interval["low"]), Decimal(interval["high"])
    return low <= x and (x <= high if interval["upper_inclusive"] else x < high)


def question_for(target, unit, bins, guarded, reverse=False, style="verbal"):
    criteria = {}
    for index, interval in enumerate(bins):
        end = "less than or equal to" if interval["upper_inclusive"] else "less than"
        if style == "compact":
            op = "<=" if interval["upper_inclusive"] else "<"
            criteria[f"range_{index + 1}"] = (
                f"{interval['low']} <= x {op} {interval['high']}; x is in {unit}."
            )
        else:
            criteria[f"range_{index + 1}"] = (
                f"Value is greater than or equal to {interval['low']} {unit} "
                f"and {end} {interval['high']} {unit}."
            )
    if guarded:
        criteria["unresolved"] = (
            "No unique exact value is available: not documented, only a bound or range, "
            "or conflicting values without a resolved choice. Do not invent a number."
        )
        criteria["outside"] = "A unique exact value is documented but is outside all the listed numeric ranges."
    if reverse:
        criteria = dict(reversed(list(criteria.items())))
    return {"type": "choice", "instructions": (
        f"Extract {target}. Select the range containing that value. "
        "Use only the supplied evidence and any conversion rule in this question. "
        "Do not round the value before comparing it with range boundaries. "
        "An exact boundary belongs to the range whose lower endpoint equals that value. "
        "Honor the stated endpoint inclusivity."
    ), "criteria": criteria}


def summarize(rows, calls, depth):
    result = {}
    for strategy in sorted({row["strategy"] for row in rows}):
        subset = [r for r in rows if r["strategy"] == strategy]
        numeric = [r for r in subset if r["case"]["expected_status"] == "numeric"]
        by_pass = []
        for step in range(depth):
            returned = [r["steps"][step] for r in numeric if len(r["steps"]) > step and r["steps"][step]["interval"] is not None]
            by_pass.append({
                "pass": step + 1, "eligible_numeric_cases": len(numeric),
                "numeric_estimates_returned": len(returned),
                "interval_contains_gold": sum(s["contains_gold"] for s in returned),
                "normalized_mae_among_returned": statistics.mean(s["normalized_absolute_error"] for s in returned) if returned else None,
                "max_normalized_error_among_returned": max((s["normalized_absolute_error"] for s in returned), default=None),
            })
        per_field = {}
        for field in sorted({r["case"]["field"] for r in numeric}):
            available = [r["steps"][-1] for r in numeric if r["case"]["field"] == field and len(r["steps"]) == depth and r["status"] == "numeric"]
            per_field[field] = {"unit": FIELDS[field]["unit"], "returned": len(available),
                                "mae": statistics.mean(s["absolute_error"] for s in available) if available else None,
                                "max_error": max((s["absolute_error"] for s in available), default=None)}
        special = [r for r in subset if r["case"]["expected_status"] != "numeric"]
        numeric_step_latencies = [s["latency_ms"] for r in numeric for s in r["steps"]]
        result[strategy] = {
            "n": len(subset), "numeric_cases": len(numeric), "by_pass": by_pass,
            "special_cases": {"n": len(special), "correct_status": sum(r["status"] == r["case"]["expected_status"] for r in special)},
            "numeric_cases_stopped_early": sum(r["status"] != "numeric" for r in numeric),
            "latency_ms": {"median_single_call": statistics.median(numeric_step_latencies),
                           "median_whole_numeric_case": statistics.median(sum(s["latency_ms"] for s in r["steps"]) for r in numeric)},
            "absolute_error_by_field": per_field,
            "failure_ids": [r["case"]["id"] for r in numeric if r["status"] != "numeric" or not r["steps"][-1]["contains_gold"]],
        }
    return {"strategies": result, "inference_calls": len(calls)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate-fixture", action="store_true")
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts/plumb_feasibility/model")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/plumb_feasibility/continuous_results")
    parser.add_argument("--gpu", default="GPU-184bbef2-879b-de2e-73de-45fca36cafb5")
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--interval-style", choices=("verbal", "compact"), default="verbal")
    args = parser.parse_args()
    if args.generate_fixture:
        if args.fixture.exists():
            raise FileExistsError("Refusing to overwrite the frozen fixture")
        dump(args.fixture, make_fixture())
        print(f"Saved fixture to {args.fixture}")
        return
    if not 1 <= args.passes <= 5:
        parser.error("--passes must be between 1 and 5")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/plumb-triton-cache")
    from jevk5 import JevK5, decision_options
    import torch

    fixture = json.loads(args.fixture.read_text())
    cases = fixture["cases"]
    assert len({c["id"] for c in cases}) == len(cases)
    for case in cases:
        field = fixture["fields"][case["field"]]
        if case["expected_status"] == "numeric":
            assert Decimal(field["low"]) <= Decimal(case["gold"]) <= Decimal(field["high"])
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "results.jsonl").exists():
        raise FileExistsError("Use a new output directory rather than overwriting an experiment")
    dump(args.output / "fixture.json", fixture)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible GPU")
    free_before, _ = torch.cuda.mem_get_info()
    if free_before < 12 * 1024**3:
        raise RuntimeError("Not enough free GPU memory")
    torch.set_num_threads(4)
    print("Loading Plumb on GPU 1", flush=True)
    started = time.perf_counter()
    model = JevK5(str(args.model), device="cuda:0", graphs=False)
    load_seconds = time.perf_counter() - started
    # One representative prompt of each strategy; later new-shape compilation
    # may still be counted. Save all timings and disclose this in the report.
    for guarded, count in ((False, 8), (True, 6)):
        q = question_for("the parcel weight in kg", "kg", intervals(Decimal(0), Decimal(256), count, True), guarded, style=args.interval_style)
        for _ in range(3):
            model.decide("The parcel weighs 53.2 kg.", q)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    print("Model ready; running frozen numeric fixture", flush=True)
    rows, calls = [], []
    with (args.output / "results.jsonl").open("w") as output:
        for strategy, count, guarded in (("eight_numeric_bins", 8, False), ("six_bins_with_abstention", 6, True)):
            for case in cases:
                if not guarded and case["expected_status"] != "numeric":
                    continue
                field = fixture["fields"][case["field"]]
                low, high = Decimal(field["low"]), Decimal(field["high"])
                global_width = high - low
                upper_inclusive = True
                steps = []
                status = "numeric"
                for step in range(args.passes):
                    bins = intervals(low, high, count, upper_inclusive)
                    question = question_for(case["target"], field["unit"], bins, guarded, style=args.interval_style)
                    options = decision_options(question)
                    assert len(options) <= 8
                    tokens = len(model.encode(case["state"], question["instructions"], [v for _, v in options]))
                    if tokens > 2048:
                        raise ValueError("Short-prompt test exceeded its token limit; no truncation allowed")
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    answer = model.decide(case["state"], question)
                    torch.cuda.synchronize()
                    latency = (time.perf_counter() - t0) * 1000
                    probs = answer["probabilities"]
                    assert set(probs) == set(question["criteria"])
                    assert all(math.isfinite(p) and 0 <= p <= 1 for p in probs.values())
                    assert abs(sum(probs.values()) - 1) < 1e-5
                    chosen = answer["choice"]
                    selected = bins[int(chosen.split("_")[1]) - 1] if chosen.startswith("range_") else None
                    entry = {"pass": step + 1, "question": question, "answer": answer,
                             "interval": selected, "latency_ms": latency}
                    if selected is None:
                        status = chosen
                    else:
                        low, high = Decimal(selected["low"]), Decimal(selected["high"])
                        upper_inclusive = selected["upper_inclusive"]
                        midpoint = (low + high) / 2
                        entry.update(midpoint=number(midpoint), width=number(high - low),
                                     contains_gold=contains(selected, case["gold"]))
                        if case["gold"] is not None:
                            error = abs(midpoint - Decimal(case["gold"]))
                            entry.update(absolute_error=float(error), normalized_absolute_error=float(error / global_width))
                    steps.append(entry)
                    calls.append({"strategy": strategy, "case_id": case["id"], **entry})
                    if selected is None:
                        break
                row = {"strategy": strategy, "case": case, "steps": steps, "status": status}
                rows.append(row)
                output.write(json.dumps(row, allow_nan=False) + "\n")
                output.flush()
                print(json.dumps({"strategy": strategy, "id": case["id"], "gold": case["gold"],
                                  "path": [s["answer"]["choice"] for s in steps],
                                  "midpoint": steps[-1].get("midpoint"), "status": status,
                                  "contains_gold": steps[-1].get("contains_gold")}), flush=True)
    summary = summarize(rows, calls, args.passes)
    summary.update({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": json.loads((args.model.parent / "model_revision.json").read_text()),
        "runtime_source_sha256": hashlib.sha256(Path(inspect.getfile(JevK5)).read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(), "python_executable": sys.executable,
        "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "jevk5", "flash-linear-attention")},
        "gpu": {"selector": args.gpu, "name": torch.cuda.get_device_name(),
                "free_before_gib": free_before / 1024**3,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3},
        "settings": {"passes": args.passes, "temperature": model.temperature,
                     "dtype": "bfloat16", "graphs": False, "generated_tokens": 0,
                     "interval_style": args.interval_style,
                     "intervals": "lower-inclusive, upper-exclusive except global upper endpoint",
                     "point_estimate": "selected interval midpoint", "ascending_option_order": True},
        "load_seconds": load_seconds,
        "limitations": ["Synthetic short notes, not clinical validation.",
                        "Greedy choice at each pass; no gold-assisted corrections, retries, or beam search.",
                        "Midpoint intervals are search bins, not statistical confidence intervals.",
                        "Point-error aggregates across fields are normalized by each initial domain width.",
                        "Shared GPU; three warm-up calls per strategy; new-shape compilation may affect timing."]})
    dump(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
