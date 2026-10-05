"""One-feature, ColBERT-retrieved, auditable decision-model measurements."""

from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import math
from pathlib import Path

from .stage2_decision_client import decision_messages

SCHEMA = "stage2-plumb-decision-v2-grouped-categories"
MISSING = "__not_documented__"
NONE = "__none_of_above__"


def validate_ontology(feature):
    """Only closed categories or a bounded numeric domain reach the classifier."""
    kind = feature.get("value_type")
    values = feature.get("categories_or_unit")
    if not isinstance(values, (list, tuple)) or any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError("decision features require categories_or_unit strings")
    if kind in {"binary", "categorical", "ordinal"}:
        if not values or len(set(values)) != len(values):
            raise ValueError("decision categorical ontologies require distinct nonempty categories")
        if kind == "binary" and len(values) != 2:
            raise ValueError("decision binary ontologies require exactly two categories")
    elif kind == "continuous":
        if len(values) != 1:
            raise ValueError("numeric decision features require one canonical unit (or 'unitless')")
        ontology = feature.get("decision_ontology")
        if not isinstance(ontology, dict) or set(ontology) != {"minimum", "maximum"}:
            raise ValueError("numeric decision_ontology requires minimum and maximum")
        lo, hi = ontology["minimum"], ontology["maximum"]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (lo, hi)):
            raise ValueError("numeric decision bounds must be finite numbers")
        if not lo < hi or not math.isfinite(hi - lo):
            raise ValueError("numeric decision minimum must be below maximum with finite width")
    else:
        raise ValueError("decision extraction requires binary, categorical, ordinal, or continuous ontologies")
    return feature


def policy_identity(policy, colbert):
    from ..extraction.colbert import retrieval_identity

    return {"method": "plumb_decision", "schema": SCHEMA, "decision": policy.public_dict(),
            "retrieval": retrieval_identity(colbert), "readout": "LAST/raw_A_to_P_logits",
            "option_order": "declared_categories_or_ascending_bins_then_exits",
            "categorical_readout": "recursive_declared_order_groups_at_most_14_plus_exits"}


def claim_output_method(output_dir, policy, colbert, *, fold_root=False):
    from .plain_handoff_stage2_analysis import _write_json

    path = Path(output_dir) / "measurement_method.json"
    identity = policy_identity(policy, colbert)
    if path.is_file():
        if json.loads(path.read_text()) != identity:
            raise ValueError("Decision measurement policy changed; use a fresh Stage 2 output directory")
    elif any((Path(output_dir) / p).exists() for p in (
        ("ontology_supervision", "extraction", "preselection") if fold_root
        else ("colbert", "batches", "pages", "extracted.csv", "decisions")
    )):
        raise ValueError("Existing measurements have no compatible decision policy; use a fresh output directory")
    _write_json(path, identity)
    return identity


def numeric_options(lo, hi, bins, *, upper_inclusive):
    edges = [float(format(lo + (hi - lo) * i / bins, ".12g")) for i in range(bins + 1)]
    edges[0], edges[-1] = lo, hi
    if not all(a < b for a, b in zip(edges, edges[1:])):
        raise ValueError("numeric range is too narrow to subdivide reliably")
    options = []
    for i, (left, right) in enumerate(zip(edges, edges[1:])):
        right_operator = "<=" if i == bins - 1 and upper_inclusive else "<"
        key = f"bin_{i + 1}"
        options.append((key, f"{key}: {left:.12g} <= value {right_operator} {right:.12g}"))
    options.extend([(MISSING, "not_documented: No unambiguous numeric value is documented in the supplied excerpts."),
                    (NONE, "none_of_above: A numeric value is documented but lies outside ALL the listed ranges.")])
    return edges, options


def _contract(feature):
    from .plain_handoff_stage2_analysis import _prompt_feature_definitions

    contract = _prompt_feature_definitions([feature])[0]
    contract.pop("accepted_representations", None)
    return "Feature contract: " + json.dumps(contract, ensure_ascii=False, separators=(",", ":"))


def packed_decision(client, *, source, evidence, criterion, options):
    """Drop whole low-ranked retrieval chunks; never truncate a prompt or chunk."""
    from ..extraction.colbert import render_context

    ranked = [hit for group in evidence["hits"] for hit in group]
    # Retrieval is for exactly one feature. Source order is restored by render_context.
    selected = []
    empty_tokens = len(client.encode(decision_messages("", criterion, options)))
    if empty_tokens > client.policy.max_prompt_tokens:
        raise ValueError("Feature ontology alone exceeds the decision prompt token budget")
    for hit in ranked:
        candidate = render_context(source, [*selected, hit])
        if len(client.encode(decision_messages(candidate, criterion, options))) > client.policy.max_prompt_tokens:
            break
        selected.append(hit)
    if ranked and not selected:
        raise ValueError("No complete retrieved chunk fits the decision prompt; shorten the feature contract")
    context = render_context(source, selected) if selected else ""
    result = client.decide(context, criterion, options)
    result["retrieval_budget"] = {"selected_chunk_indices": [h["chunk_index"] for h in selected],
        "omitted_chunk_indices": [h["chunk_index"] for h in ranked[len(selected):]],
        "max_prompt_tokens": client.policy.max_prompt_tokens}
    return result


def measure_feature(*, feature, source, evidence, client):
    validate_ontology(feature)
    calls = []

    def decide(criterion, options, phase):
        result = packed_decision(client, source=source, evidence=evidence,
                                 criterion=criterion, options=options)
        calls.append({"phase": phase, **result})
        return result

    if not source.strip():
        return {"value": None, "status": "not_documented", "calls": calls}
    contract = _contract(feature)
    if feature["value_type"] != "continuous":
        values = list(feature["categories_or_unit"])
        offset, group_level = 0, 0
        exits = [(MISSING, "not_documented: The excerpts do not establish the feature's value; missing or unresolved evidence."),
                 (NONE, "none_of_above: The excerpts establish a value, but NONE of the listed categories represents it.")]
        while len(values) > 14:
            group_level += 1
            group_size = math.ceil(len(values) / 14)
            groups = [values[i:i + group_size] for i in range(0, len(values), group_size)]
            options = [(f"group_{i}", "The documented category is one of: " + json.dumps(group, ensure_ascii=False))
                       for i, group in enumerate(groups)] + exits
            answer = decide(contract + "\nSelect the group containing the documented category under the "
                            "measurement and conflict rules. Category groups only organize the choices; "
                            "they do not combine or rename the original categories. Do not infer absence "
                            "from silence. Use only the supplied evidence.", options, f"category_group_{group_level}")
            key = answer["selected"]
            if key in {MISSING, NONE}:
                status = ("none_of_above" if group_level == 1 else "category_selection_failed") if key == NONE else "not_documented"
                return {"value": None, "status": status,
                        "outside_initial_domain": key == NONE and group_level == 1, "calls": calls}
            group_index = int(key.removeprefix("group_"))
            offset += group_index * group_size
            values = groups[group_index]
        options = [(f"category_{offset + i}", f"category_{offset + i}: {v}") for i, v in enumerate(values)] + exits
        answer = decide(contract + "\nSelect the documented category under the measurement and conflict rules. "
                        "Do not infer absence from silence. Use only the supplied evidence.", options, "category")
        key = answer["selected"]
        value = feature["categories_or_unit"][int(key.removeprefix("category_"))] if key not in {MISSING, NONE} else None
        status = ("accepted" if value is not None else
                  ("category_selection_failed" if group_level else "none_of_above") if key == NONE else "not_documented")
        return {"value": value, "status": status,
                "outside_initial_domain": key == NONE and group_level == 0, "calls": calls}

    lo, hi = (float(feature["decision_ontology"][k]) for k in ("minimum", "maximum"))
    upper_inclusive = True
    for pass_index in range(client.policy.numeric_passes):
        edges, options = numeric_options(lo, hi, client.policy.numeric_bins, upper_inclusive=upper_inclusive)
        criterion = contract + (
            "\nSelect the interval containing the actual documented numeric value, in the canonical unit. "
            "Convert units if needed. Follow the feature's time and conflict rules; do not use a prior estimate. "
            "A threshold, ambiguous value, or unavailable conversion is not an exact numeric observation. "
            "If the documented value is outside every interval, choose none_of_above."
        )
        answer = decide(criterion, options, f"numeric_pass_{pass_index + 1}")
        key = answer["selected"]
        if key in {NONE, MISSING}:
            return {"value": None, "status": "none_of_above" if key == NONE else "not_documented",
                    "outside_initial_domain": key == NONE and pass_index == 0,
                    "rejected_interval": [lo, hi], "calls": calls}
        selected_bin = int(key.removeprefix("bin_")) - 1
        lo, hi = edges[selected_bin:selected_bin+2]
        upper_inclusive = upper_inclusive and selected_bin == client.policy.numeric_bins - 1
    estimate = lo + (hi - lo) / 2
    tolerance = client.policy.verification_relative_tolerance
    # Algebraically equivalent to |estimate - actual| <= tolerance*|actual|.
    # Ask the small model for a direct interval decision instead of arithmetic.
    allowed = sorted((estimate / (1 + tolerance), estimate / (1 - tolerance)))
    criterion = contract + (
        "\nIndependently read the actual documented value from the evidence, in the canonical unit. "
        f"Is that documented value between {allowed[0]:.12g} and {allowed[1]:.12g} "
        f"{feature['categories_or_unit'][0]}, inclusive? "
        f"This checks whether candidate {estimate:.12g} is within {100*tolerance:g}% of the documented value. "
        "Convert units if needed. Answer false for missing, ambiguous, incompatible-unit, or out-of-range evidence."
    )
    check = decide(criterion, [("true", "The proposition is true."),
                               ("false", "The proposition is false.")], "numeric_verification")
    accepted = check["selected"] == "true" and check["probabilities"]["true"] >= client.policy.verification_min_probability
    return {"value": estimate if accepted else None,
            "status": "accepted" if accepted else "numeric_verification_failed",
            "candidate": estimate, "interval": [lo, hi], "upper_inclusive": upper_inclusive,
            "verification_documented_range": allowed,
            "calls": calls}


def extract_rows(*, dataset, row_ids, text_column, definitions, output_dir, workers,
                 request_identity, policy, client, colbert):
    import pandas as pd
    from ..extraction.colbert import get_retriever
    from . import plain_handoff_stage2_analysis as analysis

    if client is None:
        raise ValueError("decision extraction requires a decision client")
    if client.policy != policy:
        raise ValueError("decision client and extraction policy differ")
    identity = claim_output_method(output_dir, policy, colbert)
    names = [str(f["name"]) for f in definitions]
    if len(set(names)) != len(names) or len(set(row_ids)) != len(row_ids):
        raise ValueError("decision extraction requires distinct feature names and row IDs")
    for feature in definitions:
        validate_ontology(feature)
    retriever = get_retriever(colbert)
    values = {int(row_id): {} for row_id in row_ids}
    results = {name: [] for name in names}

    def run(task):
        row_id, feature = task
        raw = dataset.iloc[row_id][text_column]
        source = "" if pd.isna(raw) else str(raw)
        fingerprint = analysis._value_fingerprint({"schema": SCHEMA,
            "feature": analysis._prompt_feature_definitions([feature])[0],
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "measurement": identity, "request": dict(request_identity or {})})
        directory = Path(output_dir) / "decisions" / f"row_{row_id:08d}" / fingerprint
        result_path = directory / "result.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text())
            if result.get("fingerprint") != fingerprint or result.get("row_id") != row_id:
                raise ValueError("invalid decision checkpoint identity")
        else:
            evidence = retriever.retrieve(source, [feature], top_k=colbert.top_k)
            analysis._write_json(directory / "retrieval.json", {"row_id": row_id, **evidence})
            result = {"fingerprint": fingerprint, "row_id": row_id, "feature_name": feature["name"],
                "schema": SCHEMA, **measure_feature(feature=feature, source=source, evidence=evidence, client=client)}
            analysis._write_json(result_path, result)
        return row_id, feature["name"], result

    tasks = iter((int(row_id), feature) for row_id in row_ids for feature in definitions)
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        pending = set()
        try:
            for _ in range(max(1, int(workers))):
                task = next(tasks, None)
                if task is not None:
                    pending.add(pool.submit(run, task))
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    row_id, name, result = future.result()
                    values[row_id][name] = result["value"]
                    results[name].append(result)
                    task = next(tasks, None)
                    if task is not None:
                        pending.add(pool.submit(run, task))
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    patterns, feature_summary = [], {}
    for feature in definitions:
        name = feature["name"]
        counts = Counter(r["status"] for r in results[name])
        feature_summary[name] = {"patient_count": len(row_ids), "status_counts": dict(counts)}
        for status in ("none_of_above", "numeric_verification_failed", "category_selection_failed"):
            rows = sorted(r["row_id"] for r in results[name] if r["status"] == status)
            if rows:
                patterns.append({"feature_name": name, "failure_kind": status, "reason": status,
                    "patient_count": len(rows), "patient_row_ids": rows, "total_patients": len(row_ids),
                    "patient_fraction": len(rows) / len(row_ids), "example_values": [],
                    "allowed_categories": feature["categories_or_unit"],
                    "outside_initial_domain_count": sum(bool(r.get("outside_initial_domain")) for r in results[name])})
    frame = pd.DataFrame([{"_oci_row_id": int(i), **values[int(i)]} for i in row_ids],
                         columns=["_oci_row_id", *names])
    analysis._write_frame(Path(output_dir) / "extracted.csv", frame)
    analysis._write_json(Path(output_dir) / "failure_summary.json", {
        "schema_version": analysis.EXTRACTION_ISSUE_SCHEMA_VERSION, "completed_at": analysis._now(),
        "issue_files": 0, "feature_failure_patterns": patterns, "decision_feature_summary": feature_summary,
        "structural_failure_patient_count": 0, "structural_failure_patient_row_ids": []})
    analysis._write_json(Path(output_dir) / "complete.json", {"status": "complete", "schema": SCHEMA,
        "rows": len(frame), "features": len(names), "measurement_method": identity,
        "scope": "retrieved_excerpts", "completed_at": analysis._now()})
    return frame
