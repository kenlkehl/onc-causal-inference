"""Bind a study design's candidate variables to a patient's pretreatment observations."""

import hashlib
import math
import re

import numpy as np

from .dgp_terms import ROLE_GROUPS
from .longitudinal_history import history_features
from .study_designs import MEASUREMENTS, sanitize_note_findings


def _expit(x):
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _clip(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)

_MAX_ACTIVE_CAP = 16
_NOTE_PREFIXES = (
    "",
    "History: ",
    "Assessment: ",
    "Workup: ",
    "Clinical findings: ",
    "Review of systems: ",
)

# Couple legacy bank findings that overlap measured physiology to actual patient
# measurements so notes do not blatantly contradict labs, vitals, or PROs.
_PHYSIOLOGY_RULES = (
    (re.compile(r"\b(anemi[ac]|anaemi[ac])\b", re.I), "anemia"),
    (re.compile(r"\b(weight loss|cachexia)\b", re.I), "weight_loss"),
    (re.compile(r"\b(chronic kidney|renal insufficienc|ckd)\b", re.I), "ckd"),
    (re.compile(r"\bhypertension\b", re.I), "hypertension"),
    (re.compile(r"\bneutropeni", re.I), "neutropenia"),
    (re.compile(r"\bthrombocytopeni", re.I), "thrombocytopenia"),
    (re.compile(r"\b(dyspnea|shortness of breath|hypoxi)", re.I), "dyspnea"),
    (re.compile(r"\bfatigue\b", re.I), "fatigue"),
    (re.compile(r"\bnausea\b", re.I), "nausea"),
    (re.compile(r"\b(anorexia|decreased appetite|loss of appetite)\b", re.I), "appetite"),
    (re.compile(r"\bpain\b", re.I), "pain"),
)


def sanitized_design_candidates(design):
    """Return measurements and sentence-deduplicated note_findings for a design."""
    return (
        list(design["candidates"]["measurements"]),
        sanitize_note_findings(design["candidates"]["note_findings"]),
    )


def candidate_features(design, rng, *, max_active=_MAX_ACTIVE_CAP):
    """Draw one permitted role per candidate and keep inactive features in the record."""
    measurements, note_findings = sanitized_design_candidates(design)
    features, reference = [], {}
    permitted_active = []
    for kind, specs in (("measurements", measurements), ("note_findings", note_findings)):
        for spec in specs:
            name = spec["name"].replace(".", "_")
            role = str(rng.choice(spec["roles"]))
            feature = dict(
                name=name, description=spec["name"], roles=list(ROLE_GROUPS.get(role, ()))
            )
            if kind == "measurements":
                mean, std = MEASUREMENTS[spec["name"]]
                feature["type"] = "continuous"
                reference[name] = dict(type="continuous", mean=mean, std=std)
            else:
                feature.update(type="categorical", categories=["absent", "present"])
                p = spec["prevalence"]
                reference[name] = dict(
                    type="categorical", proportions={"absent": 1 - p, "present": p}
                )
            idx = len(features)
            non_unused = [r for r in spec["roles"] if r != "unused"]
            if non_unused:
                permitted_active.append((idx, non_unused))
            features.append(feature)

    active_indices = [i for i, f in enumerate(features) if f["roles"]]
    if not active_indices and permitted_active:
        idx, non_unused = permitted_active[int(rng.integers(len(permitted_active)))]
        chosen_role = str(rng.choice(non_unused))
        features[idx]["roles"] = list(ROLE_GROUPS[chosen_role])
        active_indices = [idx]

    # Keep wide designs identifiable by capping active causal variables to a
    # realistic sparse subset while retaining all remaining candidates as distractors.
    if max_active is not None and len(active_indices) > max_active:
        budget = int(rng.integers(max(3, min(6, max_active)), max_active + 1))
        keep = set(rng.choice(active_indices, size=budget, replace=False).tolist())
        for idx in active_indices:
            if idx not in keep:
                features[idx]["roles"] = []

    return features, reference


def measured_values(streams, candidates):
    """Read each numeric oracle input from its recorded measurement series."""
    derived = history_features(streams) if any("." not in s["name"] for s in candidates) else {}
    labs = {}
    for panel in streams["labs"]:
        for item in panel["results"]:
            labs.setdefault(item["component"], []).append(item["value"])
    pros = {}
    for survey in streams["pros"]:
        for item in survey["responses"]:
            if item["score"] is not None:
                pros.setdefault((item["symptom"], item["attribute"]), []).append(item["score"])
    values = {}
    for spec in candidates:
        key = spec["name"]
        if key in derived:
            value = derived[key]
        else:
            source, field, *rest = key.split(".")
            summary = rest[-1]
            if source == "lab":
                series = labs[field]
            elif source == "vital":
                series = [e[field] for e in streams["vitals"]]
            else:
                # An entirely unanswered PRO item gets its fixed reference mean.
                series = pros.get((field, rest[0]), [MEASUREMENTS[key][0]])
            if summary == "mean":
                value = np.mean(series)
            elif summary == "change":
                value = series[-1] - series[0]
            else:
                value = series[-1]
        values[key.replace(".", "_")] = float(value)
    return values


def _physiology_signals(streams):
    """Summarize standardized physiological signals from simulated streams."""
    if not streams:
        return {}
    labs = {}
    for panel in streams.get("labs", ()):
        for item in panel["results"]:
            labs.setdefault(item["component"], []).append(float(item["value"]))
    vitals = streams.get("vitals", ())
    pros = {}
    for survey in streams.get("pros", ()):
        for item in survey["responses"]:
            if item["score"] is not None:
                pros.setdefault((item["symptom"], item["attribute"]), []).append(float(item["score"]))

    signals = {}
    if "hemoglobin" in labs:
        signals["anemia"] = (12.0 - min(labs["hemoglobin"])) / 1.0
    if "neutrophils_abs" in labs:
        signals["neutropenia"] = (2.5 - min(labs["neutrophils_abs"])) / 1.0
    if "platelets" in labs:
        signals["thrombocytopenia"] = (190.0 - min(labs["platelets"])) / 35.0
    if "creatinine" in labs:
        signals["ckd"] = (labs["creatinine"][-1] - 1.15) / 0.28
    if vitals:
        w0, w1 = vitals[0]["weight_kg"], vitals[-1]["weight_kg"]
        signals["weight_loss"] = ((w0 - w1) / max(w0, 1.0) * 100.0 - 2.0) / 4.0
        mean_sbp = float(np.mean([v["systolic_mmhg"] for v in vitals]))
        signals["hypertension"] = (mean_sbp - 126.0) / 10.0
    if ("fatigue", "severity") in pros:
        vals = pros[("fatigue", "severity")]
        signals["fatigue"] = (sum(vals) / len(vals) - 1.7) / 1.0
    if ("shortness_of_breath", "severity") in pros:
        vals = pros[("shortness_of_breath", "severity")]
        signals["dyspnea"] = (sum(vals) / len(vals) - 1.2) / 1.0
    if ("nausea", "severity") in pros:
        vals = pros[("nausea", "severity")]
        signals["nausea"] = (sum(vals) / len(vals) - 0.65) / 0.85
    if ("decreased_appetite", "severity") in pros:
        vals = pros[("decreased_appetite", "severity")]
        signals["appetite"] = (sum(vals) / len(vals) - 1.0) / 0.95
    if ("general_pain", "severity") in pros:
        vals = pros[("general_pain", "severity")]
        signals["pain"] = (sum(vals) / len(vals) - 1.15) / 1.0

    # Composite clinical burden index for realistic cross-modal correlation.
    burden_terms = [
        signals[k]
        for k in ("anemia", "ckd", "weight_loss", "fatigue", "dyspnea")
        if k in signals and math.isfinite(signals[k])
    ]
    if burden_terms:
        signals["_burden"] = _clip(sum(burden_terms) / len(burden_terms), -2.5, 2.5)
    return signals


def _physiology_key(spec):
    blob = f"{spec['name'].replace('_', ' ')} {spec['present']} {spec['absent']}"
    for pattern, key in _PHYSIOLOGY_RULES:
        if pattern.search(blob):
            return key
    return None


def _burden_coupling(name):
    """Consistent finding-specific correlation with patient clinical burden."""
    raw = int.from_bytes(hashlib.blake2b(name.encode(), digest_size=4).digest(), "little")
    unit = raw / 0xFFFFFFFF
    # Most oncology comorbidities/adverse findings correlate positively with burden.
    return -0.25 + 0.90 * unit


def prepare_note_findings(candidates, *, paraphrase_bank=None):
    """Precompute deduplicated findings, base logits, and couplings once per design."""
    from .paraphrase_bank import finding_key

    deduped = sanitize_note_findings(candidates)
    p_lookup = paraphrase_bank.get("findings") if paraphrase_bank else None
    prepared = []
    for spec in deduped:
        base_p = _clip(float(spec["prevalence"]), 1e-4, 1.0 - 1e-4)
        base_logit = math.log(base_p / (1.0 - base_p))
        phys_key = _physiology_key(spec)
        coupling = _burden_coupling(spec["name"])
        if p_lookup is not None:
            entry = p_lookup.get(finding_key(spec["present"], spec["absent"]))
            if entry:
                present_opts = tuple(entry["present"])
                absent_opts = tuple(entry["absent"])
                section = entry.get("section", "oncology_history")
            else:
                present_opts = (spec["present"],)
                absent_opts = (spec["absent"],)
                section = "oncology_history"
            prepared.append(
                (
                    spec["name"],
                    base_p,
                    base_logit,
                    phys_key,
                    coupling,
                    present_opts,
                    absent_opts,
                    section,
                )
            )
        else:
            prepared.append(
                (spec["name"], base_p, base_logit, phys_key, coupling, spec["present"], spec["absent"])
            )
    return tuple(prepared)


def _format_realistic_note(narrative, ecog_str, sectioned_findings, paraphrase_bank, rng):
    """Assemble a multi-section oncology note from paraphrased findings and benign filler."""
    from .paraphrase_bank import SECTION_HEADERS, SECTIONS

    fillers = paraphrase_bank.get("filler_paragraphs") or {}
    ecog_templates = paraphrase_bank.get("ecog_templates") or {}
    ecog_digit = ecog_str.strip().rstrip(".")
    ecog_opts = ecog_templates.get(ecog_digit)
    ecog_line = (
        ecog_opts[int(rng.integers(len(ecog_opts)))]
        if ecog_opts
        else f"ECOG performance status: {ecog_str}"
    )

    parts = []
    hpi_fillers = fillers.get("hpi_intro")
    hpi_header = ("INTERVAL HISTORY:", "HPI:", "ONCOLOGY FOLLOW-UP:")[int(rng.integers(3))]
    if hpi_fillers and rng.random() < 0.75:
        intro = hpi_fillers[int(rng.integers(len(hpi_fillers)))]
        parts.append(f"{hpi_header} {intro} {narrative}")
    else:
        parts.append(f"{hpi_header} {narrative}")

    # Emit populated clinical sections in a mildly varied order.
    sec_order = list(SECTIONS)
    if rng.random() < 0.5:
        rng.shuffle(sec_order)
    for sec in sec_order:
        items = sectioned_findings.get(sec)
        if not items:
            continue
        if len(items) > 1:
            rng.shuffle(items)
        headers = SECTION_HEADERS.get(sec, ("CLINICAL NOTES:",))
        hdr = headers[int(rng.integers(len(headers)))]
        parts.append(f"{hdr} {' '.join(items)}")

    ros_fillers = fillers.get("ros_general")
    if ros_fillers and rng.random() < 0.65:
        parts.append(f"ROS: {ros_fillers[int(rng.integers(len(ros_fillers)))]}")

    exam_fillers = fillers.get("exam_general")
    exam_hdr = ("PHYSICAL EXAM:", "EXAM:", "OBJECTIVE:")[int(rng.integers(3))]
    if exam_fillers and rng.random() < 0.75:
        exam_line = exam_fillers[int(rng.integers(len(exam_fillers)))]
        if rng.random() < 0.5:
            parts.append(f"{exam_hdr} {exam_line} {ecog_line}")
            ecog_placed = True
        else:
            parts.append(f"{exam_hdr} {exam_line}")
            ecog_placed = False
    else:
        parts.append(f"{exam_hdr} {ecog_line}")
        ecog_placed = True

    plan_fillers = fillers.get("plan_followup")
    plan_hdr = ("ASSESSMENT & PLAN:", "IMPRESSION / PLAN:", "A/P:")[int(rng.integers(3))]
    plan_bits = []
    if not ecog_placed:
        plan_bits.append(ecog_line)
    if plan_fillers and rng.random() < 0.8:
        plan_bits.append(plan_fillers[int(rng.integers(len(plan_fillers)))])
    if plan_bits:
        parts.append(f"{plan_hdr} {' '.join(plan_bits)}")
    return "\n".join(parts)


def document_findings(
    notes,
    candidates,
    rng,
    *,
    streams=None,
    omit_absent_prob=0.0,
    style_variation=False,
    prepared=None,
    paraphrase_bank=None,
):
    """Sample design-specific findings and record each one in a dated pretreatment note.

    When ``streams`` is supplied, findings that overlap measured physiology couple
    strongly to the matching simulated stream, and other findings carry a
    finding-specific correlation with overall clinical burden. ``omit_absent_prob``
    mimics real-EHR positive-only charting where absent findings may go unstated
    rather than explicitly negated. When ``paraphrase_bank`` is supplied, findings
    are rendered with diverse LLM-authored clinical paraphrases inside multi-section
    oncology notes.
    """
    items = (
        prepared
        if prepared is not None
        else prepare_note_findings(candidates, paraphrase_bank=paraphrase_bank)
    )
    use_paraphrases = paraphrase_bank is not None or (items and len(items[0]) == 8)
    signals = _physiology_signals(streams)
    burden = signals.get("_burden", 0.0)
    has_signals = bool(signals)
    n_notes = len(notes)
    n_prefixes = len(_NOTE_PREFIXES)
    values = {}

    if not use_paraphrases:
        additions = [[] for _ in range(n_notes)]
        for name, base_p, base_logit, phys_key, coupling, present_stmt, absent_stmt in items:
            if phys_key is not None and phys_key in signals:
                prob = _expit(base_logit + 1.8 * _clip(signals[phys_key], -2.5, 2.5))
            elif has_signals:
                prob = _expit(base_logit + coupling * burden)
            else:
                prob = base_p

            is_present = rng.random() < prob
            values[name] = "present" if is_present else "absent"
            if not is_present and omit_absent_prob > 0.0 and rng.random() < omit_absent_prob:
                continue
            statement = present_stmt if is_present else absent_stmt
            if style_variation:
                prefix = _NOTE_PREFIXES[int(rng.integers(n_prefixes))]
                if prefix:
                    statement = prefix + statement
            additions[int(rng.integers(n_notes))].append(statement)

        for note, extra in zip(notes, additions):
            if extra:
                narrative, ecog = note["text"].rsplit(" ECOG performance status: ", 1)
                note["text"] = f"{narrative} {' '.join(extra)} ECOG performance status: {ecog}"
        return values

    # Loop 1: sample causal finding presence, omission, and note placement using the exact
    # same RNG consumption order as the plain path so ground-truth labels are bit-identical.
    placements = []
    for name, base_p, base_logit, phys_key, coupling, present_opts, absent_opts, section in items:
        if phys_key is not None and phys_key in signals:
            prob = _expit(base_logit + 1.8 * _clip(signals[phys_key], -2.5, 2.5))
        elif has_signals:
            prob = _expit(base_logit + coupling * burden)
        else:
            prob = base_p

        is_present = rng.random() < prob
        values[name] = "present" if is_present else "absent"
        if not is_present and omit_absent_prob > 0.0 and rng.random() < omit_absent_prob:
            continue
        if style_variation:
            _ = int(rng.integers(n_prefixes))
        note_idx = int(rng.integers(n_notes))
        placements.append((note_idx, section, present_opts if is_present else absent_opts))

    # Loop 2: render diverse clinical paraphrases and multi-section formatting after all
    # causal ground-truth draws have completed.
    render_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
    sectioned = [{} for _ in range(n_notes)]
    for note_idx, section, pool in placements:
        statement = (
            pool[int(render_rng.integers(len(pool)))] if isinstance(pool, (tuple, list)) else pool
        )
        sectioned[note_idx].setdefault(section, []).append(statement)

    for note, sec_map in zip(notes, sectioned):
        narrative, ecog = note["text"].rsplit(" ECOG performance status: ", 1)
        note["text"] = _format_realistic_note(
            narrative, ecog, sec_map, paraphrase_bank or {}, render_rng
        )
    return values
