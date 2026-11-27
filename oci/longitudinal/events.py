"""Convert longitudinal EHR / simulator event streams into model-ready inputs.

This module is self-contained (no runtime dependency on ``synthetic_data``) and
handles both the simulator's five-stream schema and open-vocabulary real EHR
panels, lab components, diagnosis/procedure codes, and PRO instruments.
"""

import hashlib
import math
import re
from functools import lru_cache
from numbers import Integral, Real

import pyarrow.dataset as ds
import torch

DAYS_PER_YEAR = 365.25
LEGACY_STREAMS = ("notes", "labs", "encounters", "vitals")
STREAMS = (*LEGACY_STREAMS, "pros")
DIFFERENTIAL = ("neutrophils", "lymphocytes", "monocytes", "eosinophils", "basophils")

# Canonical lab reference ranges (unit, low, high) matching standard adult EHR panels.
_CANONICAL_LAB_RANGES = {
    "wbc": ("k/uL", 4.0, 11.0),
    "hemoglobin": ("g/dL", 12.0, 17.5),
    "hematocrit": ("%", 36.0, 50.0),
    "platelets": ("k/uL", 150.0, 400.0),
    "rbc": ("M/uL", 4.2, 5.9),
    "sodium": ("mEq/L", 136.0, 145.0),
    "potassium": ("mEq/L", 3.5, 5.1),
    "chloride": ("mEq/L", 98.0, 107.0),
    "co2": ("mEq/L", 22.0, 29.0),
    "bun": ("mg/dL", 7.0, 20.0),
    "creatinine": ("mg/dL", 0.6, 1.3),
    "glucose": ("mg/dL", 70.0, 100.0),
    "calcium": ("mg/dL", 8.5, 10.5),
    "albumin": ("g/dL", 3.5, 5.0),
    "bilirubin": ("mg/dL", 0.1, 1.2),
    "ast": ("U/L", 10.0, 40.0),
    "alt": ("U/L", 7.0, 56.0),
    "alp": ("U/L", 44.0, 147.0),
}
LAB_UNITS = {name: info[0] for name, info in _CANONICAL_LAB_RANGES.items()}
LAB_UNITS.update(mcv="fL", mch="pg", mchc="g/dL", rdw="%", total_protein="g/dL")
LAB_UNITS.update({f"{cell}_abs": "k/uL" for cell in DIFFERENTIAL})
LAB_UNITS.update({f"{cell}_pct": "%" for cell in DIFFERENTIAL})

# Fixed reference coordinates, independent of the context and query samples.
LAB_SCALES = {
    k: ((low + high) / 2.0, (high - low) / 2.0) for k, (_, low, high) in _CANONICAL_LAB_RANGES.items()
}
LAB_SCALES.update(mcv=(90, 10), mch=(30, 4), mchc=(33, 3), rdw=(14, 2), total_protein=(7, 1))
for _name, _absolute, _percent in zip(
    DIFFERENTIAL, [4, 2, 0.5, 0.2, 0.05], [60, 30, 7, 2, 1]
):
    LAB_SCALES[_name + "_abs"] = (_absolute, _absolute)
    LAB_SCALES[_name + "_pct"] = (_percent, _percent)
LAB_NAMES = tuple(sorted(LAB_SCALES))
LAB_INDEX = {name: i for i, name in enumerate(LAB_NAMES)}
PANELS = ("CBC_DIFF", "BMP", "CMP")
# Structured lab vector: values + presence + known panels + other_panel flag + log1p(extra_count).
LAB_INPUT_SIZE = 2 * len(LAB_NAMES) + len(PANELS) + 2

VITAL_SCALES = {
    "weight_kg": (75, 20),
    "temperature_c": (37, 1),
    "systolic_mmhg": (120, 20),
    "diastolic_mmhg": (75, 12),
    "pulse_bpm": (80, 15),
    "respirations_per_min": (16, 4),
    "spo2_pct": (96, 4),
}
VITAL_INPUT_SIZE = 2 * len(VITAL_SCALES)

CODE_BUCKETS = 4096
EXTRA_LAB_BUCKETS = 1024
EXTRA_PRO_BUCKETS = 512
LEXICAL_BUCKETS = 2048
_LEXICAL_TOKEN_RE = re.compile(r"[a-z0-9]+")

PRO_INSTRUMENT = "PRO-CTCAE"
RECALL_DAYS = 7
PRO_ATTRIBUTES = {
    "fatigue": ("severity", "interference"),
    "shortness_of_breath": ("severity", "interference"),
    "nausea": ("frequency", "severity"),
    "general_pain": ("frequency", "severity", "interference"),
    "decreased_appetite": ("severity", "interference"),
}
PRO_ITEMS = tuple((s, a) for s, attributes in PRO_ATTRIBUTES.items() for a in attributes)
PRO_INDEX = {item: i for i, item in enumerate(PRO_ITEMS)}
# Per item: score observed, then four cumulative ordinal thresholds.
PRO_INPUT_SIZE = 5 * len(PRO_ITEMS)

# Explicit per-modality longitudinal summary sizes (latest, mean, slope/yr, obs + count/recency).
LAB_SUMMARY_SIZE = 4 * len(LAB_NAMES) + 2
VITAL_SUMMARY_SIZE = 4 * len(VITAL_SCALES) + 2
PRO_SUMMARY_SIZE = 4 * len(PRO_ITEMS) + 2
ENCOUNTER_SUMMARY_SIZE = 9


def read_patient_table(path, streams=STREAMS, labels=()):
    """Project input fields explicitly, allowing pre-PRO cohorts to omit the new column."""
    dataset = ds.dataset(path, format="parquet", partitioning="hive")
    fields = [s for s in streams if s != "pros" or s in dataset.schema.names]
    return dataset.to_table(columns=["patient_id", *fields, *labels])


@lru_cache(maxsize=16384)
def _hash_bucket(name, buckets=CODE_BUCKETS):
    digest = hashlib.blake2b(name.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % buckets


def _code_id(name):
    return _hash_bucket(name, CODE_BUCKETS)


_SENTENCE_SPLIT_RE = re.compile(r"(?:\n+|;\s+|(?<!\d)\.(?:\s+|$)|(?:\s+-\s+))")


def note_sentences(notes, *, max_sentences=128, min_chars=8):
    """Split a patient's pretreatment notes into deduplicated short clinical clauses."""
    seen = set()
    sentences = []
    for note in notes or ():
        text = note.get("text")
        if not text:
            continue
        for chunk in _SENTENCE_SPLIT_RE.split(str(text)):
            clause = chunk.strip(" \t\r\n*-•:")
            if len(clause) < min_chars or clause in seen:
                continue
            seen.add(clause)
            sentences.append(clause)
            if len(sentences) >= max_sentences:
                return sentences
    return sentences


def note_sentences_balanced(notes, *, max_sentences=320, min_chars=12):
    """Balanced multi-note sentence extractor that never drops late notes on long charts.

    For each note, orders clauses alternating from end (Assessment/Impression/Plan) and
    beginning (headers/pathology/biomarkers), then round-robins across all notes so every
    note in a long multi-note longitudinal history (e.g., 14+ notes, 5,000+ words) is
    represented before hitting ``max_sentences``.
    """
    per_note_ordered: list[list[str]] = []
    for note in notes or ():
        text = note.get("text") if isinstance(note, dict) else note
        if not text:
            continue
        raw: list[str] = []
        for chunk in _SENTENCE_SPLIT_RE.split(str(text)):
            clause = chunk.strip(" \t\r\n*-•:|#`")
            if len(clause) < min_chars or not any(ch.isalpha() for ch in clause):
                continue
            raw.append(clause)
        if not raw:
            continue
        # Alternate tail (impressions/conclusions) and head (diagnosis/findings) within each note
        lo, hi = 0, len(raw) - 1
        ordered: list[str] = []
        take_tail = True
        while lo <= hi:
            if take_tail:
                ordered.append(raw[hi])
                hi -= 1
            else:
                ordered.append(raw[lo])
                lo += 1
            take_tail = not take_tail
        per_note_ordered.append(ordered)

    if not per_note_ordered:
        return []

    seen: set[str] = set()
    sentences: list[str] = []
    # Reverse note order so when max_sentences is tight, most recent notes get tie-break priority
    # while still giving every note across the full history equal round-robin slots.
    note_lists = list(reversed(per_note_ordered))
    cursors = [0] * len(note_lists)
    active = len(note_lists)
    while active > 0 and len(sentences) < max_sentences:
        active = 0
        for idx, clauses in enumerate(note_lists):
            c = cursors[idx]
            while c < len(clauses) and clauses[c] in seen:
                c += 1
            if c < len(clauses):
                cl = clauses[c]
                seen.add(cl)
                sentences.append(cl)
                cursors[idx] = c + 1
                active += 1
                if len(sentences) >= max_sentences:
                    return sentences
            else:
                cursors[idx] = len(clauses)
    return sentences


def note_lexical_vector(notes):
    """Deterministic hashed 1-3gram log-count vector across a patient's pretreatment notes."""
    counts = [0.0] * LEXICAL_BUCKETS
    for note in notes or ():
        text = note.get("text")
        if not text:
            continue
        tokens = _LEXICAL_TOKEN_RE.findall(str(text).lower())
        n_tok = len(tokens)
        for i, tok in enumerate(tokens):
            counts[_hash_bucket(tok, LEXICAL_BUCKETS)] += 1.0
            if i + 1 < n_tok:
                bi = f"{tok}_{tokens[i + 1]}"
                counts[_hash_bucket(bi, LEXICAL_BUCKETS)] += 1.0
                if i + 2 < n_tok:
                    counts[_hash_bucket(f"{bi}_{tokens[i + 2]}", LEXICAL_BUCKETS)] += 1.0
    return [math.log1p(c) if c else 0.0 for c in counts]


def _numeric(value, mean, scale):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Nonfinite measurement; omit unavailable values")
    return math.asinh((value - mean) / scale)


def _ordinal_pro_score(score, key):
    """Validate an ordinal 0..4 PRO-CTCAE score, accepting Parquet-promoted integer floats."""
    if isinstance(score, bool):
        raise ValueError(f"Expected an ordinal PRO-CTCAE score from 0 to 4 for {key}")
    if isinstance(score, Integral):
        ival = int(score)
    elif isinstance(score, Real):
        fval = float(score)
        if not math.isfinite(fval) or not fval.is_integer():
            raise ValueError(f"Expected an ordinal PRO-CTCAE score from 0 to 4 for {key}")
        ival = int(fval)
    else:
        raise ValueError(f"Expected an ordinal PRO-CTCAE score from 0 to 4 for {key}")
    if not 0 <= ival <= 4:
        raise ValueError(f"Expected an ordinal PRO-CTCAE score from 0 to 4 for {key}")
    return ival


def pro_features(event, *, strict=False):
    """Return dense canonical PRO-CTCAE vector plus open-vocabulary ``(bucket, value)`` items."""
    instrument = str(event.get("instrument", PRO_INSTRUMENT))
    recall = event.get("recall_days", RECALL_DAYS)
    is_canonical_pro_ctcae = instrument == PRO_INSTRUMENT and recall == RECALL_DAYS
    if strict and not is_canonical_pro_ctcae:
        raise ValueError("Expected an adult PRO-CTCAE assessment with seven-day recall")
    values = [0.0] * PRO_INPUT_SIZE
    extra = []
    seen = set()
    for item in event.get("responses") or ():
        key = (item["symptom"], item["attribute"])
        if key in seen:
            raise ValueError(f"Duplicate PRO item: {key}")
        seen.add(key)
        score = item["score"]
        if key in PRO_INDEX and is_canonical_pro_ctcae:
            if score is None:
                continue
            ival = _ordinal_pro_score(score, key)
            offset = 5 * PRO_INDEX[key]
            values[offset : offset + 5] = [float(ival >= threshold) for threshold in range(5)]
        else:
            if strict:
                raise ValueError(f"Unknown PRO-CTCAE item: {key}")
            if score is None:
                continue
            if isinstance(score, bool):
                raise ValueError(f"Expected a numeric PRO score for {key}")
            if instrument == PRO_INSTRUMENT:
                fval = float(_ordinal_pro_score(score, key))
            else:
                fval = float(score)
                if not math.isfinite(fval):
                    raise ValueError(f"Nonfinite PRO score for {key}")
            bucket = _hash_bucket(
                f"pro:{instrument}:{key[0]}:{key[1]}", EXTRA_PRO_BUCKETS
            )
            extra.append((bucket, math.asinh(fval)))
    return values, extra


def pro_vector(event):
    """Strict 11-item PRO-CTCAE cumulative-threshold vector."""
    values, _ = pro_features(event, strict=True)
    return values


def lab_features(event, *, strict=False):
    """Return dense canonical lab vector plus open-vocabulary ``(bucket, value)`` pairs.

    Known CBC/BMP/CMP components with canonical units populate the structured
    vector; additional real-EHR panels, components, or alternative units are
    hashed into ``extra`` instead of failing closed.
    """
    values = [0.0] * LAB_INPUT_SIZE
    extra = []
    panel = str(event.get("panel", ""))
    panel_offset = 2 * len(LAB_NAMES)
    if panel in PANELS:
        values[panel_offset + PANELS.index(panel)] = 1.0
    elif strict:
        raise ValueError(f"Unsupported lab panel: {panel}")
    else:
        values[panel_offset + len(PANELS)] = 1.0
        if panel:
            extra.append((_hash_bucket("panel:" + panel, EXTRA_LAB_BUCKETS), 1.0))

    seen = set()
    for item in event.get("results") or ():
        name = str(item["component"])
        unit = str(item.get("unit", ""))
        if name in seen:
            raise ValueError(f"Duplicate result in lab panel: {name}")
        seen.add(name)
        val = item["value"]
        if name in LAB_INDEX and unit == LAB_UNITS[name]:
            i = LAB_INDEX[name]
            if val is not None:
                values[i] = _numeric(val, *LAB_SCALES[name])
                values[len(LAB_NAMES) + i] = 1.0
        elif strict:
            raise ValueError(f"Unsupported lab component or unit: {name} {unit}")
        elif val is not None:
            fval = float(val)
            if not math.isfinite(fval):
                raise ValueError("Nonfinite measurement; omit unavailable values")
            bucket = _hash_bucket(f"lab:{name}:{unit}", EXTRA_LAB_BUCKETS)
            extra.append((bucket, math.asinh(fval)))
    values[panel_offset + len(PANELS) + 1] = math.log1p(len(extra))
    return values, extra


def lab_vector(event):
    """Structured vector for canonical CBC/BMP/CMP lab panels."""
    values, _ = lab_features(event, strict=True)
    return values[: 2 * len(LAB_NAMES) + len(PANELS)]


def vital_vector(event):
    values = [0.0] * VITAL_INPUT_SIZE
    for i, (name, scale) in enumerate(VITAL_SCALES.items()):
        if event.get(name) is not None:
            values[i] = _numeric(event[name], *scale)
            values[len(VITAL_SCALES) + i] = 1.0
    return values


def encounter_codes(event):
    """Hash setting, exact ICD-10-CM/CPT codes, and 3-char category prefixes."""
    codes = ["setting:" + str(event.get("setting", "unknown"))]
    for system in ("icd10cm", "cpt"):
        for raw in event.get(system) or ():
            code = str(raw).strip()
            if not code:
                continue
            codes.append(f"{system}:{code}")
            prefix = code.split(".", 1)[0][:3]
            if len(prefix) == 3:
                codes.append(f"{system}_cat:{prefix}")
    return [_code_id(s) for s in sorted(set(codes))]


def event_times(stream):
    """Log elapsed days and time to index on a common scale; distinguish the first event.

    Accepts pretreatment and index-day events (``day_from_index <= 0``). The
    redundant ``delta_years`` field is optional for real-EHR inputs and validated
    when present.
    """
    times = []
    previous = None
    for event in stream:
        day = float(event["day_from_index"])
        gap = event.get("delta_years")
        if not math.isfinite(day) or day > 0 or (previous is not None and day < previous):
            raise ValueError("Events must be chronological and on or before the index day")
        if previous is None:
            if gap is not None:
                raise ValueError("Expected a null first delta_years")
            elapsed = 0.0
        else:
            elapsed = day - previous
            if gap is not None:
                gap_days = float(gap) * DAYS_PER_YEAR
                if not math.isclose(gap_days, elapsed, abs_tol=1e-4):
                    raise ValueError("delta_years disagrees with consecutive event days")
        times.append([math.log1p(-day), math.log1p(elapsed), float(previous is None)])
        previous = day
    return times


def binary_observations(rows, device):
    if not rows:
        raise ValueError("Expected at least one labeled context patient")
    observations = torch.tensor(
        [[row["treatment_indicator"], row["outcome_indicator"]] for row in rows],
        device=device,
        dtype=torch.float32,
    )
    if not ((observations == 0) | (observations == 1)).all():
        raise ValueError("Expected binary observed treatment and outcome")
    return observations.unbind(dim=1)


def _series_stats(points):
    """Return ``[latest, mean, asinh(ols_slope_per_year), observed]`` for ``(day, val)`` pairs."""
    if not points:
        return [0.0, 0.0, 0.0, 0.0]
    latest = points[-1][1]
    n = len(points)
    v_mean = sum(v for _, v in points) / n
    slope = 0.0
    if n >= 2 and (points[-1][0] - points[0][0]) >= 1.0:
        ts = [day / DAYS_PER_YEAR for day, _ in points]
        t_mean = sum(ts) / n
        cov = sum((t - t_mean) * (v - v_mean) for t, (_, v) in zip(ts, points))
        var = sum((t - t_mean) ** 2 for t in ts)
        slope = math.asinh(cov / (var + 1e-4))
    return [latest, v_mean, slope, 1.0]


def lab_stream_summary(stream):
    """Fixed-length longitudinal summary (latest, mean, slope/yr, obs) for canonical labs."""
    stream = stream or ()
    series = {name: [] for name in LAB_NAMES}
    for panel in stream:
        day = float(panel["day_from_index"])
        for item in panel.get("results") or ():
            name = item.get("component")
            val = item.get("value")
            if name in series and val is not None and item.get("unit") == LAB_UNITS[name]:
                series[name].append((day, _numeric(val, *LAB_SCALES[name])))
    feats = []
    for name in LAB_NAMES:
        feats.extend(_series_stats(series[name]))
    recency = math.log1p(max(0.0, -float(stream[-1]["day_from_index"]))) if stream else 0.0
    feats.extend([math.log1p(len(stream)), recency])
    return feats


def vital_stream_summary(stream):
    """Fixed-length longitudinal summary (latest, mean, slope/yr, obs) for canonical vitals."""
    stream = stream or ()
    series = {name: [] for name in VITAL_SCALES}
    for vital in stream:
        day = float(vital["day_from_index"])
        for name, scale in VITAL_SCALES.items():
            if vital.get(name) is not None:
                series[name].append((day, _numeric(vital[name], *scale)))
    feats = []
    for name in VITAL_SCALES:
        feats.extend(_series_stats(series[name]))
    recency = math.log1p(max(0.0, -float(stream[-1]["day_from_index"]))) if stream else 0.0
    feats.extend([math.log1p(len(stream)), recency])
    return feats


def pro_stream_summary(stream):
    """Fixed-length longitudinal summary (latest, mean, slope/yr, obs) for canonical PROs."""
    stream = stream or ()
    series = {item: [] for item in PRO_ITEMS}
    for form in stream:
        instrument = str(form.get("instrument", PRO_INSTRUMENT))
        recall = form.get("recall_days", RECALL_DAYS)
        if instrument != PRO_INSTRUMENT or recall != RECALL_DAYS:
            continue
        day = float(form["day_from_index"])
        for resp in form.get("responses") or ():
            key = (resp.get("symptom"), resp.get("attribute"))
            score = resp.get("score")
            if key in series and score is not None:
                series[key].append((day, float(score) / 4.0))
    feats = []
    for item in PRO_ITEMS:
        feats.extend(_series_stats(series[item]))
    recency = math.log1p(max(0.0, -float(stream[-1]["day_from_index"]))) if stream else 0.0
    feats.extend([math.log1p(len(stream)), recency])
    return feats


def encounter_stream_summary(stream):
    """Fixed-length utilization and recency summary for longitudinal encounters."""
    stream = stream or ()
    ed_total = ed_180 = ed_90 = 0.0
    inpt_total = inpt_180 = outpt_total = 0.0
    code_total = 0.0
    for enc in stream:
        day = float(enc.get("day_from_index", -9999.0))
        setting = enc.get("setting")
        if setting == "emergency":
            ed_total += 1.0
            if day >= -180.0:
                ed_180 += 1.0
            if day >= -90.0:
                ed_90 += 1.0
        elif setting == "inpatient":
            inpt_total += 1.0
            if day >= -180.0:
                inpt_180 += 1.0
        elif setting == "outpatient":
            outpt_total += 1.0
        code_total += len(enc.get("icd10cm") or ()) + len(enc.get("cpt") or ())
    recency = math.log1p(max(0.0, -float(stream[-1]["day_from_index"]))) if stream else 0.0
    return [
        math.log1p(len(stream)),
        math.log1p(ed_total),
        math.log1p(ed_180),
        math.log1p(ed_90),
        math.log1p(inpt_total),
        math.log1p(inpt_180),
        math.log1p(outpt_total),
        math.log1p(code_total),
        recency,
    ]


def patient_summary_vector(row):
    """Fixed-length observational numeric summary of a patient for tabular baselines.

    Uses only observed pretreatment histories (never oracle columns or outcomes).
    """
    notes = row.get("notes") or ()
    return [
        *lab_stream_summary(row.get("labs")),
        *vital_stream_summary(row.get("vitals")),
        *pro_stream_summary(row.get("pros")),
        *encounter_stream_summary(row.get("encounters")),
        math.log1p(len(notes)),
    ]
