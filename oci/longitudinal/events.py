"""Convert longitudinal EHR / simulator event streams into model-ready inputs.

This module is self-contained (no runtime dependency on ``synthetic_data``) and
handles both the simulator's five-stream schema and open-vocabulary real EHR
panels, lab components, diagnosis/procedure codes, and PRO instruments.
"""

import hashlib
import math
from functools import lru_cache
from numbers import Integral

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


def _numeric(value, mean, scale):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Nonfinite measurement; omit unavailable values")
    return math.asinh((value - mean) / scale)


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
            if isinstance(score, bool) or not isinstance(score, Integral) or not 0 <= score <= 4:
                raise ValueError(f"Expected an ordinal PRO-CTCAE score from 0 to 4 for {key}")
            offset = 5 * PRO_INDEX[key]
            values[offset : offset + 5] = [float(score >= threshold) for threshold in range(5)]
        else:
            if strict:
                raise ValueError(f"Unknown PRO-CTCAE item: {key}")
            if score is None:
                continue
            if isinstance(score, bool):
                raise ValueError(f"Expected a numeric PRO score for {key}")
            fval = float(score)
            if not math.isfinite(fval):
                raise ValueError(f"Nonfinite PRO score for {key}")
            if instrument == PRO_INSTRUMENT and (not isinstance(score, Integral) or not 0 <= score <= 4):
                raise ValueError(f"Expected an ordinal PRO-CTCAE score from 0 to 4 for {key}")
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


def patient_summary_vector(row):
    """Fixed-length observational numeric summary of a patient for tabular baselines.

    Uses only observed pretreatment histories (never oracle columns or outcomes).
    """
    # Labs: last, mean, and change across canonical components (3 * len(LAB_NAMES)).
    lab_series = {name: [] for name in LAB_NAMES}
    for panel in row.get("labs") or ():
        for item in panel.get("results") or ():
            name = item.get("component")
            val = item.get("value")
            if name in lab_series and val is not None and item.get("unit") == LAB_UNITS[name]:
                lab_series[name].append(_numeric(val, *LAB_SCALES[name]))
    feats = []
    for name in LAB_NAMES:
        vals = lab_series[name]
        if vals:
            feats.extend([vals[-1], sum(vals) / len(vals), vals[-1] - vals[0], 1.0])
        else:
            feats.extend([0.0, 0.0, 0.0, 0.0])

    # Vitals: last, mean, change, observed.
    vital_series = {name: [] for name in VITAL_SCALES}
    for vital in row.get("vitals") or ():
        for name, scale in VITAL_SCALES.items():
            if vital.get(name) is not None:
                vital_series[name].append(_numeric(vital[name], *scale))
    for name in VITAL_SCALES:
        vals = vital_series[name]
        if vals:
            feats.extend([vals[-1], sum(vals) / len(vals), vals[-1] - vals[0], 1.0])
        else:
            feats.extend([0.0, 0.0, 0.0, 0.0])

    # PROs: mean and last score for canonical items.
    pro_series = {item: [] for item in PRO_ITEMS}
    for form in row.get("pros") or ():
        for resp in form.get("responses") or ():
            key = (resp.get("symptom"), resp.get("attribute"))
            if key in pro_series and resp.get("score") is not None:
                pro_series[key].append(float(resp["score"]) / 4.0)
    for item in PRO_ITEMS:
        vals = pro_series[item]
        if vals:
            feats.extend([sum(vals) / len(vals), vals[-1], 1.0])
        else:
            feats.extend([0.0, 0.0, 0.0])

    # Encounters & note counts / ECOG if present.
    encounters = row.get("encounters") or ()
    ed_180 = sum(
        1.0
        for e in encounters
        if e.get("setting") == "emergency" and float(e.get("day_from_index", -9999)) >= -180
    )
    inpatient_180 = sum(
        1.0
        for e in encounters
        if e.get("setting") == "inpatient" and float(e.get("day_from_index", -9999)) >= -180
    )
    notes = row.get("notes") or ()
    feats.extend(
        [
            math.log1p(len(encounters)),
            math.log1p(ed_180),
            math.log1p(inpatient_180),
            math.log1p(len(notes)),
        ]
    )
    return feats
