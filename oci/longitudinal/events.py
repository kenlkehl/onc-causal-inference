"""Convert the simulator's observed event schema to modality-specific inputs."""

import hashlib
import math
from functools import lru_cache
from numbers import Integral

import pyarrow.dataset as ds
import torch

from synthetic_data.longitudinal_history import DAYS_PER_YEAR, DIFFERENTIAL, LAB_UNITS, STREAMS
from synthetic_data.pro_ctcae import PRO_INSTRUMENT, PRO_ITEMS, RECALL_DAYS
from synthetic_data.structured_data import LAB_REFERENCE_RANGES

# Fixed reference coordinates, independent of the context and query samples.
LAB_SCALES = {
    k: ((low + high) / 2, (high - low) / 2) for k, (_, low, high) in LAB_REFERENCE_RANGES.items()
}
LAB_SCALES.update(mcv=(90, 10), mch=(30, 4), mchc=(33, 3), rdw=(14, 2), total_protein=(7, 1))
for name, absolute, percent in zip(DIFFERENTIAL, [4, 2, 0.5, 0.2, 0.05], [60, 30, 7, 2, 1]):
    LAB_SCALES[name + "_abs"] = (absolute, absolute)
    LAB_SCALES[name + "_pct"] = (percent, percent)
LAB_NAMES = tuple(sorted(LAB_SCALES))
LAB_INDEX = {name: i for i, name in enumerate(LAB_NAMES)}
PANELS = ("CBC_DIFF", "BMP", "CMP")
VITAL_SCALES = {
    "weight_kg": (75, 20),
    "temperature_c": (37, 1),
    "systolic_mmhg": (120, 20),
    "diastolic_mmhg": (75, 12),
    "pulse_bpm": (80, 15),
    "respirations_per_min": (16, 4),
    "spo2_pct": (96, 4),
}
CODE_BUCKETS = 4096
PRO_INDEX = {item: i for i, item in enumerate(PRO_ITEMS)}
# Per item: score observed, then four cumulative ordinal thresholds.
PRO_INPUT_SIZE = 5 * len(PRO_ITEMS)


def read_patient_table(path, streams=STREAMS, labels=()):
    """Project input fields explicitly, allowing pre-PRO cohorts to omit the new column."""
    dataset = ds.dataset(path, format="parquet", partitioning="hive")
    fields = [s for s in streams if s != "pros" or s in dataset.schema.names]
    return dataset.to_table(columns=["patient_id", *fields, *labels])


def pro_vector(event):
    if event["instrument"] != PRO_INSTRUMENT or event["recall_days"] != RECALL_DAYS:
        raise ValueError("Expected an adult PRO-CTCAE assessment with seven-day recall")
    values = [0.0] * PRO_INPUT_SIZE
    seen = set()
    for item in event["responses"]:
        key = (item["symptom"], item["attribute"])
        if key not in PRO_INDEX or key in seen:
            raise ValueError(f"Unknown or duplicate PRO-CTCAE item: {key}")
        seen.add(key)
        score = item["score"]
        if score is None:
            continue
        if isinstance(score, bool) or not isinstance(score, Integral) or not 0 <= score <= 4:
            raise ValueError(f"Expected an ordinal PRO-CTCAE score from 0 to 4 for {key}")
        offset = 5 * PRO_INDEX[key]
        values[offset : offset + 5] = [float(score >= threshold) for threshold in range(5)]
    return values


def _numeric(value, mean, scale):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Nonfinite measurement; omit unavailable values")
    return math.asinh((value - mean) / scale)


def lab_vector(event):
    values = [0.0] * (2 * len(LAB_NAMES) + len(PANELS))
    values[2 * len(LAB_NAMES) + PANELS.index(event["panel"])] = 1.0
    for item in event["results"]:
        name = item["component"]
        if name not in LAB_INDEX or item["unit"] != LAB_UNITS[name]:
            raise ValueError(f"Unsupported lab component or unit: {name} {item['unit']}")
        i = LAB_INDEX[name]
        if values[len(LAB_NAMES) + i]:
            raise ValueError(f"Duplicate result in lab panel: {name}")
        if item["value"] is not None:
            values[i] = _numeric(item["value"], *LAB_SCALES[name])
            values[len(LAB_NAMES) + i] = 1.0
    return values


def vital_vector(event):
    values = [0.0] * (2 * len(VITAL_SCALES))
    for i, (name, scale) in enumerate(VITAL_SCALES.items()):
        if event.get(name) is not None:
            values[i] = _numeric(event[name], *scale)
            values[len(VITAL_SCALES) + i] = 1.0
    return values


@lru_cache(maxsize=8192)
def _code_id(name):
    digest = hashlib.blake2b(name.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % CODE_BUCKETS


def encounter_codes(event):
    codes = ["setting:" + event["setting"]]
    for system in ("icd10cm", "cpt"):
        codes.extend(system + ":" + code for code in event[system])
    return [_code_id(s) for s in sorted(set(codes))]


def event_times(stream):
    """Log elapsed days and time to index on a common scale; distinguish the first event."""
    times = []
    previous = None
    for event in stream:
        day = float(event["day_from_index"])
        gap = event["delta_years"]
        if not math.isfinite(day) or day >= 0 or (previous is not None and day < previous):
            raise ValueError("Events must be chronological and strictly before the index day")
        if previous is None:
            if gap is not None:
                raise ValueError("Expected a null first delta_years")
            elapsed = 0.0
        else:
            elapsed = float(gap) * DAYS_PER_YEAR
            if not math.isclose(elapsed, day - previous, abs_tol=1e-4):
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
