"""Bind a study design's candidate variables to a patient's pretreatment observations."""

import numpy as np

from .longitudinal_history import history_features
from .study_designs import MEASUREMENTS
from .dgp_terms import ROLE_GROUPS


def candidate_features(design, rng):
    """Draw one permitted role per candidate and keep inactive features in the record."""
    features, reference = [], {}
    for kind in ("measurements", "note_findings"):
        for spec in design["candidates"][kind]:
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
            features.append(feature)
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


def document_findings(notes, candidates, rng):
    """Sample design-specific findings and record each one in a dated pretreatment note."""
    values = {}
    for spec in candidates:
        level = "present" if rng.random() < spec["prevalence"] else "absent"
        values[spec["name"]] = level
        note = notes[int(rng.integers(len(notes)))]
        narrative, ecog = note["text"].rsplit(" ECOG performance status: ", 1)
        note["text"] = f"{narrative} {spec[level]} ECOG performance status: {ecog}"
    return values
