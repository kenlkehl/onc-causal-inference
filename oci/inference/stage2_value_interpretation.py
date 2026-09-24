"""Compile supported clinical numerical meanings into validated value maps.

The model describes meanings. Python builds interval boundaries and complements.
Unparsed or incompatible meanings raise a validation error, which the caller
records before retaining the original hybrid measurements as its fallback.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_NUM_RE = re.compile(_NUMBER)


def _interval(text, meaning):
    if not isinstance(text, str):
        raise ValueError("a numerical interpretation must be text")
    text = text.casefold().replace("−", "-").replace("–", " to ").replace("≥", ">=").replace("≤", "<=")
    numbers = [float(x) for x in _NUM_RE.findall(text)]
    if not numbers or any(not math.isfinite(x) for x in numbers):
        raise ValueError("the interpretation needs a supported finite numerical meaning")
    if meaning == "exact_number":
        if len(numbers) != 1 or re.search(r"[<>]|\b(?:between|range|less|greater|above|below|least|most)\b", text):
            raise ValueError("an exact numerical meaning must identify one number")
        return numbers[0], True, numbers[0], True
    if len(numbers) == 1:
        number = numbers[0]
        if re.search(r">=|at least|greater than or equal|no less than|or (?:more|greater|higher)|and (?:above|over)", text):
            return number, True, None, False
        if re.search(r"<=|at most|less than or equal|no (?:more|greater) than|or (?:less|lower)|and (?:below|under)", text):
            return None, False, number, True
        if re.search(r">|greater than|more than|above|over\b", text):
            return number, False, None, False
        if re.search(r"<|less than|below|under\b", text):
            return None, False, number, False
    if len(numbers) == 2 and numbers[0] < numbers[1]:
        # A two-sided verbal range needs explicit boundary semantics. Bare
        # 'between' is ambiguous and receives a clarification/repair request.
        if re.search(r"inclusive|including both|closed range", text) and not re.search(r"exclusive|excluding", text):
            return numbers[0], True, numbers[1], True
        if re.search(r"exclusive|excluding both|open range", text) and not re.search(r"inclusive|including", text):
            return numbers[0], False, numbers[1], False
        lower_inclusive = bool(re.search(r">=|at least|greater than or equal", text))
        upper_inclusive = bool(re.search(r"<=|at most|less than or equal", text))
        if re.search(r">|at least|greater than|above", text) and re.search(r"<|at most|less than|below", text):
            return numbers[0], lower_inclusive, numbers[1], upper_inclusive
        # Conventional half-open notation, e.g. [1, 50).
        match = re.search(rf"([\[(])\s*({_NUMBER})\s*,\s*({_NUMBER})\s*([\])])", text)
        if match:
            return numbers[0], match[1] == "[", numbers[1], match[4] == "]"
    raise ValueError("the numerical meaning is ambiguous; state the bounds and whether each boundary is included")


def _contains(interval, x):
    low, li, high, hi = interval
    return (low is None or x > low or (li and x == low)) and (high is None or x < high or (hi and x == high))


def _label(interval):
    low, li, high, hi = interval
    if low == high:
        return f"= {low:g}"
    if low is None:
        return f"{'<=' if hi else '<'} {high:g}"
    if high is None:
        return f"{'>=' if li else '>'} {low:g}"
    return f"{'[' if li else '('}{low:g}, {high:g}{']' if hi else ')'}"


def compile_value_interpretations(value, *, observations):
    required = {"status", "representation", "reason", "token_interpretations"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("value interpretation requires status, representation, reason, and token_interpretations")
    if value["status"] == "insufficient_definition":
        raise ValueError("measurement definition is insufficient: " + str(value["reason"]))
    if value["status"] != "ready" or value["representation"] not in {"continuous", "categorical"}:
        raise ValueError("a ready interpretation requires a continuous or categorical representation")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ValueError("value interpretation requires a reason")
    expected = [str(x["raw_value"]) for x in observations.get("categorical_values", [])]
    rows = value["token_interpretations"]
    if not isinstance(rows, list):
        raise ValueError("token_interpretations must be an array")
    meanings = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"raw_text", "meaning", "interpretation"}:
            raise ValueError("each phrase requires raw_text, meaning, and interpretation")
        raw = row["raw_text"]
        if raw not in expected or raw in meanings:
            raise ValueError("return each supplied phrase exactly once")
        meaning = row["meaning"]
        if meaning not in {"exact_number", "explicit_interval", "defined_category", "unusable"}:
            raise ValueError("unknown numerical meaning")
        if not isinstance(row["interpretation"], str) or not row["interpretation"].strip():
            raise ValueError("each phrase needs an explanation of its meaning")
        meanings[raw] = None if meaning == "unusable" else _interval(row["interpretation"], meaning)
    if set(meanings) != set(expected):
        raise ValueError("return an interpretation for every supplied phrase")
    result = {"target_representation": value["representation"], "reason": value["reason"],
              "canonical_categories": [], "numeric_bin_rules": [], "categorical_value_map": []}
    if value["representation"] == "continuous":
        if any(x is not None and x[0] != x[2] for x in meanings.values()):
            raise ValueError("continuous values require exact numerical meanings; thresholds require categories")
        result["categorical_value_map"] = [{"raw_value": raw, "canonical_value": interval[0] if interval else None} for raw, interval in meanings.items()]
        return result
    intervals = list(dict.fromkeys(x for x in meanings.values() if x is not None and x[0] != x[2]))
    if not intervals:
        raise ValueError("categorical representation requires at least one explicitly defined interval")
    bounds = sorted({b for x in intervals for b in (x[0], x[2]) if b is not None})
    atoms = []
    for i, bound in enumerate(bounds):
        previous = bounds[i - 1] if i else None
        midpoint = (previous + bound) / 2 if previous is not None else bound - max(1, abs(bound))
        atoms.append(((previous, False, bound, False), tuple(_contains(x, midpoint) for x in intervals)))
        atoms.append(((bound, True, bound, True), tuple(_contains(x, bound) for x in intervals)))
    last = bounds[-1]
    atoms.append(((last, False, None, False), tuple(_contains(x, last + max(1, abs(last))) for x in intervals)))
    bins, memberships = [], []
    for interval, membership in atoms:
        if bins and memberships[-1] == membership:
            bins[-1] = (*bins[-1][:2], *interval[2:])
        else:
            bins.append(interval)
            memberships.append(membership)
    labels = [_label(x) for x in bins]
    for raw, interval in meanings.items():
        if interval is None:
            category = None
        elif interval[0] == interval[2]:
            category = labels[next(i for i, x in enumerate(bins) if _contains(x, interval[0]))]
        else:
            matches = [i for i, membership in enumerate(memberships) if membership[intervals.index(interval)]]
            if len(matches) != 1:
                raise ValueError(f"the meaning of {raw!r} spans multiple required categories; a single category would lose information")
            category = labels[matches[0]]
        result["categorical_value_map"].append({"raw_value": raw, "canonical_value": category})
    result["canonical_categories"] = labels
    result["numeric_bin_rules"] = [{"lower_bound": x[0], "lower_inclusive": x[1], "upper_bound": x[2], "upper_inclusive": x[3], "canonical_value": name} for x, name in zip(bins, labels)]
    return result
