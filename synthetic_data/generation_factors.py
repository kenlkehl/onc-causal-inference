# synthetic_data/generation_factors.py
"""Generation-time factors: documentation control, post-treatment dated notes, and proxies.

These factors change the text the LLM writes, so they are chosen when a corpus
is generated. Every random choice here uses its own seeded stream, so a factor
that is off leaves the generator's random draws unchanged.
"""

from __future__ import annotations

import math
import random
import re
from datetime import date, timedelta
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .dgp_terms import CONFOUNDER, roles_of, standardized_value

# --------------------------------------------------------------------------
# Post-treatment events
# --------------------------------------------------------------------------

# Built-in treatment-caused events. ``control_rate`` is the event probability
# for an untreated patient whose covariate sits at its mean (standardized value
# 0); ``treatment_log_odds`` and ``covariate_log_odds`` add to that log odds.
POST_TREATMENT_EVENTS: Dict[str, Dict[str, Any]] = {
    "toxicity": {
        "description": "a grade 3 or higher treatment-related toxicity",
        "control_rate": 0.15,
        "treatment_log_odds": 1.0,
        "covariate_log_odds": 0.5,
        "present": (
            "The patient has developed a grade 3 or higher toxicity attributed to the "
            "current regimen (choose one typical of that regimen) that required clinical "
            "management."
        ),
        "absent": (
            "The patient has had no grade 3 or higher treatment-related toxicity on the "
            "current regimen."
        ),
    },
    "early_response": {
        "description": "an early objective tumor response",
        "control_rate": 0.25,
        "treatment_log_odds": 0.8,
        "covariate_log_odds": 0.5,
        "present": (
            "Early restaging shows an objective tumor response (partial response) on the "
            "current regimen."
        ),
        "absent": (
            "Early restaging shows no objective tumor response (stable or progressing "
            "disease) on the current regimen."
        ),
    },
}
DEFAULT_POST_TREATMENT_EVENTS = ("toxicity", "early_response")
POST_TREATMENT_KEYS = frozenset({"rate", "events", "regimen_mention_prob"})
POST_TREATMENT_EVENT_KEYS = frozenset(
    {"name", "description", "control_rate", "treatment_log_odds", "covariate",
     "covariate_log_odds", "present", "absent"}
)
# Post-index notes are dated this many days after the index date (inclusive).
POST_TREATMENT_DAY_RANGE = (7, 90)
POST_TREATMENT_MAX_NOTES = 2
POST_TREATMENT_NOTE_TYPE = "clinical_note"
_EVENT_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def _probability(value: Any, label: str, *, low_open: bool, high_open: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a number")
    value = float(value)
    if (value <= 0.0 if low_open else value < 0.0) or (value >= 1.0 if high_open else value > 1.0):
        raise ValueError(f"{label} must be in {'(' if low_open else '['}0, 1{')' if high_open else ']'}")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def validate_post_treatment_notes(value: Any) -> Dict[str, Any]:
    """Normalize a ``post_treatment_notes`` block; unknown keys are errors.

    ``events`` holds built-in event names (``toxicity``, ``early_response``) or
    objects with ``name`` and any of ``description``, ``control_rate``,
    ``treatment_log_odds``, ``covariate``, ``covariate_log_odds``, ``present``,
    and ``absent``; a built-in name supplies the defaults for the omitted keys.
    """

    if not isinstance(value, Mapping):
        raise ValueError("post_treatment_notes must be an object with 'rate' and 'events'")
    unknown = set(value) - POST_TREATMENT_KEYS
    if unknown:
        raise ValueError(f"unknown post_treatment_notes keys: {sorted(unknown)}")
    rate = _probability(value.get("rate", 1.0), "post_treatment_notes.rate",
                        low_open=True, high_open=False)
    mention = _probability(value.get("regimen_mention_prob", 0.5),
                           "post_treatment_notes.regimen_mention_prob",
                           low_open=False, high_open=False)
    raw_events = value.get("events", list(DEFAULT_POST_TREATMENT_EVENTS))
    if isinstance(raw_events, (str, Mapping)) or not isinstance(raw_events, Sequence) or not raw_events:
        raise ValueError("post_treatment_notes.events must be a non-empty list")
    events: List[Dict[str, Any]] = []
    for raw in raw_events:
        entry = {"name": raw} if isinstance(raw, str) else raw
        if not isinstance(entry, Mapping) or not isinstance(entry.get("name"), str):
            raise ValueError("each post-treatment event is a name or an object with a 'name'")
        unknown = set(entry) - POST_TREATMENT_EVENT_KEYS
        if unknown:
            raise ValueError(f"unknown post-treatment event keys: {sorted(unknown)}")
        name = entry["name"]
        if not _EVENT_NAME.match(name):
            raise ValueError(
                f"post-treatment event name {name!r} must be lowercase letters, digits, "
                "and underscores, starting with a letter"
            )
        base = dict(POST_TREATMENT_EVENTS.get(name, {}))
        merged = {**base, **{key: item for key, item in entry.items() if key != "name"}}
        missing = {"description", "control_rate", "treatment_log_odds", "covariate_log_odds"} - set(merged)
        if missing:
            raise ValueError(
                f"post-treatment event {name!r} is not built in and lacks {sorted(missing)}"
            )
        description = merged["description"]
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"post-treatment event {name!r} needs a description")
        covariate = merged.get("covariate")
        if covariate is not None and (not isinstance(covariate, str) or not covariate.strip()):
            raise ValueError(f"post-treatment event {name!r} covariate must be a variable name")
        event = {
            "name": name,
            "description": description.strip(),
            "control_rate": _probability(merged["control_rate"], f"{name}.control_rate",
                                         low_open=True, high_open=True),
            "treatment_log_odds": _finite(merged["treatment_log_odds"], f"{name}.treatment_log_odds"),
            "covariate": covariate,
            "covariate_log_odds": _finite(merged["covariate_log_odds"], f"{name}.covariate_log_odds"),
            "present": str(merged.get("present") or f"The patient has had {description.strip()}."),
            "absent": str(merged.get("absent") or f"The patient has not had {description.strip()}."),
        }
        events.append(event)
    names = [event["name"] for event in events]
    if len(set(names)) != len(names):
        raise ValueError("post-treatment event names must be distinct")
    return {"rate": rate, "events": events, "regimen_mention_prob": mention}


def post_treatment_oracle_column(event_name: str) -> str:
    return f"true_post_treatment_{event_name}"


def resolve_post_treatment_events(
    block: Mapping[str, Any], features: Sequence[Mapping[str, Any]], seed: int
) -> List[Dict[str, Any]]:
    """Fix each event's covariate: the configured one, else a seeded draw.

    The draw picks among confounder-role variables, or among all variables when
    there are none.
    """

    names = [feature["name"] for feature in features if not feature.get("proxy_of")]
    if not names:
        raise ValueError("post-treatment events need at least one variable")
    confounders = [
        feature["name"] for feature in features
        if CONFOUNDER in roles_of(feature) and not feature.get("proxy_of")
    ]
    pool = confounders or names
    rng = random.Random(f"{seed}:post_treatment_covariates")
    resolved = []
    for event in block["events"]:
        event = dict(event)
        if event["covariate"] is None:
            event["covariate"] = rng.choice(pool)
        elif event["covariate"] not in names:
            raise ValueError(
                f"post-treatment event {event['name']!r} covariate {event['covariate']!r} "
                "is not a generated variable"
            )
        event["oracle_column"] = post_treatment_oracle_column(event["name"])
        resolved.append(event)
    return resolved


def post_treatment_event_probability(
    event: Mapping[str, Any],
    characteristics: Mapping[str, Any],
    treatment: int,
    features: Sequence[Mapping[str, Any]],
    summary_stats: Mapping[str, Any],
) -> float:
    feature = next(item for item in features if item["name"] == event["covariate"])
    s = standardized_value(feature, characteristics[feature["name"]], summary_stats)
    rate = event["control_rate"]
    logit = math.log(rate / (1.0 - rate)) + event["treatment_log_odds"] * treatment
    logit += event["covariate_log_odds"] * s
    return 1.0 / (1.0 + math.exp(-logit))


def draw_post_treatment_events(
    events: Sequence[Mapping[str, Any]],
    characteristics: Mapping[str, Any],
    treatment: int,
    features: Sequence[Mapping[str, Any]],
    summary_stats: Mapping[str, Any],
    rng: np.random.Generator,
) -> Dict[str, int]:
    """One Bernoulli draw per event, in event order: ``{oracle column: 0/1}``."""

    return {
        event["oracle_column"]: int(
            rng.random()
            < post_treatment_event_probability(event, characteristics, treatment, features, summary_stats)
        )
        for event in events
    }


def post_treatment_note_plan(
    block: Mapping[str, Any], rng: random.Random
) -> List[Dict[str, Any]]:
    """Days after the index date (and regimen naming) of a patient's post-index notes.

    With probability ``rate`` the patient has one or two notes (equally
    likely), on distinct days drawn from 7 to 90; otherwise none.
    """

    if rng.random() >= block["rate"]:
        return []
    count = 1 + int(rng.random() < 0.5)
    low, high = POST_TREATMENT_DAY_RANGE
    days = sorted(rng.sample(range(low, high + 1), min(count, POST_TREATMENT_MAX_NOTES)))
    return [
        {"days_after_index": day, "mention_regimen": rng.random() < block["regimen_mention_prob"]}
        for day in days
    ]


# --------------------------------------------------------------------------
# Document dates
# --------------------------------------------------------------------------

_AGE = re.compile(r"\bage\s+(\d{1,3})\b|\b(\d{1,3})[- ]year[- ]old\b", re.IGNORECASE)
_FIRST_DATE = date(2012, 1, 1)
_FIRST_DATE_SPAN_DAYS = 7 * 365


def _event_age(text: str) -> Optional[int]:
    match = _AGE.search(text or "")
    if not match:
        return None
    return int(match.group(1) or match.group(2))


def assign_event_dates(
    events: Sequence[Mapping[str, str]], rng: random.Random
) -> Tuple[List[date], date]:
    """Calendar dates for timeline events, and the index date.

    The timeline carries no dates (events state the patient's age), so dates
    are drawn: the first event falls on a seeded day in 2012-2018; each later
    event follows the previous one by the stated age difference in years (less
    up to 90 days) when its stated age is higher, and otherwise by 3 to 45
    days. The index date is the day after the last event.
    """

    start = _FIRST_DATE + timedelta(days=rng.randrange(_FIRST_DATE_SPAN_DAYS))
    dates: List[date] = []
    previous_age: Optional[int] = None
    for event in events:
        age = _event_age(event.get("event_text", ""))
        if not dates:
            dates.append(start)
        else:
            if age is not None and previous_age is not None and age > previous_age:
                gap = max(1, int(round((age - previous_age) * 365.25)) - rng.randint(0, 90))
            else:
                gap = rng.randint(3, 45)
            dates.append(dates[-1] + timedelta(days=gap))
        if age is not None:
            previous_age = age
    index_date = (dates[-1] if dates else start) + timedelta(days=1)
    return dates, index_date


# --------------------------------------------------------------------------
# Documentation control
# --------------------------------------------------------------------------


def documentation_variables(
    config_subset: Optional[Sequence[str]], features: Sequence[Mapping[str, Any]]
) -> List[str]:
    names = [feature["name"] for feature in features]
    if config_subset is None:
        return names
    missing = sorted(set(config_subset) - set(names))
    if missing:
        raise ValueError(f"documented_variables_subset names unknown variables: {missing}")
    return [name for name in names if name in set(config_subset)]


def documentation_design(
    *,
    rate: float,
    mechanism: str,
    slope: float,
    variables: Sequence[str],
    features: Sequence[Mapping[str, Any]],
    summary_stats: Mapping[str, Any],
    sampled_characteristics: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Documentation probabilities for each controlled variable.

    MCAR shows each value with probability ``rate``. MNAR shows it with
    probability ``sigmoid(a + slope * s)``, where ``s`` is the standardized
    value (or level index) and ``a`` is solved on the Monte Carlo draws so that
    the mean probability equals ``rate``.
    """

    from scipy.optimize import brentq

    by_name = {feature["name"]: feature for feature in features}
    intercepts: Dict[str, Optional[float]] = {}
    for name in variables:
        if mechanism != "mnar" or rate in (0.0, 1.0):
            intercepts[name] = None
            continue
        feature = by_name[name]
        s = np.array([
            standardized_value(feature, chars[name], summary_stats) for chars in sampled_characteristics
        ])

        def error(intercept: float) -> float:
            return float(np.mean(1.0 / (1.0 + np.exp(-(intercept + slope * s))))) - rate

        intercepts[name] = float(brentq(error, -40.0, 40.0))
    return {
        "rate": float(rate),
        "mechanism": mechanism,
        "mnar_slope": float(slope),
        "variables": list(variables),
        "intercepts": intercepts,
    }


def documentation_probability(
    design: Mapping[str, Any],
    feature: Mapping[str, Any],
    value: Any,
    summary_stats: Mapping[str, Any],
) -> float:
    rate = design["rate"]
    intercept = design["intercepts"].get(feature["name"])
    if design["mechanism"] != "mnar" or intercept is None:
        return rate
    s = standardized_value(feature, value, summary_stats)
    return 1.0 / (1.0 + math.exp(-(intercept + design["mnar_slope"] * s)))


def documented_oracle_column(name: str) -> str:
    return f"true_{name}__documented"


def draw_documentation(
    design: Mapping[str, Any],
    characteristics: Mapping[str, Any],
    features: Sequence[Mapping[str, Any]],
    summary_stats: Mapping[str, Any],
    rng: np.random.Generator,
) -> Dict[str, int]:
    """``{variable: 1 if shown to the note writer else 0}`` for every variable.

    Variables outside the controlled set are always shown; controlled ones
    take one Bernoulli draw each, in feature order.
    """

    controlled = set(design["variables"])
    flags: Dict[str, int] = {}
    for feature in features:
        name = feature["name"]
        if name not in controlled:
            flags[name] = 1
            continue
        probability = documentation_probability(design, feature, characteristics[name], summary_stats)
        flags[name] = int(rng.random() < probability)
    return flags


# --------------------------------------------------------------------------
# Proxies
# --------------------------------------------------------------------------


def choose_proxy_sources(
    features: Sequence[Mapping[str, Any]], count: int, seed: int
) -> List[str]:
    """Seeded choice of ``count`` distinct confounder-role variables to copy."""

    pool = [feature["name"] for feature in features if CONFOUNDER in roles_of(feature)]
    if count > len(pool):
        raise ValueError(
            f"num_proxy_variables={count} exceeds the {len(pool)} confounder-role variables"
        )
    rng = random.Random(f"{seed}:proxy_sources")
    chosen = set(rng.sample(pool, count))
    return [name for name in pool if name in chosen]


def validate_proxy_definitions(
    response: Mapping[str, Any],
    sources: Sequence[str],
    existing_names: Sequence[str],
) -> List[Dict[str, Any]]:
    """Proxy feature definitions from the design LLM, one per source variable."""

    raw = response.get("proxies") if isinstance(response, Mapping) else None
    if not isinstance(raw, list):
        raise ValueError("proxy design response has no 'proxies' list")
    by_source: Dict[str, Mapping[str, Any]] = {}
    for item in raw:
        if isinstance(item, Mapping) and isinstance(item.get("source"), str):
            by_source.setdefault(item["source"], item)
    taken = set(existing_names)
    proxies = []
    for source in sources:
        item = by_source.get(source)
        if item is None:
            raise ValueError(f"proxy design response has no proxy for {source!r}")
        name = str(item.get("name") or "").strip()
        kind = str(item.get("type") or "").strip()
        if not re.match(r"^[a-z][a-z0-9_]*$", name) or "__" in name:
            raise ValueError(f"proxy name {name!r} must be a snake_case identifier")
        if name in taken:
            raise ValueError(f"proxy name {name!r} duplicates another variable")
        if kind not in {"continuous", "categorical"}:
            raise ValueError(f"proxy {name!r} has invalid type {kind!r}")
        definition: Dict[str, Any] = {
            "name": name,
            "type": kind,
            "description": str(item.get("description") or name.replace("_", " ")).strip(),
        }
        if kind == "categorical":
            categories = item.get("categories")
            if (
                not isinstance(categories, list)
                or len(categories) < 2
                or len({str(category) for category in categories}) != len(categories)
            ):
                raise ValueError(f"categorical proxy {name!r} needs two or more distinct categories")
            definition["categories"] = [str(category) for category in categories]
        definition["roles"] = []
        definition["proxy_of"] = source
        taken.add(name)
        proxies.append(definition)
    return proxies
