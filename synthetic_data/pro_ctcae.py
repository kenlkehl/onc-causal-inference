"""Adult PRO-CTCAE v1.0 items selected for the synthetic oncology cohort."""

import numpy as np

# NCI adult English item library, terms 53, 19, 9, 48, and 8 respectively.
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
PRO_SOURCE = (
    "https://healthcaredelivery.cancer.gov/pro-ctcae/instruments/pro-ctcae/pro-ctcae_english.pdf"
)


def weekly_assessments(rng, visit_days, visit_burden):
    """Yield submitted forms, using only illness states up to each assessment day."""
    # Schedule up to twelve non-overlapping recall weeks ending at the last visit.
    last = int(visit_days[-1])
    earliest = max(int(visit_days[0]) + RECALL_DAYS - 1, last - 11 * RECALL_DAYS)
    days = range(last, earliest - 1, -RECALL_DAYS)
    typical = np.array([1.7, 1.2, 0.7, 1.25, 1.0]) + rng.normal(0, 0.75, len(PRO_ATTRIBUTES))
    fluctuation = np.zeros(len(typical))
    burden = np.asarray(visit_burden)
    for day in reversed(days):
        week = np.arange(day - RECALL_DAYS + 1, day + 1)
        illness = burden[visit_days.searchsorted(week, side="right") - 1].mean()
        fluctuation = 0.6 * fluctuation + rng.normal(0, 0.45, len(typical))
        if day != last and rng.random() < 0.2:
            continue
        levels = typical + 0.65 * illness + fluctuation
        responses = []
        for (symptom, attributes), level in zip(PRO_ATTRIBUTES.items(), levels, strict=True):
            previous = None
            for i, attribute in enumerate(attributes):
                # Electronic branching propagates a preceding zero. A skipped
                # question and its dependent follow-ups have no numeric score.
                if i and previous in (None, 0):
                    score = previous
                elif (day != last or symptom != "fatigue") and rng.random() < 0.04:
                    score = None
                else:
                    shift = -0.35 if attribute == "interference" else 0.0
                    raw = round(float(level) + shift + float(rng.normal(0.0, 0.55)))
                    score = 0 if raw < 0 else (4 if raw > 4 else int(raw))
                responses.append(dict(symptom=symptom, attribute=attribute, score=score))
                previous = score
        yield day, dict(instrument=PRO_INSTRUMENT, recall_days=RECALL_DAYS, responses=responses)
