from collections import Counter
from types import SimpleNamespace

from synthetic_data.nemotron_bank import (
    ROLE_PROFILES,
    SEED,
    SITES,
    design_request,
    make_plans,
    make_questions,
    slot_options,
)


def test_width_schedule_and_unseen_cancers():
    rows = {
        site: [
            dict(
                population=cancer,
                control=f"control {i}",
                treatment=f"treatment {i}",
                outcome="hospitalization",
                horizon_days=90,
            )
            for i in range(5)
        ]
        for site, (cancer, _) in SITES.items()
    }
    questions = make_questions(rows, SEED)
    plans = make_plans(questions, SEED)
    widths = Counter(p[3] for p in plans)
    assert set(widths) == set(range(3, 101)) and set(widths.values()) == {51, 52}
    assert Counter(p[1]["split"] for p in plans) == {"train": 4000, "held_out": 1000}
    sites = {
        split: {s for s, q in questions if q["split"] == split} for split in ("train", "held_out")
    }
    assert len(sites["held_out"] - sites["train"]) == 3
    assert len(plans) == 5000 and plans == make_plans(questions, SEED)
    for n in (3, 50, 100):
        slots = slot_options(300, n, SEED)
        ids = [i for slot in slots for i in slot]
        assert len(ids) == len(set(ids)) and len(slots) == n and min(map(len, slots)) >= 2


def test_model_selects_each_feature_and_its_roles():
    question = dict(
        population="colon cancer",
        control="A",
        treatment="B",
        outcome="an event",
        horizon_days=90,
        id="colon_q01",
    )
    catalog = [
        (
            "note_findings",
            dict(
                name=f"finding_{i}",
                prevalence=0.5,
                present=f"Finding {i} present.",
                absent=f"Finding {i} absent.",
            ),
        )
        for i in range(300)
    ]
    width = 100
    plan = ("colon", question, 0, width)

    def design(code):
        client = SimpleNamespace(
            seed=SEED, request=lambda key, prompt, parse, **kw: parse(" ".join([code] * width))
        )
        return design_request(client, plan, catalog)["candidates"]["note_findings"]

    first, second = design("A"), design("H")
    assert len(first) == len(second) == width
    assert {i["name"] for i in first}.isdisjoint(i["name"] for i in second)
    assert all(i["roles"] == list(ROLE_PROFILES[0]) for i in first)
    assert all(i["roles"] == list(ROLE_PROFILES[1]) for i in second)
