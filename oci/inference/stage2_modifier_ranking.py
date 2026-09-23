"""Evidence-cited modifier ordering with bounded, resumable LLM comparisons.

The LLM orders candidates, not the number retained or the confounder set.
Sorted batches are merged by comparing their leading windows. A merge preserves
each input list's order, and stops emitting when a window is exhausted so an
unseen candidate is never skipped. Only the requested prefix is retained.
"""

from hashlib import sha256
from pathlib import Path

from .stage2_multi_model_adjudication import build_multi_model_role_evidence
from .stage2_multi_model_selection import _checkpoint
from .stage2_role_adjudication import _canonical_json, _fingerprint, _write_json

RANKING_VERSION = "stage2_modifier_ranking_v1"
SYSTEM_PROMPT = """Rank pretreatment measurements as inputs to a heterogeneous treatment
effect model, using only the supplied modeling evidence. Compare effect-signal
magnitude, held-out R-loss gains, fold consistency, evaluability, complementary
information, and redundant proxies. Prognostic importance alone is not effect
modification. Logistic interactions concern log odds; orthogonal models concern
outcome/probability differences. Methods and overlapping folds are not independent
replications; support fractions and p-values are not causal probabilities.
All modifier evidence uses the supplied propensity-eligible population; assess
its evaluability within that population, not as evidence for excluded patients.
Order every supplied candidate exactly once, including weak or unevaluable ones.
Put weak, contradictory, or redundant evidence later; do not invent support.
Cite only each candidate's supplied effect-evidence IDs. No count or hard p-value
threshold is chosen here. Confounder retention and investigator locks are handled
separately. Do not merge, rename, or alter measurements. Never infer an oracle or
hidden data-generating process. For a merge, interleave the supplied ordered
lists without changing the relative order within either list. Return JSON only.
""".strip()


def _ranking_validator(cards, ordered_lists=()):
    allowed = {
        c["feature_id"]: {
            row["evidence_id"]
            for row in c["modeling_evidence"]
            if row["role"] == "effect"
        }
        for c in cards
    }

    def validate(response):
        rows = response.get("ranking")
        if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
            raise ValueError("modifier ranking must be a list of candidate objects")
        ids = [row.get("feature_id") for row in rows]
        if (
            any(not isinstance(key, str) for key in ids)
            or len(set(ids)) != len(ids)
            or set(ids) != set(allowed)
        ):
            raise ValueError(
                "modifier ranking must contain every supplied candidate exactly once"
            )
        for group in ordered_lists:
            members = set(group)
            if [key for key in ids if key in members] != list(group):
                raise ValueError(
                    "modifier merge must preserve order within each input list"
                )
        cleaned = []
        for row in rows:
            refs, reason, key = (
                row.get("evidence_ids"),
                row.get("rationale"),
                row["feature_id"],
            )
            if (
                not isinstance(refs, list)
                or any(not isinstance(ref, str) for ref in refs)
                or not set(refs) <= allowed[key]
                or (allowed[key] and not refs)
            ):
                raise ValueError(
                    "modifier ranking must cite its own supplied effect evidence"
                )
            if not isinstance(reason, str) or not reason.strip() or len(reason) > 1600:
                raise ValueError(
                    "modifier ranking requires a rationale of 1 through 1600 characters"
                )
            cleaned.append(
                {
                    "feature_id": key,
                    "evidence_ids": list(dict.fromkeys(refs)),
                    "rationale": reason.strip(),
                }
            )
        return {"ranking": cleaned}

    return validate


def rank_modifier_candidates(
    *,
    definitions,
    statistical_report,
    request_json,
    output_dir,
    role_policy,
    maximum,
    maximum_chars,
    model_identity,
):
    evidence = build_multi_model_role_evidence(
        definitions=definitions,
        statistical_report=statistical_report,
        policy=role_policy,
    )
    # Explicit roles are exact. Locked modifiers are included outside this budget;
    # locked confounder-only features cannot be promoted to modifier by a ranking.
    locked = {
        str(f["feature_id"])
        for f in definitions
        if f.get("configured_explicit_feature") is True
    }
    cards = [c for c in evidence["candidates"] if c["feature_id"] not in locked]
    by_id = {c["feature_id"]: c for c in cards}
    directory = Path(output_dir)
    identity = {
        "version": RANKING_VERSION,
        "evidence": _fingerprint(evidence),
        "model": model_identity,
        "role_policy": role_policy.public_dict(),
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    maximum_items = max(2, role_policy.max_candidates_per_request)

    def payload(batch, groups=()):
        return {
            "task": "merge_stage2_modifier_ranking"
            if groups
            else "rank_stage2_modifiers",
            "version": RANKING_VERSION,
            "candidates": batch,
            "score_meaning": evidence["score_meaning"],
            "analysis_populations": evidence["analysis_populations"],
            "ordered_lists": list(groups),
            "required_response": {
                "ranking": [
                    {
                        "feature_id": "each supplied candidate exactly once",
                        "evidence_ids": [
                            "this candidate's supplied effect evidence IDs"
                        ],
                        "rationale": "compare evidence, redundancy, and uncertainty",
                    }
                ]
            },
        }

    def fits(batch, groups=()):
        return (
            len(SYSTEM_PROMPT) + len(_canonical_json(payload(batch, groups)))
            <= maximum_chars
        )

    def request(batch, groups=()):
        value = payload(batch, groups)
        if not fits(batch, groups):
            raise ValueError(
                "modifier ranking prompt exceeds max_prompt_chars; no evidence was truncated"
            )
        validate = _ranking_validator(batch, groups)
        fingerprint = _fingerprint(
            {"identity": identity, "system": SYSTEM_PROMPT, "payload": value}
        )
        root = directory / "requests" / fingerprint[:20]
        _write_json(
            root / "prompt.json",
            {
                "system": SYSTEM_PROMPT,
                "payload": value,
                "input_fingerprint": fingerprint,
            },
        )
        result = _checkpoint(
            root,
            "response",
            fingerprint,
            lambda: validate(
                request_json(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": _canonical_json(value)},
                    ],
                    validate,
                    request_kind="interpretation",
                )
            ),
        )
        return validate(result)["ranking"]

    batches, batch = [], []
    for card in cards:
        if batch and (len(batch) >= maximum_items or not fits([*batch, card])):
            batches.append(batch)
            batch = []
        if not fits([card]):
            raise ValueError("one modifier ranking candidate exceeds max_prompt_chars")
        batch.append(card)
    if batch:
        batches.append(batch)
    lists = [request(batch)[:maximum] for batch in batches]

    def merge(left, right):
        result = []
        while left and right and len(result) < maximum:
            window = max(1, maximum_items // 2)
            a, b = left[:window], right[:window]
            while True:
                groups = [[r["feature_id"] for r in a], [r["feature_id"] for r in b]]
                combined = [by_id[key] for group in groups for key in group]
                if fits(combined, groups):
                    break
                if len(a) == len(b) == 1:
                    raise ValueError(
                        "modifier ranking cannot compare two candidates within max_prompt_chars"
                    )
                if len(a) >= len(b) and len(a) > 1:
                    a = a[:-1]
                else:
                    b = b[:-1]
            ordered = request(combined, groups)
            consumed = [0, 0]
            first = set(groups[0])
            for row in ordered:
                side = 0 if row["feature_id"] in first else 1
                result.append(row)
                consumed[side] += 1
                if len(result) == maximum or consumed[side] == len(groups[side]):
                    break
            left, right = left[consumed[0] :], right[consumed[1] :]
        return (result + left + right)[:maximum]

    while len(lists) > 1:
        lists = [
            merge(lists[i], lists[i + 1]) if i + 1 < len(lists) else lists[i]
            for i in range(0, len(lists), 2)
        ]
    ranking = lists[0] if lists else []
    assert len(ranking) == min(maximum, len(cards))
    result = {
        "schema_version": RANKING_VERSION,
        "ranking": ranking,
        "reviewed_candidate_ids": list(by_id),
        "locked_feature_ids": sorted(locked),
        "identity": identity,
        "maximum": maximum,
    }
    _write_json(directory / "ranking.json", result)
    return result
