"""Clinical modifier ordering with Python-owned ranks, ties, and merge state."""
from hashlib import sha256
from pathlib import Path

from . import stage2_clinical_prompts as clinical_prompts
from .stage2_multi_model_adjudication import build_multi_model_role_evidence, _bounded_batches
from .stage2_prompt_io import request_review
from .stage2_role_adjudication import _fingerprint, _write_json

RANKING_VERSION = "stage2_modifier_ranking_v2_clinical_pairwise"
SYSTEM_PROMPT = clinical_prompts.SYSTEM_PROMPTS["18_rank_modifiers"]


def _ranking_validator(cards, ordered_lists=()):
    by_id = {c["feature_id"]: c for c in cards}
    labels = {clinical_prompts.label(c["definition"]): c["feature_id"] for c in cards}
    def validate(response):
        if "ordered_groups" in response:
            if set(response) != {"ordered_groups"} or not isinstance(response["ordered_groups"], list):
                raise ValueError("return ordered_groups from strongest to weakest modifier evidence")
            rows = []
            for group in response["ordered_groups"]:
                if set(group) != {"features", "rationale"} or not isinstance(group["features"], list) or not group["features"]:
                    raise ValueError("each ordered group requires clinical names and a rationale")
                keys = [clinical_prompts.resolve_label(x, labels) for x in group["features"]]
                keys.sort(key=lambda key: (clinical_prompts.label(by_id[key]["definition"]).casefold(), key))
                rows.extend({"feature_id": key, "rationale": group["rationale"]} for key in keys)
        else:
            rows = response.get("ranking")
        if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
            raise ValueError("modifier ranking requires an ordered set of clinical variables")
        ids = [r.get("feature_id") for r in rows]
        if any(not isinstance(k, str) for k in ids) or len(set(ids)) != len(ids) or set(ids) != set(by_id):
            raise ValueError("include every supplied clinical variable exactly once")
        for group in ordered_lists:
            if [key for key in ids if key in set(group)] != list(group):
                raise ValueError("internal ranking merge changed the prior order")
        cleaned = []
        for row in rows:
            reason, key = row.get("rationale"), row["feature_id"]
            if not isinstance(reason, str) or not reason.strip() or len(reason) > 4000:
                raise ValueError("each ranking group requires a concise rationale")
            available = {r["evidence_id"] for r in by_id[key]["modeling_evidence"] if r["role"] == "effect"}
            if "evidence_ids" in row and (not isinstance(row["evidence_ids"], list) or not set(row["evidence_ids"]) <= available):
                raise ValueError("ranking provenance must use its own supplied effect evidence")
            cleaned.append({"feature_id": key, "rationale": reason.strip(),
                "evidence_ids": [r["evidence_id"] for r in by_id[key]["modeling_evidence"] if r["role"] == "effect"],
                "evidence_attachment": "Available effect evidence attached by Python."})
        return {"ranking": cleaned}
    return validate


def _pair_validator(cards):
    labels = {clinical_prompts.label(c["definition"]): c["feature_id"] for c in cards}
    def validate(response):
        reason = response.get("rationale")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 4000:
            raise ValueError("explain the modifier comparison and uncertainty")
        if "preferred_feature_id" in response:
            key = response["preferred_feature_id"]
        else:
            if set(response) != {"preferred_feature", "rationale"}:
                raise ValueError("return preferred_feature and rationale")
            key = None if response["preferred_feature"] is None else clinical_prompts.resolve_label(response["preferred_feature"], labels)
        if key is not None and key not in labels.values():
            raise ValueError("choose one of the two supplied variables, or null for a tie")
        return {"preferred_feature_id": key, "rationale": reason.strip()}
    return validate


def rank_modifier_candidates(*, definitions, statistical_report, request_json, output_dir,
    role_policy, maximum, maximum_chars, model_identity):
    evidence = build_multi_model_role_evidence(definitions=definitions, statistical_report=statistical_report, policy=role_policy)
    locked = {str(f["feature_id"]) for f in definitions if f.get("configured_explicit_feature") is True}
    cards = [c for c in evidence["candidates"] if c["feature_id"] not in locked]
    by_id = {c["feature_id"]: c for c in cards}
    directory = Path(output_dir)
    identity = {"version": RANKING_VERSION, "prompt_version": clinical_prompts.PROMPT_VERSION,
        "evidence": _fingerprint(evidence), "model": model_identity, "role_policy": role_policy.public_dict(),
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}
    def messages(batch, slug="18_rank_modifiers"):
        return clinical_prompts.messages(slug, clinical_prompts.evidence_input(evidence, batch))
    requests = 0
    def request(batch, *, pair=False):
        nonlocal requests
        rendered = messages(batch, "19_merge_rankings" if pair else "18_rank_modifiers")
        key = _fingerprint({"identity": identity, "messages": rendered})
        requests += 1
        return request_review(directory / "requests" / key[:20], rendered,
            _pair_validator(batch) if pair else _ranking_validator(batch),
            request_json=request_json, identity=identity, maximum_chars=maximum_chars)
    batches = _bounded_batches(cards, messages, maximum_items=max(2, role_policy.max_candidates_per_request), maximum_chars=maximum_chars)
    lists = [request(batch)["ranking"][:maximum] for batch in batches]
    def merge(left, right):
        result, a, b = [], 0, 0
        while a < len(left) and b < len(right) and len(result) < maximum:
            keys = [left[a]["feature_id"], right[b]["feature_id"]]
            comparison = request([by_id[key] for key in keys], pair=True)
            key = comparison["preferred_feature_id"]
            if key is None:
                key = min(keys, key=lambda k: (clinical_prompts.label(by_id[k]["definition"]).casefold(), k))
            if key == keys[0]:
                result.append(left[a]); a += 1
            else:
                result.append(right[b]); b += 1
        return (result + left[a:] + right[b:])[:maximum]
    while len(lists) > 1:
        lists = [merge(lists[i], lists[i + 1]) if i + 1 < len(lists) else lists[i] for i in range(0, len(lists), 2)]
    ranking = lists[0] if lists else []
    if len(ranking) != min(maximum, len(cards)):
        raise RuntimeError("ranking merge omitted clinical variables")
    result = {"schema_version": RANKING_VERSION, "ranking": ranking, "reviewed_candidate_ids": list(by_id),
              "locked_feature_ids": sorted(locked), "identity": identity, "maximum": maximum,
              "comparisons_and_batch_requests": requests,
              "tie_policy": "clinical_name_then_internal_identity", "merge_policy": "pairwise_leading_candidates"}
    _write_json(directory / "ranking.json", result)
    return result
