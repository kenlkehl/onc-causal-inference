"""Clinical modifier ordering with Python-owned ranks, ties, and merge state."""
from collections import Counter
from collections.abc import Mapping
from hashlib import sha256
import json
from pathlib import Path

from . import stage2_clinical_prompts as clinical_prompts
from .stage2_multi_model_adjudication import build_multi_model_role_evidence, _bounded_batches
from .stage2_prompt_io import request_review, fingerprint as review_fingerprint
from .stage2_role_adjudication import _fingerprint, _write_json

RANKING_VERSION = "stage2_modifier_ranking_v2_clinical_pairwise"
SYSTEM_PROMPT = clinical_prompts.SYSTEM_PROMPTS["18_rank_modifiers"]
# Completed responses from this version have identical clinical inputs and
# normalized semantics. They must still pass the current validator and hash check.
COMPATIBLE_SOURCE_HASHES = (
    "071710f1d30dc5da2402ecee008a8edffc8e44aa61ed8f6913c6ffc152b51f0d",
)


def _card_labels(cards):
    labels = {clinical_prompts.label(c["definition"]): c["feature_id"] for c in cards}
    if (len(labels) != len(cards) or any(not label for label in labels)
            or len(set(labels.values())) != len(cards)):
        raise ValueError("ranking inputs require distinct nonempty clinical labels and feature identities")
    return labels


def _resolve(value, labels):
    try:
        return clinical_prompts.resolve_label(value, labels)
    except ValueError as exc:
        raise ValueError(f"{exc}. Copy a complete supplied name, including all colon-separated "
            "components. Allowed names: " + json.dumps(list(labels), ensure_ascii=False)) from exc


def _response_format(name, properties):
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True,
        "schema": {"type": "object", "additionalProperties": False,
            "required": list(properties), "properties": properties}}}


_RATIONALE_SCHEMA = {"type": "string", "minLength": 1, "maxLength": 4000}


def _ranking_validator(cards, ordered_lists=()):
    by_id = {c["feature_id"]: c for c in cards}
    labels = _card_labels(cards)
    display = {key: name for name, key in labels.items()}
    def names(keys):
        return json.dumps([display.get(key, key) for key in sorted(keys)], ensure_ascii=False)
    instructions = ('Return {"ordered_groups": [{"features": ["complete variable name"], '
        '"rationale": "evidence and uncertainty"}]}. Each group must contain exactly '
        '"features" (an array, even for one name) and "rationale" (nonempty text). '
        'Order groups from strongest to weakest evidence; names in one group are tied. '
        'Include each supplied name exactly once, retaining all colon-separated components. '
        'Required variable names: ' + json.dumps(list(labels), ensure_ascii=False))
    def validate(response):
        if not isinstance(response, Mapping):
            raise ValueError("modifier ranking requires an object containing ordered_groups")
        if "ordered_groups" in response:
            if set(response) != {"ordered_groups"} or not isinstance(response["ordered_groups"], list):
                raise ValueError("return ordered_groups from strongest to weakest modifier evidence")
            rows = []
            for index, group in enumerate(response["ordered_groups"], 1):
                if (not isinstance(group, Mapping) or set(group) != {"features", "rationale"}
                        or not isinstance(group["features"], list) or not group["features"]):
                    raise ValueError(f"ordered group {index} must contain exactly features "
                        "(a nonempty array of complete variable names) and rationale (nonempty text). "
                        + instructions)
                keys = [_resolve(x, labels) for x in group["features"]]
                keys.sort(key=lambda key: (clinical_prompts.label(by_id[key]["definition"]).casefold(), key))
                rows.extend({"feature_id": key, "rationale": group["rationale"]} for key in keys)
        else:
            rows = response.get("ranking")
        if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
            raise ValueError("modifier ranking requires an ordered set of clinical variables")
        ids = [r.get("feature_id") for r in rows]
        if any(not isinstance(k, str) for k in ids):
            raise ValueError("ranking entries require clinical variable identities")
        counts = Counter(ids)
        missing, unknown = set(by_id) - counts.keys(), counts.keys() - set(by_id)
        duplicates = {key for key, count in counts.items() if count > 1}
        if missing or unknown or duplicates:
            raise ValueError("include every supplied clinical variable exactly once; "
                f"missing variables: {names(missing)}; repeated variables: {names(duplicates)}; "
                f"unknown variables: {names(unknown)}. " + instructions)
        for group in ordered_lists:
            if [key for key in ids if key in set(group)] != list(group):
                raise ValueError("internal ranking merge changed the prior order; preserve this sequence: " + names_in_order(group, display))
        cleaned = []
        for row in rows:
            reason, key = row.get("rationale"), row["feature_id"]
            if not isinstance(reason, str) or not reason.strip() or len(reason) > 4000:
                raise ValueError(f"rationale for {display[key]!r} must be nonempty text of at most 4000 characters")
            available = {r["evidence_id"] for r in by_id[key]["modeling_evidence"] if r["role"] == "effect"}
            if "evidence_ids" in row and (not isinstance(row["evidence_ids"], list)
                    or any(not isinstance(x, str) for x in row["evidence_ids"])
                    or not set(row["evidence_ids"]) <= available):
                raise ValueError("ranking provenance must use its own supplied effect evidence")
            cleaned.append({"feature_id": key, "rationale": reason.strip(),
                "evidence_ids": [r["evidence_id"] for r in by_id[key]["modeling_evidence"] if r["role"] == "effect"],
                "evidence_attachment": "Available effect evidence attached by Python."})
        return {"ranking": cleaned}
    validate.repair_instructions = instructions
    if cards:
        validate.response_format = _response_format("modifier_ranking", {
            "ordered_groups": {"type": "array", "minItems": 1, "maxItems": len(cards),
                "items": {"type": "object", "additionalProperties": False,
                    "required": ["features", "rationale"], "properties": {
                        "features": {"type": "array", "minItems": 1, "maxItems": len(cards),
                            "items": {"type": "string", "enum": list(labels)}},
                        "rationale": _RATIONALE_SCHEMA}}}})
    return validate


def names_in_order(keys, display):
    return json.dumps([display.get(key, key) for key in keys], ensure_ascii=False)


def _pair_validator(cards):
    labels = _card_labels(cards)
    if len(cards) != 2:
        raise ValueError("a modifier comparison requires exactly two clinical variables")
    def validate(response):
        if not isinstance(response, Mapping):
            raise ValueError("return an object with preferred_feature and rationale")
        reason = response.get("rationale")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 4000:
            raise ValueError("explain the modifier comparison and uncertainty")
        if "preferred_feature_id" in response:
            key = response["preferred_feature_id"]
        else:
            if set(response) != {"preferred_feature", "rationale"}:
                raise ValueError("return preferred_feature and rationale")
            key = None if response["preferred_feature"] is None else _resolve(response["preferred_feature"], labels)
        if key is not None and (not isinstance(key, str) or key not in labels.values()):
            raise ValueError("choose one of the two supplied variables, or null for a tie")
        return {"preferred_feature_id": key, "rationale": reason.strip()}
    validate.repair_instructions = ('Return exactly preferred_feature and rationale. '
        'preferred_feature must be one of these complete names, or null only when the evidence '
        'supports a tie: ' + json.dumps(list(labels), ensure_ascii=False) + '. '
        'rationale must explain the comparison and uncertainty in 1 through 4000 characters.')
    validate.response_format = _response_format("modifier_comparison", {
        "preferred_feature": {"type": ["string", "null"], "enum": [*labels, None]},
        "rationale": _RATIONALE_SCHEMA})
    return validate


def rank_modifier_candidates(*, definitions, statistical_report, request_json, output_dir,
    role_policy, maximum, maximum_chars, model_identity, compatible_output_dirs=()):
    evidence = build_multi_model_role_evidence(definitions=definitions, statistical_report=statistical_report, policy=role_policy)
    locked = {str(f["feature_id"]) for f in definitions if f.get("configured_explicit_feature") is True}
    cards = [c for c in evidence["candidates"] if c["feature_id"] not in locked]
    by_id = {c["feature_id"]: c for c in cards}
    directory = Path(output_dir)
    identity = {"version": RANKING_VERSION, "prompt_version": clinical_prompts.PROMPT_VERSION,
        "evidence": _fingerprint(evidence), "model": model_identity, "role_policy": role_policy.public_dict(),
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}
    compatible_identities = [{**identity, "source_sha256": value} for value in COMPATIBLE_SOURCE_HASHES]
    def messages(batch, slug="18_rank_modifiers"):
        return clinical_prompts.messages(slug, clinical_prompts.evidence_input(evidence, batch))
    requests, recoveries = 0, {}
    def request(batch, *, pair=False, allow_split=True):
        nonlocal requests
        from .plain_handoff_stage2_analysis import Stage2ResponseValidationError
        rendered = messages(batch, "19_merge_rankings" if pair else "18_rank_modifiers")
        if not allow_split:
            # A distinct prompt/cache key prevents a failed two-item batch from
            # recursively re-entering itself through its pairwise merge fallback.
            rendered[-1] = {**rendered[-1], "content": rendered[-1]["content"]
                + "\n\nCompare these two variables using ordered_groups: one group with both "
                "names for a supported tie, or two single-name groups in preferred order. "
                "Each features field must be an array; provide a rationale for each group."}
        key = _fingerprint({"identity": identity, "messages": rendered})
        requests += 1
        destination = directory / "requests" / key[:20]
        # The parent count directory is source-versioned too. Reuse only exact
        # prompts/evidence from explicitly compatible predecessors; never scan
        # arbitrary runs or trust a response solely because its path matches.
        for root in (directory, *map(Path, compatible_output_dirs)):
            for previous in (identity, *compatible_identities):
                old_key = _fingerprint({"identity": previous, "messages": rendered})
                candidate = root / "requests" / old_key[:20]
                marker = candidate / "complete.json"
                if marker.exists() and (candidate / "response.json").exists():
                    expected = review_fingerprint({"messages": rendered, "identity": previous,
                        "version": clinical_prompts.PROMPT_VERSION})
                    if json.loads(marker.read_text()).get("input_fingerprint") == expected:
                        destination = candidate
                        break
            else:
                continue
            break
        recovery_path = directory / "requests" / key[:20] / "recovery.json"
        saved = json.loads(recovery_path.read_text()) if recovery_path.exists() else {}
        if saved.get("input_fingerprint") == key:
            recoveries[key] = saved
            recoveries.update(saved.get("child_recoveries", {}))
        used_recovery = False

        def recover_request(messages, validate, **kwargs):
            nonlocal used_recovery
            if not allow_split:
                return request_json(messages, validate, **kwargs)
            record = recoveries.get(key)
            if record is None:
                try:
                    return request_json(messages, validate, **kwargs)
                except Stage2ResponseValidationError as exc:
                    if not allow_split:
                        raise
                    record = {"input_fingerprint": key, "status": "started", "error": str(exc),
                        "strategy": "alternate_pair_format" if pair else "split_batch",
                        "candidate_ids": [c["feature_id"] for c in batch]}
                    _write_json(recovery_path, record)
            used_recovery = True
            before = requests
            if pair:
                # A second response format still requires an actual clinical
                # comparison. If it also fails, propagate the error, not a tie.
                ordered = request(batch, allow_split=False)["ranking"]
                response = {"preferred_feature_id": ordered[0]["feature_id"],
                    "rationale": ordered[0]["rationale"]}
            elif len(batch) == 1:
                response = {"ranking": [{"feature_id": batch[0]["feature_id"],
                    "rationale": "Only candidate in this sub-batch; relative placement is determined by subsequent clinical comparisons."}]}
                record = {**record, "strategy": "single_candidate"}
            else:
                middle = len(batch) // 2
                left = request(batch[:middle])["ranking"]
                right = request(batch[middle:])["ranking"]
                response = {"ranking": merge(left, right, limit=len(batch))}
            response = validate(response)
            record = {**record, "status": "complete", "additional_requests": requests - before,
                "child_recoveries": {k: v for k, v in recoveries.items() if k not in before_recoveries}}
            _write_json(recovery_path, record)
            recoveries[key] = record
            return response

        before_recoveries = set(recoveries)
        response = request_review(destination, rendered,
            _pair_validator(batch) if pair else _ranking_validator(batch),
            request_json=recover_request, identity=identity, maximum_chars=maximum_chars,
            compatible_identities=compatible_identities)
        if not used_recovery and key in recoveries:
            requests += recoveries[key].get("additional_requests", 0)
        return response
    batches = _bounded_batches(cards, messages, maximum_items=max(2, role_policy.max_candidates_per_request), maximum_chars=maximum_chars)
    def merge(left, right, *, limit):
        result, a, b = [], 0, 0
        while a < len(left) and b < len(right) and len(result) < limit:
            keys = [left[a]["feature_id"], right[b]["feature_id"]]
            comparison = request([by_id[key] for key in keys], pair=True)
            key = comparison["preferred_feature_id"]
            if key is None:
                key = min(keys, key=lambda k: (clinical_prompts.label(by_id[k]["definition"]).casefold(), k))
            if key == keys[0]:
                result.append(left[a]); a += 1
            else:
                result.append(right[b]); b += 1
        return (result + left[a:] + right[b:])[:limit]
    lists = [request(batch)["ranking"][:maximum] for batch in batches]
    while len(lists) > 1:
        lists = [merge(lists[i], lists[i + 1], limit=maximum) if i + 1 < len(lists) else lists[i] for i in range(0, len(lists), 2)]
    ranking = lists[0] if lists else []
    if len(ranking) != min(maximum, len(cards)):
        raise RuntimeError("ranking merge omitted clinical variables")
    result = {"schema_version": RANKING_VERSION, "ranking": ranking, "reviewed_candidate_ids": list(by_id),
              "locked_feature_ids": sorted(locked), "identity": identity, "maximum": maximum,
              "comparisons_and_batch_requests": requests,
              **({"validation_recoveries": recoveries} if recoveries else {}),
              "tie_policy": "clinical_name_then_internal_identity", "merge_policy": "pairwise_leading_candidates"}
    _write_json(directory / "ranking.json", result)
    return result
