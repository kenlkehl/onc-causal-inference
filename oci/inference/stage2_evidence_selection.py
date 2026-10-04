"""Python-only strength ranking, overlap control, and clinical context lookup.

Scores are comparable only within their producer's list. Nothing in this
module asks the discovery LLM to interpret scores or decide redundancy.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .neural_query_evidence_contract import query_term_normalization


SELECTION_POLICY = "within_source_percentiles_context_mean_then_overlap_v1"


def _native_strength(
    member: Mapping[str, Any], scores: Mapping[str, float]
) -> tuple[str, float] | None:
    # Losses, pseudo-outcomes, and query-level fit statistics are not item
    # importance. In particular a strong query does not make every n-gram strong.
    if member["evidence_kind"] == "clinical_text":
        priorities = ("attention", "similarity", "importance", "signed_score", "score")
    else:
        role_score = (
            "best_abs_effect_score"
            if "residual_effect" in member["evidence_axes"]
            else "best_abs_confounder_score"
        )
        priorities = (
            "tfidf_contrast", "coefficient", "loading", "importance", role_score,
            "abs_pseudo_target_score", "pseudo_target_score", "confounder_overlap_score",
            "signed_score", "fit_signed_score", "standardized_score", "score", "probe_auc",
        )
    for key in (*priorities, "rank", "fit_rank"):
        value = scores.get(key)
        if value is None or not np.isfinite(value):
            continue
        if key in {"rank", "fit_rank"}:
            if value < 1:
                continue
            return key, -float(value)
        if key == "probe_auc":
            return key, abs(float(value) - 0.5)
        # Signed regression/contrast scores represent strength in either
        # direction; retrieval similarity is already ordered best to worst.
        return key, float(value) if key == "similarity" else abs(float(value))
    return None


def annotate_strength(members: Sequence[dict[str, Any]]) -> None:
    """Rank within source lists, then average one best rank per training context.

    Contexts where an item is absent do not count as zero. This keeps a signal
    found in only one training view eligible; recurrence breaks strength ties.
    Repeated queries within one context cannot inflate its recurrence count.
    """
    pools: dict[tuple, dict[str, float]] = defaultdict(dict)
    observations = []
    for member in members:
        for ref in member["raw_references"]:
            native = _native_strength(member, ref.get("scores") or {})
            if native is None:
                continue
            metric, value = native
            context = (ref.get("scope"), ref.get("inner_fold"))
            # Strip only the final item index, preserving query/view identity.
            path = re.sub(r"\[\d+\]$", "", str(ref.get("json_path") or ""))
            pool = (
                *context, ref.get("source"), tuple(member["source_architectures"]),
                member["evidence_kind"], path, metric,
            )
            identity = str(member["member_id"])
            pools[pool][identity] = max(value, pools[pool].get(identity, -np.inf))
            observations.append((identity, context, pool, metric))
    percentiles = {}
    for pool, values in pools.items():
        ordered = np.sort(list(values.values()))
        for identity, value in values.items():
            lower = int(np.searchsorted(ordered, value, side="left"))
            upper = int(np.searchsorted(ordered, value, side="right"))
            percentiles[pool, identity] = (lower + upper) / (2.0 * len(ordered))
    contexts: dict[str, dict[tuple, float]] = defaultdict(dict)
    metrics: dict[str, set[str]] = defaultdict(set)
    for identity, context, pool, metric in observations:
        contexts[identity][context] = max(
            contexts[identity].get(context, 0.0), percentiles[pool, identity]
        )
        metrics[identity].add(metric)
    for member in members:
        identity = str(member["member_id"])
        values = list(contexts[identity].values())
        member["selection_strength"] = {
            "mean_context_percentile": float(np.mean(values)) if values else None,
            "scored_context_count": len(values),
            "metrics": sorted(metrics[identity]),
        }


def strength_key(member: Mapping[str, Any]) -> tuple:
    score = (member.get("selection_strength") or {}).get("mean_context_percentile")
    contexts = {(ref.get("scope"), ref.get("inner_fold")) for ref in member["raw_references"]}
    return (score is not None, score or 0.0, len(contexts))


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\w+", text.casefold()))


def _contains(longer: tuple[str, ...], shorter: tuple[str, ...]) -> bool:
    return bool(shorter) and any(
        longer[start:start + len(shorter)] == shorter
        for start in range(len(longer) - len(shorter) + 1)
    )


_BOUNDARY_FILLER = {
    "the", "a", "an", "is", "was", "were", "and", "of", "with", "in", "on",
    "at", "to", "for", "patient", "patients", "has", "had", "level", "levels",
    "mg", "dl", "ml", "mmol", "room", "air",
}
_NEGATIONS = {"no", "not", "never", "without", "denies", "negative", "absent"}


def redundant(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Conservative text overlap, not clinical synonym/ontology merging."""
    if left["evidence_axes"] != right["evidence_axes"] or left["polarities"] != right["polarities"]:
        return False
    a, b = _tokens(str(left["text"])), _tokens(str(right["text"]))
    if not a or not b:
        return False
    # Preserve distinct observed values and explicit negation, even in otherwise
    # near-identical chunks. No cosine-similarity threshold declares equivalence.
    def protected(text: str) -> tuple:
        return re.findall(r"[-+]?\d+(?:\.\d+)?", text), set(_tokens(text)) & _NEGATIONS
    if protected(str(left["text"])) != protected(str(right["text"])):
        return False
    if a == b:
        return True
    if min(len(a), len(b)) < 12:
        shorter, longer = sorted((a, b), key=len)
        extras = set(longer) - set(shorter)
        return _contains(longer, shorter) and extras.issubset(_BOUNDARY_FILLER)
    # A nearly identical note template with a different lab/disease name still
    # supplies distinct evidence, even at >90% shingle overlap.
    if (set(a) ^ set(b)) - _BOUNDARY_FILLER:
        return False
    def shingles(tokens):
        return {tokens[i:i + 5] for i in range(len(tokens) - 4)}
    sa, sb = shingles(a), shingles(b)
    return len(sa & sb) / len(sa | sb) >= 0.9


def select_exemplars(
    members: Sequence[Mapping[str, Any]],
    matrix: np.ndarray | None,
    center: np.ndarray | None,
    *,
    limit: int,
) -> list[Mapping[str, Any]]:
    """Strongest first; center proximity breaks ties; skip overlapping excerpts."""
    distances = (
        np.linalg.norm(matrix - center.reshape(1, -1), axis=1)
        if matrix is not None and center is not None else np.zeros(len(members))
    )
    ordered = sorted(
        range(len(members)),
        key=lambda i: (*strength_key(members[i]), -float(distances[i]), str(members[i]["member_id"])),
        reverse=True,
    )
    selected = []
    for index in ordered:
        member = members[index]
        if any(redundant(member, existing) for existing in selected):
            continue
        selected.append(member)
        if len(selected) >= limit:
            break
    return selected


def _context_key(ref: Mapping[str, Any]) -> tuple:
    return ref.get("scope"), ref.get("inner_fold")


def _source_neighborhood(ref: Mapping[str, Any]) -> tuple | None:
    if ref.get("query_id"):
        return ("query", ref.get("query_id"), ref.get("bank"))
    contrast = re.search(r"embedding_contrast_evidence\.contrasts\[\d+\]", str(ref.get("json_path")))
    if contrast:
        return ("contrast", ref.get("source"), contrast.group())
    return None


def _normalized_word_tokens(
    text: str, policy: Mapping[str, Any]
) -> tuple[tuple[str, ...], list[tuple[int, int]]]:
    """Use sklearn's source preprocessing, keeping offsets into original text."""
    vectorizer = TfidfVectorizer(**policy)
    preprocess = vectorizer.build_preprocessor()
    normalized = preprocess(text)
    stop_words = vectorizer.get_stop_words()
    pattern = re.compile(str(policy["token_pattern"]))
    if pattern.groups > 1:
        raise ValueError("Stage 1 token_pattern can have at most one capturing group")
    # Lowercasing usually preserves positions. Accents and characters such as
    # İ or ligatures can change length, so map those transformations explicitly.
    positions = None
    if policy.get("strip_accents") or len(normalized) != len(text):
        positions = [
            index for index, char in enumerate(text) for _ in preprocess(char)
        ]
        if len(positions) != len(normalized):
            raise ValueError("cannot map Stage 1 normalization to source offsets")
    tokens, spans = [], []
    for match in pattern.finditer(normalized):
        start, end = match.span(1 if pattern.groups else 0)
        token = normalized[start:end]
        if not token or (stop_words is not None and token in stop_words):
            continue
        tokens.append(token)
        spans.append((positions[start], positions[end - 1] + 1) if positions is not None else (start, end))
    return tuple(tokens), spans


class SupportingSentenceIndex:
    """Index only supplied clinical evidence in one outer fold, never patient files.

    Neural terms use their own query's retrieved chunks; embedding terms use
    their contrast's chunks. Other terms can use chunks in the same training
    context. Lookup is deferred until representatives have been selected.
    """

    def __init__(self, members: Sequence[Mapping[str, Any]]) -> None:
        self.records = []
        self.postings: dict[str, set[int]] = defaultdict(set)
        self.normalized_indexes: dict[str, tuple] = {}
        for member in members:
            if member["evidence_kind"] != "clinical_text":
                continue
            text = str(member["text"])
            matches = list(re.finditer(r"\w+", text))
            tokens = tuple(match.group().casefold() for match in matches)
            index = len(self.records)
            self.records.append((member, tokens, [(m.start(), m.end()) for m in matches]))
            for token in set(tokens):
                self.postings[token].add(index)

    def _search_modes(self, member: Mapping[str, Any], term: str):
        # Preserve literal matching for every architecture. Neural word n-grams
        # additionally use their producer's exact stop-word/tokenization rules.
        yield self.records, self.postings, _tokens(term), member["raw_references"], {"match_method": "literal_tokens"}
        if "neural_query_moments" not in member["source_architectures"]:
            return
        policies = defaultdict(list)
        for ref in member["raw_references"]:
            explicit = ref.get("term_normalization")
            policy = dict(explicit) if isinstance(explicit, Mapping) else query_term_normalization()
            if policy.get("analyzer") != "word":
                continue  # Character n-grams retain the literal path.
            policies[json.dumps(policy, sort_keys=True), explicit is None].append(ref)
        for (key, assumed), references in policies.items():
            policy = json.loads(key)
            needle, term_spans = _normalized_word_tokens(term, policy)
            # The supplied term is already a Stage 1 feature. Do not turn an
            # invalid term like '2 mg' into just 'mg' by dropping its value.
            covered = {i for start, end in term_spans for i in range(start, end)}
            if not needle or any(not char.isspace() and i not in covered for i, char in enumerate(term)):
                continue
            if key not in self.normalized_indexes:
                records, postings = [], defaultdict(set)
                for index, (chunk, _tokens_unused, _spans_unused) in enumerate(self.records):
                    tokens, spans = _normalized_word_tokens(str(chunk["text"]), policy)
                    records.append((chunk, tokens, spans))
                    for token in set(tokens):
                        postings[token].add(index)
                self.normalized_indexes[key] = records, postings
            records, postings = self.normalized_indexes[key]
            yield records, postings, needle, references, {
                "match_method": "stage1_word_ngram",
                "term_normalization": policy,
                "normalization_defaults_assumed": assumed,
            }

    def find(self, member: Mapping[str, Any], *, limit: int = 2) -> list[dict[str, Any]]:
        if member["evidence_kind"] == "clinical_text":
            return []
        # Topic terms are already a coherent multi-term representation. Look
        # for sentences for each explicit term rather than the joined topic.
        terms = [str(member["text"])]
        if member["evidence_kind"] in {"topic", "orphan_ngram_cluster"}:
            terms = str(member["text"]).split("; ")
        output = []
        seen = set()
        for term in terms:
            for records, token_index, needle, references, match_audit in self._search_modes(member, term):
                if not needle:
                    continue
                allowed = defaultdict(set)
                for ref in references:
                    allowed[_context_key(ref)].add(_source_neighborhood(ref))
                postings = [token_index.get(token, set()) for token in set(needle)]
                candidate_ids = set.intersection(*sorted(postings, key=len))
                ordered = sorted(candidate_ids, key=lambda i: (*strength_key(records[i][0]), -i), reverse=True)
                for index in ordered:
                    chunk, tokens, spans = records[index]
                    ref = next((ref for ref in chunk["raw_references"] if (
                        _context_key(ref) in allowed and (
                            None in allowed[_context_key(ref)]
                            or _source_neighborhood(ref) in allowed[_context_key(ref)]
                        )
                    )), None)
                    if ref is None:
                        continue
                    starts = [i for i in range(len(tokens) - len(needle) + 1) if tokens[i:i + len(needle)] == needle]
                    if not starts:
                        continue
                    text = str(chunk["text"])
                    start = spans[starts[0]][0]
                    end = spans[starts[0] + len(needle) - 1][1]
                    boundaries = [0, *(m.end() for m in re.finditer(r"[.!?]\s+|\n+", text)), len(text)]
                    left = max(pos for pos in boundaries if pos <= start)
                    right = min(pos for pos in boundaries if pos >= end)
                    # Long punctuation-free source fragments stay centered on the
                    # match instead of losing it to head/tail clipping.
                    clipped = right - left > 900
                    if clipped:
                        left, right = max(left, start - 300), min(right, end + 500)
                    while left < right and text[left].isspace():
                        left += 1
                    while right > left and text[right - 1].isspace():
                        right -= 1
                    excerpt = text[left:right]
                    if excerpt in seen:
                        continue
                    seen.add(excerpt)
                    output.append({
                        "text": excerpt, "source_member_id": chunk["member_id"],
                        "reference": dict(ref), "source_char_start": left,
                        "source_char_end": right, "clipped": clipped,
                        "matched_source_char_start": start, "matched_source_char_end": end,
                        **match_audit,
                    })
                    if len(output) >= limit:
                        return output
        return output
