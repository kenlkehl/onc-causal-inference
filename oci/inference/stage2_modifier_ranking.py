"""Deterministic fusion of training-only marginal and joint modifier evidence.

Ranking never calls an LLM. Scores describe predictive support, not calibrated
probabilities or independent statistical replications. Numerical caches remain
owned by stage2_multi_model_selection; legacy LLM rankings are incompatible.
"""
from bisect import bisect_left, bisect_right
from collections import defaultdict
from hashlib import sha256
import json
import logging
import math
from pathlib import Path
from statistics import median

from .stage2_role_adjudication import _fingerprint, _write_json

LOGGER = logging.getLogger(__name__)
RANKING_VERSION = "stage2_modifier_ranking_v3_statistical_fusion"
SOURCES = {
    "univariable_rlearner": "individual_effect_gain",
    "dina": "joint_dina_permutation",
    "causal_forest": "joint_forest_permutation",
}
JOINT_SOURCES = {"dina", "causal_forest"}
SELECTION_FAMILIES = {"orthogonal_linear", "penalized_interactions"}


def _finite(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def _mean(values):
    return math.fsum(sorted(values)) / len(values)


def _inputs(definitions, report):
    ids = [f["feature_id"] for f in definitions]
    if any(not isinstance(k, str) or not k for k in ids) or len(set(ids)) != len(ids):
        raise ValueError("ranking requires unique nonempty feature identities")
    cells = report.get("cells")
    if not isinstance(cells, list) or (ids and not cells):
        raise ValueError("statistical ranking requires raw numerical cells, not LLM summaries")
    families = report.get("model_families", sorted({c["family"] for c in cells}))
    sources = [family for family in SOURCES if family in families]
    if ids and not sources:
        raise ValueError("statistical ranking requires a configured predictive effect source")
    folds = sorted({c["inner_fold"] for c in cells})
    repeats = report.get("policy", {}).get("multi_model", {}).get("repeats")
    if repeats is None:
        repeats = 1 + max((c["repeat"] for c in cells), default=0)
    if type(repeats) is not int or repeats < 1:
        raise ValueError("ranking requires a positive planned repeat count")
    known = set(ids)
    rows, seen = [], set()
    for cell in cells:
        family, fold, repeat = cell["family"], cell["inner_fold"], cell["repeat"]
        if family not in {*sources, *SELECTION_FAMILIES, "univariable"}:
            continue
        if type(repeat) is not int or not 0 <= repeat < repeats:
            raise ValueError("numerical repeat is outside the planned ranking schedule")
        for row in cell["records"]:
            key = row["feature_id"]
            if key not in known or row["role"] != "effect":
                continue
            slot = (key, family, fold, repeat)
            if slot in seen:
                raise ValueError("duplicate feature exposure in a ranking fold/repeat")
            seen.add(slot)
            rows.append({"feature_id": key, "family": family, "fold": fold,
                "repeat": repeat, **{name: row.get(name) for name in
                    ("status", "score", "model_gain", "selected", "q_value")}})
    rows.sort(key=lambda r: (r["feature_id"], r["family"], r["fold"], r["repeat"]))
    return sources, folds, repeats, rows


def _component(rows, folds, repeats, *, joint):
    observed, raw, gated = defaultdict(list), defaultdict(list), 0
    for row in rows:
        score, model_gain = row["score"], row["model_gain"]
        if row["status"] != "ok" or not _finite(score) or (joint and not _finite(model_gain)):
            continue
        raw[row["fold"]].append(float(score))
        # A nonpredictive joint model cannot supply positive feature support.
        # Preserve negative permutation evidence in the fold summary.
        qualified = float(score)
        if joint and model_gain <= 0:
            qualified = min(qualified, 0.0)
            gated += 1
        observed[row["fold"]].append(qualified)
    fold_rows = [{"inner_fold": fold, "planned": repeats,
        "evaluated": len(observed[fold]),
        "raw_mean_gain": _mean(raw[fold]) if raw[fold] else None,
        "qualified_mean_gain": _mean(observed[fold]) if observed[fold] else None}
        for fold in folds]
    gains = [r["qualified_mean_gain"] for r in fold_rows if r["evaluated"]]
    evaluated = sum(r["evaluated"] for r in fold_rows)
    planned = len(folds) * repeats
    return {"status": "evaluated" if evaluated else "missing",
        "median_fold_gain": median(gains) if gains else None,
        "fold_consistency": sum(g > 0 for g in gains) / len(gains) if gains else 0.0,
        "evaluation_coverage": evaluated / planned if planned else 0.0,
        "evaluated": evaluated, "planned": planned,
        "nonpredictive_joint_model_fits": gated, "folds": fold_rows}


def _selection_stability(rows):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if (row["family"] in SELECTION_FAMILIES and row["status"] == "ok"
                and type(row["selected"]) is bool):
            grouped[row["family"]][row["fold"]].append(float(row["selected"]))
    values = [_mean([_mean(repeats) for repeats in folds.values()])
              for folds in grouped.values()]
    return _mean(values) if values else 0.0


def _compute(definitions, report, maximum, sources, folds, repeats, rows, identity):
    locked = {f["feature_id"] for f in definitions if f.get("configured_explicit_feature") is True}
    ids = sorted(f["feature_id"] for f in definitions if f["feature_id"] not in locked)
    by_key = defaultdict(list)
    for row in rows:
        by_key[row["feature_id"]].append(row)
    candidates = {}
    for key in ids:
        component = {family: _component([r for r in by_key[key] if r["family"] == family],
            folds, repeats, joint=family in JOINT_SOURCES) for family in sources}
        qs = [r["q_value"] for r in by_key[key] if r["family"] == "univariable"
              and r["status"] == "ok" and _finite(r["q_value"])]
        candidates[key] = {"feature_id": key, "components": component,
            "joint_selection_stability": _selection_stability(by_key[key]),
            "median_interaction_q": median(qs) if qs else None,
            "diagnostic_evidence": report.get("multi_model_evidence", {}).get(key, [])}
    for family in sources:
        positive = sorted(c["components"][family]["median_fold_gain"] for c in candidates.values()
            if c["components"][family]["median_fold_gain"] is not None
            and c["components"][family]["median_fold_gain"] > 0)
        for candidate in candidates.values():
            component = candidate["components"][family]
            gain = component["median_fold_gain"]
            percentile = ((bisect_left(positive, gain) + bisect_right(positive, gain))
                          / (2 * len(positive))) if positive and gain is not None and gain > 0 else 0.0
            component["positive_percentile"] = percentile
            component["support_score"] = (percentile * component["fold_consistency"]
                                            * component["evaluation_coverage"])
    tied = defaultdict(list)
    for key, candidate in candidates.items():
        scores = [c["support_score"] for c in candidate["components"].values()]
        score = 0.5 * max(scores) + 0.5 * _mean(scores)
        candidate["score"] = score
        candidate["evidence_ids"] = [f"multi:{key}:{family}:effect" for family in sources
                                     if candidate["components"][family]["evaluated"]]
        candidate["rationale"] = (f"Statistical support score {score:.6g}; " + "; ".join(
            f"{SOURCES[f]}={c['support_score']:.6g} ({c['evaluated']}/{c['planned']} fits)"
            for f, c in candidate["components"].items())
            + ". Scores combine within-method percentile, fold consistency, and coverage; they are not significance probabilities.")
        tied[(score, candidate["joint_selection_stability"])].append(candidate)
    ordered = []
    for group_key in sorted(tied, reverse=True):
        group = tied[group_key]
        # Applying a q-value only pairwise when both are present could create
        # nontransitive comparisons. Use it only for a fully comparable tie group.
        comparable = all(c["median_interaction_q"] is not None for c in group)
        ordered.extend(sorted(group, key=lambda c:
            ((c["median_interaction_q"] if comparable else 0.0), c["feature_id"])))
    for rank, candidate in enumerate(ordered, 1):
        candidate["rank"] = rank
    ranking = [{name: c[name] for name in ("feature_id", "rank", "score", "rationale", "evidence_ids")}
               for c in ordered[:maximum]]
    return {"schema_version": RANKING_VERSION, "method": "statistical_evidence_fusion",
        "identity": identity, "ranking": ranking, "candidate_scores": ordered,
        "reviewed_candidate_ids": ids, "locked_feature_ids": sorted(locked), "maximum": maximum,
        "comparisons_and_batch_requests": 0,
        "criterion": {"sources": {f: SOURCES[f] for f in sources},
            "aggregation": "mean_repeats_within_fold_then_median_folds",
            "percentile": "midrank_among_strictly_positive_medians_within_source",
            "component": "positive_percentile * fold_consistency * evaluation_coverage",
            "combination": "0.5 * max(components) + 0.5 * mean(components)",
            "joint_model_gate": "positive_support_requires_model_gain_above_zero",
            "missing": "zero_demonstrated_support_with_explicit_missing_status_no_reweighting",
            "tie_policy": "joint_selection_stability_then_comparable_interaction_q_then_feature_id",
            "formal_error_guarantee": False}}


def rank_modifier_candidates(*, definitions, statistical_report, output_dir, maximum,
    request_json=None, role_policy=None, maximum_chars=None, model_identity=None,
    compatible_output_dirs=()):
    """Rank numerical cells; LLM arguments are accepted for caller compatibility only."""
    if type(maximum) is not int or maximum < 0:
        raise ValueError("ranking maximum must be a nonnegative integer")
    sources, folds, repeats, rows = _inputs(definitions, statistical_report)
    identity = {"version": RANKING_VERSION, "maximum": maximum,
        "numerical_input_fingerprint": statistical_report.get("input_fingerprint"),
        "definitions": _fingerprint(sorted(definitions, key=lambda f: f["feature_id"])),
        "evidence": _fingerprint({"sources": sources, "folds": folds, "repeats": repeats,
            "rows": rows, "diagnostics": statistical_report.get("multi_model_evidence", {})}),
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}
    directory = Path(output_dir)
    path, marker = directory / "ranking.json", directory / "complete.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved.get("input_fingerprint") == _fingerprint(identity):
            if not path.exists():
                raise ValueError("corrupt statistical ranking checkpoint: missing ranking")
            cached = json.loads(path.read_text())
            if (saved.get("result_sha256") != _fingerprint(cached)
                    or cached.get("identity") != identity):
                raise ValueError("corrupt statistical ranking checkpoint: hash does not match")
            return cached
    result = _compute(definitions, statistical_report, maximum, sources, folds, repeats, rows, identity)
    _write_json(path, result)
    _write_json(marker, {"input_fingerprint": _fingerprint(identity), "result_sha256": _fingerprint(result)})
    LOGGER.info("Stage 2 statistical modifier ranking: %s candidates, %s sources, %s retained; zero LLM requests",
                len(result["candidate_scores"]), len(sources), len(result["ranking"]))
    return result
