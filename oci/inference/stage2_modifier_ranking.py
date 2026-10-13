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
from . import stage2_evidence_stability as stability

LOGGER = logging.getLogger(__name__)
RANKING_VERSION = "stage2_modifier_ranking_v4_stable_joint"
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
        limit = (report.get('policy', {}).get('multi_model', {}).get('stability', {}).get('subset_partitions', repeats - 1) + 1
                 if family in JOINT_SOURCES else repeats)
        if type(repeat) is not int or not 0 <= repeat < limit:
            raise ValueError("numerical repeat is outside the planned ranking schedule")
        for row in cell["records"]:
            key = row["feature_id"]
            if key not in known or row["role"] != "effect":
                continue
            slot = (key, family, fold, repeat)
            if slot in seen:
                raise ValueError("duplicate feature exposure in a ranking fold/repeat")
            seen.add(slot)
            audit = cell.get('audit', {})
            if isinstance(audit, list):
                audit = next((a for a in audit if a.get('role') == 'effect'), {})
            rows.append({"feature_id": key, "family": family, "fold": fold,
                "repeat": repeat, 'context': cell.get('context', 'legacy'),
                'partition_repeat': cell.get('partition_repeat', 0),
                'validation_row_ids': cell.get('validation_row_ids', []),
                'model_loss_deltas': audit.get('model_loss_deltas'),
                **{name: row.get(name) for name in
                    ('status', 'score', 'model_gain', 'selected', 'q_value', 'loss_deltas',
                     'measurement_group', 'group_score', 'group_loss_deltas')}})
    rows.sort(key=lambda r: (r["feature_id"], r["family"], r["fold"], r["repeat"]))
    return sources, folds, repeats, rows


def _component(rows, folds, repeats, *, joint, field='loss_deltas'):
    observed = defaultdict(list)
    valid = []
    for row in rows:
        score = row.get('group_score') if field == 'group_loss_deltas' else row.get('score')
        if row['status'] != 'ok' or not _finite(score) or (joint and not _finite(row.get('model_gain'))):
            continue
        observed[row['fold']].append(float(score))
        valid.append(row)
    fold_rows = [{'inner_fold': fold, 'planned': repeats, 'evaluated': len(observed[fold]),
                  'raw_mean_gain': _mean(observed[fold]) if observed[fold] else None}
                 for fold in folds]
    gains = [r['raw_mean_gain'] for r in fold_rows if r['evaluated']]
    summary = stability.patient_summary(valid, field)
    # Older numerical fixtures without patient vectors remain auditable, but
    # production v5 evidence always carries paired patient loss differences.
    center = summary['mean_gain'] if summary else (median(gains) if gains else None)
    noise = summary['noise_scale'] if summary else 0.0
    model_summary = stability.patient_summary(valid, 'model_loss_deltas') if joint else None
    model_gains = [r['model_gain'] for r in valid] if joint else []
    model_center = model_summary['mean_gain'] if model_summary else (_mean(model_gains) if model_gains else 0.0)
    model_noise = model_summary['noise_scale'] if model_summary else 0.0
    reliability = (max(model_center, 0.0) / (abs(model_center) + model_noise)
                   if abs(model_center) + model_noise else 0.0) if joint else 1.0
    strength = max(center, 0.0) / (abs(center) + noise) if center is not None and abs(center) + noise else 0.0
    evaluated, planned = len(valid), len(folds) * repeats
    return {'status': 'evaluated' if evaluated else 'missing',
            'mean_gain': center, 'median_fold_gain': median(gains) if gains else None,
            'fold_consistency': sum(g > 0 for g in gains) / len(gains) if gains else 0.0,
            'evaluation_coverage': evaluated / planned if planned else 0.0,
            'evaluated': evaluated, 'planned': planned, 'folds': fold_rows,
            'patient_summary': summary, 'model_patient_summary': model_summary,
            'model_mean_gain': model_center if joint else None,
            'model_reliability': reliability, 'importance_strength': strength,
            'nonpredictive_joint_model_fits': sum(r['model_gain'] <= 0 for r in valid) if joint else 0,
            'quality_policy': 'pooled_model_reliability_weights_both_signs_equally'}


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
    context_repeats = {'full': 1, 'subset': report.get('policy', {}).get('multi_model', {}).get('stability', {}).get('subset_partitions', repeats - 1),
                       'individual': repeats, 'legacy': repeats}
    family_contexts = {family: sorted({r['context'] for r in rows if r['family'] == family}) or ['legacy'] for family in sources}
    for key in ids:
        components, group_components = {}, {}
        for family in sources:
            records = [r for r in by_key[key] if r['family'] == family]
            contexts = family_contexts[family]
            for output, field in ((components, 'loss_deltas'), (group_components, 'group_loss_deltas')):
                output[family] = {'contexts': {context: _component(
                    [r for r in records if r['context'] == context], folds, context_repeats[context],
                    joint=family in JOINT_SOURCES, field=field) for context in contexts}}
        qs = [r['q_value'] for r in by_key[key] if r['family'] == 'univariable'
              and r['status'] == 'ok' and _finite(r['q_value'])]
        candidates[key] = {'feature_id': key, 'components': components, 'group_components': group_components,
            'joint_selection_stability': _selection_stability(by_key[key]),
            'median_interaction_q': median(qs) if qs else None,
            'diagnostic_evidence': report.get('multi_model_evidence', {}).get(key, [])}
    for channel in ('components', 'group_components'):
        for family in sources:
            contexts = sorted({context for c in candidates.values() for context in c[channel][family]['contexts']})
            for context in contexts:
                positive = sorted(c[channel][family]['contexts'][context]['mean_gain'] for c in candidates.values()
                                  if c[channel][family]['contexts'][context]['mean_gain'] is not None
                                  and c[channel][family]['contexts'][context]['mean_gain'] > 0)
                for candidate in candidates.values():
                    component = candidate[channel][family]['contexts'][context]
                    gain = component['mean_gain']
                    percentile = ((bisect_left(positive, gain) + bisect_right(positive, gain)) / (2 * len(positive))) if positive and gain is not None and gain > 0 else 0.0
                    component['positive_percentile'] = percentile
                    component['support_score'] = percentile * component['importance_strength'] * component['evaluation_coverage']
            for candidate in candidates.values():
                combined = candidate[channel][family]
                components = list(combined['contexts'].values())
                weight = sum(c['model_reliability'] for c in components)
                score = (sum(c['support_score'] * c['model_reliability'] for c in components) / weight
                         * max(c['model_reliability'] for c in components)) if weight else 0.0
                combined.update(status='evaluated' if any(c['evaluated'] for c in components) else 'missing',
                    support_score=score, evaluated=sum(c['evaluated'] for c in components),
                    planned=sum(c['planned'] for c in components),
                    context_aggregation='reliability_weighted_mean_times_max_reliability')
                # Preserve familiar audit fields for the single-context case.
                if len(components) == 1:
                    combined.update({k: v for k, v in components[0].items() if k != 'support_score'})
    representative_groups = report.get('measurement_groups', [])
    group_by_key = {key: g for g in representative_groups for key in g['members']}
    tied = defaultdict(list)
    for key, candidate in candidates.items():
        scores = [c["support_score"] for c in candidate["components"].values()]
        candidate['individual_measurement_score'] = max(scores) + 0.25 * _mean(scores)
        if key in group_by_key:
            scores += [c['support_score'] for c in candidate['group_components'].values()]
        score = max(scores) + 0.25 * _mean(scores)
        candidate["score"] = score
        candidate["evidence_ids"] = [f"multi:{key}:{family}:effect" for family in sources
                                     if candidate["components"][family]["evaluated"]]
        candidate["rationale"] = (f"Statistical support score {score:.6g}; " + "; ".join(
            f"{SOURCES[f]}={c['support_score']:.6g} ({c['evaluated']}/{c['planned']} fits)"
            for f, c in candidate["components"].items())
            + ". Scores combine patient-averaged gains, uncertainty, model reliability and coverage; they are not significance probabilities.")
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
    suppressed = set()
    representatives = []
    coverage = report.get('measurement_observed_fractions', {})
    for group in representative_groups:
        members = [key for key in group['members'] if key in candidates]
        if len(members) < 2 or any(key in locked for key in group['members']):
            continue
        representative = min(members, key=lambda key: (-candidates[key]['individual_measurement_score'], -coverage.get(key, 0), key))
        group_score = max(candidates[key]['score'] for key in members)
        candidates[representative]['score'] = group_score
        candidates[representative]['rationale'] += (f" Equivalent-measurement group {group['group_id']} represented by this measurement; group-supported score {group_score:.6g}.")
        suppressed.update(set(members) - {representative})
        representatives.append({**group, 'representative': representative,
                                'selection': 'individual_evidence_then_observed_coverage_then_feature_id'})
    ordered = sorted(ordered, key=lambda c: (-c['score'], c['rank']))
    for rank, candidate in enumerate(ordered, 1):
        candidate['rank'] = rank
        candidate['represented_by_equivalent_measurement'] = candidate['feature_id'] in suppressed
    shortlist = [c for c in ordered if c['feature_id'] not in suppressed]
    ranking = [{name: c[name] for name in ("feature_id", "rank", "score", "rationale", "evidence_ids")}
               for c in shortlist[:maximum]]
    return {"schema_version": RANKING_VERSION, "method": "statistical_evidence_fusion",
        "identity": identity, "ranking": ranking, "candidate_scores": ordered,
        "reviewed_candidate_ids": ids, "locked_feature_ids": sorted(locked), "maximum": maximum,
        "comparisons_and_batch_requests": 0, "measurement_representatives": representatives,
        "available_representatives": len(shortlist),
        "criterion": {"sources": {f: SOURCES[f] for f in sources},
            "aggregation": "average_exposures_per_patient_then_pool; retain_partition_sensitivity",
            "percentile": "midrank_among_strictly_positive_patient_mean_gains_within_source_and_context",
            "component": "positive_percentile * uncertainty_shrinkage * coverage; context_model_reliability",
            "combination": "max(components) + 0.25 * mean(components); protect_individual_support",
            "joint_model_gate": "pooled_model_reliability_weights_both_signs_equally",
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
        "measurement_groups": statistical_report.get('measurement_groups', []),
        "measurement_observed_fractions": statistical_report.get('measurement_observed_fractions', {}),
        "stability_policy": statistical_report.get('policy', {}).get('multi_model', {}).get('stability', {}),
        "stability_source_sha256": sha256(Path(stability.__file__).read_bytes()).hexdigest(),
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
