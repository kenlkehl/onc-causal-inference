"""Evidence fusion, dependent repeats, missingness, and versioned resume."""
from copy import deepcopy
import json

import pytest

from oci.inference import stage2_modifier_ranking as ranking


def inputs(values=None, *, repeats=3, folds=3):
    values = values or {"individual": (10., 0., 0.), "joint": (0., 10., 10.),
                        "weak": (1., 1., 1.)}
    definitions = [{"feature_id": key, "name": key, "value_type": "continuous"}
                   for key in values]
    families = list(ranking.SOURCES)
    cells = [{"family": family, "inner_fold": fold, "repeat": repeat,
              "candidate_ids": list(values), "records": [
                  {"feature_id": key, "role": "effect", "status": "ok",
                   "score": scores[index], "model_gain": .1, "selected": scores[index] > 0}
                  for key, scores in values.items()]}
             for index, family in enumerate(families)
             for fold in range(1, folds + 1) for repeat in range(repeats)]
    return definitions, {"cells": cells, "model_families": families,
        "policy": {"multi_model": {"repeats": repeats}}, "input_fingerprint": "training-only"}


def run(path, data=None, **kwargs):
    definitions, report = data or inputs()
    return ranking.rank_modifier_candidates(definitions=definitions, statistical_report=report,
        output_dir=path, maximum=kwargs.pop("maximum", 64),
        request_json=lambda *a, **k: pytest.fail("unexpected LLM request"), **kwargs)


def by_id(result):
    return {c["feature_id"]: c for c in result["candidate_scores"]}


def test_joint_only_evidence_can_outrank_marginal_evidence_without_llm(tmp_path):
    result = run(tmp_path)
    scores = by_id(result)
    assert result["ranking"][0]["feature_id"] == "joint"
    assert scores["joint"]["components"]["univariable_rlearner"]["support_score"] == 0
    assert scores["joint"]["score"] > scores["individual"]["score"]
    assert result["comparisons_and_batch_requests"] == 0
    assert not list(tmp_path.glob('requests/*'))


def test_single_joint_source_is_sufficient_and_family_units_do_not_mix(tmp_path):
    data = inputs({"forest_only": (0, 0, 10), "weak": (1, 1, 1)})
    first = run(tmp_path / 'first', data)
    assert first["ranking"][0]["feature_id"] == "forest_only"
    changed = deepcopy(data)
    scales = dict(zip(ranking.SOURCES, (10000, .0001, 3)))
    for cell in changed[1]["cells"]:
        for record in cell["records"]: record["score"] *= scales[cell["family"]]
    second = run(tmp_path / 'scaled', changed)
    assert [(c['feature_id'], c['score']) for c in first['ranking']] == [
        (c['feature_id'], c['score']) for c in second['ranking']]


def test_bad_joint_models_supply_no_positive_support(tmp_path):
    data = inputs()
    for cell in data[1]['cells']:
        if cell['family'] in ranking.JOINT_SOURCES:
            for record in cell['records']: record['model_gain'] = -.1
    result = run(tmp_path, data)
    assert result['ranking'][0]['feature_id'] == 'individual'
    for c in result['candidate_scores']:
        for family in ranking.JOINT_SOURCES:
            component = c['components'][family]
            assert component['support_score'] == 0
            assert component['nonpredictive_joint_model_fits'] == 9


def test_repeats_are_averaged_inside_folds_and_median_rejects_one_lucky_fold(tmp_path):
    data = inputs({'lucky': (0, 0, 0), 'stable': (0, 0, 0)})
    for cell in data[1]['cells']:
        for r in cell['records']:
            r['score'] = (100 if cell['inner_fold'] == 1 else -.1) if r['feature_id'] == 'lucky' else .01
    scores = by_id(run(tmp_path, data))
    c = scores['lucky']['components']['univariable_rlearner']
    assert c['median_fold_gain'] == pytest.approx(-.1) and c['fold_consistency'] == pytest.approx(1 / 3)
    assert scores['stable']['score'] > scores['lucky']['score'] == 0


def test_missing_slots_reduce_coverage_without_becoming_negative_results(tmp_path):
    data = inputs({'a': (1, 1, 1), 'b': (1, 1, 1)})
    # Remove a whole planned repeat's cell, then invalidate one remaining fit.
    data[1]['cells'] = [c for c in data[1]['cells'] if not (
        c['family'] == 'dina' and c['inner_fold'] == 1 and c['repeat'] == 2)]
    cell = next(c for c in data[1]['cells'] if c['family'] == 'dina')
    cell['records'][0].update(status='not_evaluable', score=None)
    scores = by_id(run(tmp_path, data))
    c = scores['a']['components']['dina']
    assert c['planned'] == 9 and c['evaluated'] == 7
    assert c['median_fold_gain'] == 1 and c['evaluation_coverage'] == pytest.approx(7 / 9)
    assert scores['b']['score'] > scores['a']['score']


def test_all_missing_source_remains_in_denominator(tmp_path):
    data = inputs({'a': (1, 1, 1)})
    full = by_id(run(tmp_path / 'full', data))['a']['score']
    for cell in data[1]['cells']:
        if cell['family'] == 'dina': cell['records'][0].update(status='not_evaluable', score=None)
    c = by_id(run(tmp_path / 'missing', data))['a']
    assert c['components']['dina']['status'] == 'missing' and c['score'] < full
    assert c['components']['dina']['median_fold_gain'] is None


def test_disabled_dina_for_continuous_outcomes_is_not_a_missing_vote(tmp_path):
    data = inputs()
    data[1]['model_families'].remove('dina')
    data[1]['cells'] = [c for c in data[1]['cells'] if c['family'] != 'dina']
    result = run(tmp_path, data)
    assert set(result['criterion']['sources']) == {'univariable_rlearner', 'causal_forest'}


def test_locks_budget_order_and_model_independence(tmp_path):
    data = inputs()
    data[0][0]['configured_explicit_feature'] = True
    first = run(tmp_path, data, maximum=1, model_identity='gemma')
    assert first['locked_feature_ids'] == ['individual']
    assert 'individual' not in by_id(first)
    data[0].reverse(); data[1]['cells'].reverse()
    for cell in data[1]['cells']: cell['records'].reverse()
    second = run(tmp_path, data, maximum=1, model_identity='another-model')
    assert first == second and len(first['ranking']) == 1


def test_nonpositive_and_zero_budget_candidates_still_audited(tmp_path):
    result = run(tmp_path, inputs({'b': (-1, 0, -1), 'a': (0, -1, 0)}), maximum=0)
    assert result['ranking'] == []
    assert [c['feature_id'] for c in result['candidate_scores']] == ['a', 'b']
    assert all(c['score'] == 0 for c in result['candidate_scores'])


def test_cache_integrity_and_no_legacy_llm_cache_reuse(tmp_path):
    tmp_path.joinpath('ranking.json').write_text(json.dumps({'schema_version': 'legacy', 'ranking': []}))
    first = run(tmp_path)
    assert first['schema_version'] == ranking.RANKING_VERSION
    assert run(tmp_path) == first
    path = tmp_path / 'ranking.json'; changed = json.loads(path.read_text())
    changed['ranking'][0]['score'] += .01; path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='hash does not match'): run(tmp_path)


def test_raw_evidence_changes_invalidate_ranking(tmp_path):
    data = inputs(); first = run(tmp_path, data)
    for cell in data[1]['cells']:
        for r in cell['records']: r['score'] = 10 if r['feature_id'] == 'weak' else 0
    second = run(tmp_path, data)
    assert second['ranking'][0]['feature_id'] == 'weak'
    assert first['identity'] != second['identity']


def test_foreign_diagnostics_never_enter_score_and_duplicates_rejected(tmp_path):
    data = inputs(); first = run(tmp_path / 'first', data)
    data[1]['oracle'] = {'weak': 1000}
    data[1]['cells'][0]['records'].append({'feature_id': 'foreign', 'role': 'effect', 'score': 1000})
    assert run(tmp_path / 'foreign', data)['ranking'] == first['ranking']
    data[1]['cells'][0]['records'].append(deepcopy(data[1]['cells'][0]['records'][0]))
    with pytest.raises(ValueError, match='duplicate feature exposure'): run(tmp_path / 'duplicate', data)


def test_missing_raw_cells_cannot_silently_fall_back_to_aggregate_means(tmp_path):
    definitions, report = inputs(); del report['cells']
    with pytest.raises(ValueError, match='raw numerical cells'): run(tmp_path, (definitions, report))


def test_q_tie_break_is_transitive_with_missing_values(tmp_path):
    data = inputs({'c': (1, 1, 1), 'b': (1, 1, 1), 'a': (1, 1, 1)})
    data[1]['cells'].append({'family': 'univariable', 'inner_fold': 1, 'repeat': 0,
        'records': [{'feature_id': key, 'role': 'effect', 'status': 'ok', 'q_value': q}
                    for key, q in [('a', None), ('b', .9), ('c', .01)]]})
    assert [c['feature_id'] for c in run(tmp_path / 'missing', data)['ranking']] == ['a', 'b', 'c']
    data[1]['cells'][-1]['records'][0]['q_value'] = .5
    assert [c['feature_id'] for c in run(tmp_path / 'known', data)['ranking']] == ['c', 'a', 'b']


def test_selection_stability_breaks_exact_primary_ties(tmp_path):
    data = inputs({'a': (1, 1, 1), 'b': (1, 1, 1)})
    for fold in range(1, 4):
        for repeat in range(3):
            data[1]['cells'].append({'family': 'orthogonal_linear', 'inner_fold': fold,
                'repeat': repeat, 'records': [{'feature_id': key, 'role': 'effect',
                'status': 'ok', 'selected': key == 'b'} for key in ('a', 'b')]})
    result = run(tmp_path, data)
    assert result['ranking'][0]['feature_id'] == 'b'
    assert result['ranking'][0]['score'] == result['ranking'][1]['score']


def test_optional_concept_review_uses_statistical_shortlists_and_representatives(tmp_path):
    from oci.inference import stage2_modifier_concepts as concepts
    from oci.inference import stage2_clinical_prompts as clinical
    from oci.inference.stage2_multi_model_selection import aggregate_evidence
    from oci.inference.stage2_role_adjudication import Stage2RoleAdjudicationConfig
    definitions, report = inputs({'a': (1, 1, 1), 'b': (2, 2, 2)})
    report['multi_model_evidence'] = aggregate_evidence(report['cells'], ['a', 'b'])
    calls = []
    def request(messages, validate, **kwargs):
        assert messages[0]['content'] == clinical.SYSTEM_PROMPTS['21_cross_fold_concepts']
        calls.append(messages)
        return validate({'concepts': [{'name': 'Example concept', 'members': ['a', 'b'],
            'representatives': ['b'], 'modifier_recommendation': 'retain',
            'rationale': 'Representative supported by numerical evidence.', 'unresolved_questions': ''}]})
    result = concepts.infer_modifier_concepts(definitions=definitions, statistical_report=report,
        request_json=request, output_dir=tmp_path, role_policy=Stage2RoleAdjudicationConfig(),
        top_n=100, maximum=64, maximum_chars=100000, model_identity='test')
    assert len(calls) == 1
    assert [r['feature_id'] for r in result['ranking']] == ['b']
    assert result['comparisons_and_batch_requests'] == 0
