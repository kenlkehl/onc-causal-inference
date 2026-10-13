"""Loss pairing, patient dependence, model-quality weighting and grouping."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from oci.inference import stage2_evidence_stability as stability
from oci.inference import stage2_modifier_ranking as ranking
from oci.inference import stage2_multi_model_selection as numerical
from oci.inference.stage2_multi_model_config import EvidenceStabilityConfig
from tests.test_stage2_multi_model import sample_inputs, feature
from tests.test_stage2_statistical_ranking import inputs, run, by_id


def test_repeated_partitions_cover_same_pool_without_access_to_other_patients():
    args = sample_inputs()
    frame, labels = numerical._validated_inputs(args['dataset'], args['extracted_fit'],
        args['definitions'], args['inner_splits'], 'treatment', 'outcome')
    splits = stability.repeated_partitions(args['inner_splits'], labels,
        treatment_column='treatment', outcome_column='outcome', binary=False, repeats=3, seed=42)
    assert len(splits) == 6
    for repeat in range(3):
        partition = [s for s in splits if s['partition_repeat'] == repeat]
        assert sorted(i for s in partition for i in s['heldout_row_ids']) == list(range(96))
        for split in partition:
            assert set(split['fit_row_ids']).isdisjoint(split['heldout_row_ids'])
            assert set(split['fit_row_ids']) | set(split['heldout_row_ids']) == set(range(96))
    assert splits[0]['heldout_row_ids'] != splits[2]['heldout_row_ids']


def test_patient_uncertainty_does_not_shrink_when_same_exposures_are_duplicated():
    rows = [{'validation_row_ids': [4, 7, 12], 'partition_repeat': 0,
             'loss_deltas': stability.pack([1., 2., 6.])}]
    once = stability.patient_summary(rows)
    repeated = stability.patient_summary(rows * 10)
    assert once == repeated
    assert once['patients'] == 3
    assert once['patient_standard_error_heuristic'] > 0
    assert not once['formal_error_guarantee']
    changed = deepcopy(rows)
    changed[0]['partition_repeat'] = 1
    changed[0]['loss_deltas'] = stability.pack([3., 4., 8.])
    assert stability.patient_summary(rows + changed)['partition_standard_deviation'] > 0


def test_joint_quality_weights_both_signs_without_editing_importance():
    rows = [{'status': 'ok', 'score': score, 'model_gain': quality,
             'fold': fold, 'validation_row_ids': [fold * 2, fold * 2 + 1],
             'partition_repeat': 0, 'loss_deltas': stability.pack([score, score]),
             'model_loss_deltas': stability.pack([quality, quality])}
            for fold, score, quality in [(1, .4, -.1), (2, -.1, .3), (3, .2, .3)]]
    result = ranking._component(rows, [1, 2, 3], 1, joint=True)
    assert result['mean_gain'] == pytest.approx(1 / 6)
    assert result['patient_summary']['mean_gain'] == pytest.approx(1 / 6)
    assert 0 < result['model_reliability'] < 1
    assert result['folds'][0]['raw_mean_gain'] == .4
    for row in rows:
        row['score'] *= -1
        row['loss_deltas'] = stability.pack([-v for v in stability.unpack(row['loss_deltas'])])
    opposite = ranking._component(rows, [1, 2, 3], 1, joint=True)
    assert opposite['mean_gain'] == -result['mean_gain']
    assert opposite['model_reliability'] == result['model_reliability']


def test_joint_contexts_remain_separate_and_individual_evidence_is_protected(tmp_path):
    definitions, report = inputs({'a': (2, 0, 1), 'b': (1, 0, .1)})
    report['policy']['multi_model']['stability'] = {'subset_partitions': 2}
    for cell in report['cells']:
        if cell['family'] in ranking.JOINT_SOURCES:
            cell['context'] = 'full' if cell['repeat'] == 0 else 'subset'
            for record in cell['records']:
                if cell['context'] == 'full':
                    record['model_gain'] = -100
        else:
            cell['context'] = 'individual'
    result = by_id(run(tmp_path, (definitions, report)))
    forest = result['a']['components']['causal_forest']
    assert set(forest['contexts']) == {'full', 'subset'}
    assert forest['contexts']['full']['model_reliability'] == 0
    assert forest['contexts']['subset']['model_reliability'] == 1
    assert forest['support_score'] > 0
    individual = result['a']['components']['univariable_rlearner']['support_score']
    assert result['a']['score'] >= individual


def test_equivalent_measurements_need_names_and_training_agreement_and_stay_together():
    x = np.linspace(-1, 1, 80)
    names = ['hemoglobin', 'hemoglobin_value', 'hemoglobin_level', 'creatinine']
    definitions = [feature(name) for name in names]
    train = pd.DataFrame(dict(zip(names, [x, x, -x, x])))
    groups = stability.measurement_groups(train, definitions, EvidenceStabilityConfig())
    assert len(groups) == 1
    assert set(groups[0]['members']) == {'hemoglobin', 'hemoglobin_value'}
    for seed in range(4):
        subsets = stability.balanced_subsets(names, groups, subset_size=2, seed=seed)
        assert sorted(k for subset in subsets for k in subset) == sorted(names)
        assert any(set(g['members']) <= set(subset) for subset in subsets for g in groups)


def test_group_permutation_measures_redundant_signal_before_selecting_representative(tmp_path):
    x = np.arange(40, dtype=float)
    matrix = np.column_stack([x, x])
    groups = [{'group_id': 'same_measurement', 'members': ['a', 'b']}]
    _, gains = stability.permutation_evidence(matrix, ['a', 'b'],
        lambda z: (x - z.mean(axis=1)) ** 2, repeats=10, seed=7, groups=groups)
    assert gains['same_measurement']['gain'] > gains['a']['gain']
    assert len(stability.unpack(gains['same_measurement']['loss_deltas'])) == 40
    data = inputs({'a': (2, 0, 0), 'b': (1, 0, 0), 'noise': (0, 0, 0)})
    data[1]['measurement_groups'] = groups
    for cell in data[1]['cells']:
        if cell['family'] in ranking.JOINT_SOURCES:
            for record in cell['records']:
                if record['feature_id'] in ['a', 'b']:
                    record['group_score'] = 10
    result = run(tmp_path, data, maximum=3)
    assert [r['feature_id'] for r in result['ranking']] == ['a', 'noise']
    assert result['measurement_representatives'][0]['representative'] == 'a'
    assert result['available_representatives'] == 2


@pytest.mark.parametrize('binary', [False, True])
def test_real_evidence_repeats_partitions_with_patient_vectors_and_resumes(tmp_path, binary, monkeypatch):
    args = sample_inputs()
    if binary:
        args['dataset']['outcome'] = (args['dataset']['outcome'] > args['dataset']['outcome'].median()).astype(int)
        args['outcome_type'] = 'binary'
    args['policy'] = replace(args['policy'], multi_model=replace(args['policy'].multi_model,
        repeats=1, stability=EvidenceStabilityConfig(partition_repeats=2, subset_partitions=2)))
    _, report, _, _ = numerical.select_stage2_features_multi_model(**args, checkpoint_dir=tmp_path)
    assert report['partition_repeats'] == 2
    assert len(report['cross_fitted_nuisance_models']['predictions']) == 96
    for cell in report['cells']:
        assert set(cell['fit_row_ids']).isdisjoint(cell['validation_row_ids'])
        if cell['family'] in ranking.SOURCES:
            for record in cell['records']:
                if record['status'] == 'ok':
                    assert len(stability.unpack(record['loss_deltas'])) == len(cell['validation_row_ids'])
    result = ranking.rank_modifier_candidates(definitions=args['definitions'], statistical_report=report,
        output_dir=tmp_path/'ranking', maximum=4)
    for candidate in result['candidate_scores']:
        for component in candidate['components'].values():
            for context in component['contexts'].values():
                if context['evaluated']:
                    assert context['patient_summary']['patients'] == 96
    monkeypatch.setattr(numerical, '_nuisances', lambda *a, **k: pytest.fail('cached evidence refitted'))
    _, cached, _, _ = numerical.select_stage2_features_multi_model(**args, checkpoint_dir=tmp_path)
    assert report == cached


@pytest.mark.parametrize('field,value', [('partition_repeats', 0), ('subset_partitions', True),
                                       ('group_min_association', float('nan')), ('group_min_overlap', 0)])
def test_invalid_stability_policy_rejected(field, value):
    with pytest.raises(ValueError):
        replace(EvidenceStabilityConfig(), **{field: value}).validate()
