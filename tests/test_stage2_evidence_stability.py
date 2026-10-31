"""Loss pairing, patient dependence, model-quality weighting and grouping."""


from oci.inference import stage2_evidence_stability as stability
from oci.inference import stage2_multi_model_selection as numerical
from tests.test_stage2_multi_model import sample_inputs


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
    arbitrary_ids = [{**s, 'inner_fold': 4 + i * 5} for i, s in enumerate(args['inner_splits'])]
    other = stability.repeated_partitions(arbitrary_ids, labels,
        treatment_column='treatment', outcome_column='outcome', binary=False, repeats=3, seed=42)
    assert len({s['inner_fold'] for s in other}) == 6
