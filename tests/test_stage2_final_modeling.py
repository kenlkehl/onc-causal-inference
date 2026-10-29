"""Final estimation concurrency, numerical equivalence, honesty and resume."""


import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from oci.inference import plain_handoff_stage2_analysis as analysis
from oci.inference import stage2_modeling_pool as pool
from tests.test_stage2_multi_model import feature, small_policy


def inputs(binary=True):
    rng = np.random.default_rng(121)
    n = 96
    x = rng.normal(size=n)
    t = np.tile([0, 1], n // 2)
    continuous = x + t * (0.2 + x) + rng.normal(size=n)
    dataset = pd.DataFrame(
        {
            "patient_id": range(n),
            "treatment": t,
            "outcome": (continuous > 0).astype(float) if binary else continuous,
        }
    )
    extracted = pd.DataFrame({"_oci_row_id": range(n), "x": x})
    definitions = [{
        **feature("x"),
        "roles": ["confounder", "effect_modifier"],
        "nuisance_model_roles": ["treatment", "outcome"],
    }]
    split = {"fit_row_ids": list(range(72)), "heldout_row_ids": list(range(72, n))}
    split["inner_splits"] = analysis._fallback_inner_splits(range(72), folds=3, seed=9)
    return dict(
        dataset=dataset,
        extracted_fit=extracted.iloc[:72],
        extracted_heldout=extracted.iloc[72:],
        definitions=definitions,
        split=split,
        unit_id_column="patient_id",
        treatment_column="treatment",
        outcome_column="outcome",
        outcome_type="binary" if binary else "continuous",
        inner_folds=3,
        seed=5,
        propensity_clip=0.02,
        estimation_trees=8,
    )


def candidates(kw):
    """The final fit cross-fits nuisances over the unscreened candidate catalog."""
    return dict(
        nuisance_candidate_definitions=kw["definitions"],
        nuisance_candidate_fit=kw["extracted_fit"],
        nuisance_candidate_heldout=kw["extracted_heldout"],
    )


def serial_nuisances(kw):
    """Original sequential fitting order, independent of the new dispatcher."""
    result = np.full((3, len(kw["extracted_fit"])), np.nan)
    extracted = kw["extracted_fit"].set_index("_oci_row_id", drop=False)
    for index, split in enumerate(kw["split"]["inner_splits"], 1):
        fit, hold = split["fit_row_ids"], split["heldout_row_ids"]
        train, valid = extracted.loc[fit], extracted.loc[hold]
        t, y = kw["dataset"].iloc[fit][["treatment", "outcome"]].to_numpy().T
        encoder = analysis._FeatureEncoder(kw["definitions"]).fit(train)
        x, xv = encoder.transform(train), encoder.transform(valid)
        seed = kw["seed"] + index
        result[0, hold] = analysis._fit_nuisance_prediction(x, t, xv, binary=True, seed=seed)
        for arm in (0, 1):
            mask = t.astype(int) == arm
            result[1 + arm, hold] = analysis._fit_nuisance_prediction(
                x[mask], y[mask], xv, binary=kw["outcome_type"] == "binary", seed=seed + arm
            )
    return result


@pytest.mark.parametrize("binary", [True])
def test_final_nuisances_match_original_and_resume_each_fit(tmp_path, monkeypatch, binary):
    kw = inputs(binary)
    with threadpool_limits(limits=1):
        expected = serial_nuisances(kw)
    arguments = dict(
        dataset=kw["dataset"],
        extracted=kw["extracted_fit"],
        definitions=kw["definitions"],
        fit_ids=kw["split"]["fit_row_ids"],
        inner_splits=kw["split"]["inner_splits"],
        treatment_column="treatment",
        outcome_column="outcome",
        binary=binary,
        seed=kw["seed"],
        checkpoint_dir=tmp_path,
        checkpoint_fingerprint="same_inputs",
    )
    with pool.modeling_session(2):
        actual = analysis._cross_fitted_nuisance(**arguments)
    np.testing.assert_array_equal(actual, expected)
    files = list(tmp_path.glob("cross_fit/fold_*/*.json"))
    assert len(files) == 9
    snapshots = {p: p.read_bytes() for p in files}
    missing = files[0]
    missing.unlink()
    original = analysis._fit_nuisance_prediction
    calls = []

    def record(*args, **kwargs):
        calls.append(kwargs["seed"])
        return original(*args, **kwargs)

    monkeypatch.setattr(analysis, "_fit_nuisance_prediction", record)
    with pool.modeling_session(1):
        np.testing.assert_array_equal(analysis._cross_fitted_nuisance(**arguments), expected)
    assert len(calls) == 1
    assert {p: p.read_bytes() for p in files} == snapshots
    with pytest.raises(ValueError, match="incompatible"):
        analysis._cross_fitted_nuisance(**{**arguments, "checkpoint_fingerprint": "different"})


def test_final_parallel_fits_do_not_learn_from_heldout_labels(tmp_path):
    kw = inputs()
    kw.update(estimator="linear_interactions", statistical_policy=small_policy())
    results = []
    for name in ["original", "altered"]:
        with pool.modeling_session(2):
            analysis.estimate_outer_fold(**kw, **candidates(kw), output_dir=tmp_path / name)
        results.append(pd.read_csv(tmp_path / name / "predictions.csv"))
        kw["dataset"] = kw["dataset"].copy()
        kw["dataset"].loc[72:, "outcome"] = 1 - kw["dataset"].loc[72:, "outcome"]
        kw["dataset"].loc[72:, "treatment"] = 1 - kw["dataset"].loc[72:, "treatment"]
    columns = ["_oci_row_id", "propensity", "mu0", "mu1", "estimated_cate"]
    pd.testing.assert_frame_equal(results[0][columns], results[1][columns])
