"""Final estimation concurrency, numerical equivalence, honesty and resume."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
import threading
import time

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from oci.inference import plain_handoff_stage2_analysis as analysis
from oci.inference import stage2_effect_estimators as effects
from oci.inference import stage2_final_modeling as final
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


@pytest.mark.parametrize("binary", [False, True])
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


def test_checkpoint_does_not_record_failed_fit_or_accept_corruption(tmp_path):
    def fail():
        raise ValueError("fit failed")

    with pool.modeling_session(2):
        with pytest.raises(ValueError, match="fit failed"):
            final.prediction_future(tmp_path, "p", "fp", fail, 2).result()
        assert not (tmp_path / "p.json").exists()
        with pytest.raises(ValueError, match="finite"):
            final.prediction_future(tmp_path, "p", "fp", lambda: [0, np.nan], 2).result()
        assert not (tmp_path / "p.json").exists()
        assert final.prediction_result(
            final.prediction_future(tmp_path, "p", "fp", lambda: [0.2, 0.8], 2), 2
        ).tolist() == [0.2, 0.8]
        value = json.loads((tmp_path / "p.json").read_text())
        value["result"]["prediction"][0] = 0.9
        (tmp_path / "p.json").write_text(json.dumps(value))
        with pytest.raises(ValueError, match="corrupt"):
            final.prediction_future(tmp_path, "p", "fp", fail, 2)


def test_final_resume_reuses_nuisances_after_effect_fit_failure(tmp_path, monkeypatch):
    kw = inputs()
    kw.update(estimator="linear_interactions", statistical_policy=small_policy())
    original = effects.fit_interaction_effect

    def fail(**kwargs):
        raise RuntimeError("effect fit interrupted")

    monkeypatch.setattr(effects, "fit_interaction_effect", fail)
    with pool.modeling_session(2):
        with pytest.raises(RuntimeError, match="effect fit interrupted"):
            analysis.estimate_outer_fold(**kw, **candidates(kw), output_dir=tmp_path)
    assert not (tmp_path / "complete.json").exists()
    snapshots = {p: p.read_bytes() for p in tmp_path.glob("fit_checkpoints/*/**/*.json")}
    assert len(snapshots) == 12
    monkeypatch.setattr(effects, "fit_interaction_effect", original)

    def forbidden(*args, **kwargs):
        raise AssertionError("completed nuisance fit recomputed")

    monkeypatch.setattr(analysis, "_fit_nuisance_prediction", forbidden)
    with pool.modeling_session(1):
        analysis.estimate_outer_fold(**kw, **candidates(kw), output_dir=tmp_path)
    assert (tmp_path / "complete.json").exists()
    assert {p: p.read_bytes() for p in snapshots} == snapshots


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


@pytest.mark.parametrize("binary", [False, True])
def test_parallel_interaction_paths_preserve_predictions_and_warm_starts(tmp_path, binary):
    kw = inputs(binary)
    fit = dict(
        train=kw["extracted_fit"],
        valid=kw["extracted_heldout"],
        definitions=kw["definitions"],
        modifier_ids=["x"],
        treatment=kw["dataset"].treatment[:72].to_numpy(),
        outcome=kw["dataset"].outcome[:72].to_numpy(),
        binary=binary,
        policy=small_policy(),
        seed=21,
    )
    expected = effects.fit_interaction_effect(**fit)
    with pool.modeling_session(2):
        actual = effects.fit_interaction_effect(
            **fit, path_submit=final.penalty_path_submitter(tmp_path, "fp")
        )
    assert actual["audit"] == expected["audit"]
    np.testing.assert_array_equal(actual["tau"], expected["tau"])
    paths = list(tmp_path.glob("interaction_cv/*.json"))
    assert len(paths) == 2

    def cached(index, compute):
        def forbidden():
            raise AssertionError("completed penalty path was recomputed")

        return final.penalty_path_submitter(tmp_path, "fp")(index, forbidden)

    again = effects.fit_interaction_effect(**fit, path_submit=cached)
    assert again["audit"] == actual["audit"]


@pytest.mark.parametrize("estimator", ["linear_interactions", "causal_forest", "dina"])
def test_real_final_estimation_parallel_equivalence_and_complete_resume(
    tmp_path, monkeypatch, estimator
):
    kw = inputs()
    kw.update(estimator=estimator, statistical_policy=replace(small_policy(), max_iter=3000))
    frames, reports = [], []
    for workers in (1, 2):
        destination = tmp_path / str(workers)
        with pool.modeling_session(workers):
            reports.append(
                analysis.estimate_outer_fold(**kw, **candidates(kw), output_dir=destination)
            )
        frames.append(pd.read_csv(destination / "predictions.csv"))
        assert len(list(destination.glob("fit_checkpoints/*/cross_fit/fold_*/*.json"))) == 9
        assert len(list(destination.glob("fit_checkpoints/*/full_training/*.json"))) == 3
    pd.testing.assert_frame_equal(frames[0], frames[1], rtol=1e-12, atol=1e-12)
    if estimator == "causal_forest":
        for jobs, report in zip((1, 2), reports):
            assert report["causal_forest_fit_audit"]["operational_parameters"]["n_jobs"] == jobs

    def forbidden(*args, **kwargs):
        raise AssertionError("completed estimation was recomputed")

    monkeypatch.setattr(analysis, "_cross_fitted_nuisance", forbidden)
    with pool.modeling_session(3):
        assert analysis.estimate_outer_fold(
            **kw, **candidates(kw), output_dir=tmp_path / "2"
        ) == reports[1]


def _timed_work():
    start = time.monotonic()
    time.sleep(0.08)
    return start, time.monotonic(), os.getpid()


def test_multi_cpu_reservation_is_not_starved_by_single_fit_jobs():
    with pool.modeling_session(3):
        budget = pool._pool
        budget.acquire(3)

        def wait_for_queue_size(size):
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                with budget.capacity:
                    if len(budget.waiters) == size:
                        return
                time.sleep(0.01)
            pytest.fail("CPU reservation was not queued")

        with ThreadPoolExecutor(2) as executor:
            forest = executor.submit(budget.acquire, 2)
            wait_for_queue_size(1)
            single = executor.submit(budget.acquire)
            wait_for_queue_size(2)
            budget.release(1)
            # A single free CPU cannot bypass the older two-CPU reservation.
            time.sleep(0.05)
            assert not forest.done() and not single.done()
            budget.release(1)
            assert forest.result(timeout=10) == 2
            assert not single.done()
            budget.release(2)
            assert single.result(timeout=10) == 1
            budget.release(2)
        assert budget.available == 3


def test_forest_threads_and_process_jobs_share_one_budget_and_release_on_error():
    started = threading.Event()
    release = threading.Event()
    reservations = []

    def forest():
        with pool.threaded_fit() as jobs:
            reservations.append((jobs, time.monotonic()))
            started.set()
            assert release.wait(20)
        reservations.append(time.monotonic())

    with pool.modeling_session(3, parallel_fits=2):
        with ThreadPoolExecutor(2) as executor:
            forest_future = executor.submit(forest)
            assert started.wait(20)
            assert reservations[0][0] == 1  # floor(3 / 2) per concurrent forest
            jobs = [pool.submit(_timed_work) for _ in range(4)]
            rows = [job.result() for job in jobs]
            release.set()
            forest_future.result()
        events = sorted([(row[0], 1) for row in rows] + [(row[1], -1) for row in rows])
        active = peak = 0
        for _, change in events:
            active += change
            peak = max(peak, active)
        # One CPU remains reserved for the threaded forest. Worker processes can
        # start slowly enough that the short jobs never overlap, so only the
        # upper bound is deterministic.
        assert 1 <= peak <= 2
        assert all(row[2] != os.getpid() for row in rows)
        assert pool._pool.available == 3
        with pytest.raises(RuntimeError, match="failed"):
            with pool.threaded_fit():
                raise RuntimeError("failed")
        assert pool._pool.available == 3
        assert pool.call(lambda: 7) == 7
