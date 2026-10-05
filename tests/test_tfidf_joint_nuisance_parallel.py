from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import numpy as np
import pytest

from oci.config import (
    BoWViewConfig,
    TfidfNuisanceStackScientificConfig,
)
from oci.inference import tfidf_topic_discovery as discovery
from oci.inference.tfidf_safe_artifacts import (
    safe_artifact_content_sha256,
    write_fitted_topic_context,
)
from oci.inference.tfidf_topic_discovery import (
    FittedTopicContext,
    _strata,
    fit_joint_cross_fitted_nuisance_stacks,
    stable_hash,
)


def _inputs() -> dict:
    row_count = 48
    treatment = np.asarray(
        [index % 2 for index in range(row_count)],
        dtype=float,
    )
    outcome = np.asarray(
        [(index // 2) % 2 for index in range(row_count)],
        dtype=float,
    )
    long_prefix = "paddingword " * 1500
    texts = [
        (
            f"{long_prefix}sentinelafterfourteenthousand "
            f"patient_{index} arm_{int(treatment[index])} "
            f"outcome_{int(outcome[index])} cycle_{index % 6}"
        )
        for index in range(row_count)
    ]
    return {
        "texts": texts,
        "treatment": treatment,
        "outcome": outcome,
        "outcome_binary": True,
        "strata": _strata(
            treatment,
            outcome,
            outcome_binary=True,
        ),
        "views": [
            BoWViewConfig(
                name="linear_1_2",
                min_df=1,
                max_df=1.0,
                max_features=256,
                ngram_range_min=1,
                ngram_range_max=2,
                bow_model="linear",
            )
        ],
        "folds": 3,
        "random_state": 71,
        "nuisance_stack_config": (
            TfidfNuisanceStackScientificConfig()
        ),
    }


def _safe_stack_content_sha256(result: dict, root: Path) -> str:
    fitted = FittedTopicContext(
        common_vectorizer=(
            result["treatment"]["fitted"].base_models[0][0]
        ),
        treatment_stack=result["treatment"]["fitted"],
        outcome_stack=result["outcome"]["fitted"],
        topic_banks={},
        config_hash=stable_hash(
            {
                "treatment": result["treatment"]["fitted"].config_hash,
                "outcome": result["outcome"]["fitted"].config_hash,
            }
        ),
    )
    return safe_artifact_content_sha256(
        write_fitted_topic_context(fitted, root)
    )


def _intervals_overlap(first: dict, second: dict) -> bool:
    return (
        max(
            int(first["started_monotonic_ns"]),
            int(second["started_monotonic_ns"]),
        )
        < min(
            int(first["finished_monotonic_ns"]),
            int(second["finished_monotonic_ns"]),
        )
    )


@pytest.mark.filterwarnings("ignore:'multi_class' was deprecated:FutureWarning")
def test_serial_and_loky_top_level_folds_are_exact_and_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    mpl_cache = tmp_path / "matplotlib"
    mpl_cache.mkdir()
    monkeypatch.setenv("MPLCONFIGDIR", str(mpl_cache))
    inputs = _inputs()
    serial_attestations: list[dict] = []
    process_attestations: list[dict] = []

    serial = fit_joint_cross_fitted_nuisance_stacks(
        **inputs,
        tfidf_workers=1,
        tfidf_parallel_backend="threads",
        owner_cpu_budget=1,
        operational_attestation_sink=serial_attestations.append,
    )
    parallel = fit_joint_cross_fitted_nuisance_stacks(
        **inputs,
        tfidf_workers=2,
        owner_cpu_budget=2,
        operational_attestation_sink=process_attestations.append,
    )

    assert set(serial) == set(parallel) == {"treatment", "outcome", "mu0", "mu1"}
    for target in ("treatment", "outcome", "mu0", "mu1"):
        assert np.array_equal(
            serial[target]["base_oof"],
            parallel[target]["base_oof"],
        )
        assert np.array_equal(
            serial[target]["stacked_oof"],
            parallel[target]["stacked_oof"],
        )
        assert np.array_equal(
            serial[target]["fold_ids"],
            parallel[target]["fold_ids"],
        )
        assert (
            serial[target]["fit_positions_by_row"]
            == parallel[target]["fit_positions_by_row"]
        )
        serial_external, serial_views = serial[target]["fitted"].predict(
            ["sentinelafterfourteenthousand heldout cycle_2"]
        )
        parallel_external, parallel_views = parallel[target][
            "fitted"
        ].predict(
            ["sentinelafterfourteenthousand heldout cycle_2"]
        )
        assert np.array_equal(serial_external, parallel_external)
        assert serial_views.keys() == parallel_views.keys()
        for view_name in serial_views:
            assert np.array_equal(
                serial_views[view_name],
                parallel_views[view_name],
            )
        for row, fit_positions in enumerate(
            parallel[target]["fit_positions_by_row"]
        ):
            assert fit_positions
            assert row not in fit_positions

    assert _safe_stack_content_sha256(
        serial,
        tmp_path / "serial_stack",
    ) == _safe_stack_content_sha256(
        parallel,
        tmp_path / "parallel_stack",
    )
    for result in (serial, parallel):
        vectorizer = result["treatment"]["fitted"].base_models[0][0]
        assert (
            "sentinelafterfourteenthousand"
            in vectorizer.vocabulary_
        )

    assert len(serial_attestations) == len(process_attestations) == 1
    attestation = process_attestations[0]
    assert attestation["configured_backend"] == "processes"
    assert attestation["joblib_backend"] == "loky"
    assert attestation["effective_workers"] == 2
    assert attestation["actual_peak_concurrent_fold_workers"] == 2
    assert attestation["fold_overlap_observed"] is True
    assert attestation["canonical_fold_order"] == [1, 2, 3]
    assert attestation["subfold_parallelism"] == 1
    assert attestation["subfold_joblib_pools_created"] is False
    assert (
        attestation["full_data_base_fits_after_fold_barrier"]
        is True
    )
    assert attestation["final_stack_fits_after_fold_barrier"] is True
    assert attestation["scientific_outputs_include_operational_metadata"] is False
    intervals = attestation["fold_intervals"]
    assert len({row["worker_pid"] for row in intervals}) >= 2
    assert any(
        _intervals_overlap(first, second)
        for index, first in enumerate(intervals)
        for second in intervals[index + 1 :]
    )
    body = {
        key: value
        for key, value in attestation.items()
        if key != "content_sha256"
    }
    assert attestation["content_sha256"] == stable_hash(body)


@pytest.mark.filterwarnings("ignore:'multi_class' was deprecated:FutureWarning")
def test_reversed_completion_merges_canonically_without_nested_pool(
    monkeypatch: pytest.MonkeyPatch,
):
    inputs = _inputs()
    inputs["texts"] = [
        text[-400:] for text in inputs["texts"]
    ]
    serial = fit_joint_cross_fitted_nuisance_stacks(
        **inputs,
        tfidf_workers=1,
        tfidf_parallel_backend="threads",
        owner_cpu_budget=1,
    )
    real_worker = discovery._fit_joint_nuisance_top_level_fold
    real_parallel = discovery.Parallel
    pool_calls: list[dict] = []

    def reverse_worker(task):
        started = time.monotonic_ns()
        time.sleep(0.12 * (4 - int(task.fold)))
        result = real_worker(task)
        return dataclasses.replace(
            result,
            started_monotonic_ns=started,
        )

    class RecordingParallel:
        def __init__(self, *args, **kwargs):
            pool_calls.append(dict(kwargs))
            self._parallel = real_parallel(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            return self._parallel(*args, **kwargs)

    monkeypatch.setattr(
        discovery,
        "_fit_joint_nuisance_top_level_fold",
        reverse_worker,
    )
    monkeypatch.setattr(discovery, "Parallel", RecordingParallel)
    attestations: list[dict] = []
    threaded = fit_joint_cross_fitted_nuisance_stacks(
        **inputs,
        tfidf_workers=3,
        tfidf_parallel_backend="threads",
        owner_cpu_budget=3,
        operational_attestation_sink=attestations.append,
    )

    assert len(pool_calls) == 1
    assert attestations[0]["completion_order"] != [1, 2, 3]
    assert attestations[0]["canonical_fold_order"] == [1, 2, 3]
    assert attestations[0]["subfold_joblib_pools_created"] is False
    for target in ("treatment", "outcome", "mu0", "mu1"):
        assert np.array_equal(
            serial[target]["base_oof"],
            threaded[target]["base_oof"],
        )
        assert np.array_equal(
            serial[target]["stacked_oof"],
            threaded[target]["stacked_oof"],
        )
        assert np.array_equal(
            serial[target]["fold_ids"],
            threaded[target]["fold_ids"],
        )


def test_parallel_plan_is_bounded_by_owner_budget_and_rejects_bad_backend():
    assert discovery._resolve_joint_nuisance_parallelism(
        tfidf_workers=8,
        tfidf_parallel_backend="loky",
        owner_cpu_budget=2,
        task_count=5,
    ) == (2, "processes", "loky", 2)
    with pytest.raises(ValueError, match="tfidf_parallel_backend"):
        discovery._resolve_joint_nuisance_parallelism(
            tfidf_workers=2,
            tfidf_parallel_backend="fork",
            owner_cpu_budget=2,
            task_count=3,
        )
