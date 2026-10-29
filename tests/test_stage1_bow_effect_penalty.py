"""bow_r_loss effect penalties are cross-validated on the effect folds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from oci.config import (
    DEFAULT_BOW_EFFECT_RIDGE_ALPHA_GRID,
    AppliedInferenceConfig,
    BoWViewConfig,
    ForestScientificConfig,
    ModelArchitectureConfig,
    MultiModelForestConfig,
)
from oci.inference import multi_model_forest_stage1
from oci.inference.multi_model_forest_stage1 import MultiModelForestStage1Runner
from oci.inference.sparse_text_modeling import select_ridge_alpha
from oci.inference.stage1_checkpointing import Stage1CheckpointStore, row_id_identity
from oci.models import dina

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")

FIT_ROWS = 48


def _dataset(outcome_type: str) -> pd.DataFrame:
    rng = np.random.default_rng(23)
    vocab = [f"word{index}" for index in range(20)]
    rows = []
    for index in range(64):
        treatment = index % 2
        marker = rng.random() < 0.5
        words = list(rng.choice(vocab, size=10))
        if marker:
            words += ["egfr", "mutation"]
        rng.shuffle(words)
        signal = 0.3 + 0.4 * marker * treatment
        outcome = (
            float(signal + rng.normal(scale=0.1))
            if outcome_type == "continuous"
            else int(rng.random() < signal)
        )
        rows.append(
            {
                "clinical_text": " ".join(words),
                "treatment_indicator": treatment,
                "outcome_indicator": outcome,
            }
        )
    return pd.DataFrame(rows)


def _config(outcome_type: str, **settings) -> AppliedInferenceConfig:
    views = [
        BoWViewConfig(name="linear_1_2", bow_model="linear", ngram_range_max=2, min_df=1),
        BoWViewConfig(
            name="extratrees_1_2",
            bow_model="extratrees",
            ngram_range_max=2,
            min_df=1,
            forest_scientific=ForestScientificConfig(n_estimators=10),
        ),
    ]
    config = AppliedInferenceConfig(
        outcome_type=outcome_type,
        architecture=ModelArchitectureConfig(
            model_type="multi_model_forest",
            multi_model_forest=MultiModelForestConfig(
                feature_discovery_methods=["bow"],
                matched_pair_uplift_enabled=False,
                bow_views=views,
                nuisance_folds=2,
                effect_folds=2,
                top_n_features=5,
                **settings,
            ),
        ),
    )
    config.seed = 5
    return config


def _run(tmp_path, outcome_type: str, *, store=None, **settings):
    output = tmp_path / "worker" / "predictions.parquet"
    runner = MultiModelForestStage1Runner(
        dataset=_dataset(outcome_type),
        config=_config(outcome_type, **settings),
        output_path=output,
        device=torch.device("cpu"),
        num_workers=1,
        checkpoint_store=store,
    )
    row = runner.build_discovery_handoff_row(
        train_idx=np.arange(FIT_ROWS),
        heldout_idx=np.arange(FIT_ROWS, 64),
        fold_key=1,
        outer_fold=1,
        scope="full_outer_train",
    )
    views = {view["view_name"]: view for view in row["importance"]["views"]}
    return views, pd.read_parquet(output)


def _effect_predictions(predictions: pd.DataFrame, view_name: str) -> pd.DataFrame:
    frame = predictions[predictions["source_name"] == f"bow__{view_name}__effect"]
    columns = [c for c in frame.columns if c.startswith("tau_hat_")]
    return frame.sort_values("_oci_row_id")[["_oci_row_id", *columns]].reset_index(drop=True)


def _assert_selected(record, criterion):
    assert record["selection"] == "cross_validated"
    assert record["criterion"] == criterion
    assert record["ridge_alphas"] == list(DEFAULT_BOW_EFFECT_RIDGE_ALPHA_GRID)
    losses = np.asarray(record["heldout_loss"], dtype=float)
    chosen = record["ridge_alphas"].index(record["ridge_alpha"])
    assert losses[chosen] == np.min(losses)


def test_binary_dina_penalty_is_cross_validated_and_sets_the_evidence_fit(
    tmp_path, monkeypatch
):
    calls = []
    original = dina.fit

    def recording_fit(x, y, *args, **kwargs):
        calls.append((len(y), kwargs["regularization"]))
        return original(x, y, *args, **kwargs)

    monkeypatch.setattr(dina, "fit", recording_fit)
    views, _predictions = _run(tmp_path, "binary")
    full_fit_penalties = sorted(reg * FIT_ROWS for rows, reg in calls if rows == FIT_ROWS)
    selected = []
    # DINA is linear on the view's vocabulary whatever the view's learner.
    for name in ("linear_1_2", "extratrees_1_2"):
        record = views[name]["effect_penalty"]["dina"]
        _assert_selected(record, "heldout_dina_loss")
        selected.append(record["ridge_alpha"])
    # The full-training-row evidence model uses each view's chosen penalty.
    np.testing.assert_allclose(full_fit_penalties, sorted(selected))
    effect_fold_fits = [rows for rows, _ in calls if rows != FIT_ROWS]
    assert len(effect_fold_fits) == 2 * 2 * len(DEFAULT_BOW_EFFECT_RIDGE_ALPHA_GRID)


def test_continuous_ridge_penalties_are_cross_validated_per_objective(tmp_path, monkeypatch):
    evidence_penalties = []
    original = multi_model_forest_stage1._make_bow_regressor

    def recording_regressor(params, *, random_state=17):
        if random_state == 303:
            evidence_penalties.append((params["bow_model"], params["ridge_alpha"]))
        return original(params, random_state=random_state)

    monkeypatch.setattr(multi_model_forest_stage1, "_make_bow_regressor", recording_regressor)
    views, _predictions = _run(tmp_path, "continuous")
    linear = views["linear_1_2"]["effect_penalty"]
    _assert_selected(linear["pseudo_outcome"], "heldout_pseudo_outcome_mse")
    _assert_selected(linear["weighted_r"], "heldout_r_loss")
    trees = views["extratrees_1_2"]["effect_penalty"]
    for record in trees.values():
        assert record["selection"] == "not_applicable"
        assert record["ridge_alpha"] is None
    # The evidence model is the weighted R-loss regression with its chosen penalty.
    assert ("linear", linear["weighted_r"]["ridge_alpha"]) in evidence_penalties


@pytest.mark.parametrize("outcome_type", ["binary", "continuous"])
def test_empty_grid_uses_the_fixed_view_penalty(tmp_path, outcome_type):
    fixed_views, fixed = _run(tmp_path / "fixed", outcome_type, bow_effect_ridge_alpha_grid=[])
    single_views, single = _run(
        tmp_path / "single", outcome_type, bow_effect_ridge_alpha_grid=[10.0]
    )
    pd.testing.assert_frame_equal(
        _effect_predictions(fixed, "linear_1_2"), _effect_predictions(single, "linear_1_2")
    )
    for views in (fixed_views, single_views):
        for record in views["linear_1_2"]["effect_penalty"].values():
            assert record["selection"] == "fixed"
            assert record["ridge_alpha"] == 10.0
            assert record["ridge_alphas"] == [10.0]


def test_resumed_context_reuses_the_selected_penalties(tmp_path, monkeypatch):
    def store():
        return Stage1CheckpointStore(
            tmp_path / "checkpoints",
            context_identity={
                "context": "bow-effect-penalty",
                "fit": row_id_identity(list(range(FIT_ROWS))),
                "heldout": row_id_identity(list(range(FIT_ROWS, 64))),
            },
            base_seed=7,
        )

    first_views, first = _run(tmp_path / "first", "binary", store=store())

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a resumed context must not refit DINA effects")

    monkeypatch.setattr(dina, "fit", forbidden)
    second_views, second = _run(tmp_path / "second", "binary", store=store())
    for name in first_views:
        assert second_views[name]["effect_penalty"] == first_views[name]["effect_penalty"]
        pd.testing.assert_frame_equal(
            _effect_predictions(first, name), _effect_predictions(second, name)
        )


def test_grid_setting_is_validated_and_sorted():
    assert MultiModelForestConfig().bow_effect_ridge_alpha_grid == list(
        DEFAULT_BOW_EFFECT_RIDGE_ALPHA_GRID
    )
    unsorted = MultiModelForestConfig(bow_effect_ridge_alpha_grid=[30, 1, 3.5])
    assert unsorted.bow_effect_ridge_alpha_grid == [1.0, 3.5, 30.0]
    assert MultiModelForestConfig(bow_effect_ridge_alpha_grid=[]).bow_effect_ridge_alpha_grid == []
    for grid in ([0.0], [-1.0], [float("nan")], [True], [1.0, 1.0], "1,2", [None]):
        with pytest.raises(ValueError, match="bow_effect_ridge_alpha_grid"):
            MultiModelForestConfig(bow_effect_ridge_alpha_grid=grid)


def test_penalty_selection_prefers_the_larger_penalty_on_ties():
    record = select_ridge_alpha([10.0, 1.0, 100.0], [0.5, 0.5, 0.7], criterion="loss")
    assert record["ridge_alpha"] == 10.0
    assert record["ridge_alphas"] == [1.0, 10.0, 100.0]
    assert record["heldout_loss"] == [0.5, 0.5, 0.7]
    assert record["selection"] == "cross_validated" and not record["at_grid_edge"]
    edge = select_ridge_alpha([1.0, 10.0], [float("nan"), 0.2], criterion="loss")
    assert edge["ridge_alpha"] == 10.0 and edge["at_grid_edge"]
    assert edge["heldout_loss"] == [None, 0.2]
    fixed = select_ridge_alpha([3.0], [0.4], criterion="loss")
    assert fixed["selection"] == "fixed" and not fixed["at_grid_edge"]
    with pytest.raises(ValueError):
        select_ridge_alpha([1.0], [float("nan")], criterion="loss")
    with pytest.raises(ValueError):
        select_ridge_alpha([1.0, 2.0], [0.1], criterion="loss")
