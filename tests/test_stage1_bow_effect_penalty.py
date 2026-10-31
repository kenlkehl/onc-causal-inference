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
from oci.inference.multi_model_forest_stage1 import MultiModelForestStage1Runner
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
