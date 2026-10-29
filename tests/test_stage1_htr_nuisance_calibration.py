"""HTR nuisance calibrators are fit on out-of-fold predictions (B3)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from oci.inference import multi_model_forest_stage1
from oci.utils.calibration import BinaryProbabilityCalibrator


class _FakeNet:
    def __init__(self, *, extractor, **_kwargs):
        self.extractor = extractor
        self.trained_rows: set[int] = set()

    def to(self, _device):
        return self


class _MemorizingRunner:
    """Fits its training rows perfectly and knows nothing about other rows.

    Its in-sample predictions are therefore maximally overconfident, while its
    out-of-fold predictions are uninformative. A calibrator fit in sample
    sharpens; one fit out of fold flattens toward the base rate.
    """

    def __init__(self, *, outcome_type: str = "binary"):
        self.avf_config = SimpleNamespace(
            nuisance_folds=3,
            effect_folds=3,
            nuisance_calibration="temperature",
            effect_objective="dina",
            e_clip=0.01,
        )
        self.config = SimpleNamespace(
            treatment_column="treatment",
            outcome_column="outcome",
            outcome_type=outcome_type,
            architecture=SimpleNamespace(htr_prediction_head_hidden_dim=4),
        )
        self.device = torch.device("cpu")

    def _create_extractor(self):
        return object()

    def _train_nuisance_model(self, model, df, fit_pos, **_kwargs):
        model.trained_rows = set(df.iloc[fit_pos]["_oci_row_id"].astype(int))

    def _predict_nuisance_model(self, model, frame):
        row_ids = frame["_oci_row_id"].to_numpy(dtype=int)
        memorized = np.asarray([row in model.trained_rows for row in row_ids])
        uninformative = 0.35 + 0.3 * ((row_ids * 7) % 10) / 10.0
        treatment = frame["treatment"].to_numpy(dtype=float)
        outcome = frame["outcome"].to_numpy(dtype=float)
        e = np.where(memorized, np.where(treatment == 1, 0.98, 0.02), uninformative)
        m = np.where(memorized, np.where(outcome == 1, 0.98, 0.02), uninformative[::-1])
        return e, m

    def _attention_evidence(self, _extractor, frame, *, fold, extra, **_kwargs):
        return [
            {"row_id": int(row_id), "fold": int(fold), "e_hat_raw": float(raw)}
            for row_id, raw in zip(frame["_oci_row_id"].to_numpy(), extra["e_hat_raw"])
        ]

    def _cleanup_model(self, _model):
        return None


def _provider(runner):
    provider = object.__new__(multi_model_forest_stage1.MultiModelForestStage1HTRProvider)
    provider._runner = runner
    provider.config = runner.config
    provider.checkpoint_store = None
    provider._ensure_runner = lambda _frame: runner
    return provider


def _frames():
    rows = 60
    train = pd.DataFrame(
        {
            "_oci_row_id": np.arange(rows),
            "treatment": np.tile([0, 1], rows // 2),
            "outcome": np.tile([0, 0, 1], rows // 3),
        }
    )
    test = pd.DataFrame(
        {
            "_oci_row_id": [100, 101, 102, 103],
            "treatment": [0, 1, 0, 1],
            "outcome": [1, 0, 0, 1],
        }
    )
    return train, test


def test_htr_nuisance_calibrators_are_fit_out_of_fold(monkeypatch):
    monkeypatch.setattr(multi_model_forest_stage1, "NuisanceNet", _FakeNet)
    train, test = _frames()

    result = _provider(_MemorizingRunner()).fit_nuisance_inner_ensemble_predict(
        train, test, outer_fold=1
    )
    predictions = result["train"]["predictions"]

    # An in-sample calibrator would sharpen these uninformative held-out
    # predictions toward 0 and 1 (temperature at its 0.2 floor).
    assert predictions["e_hat"].between(0.2, 0.8).all()
    assert predictions["m_hat"].between(0.2, 0.8).all()

    folds = predictions["nuisance_fold"].to_numpy(dtype=int)
    raw = predictions["e_hat_raw"].to_numpy(dtype=float)
    treatment = train["treatment"].to_numpy(dtype=float)
    expected = np.empty(len(train))
    test_expected = []
    test_raw = np.asarray(
        _MemorizingRunner()._predict_nuisance_model(
            SimpleNamespace(trained_rows=set()), test
        )[0]
    )
    for fold in np.unique(folds):
        calibrator = BinaryProbabilityCalibrator.fit(
            raw[folds != fold], treatment[folds != fold], method="temperature"
        )
        expected[folds == fold] = calibrator.transform(raw[folds == fold])
        test_expected.append(calibrator.transform(test_raw))
    np.testing.assert_allclose(predictions["e_hat"], expected)
    np.testing.assert_allclose(
        result["test_predictions"]["e_hat"], np.mean(test_expected, axis=0)
    )
    np.testing.assert_allclose(
        predictions["t_residual"], treatment - predictions["e_hat"].to_numpy()
    )

    attention_e = {row["row_id"]: row["e_hat"] for row in result["train"]["attention"]}
    np.testing.assert_allclose(
        [attention_e[row] for row in train["_oci_row_id"]], predictions["e_hat"]
    )
