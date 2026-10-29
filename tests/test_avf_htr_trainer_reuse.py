"""The standalone AVF runner reuses the Stage 1 HTR training core."""

from __future__ import annotations


import numpy as np
import pandas as pd
import pytest
import torch

from oci.config import AppliedInferenceConfig
from oci.inference.agentic_attention_variable_forest import AgenticAttentionVariableForestRunner


# --------------------------------------------------------------------------
# B3: AVF nuisance calibration is cross-fitted, never fit on in-sample output.


class _DummyNuisanceNet(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.extractor = None
        self.fit_rows: set = set()


def _calibration_frame(n: int = 120, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "clinical_text": [f"note {i}" for i in range(n)],
            "treatment_indicator": rng.binomial(1, 0.4, n),
            "outcome_indicator": rng.binomial(1, 0.3, n),
        }
    )


def _overfitting_runner(tmp_path, monkeypatch):
    from oci.inference import agentic_attention_variable_forest as avf_module

    runner = AgenticAttentionVariableForestRunner(
        dataset=_calibration_frame(),
        config=AppliedInferenceConfig(outcome_type="binary"),
        output_path=tmp_path / "predictions.parquet",
        device=torch.device("cpu"),
        num_workers=1,
        proposal_agent=object(),
        extraction_provider=object(),
    )
    monkeypatch.setattr(avf_module, "NuisanceNet", _DummyNuisanceNet)
    monkeypatch.setattr(runner, "_create_extractor", lambda: None)

    def train(model, df, positions, **kwargs):
        model.fit_rows = set(df.iloc[np.asarray(positions)]["_oci_row_id"].tolist())

    def predict(model, frame):
        # Overconfident on its own training rows, weakly informative elsewhere.
        rng = np.random.default_rng(len(frame))
        values = []
        for column in ("treatment_indicator", "outcome_indicator"):
            labels = frame[column].to_numpy(dtype=float)
            in_sample = frame["_oci_row_id"].isin(model.fit_rows).to_numpy()
            confident = np.where(
                labels == 1,
                rng.uniform(0.9, 0.999, len(frame)),
                rng.uniform(0.001, 0.1, len(frame)),
            )
            weak = rng.uniform(0.35, 0.65, len(frame))
            values.append(np.where(in_sample, confident, weak))
        return values[0], values[1]

    def attention(extractor, df, fold, outer_fold, stage, extra):
        return [
            {
                "row_id": row_id,
                "fold": fold,
                "stage": stage,
                **{key: float(values[i]) for key, values in extra.items()},
            }
            for i, row_id in enumerate(df["_oci_row_id"].tolist())
        ]

    monkeypatch.setattr(runner, "_train_nuisance_model", train)
    monkeypatch.setattr(runner, "_predict_nuisance_model", predict)
    monkeypatch.setattr(runner, "_attention_evidence", attention)
    return runner, predict


def test_avf_nuisance_calibration_is_out_of_fold(tmp_path, monkeypatch):
    from sklearn.metrics import log_loss

    from oci.utils.calibration import BinaryProbabilityCalibrator

    runner, predict = _overfitting_runner(tmp_path, monkeypatch)
    df = runner.dataset
    result = runner._crossfit_nuisance(df, outer_fold=1)
    predictions = result["predictions"]

    # Raw held-out predictions come from models that never saw those rows.
    assert predictions["e_hat_raw"].between(0.35, 0.65).all()
    # Cross-fitted calibration softens toward the base rate instead of
    # sharpening uninformative held-out predictions into near-certain ones
    # (isotonic steps may still leave an occasional boundary block).
    for column in ("e_hat", "m_hat"):
        values = predictions[column]
        assert np.mean((values < 0.02) | (values > 0.98)) < 0.05
    np.testing.assert_allclose(
        predictions["t_residual"],
        df["treatment_indicator"].to_numpy() - predictions["e_hat"].to_numpy(),
    )
    # Attention metadata carries the calibrated values, not stale placeholders.
    by_row = predictions.set_index("_oci_row_id")
    for record in result["attention"]:
        assert record["e_hat"] == pytest.approx(by_row.loc[record["row_id"], "e_hat"])
        assert record["y_residual"] == pytest.approx(by_row.loc[record["row_id"], "y_residual"])

    # The removed behaviour: a calibrator fit on a fold model's own in-sample
    # output stretches uninformative held-out predictions instead of
    # shrinking them, which costs held-out log loss.
    model = _DummyNuisanceNet()
    fit, held = df.iloc[: len(df) // 2], df.iloc[len(df) // 2 :]
    model.fit_rows = set(fit["_oci_row_id"])
    e_fit, _ = predict(model, fit)
    e_held, _ = predict(model, held)
    sharpened = BinaryProbabilityCalibrator.fit(
        e_fit, fit["treatment_indicator"].to_numpy(dtype=float)
    ).transform(e_held)
    crossfit_loss = log_loss(df["treatment_indicator"], predictions["e_hat"])
    in_sample_loss = log_loss(held["treatment_indicator"], sharpened)
    assert np.std(sharpened) > 1.5 * np.std(predictions["e_hat"])
    assert crossfit_loss < in_sample_loss - 0.1

    # A resume reuses raw checkpoints and recomputes the same calibration.
    resumed = runner._crossfit_nuisance(df, outer_fold=1)["predictions"]
    pd.testing.assert_frame_equal(resumed, predictions)
