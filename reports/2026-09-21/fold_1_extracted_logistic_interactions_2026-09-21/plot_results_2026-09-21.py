"""Plot frozen held-out effects for the two full-training logistic models."""
from pathlib import Path
import json
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/oci_extracted_logistic_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
evaluation = json.loads((HERE / "evaluation_2026-09-21.json").read_text())
truth = pd.read_parquet(evaluation["oracle_path"], columns=["true_ite_prob"])
metrics = pd.read_csv(HERE / "logistic_metrics_2026-09-21.csv")
frames = {name: pd.read_parquet(HERE / "fits/all_800" / name / "predictions.parquet") for name in ["elastic_net", "ridge"]}
ids = frames["elastic_net"]._oci_row_id.to_numpy()
assert np.array_equal(ids, frames["ridge"]._oci_row_id)
tau = truth.iloc[ids].true_ite_prob.to_numpy()
values = np.concatenate([tau, *[frame.estimated_cate.to_numpy() for frame in frames.values()]])
limits = (np.floor(values.min() * 10) / 10 - .05, np.ceil(values.max() * 10) / 10 + .05)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.7), sharex=True, sharey=True)
for ax, (name, label, color) in zip(axes, [("elastic_net", "Elastic net", "#2468A2"), ("ridge", "Ridge", "#B7532D")]):
    frame = frames[name]
    row = metrics[(metrics.model == name) & (metrics.training_cohort == "all_800") & (metrics.test_set == "full_200")].iloc[0]
    ax.scatter(tau, frame.estimated_cate, s=22, alpha=.65, color=color, edgecolors="none")
    ax.plot(limits, limits, color="#777777", linestyle="--", linewidth=1.1)
    ax.axhline(0, color="#DDE2E7", linewidth=.8)
    ax.axvline(0, color="#DDE2E7", linewidth=.8)
    ax.set(xlim=limits, ylim=limits, xlabel="True probability-scale treatment effect", title=label)
    ax.set_aspect("equal", adjustable="box")
    ax.text(.04, .96, f"Correlation: {row.correlation:.3f}\nRMSE: {row.rmse:.3f}\nMean bias: {row.bias:+.3f}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9})
axes[0].set_ylabel("Estimated probability-scale treatment effect")
fig.suptitle("Outcome interaction models using extracted candidates", fontsize=16, fontweight="bold", y=.985)
fig.text(.5, .925, "352 candidates · 2,194 coefficients including intercept · 800 training / 200 test patients", ha="center", fontsize=10)
fig.text(.5, .035, "Penalty chosen by five-fold training outcome log loss. Dashed line: perfect effect prediction.", ha="center", fontsize=9, color="#555555")
fig.subplots_adjust(left=.08, right=.985, top=.85, bottom=.15, wspace=.17)
fig.savefig(HERE / "effect_predictions_2026-09-21.png", dpi=180)
fig.savefig(HERE / "effect_predictions_2026-09-21.pdf")
plt.close(fig)
print("Saved effect prediction figure.")
