"""Plot the completed controlled sample-size comparison."""
from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/oci_oracle_power_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
table = pd.read_csv(HERE / "metrics_by_seed_2026-09-21.csv")
table = table[table.cohort == "new"]
assert len(table) == 18
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans", "axes.titleweight": "bold"})
figure, axes = plt.subplots(1, 2, figsize=(11.4, 5.3))
for scenario, label, color in [
    ("modifiers", "Modifiers in X; confounders in W", "#2267B3"),
    ("all", "All ten variables in X", "#B04D26"),
]:
    group = table[table.scenario == scenario].groupby("training_rows")
    for axis, metric in zip(axes, ["correlation", "rmse"]):
        values = group[metric].agg(["mean", "min", "max"])
        x = values.index.to_numpy()
        axis.plot(x, values["mean"], marker="o", markersize=7, linewidth=2.3, color=color, label=label)
        axis.fill_between(x, values["min"], values["max"], color=color, alpha=.15, linewidth=0)
        for n, value in zip(x, values["mean"]):
            other = table[(table.training_rows == n) & (table.scenario != scenario)][metric].mean()
            offset = 11 if value >= other else -19
            axis.annotate(f"{value:.3f}", (n, value), xytext=(0, offset), textcoords="offset points",
                          ha="center", fontsize=9, color=color)
for axis, title, ylabel in zip(axes, ["Effect correlation", "Effect estimation error"],
                              ["Pearson correlation with true effect", "RMSE on probability-difference scale"]):
    axis.set_xscale("log")
    axis.set_xticks([800, 8000, 80000], ["800", "8,000", "80,000"])
    axis.set_xlabel("Training patients")
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontsize=13, pad=13)
    axis.grid(axis="y", color="#E5E8EC", linewidth=.7)
    axis.margins(x=.12, y=.18)
axes[0].set_ylim(0, 1)
axes[1].set_ylim(bottom=0)
figure.suptitle("Oracle causal forest learning curves", fontsize=17, fontweight="bold", y=.98)
figure.text(.5, .917, "200 trees · 45% sampling · honesty and inference on · minimum leaf size 10", ha="center", fontsize=10, color="#444444")
handles, labels = axes[0].get_legend_handles_labels()
figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .072), ncol=2, frameon=False, fontsize=10)
figure.text(.5, .025, "Same 20,000 test patients. Lines: means of 3 forest seeds; shading: seed ranges, not confidence intervals.",
            ha="center", fontsize=9, color="#555555")
figure.subplots_adjust(left=.085, right=.98, top=.80, bottom=.26, wspace=.34)
figure.savefig(HERE / "learning_curve_2026-09-21.png", dpi=170)
figure.savefig(HERE / "learning_curve_2026-09-21.pdf")
plt.close(figure)
print("Saved learning-curve PNG and PDF.")
