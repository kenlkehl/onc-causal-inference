"""Score only inner validation, after numerical and LLM artifacts are frozen."""
from collections import Counter
import json
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/query-perturbation-mpl")
import numpy as np
import pandas as pd

from common import HERE, RUN, POLICY, labels, nuisance_path, read, sha, write, now, verify_manifest
from probes import loss


def moment(values, t, y, u, v, tau0, target):
    if target == "effect":
        c, w = u * (v - tau0 * u), u ** 2
    else:
        observed = t if target == "treatment" else y
        prevalence = float(observed.mean())
        assert 0 < prevalence < 1
        c, w = observed / prevalence - (1-observed) / (1-prevalence), np.ones(len(observed))
    centered = values - np.average(values, weights=w)
    rows = centered * c
    return float(rows.mean()), float(rows.std(ddof=0))


def table(headers, rows):
    escape = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
                       *["| " + " | ".join(escape(x) for x in row) + " |" for row in rows]])


def verify_freezes():
    manifest = verify_manifest()
    numerical, interpretations = read(RUN / "numerical_frozen.json"), read(RUN / "llm_frozen.json")
    assert numerical["manifest_sha256"] == interpretations["manifest_sha256"] == sha(RUN / "manifest.json")
    assert len(numerical["folds"]) == 5 and len(interpretations["requests"]) == 75
    for name, expected in numerical["folds"].items():
        folder = RUN / "folds" / name
        assert sha(folder / "frozen.json") == expected
        frozen = read(folder / "frozen.json")
        assert frozen["validation_outcomes_loaded"] is False and not frozen["outer_test_rows_used"]
        for file, digest in frozen["files"].items():
            assert sha(folder / file) == digest
        for identifier, digest in frozen["packets"].items():
            assert sha(RUN / "packets" / (identifier + ".json")) == digest
    for name, expected in interpretations["requests"].items():
        folder = RUN / "llm" / name
        assert sha(folder / "complete.json") == expected
        complete = read(folder / "complete.json")
        assert sha(folder / "response.json") == complete["response_sha256"]
        assert sha(folder / "input.json") == complete["input_sha256"]
    return manifest, interpretations


def main():
    manifest, interpretations = verify_freezes()
    write(RUN / "inner_validation_access_started.json", {"at": now(), "outer_test_labels_accessed": False,
          "numerical_freeze_sha256": sha(RUN / "numerical_frozen.json"), "llm_freeze_sha256": sha(RUN / "llm_frozen.json"),
          "evaluator_sha256": sha(__file__), "scope": "the five inner-validation sets within outer fold 1 training"})
    signals, losses = [], []
    for position, part in enumerate(manifest["inner_splits"], 1):
        folder = RUN / "folds" / f"inner_{position:03d}"
        variants = read(folder / "variants.json")
        optimization = read(folder / "optimization.json")
        tau0 = optimization["tau0"]
        nuisance = pd.read_parquet(nuisance_path(position)).set_index("_oci_row_id")
        arrays = np.load(folder / "activations.npz", allow_pickle=False)
        residual = {}
        for sample, row_ids in (("train", part["fit_row_ids"]), ("validation", part["heldout_row_ids"])):
            observed = labels(row_ids)
            t, y = observed.treatment.to_numpy(), observed.outcome.to_numpy()
            e, m = nuisance.loc[row_ids, ["treatment_stacked", "outcome_stacked"]].to_numpy().T
            residual[sample] = t, y, t - e, y - m
        for target in ("effect", "treatment", "outcome"):
            training_scales = [moment(arrays["train"][:, j], *residual["train"], tau0, target)[1] for j in range(5)]
            for sample in ("train", "validation"):
                for index, variant in enumerate(variants):
                    values = arrays[sample][:, index]
                    mean, scale = moment(values, *residual[sample], tau0, target)
                    base = arrays[sample][:, variant["query"]]
                    signals.append({"inner_fold": position, "sample": sample, "target": target, "variant": index, **variant,
                                    "n": len(values), "moment": mean, "row_score_sd": scale,
                                    "standardized_score": np.sqrt(len(values)) * mean / max(scale, 1e-10),
                                    "abs_studentized_score": abs(np.sqrt(len(values)) * mean / max(scale, 1e-10)),
                                    "fixed_scale_score": np.sqrt(len(values)) * mean / max(training_scales[variant["query"]], 1e-10),
                                    "abs_fixed_scale_score": abs(np.sqrt(len(values)) * mean / max(training_scales[variant["query"]], 1e-10)),
                                    "activation_sd": float(values.std()),
                                    "activation_correlation_with_original": float(np.corrcoef(base, values)[0, 1])})
        metadata = read(folder / "probe_metadata.json")
        pred_arrays = np.load(folder / "probe_predictions.npz", allow_pickle=False)
        assert pred_arrays["row_ids"].tolist() == part["heldout_row_ids"]
        fold_losses = []
        for index, item in enumerate(metadata):
            variant = variants[item["variant"]] if item["variant"] >= 0 else {"kind": item["mode"], "radius": 0, "control": -1, "distance": 0}
            record = {"inner_fold": position, **item, **{k: variant[k] for k in ("kind", "radius", "control", "distance")},
                      "validation_n": len(part["heldout_row_ids"]),
                      "loss": loss(item["target"], pred_arrays["predictions"][index], *residual["validation"])}
            fold_losses.append(record)
        key = lambda r: (r["query"], r["scope"], r["family"], r["target"])
        original = {key(r): r["loss"] for r in fold_losses if r["mode"] == "original"}
        constants = {r["target"]: r["loss"] for r in fold_losses if r["mode"] == "constant"}
        remaining = {key(r): r["loss"] for r in fold_losses if r["mode"] == "without_query"}
        for record in fold_losses:
            record["constant_loss"] = constants[record["target"]]
            record["gain_over_constant"] = record["constant_loss"] - record["loss"]
            if key(record) in original:
                record["original_loss"] = original[key(record)]
                record["loss_increase_from_original"] = record["loss"] - original[key(record)]
            if key(record) in remaining:
                record["without_query_loss"] = remaining[key(record)]
                record["gain_over_without_query"] = remaining[key(record)] - record["loss"]
        losses.extend(fold_losses)
        arrays.close()
        pred_arrays.close()
    signal_frame, loss_frame = pd.DataFrame(signals), pd.DataFrame(losses)
    signal_frame.to_csv(RUN / "signal_metrics_2026-09-22.csv", index=False)
    loss_frame.to_csv(RUN / "probe_losses_2026-09-22.csv", index=False)
    signal_summary = signal_frame.groupby(["sample", "target", "kind", "radius"], as_index=False).agg(
        mean_abs_fixed_scale_score=("abs_fixed_scale_score", "mean"), mean_distance=("distance", "mean"),
        mean_abs_studentized_score=("abs_studentized_score", "mean"),
        mean_training_sd_ratio=("training_sd_ratio", "mean"),
        mean_activation_correlation=("activation_correlation_with_original", "mean"))
    probe_summary = loss_frame.groupby(["target", "family", "scope", "mode", "kind", "radius"], as_index=False).agg(
        loss=("loss", "mean"), gain_over_constant=("gain_over_constant", "mean"),
        loss_increase_from_original=("loss_increase_from_original", "mean"))
    signal_summary.to_csv(RUN / "signal_summary_2026-09-22.csv", index=False)
    probe_summary.to_csv(RUN / "probe_summary_2026-09-22.csv", index=False)
    candidate_records, request_records = [], []
    for identifier in sorted(interpretations["requests"]):
        directory = RUN / "llm" / identifier
        complete, response = read(directory / "complete.json"), read(directory / "response.json")
        request_records.append({"packet_id": identifier, **complete, "summary": response["summary"]})
        for index, candidate in enumerate(response["candidates"]):
            candidate_records.append({"packet_id": identifier, "inner_fold": complete["inner_fold"], "query": complete["query"],
                                      "condition": complete["condition"], "candidate": index, **candidate})
    write(RUN / "candidate_proposals_2026-09-22.json", candidate_records)
    pd.DataFrame([{**c, "citations": json.dumps(c["citations"]), "role_hypotheses": json.dumps(c["role_hypotheses"]),
                   "categories": json.dumps(c["categories"])} for c in candidate_records]).to_csv(RUN / "candidate_proposals_2026-09-22.csv", index=False)
    pd.DataFrame(request_records).to_csv(RUN / "interpretations_2026-09-22.csv", index=False)
    write(RUN / "evaluation_2026-09-22.json", {"at": now(), "outer_fold": 1, "outer_test_patients_used": 0,
          "oracle_columns_loaded": False, "signal_rows": len(signals), "probe_loss_rows": len(losses),
          "llm_requests": len(request_records), "candidate_proposals": len(candidate_records),
          "numerical_freeze_sha256": sha(RUN / "numerical_frozen.json"), "llm_freeze_sha256": sha(RUN / "llm_frozen.json")})
    make_plot(signal_summary, probe_summary)
    build_report(signal_frame, loss_frame, signal_summary, probe_summary, candidate_records, request_records)
    write(RUN / "complete.json", {"at": now(), "report": str(HERE / "REPORT_2026-09-22.md"),
          "report_sha256": sha(HERE / "REPORT_2026-09-22.md"), "evaluation_sha256": sha(RUN / "evaluation_2026-09-22.json")})
    print(json.dumps(read(RUN / "evaluation_2026-09-22.json")), flush=True)


def make_plot(signal_summary, probe_summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, sample in zip(axes[0], ("train", "validation")):
        data = signal_summary[(signal_summary.target == "effect") & (signal_summary["sample"] == sample)]
        base = data.loc[data.kind == "original", "mean_abs_fixed_scale_score"].iloc[0]
        for kind, color in (("targeted", "#b33a3a"), ("random", "#3574a0")):
            block = data[data.kind == kind].sort_values("radius")
            axis.plot([0, *block.radius], [base, *block.mean_abs_fixed_scale_score], marker="o", label=kind, color=color)
        axis.set(title=f"{sample.capitalize()}: effect moment", xlabel="Maximum query distance", ylabel="Mean absolute score, fixed training scale")
        axis.legend()
    for axis, scope in zip(axes[1], ("solo", "conditional")):
        data = probe_summary[(probe_summary.target == "effect") & (probe_summary.family == "linear") & (probe_summary.scope == scope)]
        for kind, color in (("targeted", "#b33a3a"), ("random", "#3574a0")):
            for mode, style in (("frozen", "--"), ("refit", "-")):
                block = data[(data.kind == kind) & (data["mode"] == mode)].sort_values("radius")
                axis.plot(block.radius, block.loss_increase_from_original, style, marker="o", color=color, label=f"{kind}, {mode}")
        axis.axhline(0, color="gray", linewidth=0.8)
        axis.set(title=f"Inner-validation R-loss: {scope}", xlabel="Maximum query distance", ylabel="Loss increase from original (higher = worse)")
        axis.legend(fontsize=8)
    fig.suptitle("Outer fold 1 training only: 25 effect queries across five inner splits", fontsize=12)
    fig.savefig(HERE / "query_perturbation_2026-09-22.png", dpi=170)
    plt.close(fig)


def build_report(signals, losses, signal_summary, probe_summary, candidates, requests):
    effect = signal_summary[signal_summary.target == "effect"]
    rows = []
    for radius in POLICY["radii"]:
        row = [radius]
        for sample in ("train", "validation"):
            for kind in ("targeted", "random"):
                value = effect[(effect["sample"] == sample) & (effect.kind == kind) & (effect.radius == radius)].iloc[0]
                row.append(f"{value.mean_abs_fixed_scale_score:.4f}")
        rows.append(row)
    baseline_scores = {sample: float(effect[(effect["sample"] == sample) & (effect.kind == "original")].mean_abs_fixed_scale_score.iloc[0])
                       for sample in ("train", "validation")}
    studentized_rows = []
    for sample in ("train", "validation"):
        row = [sample]
        for kind in ("original", "targeted", "random"):
            radius = 0 if kind == "original" else POLICY["llm_radius"]
            block = effect[(effect["sample"] == sample) & (effect.kind == kind) & (effect.radius == radius)]
            row.append(f"{block.mean_abs_studentized_score.iloc[0]:.4f}")
        studentized_rows.append(row)
    baselines = probe_summary[(probe_summary.target == "effect") & (probe_summary["mode"] == "original")]
    probe_rows = []
    for scope in ("solo", "conditional"):
        for family in ("linear", "spline"):
            base = baselines[(baselines.scope == scope) & (baselines.family == family)].iloc[0]
            row = [scope, family, f"{base.loss:.6f}", f"{base.gain_over_constant:.6f}"]
            for kind, mode in (("targeted", "frozen"), ("targeted", "refit"), ("random", "frozen"), ("random", "refit")):
                block = probe_summary[(probe_summary.target == "effect") & (probe_summary.scope == scope) &
                         (probe_summary.family == family) & (probe_summary.kind == kind) & (probe_summary["mode"] == mode) &
                         (probe_summary.radius == POLICY["llm_radius"])]
                row.append(f"{block.loss_increase_from_original.iloc[0]:+.6f}")
            probe_rows.append(row)
    proposal_rows = []
    peak_changes = {}
    for condition in ("targeted", "random"):
        changed, total = 0, 0
        for request in requests:
            if request["condition"] != condition:
                continue
            packet = read(RUN / "packets" / (request["packet_id"] + ".json"))
            patient_peaks = {}
            for excerpt in packet["evidence"]:
                for reason in excerpt["selection_reasons"]:
                    patient_peaks.setdefault(excerpt["row_id"], {})[reason] = excerpt["chunk_index"]
            for peaks in patient_peaks.values():
                changed += peaks["original_peak"] != peaks["edited_peak"]
                total += 1
        peak_changes[condition] = (changed, total)
    for condition in POLICY["llm_conditions"]:
        subset = [x for x in candidates if x["condition"] == condition]
        names = Counter(x["name"].strip().casefold() for x in subset)
        request_subset = [r for r in requests if r["condition"] == condition]
        proposal_rows.append([condition, len(request_subset), len(subset), len(names),
                              f"{np.mean([r['excerpts'] for r in request_subset]):.1f}",
                              "; ".join(f"{name} ({count})" for name, count in names.most_common(6))])
    secondary_rows = []
    for target in ("treatment", "outcome"):
        for scope in ("solo", "conditional"):
            for family in ("linear", "spline"):
                row = [target, scope, family]
                for kind in ("targeted", "random"):
                    block = probe_summary[(probe_summary.target == target) & (probe_summary.scope == scope) &
                            (probe_summary.family == family) & (probe_summary.kind == kind) &
                            (probe_summary["mode"] == "refit") & (probe_summary.radius == POLICY["llm_radius"])]
                    row.append(f"{block.loss_increase_from_original.iloc[0]:+.6f}")
                secondary_rows.append(row)
    fold_rows = []
    for fold in range(1, 6):
        row = [fold]
        block = signals[(signals.inner_fold == fold) & (signals.target == "effect") &
                        (signals["sample"] == "validation")]
        for kind in ("original", "targeted", "random"):
            radius = 0 if kind == "original" else POLICY["llm_radius"]
            row.append(f"{block[(block.kind == kind) & (block.radius == radius)].abs_studentized_score.mean():.4f}")
        for scope in ("solo", "conditional"):
            probes = losses[(losses.inner_fold == fold) & (losses.target == "effect") &
                            (losses["mode"] == "refit") & (losses.family == "linear") &
                            (losses.scope == scope) & (losses.radius == POLICY["llm_radius"])]
            difference = probes[probes.kind == "targeted"].loss.mean() - probes[probes.kind == "random"].loss.mean()
            row.append(f"{difference:+.6f}")
        fold_rows.append(row)
    report = ["# Stage 1 query-perturbation pilot — September 22, 2026", "",
        "1. **Experiment and scope**",
        "   1. Evaluated 25 saved effect queries across the five inner splits of outer fold 1. Each edit, probe, and LLM packet used only its 640 training patients; evaluation used its corresponding 160 inner-validation patients.",
        "   2. **No outer-test patient, outer-test outcome, or oracle variable was used.** All edits, predictions, and 75 independent LLM responses were frozen before inner-validation scoring. Every one of 4,000 nuisance-prediction training registries was checked for self-training and split leakage.",
        "   3. Exact-context cached TF-IDF nuisances were held fixed. They are correctly scoped but differ from the unpersisted nuisance fits originally used to optimize these neural queries. This pilot recomputes its own baseline moments.",
        "   4. Targeted edits minimize a fixed-scale effect moment with bounded query distance and preserved activation variance. Five seeded random edits match each actual distance. Radius selection, restart selection, and LLM evidence selection did not use validation outcomes.",
        "", "2. **Effect-score attenuation**",
        f"   1. Before editing, mean absolute effect score on the fixed training scale was {baseline_scores['train']:.4f} in training and {baseline_scores['validation']:.4f} in inner validation.",
        "   2. Scores below use each query's original training row-score standard deviation. They are descriptive diagnostics; a small moment can conceal nonlinear or cancelling signal. Random controls are averaged within the balanced query/fold design.", "",
        table(["Distance budget", "Train: targeted", "Train: random", "Validation: targeted", "Validation: random"], rows), "",
        "   3. A fixed-scale moment can fall partly because activation amplitude shrinks. The optimizer permits SD down to half its original value. The next table recomputes the row-score standard deviation for each edited query, making a simple activation rescaling insufficient to reduce the score. Neither score alone proves information erasure.", "",
        table(["Sample", "Original mean absolute standardized score", "Targeted at 0.20", "Random at 0.20"], studentized_rows), "",
        "3. **Does a new predictor recover the information?**",
        "   1. A frozen probe retains the model fitted to original activations. A refitted probe learns from edited training activations. The conditional probe uses all 15 queries with one effect query replaced; the solo probe uses only that query.",
        "   2. The table uses the prespecified 0.20 radius. Positive loss increases mean worse inner-validation effect prediction. Positive original gain means improvement over a fitted constant-effect model.", "",
        table(["Scope", "Probe", "Original R-loss", "Original gain vs constant", "Targeted frozen Δ", "Targeted refit Δ", "Random frozen Δ", "Random refit Δ"], probe_rows), "",
        "   3. These fixed linear and additive spline probes test a limited family of readouts. They do not establish removal of every nonlinear interaction, and this is not a new outer-test CATE evaluation.", "",
        "   4. The following split-level results use radius 0.20. Score columns use each query's own row-score standard deviation. The last two columns compare refitted linear probe R-loss for targeted versus matched random edits; positive differences mean targeted edits hurt prediction more. Each row averages five queries and their balanced random controls. The five training sets overlap, so these are descriptive repeats rather than five independent experiments.", "",
        table(["Inner split", "Original score", "Targeted score", "Random score", "Solo targeted − random R-loss", "Conditional targeted − random R-loss"], fold_rows), "",
        "4. **Secondary nuisance prediction changes**",
        "   1. Changes in inner-validation log loss after refitting, at radius 0.20. Positive values mean worse prediction. Nuisance independence was not imposed as a requirement for a valid modifier.", "",
        table(["Target", "Scope", "Probe", "Targeted Δ log loss", "Random Δ log loss"], secondary_rows), "",
        "5. **LLM candidate proposals**",
        "   1. Gemma 4 31B at the requested endpoint interpreted three independent conditions per query: original retrieval, targeted-edit contrast, and random-edit contrast. Each request allowed at most six training patients, four complete excerpts per patient, and two proposed variables. Literal citation quotes were validated, with errors returned for repair.",
        "   2. Unique-name counts below mean exact case-normalized names, not independently established clinical concepts. Differences in wording are not evidence of novel feature discovery. Duplicate excerpts were combined; the table reports actual excerpt counts.", "",
        table(["Condition", "Requests", "Proposals", "Unique names", "Mean excerpts", "Most frequent names (count)"], proposal_rows), "",
        "   3. Complete definitions, extraction instructions, role hypotheses, citations, and uncertainties are saved in [candidate proposals](pilot_v1/candidate_proposals_2026-09-22.json). Per-query interpretations are in [the interpretation table](pilot_v1/interpretations_2026-09-22.csv). These variables have not yet been newly extracted or validated in Stage 2.", "",
        "   4. There is one stochastic LLM draw per query/condition. Differences between candidate lists can reflect model sampling as well as changed retrieval. There is no repeated-identical-prompt control, and exact-name counts are not a test of discovery superiority.", "",
        "   5. Original packets show high-activation retrieval, whereas both edited conditions show retrieval contrasts. Packets identify their condition, so the LLM comparison is not blinded. A future discovery comparison should match packet format, hide condition labels, and repeat identical-prompt controls.", "",
        f"   6. In the training patients selected for contrast packets, targeted edits changed the highest-similarity chunk in {peak_changes['targeted'][0]}/{peak_changes['targeted'][1]} patient-query instances; random edits did so in {peak_changes['random'][0]}/{peak_changes['random'][1]}. These are descriptive retrieval diagnostics: conditions select different patients, and repeated patient-query instances are not independent observations.", "",
        "6. **How to interpret this pilot**",
        "   1. Training-score suppression establishes that an optimization can change its objective. Improvement over equal-distance random edits on inner validation is the relevant robustness check.",
        "   2. If refitting restores predictive value, the edit changed the representation more than it erased information. If the original probe has little validation gain, there is little demonstrated useful signal to remove.",
        "   3. Candidate proposals are hypotheses. More proposals or different clinical names do not establish greater sensitivity, better extraction, or improved CATE estimation. A subsequent extraction and modeling experiment is needed for that claim.",
        "   4. All results are exploratory on a previously examined cohort. Overlapping training sets, queries, and random seeds are not independent cohorts. No hyperparameters were selected from these new validation results.", "",
        "7. **Artifacts and reproducibility**",
        "   1. [Prespecified protocol](PILOT_PROTOCOL_2026-09-22.md); [input and source manifest](pilot_v1/manifest.json); [isolation audit](pilot_v1/ISOLATION_AUDIT_2026-09-22.json).",
        "   2. [Signal metrics](pilot_v1/signal_metrics_2026-09-22.csv); [probe losses](pilot_v1/probe_losses_2026-09-22.csv); [probe summary](pilot_v1/probe_summary_2026-09-22.csv).",
        "   3. Independent frozen numerical and LLM artifacts support a checked evaluation replay. Reproduction scripts are contained in this research directory; production workflow files were not edited.", "",
        "   4. Use `run_numerical.py` as the numerical entry point. It gives each fold a fresh process, avoiding repeated PyTorch interop initialization in the original scheduler. The frozen fitting implementation and completed checkpoints were preserved. Six focused tests passed; 36 real frozen prediction vectors were reproduced exactly without validation labels.", "",
        "![Query perturbation diagnostics](query_perturbation_2026-09-22.png)", ""]
    (HERE / "REPORT_2026-09-22.md").write_text("\n".join(report))


if __name__ == "__main__":
    main()
