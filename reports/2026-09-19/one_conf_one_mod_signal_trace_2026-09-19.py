"""Post-hoc audit of frozen artifacts; never refits or changes the study run."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = ROOT / "artifacts/research_all_evidence/one_conf_one_mod_nsclc_full/stage2_pre_roles_refactor"
DATA = ROOT / "synthetic_data/example_synthetic_datasets/one_confounder_one_effect_modifier_nsclc_with_structured"
PDL1 = re.compile(r"pd[\W_]*l[\W_]*1|programmed.{0,12}death|tumou?r.{0,12}proportion", re.I)
GROUPS = ["<1%", "1-49%", ">=50%"]
sources = {}
cache = {}


def frozen(path):
    content = path.read_bytes()
    sources[str(path.relative_to(ROOT))] = {
        "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)
    }
    return content


def read_json(path):
    if path not in cache:
        cache[path] = json.loads(frozen(path))
    return cache[path]


def category(value):
    if pd.isna(value):
        return "missing"
    s = str(value).lower().replace("≥", ">=").replace("≤", "<=")
    s = re.sub(r"[–—−‑]", "-", s)
    s = re.sub(r"\s+|%", "", s).replace("tps", "")
    if "1-49" in s or "1to49" in s:
        return "1-49%"
    if "<1" in s or "lessthan1" in s:
        return "<1%"
    if ">=50" in s or ">50" in s or "greaterthan50" in s:
        return ">=50%"
    try:
        x = float(s)
    except ValueError:
        return "unparsed"
    if not np.isfinite(x) or not 0 <= x <= 100:
        return "unparsed"
    return "<1%" if x < 1 else ("1-49%" if x < 50 else ">=50%")


def metrics(predicted, actual):
    p = np.asarray(predicted, dtype=float)
    a = np.asarray(actual, dtype=float)
    return {
        "n": len(a), "pearson": float(np.corrcoef(p, a)[0, 1]),
        "rmse": float(np.sqrt(np.mean((p - a) ** 2))),
        "bias": float(np.mean(p - a)), "predicted_sd": float(np.std(p)),
        "oracle_sd": float(np.std(a)), "negative_predictions": int((p < 0).sum()),
    }


def measurement(frame, name, truth):
    joined = frame[["_oci_row_id", name]].merge(truth, on="_oci_row_id", validate="one_to_one")
    mapped = joined[name].map(category)
    actual = joined["true_pdl1_expression"].map(category)
    present = ~mapped.isin(["missing", "unparsed"])
    counts = pd.crosstab(actual, mapped).reindex(index=GROUPS, columns=GROUPS + ["missing", "unparsed"], fill_value=0)
    return {
        "rows": len(joined), "missing": int(joined[name].isna().sum()),
        "unparsed": int((mapped == "unparsed").sum()),
        "correct": int((mapped == actual).sum()),
        "wrong_category": int((present & (mapped != actual)).sum()),
        "accuracy_all_rows": float((mapped == actual).mean()),
        "accuracy_nonmissing": float((mapped[present] == actual[present]).mean()),
        "missing_with_pdl1_text_mention": int((~present & joined["text_mentions_pdl1"]).sum()),
        "confusion_matrix": {g: {c: int(v) for c, v in counts.loc[g].items()} for g in GROUPS},
        "unparsed_examples": joined.loc[mapped == "unparsed", name].astype(str).value_counts().head(10).to_dict(),
    }, joined.assign(measured_category=mapped.to_numpy(), true_category=actual.to_numpy())


# Freeze every analyzed pipeline leaf before joining oracle values. Raw Stage 1
# member files are not rescanned: the saved compiled cards and discovery lineage
# establish that the PD-L1 concept crossed the Stage 1 -> Stage 2 boundary.
prediction_path = RUN / "cross_fitted_predictions.csv"
frozen(prediction_path)
evaluation = read_json(RUN / "posthoc_oracle_ite_metrics.json")
assert sources[str(prediction_path.relative_to(ROOT))]["sha256"] == evaluation["frozen_prediction_sha256"]
for fold in range(1, 6):
    directory = RUN / f"outer_{fold:03d}"
    for rel in [
        "feature_definitions.json", "final_definitions.json", "interpreted_candidates.json",
        "selection/statistical_evidence.json", "selection/role_adjudication/response.json",
        "estimation/diagnostics.json",
    ]:
        read_json(directory / rel)
    for rel in [
        "extraction/all_candidates_fit/extracted.csv", "extraction/fit/harmonized.csv",
        "extraction/heldout/extracted.csv", "extraction/heldout/harmonized.csv",
        "ontology_supervision/round_001/extraction/extracted.csv",
    ]:
        frozen(directory / rel)
    frozen(RUN / "evidence_compilation" / f"outer_{fold:03d}" / "cards.jsonl")
configuration = read_json(RUN / "config.json")
completion = read_json(RUN / "complete.json")

frozen(DATA / "metadata.json")
frozen(DATA / "dataset.parquet")
truth = pd.read_parquet(DATA / "dataset.parquet", columns=[
    "patient_id", "clinical_text", "true_age", "true_pdl1_expression",
    "true_ite_prob", "true_treatment_prob", "true_y0_prob", "true_y1_prob",
    "treatment_indicator", "outcome_indicator",
]).reset_index(drop=True).rename_axis("_oci_row_id").reset_index()
truth["text_mentions_pdl1"] = truth["clinical_text"].fillna("").map(lambda text: bool(PDL1.search(text)))
truth = truth.drop(columns="clinical_text")
predictions = pd.read_csv(prediction_path)
assert len(predictions) == len(truth) == 1000
assert set(predictions["_oci_row_id"]) == set(truth["_oci_row_id"])
if "patient_id" in predictions:
    joined_ids = predictions.merge(truth[["_oci_row_id", "patient_id"]], on="_oci_row_id")
    assert (joined_ids["patient_id_x"].astype(str) == joined_ids["patient_id_y"].astype(str)).all()

results = {
    "date": "2026-09-19", "run": str(RUN.relative_to(ROOT)),
    "completed_at": completion["completed_at"], "selection_mode": "llm_roles",
    "selection_consolidation_enabled": configuration["selection_consolidation"]["enabled"],
    "frozen_prediction_hash_matches_saved_evaluation": True,
    "method": "Post-hoc frozen-artifact audit. No LLM requests or causal model refits. Oracle lookup diagnostics use training-fold oracle effects only and are not deployable causal estimators.",
    "folds": [], "source_files": sources,
}
for fold in range(1, 6):
    directory = RUN / f"outer_{fold:03d}"
    original = read_json(directory / "feature_definitions.json")["features"]
    selected = read_json(directory / "final_definitions.json")["features"]
    feature = next(f for f in selected if "pd_l1" in f["name"])
    name, fid = feature["name"], feature["feature_id"]
    initial_feature = next(f for f in original if f["feature_id"] == fid)
    statistic = read_json(directory / "selection/statistical_evidence.json")
    decision = next(f for f in statistic["decisions"] if f["feature_id"] == fid)
    role = next(f for f in read_json(directory / "selection/role_adjudication/response.json")["decisions"] if f["feature_id"] == fid)
    candidates = read_json(directory / "interpreted_candidates.json")
    named = [c for c in candidates if PDL1.search(c.get("name", "") + " " + c.get("description", ""))]
    cards = [json.loads(line) for line in (RUN / "evidence_compilation" / f"outer_{fold:03d}" / "cards.jsonl").open()]
    visible_cards = [c for c in cards if any(PDL1.search(r.get("text", "")) for r in c["representative_evidence"])]
    result = {
        "outer_fold": fold, "feature_id": fid, "name": name,
        "stage1_derived_cards_with_pdl1_text": len(visible_cards),
        "card_architectures_with_pdl1_text": sorted({a for c in visible_cards for a in c["source_architectures"]}),
        "pdl1_discovery_candidates": len(named),
        "pdl1_discovery_architectures": sorted({c["architecture"] for c in named}),
        "initial_ontology": {k: initial_feature.get(k) for k in ["value_type", "categories_or_unit", "measurement_definition"]},
        "final_ontology": {k: feature.get(k) for k in ["value_type", "categories_or_unit", "measurement_definition", "roles"]},
        "statistical_decision": decision, "llm_decision": role,
        "age_roles": next(f["roles"] for f in selected if f["name"] == "age"),
        "measurement": {},
    }
    frames = {}
    stages = {
        "first_round_training": "ontology_supervision/round_001/extraction/extracted.csv",
        "final_raw_training": "extraction/all_candidates_fit/extracted.csv",
        "final_model_training": "extraction/fit/harmonized.csv",
        "raw_heldout": "extraction/heldout/extracted.csv",
        "model_heldout": "extraction/heldout/harmonized.csv",
    }
    for label, rel in stages.items():
        frame = pd.read_csv(directory / rel)
        result["measurement"][label], joined = measurement(frame, name, truth)
        frames[label] = (frame, joined)
    training, train_join = frames["final_model_training"]
    heldout, test_join = frames["model_heldout"]
    assert len(training) == 800 and len(heldout) == 200
    assert not set(training["_oci_row_id"]) & set(heldout["_oci_row_id"])
    fold_predictions = predictions.merge(test_join, on="_oci_row_id", validate="one_to_one", suffixes=("", "_truth"))
    assert len(fold_predictions) == 200
    if "treatment" in fold_predictions:
        assert (fold_predictions["treatment"] == fold_predictions["treatment_indicator"]).all()
        assert (fold_predictions["outcome"] == fold_predictions["outcome_indicator"]).all()
    result["final_predictions"] = metrics(fold_predictions.estimated_cate, fold_predictions.true_ite_prob)
    result["pdl1_group_predictions"] = {
        label: {"n": len(group), "true_effect": float(group.true_ite_prob.mean()),
                "estimated_effect": float(group.estimated_cate.mean()),
                "negative_predictions": int((group.estimated_cate < 0).sum())}
        for label, group in fold_predictions.groupby("true_category")
    }
    correctly_measured = fold_predictions[
        fold_predictions.measured_category == fold_predictions.true_category
    ]
    result["correctly_measured_group_predictions"] = {
        label: {"n": len(group), "true_effect": float(group.true_ite_prob.mean()),
                "estimated_effect": float(group.estimated_cate.mean())}
        for label, group in correctly_measured.groupby("true_category")
    }
    group_means = result["pdl1_group_predictions"]
    true_gap = group_means[">=50%"]["true_effect"] - group_means["1-49%"]["true_effect"]
    estimated_gap = group_means[">=50%"]["estimated_effect"] - group_means["1-49%"]["estimated_effect"]
    result["high_minus_middle_contrast"] = {
        "true_effect_gap": true_gap, "estimated_effect_gap": estimated_gap,
        "fraction_retained": estimated_gap / true_gap,
    }
    raw_heldout = frames["raw_heldout"][0].set_index("_oci_row_id")[name]
    model_heldout = heldout.set_index("_oci_row_id")[name].reindex(raw_heldout.index)
    dropped = raw_heldout.notna() & model_heldout.isna()
    result["harmonization_newly_missing_heldout"] = {
        "n": int(dropped.sum()),
        "raw_values": raw_heldout[dropped].astype(str).value_counts().to_dict(),
    }
    result["true_negative_heldout_encoded_as_exactly_one"] = int((
        (test_join.true_category == "<1%")
        & (pd.to_numeric(test_join[name], errors="coerce") == 1)
    ).sum())
    for group_key, output_key in [("true_category", "oracle_true_category_lookup"), ("measured_category", "oracle_measured_category_lookup")]:
        means = train_join.groupby(group_key).true_ite_prob.mean()
        oracle_predictions = fold_predictions[group_key].map(means).fillna(train_join.true_ite_prob.mean())
        result[output_key] = metrics(oracle_predictions, fold_predictions.true_ite_prob)
    result["candidate_r_loss_by_inner_fold"] = [
        {"inner_fold": row["inner_fold"], **{k: test.get(k) for k in ["status", "rank", "selected_top_n", "heldout_r_loss_improvement", "heldout_relative_r_loss_improvement", "tested_interaction_columns"]}}
        for row in statistic["effect_modifier_screen"]["folds"]
        for test in row["tests"] if test["feature_id"] == fid
    ]
    result["joint_r_loss_by_inner_fold"] = [
        {"inner_fold": row["inner_fold"], "pdl1_selected": fid in row["selected_feature_ids"],
         "total_selected": len(row["selected_feature_ids"]), "regularization_alpha": row["regularization_alpha"],
         "heldout_r_loss_improvement": row["heldout_r_loss_improvement"]}
        for row in statistic["multivariable_modifier_elastic_net_screen"]["folds"]
    ]
    x = [f for f in selected if "effect_modifier" in f["roles"]]
    w = [f for f in selected if f["roles"] == ["confounder"]]
    result["forest_design"] = {
        "source": "Saved final feature definitions; counts are concepts, not encoded matrix columns.",
        "effect_features": len(x), "control_features": len(w),
        "effect_feature_names": [f["name"] for f in x],
        "control_feature_names": [f["name"] for f in w],
    }
    result["age_measurement"] = {
        "train_mae_nonmissing": float(np.nanmean(np.abs(pd.to_numeric(training.age, errors="coerce").to_numpy() - truth.set_index("_oci_row_id").loc[training._oci_row_id, "true_age"].to_numpy()))),
        "age_in_forest_x": any(f["name"] == "age" for f in x),
    }
    results["folds"].append(result)

target = OUT / "one_conf_one_mod_signal_trace_2026-09-19.json"
target.write_text(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
for f in results["folds"]:
    print(json.dumps({
        "fold": f["outer_fold"], "cards": f["stage1_derived_cards_with_pdl1_text"],
        "candidates": f["pdl1_discovery_candidates"],
        "train_measurement": f["measurement"]["final_model_training"],
        "test_measurement": f["measurement"]["model_heldout"],
        "forest": f["forest_design"], "predictions": f["final_predictions"],
        "oracle_measured_lookup": f["oracle_measured_category_lookup"],
        "groups": f["pdl1_group_predictions"],
    }, ensure_ascii=False))
print(target)
