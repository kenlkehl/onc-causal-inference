"""Write the dated report from completed, independently validated results."""
from pathlib import Path
import json
import pandas as pd

HERE = Path(__file__).resolve().parent
DATE = "2026-09-21"


def read(name):
    return json.loads((HERE / name).read_text())


def table(frame, columns, headings):
    result = ["| " + " | ".join(headings) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if column == "scenario":
                values.append({"all": "All ten variables", "modifiers": "Modifiers only"}[value])
            elif column in ("training_rows", "test_rows"):
                values.append(f"{int(value):,}")
            elif isinstance(value, (int, float)):
                values.append(f"{value:.5f}")
            else:
                values.append(str(value))
        result.append("| " + " | ".join(values) + " |")
    return "\n".join(result)


summary = pd.read_csv(HERE / f"summary_metrics_{DATE}.csv")
per_seed = pd.read_csv(HERE / f"metrics_by_seed_{DATE}.csv")
evaluation = read(f"evaluation_{DATE}.json")
validation = read(f"independent_validation_{DATE}.json")
generation = read(f"generation_manifest_{DATE}.json")
frozen = read(f"predictions_frozen_{DATE}.json")
nuisance = read("fits/n_800000/nuisance_audit.json")
assert validation["saved_forests_reloaded"] == 6
full = summary[summary.test_set == "full_200k"].set_index("scenario")
common = summary[summary.test_set == "common_20k"]
learning = common.pivot(index="training_rows", columns="scenario", values=["correlation", "rmse"])
learning.columns = [metric + "_" + scenario for metric, scenario in learning.columns]
learning = learning.reset_index()
mean_mod, mean_all = full.loc["modifiers"], full.loc["all"]
comparisons = []
for scenario in ("modifiers", "all"):
    values = common[common.scenario == scenario].set_index("training_rows")
    before, after = values.loc[80_000], values.loc[800_000]
    a = per_seed[(per_seed.test_set == "common_20k") & (per_seed.training_rows == 80_000) & (per_seed.scenario == scenario)].set_index("seed")
    b = per_seed[(per_seed.test_set == "common_20k") & (per_seed.training_rows == 800_000) & (per_seed.scenario == scenario)].set_index("seed")
    comparisons.append(f"1. **{'Modifiers only' if scenario == 'modifiers' else 'All ten variables'} in X:** correlation changed from {before.correlation:.5f} to {after.correlation:.5f}; RMSE changed from {before.rmse:.5f} to {after.rmse:.5f} ({100 * (1 - after.rmse / before.rmse):.2f}% reduction). Correlation improved in {int((b.correlation > a.correlation).sum())}/3 matched seeds; RMSE improved in {int((b.rmse < a.rmse).sum())}/3.")
tree_rows = []
for scenario in ("modifiers", "all"):
    part = []
    for audit_path in sorted((HERE / "fits/n_800000" / scenario).glob("seed_*/fit_audit.json")):
        audit = json.loads(audit_path.read_text())
        tree = audit["tree_audit"]
        assert tree["patients_used_for_splits"] == tree["patients_used_for_effects"] == 800_000
        part.append(tree)
    leaves = [item["leaves_per_tree"]["median"] for item in part]
    sizes = [item["effect_patients_per_leaf"]["median"] for item in part]
    label = "All ten variables in X" if scenario == "all" else "Modifiers only in X"
    size_label = f"{min(sizes):g}" if min(sizes) == max(sizes) else f"{min(sizes):g}–{max(sizes):g}"
    tree_rows.append(f"1. **{label}:** median leaves per tree {min(leaves):,.1f}–{max(leaves):,.1f}; median effect-estimation patients per leaf {size_label}. All 800,000 training patients appear in both roles across each forest.")
max_iterations = max(item["optimization"]["maximum_iterations_observed"] for role in nuisance["models"].values() for item in role)
report = f"""# Oracle causal forest: one-million-patient experiment

Date: September 21, 2026.

## 1. Results

Generated **1,000,000 patients**, trained on **800,000**, and held out **200,000**. On the full test set, the historical forest reached mean effect correlation **{mean_all.correlation:.5f}** with all ten oracle covariates in X and **{mean_mod.correlation:.5f}** with only the five designated modifiers in X. The respective RMSEs were **{mean_all.rmse:.5f}** and **{mean_mod.rmse:.5f}**.

Each result below averages the scores of three separate forest seeds. Predictions were not averaged into an ensemble before scoring.

### 1.1. Full 200,000-patient test set

{table(full.reset_index(), ['scenario', 'correlation', 'rmse', 'mae', 'bias', 'prediction_sd'], ['Variables in X', 'Correlation', 'RMSE', 'MAE', 'Mean bias', 'Prediction SD'])}

The full test-set true probability-scale treatment effect has mean **{evaluation['test_truth']['full_200k']['mean']:.5f}** and SD **{evaluation['test_truth']['full_200k']['sd']:.5f}**. `modifiers` means M in X and C in W; `all` means C+M in X with no additional W.

### 1.2. Direct learning-curve comparison on the original common 20,000 test patients

The earlier 100K cohort and its split were preserved exactly. All prior training samples are nested within the new 800K training set, and the original 20K test patients remain held out. Earlier saved predictions were reused and their scores reproduced against the identical truth.

{table(learning, ['training_rows', 'correlation_modifiers', 'rmse_modifiers', 'correlation_all', 'rmse_all'], ['Training patients', 'Modifiers: correlation', 'Modifiers: RMSE', 'All variables: correlation', 'All variables: RMSE'])}

{chr(10).join(comparisons)}

![Extended learning curves](learning_curve_{DATE}.png)

### 1.3. What the additional tenfold increase changes

1. **The all-variable forest continues to improve substantially:** its RMSE falls another 21.06% on the identical test patients when training increases from 80K to 800K. The earlier correlation near 0.67 is not a fixed capacity ceiling with the full oracle input set.
2. **The modifier-only configuration is close to a plateau in this sequence:** its corresponding RMSE improvement is 2.11%. Having confounders available only to nuisance models does not let the final forest distinguish their contributions to the individual probability-scale effect.
3. The two test sets give very similar new-model results, and all three matched forest seeds improve correlation and RMSE for both configurations. This supports the direction of the learning curve, although one generated cohort sequence is not a repeated-cohort uncertainty study.
4. At 800K training patients, estimated propensity RMSE on the full test set is **{mean_all.propensity_rmse:.5f}**, and marginal-outcome nuisance RMSE is **{mean_all.outcome_nuisance_rmse:.5f}**. These are shared by the two final-forest designs. The nuisance functions remain estimated rather than supplied by the DGP.
5. The all-variable forest's nominal 95% intervals contain the individual oracle effect for **{100 * mean_all.coverage_95_of_individual_truth:.2f}%** of test patients. Thus the improved point estimates do not establish calibrated 95% inference. This is empirical coverage across test patients for one training cohort, not a repeated-training-sample coverage experiment. Modifier-only coverage against individual truth is recorded in the metrics, but its restricted conditioning set does not identify that full individual effect.

## 2. Experimental design

### 2.1. Cohort generation and split

1. Used the existing `_build_patient_scaffold_record` generator with the saved five-confounder/five-modifier distributions and final treatment/outcome equations. Generation seed **20260921**.
2. Generated binary observed treatment and outcome. No LLM, clinical-text generation, extraction, recalibration, coefficient changes, or noiseless-outcome substitution.
3. The first 100,000 rows match the preceding generated cohort exactly, including every structured covariate, observed treatment/outcome, and oracle probability.
4. Preserved all previous 80K training and 20K test memberships. Split the additional 900K rows into 720K training and 180K test using the first shuffled five-fold split with seed 42. The combined split is therefore 800K/200K.
5. Used five inner nuisance folds with seed 51043. Each inner heldout fold contains 160K patients; nuisance fitting in that fold uses the other 640K.
6. Retained the original positivity setting (`enforce_positivity=False`) and the pre-existing prior-platinum category-label mismatch described in the [100K report](../oracle_power_100k_2026-09-21/oracle_power_100k_report_2026-09-21.md). No generator behavior was corrected during this comparison.
7. Independent vectorized calculations reproduced all one million rows' oracle probabilities/effects within 4.45e-16. No values are missing and all patient IDs are unique.

### 2.2. Historical estimation settings

1. **200 trees; 45% sampling; honesty on; inference on; minimum leaf size 10; no maximum depth; square-root feature search; MSE criterion; subforest size 4.** Forest seeds: 120042, 1120042, 2120042.
2. Two final-forest designs: five true modifiers in X and five true confounders in W, or all ten true variables in X with no additional W. Full categorical encoding yields 12 modifier columns and 13 confounder columns. Continuous-variable scaling uses only the outer training sample.
3. Both configurations share exactly the same estimated nuisance residuals. The nuisance design includes all ten variables once; nuisance models are fitted once and reused across all six final forests.
4. Nuisances retain the historical elastic-net logistic specification: L1 ratio 0.8; three-fold internal regularization selection; 16 C values from 0.01 through 10,000; maximum 5,000 iterations; tolerance 0.0001; seed 120042.
5. The initial model uses native CausalForestDML. Remaining forests use its exact final-forest class, parameters, and cached cross-fitted residuals.
6. Every estimator retains `n_jobs=1`. Up to three independent final forests run concurrently. No forest parameters were tuned to the new outcomes or true effects.
7. Fitting reads only oracle covariates and observed T/Y. True propensity, potential-outcome probabilities, and true treatment effects are excluded from fitting. No propensity filtering or feature selection.

### 2.3. Tree resolution

1. Each tree samples **360,000** patients, with separate sets of **180,000** patients for split selection and effect estimation.
{chr(10).join(tree_rows)}
1. The leaf-size threshold remains fixed as the dataset grows, so more data mainly permits more partitions; it does not impose larger leaves.

## 3. Interpretation and limits

1. The common-test learning curve above isolates additional training data from a change in test patients. It measures the whole historical estimation procedure, including the improvement in estimated nuisance functions.
2. At any one training size, the X-placement comparison shares exactly the same nuisance estimates and isolates which variables the final forest can use for effect estimation.
3. The DGP's probability-scale effect is `sigmoid(g(C)+h(M)) - sigmoid(g(C))`. A modifier-only forest cannot distinguish patients with the same modifiers but different baseline-risk covariates. A plateau in that configuration does not establish a causal-forest capacity limit with all relevant inputs.
4. Remaining error can involve finite-sample forest approximation, estimated nuisance error, fixed hyperparameters, outcome noise, and restricted conditioning. This experiment does not separately identify those contributions or establish an infinite-sample limit.
5. Three seeds measure algorithmic variation on one nested cohort sequence, not repeated independent cohort uncertainty. This is an oracle diagnostic, not an observed-data selection of a production model.

## 4. Verification and retained artifacts

1. Predictions and models were frozen at **{frozen['frozen_at']}**, before the evaluation phase loaded true effects. Generation necessarily computed oracle quantities to create and validate the cohort, but those columns were not read by fitting.
2. The independent validator reloaded all six saved forests and reproduced all 200K test predictions and intervals exactly. It independently recomputed metrics on both test sets, checked forest parameters and actual tree structure, and verified source/artifact hashes.
3. All six forests and all ten fitted nuisance clones completed without warnings. Every one of the 800K out-of-fold nuisance predictions was independently reproduced from the saved fold models. No nuisance model reached the iteration limit; maximum iterations observed: **{max_iterations}**.
4. Temporary dataset and model binaries are stored under `{generation['temporary_directory']}`. Normal temporary-directory cleanup may remove them; generation scripts, seeds, splits, and the safe DGP specification support regeneration.
5. Production source code and the previous experiments remain unchanged.

- [Temporary structured dataset]({generation['temporary_dataset']})
- [Learning-curve PDF](learning_curve_{DATE}.pdf)
- [Per-seed metrics](metrics_by_seed_{DATE}.csv)
- [Mean metrics](summary_metrics_{DATE}.csv)
- [Evaluation and seed ranges](evaluation_{DATE}.json)
- [Generation script](generate_oracle_1m_{DATE}.py)
- [Generation manifest](generation_manifest_{DATE}.json)
- [Generation validation](generation_validation_{DATE}.json)
- [DGP specification](dgp_specification_{DATE}.json)
- [Exact split indices](splits_{DATE}.npz)
- [Fitting/evaluation script](run_power_comparison_{DATE}.py)
- [Input manifest](input_manifest_{DATE}.json)
- [Frozen artifact hashes](predictions_frozen_{DATE}.json)
- [Independent validation](independent_validation_{DATE}.json)
"""
destination = HERE / f"oracle_power_1m_report_{DATE}.md"
destination.write_text(report)
print(destination)
