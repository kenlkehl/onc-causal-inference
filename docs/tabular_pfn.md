# Sparse Tabular CausalPFN (`oci.tabular_pfn`)

`oci.tabular_pfn` implements `SparseTabularCausalPFN`, a zero-shot tabular causal foundation model designed specifically for high-dimensional, noisy, sparse-signal candidate tables such as those produced by OCI's all-evidence Stage-2 workflow (scaling cleanly to $P = 2{,}000+$ candidate columns and $N = 10{,}000+$ patients).

## Architecture Overview

Unlike vanilla TabPFN / CausalPFN architectures whose $O(N P^2)$ attention and dense L2 shrinkage degrade sharply when 90–99% of candidate columns are noise or Stage-1-induced instruments, `SparseTabularCausalPFN` uses a two-round sparse-gated orthogonalized architecture:

1. **Context-Only Standardization & $O(N \cdot P)$ Observational Column Statistics** (`_column_sufficient_stats`):
   Computes 24 strictly observational descriptors per column (propensity linear/quadratic $t$-stats, overall and arm-stratified outcome $t$-stats within $T=0$ and $T=1$, arm-difference interaction $z$-stats, missingness rate, distributional shape, cohort scale $\log N, \log P$, and fast redundancy/collinearity proxies against the top-32 signal columns). Because pure instruments show strong $T$ association but near-zero within-arm $Y$ association, the model sees the instrument signature on step 0.
2. **$O(P)$ Column Role Encoder & Round-1 Top-$K_{\text{nuis}}$ Correlation-Biased Attention**:
   A permutation-equivariant DeepSets encoder screens all $P$ columns and selects the top-$K_{\text{nuis}}$ (default `64`) nuisance candidates. Pairwise column self-attention biased by the empirical $K_{\text{nuis}} \times K_{\text{nuis}}$ correlation matrix resolves collinear duplicates and outputs soft ARD spike-and-slab gates $(g_e, g_m)$ for linear and nonlinear ($X^2$) terms, feeding exact differentiable closed-form LOO-CV Ridge fits for $\hat{e}^{(-i)}$ and $\hat{m}^{(-i)}$.
3. **Round-2 All-$P$ Adjusted Robinson Interaction Scan & ARD R-Learner**:
   Using the Round-1 adjusted residuals $(T - \hat{e}^{(-i)}, Y - \hat{m}^{(-i)})$, the model scans **all $P$ columns** in a single $O(N \cdot P)$ pass to compute exact adjusted Robinson interaction scores, refines its 6-class Bayesian role posteriors (`noise`, `instrument`, `prognostic`, `confounder`, `modifier`, `dual_role`), promotes masked effect modifiers into the top-$K_{\text{cate}}$ (default `48`) active set, applies correlation-biased attention for soft ARD gates $g_\tau$, and fits the differentiable LOO-CV Robinson R-learner.
4. **Residual Two-Way TabPFN Study Layers & 129-Bin Sinh Posterior Head**:
   Two lightweight two-way attention layers over the top `attn_slots` (default `16`) gated columns refine pairwise feature interactions and calibrate arm-specific risk means, epistemic SDs, and propensity logits.

## Synthetic 6-Family Interacting & Embedding Hybrid Causal Prior (`oci/tabular_pfn/prior.py`)

Training draws log-uniform $N_{\text{ctx}} \in [80, 8{,}000]$ and $P \in [20, 3{,}500]$ (v3/v4) from a de-aliased 6-family mixture prior with real Stage-1 LLM extraction pathologies and rich confounder/modifier interactions:

1. **`sparse_oci` (20%):** High-dimensional sparse all-evidence tables with mixed clinical types (binary, ordinal, Gaussian, log-normal, heavy-tailed Student-$t$ lab outliers, zero-inflated utilization counts), multi-level Dirichlet one-hot categorical blocks, strong pure instruments, collinear noisy proxies, optional dense weak-confounding regimes (`15%`), rich nonlinearities (including non-monotonic clinical reference bands and asymmetric sigmoids) + confounder×confounder, confounder×modifier, and modifier×modifier interactions (`20%` pure-epistatic HTE), optional unmeasured confounding (creating Z-bias on instruments), Stage-1 LLM extraction noise, MCAR/MAR/**MNAR** + 3-state EHR documentation, occasional RCTs (8%), 25% strictly null-HTE studies, and both `logOR` and bounded additive `RD` scales.
2. **`reequation_calc` (12%):** Vectorized implementation of OCI's `synthetic_data/reequation.py` and `sample_size_calculator/engine.R` tabular DGP family with optional confounder×confounder and confounder×modifier interactions embedded in $P = 20\text{–}3{,}500$.
3. **`nemotron_design` (18%):** Direct tabular studies sampled strictly from the 4,000 `split == "train"` oncology designs in `synthetic_data/designs/nemotron_5000.json.gz` (plus clinical pairwise interactions) embedded in $P = 20\text{–}3{,}500$. The 1,000 `split == "held_out"` designs (including 3 unseen cancers) are strictly excluded from training and used automatically inside `evaluate`.
4. **`tree_subgroup` (18%):** Clinical decision-tree / multi-way biomarker rule-list subgroup DGP (depth 2–4 logical conjunctions including explicit confounder×modifier rules, crisp/soft cutoffs, piecewise subgroups) requiring genuine nonlinear interaction discovery beyond linear/quadratic ARD Ridge.
5. **`causal_scm` (16%):** General layered nonlinear Structural Causal Model (DAG + 2-layer MLP treatment, outcome, and joint confounder×modifier HTE surfaces) with explicit observed-parent ($Z \to C \leftarrow W$) and latent-parent ($U_T \to C \leftarrow U_Y$) M-bias colliders, heavy tails, dense confounding, and near-positivity extremes.
6. **`interacting_clinical` (16%):** Interaction-heavy clinical & multimodal-embedding hybrid prior featuring coupled confounder×confounder propensity interactions, prognostic×confounder outcome interactions, strong confounder×modifier / prognostic×modifier / modifier×modifier HTE (`30%` mathematically orthogonal pure-epistatic crossover HTE with zero marginal modifier slope), and in 50% of wide cohorts a bounded correlated cosine-MaxSim embedding block mimicking frozen clinical-text representations (e.g., `TinyOncBERT SentKM`) coupled with structured covariates.
- **De-Aliased High-Missingness & Fragmented Stage-1 Synonym Clusters (applied across all 6 families, `55%`):** Splits both active and noise features into 2–6 synonymous opposite-polarity, coarsened, and partial-coverage (`NaN`-masked) columns alongside heavy-tailed EHR missingness up to `95%` so the model pools across fragmented Stage-1 extractions with zero missingness-rate shortcut (`miss_rate` AUROC `~0.51–0.53` across all 6 families).

## Usage

### Train on 1× H200 or RTX A6000
```bash
# v1 compact (2.11M params, ~2.3 GiB peak VRAM, 12,000 steps)
./scripts/run_tabular_pfn_h200.sh

# v2 (7.38M params, ~6-7 GiB peak VRAM with bf16 FlashAttention-2, 15,000 steps):
# decoupled outcome-adaptive R-learner propensity + 4-layer/192-wide/24-slot TabPFN + wider top-K recall
./scripts/run_tabular_pfn_v2_h200.sh

# v3 H200 flagship (18.88M params, N_ctx = 80..8,000, P = 20..3,500, 30,000 steps / 360k studies):
# 5-family prior + fragmented extractions + M-bias colliders + 4-channel ARD basis (X, X^2, tanh, M_c)
# + 6-d pairwise-complete co-observed correlation attention + 6-layer/256-wide/32-slot TabPFN
./scripts/run_tabular_pfn_v3_h200.sh

# v4 H200 flagship (34.60M params, N_ctx = 80..8,000, P = 20..3,500, 40,000 steps / 480k studies):
# de-aliased 6-family interacting + cosine-MaxSim embedding hybrid prior + O(8*N*P) confounder-modulated
# Robinson scan + symmetric pairwise interaction basis inside closed-form ARD solver (pair_top_k=16 -> 120 pairs)
# + 8-layer/320-wide/40-slot TabPFN with automatic gradient checkpointing above N_ctx=4,000 (24.19 GiB worst-case VRAM)
./scripts/run_tabular_pfn_v4_h200.sh
# Dedicated H200: NUM_WORKERS=16 ./scripts/run_tabular_pfn_v4_h200.sh
# Sharing host RAM with another training run: NUM_WORKERS=6 ./scripts/run_tabular_pfn_v4_h200.sh
```

### Zero-Shot Prediction + Calibrated Feature Role Report on Any Tabular Cohort
```bash
# Out-of-fold zero-shot predictions for every patient in a Stage-2 cohort (5-fold cross-fitted)
# Supports .json or .csv for --role-report and prints top confounders, modifiers, & instruments
python -m oci.tabular_pfn.cli predict \
  --run-dir artifacts/tabular_causal_pfn_p2000 \
  --context path/to/stage2_catalog.parquet \
  --cv-folds 5 \
  --output predictions.parquet \
  --role-report feature_roles.csv
```
(You can also pass explicit `--context` + `--queries` tables, `--context-size K` for a single holdout split, or `--print-roles` to display the summary table without writing a file. By default, role probabilities are calibrated to the cohort's empirical sparsity via Normalized Saerens EM.)

`TabularPreprocessor` automatically handles mixed numeric, binary, categorical, and 3-state EHR missingness (`not_documented` vs. `documented_unknown`) strictly fitted on context rows with zero query or oracle leakage.

### Feature Role Report & CLI Reference

Every column in `--role-report` (`.csv` or `.json`) reports:
- **Calibrated role probabilities** (adjusted from the 50/50 balanced training prior to the cohort's empirical sparsity via Normalized Saerens EM, preserving exact rank order): `p_signal` ($1 - p_{\text{noise}}$), `p_any_confounder` ($p_{\text{confounder}} + p_{\text{dual\_role}}$), `p_any_modifier` ($p_{\text{modifier}} + p_{\text{dual\_role}}$), plus all six individual probabilities (`p_noise`, `p_instrument`, `p_prognostic`, `p_confounder`, `p_modifier`, `p_dual_role`) and `predicted_role` (most likely non-noise role when `p_signal >= 0.50`, else `"noise"`).
- **Solver ARD gate weights** (`0.0–1.0`): `gate_propensity` ($g_e$), `gate_outcome` ($g_m$), and `gate_cate` ($g_\tau$) — the actual feature weights used by the differentiable LOO-CV nuisances and Robinson R-learner (`0.0` if filtered out before top-$K$).
- **Raw uncalibrated score**: `raw_signal_score` ($1 - p_{\text{noise}}^{\text{raw}}$).

Key CLI flags across subcommands (`python -m oci.tabular_pfn.cli {train,resume,evaluate,predict}`):
- `predict`: `--cv-folds K` (stratified out-of-fold predictions for all $N$ patients + fold-averaged role report), `--context-size K` or `--queries FILE`, `--role-report PATH.{csv,json}`, `--print-roles` (print summary without writing a file), `--role-summary-top-k 15` (`0` disables stdout table), `--uncalibrated-roles` (disable EM sparsity calibration), `--query-batch-size 512`.
- `evaluate`: `--num-features P`, `--family {all,sparse_oci,reequation_calc,nemotron_design,tree_subgroup,causal_scm,interacting_clinical}` (always locks `nemotron_design` to the 1,000 `split == "held_out"` designs), `--no-baselines`.
- `resume`: `--run-dir DIR --steps TOTAL_STEPS` resumes optimizer state, model config, and saved prior seamlessly.

## Proof-of-Concept Stress-Test Benchmark (`P = 50, 500, 2,000`)

Measured after a 6.1-minute, 500-step proof-of-concept training run (2.11M parameters, log-uniform `N_ctx = 80..1,200`, `P = 20..2,000`, 1× RTX A6000), evaluated across 48 held-out synthetic all-evidence studies per width (`N_ctx = 800`, `N_q = 128`, 25% strictly null-HTE, 70% strong instruments, collinear proxies, mixed types, MCAR/MAR missingness):

| Candidate Width `P` | **PFN `cate_rmse` ↓** | Arm-Mean ↓ | Dense Ridge-R ↓ | Screened-32 Ridge-R ↓ | Cross-Fit GBM-R ↓ | **PFN `within_r` ↑** (vs. Scr-32) | **PFN `null_rmse` ↓** | Signal / Modifier AUROC ↑ | `g_e` (Conf / Inst) | Median Infer / Peak VRAM |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`P = 50`** | **`0.1209`** | `0.1521` | `0.1639` | `0.1585` | `0.2325` | **`+0.473`** (`+0.391`) | `0.0724` | `0.838 / 0.806` | `0.59 / 0.43` | `20.5 ms` / `134 MB` |
| **`P = 500`** | **`0.1246`** | `0.1490` | `0.1480` | `0.1634` | `0.2392` | **`+0.440`** (`+0.324`) | `0.0765` | `0.872 / 0.878` | `0.53 / 0.33` | `22.0 ms` / `143 MB` |
| **`P = 2,000`** | **`0.1323`** | `0.1630` | `—` | `0.1467` | `0.2376` | **`+0.388`** (`+0.302`) | `0.0711` | **`0.863 / 0.866`** | **`0.42 / 0.29`** | **`27.5 ms` / `170 MB`** |

For the full architectural iteration history, bug-fix notes, 3-family zero-shot out-of-distribution transfer results, and ongoing H200 training run tracker, see [`docs/tabular_pfn_experiments.md`](tabular_pfn_experiments.md).
