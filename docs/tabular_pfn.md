# Sparse Tabular CausalPFN (`oci.tabular_pfn`)

`oci.tabular_pfn` implements `SparseTabularCausalPFN`, a zero-shot tabular causal foundation model designed specifically for high-dimensional, noisy, sparse-signal candidate tables such as those produced by OCI's all-evidence Stage-2 workflow (scaling cleanly to $P = 2{,}000+$ candidate columns and $N = 10{,}000+$ patients).

## Architecture Overview

Unlike vanilla TabPFN / CausalPFN architectures whose $O(N P^2)$ attention and dense L2 shrinkage degrade sharply when 90–99% of candidate columns are noise or Stage-1-induced instruments, `SparseTabularCausalPFN` uses a two-round sparse-gated orthogonalized architecture:

1. **Context-Only Standardization & $O(N \cdot P)$ Observational Column Statistics** (`_column_sufficient_stats`):
   Computes 24 (v1/v2) or 32 (v3–v5, adding smooth saturating $\tanh(1.5 X)$ and variance-gated missingness $M_c$ channels) strictly observational descriptors per column (propensity linear/quadratic $t$-stats, overall and arm-stratified outcome $t$-stats within $T=0$ and $T=1$, arm-difference interaction $z$-stats, missingness rate, smooth $C^\infty$ distributional shape, cohort scale $\log N, \log P$, and fast redundancy/collinearity proxies against the top-32 signal columns). Because pure instruments show strong $T$ association but near-zero within-arm $Y$ association, the model sees the instrument signature on step 0.
2. **$O(P)$ Column Role Encoder & Round-1 Top-$K_{\text{nuis}}$ Correlation-Biased Attention**:
   A permutation-equivariant DeepSets encoder screens all $P$ columns and selects the top-$K_{\text{nuis}}$ (`64` v1 → `128` v3/v4 → `160` v5) nuisance candidates. Pairwise column self-attention biased by the empirical correlation matrix (`6`-d pairwise-complete co-observed correlation + overlap in v3+) resolves collinear/fragmented duplicates and outputs soft ARD spike-and-slab gates $(g_e, g_{e_r}, g_m)$ — with exact zero `prop_loss` gradient into decoupled R-learner propensity $g_{e_r}$ (`--decouple-rlearner-propensity`) — plus symmetric pairwise interaction gates (`pair_top_k=16→20` in v4/v5), feeding exact differentiable closed-form LOO-CV Ridge fits for $\hat{e}^{(-i)}$, $\hat{e}_r^{(-i)}$, and $\hat{m}^{(-i)}$.
3. **Round-2 All-$P$ Adjusted Robinson & Epistasis Scans + ARD R-Learner**:
   Using the Round-1 adjusted residuals $(T - \hat{e}_r^{(-i)}, Y - \hat{m}^{(-i)})$, the model scans **all $P$ columns** for 1D adjusted Robinson interaction scores, adds an exact $O(8 N P)$ centered confounder-modulated `C × M` scan (`--use-v4-interactions`), and in **v5 (`--use-v5-epistasis`)** runs a memory-bounded exact tiled all-pairs symmetric Robinson epistasis scan (`_robinson_epistasis_all_pairs`, `ROB_STAT_DIM = 20`) over all $P(P-1)/2$ pairs with smooth orthogonal deduplication $\sqrt{1 - \rho^2}$ and Bonferroni excess calibration $\operatorname{softplus}(\max |Z| - \sqrt{4\log P})$. This rescues strictly pure `M₁ × M₂` crossover epistasis (zero main effects on $T$, $Y$, or 1D HTE), refines 6-class Bayesian role posteriors (`noise`, `instrument`, `prognostic`, `confounder`, `modifier`, `dual_role`), promotes masked modifiers into top-$K_{\text{cate}}$ (`48` → `96` → `128`), and injects top-node pairs (`pair_top_k`) plus up to `--epi-pair-k` (default `32`) globally discovered pure-epistatic pairs directly into the differentiable LOO-CV Robinson R-learner.
4. **Residual Two-Way TabPFN Study Layers & 129-Bin Sinh Posterior Head**:
   Two-way alternating feature-slot and context-sample attention layers (`2L/128W/16S` v1 → `8L/320W/40S` v4 → **`8L/320W/64S` v5** with detached dense-confounder slot boost and automatic gradient checkpointing above `N_ctx = 1,200`) refine arbitrary higher-order interactions and calibrate arm-specific risk means, epistemic SDs, and propensity logits.

## Synthetic 7-Family Universal Causal Prior (`oci/tabular_pfn/prior.py`)

Training draws log-uniform $N_{\text{ctx}} \in [80, 8{,}000]$ and $P \in [20, 3{,}500]$ (v3–v5) from a de-aliased 7-family mixture prior with real Stage-1 LLM extraction pathologies, rich higher-order interactions, and dense polygenic/multi-biomarker regimes (v5 default weights `16/10/16/16/14/14/14%`; legacy 3/5/6-family configs pad automatically):

1. **`sparse_oci` (16%):** High-dimensional sparse & dense (`--dense-conf-prob 0.30` in v5: `12–42` confounders + `4–26` modifiers) all-evidence tables with mixed clinical types (binary, ordinal, Gaussian, log-normal, heavy-tailed Student-$t$ lab outliers, zero-inflated utilization counts), multi-level Dirichlet one-hot categorical blocks, strong pure instruments, collinear noisy proxies, rich nonlinearities (including non-monotonic clinical reference bands and asymmetric sigmoids) + `C×C`, `C×M`, and `M×M` interactions (`20%` pure-epistatic HTE), optional unmeasured confounding (Z-bias on instruments), Stage-1 LLM extraction noise, MCAR/MAR/**MNAR** + 3-state EHR documentation, occasional RCTs (8%), 25% strictly null-HTE studies, and both `logOR` and bounded additive `RD` scales.
2. **`reequation_calc` (10%):** Vectorized implementation of OCI's `synthetic_data/reequation.py` and `sample_size_calculator/engine.R` tabular DGP family with optional `C×C` and `C×M` interactions embedded in $P = 20\text{–}3{,}500$.
3. **`nemotron_design` (16%):** Direct tabular studies sampled strictly from the 4,000 `split == "train"` oncology designs in `synthetic_data/designs/nemotron_5000.json.gz` (plus clinical pairwise interactions) embedded in $P = 20\text{–}3{,}500$. The 1,000 `split == "held_out"` designs (including 3 unseen cancers) are strictly excluded from training and used automatically inside `evaluate`.
4. **`tree_subgroup` (16%):** Clinical decision-tree / multi-way biomarker rule-list subgroup DGP (depth 2–4 logical conjunctions including explicit `C×M` rules, crisp/soft cutoffs, piecewise subgroups).
5. **`causal_scm` (14%):** General layered nonlinear Structural Causal Model (DAG + 2-layer MLP treatment, outcome, and joint `C×M` HTE surfaces) with explicit observed-parent ($Z \to C \leftarrow W$) and latent-parent ($U_T \to C \leftarrow U_Y$) M-bias colliders, heavy tails, dense confounding, and near-positivity extremes.
6. **`interacting_clinical` (14%):** Interaction-heavy clinical & multimodal-embedding hybrid prior featuring coupled `C×C` propensity interactions, `P×C` outcome interactions, strong `C×M`/`P×M`/`M×M` HTE (`38%` pure-epistatic crossover HTE in v5, half of which are strictly pure `M₁ × M₂` with zero main effects on $T$ or $Y$), and in 50% of wide cohorts a bounded correlated cosine-MaxSim embedding block mimicking frozen clinical-text representations (e.g., `TinyOncBERT SentKM`).
7. **`universal_nonparametric` (14%, new in v5):** Firewall-clean universal nonparametric, higher-order epistatic, and ultra-dense prior mixing four general function families distinct from `S10` benchmark equations — localized Gaussian/Laplace RBF kernel mixtures, 1-to-3-way MARS spline hinge products, 3-layer Swish residual neural fields, and orthogonalized 2-/3-way epistasis — across both sparse and ultra-dense (`15–42` confounders + `10–32` modifiers) regimes.
- **De-Aliased High-Missingness & Fragmented Stage-1 Synonym Clusters (applied across all 7 families, `55%`):** Splits both active and noise features into 2–6 synonymous opposite-polarity, coarsened, and partial-coverage (`NaN`-masked) columns alongside heavy-tailed EHR missingness up to `95%` so the model pools across fragmented Stage-1 extractions with zero missingness-rate shortcut (`miss_rate` AUROC `~0.50–0.53` across all 7 families).

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

# v5 H200 universal flagship (34.60M params, N_ctx = 80..8,000, P = 20..3,500, 45,000 steps / 540k studies):
# de-aliased 7-family universal prior (+ universal_nonparametric RBF/MARS/ResMLP/3-way epistasis,
# dense_conf_prob=0.30, pure M1 x M2 epistasis) + exact tiled all-pairs Robinson epistasis scan
# (--use-v5-epistasis --epi-pair-k 32 --pair-top-k 20) + expanded dense capacity (top_k 160/128,
# attn_slots=64, auto checkpointing above N_ctx=1,200) + auto-resume & post-training Option A suite
NUM_WORKERS=16 ./scripts/run_tabular_pfn_v5_h200.sh
# Sharing host RAM with another training run: NUM_WORKERS=8 ./scripts/run_tabular_pfn_v5_h200.sh
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
- `evaluate`: `--num-features P`, `--family {all,sparse_oci,reequation_calc,nemotron_design,tree_subgroup,causal_scm,interacting_clinical,universal_nonparametric}` (always locks `nemotron_design` to the 1,000 `split == "held_out"` designs), `--no-baselines`.
- `train` (v5 flags): `--use-v5-epistasis`, `--epi-pair-k 32`, `--pair-top-k 20`, `--top-k-nuis 160`, `--top-k-cate 128`, `--attn-slots 64`, `--dense-conf-prob 0.30`.
- `resume`: `--run-dir DIR --steps TOTAL_STEPS` resumes optimizer state, model config, and saved prior seamlessly.

## Proof-of-Concept Stress-Test Benchmark (`P = 50, 500, 2,000`)

Measured after a 6.1-minute, 500-step proof-of-concept training run (2.11M parameters, log-uniform `N_ctx = 80..1,200`, `P = 20..2,000`, 1× RTX A6000), evaluated across 48 held-out synthetic all-evidence studies per width (`N_ctx = 800`, `N_q = 128`, 25% strictly null-HTE, 70% strong instruments, collinear proxies, mixed types, MCAR/MAR missingness):

| Candidate Width `P` | **PFN `cate_rmse` ↓** | Arm-Mean ↓ | Dense Ridge-R ↓ | Screened-32 Ridge-R ↓ | Cross-Fit GBM-R ↓ | **PFN `within_r` ↑** (vs. Scr-32) | **PFN `null_rmse` ↓** | Signal / Modifier AUROC ↑ | `g_e` (Conf / Inst) | Median Infer / Peak VRAM |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`P = 50`** | **`0.1209`** | `0.1521` | `0.1639` | `0.1585` | `0.2325` | **`+0.473`** (`+0.391`) | `0.0724` | `0.838 / 0.806` | `0.59 / 0.43` | `20.5 ms` / `134 MB` |
| **`P = 500`** | **`0.1246`** | `0.1490` | `0.1480` | `0.1634` | `0.2392` | **`+0.440`** (`+0.324`) | `0.0765` | `0.872 / 0.878` | `0.53 / 0.33` | `22.0 ms` / `143 MB` |
| **`P = 2,000`** | **`0.1323`** | `0.1630` | `—` | `0.1467` | `0.2376` | **`+0.388`** (`+0.302`) | `0.0711` | **`0.863 / 0.866`** | **`0.42 / 0.29`** | **`27.5 ms` / `170 MB`** |

For the full architectural iteration history, bug-fix notes, 3-family zero-shot out-of-distribution transfer results, and ongoing H200 training run tracker, see [`docs/tabular_pfn_experiments.md`](tabular_pfn_experiments.md).
