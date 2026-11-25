# Sparse Tabular CausalPFN (`oci.tabular_pfn`) — Design & Experiment Log

This document records the architectural design choices, engineering iterations, synthetic prior upgrades, and held-out benchmark results for `SparseTabularCausalPFN` (`oci.tabular_pfn`) — a zero-shot high-dimensional tabular causal foundation model built specifically for OCI's Stage-2 all-evidence extracted candidate tables (scaling cleanly to $P = 2{,}000+$ candidate features and $N = 10{,}000+$ patients).

All evaluations use strictly observational context inputs `(X_ctx, T_ctx, Y_ctx)` with zero oracle leakage and strict separation between training and held-out synthetic study designs.

---

## 1. Architectural Rationale: Why Not Vanilla TabPFN / CausalPFN?

Original CausalPFN (Balazadeh et al. / TabPFN-style in-context transformers) applies dense cell-level self-attention across all `(patient × feature)` tokens. That paradigm breaks down on OCI's Stage-2 all-evidence tables ($P \approx 2{,}000$ candidate columns, 90–99.5% noise, collinear extracted duplicates, strong Stage-1 instruments, 3-state EHR missingness, and null-HTE studies) for three reasons:
1. **Complexity:** Dense $(N \times P)$ attention scales as $O(N P^2 + P N^2)$, becoming prohibitive at $P = 2{,}000, N = 3{,}000$.
2. **Confounding & Orthogonalization Saddle:** Pure attention layers struggle to solve high-dimensional confounding adjustment and Robinson orthogonalization from scratch (observed directly in our longitudinal pure-PFN baseline, which plateaued near `within_r ≈ 0.00`).
3. **Noise Dilution & Instrument Amplification:** Uniform shrinkage or dense attention lets ~1,985 noise columns drown out 5–15 true confounders/modifiers and lets strong $T$-only instruments blow up propensity weights and induce Z-bias under residual confounding.

### Hybrid Neuro-Statistical Architecture (`SparseTabularCausalPFN`, 2.11M params)

Instead of asking a pure transformer to approximate linear algebra over 2,000 columns, `SparseTabularCausalPFN` (`oci/tabular_pfn/model.py`) separates **column-role inference & sparse ARD gating** (neural) from **orthogonalized estimation** (exact differentiable closed-form LOO-CV algebra), with a compact residual TabPFN layer on top:

| Stage | Module / Function | Complexity | Purpose |
|---|---|---|---|
| **1. Observational Column Fingerprint** | `_standardize_context`, `_column_sufficient_stats` | $O(N \cdot P)$ | Context-only standardization + centered $X^2$ channel + 24 strictly observational column stats (propensity linear/quadratic $t$-stats, overall & arm-stratified $Y$-stats within $T=0/1$, raw interaction $z$-stats, missingness, smooth endpoint/tail shape, $\log N, \log P$, top-32 collinearity proxies). Separates instruments (high $|t_t|$, low $\max(|t_{y0}|, |t_{y1}|)$) from confounders on step 0. |
| **2. Round-1 Column Role Encoder & Screener** | `col_in`, `_DeepSetsColumnBlock`, `nuis_screen_head` | $O(P)$ | Permutation-equivariant DeepSets global context exchange across all $P$ columns; supervised nuisance screener (`r_tgt >= 2`: keeps confounders/prognostics/modifiers, filters noise & pure instruments) selects top-$K_{\text{nuis}}$ (`64`). |
| **3. Round-1 Correlation-Biased Attention & ARD Nuisances** | `_CorrBiasedColumnLayer`, `nuis_gate_head`, `_fit_ard_loo_cv_ridge` | $O(K_1^2 + N K_1^2)$ | Pairwise attention across top-64 columns biased by empirical $(\rho_{jk}, |\rho_{jk}|, \rho_{jk}^2)$ resolves collinear proxies → soft ARD spike-and-slab gates `(g_e_lin, g_e_nl, g_m_lin, g_m_nl)` → differentiable closed-form batched Cholesky LOO-CV Ridge fits for $\hat{e}^{(-i)}$ and $\hat{m}^{(-i)}$. |
| **4. Round-2 All-$P$ Adjusted Robinson Scan & ARD R-Learner** | `_robinson_all_p_stats`, `rob_update`, `role_head`, `mod_screen_head`, `cate_col_layers`, `cate_gate_head` | $O(N \cdot P + N K_2^2)$ | Scans **all $P$ columns** against adjusted residuals $(T - \hat{e}^{(-i)}, Y - \hat{m}^{(-i)})$ to rescue subtle/pure-crossover modifiers masked by strong confounders → final 6-class Bayesian `role_logits` (`noise`, `instrument`, `prognostic`, `confounder`, `modifier`, `dual_role`) + top-$K_{\text{cate}}$ (`48`) correlation-biased attention → ARD gates `(g_tau_lin, g_tau_nl)` → differentiable LOO-CV Robinson R-learner + exact leverages $(h_e, h_m, h_\tau)$. |
| **5. Compact Residual Two-Way TabPFN & Posterior Head** | `_TabularTwoWayLayer`, `arm_cal`, `prop_cal`, `head` | $O(N_{\text{ctx}} N_q S)$ ($S=16$) | Calibrates ARD R-learner arm risks and epistemic SDs into a 129-bin sinh-warped logit posterior, plus a lightweight 2-layer two-way TabPFN residual (`attn_residual_scale`, init `0.1`) over only the top `attn_slots = 16` gated columns to capture pairwise feature interactions and non-quadratic nonlinearities. |

---

## 2. Design & Engineering Iteration History

### Iteration 1 — Initial Implementation & Permutation-Equivariance Bug Catch
- **Implemented:** Full `oci/tabular_pfn/` package (`prior.py`, `preprocessing.py`, `model.py`, `training.py`, `cli.py`), H200 launcher (`scripts/run_tabular_pfn_h200.sh`), and offline unit tests (`tests/test_tabular_pfn.py`).
- **Subtle Bug Caught by Invariance Tests:** Under simultaneous context row permutation + column permutation (`test_model_p2000_forward_backward_and_invariance`), predictions `(m, s, p)` matched within `5e-4`, **but `mem.role_logits` drifted by `0.118`** (`157/240` elements).
- **Root Cause:** Hard indicator thresholds in `_column_sufficient_stats` — `near_end` (`< 0.05 * span`) and `tail_frac` (`|X_s| > 2.0`) — flipped cleanly across dozens of rows of discrete/ordinal columns when floating-point reduction order changed `col_mean` by `~4.8e-7` (`stat[18]` jumped by `0.429`!).
- **Fix:** Replaced both hard indicators with smooth $C^\infty$ bounded equivalents (`near_end = exp(-(d_end / 0.05)^2).mean()`, `tail_frac = sigmoid(4 * (|X_s| - 2)).mean()`), eliminating threshold discontinuities and achieving `< 5e-4` exact simultaneous row- and column-permutation invariance/equivariance across predictions, `role_logits`, and screener logits.

### Iteration 2 — Dead-Gradient Screener Fix, Async Prefetch, & Out-of-Fold Stage-2 CLI (Commit `7d6b661`)
- **Subtle Bug Caught by Code Audit:** `nuis_screen_head(h_col)` and `mod_screen_head(h_col2)` fed only into non-differentiable `torch.topk(...).indices`, receiving **zero gradient** during training.
- **Fix:** Exposed `nuis_screen_logits` and `mod_screen_logits` on `TabularMemory`, increased learned screener blend weight (`1.5×`), and added class-balanced BCE supervision inside `loss_components()` (`nuis_screen_logits` vs. `roles >= 2` to filter noise AND pure instruments before top-$K_1$; `mod_screen_logits` vs. `roles in {4, 5}` to promote modifiers in Round 2). Verified via unit test that **100% of model parameters** receive finite non-zero gradients (`len(grads) == num_parameters`).
- **Throughput & Stage-2 Usability:**
  - Added 1-step asynchronous `ProcessPoolExecutor` prefetching in `training.py` so CPU high-$P$ episode synthesis overlaps 100% with GPU forward/backward.
  - Added `--cv-folds K` stratified out-of-fold zero-shot prediction + averaged `--role-report` + batched query scoring (`--query-batch-size 512`) in `cli.py` so a single Stage-2 cohort yields leak-free predictions for all $N$ patients.

### Iteration 3 — 500-Step Proof-of-Concept & `P = 50 / 500 / 2,000` Head-to-Head Benchmark (Commit `f0afb22`)
- **Setup:** Trained a 500-step proof-of-concept (`4,000` studies, log-uniform $N_{\text{ctx}} \in [80, 1{,}200]$, $P \in [20, 2{,}000]$, 1× RTX A6000, `grad_accum=8`, `num_workers=8`) on the v1 single-family sparse prior.
- **Runtime:** **367.2 s (`6.1 min`) total** (`~0.73 s/step`).
- **Mid-training trajectory (32 held-out studies, `P = 500`, `N_ctx = 800`):**
  - Step 250: `cate_rmse = 0.1177`, `within_r = +0.404`, `null_hte = 0.0813`, `role_auroc = 0.855`, `g_e(conf/inst) = 0.42 / 0.17`
  - Step 500: `cate_rmse = 0.1174`, `within_r = +0.410`, `null_hte = 0.0767`, `role_auroc = 0.856`, `g_e(conf/inst) = 0.50 / 0.21` (vs. Arm-Mean `0.1422`, Dense Ridge-R `0.1458`, Screened-32 Ridge-R `0.1651`).
- **Full `P = 50 / 500 / 2,000` Head-to-Head Stress Test (48 held-out studies per width, `N_ctx = 800`, `N_q = 128`, 25% null-HTE, 70% strong instruments):**

| Candidate Width `P` | **PFN `cate_rmse` ↓** | Arm-Mean ↓ | Dense Ridge-R ↓ | Screened-32 Ridge-R ↓ | Cross-Fit GBM-R ↓ | **PFN `within_r` ↑** (vs. Scr-32) | **PFN `null_rmse` ↓** | Signal / Modifier AUROC ↑ | `g_e` (Conf / Inst) | Median Infer / Peak VRAM |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`P = 50`** | **`0.1209`** | `0.1521` | `0.1639` | `0.1585` | `0.2325` | **`+0.473`** (`+0.391`) | `0.0724` | `0.838 / 0.806` | `0.59 / 0.43` | `20.5 ms` / `134 MB` |
| **`P = 500`** | **`0.1246`** | `0.1490` | `0.1480` | `0.1634` | `0.2392` | **`+0.440`** (`+0.324`) | `0.0765` | `0.872 / 0.878` | `0.53 / 0.33` | `22.0 ms` / `143 MB` |
| **`P = 2,000`** | **`0.1323`** | `0.1630` | `—` | `0.1467` | `0.2376` | **`+0.388`** (`+0.302`) | `0.0711` | **`0.863 / 0.866`** | **`0.42 / 0.29`** | **`27.5 ms` / `170 MB`** |

- **Key Findings:**
  - Scaling 40× from `P = 50` to `P = 2,000` (>99% noise) increased latency by only `+7 ms` (`20.5 → 27.5 ms`) and VRAM by `+36 MB` (`134 → 170 MB`).
  - Univariate top-32 screening (`Screened-32 Ridge-R`) degraded sharply when pure instruments dominated marginal $T$-correlation and pure crossover modifiers had weak marginal $Y$-correlation; `SparseTabularCausalPFN` beat every baseline across all widths while achieving `0.866` modifier AUROC at `P = 2,000`.
  - Cross-fitted `HistGradientBoosting` R-learner catastrophically overfit binary R-learner pseudo-outcomes (`0.21–0.22` null-HTE error vs. PFN's `0.071`).

### Iteration 4 — Upgrade to 3-Family OCI-Aligned Multi-DGP Mixture Prior (Commit `424bfb1`)
- **Motivation (Realism Gap Analysis vs. OCI's Existing Synthetic DGPs):** Auditing `synthetic_data/{dgp_terms.py,config.py,reequation.py,longitudinal.py}` and `sample_size_calculator/{population.py,engine.R}` revealed six clinical/statistical regimes missing from the v1 tabular prior:
  1. Multi-level mutually exclusive Dirichlet one-hot categorical blocks (where 1–2 levels drive treatment or effect modification).
  2. True **MNAR** missingness (`documentation_mechanism="mnar"`) plus explicit 3-state EHR missingness/unknown indicator columns.
  3. **Stage-1 LLM extraction noise** (asymmetric binary FN/FP + continuous measurement noise) and **proxy-only latent confounders** (teaching the model to pool across multiple noisy proxies).
  4. **Unmeasured confounding** (`unmeasured_confounders`, `unmeasured_strength`) — calibrating wider epistemic uncertainty and inducing true **Z-bias** whenever instruments enter `e(X)`.
  5. Harsher oncology prevalence/positivity (`treatment_logit_std` up to `2.8`, rare outcomes down to `8%`, 8% unconfounded RCTs, 50% `modifier_main_effects=False` pure crossover modifiers, and both `logOR` and bounded additive `RD` scales).
  6. Direct tabular episodes from OCI's bundled 5,000-design oncology bank (`synthetic_data/designs/nemotron_5000.json.gz`), with strict `split == "train"` (4,000 designs) vs. `split == "held_out"` (1,000 designs, 3 unseen cancers) separation.
- **Upgraded 3-Family Mixture in `oci/tabular_pfn/prior.py`:**
  - `sparse_oci` (50%) + `reequation_calc` (25%) + `nemotron_design` (25%), vectorized with `scipy.special.ndtri`/`ndtr`.
  - **Worst-case synthesis speed (`P = 2,000, N = 1,200`):** `sparse_oci = 114.6 ms/study`, `reequation_calc = 119.5 ms/study`, `nemotron_design = 102.3 ms/study` (~`7 ms/study` effective across 16 H200 CPU workers, overlapped with GPU).
- **Zero-Shot Out-of-Distribution Transfer Check (v1 500-Step PoC Evaluated on New 3-Family Held-Out Benchmark, `P = 500`, `N_ctx = 800`, 36 studies):**
  Even though the v1 PoC checkpoint had *never seen* `reequation_calc`, `nemotron_design`, unmeasured confounding, MNAR, one-hot categoricals, extraction noise, or `RD` scales during training, it transferred strongly zero-shot across every family:

| Held-Out Slice (`P = 500`, `N_ctx = 800`) | Studies | **v1 PoC PFN `cate_rmse` ↓** | Arm-Mean ↓ | Screened-32 Ridge-R ↓ | Dense Ridge-R ↓ | **PFN `within_r` ↑** |
|---|---:|---:|---:|---:|---:|---:|
| **All 3 OCI Families Combined** | 36 | **`0.1192`** | `0.1576` | `0.1502` | `0.1512` | **`+0.4315`** |
| `sparse_oci` (upgraded: MNAR, U-conf, noise, RD, cats) | 18 | **`0.1155`** | `0.1293` | `—` | `—` | **`+0.3555`** |
| `reequation_calc` (OCI `reequation`/`engine.R` DGP) | 8 | **`0.1196`** | `0.2026` | `—` | `—` | **`+0.4941`** |
| `nemotron_design` (unseen `held_out` oncology designs) | 10 | **`0.1254`** | `0.1627` | `—` | `—` | **`+0.5049`** |

*(Overall 36-study diagnostics: `null_hte_cate_rmse = 0.0690`, `role_signal_auroc = 0.8765`, `confounder/instrument` propensity gate `0.57 / 0.36`, `risk_rmse = 0.1648`, `propensity_rmse = 0.1448`, `ate_rmse = 0.0488`.)*

### Iteration 5 — Empirical-Sparsity Role Calibration (Normalized Saerens EM) & Clinical Role Summary CLI
- **Problem Identified:** Because `loss_components()` balances noise vs. non-noise columns 50/50 during training (so 99% noise doesn't collapse `role_logits` to all-noise), raw softmax probabilities over-called non-noise at large $P$ (`61.4` non-noise calls at `P = 2,000` with `13%` precision and `0.0982` Brier).
- **Calibration Ablation (measured on Run 1 step-500 checkpoint, 15 studies per $P$, `N_ctx = 800`, ~15 true active columns):**
  1. *Naive Saerens EM* (unnormalized `LR = exp(l_sig - l_0)`): Collapsed $\hat{\pi} \to 0$ at `P = 2,000` (`est_K = 1.1`, `top10 mean_p = 0.04` vs. `0.54` empirical accuracy) because the neural network's null $\mathbb{E}[\text{LR} \mid \text{noise}]$ drifted to `~0.50 < 1.0`.
  2. *Gaussian 2-component local-FDR mixture*: Over-called (`est_K = 221.8`, `prec = 0.06`) because the null `log_lr` tail is right-skewed (non-Gaussian).
  3. **Normalized Saerens EM** (`calibrate_role_probabilities` in `oci/tabular_pfn/model.py`): Enforces the exact distribution-free likelihood-ratio identity $\mathbb{E}[\text{LR} \mid \text{noise}] = 1$ inside EM (zero Gaussian assumption, zero checkpoint changes, preserves exact ranking):

| `P` | True Active `K` | Raw Calls / Prec / Brier ↓ | **Calibrated Calls (`p_sig ≥ 0.5`) / Prec / Brier ↓** | **Top-10 Empirical Hit vs. Mean Calibrated `p_sig`** |
|---:|---:|---|---|---|
| `50` | `15.9` | `11.3` / `0.70` / `0.1634` | `13.7` / **`0.74`** / **`0.1403`** | **`0.81` vs. `0.76`** |
| `500` | `15.3` | `28.4` / `0.31` / `0.1086` | `10.1` / **`0.69`** / **`0.0204`** (`5.3×` ↓) | **`0.67` vs. `0.70`** |
| `2,000` | `15.3` | `61.4` / `0.13` / `0.0982` | `4.7` / **`0.68`** / **`0.0060`** (`16.4×` ↓) | **`0.54` vs. `0.50`** |

- **Clinical Summary & Export (`oci/tabular_pfn/cli.py`, `format_role_summary`):** `predict` now supports `.csv` and `.json` for `--role-report` (with composite `p_signal`, `p_any_confounder`, `p_any_modifier`, and `raw_signal_score` columns) and automatically prints a ranked summary table of top suspected confounders, effect modifiers, and suppressed pure instruments (also available standalone via `--print-roles` / `--role-summary-top-k`).

### Iteration 6 — Step-4,500 Bottleneck Diagnosis & v2 Architecture (`scripts/run_tabular_pfn_v2_h200.sh`)
- **Live Diagnostic on Run 1 (`step 4,500` checkpoint, 48 held-out `P = 500, N_ctx = 800` 3-family studies):**
  - Full v1 PFN (`cate_rmse = 0.1163`, `within_r = +0.5001`) already outperformed an **Oracle-Columns Ridge R-learner** handed only the ~12 true causal features (`cate_rmse = 0.1994`, `within_r = +0.4954`).
  - Zeroing `attn_residual_scale` (which had grown `0.10 → 0.261`) dropped performance back to `cate_rmse = 0.1237, within_r = +0.4599`, proving that **almost all post-step-500 gain came from the residual Two-Way TabPFN layers** capturing nonlinearities and pairwise interactions.
  - ARD gates by true role showed 50–77× noise suppression (`noise: ge=0.0019, gm=0.0043, gtau=0.0016` vs. `confounder gm=0.233`, `modifier gtau=0.123`, `dual_role gm=0.300`), **except `instrument ge = 0.0876` tied `confounder ge = 0.0821`** because `prop_loss` against `true_propensity` (which includes instruments) fought `cate_loss` through the shared `g_e` gate.
  - Screener recall at `P = 500`: Top-64 confounder recall `0.828`, Top-48 modifier recall `0.736`.
- **v2 Architectural Upgrades (100% backward-compatible with v1 checkpoints):**
  1. **`--decouple-rlearner-propensity`:** Adds dedicated outcome-adaptive propensity gates `(g_er_lin, g_er_nl)` + `log_lam_er` used strictly inside Robinson `e_r_loo` and `(y0_raw, y1_raw)` reconstruction with **zero gradient from `prop_loss`** (verified by unit test), completely separating treatment-model propensity (`g_e`) from outcome-adaptive instrument-suppressing causal propensity (`g_er`).
  2. **Higher-capacity residual TabPFN & column attention (`7.38M` params):** `col_dim=192`, `col_attn_layers=3`, `study_width=192`, `study_layers=4`, `heads=6`, `attn_slots=24`, plus **bf16 FlashAttention-2 / SDPA autocast** strictly inside `study_layers` on CUDA (keeping Cholesky ARD solvers in exact fp32, cutting worst-case `N_ctx=3,000, P=2,000` VRAM to `~6–7 GiB`).
  3. **Higher Top-$K$ recall & smoother schedule:** `top_k_nuis=80`, `top_k_cate=64`, `lr=4e-4`, `warmup_steps=300`, `eval_studies=64`, default `STEPS=15000` (`artifacts/tabular_causal_pfn_v2_p2000`).

### Iteration 7 — v3 Flagship Architecture & 5-Family Real-World Pathology Prior (`scripts/run_tabular_pfn_v3_h200.sh`)
- **Motivation (Closing the Real Stage-2 & Non-Additive Generalization Gaps):** Real Stage-2 catalog evaluations (`S0/n1000/seed101` & `seed102`) exposed fragmented multi-column synonymous extractions (e.g., 5–8 partial-coverage/opposite-polarity columns per latent confounder like `sex` or `ECOG`), while v1/v2's 3-family prior remained largely additive-1D + 1 pairwise product and capped at `N_ctx = 3,000, P = 2,000`.
- **5-Family Prior Upgrades (`oci/tabular_pfn/prior.py`, `178–219 ms/study` at `N=1,800, P=2,000`):**
  1. **New Family 4 (`tree_subgroup`, 18%):** Multi-way depth 2–4 clinical decision-rule & biomarker subgroup HTE ensembles (crisp/soft conjunctions impossible for linear/quadratic ARD Ridge alone).
  2. **New Family 5 (`causal_scm`, 17%):** Layered nonlinear DAG + 2-layer MLP treatment/outcome/HTE surfaces with explicit **observed-parent ($Z \to C \leftarrow W$, 100% identifiable by column attention)** and **latent-parent ($U_T \to C \leftarrow U_Y$)** M-bias colliders, heavy tails, dense weak confounding, and near-positivity extremes.
  3. **Fragmented Stage-1 Extraction Synonym Clusters (`55%` across all 5 families):** Splits active causal variables into 2–6 synonymous opposite-polarity, coarsened, and partial-coverage (`NaN`-masked) clones inheriting the true causal role, forcing the model to pool across fragmented columns.
  4. **Expanded Clinical Distributions & Scale:** Heavy-tailed Student-$t$ lab outliers, zero-inflated utilization counts, non-monotonic clinical reference bands, optional dense weak confounding (`15%`), and H200 training envelope **`N_ctx = 80..8,000`, `P = 20..3,500`, `query_size = 128`**.
- **v3 Architecture (`18.88M` params, `--use-v3-solver-basis --decouple-rlearner-propensity`, 100% v1/v2 checkpoint compatible):**
  1. **4-Channel $C^\infty$ ARD Solver Basis:** Extends closed-form LOO-CV ARD nuisances & R-learner from `(X, X^2)` to `(X, X^2, tanh(1.5 X), M_c)` (adding outlier/threshold-robust saturation and smooth variance-gated missingness `M_c` directly inside orthogonalization, expanding column stats `24 → 32` and Robinson stats `6 → 12`).
  2. **6-d Pairwise-Complete Co-Observed Correlation Bias (`_pairwise_corr_features`):** Feeds both zero-imputed Pearson correlation AND pairwise-complete co-observed correlation + overlap fraction into `_CorrBiasedColumnLayer`, allowing column attention to distinguish redundant complete duplicates (`overlap ≈ 1`) from complementary fragmented extractions (`corr_coobs ≈ 0.9, overlap ≈ 0.25`) and observed-parent colliders.
  3. **Expanded H200 Capacity & Budget:** `col_dim=256`, `col_attn_layers=4`, `study_width=256`, `study_layers=6`, `heads=8`, `top_k_nuis=128`, `top_k_cate=96`, `attn_slots=32`, `ridge_grid_size=9` (`[-0.5, 11.5]`), `30,000` steps × `grad_accum=12` (**`360,000` unique studies**). Verified worst-case peak VRAM at `N_ctx=8,000, P=3,500`: **`38.21 GiB`** (`27%` of H200 141 GB; `9.76 GiB` at typical `N=1,800, P=1,000`).

### Iteration 8 — v4 Interaction Architecture & De-Aliased 6-Family Interacting + Embedding Prior (`scripts/run_tabular_pfn_v4_h200.sh`)
- **Motivation (Closing the Epistatic Interaction, Missingness-Shortcut, & Multimodal-Embedding Gaps):**
  1. **Clear scaling & non-saturation on v3:** Zero-shot CATE RMSE on all 16 Full-LLM realistic cohorts (`N = 800`, `Struct + TinyOncBERT SentKM-256`) jumped from `v2 0.1110 (+0.173 within_r)` to **`v3 0.1043 (+0.234 within_r, 0.0392 ATE RMSE)`** (crushing `Applied GBM-R 0.1127` and live end-to-end Run 5 `0.1217`), and v3's 5-family loss was still decreasing at step `26,000 / 30,000`.
  2. **Structural bottleneck in v3's closed-form ARD solver:** v3's ARD basis `(X, X^2, tanh(1.5 X), M_c)` remained strictly 1D additive — forcing the residual TabPFN head to absorb all confounder×confounder, confounder×modifier, and modifier×modifier interactions (`tree_subgroup` lagged at `within_r = +0.343` vs. `nemotron_design +0.639`), while Round 2's 1D Robinson scan was blind to pure-epistatic `C × M` crossover modifiers where marginal `E[tau | M] = 0`.
  3. **De-aliased EHR missingness & multimodal embedding geometry:** Incorporating the verified zero-shortcut high-missingness prior (`miss_rate` AUROC `~0.51–0.53` across all 6 families) from step 0 eliminates v3's `miss_rate > 0.55` Stage-2 failure mode, while adding bounded correlated cosine-MaxSim embedding blocks directly trains v4 on the geometry of `TinyOncBERT SentKM` multimodal representations.
- **6-Family Interacting Prior (`oci/tabular_pfn/prior.py`, `217–258 ms/study` at `N=1,800, P=2,000`):**
  1. Upgraded all five existing families (`sparse_oci` 20%, `reequation_calc` 12%, `nemotron_design` 18%, `tree_subgroup` 18%, `causal_scm` 16%) with explicit confounder×confounder propensity interactions, confounder×modifier & modifier×modifier HTE interactions (20% pure-epistatic in `sparse_oci`, cross-role confounder×modifier conjunctions in `tree_subgroup`, 45% joint confounder×modifier HTE MLPs in `causal_scm`), keeping ground-truth `dual_role` labels exact.
  2. **New Family 6 (`interacting_clinical`, 16%):** Coupled bilinear/threshold/categorical×continuous clinical interactions across treatment, outcome, and HTE (**30% mathematically orthogonalized pure-epistatic crossover HTE** with zero marginal modifier slope), plus **50% correlated cosine-MaxSim embedding blocks** (`K = 48..320` unit-sphere prototypes with uniform synonymous clusters mimicking `TinyOncBERT SentKM` coupled with structured covariates; verified `miss_rate` AUROC = `0.525`).
- **v4 Architecture (`34.60M` params, `--use-v3-solver-basis --use-v4-interactions --decouple-rlearner-propensity`, 100% v1/v2/v3 checkpoint compatible):**
  1. **Confounder-Modulated All-$P$ Robinson Scan (`_robinson_confounder_mod_stats`):** Exact centered $O(R \cdot N \cdot P)$ ($R \le 8$ top Round-1 confounders/prognostics) scan via 3 fast `(R, N) @ (N, P)` GEMMs with zero `(R, N, P)` memory allocation (`ROB_STAT_DIM: 12 → 16`), rescuing pure-epistatic `C × M` modifiers whose marginal 1D Robinson z-score is zero.
  2. **Symmetric $C^\infty$ Pairwise Interaction Basis inside Closed-Form ARD Solvers (`pair_top_k = 16` → `120` unordered pairs):** Smooth variance-weighted products `(p_var / (p_var + 2e-3))^0.5` (auto-zeroing mutually exclusive categorical dummy pairs) gated by strictly symmetric MLPs (`[h_a + h_b, sqrt((h_a - h_b)^2 + 1e-6), pair_stats]`) enter `A_e`, `A_er` (**verified zero `prop_loss` gradient into `g_per`**), `A_m`, and `A_tau` (mixing top modifiers, dual-role, confounders, and prognostics so `C × M`, `P × M`, and `M × M` interactions enter closed-form R-learner orthogonalization directly). Verified `< 5e-4` exact simultaneous row + column permutation invariance.
  3. **Scaled H200 Capacity, Auto Checkpointing, & Host-RAM Hygiene:** `col_dim=320`, `study_width=320`, `study_layers=8`, `col_attn_layers=4`, `heads=8`, `top_k_nuis=128`, `top_k_cate=96`, `attn_slots=40`, `pair_top_k=16`, `ridge_grid_size=9`, `40,000` steps × `grad_accum=12` (**`480,000` unique studies**). Automatic gradient checkpointing inside `study_layers` for `N_ctx > 4,000` keeps verified peak VRAM at **`18.68 GiB`** typical (`N=1,800, P=1,000`) and **`24.19 GiB`** worst-case (`N_ctx=8,000, P=3,500`), while periodic `gc.collect()` + `malloc_trim(0)` and bounded nested eval workers (`min(num_workers, 4)`) keep host RSS flat.

---

## 3. Training Runs Summary

| Run | Artifact Dir | Prior Version | Budget (`Steps` / `Studies`) | `N_ctx` / `P` Range | Status | Held-Out `P = 500` `cate_rmse` ↓ | Held-Out `P = 2,000` `cate_rmse` ↓ | `P = 2,000` `within_r` ↑ | `P = 2,000` Role / Mod AUROC ↑ | `g_e` (Conf / Inst) |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| **PoC v1 (A6000)** | *(scratch)* | v1 single-family sparse | `500` / `4,000` (`6.1 min`) | `80..1,200` / `20..2,000` | Complete | `0.1246` (v1) / `0.1192` (3-fam OOD) | `0.1323` | `+0.388` | `0.863 / 0.866` | `0.42 / 0.29` |
| **Run 1 (v1 2.11M, A6000)** | `artifacts/tabular_causal_pfn_p2000` | v2 3-family OCI mixture (`50/25/25`) | `12,000` / `96,000` | `80..3,000` / `20..2,000` | **Complete** | `0.1186` (`within_r = +0.491`) | `0.1155` | `+0.492` | `0.894` (sig) | `0.075 / 0.066` (`1.1×`) |
| **Run 2 (v2 7.38M, Decoupled)** | `artifacts/tabular_causal_pfn_v2_p2000` | v2 3-family OCI mixture (`50/25/25`) | `15,000` / `120,000` | `80..3,000` / `20..2,000` | **Complete** (`best = s13,000`) | `0.1187` (`within_r = +0.502`) | **`0.1150`** | **`+0.503`** | **`0.895`** (sig) | **`0.032 / 0.015` (`2.1×`)** |
| **Run 3 (v3 18.88M, H200 Flagship)** | `artifacts/tabular_causal_pfn_v3_h200` | v3 5-family + fragmented + colliders (`30/15/20/18/17`) | `30,000` / `360,000` (`16.79h`) | `80..8,000` / `20..3,500` | **Complete** (`best = s25,000`) | `0.1189` (5-fam) / **`0.0979`** (nemotron) | **`0.1149`** (`0.0941` @ `N=4k`) | `+0.437` (**`+0.703`** @ `N=4k`) | **`0.905`** (**`0.959`** @ `N=4k`) | **`0.014 / 0.009` (`1.6×`)** |
| **Run 4 (v4 34.60M, H200 Pairwise-ARD)** | `artifacts/tabular_causal_pfn_v4_h200` | v4 de-aliased 6-family interacting + cosine-MaxSim (`20/12/18/18/16/16`) | `40,000` / `480,000` | `80..8,000` / `20..3,500` | **Ready to launch** (`scripts/run_tabular_pfn_v4_h200.sh`) | `—` | `—` | `—` | `—` | `—` |

### Final Held-Out Stress Tests: v3 (`18.88M`, `30,000` steps / `360k` studies, `best = s25,000`) on 5-Family Suite

Evaluated across `P = 50..3,500` (`N_ctx = 800`, 128 studies), large-cohort `N_ctx = 4,000, P = 2,000` (64 studies), and all 5 individual families (`P = 500` and `P = 2,000`, 96 studies each):

| Benchmark | `cate_rmse` ↓ | Arm-Mean ↓ | Scr-32 Ridge-R ↓ | `within_r` ↑ | `pool_r` ↑ | `risk_rmse` ↓ | `prop_rmse` ↓ | `ate_rmse` ↓ | `null_hte` ↓ | `role_auroc` ↑ | `g_e` (Conf / Inst) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **5-Family Mix `P = 50` (`N=800`)** | **`0.1099`** | `0.1556` | `0.1823` | **`+0.548`** | `+0.765` | `0.1396` | `0.1334` | `0.0454` | `0.0600` | `0.881` | `0.020 / 0.013` |
| **5-Family Mix `P = 200` (`N=800`)** | **`0.1118`** | `0.1474` | `0.2223` | **`+0.542`** | `+0.772` | `0.1367` | `0.1373` | `0.0376` | `0.0656` | `0.900` | `0.017 / 0.011` |
| **5-Family Mix `P = 500` (`N=800`)** | **`0.1189`** | `0.1461` | `0.1636` | `+0.456` | `+0.723` | `0.1492` | `0.1380` | `0.0446` | `0.0705` | `0.913` | `0.015 / 0.009` |
| **5-Family Mix `P = 2,000` (`N=800`)** | **`0.1149`** | `0.1496` | `0.1588` | `+0.437` | `+0.767` | `0.1464` | `0.1466` | `0.0389` | `0.0694` | `0.905` | `0.014 / 0.009` |
| **5-Family Mix `P = 3,500` (`N=800`)** | **`0.1143`** | `0.1439` | `0.1758` | `+0.465` | `+0.720` | `0.1453` | `0.1441` | `0.0406` | `0.0709` | **`0.913`** | `0.014 / 0.008` |
| **5-Family Mix `P = 2,000` (`N=4,000`)** | **`0.0941`** | `0.1563` | `0.1708` | **`+0.703`** | **`+0.858`** | **`0.1215`** | **`0.1226`** | **`0.0273`** | **`0.0435`** | **`0.959`** | `0.020 / 0.011` |
| `nemotron_design P = 500` (`N=800`) | **`0.0979`** | `0.1427` | `0.3286` | **`+0.639`** | **`+0.815`** | **`0.1252`** | `0.1319` | **`0.0361`** | `0.0642` | `0.938` | `0.017 / 0.010` |
| `nemotron_design P = 2,000` (`N=800`) | **`0.1008`** | `0.1398` | `0.2325` | **`+0.590`** | `+0.793` | **`0.1280`** | `0.1315` | `0.0490` | `0.0596` | **`0.948`** | `0.019 / 0.010` |
| `reequation_calc P = 500` (`N=800`) | **`0.1173`** | `0.1619` | `0.1865` | **`+0.503`** | `+0.756` | **`0.1386`** | `0.1330` | `0.0462` | `0.0695` | `0.947` | `0.021 / 0.018` |
| `reequation_calc P = 2,000` (`N=800`) | **`0.1260`** | `0.1719` | `0.1847` | `+0.518` | `+0.735` | **`0.1413`** | `0.1384` | `0.0450` | `0.0594` | `0.944` | `0.017 / 0.014` |
| `sparse_oci P = 500` / `P = 2,000` | `0.1137` / `0.1194` | `0.1537` / `0.1475` | `0.1616` / `0.1490` | `+0.503` / `+0.441` | `+0.762` / `+0.721` | `0.1369` / `0.1540` | `0.1383` / `0.1509` | `0.0368` / `0.0394` | `0.0617` / `0.0661` | `0.867` / `0.870` | `0.014/0.005` / `0.011/0.005` |
| `causal_scm P = 500` / `P = 2,000` | `0.1095` / `0.1114` | `0.1399` / `0.1472` | `0.1515` / `0.1595` | **`+0.510` / `+0.512`** | `+0.755` / `+0.781` | `0.1466` / `0.1452` | `0.1577` / `0.1604` | `0.0399` / `0.0447` | `0.0579` / `0.0650` | **`0.956` / `0.955`** | `0.015/0.012` |
| `tree_subgroup P = 500` / `P = 2,000` | `0.1270` / `0.1266` | `0.1482` / `0.1437` | `0.1567` / `0.1695` | `+0.343` / `+0.304` | `+0.695` / `+0.717` | `0.1530` / `0.1638` | `0.1376` / `0.1605` | `0.0408` / `0.0399` | `0.0532` / `0.0639` | `0.852` / `0.872` | `0.012/0.006` / `0.011/0.005` |

### Final Held-Out Stress Tests: v1 (`2.11M`, 12k) vs. v2 (`7.38M`, 15k) at `N_ctx = 800`, `N_q = 128`

Mixture benchmarks (`128` studies each, `seed = 700001`) + family-specific held-out benchmarks (`96` studies each, `seed = 800001`; `nemotron_design` strictly from the 1,000 `split == "held_out"` designs):

| Benchmark | Model | `cate_rmse` ↓ | Arm-Mean ↓ | Scr-32 Ridge-R ↓ | Dense Ridge-R ↓ | `within_r` ↑ | `pool_r` ↑ | `risk_rmse` ↓ | `prop_rmse` ↓ | `ate_rmse` ↓ | `null_hte` ↓ | `role_auroc` ↑ | `g_e` (Conf / Inst) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Mixture `P = 50`** | v1 | **`0.1126`** | `0.1486` | `0.1953` | `0.1888` | `+0.534` | `+0.758` | `0.1482` | `0.1282` | `0.0442` | `0.0678` | `0.858` | `0.107 / 0.113` |
| | **v2** | `0.1127` | `0.1486` | `0.1953` | `0.1888` | **`+0.546`** | `+0.757` | **`0.1477`** | **`0.1265`** | `0.0451` | **`0.0647`** | **`0.861`** | **`0.042 / 0.023`** |
| **Mixture `P = 200`** | v1 | `0.1114` | `0.1470` | `0.1708` | `0.1646` | `+0.492` | `+0.752` | `0.1465` | `0.1313` | `0.0443` | `0.0625` | `0.884` | `0.086 / 0.084` |
| | **v2** | **`0.1112`** | `0.1470` | `0.1708` | `0.1646` | **`+0.501`** | **`+0.754`** | **`0.1454`** | **`0.1307`** | `0.0459` | `0.0644` | **`0.886`** | **`0.038 / 0.018`** |
| **Mixture `P = 500`** | v1 | **`0.1186`** | `0.1602` | `0.1742` | `0.1638` | `+0.491` | `+0.735` | `0.1509` | `0.1341` | `0.0516` | `0.0712` | `0.901` | `0.082 / 0.078` |
| | **v2** | `0.1187` | `0.1602` | `0.1742` | `0.1638` | **`+0.502`** | `+0.734` | **`0.1496`** | **`0.1333`** | `0.0532` | `0.0724` | `0.901` | **`0.038 / 0.017`** |
| **Mixture `P = 2,000`** | v1 | `0.1155` | `0.1512` | `0.1714` | `—` | `+0.492` | `+0.744` | `0.1504` | `0.1332` | `0.0406` | `0.0625` | `0.894` | `0.075 / 0.066` |
| | **v2** | **`0.1150`** | `0.1512` | `0.1714` | `—` | **`+0.503`** | **`+0.747`** | **`0.1497`** | **`0.1327`** | `0.0412` | `0.0629` | **`0.895`** | **`0.032 / 0.015`** |
| **`nemotron_design P = 500`** | v1 | `0.1094` | `0.1484` | `0.2785` | `0.2572` | `+0.590` | `+0.761` | `0.1390` | `0.1244` | `0.0424` | `0.0542` | `0.927` | `0.103 / 0.097` |
| | **v2** | **`0.1077`** | `0.1484` | `0.2785` | `0.2572` | **`+0.602`** | **`+0.769`** | **`0.1373`** | **`0.1238`** | `0.0426` | `0.0562` | **`0.928`** | **`0.043 / 0.017`** |
| **`nemotron_design P = 2,000`** | v1 | `0.1041` | `0.1426` | `0.2699` | `—` | `+0.533` | `+0.787` | `0.1449` | `0.1172` | `0.0413` | `0.0652` | `0.930` | `0.096 / 0.093` |
| | **v2** | **`0.1021`** | `0.1426` | `0.2699` | `—` | **`+0.541`** | **`+0.796`** | **`0.1435`** | **`0.1162`** | **`0.0401`** | **`0.0647`** | **`0.931`** | **`0.036 / 0.016`** |
| **`reequation_calc P = 500`** | v1 | `0.1193` | `0.1601` | `0.1494` | `0.1555` | `+0.472` | `+0.741` | `0.1497` | `0.1344` | `0.0407` | `0.0659` | `0.934` | `0.094 / 0.095` |
| | **v2** | **`0.1181`** | `0.1601` | `0.1494` | `0.1555` | **`+0.488`** | **`+0.747`** | **`0.1486`** | **`0.1327`** | **`0.0396`** | `0.0669` | **`0.936`** | **`0.044 / 0.030`** |
| **`reequation_calc P = 2,000`** | v1 | `0.1342` | `0.1762` | `0.1785` | `—` | `+0.529` | `+0.701` | `0.1573` | `0.1394` | `0.0528` | `0.0679` | `0.928` | `0.085 / 0.082` |
| | **v2** | **`0.1317`** | `0.1762` | `0.1785` | `—` | **`+0.544`** | **`+0.714`** | **`0.1553`** | **`0.1382`** | **`0.0515`** | **`0.0651`** | **`0.930`** | **`0.039 / 0.027`** |

### Diagnostic: Why `cate_rmse ≈ 0.115` at `N_ctx = 800` Even With Strong `within_r` (`+0.50..+0.60`)

Exact MSE decomposition on v2 (`best_checkpoint.pt`, `step 13,000`) over the 128 held-out `P = 500, N_ctx = 800` mixture studies:
1. **Amplitude calibration is already near-optimal (`slope = 1.077`, vs. `1.000` ideal):**
   - `Total MSE = 0.01408` (`RMSE = 0.1187`) splits cleanly into `ATE MSE = 0.00282` (`20.1%`, `ATE RMSE = 0.0532`) + `Within-study MSE = 0.01126` (`79.9%`, `Within RMSE = 0.1061`).
   - Because true within-study CATE heterogeneity in this prior is large (`true_SD = 0.1340` overall, `0.1436` on non-null studies), the theoretical minimum MSE at correlation $r \approx 0.55$ is $\sigma_{\text{true}}\sqrt{1 - r^2} \approx 0.112$ with optimal shrunk prediction SD $r \cdot \sigma_{\text{true}} \approx 0.074$ (v2 predicted SD: **`0.0761`**, within-study RMSE: **`0.1061`**).
2. **Error by data-pathology regime (`P = 500, N_ctx = 800`):**
   - Clean complete observational (`no U, no extraction noise, no missingness`, `n=12`): **`0.1011`**
   - Clean observed (`no U, no extraction noise`, with MAR/MNAR missingness, `n=61`): **`0.1109`**
   - Stage-1 extraction noise (no U, `n=36`): **`0.1269`** (irreducible Bayes attenuation because `true_ite` is defined on noiseless latents)
   - Unmeasured confounding (`n=31`): **`0.1234`** (irreducible confounding bias)
   - Strictly null-HTE (`n=36`): **`0.0724`** | Non-null with extraction noise or U (`n=45`): **`0.1417`**
3. **Head-to-head vs. Oracle-True-Columns baselines (first 48 `P = 500, N_ctx = 800` studies — baselines handed ONLY the ~12 true causal columns, zero noise/instruments):**
   - **v2 PFN (blind across all 500 cols):** **`cate_rmse = 0.1162`**, **`within_r = +0.502`**
   - Oracle-Cols HistGBM T-Learner: `cate_rmse = 0.2212`, `within_r = +0.350`
   - Oracle-Cols Logistic T-Learner: `cate_rmse = 0.2764`, `within_r = +0.423`
4. **Sample-size scaling on the same 32 `P = 500` held-out studies (`N_ctx = 800 → 2,500`):**
   - `N_ctx =   800`: `cate_rmse = 0.1038`, `ate_rmse = 0.0332`, `within_r = +0.469`
   - `N_ctx = 2,500`: **`cate_rmse = 0.0845`**, **`ate_rmse = 0.0269`**, **`within_r = +0.627`**

### Run 1 Mid-Training Trajectory (`48` Held-Out 3-Family Studies, `P = 500`, `N_ctx = 800`)
Reference baselines on this fixed 48-study held-out set: `arm_mean = 0.1541`, `screened_ridge_r = 0.1519`, `dense_ridge_r = 0.1651`.

| Step | Elapsed | `cate_rmse` ↓ | `within_r` ↑ | `pool_r` ↑ | `spearman` ↑ | `risk_rmse` ↓ | `prop_rmse` ↓ | `ate_rmse` ↓ | `null_hte` ↓ | `role_auroc` ↑ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | `7.1m` | `0.1234` | `+0.4359` | `+0.7200` | `+0.7376` | `0.1558` | `0.1391` | `0.0476` | `0.0729` | `0.8490` |
| 1000 | `14.1m` | `0.1216` | `+0.4562` | `+0.7300` | `+0.7354` | `0.1526` | `0.1397` | `0.0478` | `0.0810` | `0.8530` |
| 1500 | `21.1m` | `0.1217` | `+0.4822` | `+0.7350` | `+0.7358` | `0.1515` | `0.1386` | `0.0520` | `0.0782` | `0.8517` |
| 2000 | `28.1m` | `0.1199` | `+0.4593` | `+0.7390` | `+0.7446` | `0.1507` | `0.1386` | `0.0481` | `0.0790` | `0.8539` |
| 2500 | `36.8m` | `0.1217` | `+0.4703` | `+0.7394` | `+0.7323` | `0.1508` | `0.1376` | `0.0539` | `0.0774` | `0.8570` |
| 3000 | `44.3m` | `0.1186` | `+0.4765` | `+0.7455` | `+0.7474` | `0.1500` | `0.1390` | `0.0478` | **`0.0713`** | `0.8551` |
| 3500 | `51.3m` | `0.1187` | `+0.4774` | `+0.7450` | `+0.7443` | `0.1487` | **`0.1365`** | `0.0484` | `0.0751` | `0.8540` |
| 4000 | `58.3m` | `0.1201` | `+0.4763` | `+0.7433` | `+0.7562` | `0.1497` | `0.1373` | `0.0510` | `0.0738` | `0.8552` |
| **4500** | **`65.3m`** | **`0.1163`** | **`+0.5001`** | **`+0.7568`** | **`+0.7650`** | **`0.1475`** | `0.1367` | **`0.0459`** | `0.0715` | `0.8537` |

---

## 4. Real OCI Stage-2 Extracted All-Evidence Candidate Catalog Benchmark (`S0/n1000`)

Evaluated **v1 (`step 12,000`)** and **v2 (`best_checkpoint.pt`, `step 13,000`)** strictly **zero-shot** on the exact 5 outer cross-validation folds (`800` fit / `200` held-out per fold, zero leakage) of real LLM-extracted Stage-2 feature catalogs (`outer_001..005/extraction/extracted_features.csv`) from two independent `S0/n1000` cohorts under `/data1/ken/pcori_dev/oci_bench/` (`seed101` and `seed102`). Each fold's raw LLM-extracted schema (~600–780 columns) was preprocessed inside the fold via `TabularPreprocessor` into **1,310–1,744** 3-state EHR features (`fit` on outer-train only) and scored directly against each cohort's OCI Stage-2 baseline predictions (`multi_model` ensemble + constituent heads) and oracle `true_cate` / `true_treatment_prob`.

### Cohort 1: `S0/n1000/seed101`
- **Oracle DGP roles:** True confounders = `age`, `sex`, `baseline_nlr`; True modifiers = `creatinine_clearance`, `prior_platinum_therapy`, `brain_metastases_status`.
- **Extracted catalog width across 5 folds:** raw `[703, 708, 690, 779, 678]` → preprocessed 3-state **`[1,590, 1,520, 1,612, 1,744, 1,354]`**.

| Method | `CATE RMSE` ↓ | `MAE` ↓ | `ATE Bias` | `Pearson r` ↑ | `Spearman ρ` ↑ | `Pred SD` | `Prop RMSE` ↓ | 5-Fold Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OCI Stage-2 Ensemble (`multi_model`) | `0.1344` | `0.1081` | `+0.0637` | `+0.349` | `+0.361` | `0.0673` | `0.2028` | `—` |
| ↳ Stage-2 Causal Forest | `0.1313` | `0.1092` | `+0.0574` | `+0.320` | `+0.343` | `0.0250` | `—` | `—` |
| ↳ Stage-2 Penalized Interactions | `0.1362` | `0.1083` | `+0.0676` | `+0.300` | `+0.314` | `0.0441` | `—` | `—` |
| ↳ Stage-2 DINA | `0.1969` | `0.1516` | `+0.0662` | `+0.278` | `+0.272` | `0.1768` | `—` | `—` |
| Arm-Mean (outer-fold ATE only) | `0.2319` | `0.2058` | `+0.1961` | `+0.047` | `+0.040` | `0.0001` | `—` | `—` |
| v1 `SparseTabularCausalPFN` (zero-shot) | `0.1220` | `0.0957` | `+0.0455` | `+0.406` | `+0.420` | `0.0560` | `—` | `10.25s` |
| **v2 `SparseTabularCausalPFN` (zero-shot)** | `0.1175` | `0.0911` | **`+0.0318`** | `+0.417` | `+0.444` | `0.0637` | `0.1904` | `9.97s` |
| v3 `SparseTabularCausalPFN` (raw, *hit `miss>0.55` shortcut*) | `0.2036` | `0.1752` | `+0.1540` | `+0.217` | `+0.277` | `0.0832` | `0.3214` | `10.55s` |
| **v3 `SparseTabularCausalPFN` (`max_miss <= 0.55`, diagnostic)** | **`0.1150`** | **`0.0864`** | `+0.0414` | **`+0.568`** | **`+0.577`** | `—` | **`0.1843`** | `~10s` |

- **Blind confounder discovery (zero knowledge of column names):** **11 of the top 11** ranked confounders across ~1,600 columns were direct LLM extractions of all 3 true confounders (`neutrophil_to_lymphocyte_ratio (5/5) p_any_c=0.880`, `age (5/5) 0.395`, `biological_sex==female/male`, `age_group==adult/elderly`, `gender==Female/Male`, `sex==female/male`, `age_at_diagnosis`).
- **Blind modifier discovery:** Weaker at `N_fit = 800` (`p_any_m ≤ 0.095`), though `brain_metastases_progression_status`, `brain_lesion_characteristics`, `chronic_kidney_disease_status`, and `pemetrexed_cycle_status` surfaced in the top 12.

### Cohort 2: `S0/n1000/seed102`
- **Oracle DGP roles:** True confounders = `sex`, `ecog_performance_status`, `baseline_nlr`; True modifiers = `creatinine_clearance`, `prior_platinum_therapy`, `brain_metastases_status`.
- **Extracted catalog width across 5 folds:** raw `[655, 627, 601, 718, 680]` → preprocessed 3-state **`[1,443, 1,431, 1,310, 1,583, 1,586]`**.

| Method | `CATE RMSE` ↓ | `MAE` ↓ | `ATE Bias` | `Pearson r` ↑ | `Spearman ρ` ↑ | `Pred SD` | `Prop RMSE` ↓ | 5-Fold Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OCI Stage-2 Ensemble (`multi_model`) | `0.2292` | `0.1817` | `+0.1238` | `+0.436` | `+0.424` | `0.1018` | **`0.1911`** | `—` |
| ↳ Stage-2 Causal Forest | `0.2438` | `0.1887` | `+0.1272` | `+0.268` | `+0.255` | `0.0318` | `—` | `—` |
| ↳ Stage-2 Penalized Interactions | `0.2149` | `0.1697` | `+0.1257` | `+0.620` | `+0.608` | `0.0866` | `—` | `—` |
| ↳ Stage-2 DINA | `0.2964` | `0.2358` | `+0.1186` | `+0.294` | `+0.298` | `0.2415` | `—` | `—` |
| Arm-Mean (outer-fold ATE only) | `0.2613` | `0.2046` | `+0.1498` | `+0.031` | `+0.056` | `0.0010` | `—` | `—` |
| v1 `SparseTabularCausalPFN` (zero-shot) | `0.2015` | `0.1632` | `+0.1215` | **`+0.666`** | **`+0.653`** | `0.1249` | `—` | `9.15s` |
| **v2 `SparseTabularCausalPFN` (zero-shot)** | **`0.1963`** | **`0.1580`** | `+0.1091` | `+0.648` | `+0.640` | `0.1353` | `0.2240` | **`9.65s`** |
| v3 `SparseTabularCausalPFN` (raw, *hit `miss>0.55` shortcut*) | `0.2225` | `0.1821` | `+0.1420` | `+0.604` | `+0.621` | `0.1441` | `0.3252` | `11.56s` |
| **v3 `SparseTabularCausalPFN` (`max_miss <= 0.55`, diagnostic)** | `0.1966` | `0.1593` | **`+0.1046`** | `+0.641` | `+0.623` | `—` | `0.2118` | `~10s` |

- **Blind confounder discovery:** Top 12 ranked confounders were again dominated by all 3 true confounders (`neutrophil_to_lymphocyte_ratio (5/5) p_any_c=0.788`, `gender==male/female`, `ecog_components==ECOG 2/1`, `biological_sex`, `activity_tolerance==ECOG 1`, `performance_status==1`, `functional_capacity==impaired`).
- **Blind modifier discovery:** Top 5 ranked modifiers surfaced `comprehensive_metabolic_panel_status`, `renal_profile_stability`, and **`creatinine_clearance (5/5)`** — which received the **highest `gate_tau = 0.143`** across all ~1,500 columns.

### Takeaways & v3 Missingness-Shortcut Diagnosis (Fixed for v3.1)
1. **Zero-shot `SparseTabularCausalPFN` v2 outperformed OCI's entire Stage-2 ensemble (`multi_model`) AND every single Stage-2 constituent estimator on both real LLM-extracted cohorts** (`0.1175` vs. `0.1344` on `seed101`; `0.1963` vs. `0.2292` on `seed102`) in `< 10 seconds` on 1× A6000.
2. **Real-World Shortcut Caught on v3 (`miss_rate > 0.55`):** In v3's initial prior, noise columns were capped at `max_missing_rate = 0.45` while fragmented active columns received extra partial-coverage `NaN`s (`0.15..0.68`, stacking to `0.50..0.85`). `nuis_screen_head` exploited that artifact (`scr = +15.1` for any column with `miss_rate > 0.55`), which on real Stage-2 catalogs (where ~35% of noise columns are 65–97% `not_documented`) evicted `age`, `NLR`, and `sex` out of `nuis_idx` (`gate_e = gate_m = 0.000` despite `role_logits` nailing `age=0.900`, `NLR=0.975`, and `creatinine_clearance=0.925`!).
3. **Diagnostic Proof & v3.1 Fix:** Simply masking `miss_rate > 0.55` immediately snapped v3 to **`RMSE = 0.1150, MAE = 0.0864, Pearson = +0.568 (+0.151 over v2!), Spearman = +0.577, Prop RMSE = 0.1843`** on `seed101`, and **`RMSE = 0.1966, ATEbias = +0.1046, Prop RMSE = 0.2118`** on `seed102`. Fixed permanently in `oci/tabular_pfn/prior.py` by giving noise columns heavy-tailed EHR missingness up to `95%` AND matching noise fragmented synonym clusters (verified `miss_rate` AUROC for active vs. noise = **`0.532`**), plus `--init-from` warm-start fine-tuning (`scripts/run_tabular_pfn_v3_1_finetune_h200.sh`, 4k steps / ~2h H200).

### Next External Benchmarks
- Evaluate v2 (`artifacts/tabular_causal_pfn_v2_p2000`) on additional OCI Stage-2 scenarios (e.g., `S1/S2/S3`, larger `N`, unmeasured confounding) and `sample_size_calculator` reference configurations, and assess integrating `SparseTabularCausalPFN` as an optional Stage-2 estimator / candidate screener.
