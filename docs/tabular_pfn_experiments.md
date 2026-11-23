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

---

## 3. Training Runs Summary

| Run | Artifact Dir | Prior Version | Budget (`Steps` / `Studies`) | `N_ctx` / `P` Range | Status | Held-Out `P = 500` `cate_rmse` ↓ | Held-Out `P = 2,000` `cate_rmse` ↓ | `P = 2,000` `within_r` ↑ | `P = 2,000` Role / Mod AUROC ↑ | `g_e` (Conf / Inst) |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| **PoC v1 (A6000)** | *(scratch)* | v1 single-family sparse | `500` / `4,000` (`6.1 min`) | `80..1,200` / `20..2,000` | Complete | `0.1246` (v1) / `0.1192` (3-fam OOD) | `0.1323` | `+0.388` | `0.863 / 0.866` | `0.42 / 0.29` |
| **Run 1 (H200 Full)** | `artifacts/tabular_causal_pfn_p2000` | v2 3-family OCI mixture (`50/25/25`) | `12,000` / `96,000` (~2–2.5h) | `80..3,000` / `20..2,000` | Ready (`./scripts/run_tabular_pfn_h200.sh`) | `—` | `—` | `—` | `—` | `—` |

### Planned Post-Training Evaluations for Run 1
1. Held-out 3-family mixture stress tests (`128` studies each, `N_ctx = 800`) at `P = 50, 200, 500, 2,000`.
2. Dedicated held-out `--family nemotron_design` (1,000 unseen oncology designs) and `--family reequation_calc` evaluations at `P = 500` and `P = 2,000`.
3. External zero-shot benchmark on real OCI Stage-2 extracted all-evidence candidate catalogs and `sample_size_calculator` reference configurations.
