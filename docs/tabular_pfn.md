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

## Synthetic OCI Multi-DGP Prior (`oci/tabular_pfn/prior.py`)

Training draws log-uniform $N_{\text{ctx}} \in [80, 3{,}000]$ and $P \in [20, 2{,}000]$ from a 3-family OCI-aligned mixture prior:

1. **`sparse_oci` (50%):** High-dimensional sparse all-evidence tables with 2–18 active features inside 50–99.5% noise, multi-level mutually exclusive Dirichlet one-hot categorical blocks, strong pure instruments, collinear noisy proxies ($\rho \in [0.55, 0.94]$) plus proxy-only latent confounders, block-correlated factor noise, linear/threshold/quadratic/hinge/saturating + pairwise interactions, optional unmeasured confounding (creating real Z-bias on instruments), Stage-1 LLM extraction noise, MCAR/MAR/**MNAR** + 3-state EHR documentation, occasional RCTs (8%), 25% strictly null-HTE studies, and both `logOR` and bounded additive `RD` scales.
2. **`reequation_calc` (25%):** Vectorized implementation of OCI's `synthetic_data/reequation.py` and `sample_size_calculator/engine.R` tabular DGP family (exchangeable Gaussian copula, uniform-CDF & positive log-normal continuous, Beta-prevalence binary, Dirichlet categorical contrasts, `none`/`threshold`/`quadratic` nonlinearities, unmeasured confounders, `logOR`/`RD` scales) padded to $P = 20\text{–}2{,}000$.
3. **`nemotron_design` (25%):** Direct tabular studies sampled strictly from the 4,000 `split == "train"` oncology designs in `synthetic_data/designs/nemotron_5000.json.gz` (real lab/vital/PRO moments, cancer-specific note-finding prevalences and causal role menus, pairwise interactions, extraction noise, MNAR/MAR missingness) padded to $P = 20\text{–}2{,}000$. The 1,000 `split == "held_out"` designs (including 3 unseen cancers) are strictly excluded from training and used automatically inside `evaluate`.

## Usage

### Train on 1× H200 (~2–2.5 hours for 12,000 steps / 96,000 studies up to $P = 2{,}000$)
```bash
./scripts/run_tabular_pfn_h200.sh
# Override step budget as desired, e.g. STEPS=20000 ./scripts/run_tabular_pfn_h200.sh
```

### Zero-Shot Prediction + Interpretable Feature Role Report on Any Tabular Cohort
```bash
# Out-of-fold zero-shot predictions for every patient in a Stage-2 cohort (5-fold cross-fitted)
python -m oci.tabular_pfn.cli predict \
  --run-dir artifacts/tabular_causal_pfn_p2000 \
  --context path/to/stage2_catalog.parquet \
  --cv-folds 5 \
  --output predictions.parquet \
  --role-report feature_roles.json
```
(You can also pass explicit `--context` + `--queries` tables, or `--context-size K` for a single holdout split.)

`TabularPreprocessor` automatically handles mixed numeric, binary, categorical, and 3-state EHR missingness (`not_documented` vs. `documented_unknown`) strictly fitted on context rows with zero query or oracle leakage.

## Proof-of-Concept Stress-Test Benchmark (`P = 50, 500, 2,000`)

Measured after a 6.1-minute, 500-step proof-of-concept training run (2.11M parameters, log-uniform `N_ctx = 80..1,200`, `P = 20..2,000`, 1× RTX A6000), evaluated across 48 held-out synthetic all-evidence studies per width (`N_ctx = 800`, `N_q = 128`, 25% strictly null-HTE, 70% strong instruments, collinear proxies, mixed types, MCAR/MAR missingness):

| Candidate Width `P` | **PFN `cate_rmse` ↓** | Arm-Mean ↓ | Dense Ridge-R ↓ | Screened-32 Ridge-R ↓ | Cross-Fit GBM-R ↓ | **PFN `within_r` ↑** (vs. Scr-32) | **PFN `null_rmse` ↓** | Signal / Modifier AUROC ↑ | `g_e` (Conf / Inst) | Median Infer / Peak VRAM |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`P = 50`** | **`0.1209`** | `0.1521` | `0.1639` | `0.1585` | `0.2325` | **`+0.473`** (`+0.391`) | `0.0724` | `0.838 / 0.806` | `0.59 / 0.43` | `20.5 ms` / `134 MB` |
| **`P = 500`** | **`0.1246`** | `0.1490` | `0.1480` | `0.1634` | `0.2392` | **`+0.440`** (`+0.324`) | `0.0765` | `0.872 / 0.878` | `0.53 / 0.33` | `22.0 ms` / `143 MB` |
| **`P = 2,000`** | **`0.1323`** | `0.1630` | `—` | `0.1467` | `0.2376` | **`+0.388`** (`+0.302`) | `0.0711` | **`0.863 / 0.866`** | **`0.42 / 0.29`** | **`27.5 ms` / `170 MB`** |

For the full architectural iteration history, bug-fix notes, 3-family zero-shot out-of-distribution transfer results, and ongoing H200 training run tracker, see [`docs/tabular_pfn_experiments.md`](tabular_pfn_experiments.md).
