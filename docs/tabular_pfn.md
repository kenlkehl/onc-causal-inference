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

## Synthetic All-Evidence Prior (`oci/tabular_pfn/prior.py`)

Training draws log-uniform $N_{\text{ctx}} \in [80, 3{,}000]$ and $P \in [20, 2{,}000]$ with:
- 2–18 sparse active causal features (confounders, prognostics, modifiers, dual-role) embedded inside 50–99.5% noise,
- Strong pure instruments ($T$-only drivers, zero direct effect on $Y$),
- Collinear noisy proxy duplicates ($\rho \in [0.55, 0.94]$) plus block-correlated factor noise,
- Mixed rare/common binary, ordinal, Gaussian, and heavy-tailed log-normal columns with MCAR/MAR missingness,
- Linear, threshold, quadratic, hinge, saturating, and pairwise-interaction mechanisms,
- 25% strictly null-HTE studies ($\tau(x) \equiv \text{ATE}$) to calibrate shrinkage against false heterogeneity discovery.

## Usage

### Train on 1× H200 (~1–1.5 hours for 6,000 steps / 48,000 studies up to $P = 2{,}000$)
```bash
./scripts/run_tabular_pfn_h200.sh
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
