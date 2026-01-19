# Causal DragonNet Text (CDT)

A framework for causal inference from clinical text using DragonNet models, with traditional propensity score matching analysis for validation.

## Key Features

- **DragonNet/UpliftNet for ITE Estimation**: Estimates individual treatment effects (ITE) from clinical text, enabling heterogeneous treatment effect analysis
- **Traditional PSM Validation**: Automatically runs propensity score matching analysis using DragonNet's propensity scores, providing ATT/ATE estimates with confidence intervals
- **Balance Diagnostics**: Standardized mean differences before/after matching
- **Sensitivity Analysis**: Rosenbaum bounds for unmeasured confounding
- **Plasmode Simulation**: Validate methods with known ground truth

## How It Works

CDT uses a two-stage approach:

### 1. DragonNet for Treatment Effect Estimation

DragonNet jointly learns:
- **Propensity scores**: P(T=1|X) - probability of treatment given text features
- **Potential outcomes**: E[Y|T=0,X] and E[Y|T=1,X]
- **Individual treatment effects**: ITE = E[Y|T=1,X] - E[Y|T=0,X]

This enables analysis of treatment effect heterogeneity - understanding which patients benefit most.

### 2. PSM Analysis for Validation

Using DragonNet's propensity scores, traditional PSM analysis provides:
- **Matching**: Nearest neighbor, optimal, or caliper matching
- **ATT/ATE estimation**: Average treatment effects with bootstrap confidence intervals
- **Statistical tests**: McNemar's test (binary outcomes), paired t-test (continuous)
- **Sensitivity analysis**: How robust are results to unmeasured confounding?

This dual approach gives you:
- ITE estimates for personalized medicine (from DragonNet)
- Validated average effects with proper statistical inference (from PSM)

## Installation

```bash
git clone https://github.com/kenlkehl/causal-dragonnet-text.git
cd causal-dragonnet-text
pip install -e .
```

## Quick Start

```bash
# Create default config
cdt init --output config.json

# Edit config.json to set your dataset path

# Run experiment (DragonNet + PSM analysis)
cdt run --config config.json

# Skip PSM analysis if you only want DragonNet
cdt run --config config.json --skip-psm
```

## Configuration

```json
{
  "output_dir": "./results",
  "device": "cuda:0",

  "applied_inference": {
    "dataset_path": "./data.parquet",
    "cv_folds": 5,

    "architecture": {
      "model_type": "dragonnet",
      "num_latent_confounders": 20,
      "features_per_confounder": 4
    },

    "training": {
      "epochs": 50,
      "batch_size": 8,
      "learning_rate": 0.0001
    },

    "matching_analysis": {
      "enabled": true,
      "method": "nearest",
      "caliper": 0.2,
      "caliper_scale": "std"
    }
  }
}
```

## Output

```
results/
├── applied_inference/
│   ├── predictions.parquet    # DragonNet predictions (ITE, propensity, Y0, Y1)
│   ├── training_log.csv       # Training metrics
│   └── psm_analysis/          # PSM analysis results
│       ├── psm_summary.json   # ATT, ATE estimates with CIs
│       ├── matched_pairs.csv  # Matched treatment-control pairs
│       ├── balance_statistics.csv
│       └── sensitivity_analysis.csv
└── summary.json
```

## Key Results

The output includes:

**From DragonNet:**
- `ite_pred`: Individual treatment effect for each patient
- `y0_pred`, `y1_pred`: Predicted outcomes under control/treatment
- `propensity_pred`: Propensity score

**From PSM Analysis:**
- `att_matched`: Average treatment effect on treated (from matched pairs)
- `ate_ipw`: Average treatment effect (inverse probability weighting)
- `ate_stratified`: Average treatment effect (stratification)
- Confidence intervals and p-values for each
- Sensitivity analysis (Rosenbaum bounds)

## CLI Options

```bash
cdt run --config config.json \
    --device cuda:0 \           # GPU device
    --workers 4 \               # Parallel CV folds
    --skip-psm \                # Skip PSM analysis
    --matching-method optimal \ # Override matching method
    --caliper 0.1 \             # Override caliper
    --verbose                   # Debug logging
```

## Python API

```python
from cdt import (
    CausalDragonnetText,
    PropensityMatcher,
    estimate_att_matched,
    run_psm_analysis
)

# Train DragonNet model
model = CausalDragonnetText(...)
# ... training ...

# Get predictions
predictions = model.predict(embeddings)
# predictions contains: y0_pred, y1_pred, ite_pred, propensity

# Run PSM analysis on DragonNet's propensity scores
from cdt.config import MatchingAnalysisConfig
psm_config = MatchingAnalysisConfig(method="nearest", caliper=0.2)
psm_results = run_psm_analysis(predictions_df, psm_config)

print(f"DragonNet ATE: {predictions['ite_pred'].mean():.4f}")
print(f"PSM ATT: {psm_results['att_matched']}")
```

## Why Both DragonNet and PSM?

| Aspect | DragonNet | PSM |
|--------|-----------|-----|
| **Estimand** | ITE (individual effects) | ATT/ATE (average effects) |
| **Heterogeneity** | Yes - per-patient estimates | No - single average |
| **Inference** | Point estimates | CIs, p-values, tests |
| **Sensitivity** | - | Rosenbaum bounds |
| **Validation** | - | Balance diagnostics |

Using both gives you:
1. Rich heterogeneity analysis from DragonNet
2. Rigorous statistical validation from PSM
3. Confidence that average effects agree between methods

## Citation

```bibtex
@software{cdt2024,
  author = {Kehl, Ken},
  title = {Causal DragonNet Text},
  year = {2024},
  url = {https://github.com/kenlkehl/causal-dragonnet-text}
}
```

## License

MIT License
