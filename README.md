# Propensity Score Matching for Clinical Text (PSM-CT)

A framework for causal inference from clinical text using propensity score matching with deep learning text encoders. PSM-CT estimates treatment effects from unstructured clinical narratives by learning propensity scores from text and performing traditional statistical analysis.

## Clinical Research Objective

Observational studies using EHR data are essential for comparative effectiveness research when randomized trials are infeasible or unethical. However, standard approaches rely on structured covariates (diagnoses, labs, demographics) which may fail to capture critical confounders documented only in clinical notes—such as functional status, symptom severity, patient preferences, or nuanced disease characteristics.

PSM-CT addresses this gap by:
- **Learning propensity scores from clinical text** using deep learning encoders (CNN, Transformer, or GRU with attention)
- **Performing traditional propensity score matching** with well-established statistical methods
- **Providing comprehensive statistical analysis** including ATE/ATT estimation, balance diagnostics, and sensitivity analysis

This enables researchers to leverage the rich information in clinical narratives for causal inference while maintaining the interpretability and statistical rigor of traditional propensity score methods.

## How It Works

### Architecture Overview

PSM-CT processes clinical text through two main stages:

1. **Propensity Score Estimation**: Train a deep learning model to predict treatment assignment
   - Text is chunked and embedded using a sentence transformer (default: `all-MiniLM-L6-v2`)
   - Optional confounder feature extraction learns interpretable confounding patterns
   - Encoder architecture options: CNN, Transformer (BERT-style), or GRU with attention
   - Optional joint outcome prediction encourages learning of true confounders

2. **Traditional Propensity Score Analysis**:
   - Propensity score matching (nearest neighbor, optimal, caliper-based)
   - Balance diagnostics (standardized mean differences)
   - Treatment effect estimation (ATT from matched pairs, ATE via IPW or stratification)
   - Sensitivity analysis (Rosenbaum bounds)

### Model Architecture Options

| Encoder | Description | Best For |
|---------|-------------|----------|
| `gru` | Bidirectional GRU with attention | Variable-length texts, default choice |
| `transformer` | BERT-style transformer encoder | Long documents, complex patterns |
| `cnn` | Multi-kernel 1D CNN | Fast training, shorter texts |

### Joint Outcome Prediction

Optionally, the propensity model can jointly predict outcomes during training. This encourages the model to learn features that are true confounders (predictive of both treatment and outcome) rather than just treatment predictors. This is controlled by the `joint_outcome_prediction` and `outcome_weight` parameters.

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for practical use)

### Clone the Repository

```bash
git clone https://github.com/kenlkehl/causal-dragonnet-text.git
cd causal-dragonnet-text
```

### Install with uv (Recommended)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e .
```

### Alternative: Standard pip Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Dataset Requirements

PSM-CT expects datasets in Parquet or CSV format with these columns:

| Column | Description | Type |
|--------|-------------|------|
| `clinical_text` | The clinical narrative text | string |
| `treatment_indicator` | Binary treatment assignment (0 or 1) | int/float |
| `outcome_indicator` | Binary outcome (0 or 1) | int/float |
| `split` (optional) | Data split: "train", "val", "test" | string |

The `split` column is only needed for fixed-split mode. For cross-validation, all data is used and split automatically.

## Quick Start

### 1. Create Configuration

```bash
psm init --output config.json
```

### 2. Edit Configuration

```json
{
  "output_dir": "./results",
  "dataset_path": "./my_data.parquet",
  "cv_folds": 5,

  "model": {
    "encoder_type": "gru",
    "hidden_dim": 256,
    "joint_outcome_prediction": true,
    "outcome_weight": 0.3
  },

  "training": {
    "epochs": 50,
    "batch_size": 8,
    "learning_rate": 0.0001
  },

  "matching": {
    "method": "nearest",
    "caliper": 0.2,
    "caliper_scale": "std"
  }
}
```

### 3. Run Analysis

```bash
psm run --config config.json
```

## Configuration Reference

### Model Configuration (`model`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `encoder_type` | `"gru"` | Encoder architecture: `"cnn"`, `"transformer"`, `"gru"` |
| `hidden_dim` | `256` | Hidden dimension for encoder and heads |
| `dropout` | `0.1` | Dropout rate |
| `num_latent_confounders` | `20` | Number of learnable confounder patterns |
| `joint_outcome_prediction` | `false` | Whether to jointly predict outcomes |
| `outcome_weight` | `0.5` | Weight for outcome loss (0-1) |
| `use_confounder_features` | `true` | Use confounder feature extraction |

### Training Configuration (`training`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | `1e-4` | Learning rate |
| `epochs` | `50` | Maximum training epochs |
| `batch_size` | `8` | Batch size |
| `early_stopping_patience` | `10` | Epochs without improvement before stopping |

### Matching Configuration (`matching`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `method` | `"nearest"` | Matching method: `"nearest"`, `"optimal"`, `"caliper"` |
| `caliper` | `0.2` | Maximum allowed distance for a match |
| `caliper_scale` | `"std"` | Scale for caliper: `"propensity"`, `"logit"`, `"std"` |
| `ratio` | `1` | Matching ratio (1:k matching) |
| `replacement` | `false` | Whether to match with replacement |

## CLI Options

```bash
psm run --config config.json \
    --device cuda:0 \           # Override GPU device
    --cv-folds 10 \             # Override CV folds
    --encoder transformer \     # Override encoder type
    --joint-outcome \           # Enable joint outcome prediction
    --matching-method optimal \ # Override matching method
    --caliper 0.1 \             # Override caliper value
    --verbose                   # Enable debug logging
```

## Output Files

Results are saved to the output directory:

```
output_dir/
├── summary.json              # High-level results summary
├── predictions.parquet       # Per-sample propensity scores
├── matched_pairs.csv         # Matched treatment-control pairs
├── balance_statistics.csv    # Covariate balance before/after matching
├── sensitivity_analysis.csv  # Rosenbaum bounds
└── training_history.csv      # Training metrics per epoch
```

### Key Results

The `summary.json` file contains:
- **Crude difference**: Unadjusted difference in outcome rates
- **IPW ATE**: Inverse probability weighted average treatment effect
- **Stratified ATE**: Propensity score stratification estimate
- **Matched ATT**: Average treatment effect on the treated from matched pairs
- **Overlap coefficient**: Measure of propensity score overlap between groups

## Python API

```python
import pandas as pd
from cdt import (
    PropensityModel,
    PropensityMatcher,
    estimate_att_matched,
    estimate_ate_ipw,
    summarize_analysis
)

# Load data
data = pd.read_parquet("clinical_data.parquet")

# Train propensity model
model = PropensityModel(
    encoder_type="gru",
    joint_outcome_prediction=True
)
# ... training code ...

# Get propensity scores
propensity_scores = model.predict(embeddings)['propensity']

# Perform matching
matcher = PropensityMatcher(method="nearest", caliper=0.2)
match_result = matcher.match(propensity_scores, treatment)

# Estimate treatment effects
att = estimate_att_matched(outcomes, treatment, match_result)
ate = estimate_ate_ipw(outcomes, treatment, propensity_scores)

print(f"ATT: {att.estimate:.3f} [{att.ci_lower:.3f}, {att.ci_upper:.3f}]")
print(f"ATE: {ate.estimate:.3f} [{ate.ci_lower:.3f}, {ate.ci_upper:.3f}]")
```

## Statistical Methods

### Treatment Effect Estimators

1. **ATT from Matched Pairs**: Difference in means between treated and matched controls
2. **IPW (Inverse Probability Weighting)**: Horvitz-Thompson-style estimator
3. **Stratification**: Weighted average of within-stratum effects

### Matching Methods

1. **Nearest Neighbor**: Greedy matching to closest control
2. **Optimal**: Hungarian algorithm minimizing total distance
3. **Caliper**: Nearest neighbor with maximum distance constraint

### Inference

- Bootstrap confidence intervals (default: 1000 iterations)
- Paired t-test for continuous outcomes
- McNemar's test for binary outcomes
- Rosenbaum sensitivity analysis for unmeasured confounding

## Legacy DragonNet Mode

The original DragonNet/UpliftNet approach is still available but deprecated:

```bash
# Create legacy config
psm init --legacy --output legacy_config.json

# Run legacy mode
psm legacy-run --config legacy_config.json
```

## Citation

If you use PSM-CT in your research, please cite:

```bibtex
@software{psmct2024,
  author = {Kehl, Ken},
  title = {Propensity Score Matching for Clinical Text},
  year = {2024},
  url = {https://github.com/kenlkehl/causal-dragonnet-text}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contact

Ken Kehl - kenneth_kehl@dfci.harvard.edu
