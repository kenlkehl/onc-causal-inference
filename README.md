# Causal DragonNet Text (CDT)

CDT estimates treatment effects from clinical text by combining neural network feature extraction with causal inference methods. It extracts confounders from unstructured EHR narratives to estimate individual treatment effects (ITE) and average treatment effects (ATE) for comparative effectiveness research.

## Installation & Quickstart

### 1. Install uv (if not already installed)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

### 2. Clone and install

```bash
git clone https://github.com/kenlkehl/causal-dragonnet-text.git
cd causal-dragonnet-text
uv venv --python 3.12
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e .
```

### 3. Run an experiment

```bash
python oracle_experiment_scripts/run_causal_forest_experiment.py \
    --dataset example_synthetic_data_one_confounder/dataset_with_extraction.parquet \
    --output-dir ../quickstart_results \
    --device cuda:0 \
    --epochs 20 \
    --n-folds 3
```

For CPU-only machines, use `--device cpu`.  On Apple Silicon Macs, you can also use `--device mps`.

Results include:
- `metrics_summary.csv` - ITE correlation, ATE bias, and CI coverage metrics
- `*/predictions.parquet` - Per-sample treatment effect predictions with confidence intervals

## Recommended Approach: Causal Forest + GRU-Pool

The best-performing configuration uses a two-stage approach combining neural feature extraction with econml's CausalForestDML:

```
Stage 1: Representation Learning (Neural Network)
├── GRU-Pool Extractor: Chunk BiGRU + transformer + gated attention
├── Propensity Head: P(T=1|X) → BCE loss
├── Outcome Head: E[Y|X] → BCE loss
└── [Optional] Effect Head: τ(X) → R-loss (when use_rlearner_representation=True)

Stage 2: Effect Estimation (Causal Forest)
├── Extract learned representations from Stage 1
├── Fit CausalForestDML on extracted features
└── Estimate τ(X) = E[Y(1)-Y(0)|X] with confidence intervals
```

**Optional: R-Learner Representation Training** - Enable `use_rlearner_representation: true` in the causal forest config to add an R-learner loss during Stage 1, encouraging the neural network to learn representations that capture treatment effect heterogeneity. For even stronger separation, enable `rlearner_dual_extractors: true` to use separate feature extractors for nuisance functions vs treatment effect.

**Key advantages:**
- **Doubly-robust estimation**: Robust to misspecification of either propensity or outcome model
- **Honest trees**: Unbiased effect estimates via sample splitting within trees
- **Confidence intervals**: Built-in uncertainty quantification for treatment effects
- **No gradient competition**: Representation learning completes before effect estimation
- **Long document support**: GRU-Pool handles documents of any length via chunking

See `example_configs/causal_forest_config.json` for a complete configuration.

## Architecture

### Feature Extractors

| Type | Description | Long docs | fit_tokenizer |
|------|-------------|-----------|---------------|
| `gru_pool` | Chunk BiGRU + transformer + gated attention pooling | Yes | Required |
| `gru_transformer_mil` | Chunk BiGRU + transformer + gated MIL with K confounders | Yes | Required |
| `gated_mil_hierarchical` | Gated MIL + K confounders + task-specific weighting | Yes | No |
| `hierarchical_transformer` | Chunk BERT + transformer pooling | Yes | No |
| `bert_cross_chunk` | Chunk BERT + token-level cross-chunk attention + gated pooling | Yes | No |
| `llm` | Decoder-only LLM (Qwen3) with random init, last token embedding | Yes (32K) | No |
| `confounder` | Perceiver-style sparse cross-attention | Yes | GRU mode only |
| `bert` | HuggingFace transformer [CLS] | No (512 tokens) | No |
| `gru` | BiGRU + attention | Yes | Required |
| `cnn` | 1D CNN with optional semantic filter init | No (truncates) | Required |

### Causal Heads

| Type | Description | Key output |
|------|-------------|------------|
| `causal_forest` | Two-stage: neural features + CausalForestDML; optional R-learner representation and dual extractor mode | τ with confidence intervals |
| `rlearner` | Direct τ(X) optimization, detached nuisance functions; optional dual extractor mode | τ directly predicts ITE |
| `dragonnet` | Propensity + Y0/Y1 potential outcomes | ITE = σ(y1) - σ(y0) |
| `uplift` | Base outcome + treatment effect parametrization | ITE from effect head |
| `traditional_logreg` | Logistic regression with treatment as feature | ITE = σ(y\|T=1) - σ(y\|T=0) |

**R-Learner Dual Extractor Mode**: For R-learner, you can optionally enable `rlearner_dual_extractors: true` to use separate feature extractors for nuisance functions (propensity, outcome) and treatment effect (τ). This prevents gradient interference between confounder learning and effect modifier learning:

```json
{
  "architecture": {
    "model_type": "rlearner",
    "feature_extractor_type": "gru_pool",
    "rlearner_dual_extractors": true
  }
}
```

**Causal Forest Dual Extractor Mode**: The two-stage causal forest approach also supports dual extractors when R-learner representation training is enabled. In dual mode, Stage 2 uses features from the effect extractor (optimized for τ via R-loss) rather than the nuisance extractor:

```json
{
  "architecture": {
    "model_type": "causal_forest",
    "feature_extractor_type": "gru_pool",
    "causal_forest": {
      "n_estimators": 200,
      "use_rlearner_representation": true,
      "rlearner_dual_extractors": true
    }
  }
}
```

**Note**: Dual extractor mode approximately doubles feature extraction memory/compute.

## Dataset Requirements

CDT expects datasets in Parquet or CSV format:

| Column | Description | Type |
|--------|-------------|------|
| `clinical_text` | Clinical narrative text | string |
| `treatment_indicator` | Binary treatment (0/1) | int |
| `outcome_indicator` | Binary outcome (0/1) | int |
| `split` | "train"/"val"/"test" (optional for CV) | string |

## Running Experiments

### CLI Usage

```bash
# Generate a default configuration
cdt init --output my_config.json

# Run experiment
cdt run --config my_config.json --device cuda:0 --workers 4
```

### Configuration

Example causal forest configuration:

```json
{
  "output_dir": "./cdt_results",
  "seed": 42,
  "device": "cuda:0",

  "applied_inference": {
    "dataset_path": "./data/clinical_notes.parquet",
    "cv_folds": 5,

    "architecture": {
      "model_type": "causal_forest",
      "feature_extractor_type": "gru_pool",

      "gru_pool_embedding_dim": 128,
      "gru_pool_gru_hidden_dim": 128,
      "gru_pool_transformer_layers": 2,
      "gru_pool_chunk_size": 128,
      "gru_pool_projection_dim": 128,

      "causal_forest": {
        "n_estimators": 200,
        "min_samples_leaf": 10,
        "honest": true,
        "inference": true,
        "use_rlearner_representation": false,
        "rlearner_dual_extractors": false
      }
    },

    "training": {
      "epochs": 30,
      "batch_size": 8,
      "learning_rate": 1e-4
    }
  }
}
```

See `example_configs/` for configurations for each extractor and causal head type.

## Feature Extractor Details

### GRU-Pool (Recommended for long documents)

Combines BiGRU chunk encoding with transformer cross-chunk context and gated attention pooling. Learns from scratch via the causal objective.

```
Long Clinical Text → Token-based Chunking → BiGRU per Chunk
    → Transformer Cross-Chunk Context → Gated Attention Pooling
    → Single Document Vector → Causal Head
```

Key parameters: `gru_pool_chunk_size`, `gru_pool_transformer_layers`, `gru_pool_gated_attention_dim`

### GRU-Transformer-MIL

Similar to GRU-Pool but uses K confounder queries with task-specific weighting (propensity, tau, outcome can weight confounders differently).

Key parameters: `gru_mil_num_confounders`, `gru_mil_chunk_size`

### Gated MIL Hierarchical

Uses pretrained BERT for chunk encoding with gated MIL attention and K confounder queries.

Key parameters: `gated_mil_num_confounders`, `gated_mil_sentence_model`

### Hierarchical Transformer

Simple hierarchical encoding: chunk BERT + transformer layers + learnable [POOL] token aggregation.

Key parameters: `hier_transformer_num_layers`, `hier_transformer_chunk_size`

### CNN with Semantic Filters

1D CNN with optional semantic filter initialization from clinical concepts. Filters can be initialized from explicit phrases or learned via k-means clustering.

Key parameters: `cnn_kernel_sizes`, `cnn_explicit_filter_concepts`, `cnn_num_latent_filters`

### LLM (Decoder-Only with Random Init)

Uses a decoder-only LLM architecture (e.g., Qwen3-0.6B) initialized with **random weights** and trained entirely from scratch via the supervised causal objective. Extracts features using the last token embedding (GPT-style).

```
Clinical Text → Pretrained BBPE Tokenizer (left-padded)
    → Randomly-initialized Decoder-only LLM
    → Last Token Hidden State → Projection MLP
    → Document Vector → Causal Head
```

Key parameters: `llm_model_name`, `llm_max_length`, `llm_projection_dim`, `llm_gradient_checkpointing`

**Memory considerations:**
| Context Length | Recommended Batch Size | Notes |
|----------------|------------------------|-------|
| 32K | 1-2 | Requires gradient checkpointing |
| 8K | 4-8 | Good balance for most use cases |
| 2K | 16-32 | Fast iteration |

See `example_configs/llm_config.json` for a complete configuration.

### Numeric Feature Extraction

All extractors support an optional parallel numeric feature channel that provides magnitude-aware
encoding of clinical numbers (lab values, vitals, scores, doses, ages). Numbers are detected via
regex, binned on a log scale, and classified by context keywords. This helps models understand that
5 < 10 < 100 and that "creatinine 1.8" carries different meaning than "ECOG 2".

Enable in config:
```json
{
  "architecture": {
    "numeric_features_enabled": true,
    "numeric_embedding_dim": 32,
    "numeric_magnitude_bins": 8,
    "numeric_type_categories": 10
  }
}
```

Disabled by default (`numeric_features_enabled: false`) — no behavior change unless enabled.

### Intra-Batch Contrastive Learning

Optional supervised contrastive loss (SupCon) within similarity clusters improves confounder
detection by encouraging the model to discriminate treatment/outcome status among similar patients.

**How it works**: Features are clustered via K-means (detached, no gradient), then SupCon loss
is computed within each cluster using treatment × outcome as 4-class labels. A projection head
keeps the contrastive space separate from causal head features.

**Why cluster-then-contrast?** Global SupCon would collapse all treated patients' representations
together, destroying heterogeneity. Intra-cluster contrast targets subtle confounders among
clinically similar patients.

Enable in config:
```json
{
  "architecture": {
    "contrastive_enabled": true,
    "contrastive_num_clusters": 4,
    "contrastive_temperature": 0.1,
    "contrastive_label_mode": "joint"
  },
  "training": {
    "contrastive_weight": 0.1
  }
}
```

Key parameters: `contrastive_num_clusters`, `contrastive_temperature`, `contrastive_label_mode` ("treatment", "outcome", or "joint"), `contrastive_weight`

Works with all causal heads and extractors, including dual extractor modes (targets nuisance extractor). Gracefully degrades to zero loss for small batches or uniform-label clusters.

## Workflow Modes

### Applied Inference

Estimates treatment effects on real clinical data using K-fold CV or fixed splits.

### Plasmode Simulation

Generates synthetic outcomes with known treatment effects for method validation:

```json
{
  "plasmode_experiments": {
    "enabled": true,
    "plasmode_scenarios": [{
      "generation_mode": "phi_linear",
      "target_ate_prob": 0.10
    }]
  }
}
```

### Propensity Score Matching Analysis

Post-hoc PSM analysis using learned propensity scores:
- ATT from matched pairs
- ATE via IPW or stratification
- Balance diagnostics and Rosenbaum bounds

## Output Files

```
output_dir/
├── config.json
├── applied_inference/
│   ├── predictions.parquet        # Treatment effect estimates
│   ├── training_log.csv
│   └── psm_analysis/              # If enabled
└── plasmode_experiments/          # If enabled
```

The `predictions.parquet` contains:
- `pred_y0_prob`, `pred_y1_prob`: Predicted potential outcomes
- `pred_ite_prob`: Individual treatment effect (y1 - y0)
- `pred_propensity_prob`: Treatment propensity
- `pred_ite_lower`, `pred_ite_upper`: 95% CI (causal forest only)

## Dependencies

**Core**: torch, transformers, pandas, numpy, scikit-learn, econml

**Optional**: sentence-transformers (confounder extractor), entmax (sparse attention)

## Citation

```bibtex
@software{cdt2024,
  author = {Kehl, Ken},
  title = {Causal DragonNet Text: Clinical Causal Inference from EHR Text},
  year = {2024},
  url = {https://github.com/kenlkehl/causal-dragonnet-text}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

