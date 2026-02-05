# CLAUDE.md - CDT (Causal DragonNet Text)

## Overview

CDT estimates treatment effects from clinical text by combining text feature extraction with DragonNet causal inference heads. It extracts confounders from unstructured EHR narratives to estimate individual (ITE) and average (ATE) treatment effects.

## Repository Structure

```
cdt/
├── cli.py                 # CLI: `cdt init`, `cdt run`
├── config.py              # Dataclass configs
├── data/dataset.py        # ClinicalTextDataset
├── experiments/runner.py  # Orchestrates inference & plasmode
├── extraction/
│   ├── explicit_confounders.py   # LLM-based confounder extraction via vLLM
│   └── cache.py                  # Extraction result caching
├── inference/
│   ├── applied.py         # Applied inference (CV or fixed split)
│   └── applied_forest.py  # Causal forest inference pipeline
├── models/
│   ├── causal_text.py     # Main model (extractor + causal head)
│   ├── causal_text_forest.py  # Two-stage neural + causal forest model
│   ├── causal_forest_head.py  # CausalForestDML wrapper
│   ├── cnn_extractor.py, bert_extractor.py, gru_extractor.py
│   ├── llm_extractor.py                  # Decoder-only LLM with random init
│   ├── numeric_features.py               # Numeric value featurization (magnitude + type)
│   ├── explicit_confounder_featurizer.py # MLP featurization of extracted confounders
│   ├── chunking.py                       # Token-based text chunking utilities
│   ├── confounder_extractor.py           # Perceiver-style sparse attention
│   ├── hierarchical_transformer_extractor.py
│   ├── gated_mil_hierarchical_extractor.py
│   ├── gru_transformer_mil_extractor.py
│   ├── gru_pool_extractor.py
│   ├── dragonnet.py, uplift.py, rlearner.py, traditional_logreg.py  # Causal heads
│   └── sparse_attention.py               # entmax, top-k attention
├── training/plasmode.py   # Plasmode simulation
├── matching/              # PropensityMatcher, balance utilities
└── analysis/              # ATT/ATE estimation, PSM analysis

example_configs/           # Config files for each extractor type
synthetic_data/            # LLM-based synthetic data generation
```

## Architecture

### Feature Extractors

| Type | Description | Long docs | fit_tokenizer |
|------|-------------|-----------|---------------|
| `cnn` | 1D CNN, semantic filter init, fastest | No (truncates) | Required |
| `bert` | HuggingFace transformer [CLS] | No (512 tokens) | No |
| `gru` | BiGRU + attention, O(N) | Yes | Required |
| `confounder` | Perceiver-style sparse cross-attention, K latent confounders | Yes | GRU mode only |
| `hierarchical_transformer` | Chunk BERT + transformer pooling | Yes | No |
| `gated_mil_hierarchical` | Gated MIL + K confounders + task-specific weighting | Yes | No |
| `gru_transformer_mil` | Chunk BiGRU + transformer + gated MIL with K confounders | Yes | Required |
| `gru_pool` | Chunk BiGRU + transformer + gated attention pooling (single vector) | Yes | Required |
| `llm` | Decoder-only LLM (Qwen3) with last token embedding, random init | Yes (32K) | No |

**Note**: Hierarchical extractors use overlapping token-based chunking (`chunk_size`, `chunk_overlap`) instead of sentence splitting for more consistent context windows.

### Causal Heads

| Type | Description | Key output |
|------|-------------|------------|
| `dragonnet` | Propensity + Y0/Y1 potential outcomes | ITE = σ(y1) - σ(y0) |
| `uplift` | Base outcome + treatment effect parametrization | ITE from effect head |
| `rlearner` | Direct τ(X) optimization, detached nuisance functions | τ directly predicts ITE |
| `traditional_logreg` | Traditional logistic regression with treatment as feature | ITE = σ(y\|T=1) - σ(y\|T=0) |
| `causal_forest` | Two-stage: neural features + econml CausalForestDML | τ with confidence intervals |

**R-Learner advantage**: Nuisance functions (e, m) are detached in R-loss, providing stronger gradient signal for treatment effect modifiers.

**R-Learner Dual Extractor Mode**: When `rlearner_dual_extractors=True`, the R-Learner uses two independent feature extractors:

| Component | Purpose | Training Signal |
|-----------|---------|-----------------|
| Nuisance Extractor | e(X), m(X) | Propensity BCE + Outcome BCE |
| Effect Extractor | τ(X) | R-learner loss only |

This separation prevents gradient interference between confounder learning (nuisance) and effect modifier learning (τ). The effect extractor learns representations optimized specifically for treatment effect heterogeneity.

**Memory Note**: Dual mode approximately doubles feature extraction memory/compute.

**Config:**
```json
{
  "architecture": {
    "model_type": "rlearner",
    "feature_extractor_type": "gru_pool",
    "rlearner_dual_extractors": true
  }
}
```

**Traditional LogReg approach**: Models P(Y|X, T) directly with treatment concatenated as a feature input to the outcome head. At inference, computes counterfactuals by running the outcome head twice with T=0 and T=1. Simpler loss function (outcome + propensity, no targeted regularization needed). Supports `stop_grad_propensity` but off by default.

**Causal Forest approach**: Two-stage method combining neural network feature extraction with econml's CausalForestDML:
1. **Stage 1**: Train neural feature extractor with propensity + outcome BCE losses to learn confounder representations
2. **Stage 2**: Train CausalForestDML on extracted features to estimate τ(X) directly

Advantages:
- Doubly-robust estimation (robust to misspecification of either nuisance model)
- Honest trees for unbiased effect estimates
- Built-in confidence intervals for treatment effects
- No gradient competition between representation learning and effect estimation
- Theoretical guarantees from the causal forest literature

## CLI

```bash
cdt init --output config.json
cdt run --config config.json --device cuda:0 --workers 4 [--skip-plasmode] [--verbose]

# Apple Silicon (MPS)
cdt run --config config.json --device mps --workers 1

# CPU fallback
cdt run --config config.json --device cpu --workers 1
```

**Device options:**
- `cuda:N` - NVIDIA GPU (N = device index)
- `mps` - Apple Silicon GPU (M1/M2/M3)
- `cpu` - CPU fallback

## Dataset Format

| Column | Type | Description |
|--------|------|-------------|
| `clinical_text` | string | Clinical narrative |
| `treatment_indicator` | int | Binary (0/1) |
| `outcome_indicator` | int | Binary (0/1) |
| `split` | string | Optional: "train"/"val"/"test" |

## Training Pattern

All extractors follow the same pattern:

```python
from cdt.models import CausalText

model = CausalText(
    feature_extractor_type="gated_mil_hierarchical",  # or cnn, bert, gru, confounder, hierarchical_transformer
    model_type="rlearner",  # or dragonnet, uplift, traditional_logreg
    device="cuda:0",
    # ... extractor-specific params (see examples/ configs)
)

# Required for cnn, gru, confounder (GRU mode only)
model.fit_tokenizer(train_texts)

# Training loop
for batch in dataloader:
    losses = model.train_step(
        batch,
        alpha_propensity=1.0,
        gamma_rlearner=1.0,  # R-learner weight
        beta_targreg=0.1,    # DragonNet targeted regularization
        stop_grad_propensity=False,  # Prevent propensity dominating features
        attention_entropy_weight=0.0  # Encourage focused attention
    )
    losses['loss'].backward()
    optimizer.step()

# Predictions
preds = model.predict(texts)
ite = preds['y1_prob'] - preds['y0_prob']
```

See `example_configs/` for complete config files for each extractor type.

## Extractor-Specific Notes

### CNN (`cnn_extractor.py`)
- Semantic filter init from explicit clinical concepts
- K-means filter init from training n-grams
- `interpret_filters()` for filter interpretability

### Confounder (`confounder_extractor.py`)
Perceiver-style with K learnable latent queries and sparse attention (entmax).

| Mode | Flag | Encoder | Notes |
|------|------|---------|-------|
| Sentence-level | default | SentenceTransformer | Fast, pools sentences |
| Hierarchical | `confounder_hierarchical=True` | BERT per sentence | Token-level attention |
| GRU | `confounder_use_gru=True` | Learnable BiGRU | Learns from scratch, needs fit_tokenizer |

Key params: `confounder_num_latents`, `confounder_sparse_alpha` (1.5=entmax15), `confounder_explicit_texts`

### Gated MIL (`gated_mil_hierarchical_extractor.py`)
Gated attention (tanh × sigmoid) with K confounder queries and task-specific weighting.

| Mode | Flag | Notes |
|------|------|-------|
| Chunk-level | default | [CLS] per chunk |
| Token-level | `gated_mil_hierarchical=True` | Token-level gated pooling |
| Mean pooling | `gated_mil_use_mean_pooling=True` | Mean pool vs [CLS] |

Key params: `gated_mil_max_chunks`, `gated_mil_chunk_size`, `gated_mil_chunk_overlap`, `gated_mil_num_confounders`

Interpretability: `interpret_attention()`, `get_task_weights()`

### Hierarchical Transformer (`hierarchical_transformer_extractor.py`)
Simple: chunk BERT → transformer layers → [POOL] token aggregation.

Key params: `hier_transformer_max_chunks`, `hier_transformer_chunk_size`, `hier_transformer_chunk_overlap`

### GRU-Transformer-MIL (`gru_transformer_mil_extractor.py`)
Combines BiGRU chunk encoding (learns from scratch) with transformer cross-chunk processing
and gated MIL attention with K confounder queries.

| Stage | Component | Description |
|-------|-----------|-------------|
| Chunk encoding | BiGRU + attention | Shared GRU pools tokens within each chunk |
| Cross-chunk | Transformer | Adds positional info and cross-chunk context |
| Aggregation | Gated MIL | K confounder queries with task-specific weighting |

Key params: `gru_mil_embedding_dim`, `gru_mil_gru_hidden_dim`, `gru_mil_transformer_layers`,
`gru_mil_num_confounders`, `gru_mil_chunk_size`

Requires `fit_tokenizer()` since it learns vocabulary from scratch.

Interpretability: `interpret_attention()`, `get_task_weights()`

### GRU-Pool (`gru_pool_extractor.py`)
Simpler variant of GRU-Transformer-MIL: BiGRU chunk encoding + transformer cross-chunk context
+ gated attention pooling for final aggregation. Produces a single feature vector (no task-specific
K confounder queries).

| Stage | Component | Description |
|-------|-----------|-------------|
| Chunk encoding | BiGRU + attention | Shared GRU pools tokens within each chunk |
| Cross-chunk | Transformer | Adds positional info and cross-chunk context |
| Aggregation | Gated attention pooling | Single document vector via tanh×sigmoid gating |

Key params: `gru_pool_embedding_dim`, `gru_pool_gru_hidden_dim`, `gru_pool_transformer_layers`,
`gru_pool_gated_attention_dim`, `gru_pool_chunk_size`

Requires `fit_tokenizer()` since it learns vocabulary from scratch.

Interpretability: `interpret_attention()`, `get_attention_weights()`

### LLM (`llm_extractor.py`)
Decoder-only LLM (e.g., Qwen3-0.6B-Base) with **random weight initialization** and last token embedding.
Uses the architecture from a pretrained model but trains entirely from scratch via the supervised causal objective.

| Component | Description |
|-----------|-------------|
| Architecture | Qwen3-0.6B (28 layers, GQA, RoPE, SwiGLU) |
| Tokenizer | Pretrained BBPE tokenizer (151K vocab) |
| Embedding | Last token hidden state (GPT-style, left-padded) |
| Projection | 2-layer MLP with LayerNorm |

Key params: `llm_model_name`, `llm_max_length`, `llm_projection_dim`, `llm_gradient_checkpointing`

No `fit_tokenizer()` required - uses pretrained tokenizer from HuggingFace.

**Memory Considerations:**
| Context Length | Recommended Batch Size | Notes |
|----------------|------------------------|-------|
| 32K | 1-2 | Requires gradient checkpointing |
| 8K | 4-8 | Good balance for most use cases |
| 2K | 16-32 | Fast iteration |

Gradient checkpointing is enabled by default for memory efficiency.

## Numeric Feature Extraction

Clinical text contains numbers critical for causal inference (lab values, vitals, scores, doses, ages)
that receive no special treatment from standard tokenizers. The numeric features module (`cdt/models/numeric_features.py`)
adds magnitude-aware numeric featurization as a parallel channel to all extractors.

### How It Works

1. **Regex extraction**: Detects integers, decimals, and fractions (e.g., BP 120/80) in raw text
2. **Log-scale magnitude binning**: Maps values into 8 bins: `[0, 0.1, 1, 10, 100, 1000, 10000, 100000]`
3. **Context-based type detection**: Classifies numbers by preceding keywords into 10 categories
   (vitals, labs, scores, demographics, doses, etc.)
4. **Injection into extractor pipeline**: Two strategies depending on architecture

### Injection Strategies

| Strategy | Used By | Method |
|----------|---------|--------|
| `NumericEmbedding` (position-aligned) | `cnn`, `gru` | Added to word embeddings at token positions |
| `NumericFeatureVector` (document-level) | `bert`, `llm`, `gru_pool`, `hierarchical_transformer`, `gated_mil_hierarchical`, `gru_transformer_mil`, `confounder` | Aggregate histogram merged before output projection |

### Config Parameters

| Param | Description | Default |
|-------|-------------|---------|
| `numeric_features_enabled` | Enable numeric feature extraction | `False` |
| `numeric_embedding_dim` | Output dimension of numeric feature vectors | `32` |
| `numeric_magnitude_bins` | Number of log-scale magnitude bins | `8` |
| `numeric_type_categories` | Number of numeric type categories | `10` |

When `numeric_features_enabled` is `False` (default), there is no behavior change to any extractor.

## Explicit Confounder Extraction

Researchers can specify explicit confounder variables to be extracted from clinical text using an LLM
(via vLLM). The extracted confounders are featurized and concatenated to text embeddings before the
causal heads.

### How It Works

```
1. Config specifies explicit confounders (name, type, categories)
2. vLLM extracts confounders from clinical text (preprocessing step)
3. Generates structured values per patient with missingness flags
4. ExplicitConfounderFeaturizer MLP encodes confounders
5. Concatenated to text feature extractor output
6. Combined representation -> Causal heads (DragonNet, R-Learner, etc.)

For Causal Forest: Raw confounder features added directly to neural features
```

### Confounder Specification

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Confounder name (e.g., "performance_status") |
| `type` | string | "categorical" or "continuous" |
| `categories` | list | Valid categories for categorical (e.g., ["0", "1", "2", "3", "4"]) |
| `description` | string | Description used in LLM prompt |

### vLLM Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `server` | Connect to running vLLM OpenAI-compatible server | Production, shared infrastructure |
| `start_server` | Start vLLM server subprocess, then connect | Batch jobs with cleanup |
| `python_api` | Use vLLM Python API directly (in-process) | Single-run experiments |

### Config Parameters

| Param | Description | Default |
|-------|-------------|---------|
| `enabled` | Enable explicit confounder extraction | `False` |
| `confounders` | List of ExplicitConfounderSpec | `[]` |
| `vllm_mode` | "server", "start_server", or "python_api" | `"server"` |
| `vllm_server_url` | URL for vLLM server | `"http://localhost:8000/v1"` |
| `vllm_model_name` | Model name for extraction | `"Qwen/Qwen2.5-7B-Instruct"` |
| `vllm_tensor_parallel_size` | Number of GPUs | `1` |
| `extraction_batch_size` | Batch size for extraction | `32` |
| `extraction_max_retries` | Retries before marking missing | `3` |
| `cache_enabled` | Cache extraction results | `True` |
| `featurizer_output_dim` | MLP output dimension | `64` |
| `featurizer_hidden_dim` | MLP hidden dimension | `128` |

### Example Config

```json
{
  "explicit_confounders": {
    "enabled": true,
    "confounders": [
      {
        "name": "performance_status",
        "type": "categorical",
        "categories": ["0", "1", "2", "3", "4"],
        "description": "ECOG performance status"
      },
      {
        "name": "age_at_diagnosis",
        "type": "continuous",
        "description": "Patient age at diagnosis in years"
      }
    ],
    "vllm_mode": "python_api",
    "vllm_model_name": "Qwen/Qwen2.5-7B-Instruct",
    "cache_enabled": true,
    "featurizer_output_dim": 64
  }
}
```

### Featurization

For **neural models** (DragonNet, R-Learner, etc.):
- Categorical: k-1 dummy variables (reference coding)
- Continuous: Z-score normalized
- Missingness: Binary indicator per confounder
- MLP projection to `featurizer_output_dim`

For **Causal Forest**:
- Raw features (no MLP) for interpretability
- One-hot categoricals + normalized continuous + missingness indicators

### Caching

Extraction results are cached to avoid redundant LLM calls:
- Cache keyed by: dataset path hash + extraction config hash
- Cache location: `{dataset_dir}/.cdt_cache/extraction_{hash}.parquet`
- Invalidated automatically if config changes

## CLAM Instance-Level Loss

CLAM-style (Lu et al., Nature BME 2021) instance-level supervision is available for all hierarchical
extractors to improve ITE correlation. When enabled, a separate lightweight causal head supervises
the top-B attended chunks with document-level labels.

### Supported Extractors

| Extractor | Instance Embedding Dim | Attention Aggregation |
|-----------|----------------------|----------------------|
| `gru_pool` | `transformer_dim` (256) | Gated attention weights |
| `hierarchical_transformer` | `transformer_dim` (256) | [POOL] token attention to chunks |
| `gated_mil_hierarchical` | `sentence_dim` (128 for bert-tiny) | Tau-weighted aggregation across K confounders |
| `gru_transformer_mil` | `transformer_dim` (256) | Tau-weighted aggregation across K confounders |

**Tau-Weighted Aggregation**: For extractors with K confounder queries (gated_mil_hierarchical,
gru_transformer_mil), attention is aggregated using the task-specific tau weights. This prioritizes
confounders most relevant to treatment effect modification, aligning CLAM supervision with the
causal objective.

### CLAM Parameters

| CLAM Param | Description | Default |
|------------|-------------|---------|
| `clam_enabled` | Enable CLAM instance-level loss | `False` |
| `clam_num_instances` | Number of top-attended chunks to supervise (B) | `5` |
| `clam_instance_hidden_dim` | Hidden dimension for instance causal head | `64` |
| `clam_instance_weight` | Weight for instance-level loss (training config) | `0.5` |

The instance head is completely independent from the document head (no weight sharing).
Works with DragonNet, UpliftNet, R-Learner, and TraditionalLogReg causal heads.

## Causal Forest Mode

When `model_type="causal_forest"`, CDT uses a two-stage approach combining neural feature extraction
with econml's CausalForestDML for treatment effect estimation.

### Architecture

```
Stage 1: Representation Learning (Neural Network)
├── Feature Extractor (any supported type: gru_pool, bert, etc.)
├── Propensity Head: P(T=1|X) → BCE loss
├── Outcome Head: E[Y|X] → BCE loss
└── [Optional] Effect Head: τ(X) → R-loss (when use_rlearner_representation=True)

Stage 1 with Dual Extractors (when rlearner_dual_extractors=True):
├── Nuisance Extractor (feature_extractor)
│   ├── Text → Features_nuisance
│   ├── Propensity Head → e(X) [BCE loss]
│   └── Outcome Head → m(X) [BCE loss]
└── Effect Extractor (effect_feature_extractor)
    └── Text → Features_effect → effect_mlp → τ(X) [R-loss]

Stage 2: Effect Estimation (Causal Forest)
├── Extract learned representations from Stage 1
│   (In dual mode: uses Effect Extractor features, optimized for τ)
├── Fit CausalForestDML on extracted features
└── Estimate τ(X) = E[Y(1)-Y(0)|X] with confidence intervals
```

**Key insight for dual mode**: In dual extractor mode, Stage 2 uses the effect extractor's features
because they are specifically optimized to capture treatment effect heterogeneity via the R-loss.
This provides the causal forest with representations that focus on effect modifiers rather than confounders.

### Causal Forest Parameters

| Param | Description | Default |
|-------|-------------|---------|
| `n_estimators` | Number of trees in the forest (must be divisible by 4) | `100` |
| `max_depth` | Maximum depth of trees (None = unlimited) | `None` |
| `min_samples_leaf` | Minimum samples per leaf | `5` |
| `max_features` | Feature subset strategy for splitting | `"sqrt"` |
| `honest` | Use honest estimation (sample splitting within trees) | `True` |
| `inference` | Enable confidence intervals | `True` |
| `use_rlearner_representation` | Add τ head and R-loss to Stage 1 training | `False` |
| `gamma_rlearner` | Weight for R-learner loss during representation training | `1.0` |
| `rlearner_dual_extractors` | Use separate extractors for nuisance vs effect (with `use_rlearner_representation`) | `False` |

**Note**: Nuisance functions (propensity and outcome) are estimated using sklearn random forests
on the neural network's learned features. The neural network's key contribution is the
learned text representation that captures confounders.

**Memory Note**: Dual extractor mode approximately doubles feature extraction memory/compute.

### R-Learner Representation Training

When `use_rlearner_representation=True`, Stage 1 adds a treatment effect head (τ) and trains
with the R-learner loss in addition to propensity and outcome losses. This encourages the
neural network to learn representations that capture treatment effect heterogeneity, not just
confounders.

**R-loss formula**: `E[((Y - m(X)) - τ(X)(T - e(X)))²]`

**Key insight**: Nuisance functions (e, m) are **DETACHED** during R-loss computation, so gradients
flow only through the τ head. This provides direct signal for learning treatment effect modifiers
from text without interference from nuisance estimation.

### Usage

```python
# Config with causal forest (basic)
config = {
    "architecture": {
        "model_type": "causal_forest",
        "feature_extractor_type": "gru_pool",
        "causal_forest": {
            "n_estimators": 200,
            "min_samples_leaf": 10,
            "honest": True,
            "inference": True
        }
    }
}

# Config with R-learner representation training
config = {
    "architecture": {
        "model_type": "causal_forest",
        "feature_extractor_type": "gru_pool",
        "causal_forest": {
            "n_estimators": 200,
            "min_samples_leaf": 10,
            "honest": True,
            "inference": True,
            "use_rlearner_representation": True,
            "gamma_rlearner": 1.0
        }
    }
}

# Config with dual extractor mode (separate nuisance and effect extractors)
config = {
    "architecture": {
        "model_type": "causal_forest",
        "feature_extractor_type": "gru_pool",
        "causal_forest": {
            "n_estimators": 200,
            "min_samples_leaf": 10,
            "honest": True,
            "inference": True,
            "use_rlearner_representation": True,
            "gamma_rlearner": 1.0,
            "rlearner_dual_extractors": True
        }
    }
}

# Predictions include confidence intervals
preds = model.predict(texts)
# preds['tau_pred'] - point estimates
# preds['tau_lower'], preds['tau_upper'] - 95% CIs
```

### Advantages

1. **Doubly-robust estimation**: Robust to misspecification of either propensity or outcome model
2. **Honest trees**: Unbiased effect estimates via sample splitting within trees
3. **Confidence intervals**: Built-in uncertainty quantification
4. **No gradient interference**: Representation learning is complete before effect estimation
5. **Theoretical guarantees**: Asymptotic normality and coverage guarantees

## Training Options for τ Learning

| Option | Effect |
|--------|--------|
| `stop_grad_propensity=True` | Prevents propensity from dominating representation |
| `attention_entropy_weight>0` | Encourages focused attention (low entropy) |
| `gamma_rlearner>1.0` | Stronger treatment effect signal |
| `clam_enabled=True` | Enables CLAM instance-level loss (hierarchical extractors) |
| `clam_instance_weight>0` | Weight for instance-level loss on top-attended chunks |
| `numeric_features_enabled=True` | Adds magnitude-aware numeric featurization from clinical text |
| `rlearner_dual_extractors=True` | Uses separate extractors for nuisance (e,m) and effect (τ) in R-Learner |

## Matching & Analysis

```python
from cdt.matching import PropensityMatcher
from cdt.analysis import run_psm_analysis, estimate_att_matched, estimate_ate_ipw

# Matching
matcher = PropensityMatcher(method='nearest', caliper=0.2)
match_result = matcher.match(propensity_scores, treatment)

# Full PSM analysis
results = run_psm_analysis(predictions_df, config, output_dir)
```

## Workflow Modes

1. **Applied Inference**: K-fold CV or fixed splits → `predictions.parquet`
2. **Plasmode Simulation**: Synthetic outcomes with known ATE for validation
3. **PSM Analysis**: Post-hoc matching with ATT/ATE estimation, Rosenbaum bounds

## Output Files

```
output_dir/
├── config.json
├── applied_inference/
│   ├── predictions.parquet
│   ├── training_log.csv
│   ├── *_interpretations.json  # Filter/confounder attention
│   └── psm_analysis/           # If enabled
└── plasmode_experiments/       # If enabled
```

## Key Files

| Purpose | Files |
|---------|-------|
| Main model | `cdt/models/causal_text.py` |
| Causal forest model | `cdt/models/causal_text_forest.py`, `cdt/models/causal_forest_head.py` |
| Causal heads | `dragonnet.py`, `rlearner.py`, `uplift.py`, `traditional_logreg.py` |
| Extractors | `cnn_extractor.py`, `bert_extractor.py`, `gru_extractor.py`, `confounder_extractor.py`, `hierarchical_transformer_extractor.py`, `gated_mil_hierarchical_extractor.py`, `gru_transformer_mil_extractor.py`, `gru_pool_extractor.py`, `llm_extractor.py` |
| Numeric features | `cdt/models/numeric_features.py` |
| Explicit confounders | `cdt/extraction/explicit_confounders.py`, `cdt/extraction/cache.py`, `cdt/models/explicit_confounder_featurizer.py` |
| Text chunking | `cdt/models/chunking.py` |
| Training | `cdt/inference/applied.py`, `cdt/inference/applied_forest.py` |
| Config | `cdt/config.py` |
| PSM | `cdt/analysis/psm_analysis.py`, `cdt/matching/propensity_matcher.py` |

## Dependencies

**Core**: torch, transformers, pandas, numpy, scikit-learn, tqdm, pyarrow, econml

**Optional**: openai (synthetic data), sentence-transformers (confounder), entmax (sparse attention; fallback provided), vllm (explicit confounder extraction)

**Device support**: CUDA (NVIDIA GPUs), MPS (Apple Silicon M1/M2/M3), CPU

## Adding a New Feature Extractor

When adding a new feature extractor type, update ALL of the following files:

| File | What to Update |
|------|----------------|
| `cdt/models/new_extractor.py` | Create the new extractor module |
| `cdt/models/__init__.py` | Add exports for new classes |
| `cdt/config.py` | Add `normalize_feature_extractor_type()` entry and config options |
| `cdt/models/causal_text.py` | Add import, `__init__` params, config storage, instantiation case |
| `cdt/inference/applied.py` | Add CausalText params and initialization path |
| `cdt/training/plasmode.py` | Add CausalText params and initialization path |
| `cdt/models/propensity_model.py` | Add import, `__init__` params, config, instantiation, `create_propensity_model_from_config()` |
| `cdt/training/propensity_trimming.py` | Add initialization path in `_train_propensity_model()` |
| `example_configs/new_config.json` | Create example configuration file |
| `CLAUDE.md` | Update Feature Extractors table, architecture docs, file lists |
| `README.md` | Update documentation if significant user-facing changes |

**Checklist for new extractor:**
1. Create extractor module with `forward()`, `fit_tokenizer()` (if needed), `interpret_attention()`, `get_state()`
2. Add to `__init__.py` exports
3. Add normalization alias in `config.py` (e.g., `"gru_pool"` -> `"gru_pool"`)
4. Add all config options to `ModelArchitectureConfig` dataclass
5. Add instantiation in `CausalText.__init__()` with logging
6. Add params to `CausalText` constructor and config dict
7. Mirror changes to `applied.py`, `plasmode.py`, `propensity_model.py`, `propensity_trimming.py`
8. Create example config JSON
9. Update this file's documentation tables
10. Test with unit tests and integration tests

## Quick Reference

- **ITE**: `preds['y1_prob'] - preds['y0_prob']` (probability scale)
- **Tokenizer**: Required for `cnn`, `gru`, `confounder` with GRU mode, `gru_transformer_mil`, `gru_pool`
- **Long docs**: Use `confounder`, `hierarchical_transformer`, `gated_mil_hierarchical`, `gru_transformer_mil`, `gru_pool`, or `llm`
- **Interpretability**: `interpret_filters()` (CNN), `interpret_attention()` (others)
- **R-Learner vs DragonNet**: R-Learner for heterogeneous treatment effects; DragonNet for general use
- **LLM extractor**: Random init, pretrained tokenizer, up to 32K context, use small batch sizes
