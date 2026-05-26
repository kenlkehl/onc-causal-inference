# CLAUDE.md - OCI (Oncology Causal Inference)

## Overview

OCI estimates treatment effects from clinical text by combining text feature extraction with causal inference heads. Optional explicit feature extraction turns selected clinical variables into role-tagged structured features for confounding control (`W`) and effect modification (`X`).

## Repository Structure

```
oci/
├── cli.py                 # CLI: `oci init`, `oci run`
├── config.py              # Dataclass configs
├── data/
│   ├── dataset.py                    # ClinicalTextDataset
│   ├── cached_hidden_state_dataset.py  # Dataset for pre-cached hidden states
│   └── collators.py                  # Collator utilities (returns None for frozen LLM)
├── experiments/runner.py  # Orchestrates inference
├── extraction/
│   ├── explicit_confounders.py   # LLM-based explicit feature extraction via vLLM
│   └── cache.py                  # Extraction result caching
├── inference/
│   ├── applied.py             # Applied inference (CV or fixed split)
│   ├── applied_forest.py      # Causal forest inference pipeline
│   ├── applied_tfidf_forest.py  # TF-IDF forest baseline pipeline
│   └── applied_confounder_forest.py  # Legacy confounders-only forest pipeline
├── models/
│   ├── causal_text.py                # Main model (extractor + causal head)
│   ├── causal_text_forest.py         # Two-stage neural + causal forest model
│   ├── causal_forest_head.py         # CausalForestDML wrapper
│   ├── frozen_llm_pooler_extractor.py  # Frozen LLM + gated attention pooling
│   ├── hierarchical_llm_extractor.py  # Frozen LLM on overlapping chunks + two-level pooling
│   ├── hierarchical_cnn_extractor.py  # Dilated CNN on overlapping chunks + two-level pooling
│   ├── hierarchical_gru_extractor.py  # BiGRU on overlapping chunks + two-level pooling
│   ├── simple_cnn_extractor.py        # Dilated CNN on whole text + gated attention pooling
│   ├── text_chunking.py               # Shared token-based overlapping chunking utility
│   ├── learned_tokenizer.py           # Word-level tokenizer (learned from training data)
│   ├── gated_attention_pooling.py    # GatedAttentionPooling module
│   ├── hidden_state_cache.py         # Disk-based hidden state cache
│   ├── gpu_hidden_state_store.py     # GPU-resident hidden state store
│   ├── extractor_factory.py          # Factory for creating feature extractors
│   ├── dragonnet.py                  # DragonNet causal head
│   ├── rlearner.py                   # R-Learner causal head
│   ├── explicit_confounder_featurizer.py  # MLP featurization of extracted explicit features
│   ├── propensity_model.py           # Propensity-only model for trimming
│   └── outcome_model.py              # Outcome-only model for assessment
├── training/
│   ├── propensity_trimming.py # Propensity score trimming
│   └── outcome_training.py   # Standalone outcome model training
├── matching/
│   └── propensity_matcher.py  # PropensityMatcher, balance utilities
├── analysis/
│   ├── psm_analysis.py        # PSM analysis pipeline
│   └── statistical_analysis.py  # ATT/ATE estimation, Rosenbaum bounds
└── utils/
    ├── io.py                  # File I/O, hashing, atomic save
    └── system.py              # Thread limiting, seeding, CUDA cleanup

oracle_experiment_scripts/   # Oracle experiment runner and analysis
example_configs/             # Config files for frozen_llm_pooler and tfidf_forest
synthetic_data/              # LLM-based synthetic data generation
├── cli.py                 # CLI: `python -m synthetic_data.cli`
├── config.py              # SyntheticDataConfig, StructuredDataConfig
├── generator.py           # Main generation pipeline (HTTP API + vLLM batch)
├── prompts.py             # LLM prompt templates, build_event_timeline_prompt()
├── structured_data.py     # Structured event parsing + template text conversion
├── llm_client.py          # OpenAI-compatible LLM client
└── vllm_batch_client.py   # Direct vLLM batch inference client
```

## Architecture

### Feature Extractors

| Type | Description | Chunking | Tokenizer | Requires fit_tokenizer | Config Prefix |
|------|-------------|----------|-----------|----------------------|---------------|
| `frozen_llm_pooler` | Frozen pretrained LLM + gated attention pooling | No | Pretrained HF | No | `flp_*` |
| `hierarchical_llm` | Frozen pretrained LLM on overlapping chunks + two-level pooling | Yes (token-based) | Pretrained HF | No | `hlm_*` |
| `hierarchical_cnn` | Dilated 1D CNN on overlapping chunks + two-level pooling | Yes (token-based) | Learned (word-level) | Yes | `hcnn_*` |
| `hierarchical_gru` | BiGRU on overlapping chunks + two-level pooling | Yes (token-based) | Learned (word-level) | Yes | `hgru_*` |
| `simple_cnn` | Dilated 1D CNN on whole text + gated attention pooling | No | Learned (word-level) | Yes | `scnn_*` |

All extractors produce a fixed-size feature vector per document and share a common two-stage structure: encode token sequences, then pool into a single vector via `GatedAttentionPooling` (tanh x sigmoid gating + softmax attention). Hierarchical extractors add a second pooling level: tokens are pooled within each chunk, then chunk vectors are pooled into a document vector.

Extractors are instantiated via `extractor_factory.py`, which centralizes all creation logic used by `CausalText`, `CausalTextForest`, `PropensityOnlyModel`, and `OutcomeOnlyModel`.

### Causal Heads

| Type | Description | Key output |
|------|-------------|------------|
| `dragonnet` | Propensity + Y0/Y1 potential outcomes | ITE = sigma(y1) - sigma(y0) |
| `rlearner` | Direct tau(X) optimization, detached nuisance functions | tau directly predicts ITE |
| `causal_forest` | Two-stage: neural features + econml CausalForestDML | tau with confidence intervals |
| `tfidf_forest` | TF-IDF features + econml CausalForestDML (no neural network) | tau with confidence intervals |
| `explicit_feature_forest` | Role-tagged explicit features + econml CausalForestDML (no text features) | tau with confidence intervals |

**R-Learner advantage**: Nuisance functions (e, m) are detached in R-loss, providing stronger gradient signal for treatment effect modifiers.

### Staged R-Learner Causal Forest

For `causal_forest` with `use_rlearner_representation=True`, the forest path trains separate nuisance and effect representations. Nuisance nets learn `e(W)` and `m(W)`; inner-fold out-of-fold nuisance predictions train the effect net with R-loss; the causal forest receives effect features as `X` and nuisance features as `W`.

## CLI

```bash
oci init --output config.json
oci run --config config.json --device cuda:0 --workers 4 [--skip-pretraining] [--verbose]

# Apple Silicon (MPS)
oci run --config config.json --device mps --workers 1

# CPU fallback
oci run --config config.json --device cpu --workers 1
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
| `outcome_indicator` | int/float | Binary (0/1) or continuous |
| `split` | string | Optional: "train"/"val"/"test" |

Set `outcome_type` in config: `"binary"` (default, BCE loss + sigmoid) or `"continuous"` (MSE loss, no sigmoid). Treatment/propensity is always binary.

## Training Pattern

```python
from oci.models import CausalText

model = CausalText(
    feature_extractor_type="frozen_llm_pooler",
    model_type="rlearner",  # or dragonnet
    device="cuda:0",
    flp_model_name="Qwen/Qwen3-0.6B-Base",
    flp_max_length=8192,
    flp_freeze_llm=True,
    flp_projection_dim=128,
)

# No fit_tokenizer() needed -- uses pretrained HF tokenizer

# Training loop
for batch in dataloader:
    losses = model.train_step(
        batch,
        alpha_propensity=1.0,
        gamma_rlearner=1.0,  # R-learner weight
        beta_targreg=0.1,    # DragonNet targeted regularization
        stop_grad_propensity=False,  # Prevent propensity dominating features
    )
    losses['loss'].backward()
    optimizer.step()

# Predictions (binary: probabilities, continuous: raw values)
preds = model.predict(texts)
ite = preds['y1_prob'] - preds['y0_prob']
```

See `example_configs/` for complete config files.

## Frozen LLM Pooler Details

Pretrained decoder-only LLM with frozen weights + GatedAttentionPooling over all token hidden states. Pools information from ALL tokens via gated attention, producing a rich representation while keeping the LLM frozen.

**Default mode (live forward)**: The frozen LLM runs per batch with `torch.no_grad()` and `torch.cuda.amp.autocast(float16)`. An optional trainable downprojection layer reduces the hidden state dimensionality before pooling.

### Config Parameters

| Param | Description | Default |
|-------|-------------|---------|
| `flp_model_name` | HuggingFace model name | `"Qwen/Qwen3-0.6B-Base"` |
| `flp_max_length` | Max sequence length | `8192` |
| `flp_freeze_llm` | Freeze LLM backbone | `True` |
| `flp_gated_attention_dim` | Hidden dim for gated attention pooling | `128` |
| `flp_projection_dim` | Final output dimension | `128` |
| `flp_dropout` | Dropout rate for projection layers | `0.1` |
| `flp_gradient_checkpointing` | Gradient checkpointing (when not frozen) | `True` |
| `flp_downprojection_dim` | Trainable linear projection dim before pooling (None = no downprojection) | `None` |
| `flp_cache_hidden_states` | Pre-compute and cache LLM hidden states to disk | `False` |
| `flp_gpu_cache` | Keep hidden states on GPU VRAM instead of disk | `False` |
| `flp_random_projection_dim` | Random linear projection for cached hidden states | `None` |
| `flp_chat_template_prompt` | Chat template prompt for instruct models. Wraps each text in the model's chat template with this prompt preceding the clinical text. `None` = disabled (raw text). | `None` |

Interpretability: `interpret_attention()`, `get_attention_weights()` (not available in cached mode).

### Hidden State Caching

When caching is enabled (`flp_cache_hidden_states: true`) and the LLM is frozen, hidden states are pre-computed once for the entire dataset, cached to disk as float16 memmap files, and reused across K-fold CV folds and experiment runs. During training, the LLM is not loaded, saving approximately 2.4 GB of GPU memory.

**Cache details:**
- **Location**: `{dataset_dir}/.oci_cache/flp_hidden_states_{hash}/`
- **Key**: `(model_name, max_length, dataset_path, random_projection_dim)` -- different causal heads, learning rates, fold counts all share the same cache
- **Format**: Variable-length flat format: `hidden_states.npy` (float16 memmap, total_tokens x hidden_size) + `offsets.npy` (int64, N+1 sample boundaries) + `metadata.json`
- **Storage**: No padding waste -- per-batch padding happens during collation
- **Reuse**: Cache is automatically reused across experiments with the same model/dataset
- **Random projection**: When `flp_random_projection_dim` is set, a deterministic random Gaussian matrix projects hidden states before caching

**GPU Cache** (`flp_gpu_cache: true`): Keeps hidden states on GPU VRAM as a flat float16 tensor instead of disk memmap. Zero CPU-GPU transfer during training. Falls back to disk if insufficient VRAM.

### Memory Considerations

| Context Length | Recommended Batch Size | Notes |
|----------------|------------------------|-------|
| 32K | 1-2 | Requires gradient checkpointing |
| 8K | 4-8 | Good balance for most use cases |
| 2K | 16-32 | Fast iteration |

## Hierarchical LLM Details

Frozen pretrained LLM applied to overlapping token chunks with two-level gated attention pooling. Handles documents longer than the LLM's context window by splitting into overlapping chunks, running the LLM on each chunk independently, then aggregating with two pooling layers.

```
Raw text
  -> Pretrained tokenizer (full text, truncated to chunk_size * max_chunks)
  -> Overlapping token-based chunking (chunk_size tokens, chunk_overlap overlap)
  -> For each chunk: Frozen LLM -> hidden states -> [optional downprojection]
     -> GatedAttentionPooling (token-level) -> chunk_vector
  -> All chunk vectors -> GatedAttentionPooling (document-level) -> document_vector
  -> 2-layer Projection MLP -> (batch, output_dim)
```

Supports hidden state caching (`hlm_cache_hidden_states`) with the same disk/GPU cache infrastructure as `frozen_llm_pooler`. No `fit_tokenizer()` required.

### Config Parameters

| Param | Description | Default |
|-------|-------------|---------|
| `hlm_model_name` | HuggingFace model name | `"Qwen/Qwen3-0.6B-Base"` |
| `hlm_chunk_size` | Tokens per chunk | `2048` |
| `hlm_chunk_overlap` | Overlapping tokens between consecutive chunks | `256` |
| `hlm_max_chunks` | Maximum chunks per document | `16` |
| `hlm_freeze_llm` | Freeze LLM backbone | `True` |
| `hlm_gated_attention_dim` | Hidden dim for gated attention pooling | `128` |
| `hlm_projection_dim` | Final output dimension | `128` |
| `hlm_dropout` | Dropout rate | `0.1` |
| `hlm_gradient_checkpointing` | Gradient checkpointing (when not frozen) | `True` |
| `hlm_downprojection_dim` | Trainable linear projection before pooling (None = no downprojection) | `None` |
| `hlm_cache_hidden_states` | Pre-compute and cache LLM hidden states to disk | `False` |
| `hlm_gpu_cache` | Keep hidden states on GPU VRAM instead of disk | `False` |
| `hlm_chat_template_prompt` | Chat template prompt for instruct models (None = disabled) | `None` |

Interpretability: `interpret_attention()` returns chunk-level attention weights (not available in cached mode).

## Hierarchical CNN Details

Dilated 1D CNN applied to overlapping token chunks with two-level gated attention pooling. Trains entirely from scratch with a learned word-level tokenizer. Lightweight alternative to LLM-based extractors.

```
Raw text
  -> LearnedTokenizer (word-level, truncated to chunk_size * max_chunks)
  -> Overlapping token-based chunking
  -> For each chunk: nn.Embedding -> DilatedConvStack (dilation 1,2,4,8,...)
     -> GatedAttentionPooling (token-level) -> chunk_vector
  -> All chunk vectors -> GatedAttentionPooling (document-level) -> document_vector
  -> 2-layer Projection MLP -> (batch, output_dim)
```

The `DilatedConvStack` uses residual blocks with exponentially increasing dilation (1, 2, 4, 8, ...) for a wide receptive field. Shared CNN weights process all chunks. Requires `fit_tokenizer()` before training.

### Config Parameters

| Param | Description | Default |
|-------|-------------|---------|
| `hcnn_embedding_dim` | Word embedding dimension | `256` |
| `hcnn_conv_dim` | CNN hidden dimension | `256` |
| `hcnn_kernel_size` | Convolution kernel size | `5` |
| `hcnn_num_conv_blocks` | Number of dilated residual conv blocks | `4` |
| `hcnn_chunk_size` | Tokens per chunk | `512` |
| `hcnn_chunk_overlap` | Overlapping tokens between consecutive chunks | `64` |
| `hcnn_max_chunks` | Maximum chunks per document | `32` |
| `hcnn_vocab_size` | Vocabulary size | `50000` |
| `hcnn_gated_attention_dim` | Hidden dim for gated attention pooling | `128` |
| `hcnn_projection_dim` | Final output dimension | `128` |
| `hcnn_dropout` | Dropout rate | `0.1` |

## Hierarchical GRU Details

Bidirectional GRU applied to overlapping token chunks with two-level gated attention pooling. Trains entirely from scratch with a learned word-level tokenizer. Uses packed sequences for efficient variable-length processing.

```
Raw text
  -> LearnedTokenizer (word-level, truncated to chunk_size * max_chunks)
  -> Overlapping token-based chunking
  -> For each chunk: nn.Embedding -> BiGRU (output = 2 * gru_hidden_dim)
     -> GatedAttentionPooling (token-level) -> chunk_vector
  -> All chunk vectors -> GatedAttentionPooling (document-level) -> document_vector
  -> 2-layer Projection MLP -> (batch, output_dim)
```

Shared BiGRU weights process all chunks. The bidirectional GRU produces forward + backward hidden states concatenated to `2 * gru_hidden_dim`. Requires `fit_tokenizer()` before training.

### Config Parameters

| Param | Description | Default |
|-------|-------------|---------|
| `hgru_embedding_dim` | Word embedding dimension | `256` |
| `hgru_gru_hidden_dim` | Hidden dimension per GRU direction (output = 2x) | `256` |
| `hgru_num_gru_layers` | Number of stacked BiGRU layers | `2` |
| `hgru_chunk_size` | Tokens per chunk | `512` |
| `hgru_chunk_overlap` | Overlapping tokens between consecutive chunks | `64` |
| `hgru_max_chunks` | Maximum chunks per document | `32` |
| `hgru_vocab_size` | Vocabulary size | `50000` |
| `hgru_gated_attention_dim` | Hidden dim for gated attention pooling | `128` |
| `hgru_projection_dim` | Final output dimension | `128` |
| `hgru_dropout` | Dropout rate | `0.1` |

## Simple CNN Details

Dilated 1D CNN applied to the whole document (no chunking) with gated attention pooling. Trains from scratch with a learned word-level tokenizer. Simplest and fastest extractor -- suitable for shorter documents or as a baseline.

```
Raw text
  -> LearnedTokenizer (word-level, truncated to max_length)
  -> nn.Embedding
  -> DilatedConvStack (dilation 1,2,4,8,...)
  -> GatedAttentionPooling -> document_vector
  -> 2-layer Projection MLP -> (batch, output_dim)
```

No chunking -- processes the full tokenized sequence up to `scnn_max_length`. Requires `fit_tokenizer()` before training.

### Config Parameters

| Param | Description | Default |
|-------|-------------|---------|
| `scnn_embedding_dim` | Word embedding dimension | `256` |
| `scnn_conv_dim` | CNN hidden dimension | `256` |
| `scnn_kernel_size` | Convolution kernel size | `5` |
| `scnn_num_conv_blocks` | Number of dilated residual conv blocks | `4` |
| `scnn_max_length` | Maximum token sequence length | `10000` |
| `scnn_vocab_size` | Vocabulary size | `50000` |
| `scnn_gated_attention_dim` | Hidden dim for gated attention pooling | `128` |
| `scnn_projection_dim` | Final output dimension | `128` |
| `scnn_dropout` | Dropout rate | `0.1` |

## Explicit Feature Extraction

Researchers can specify structured variables to extract from clinical text using an LLM (via vLLM). Each feature declares one or more causal roles: `confounder`, `effect_modifier`, or both. Confounder-role features are used for nuisance adjustment (`W`); effect-modifier-role features are used for heterogeneity/effect estimation (`X`).

### How It Works

```
1. Config specifies explicit features (name, type, categories, roles)
2. vLLM extracts structured feature values from clinical text (preprocessing step)
3. Generates structured values per patient with missingness flags
4. ExplicitFeatureFeaturizer MLP encodes role-specific features for neural heads
5. Role-specific embeddings are concatenated to text features before heads
6. Combined representation -> Causal heads (DragonNet, R-Learner, etc.)

For Causal Forest:
- Raw confounder-role features are passed to W
- Raw effect-modifier-role features are passed to X
- Features with both roles are included in both matrices
```

### Feature Specification

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Feature name (e.g., "performance_status") |
| `type` | string | "categorical" or "continuous" |
| `categories` | list | Valid categories for categorical (e.g., ["0", "1", "2", "3", "4"]) |
| `description` | string | Description used in LLM prompt |
| `roles` | list | One or both of `"confounder"`, `"effect_modifier"` |

### vLLM Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `server` | Connect to running vLLM OpenAI-compatible server | Production, shared infrastructure |
| `start_server` | Start vLLM server subprocess, then connect | Batch jobs with cleanup |
| `python_api` | Use vLLM Python API directly (in-process) | Single-run experiments |

### Config Parameters

| Param | Description | Default |
|-------|-------------|---------|
| `enabled` | Enable explicit feature extraction | `False` |
| `features` | List of role-tagged ExplicitFeatureSpec | `[]` |
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
  "applied_inference": {
    "explicit_features": {
      "enabled": true,
      "features": [
        {
          "name": "performance_status",
          "type": "categorical",
          "categories": ["0", "1", "2", "3", "4"],
          "description": "ECOG performance status",
          "roles": ["confounder", "effect_modifier"]
        },
        {
          "name": "age_at_diagnosis",
          "type": "continuous",
          "description": "Patient age at diagnosis in years",
          "roles": ["confounder"]
        },
        {
          "name": "pdl1_expression",
          "type": "continuous",
          "description": "Tumor PD-L1 expression percentage",
          "roles": ["effect_modifier"]
        }
      ],
      "vllm_mode": "python_api",
      "vllm_model_name": "Qwen/Qwen2.5-7B-Instruct",
      "cache_enabled": true,
      "featurizer_output_dim": 64
    }
  }
}
```

### Featurization

For **neural models** (DragonNet, R-Learner):
- Categorical: k-1 dummy variables (reference coding)
- Continuous: Z-score normalized
- Missingness: Binary indicator per feature
- MLP projection to `featurizer_output_dim`

For **Causal Forest**:
- Raw role-specific features (no MLP) for interpretability
- Confounder-role features -> forest `W`
- Effect-modifier-role features -> forest `X`
- One-hot categoricals + normalized continuous + missingness indicators

### Caching

Extraction results are cached to avoid redundant LLM calls:
- Cache keyed by: dataset path hash + extraction config hash
- Cache location: `{dataset_dir}/.oci_cache/extraction_{hash}.parquet`
- Invalidated automatically if config changes

## Causal Forest Mode

When `model_type="causal_forest"`, OCI uses a two-stage approach combining neural feature extraction with econml's CausalForestDML for treatment effect estimation.

### Architecture

```
Stage 1a: Nuisance Representation
+-- Text/structured W -> W features
+-- Propensity Head -> e(W) [BCE loss]
+-- Outcome Head -> m(W) [BCE/MSE loss]

Stage 1b: Effect Representation
+-- Text/structured X -> X features
+-- Tau Head -> tau(X) [R-loss from OOF nuisance predictions]

Stage 2: Effect Estimation (Causal Forest)
+-- X: effect features plus raw effect-modifier features
+-- W: nuisance features plus raw confounder features
+-- Fit CausalForestDML on extracted features
+-- Estimate tau(X) = E[Y(1)-Y(0)|X] with confidence intervals
```

### Causal Forest Parameters

| Param | Description | Default |
|-------|-------------|---------|
| `n_estimators` | Number of trees in the forest (must be divisible by 4) | `100` |
| `max_depth` | Maximum depth of trees (None = unlimited) | `None` |
| `min_samples_leaf` | Minimum samples per leaf | `5` |
| `max_features` | Feature subset strategy for splitting | `"sqrt"` |
| `honest` | Use honest estimation (sample splitting within trees) | `True` |
| `inference` | Enable confidence intervals | `True` |
| `use_rlearner_representation` | Use staged nuisance/effect R-learner representation training | `False` |
| `gamma_rlearner` | Weight for R-learner loss during representation training | `1.0` |
| `rlearner_nuisance_folds` | Inner folds for out-of-fold nuisance predictions | `5` |

### R-Learner Representation Training

When `use_rlearner_representation=True`, the causal forest runner trains nuisance models first, then trains an effect representation with the R-learner loss using out-of-fold nuisance predictions.

**R-loss formula**: `E[((Y - m(X)) - tau(X)(T - e(X)))^2]`

Nuisance functions (e, m) are **DETACHED** during R-loss computation, so gradients flow only through the tau head.

### Advantages

1. **Doubly-robust estimation**: Robust to misspecification of either propensity or outcome model
2. **Honest trees**: Unbiased effect estimates via sample splitting within trees
3. **Confidence intervals**: Built-in uncertainty quantification
4. **No gradient interference**: Representation learning is complete before effect estimation
5. **Theoretical guarantees**: Asymptotic normality and coverage guarantees

## TF-IDF Forest Baseline

When `model_type="tfidf_forest"`, OCI uses a non-neural baseline: TF-IDF features directly with CausalForestDML. No GPU, no training epochs, no neural network.

### TF-IDF Forest Parameters

| Param | Description | Default |
|-------|-------------|---------|
| `max_features` | Maximum number of TF-IDF features | `10000` |
| `ngram_range_min` | Minimum n-gram size | `1` |
| `ngram_range_max` | Maximum n-gram size | `2` |
| `min_df` | Minimum document frequency (absolute count) | `5` |
| `max_df` | Maximum document frequency (proportion) | `0.95` |
| `sublinear_tf` | Use sublinear TF scaling (1 + log(tf)) | `True` |
| `n_estimators` | Number of trees (must be divisible by 4) | `200` |
| `min_samples_leaf` | Minimum samples per leaf | `10` |
| `honest` | Honest estimation | `True` |
| `inference` | Enable confidence intervals | `True` |

## Training Options for tau Learning

| Option | Effect |
|--------|--------|
| `stop_grad_propensity=True` | Prevents propensity from dominating representation |
| `attention_entropy_weight>0` | Encourages focused attention (low entropy) |
| `gamma_rlearner>1.0` | Stronger treatment effect signal |

## Propensity Trimming

When enabled, trains a propensity-only model using k-fold cross-validation to generate out-of-sample propensity scores, then trims the dataset by removing patients with extreme propensity scores. This enforces the positivity assumption for causal inference.

| Param | Description | Default |
|-------|-------------|---------|
| `enabled` | Enable propensity trimming | `False` |
| `min_propensity` | Remove patients below this threshold | `0.1` |
| `max_propensity` | Remove patients above this threshold | `0.9` |
| `cv_folds` | CV folds for propensity model | `5` |
| `propensity_epochs` | Training epochs | `20` |

## Outcome Model Pre-Assessment

When enabled, trains an outcome-only model using k-fold cross-validation to assess prognostic signal in the data before causal model training. Does NOT trim the dataset.

| Param | Description | Default |
|-------|-------------|---------|
| `enabled` | Enable outcome model training | `False` |
| `cv_folds` | CV folds for outcome model | `5` |
| `outcome_epochs` | Training epochs | `20` |

## Matching and Analysis

```python
from oci.matching import PropensityMatcher
from oci.analysis import run_psm_analysis, estimate_att_matched, estimate_ate_ipw

# Matching
matcher = PropensityMatcher(method='nearest', caliper=0.2)
match_result = matcher.match(propensity_scores, treatment)

# Full PSM analysis
results = run_psm_analysis(predictions_df, config, output_dir)
```

## Workflow Modes

1. **Applied Inference**: K-fold CV or fixed splits -> `predictions.parquet`
2. **Semi-Synthetic Simulation**: Real text + simulated T/Y with known ITE for sensitivity analysis (see `oracle_experiment_scripts/`)
3. **PSM Analysis**: Post-hoc matching with ATT/ATE estimation, Rosenbaum bounds

## Output Files

```
output_dir/
+-- config.json
+-- applied_inference/
|   +-- predictions.parquet
|   +-- training_log.csv
|   +-- *_interpretations.json  # Attention interpretations
|   +-- psm_analysis/           # If enabled
```

## Semi-Synthetic Simulation (Sensitivity Analysis)

The `oracle_experiment_scripts/` module provides a semi-synthetic simulation framework for
evaluating how well text extractors capture confounding beyond explicitly specified confounders.

### How It Works

1. LLM generates K realistic confounders for the clinical question
2. vLLM extracts confounders from real clinical text
3. Regression equations produce simulated treatment/outcome with known true ITE
4. Applied inference uses real text + simulated T/Y
5. ITE correlation measures recovery of true treatment effects

### Two Equation Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `random` | LLM generates regression equations with random coefficients, calibrated to target rates | Stress-testing across diverse DGPs |
| `fitted` | Logistic regression fit on extracted structured features to predict real T/Y | Stability analysis for specific dataset |

### Key Files

| File | Purpose |
|------|---------|
| `oracle_experiment_scripts/semisynthetic_dgp.py` | DGP generation: confounder extraction, equation generation/fitting, outcome simulation |
| `oracle_experiment_scripts/run_semisynthetic_experiments.py` | Main runner: outer/inner loops, CLI, checkpoint/resume |
| `oracle_experiment_scripts/analyze_semisynthetic_results.py` | Results aggregation, summary statistics, plots |

### CLI

```bash
# Random mode: M=5 DGPs, N=5 repeats each
python oracle_experiment_scripts/run_semisynthetic_experiments.py \
    --dataset-path /path/to/real/dataset.parquet \
    --clinical-question "Compare pembrolizumab vs docetaxel for advanced NSCLC" \
    --output-dir ../pcori_experiments/semisynthetic \
    --equation-mode random --num-dgps 5 --num-repeats 5 --devices cuda:0

# Fitted mode: learn equations from real T/Y, measure stability
python oracle_experiment_scripts/run_semisynthetic_experiments.py \
    --dataset-path /path/to/real/dataset.parquet \
    --clinical-question "Compare pembrolizumab vs docetaxel for advanced NSCLC" \
    --output-dir ../pcori_experiments/semisynthetic_fitted \
    --equation-mode fitted --num-dgps 5 --num-repeats 10 --devices cuda:0

# Analyze results
python oracle_experiment_scripts/analyze_semisynthetic_results.py \
    --results-dir ../pcori_experiments/semisynthetic
```

### Experiment Arms

For each DGP and confounder fraction f (0, 0.25, 0.5, 0.75, 1.0):
- **confounder_forest**: Confounder-only CausalForestDML (no text)
- **text_forest**: Frozen LLM pooler + CausalForestDML (with optional confounder subset)
- **best_attainable**: All K confounders (upper bound reference)

### Key Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--equation-mode` | "random" or "fitted" | `random` |
| `--num-dgps` | Number of DGPs (M) | `5` |
| `--num-repeats` | Repeats per DGP (N) | `5` |
| `--vary-confounders-per-dgp` | Fresh confounders per DGP | `True` for random, `False` for fitted |
| `--cache` / `--gpu-cache` | Pre-cache frozen LLM hidden states | `False` |
| `--resume` | Skip completed arms | `False` |

## Synthetic Data Generation

The `synthetic_data/` module generates synthetic clinical datasets with known causal structure for benchmarking. It uses an LLM to create realistic confounders, regression equations, and clinical narratives.

### Generation Modes

| Mode | Description |
|------|-------------|
| `two_stage` (default) | Event timeline -> note expansion. Produces multi-note documents. |
| `single_document` (legacy) | Single concatenated narrative per patient. |

### Two-Stage Pipeline

1. LLM generates confounders, regression equations, and summary statistics for the clinical question
2. Patient characteristics are sampled; treatment/outcome are determined by the regression equations
3. **Stage 1**: LLM generates a chronological event timeline per patient (tagged events)
4. **Stage 2**: Narrative events (clinical notes, imaging, pathology, NGS) are expanded into detailed documents via LLM; structured data events are converted to text via deterministic templates
5. All text blocks are interleaved chronologically and concatenated into `clinical_text`

### Structured Clinical Data Events

When `structured_data.enabled=True`, the event timeline includes four additional event types that simulate structured EHR/claims data converted to text:

| Event Type | Description | Text Template Output |
|------------|-------------|---------------------|
| `<encounter>` | Outpatient/ED visit with ICD-10 diagnosis codes and CPT/HCPCS procedure codes | Encounter Record with coded diagnoses and procedures |
| `<lab_result>` | CBC, CMP, tumor markers with values, units, and normal/abnormal flags | Laboratory Results with reference ranges |
| `<hospitalization>` | Hospital admission with principal diagnosis, LOS, discharge disposition | Hospital Admission Record |
| `<pro_assessment>` | EORTC QLQ-C30 subscale scores (0-100) and PRO-CTCAE symptom severity (0-4) | Patient-Reported Outcomes Assessment |

Structured events are generated by the LLM as part of the timeline (ensuring clinical coherence with the patient's trajectory), then converted to standardized text using deterministic templates in `structured_data.py` -- NOT expanded via the LLM note expansion prompt. The prompt includes reference schemas (common ICD-10 codes, CPT codes, lab reference ranges, PRO instrument definitions) so the LLM generates realistic values.

### Structured Data Config

```python
@dataclass
class StructuredDataConfig:
    enabled: bool = False
    include_encounters: bool = True       # ICD-10 + CPT encounter records
    include_hospitalizations: bool = True  # Hospital admission/discharge records
    include_labs: bool = True              # CBC, CMP, tumor markers
    include_pros: bool = True              # Patient-reported outcomes
    pro_instruments: List[str] = ["EORTC_QLQ_C30", "PRO_CTCAE"]
```

### CLI Usage

```bash
# Basic synthetic data generation (no structured data)
python -m synthetic_data.cli --use-vllm-batch --dataset-size 100

# With structured clinical data events
python -m synthetic_data.cli --use-vllm-batch --dataset-size 100 --structured-data

# Selective: only encounters and labs (no hospitalizations or PROs)
python -m synthetic_data.cli --use-vllm-batch --dataset-size 100 \
  --structured-data --no-hospitalizations --no-pros

# Via JSON config
python -m synthetic_data.cli --config my_config.json
```

### JSON Config Example

```json
{
  "clinical_question": "Compare pembrolizumab with nivolumab for advanced NSCLC",
  "dataset_size": 500,
  "generation_mode": "two_stage",
  "structured_data": {
    "enabled": true,
    "include_encounters": true,
    "include_labs": true,
    "include_hospitalizations": true,
    "include_pros": true,
    "pro_instruments": ["EORTC_QLQ_C30", "PRO_CTCAE"]
  }
}
```

### Structured Data Key Files

| File | Purpose |
|------|---------|
| `synthetic_data/config.py` | `StructuredDataConfig` dataclass |
| `synthetic_data/structured_data.py` | Parsing functions, template converters, reference data (lab ranges, PRO scales) |
| `synthetic_data/prompts.py` | `build_event_timeline_prompt()` conditionally adds structured event types and reference schemas |
| `synthetic_data/generator.py` | Interleaving logic in both HTTP API and vLLM batch paths |

## Key Files

| Purpose | Files |
|---------|-------|
| Main model | `oci/models/causal_text.py` |
| Causal forest model | `oci/models/causal_text_forest.py`, `oci/models/causal_forest_head.py` |
| Causal heads | `oci/models/dragonnet.py`, `oci/models/rlearner.py` |
| Feature extractors | `oci/models/frozen_llm_pooler_extractor.py`, `oci/models/hierarchical_llm_extractor.py`, `oci/models/hierarchical_cnn_extractor.py`, `oci/models/hierarchical_gru_extractor.py`, `oci/models/simple_cnn_extractor.py` |
| Extractor factory | `oci/models/extractor_factory.py` |
| Text chunking | `oci/models/text_chunking.py` |
| Learned tokenizer | `oci/models/learned_tokenizer.py` |
| Gated attention | `oci/models/gated_attention_pooling.py` |
| Hidden state cache | `oci/models/hidden_state_cache.py`, `oci/models/gpu_hidden_state_store.py` |
| Explicit features | `oci/extraction/explicit_confounders.py`, `oci/extraction/cache.py`, `oci/models/explicit_confounder_featurizer.py` |
| Propensity/Outcome models | `oci/models/propensity_model.py`, `oci/models/outcome_model.py` |
| Training | `oci/inference/applied.py`, `oci/inference/applied_forest.py`, `oci/inference/applied_tfidf_forest.py`, `oci/inference/applied_confounder_forest.py` |
| Semi-synthetic simulation | `oracle_experiment_scripts/semisynthetic_dgp.py`, `run_semisynthetic_experiments.py` |
| Config | `oci/config.py` |
| PSM | `oci/analysis/psm_analysis.py`, `oci/analysis/statistical_analysis.py`, `oci/matching/propensity_matcher.py` |
| Utilities | `oci/utils/io.py`, `oci/utils/system.py` |
| Data | `oci/data/dataset.py`, `oci/data/cached_hidden_state_dataset.py`, `oci/data/collators.py` |
| Synthetic data | `synthetic_data/generator.py`, `synthetic_data/config.py`, `synthetic_data/prompts.py`, `synthetic_data/structured_data.py` |

## Dependencies

**Core**: torch, transformers, pandas, numpy, scikit-learn, tqdm, pyarrow, joblib, accelerate, econml

**Optional**: openai (explicit feature extraction via vLLM server)

**Device support**: CUDA (NVIDIA GPUs), MPS (Apple Silicon M1/M2/M3), CPU

## Documentation Maintenance

**IMPORTANT**: When updating `CLAUDE.md`, always update `README.md` accordingly to keep user-facing documentation in sync. CLAUDE.md is the detailed developer reference; README.md is the user-facing overview.

## Adding a New Causal Head

When adding a new causal head type, update the following files:

| File | What to Update |
|------|----------------|
| `oci/models/new_head.py` | Create the new causal head module |
| `oci/models/__init__.py` | Add exports |
| `oci/config.py` | Add model_type validation and any new config options |
| `oci/models/causal_text.py` | Add import, instantiation case, train_step/predict logic |
| `oci/inference/applied.py` | Add any head-specific inference logic |

## Adding a New Feature Extractor

All extractors are created via `extractor_factory.py`. When adding a new extractor type, update the following files:

| File | What to Update |
|------|----------------|
| `oci/models/new_extractor.py` | Create the new extractor module (must implement `forward()`, `output_dim`, `get_state()`, `fit_tokenizer()` if needed) |
| `oci/models/__init__.py` | Add exports for new classes |
| `oci/config.py` | Add `normalize_feature_extractor_type()` entry, add to `VALID_EXTRACTOR_TYPES`, add config parameters with prefix |
| `oci/models/extractor_factory.py` | Add instantiation case in `create_feature_extractor()` |
| `oci/models/causal_text.py` | Add fit_tokenizer path if extractor requires it |
| `oci/inference/applied.py` | Add fit_tokenizer path if extractor requires it |
| `oci/training/propensity_trimming.py` | Uses extractor factory (no changes needed unless extractor has special requirements) |
| `oci/training/outcome_training.py` | Uses extractor factory (no changes needed unless extractor has special requirements) |
| `example_configs/` | Create example configuration file |
| `CLAUDE.md` | Update Feature Extractors table, add architecture section, update Key Files |

The factory pattern means `propensity_model.py`, `outcome_model.py`, and `causal_text_forest.py` automatically support new extractors without code changes -- they all delegate to `extractor_factory.create_feature_extractor()`.

## Quick Reference

- **ITE**: `preds['y1_prob'] - preds['y0_prob']` (probability scale for binary, raw values for continuous)
- **Outcome type**: `outcome_type="binary"` (BCE + sigmoid) or `"continuous"` (MSE, no sigmoid). Treatment always binary.
- **fit_tokenizer**: Required for `hierarchical_cnn`, `hierarchical_gru`, `simple_cnn`. Not needed for `frozen_llm_pooler`, `hierarchical_llm`.
- **Long docs**: Use `hierarchical_llm`, `hierarchical_cnn`, or `hierarchical_gru` (chunked). `frozen_llm_pooler` truncates at `flp_max_length`. `simple_cnn` truncates at `scnn_max_length`.
- **Interpretability**: `interpret_attention()`, `get_attention_weights()` (not available in cached mode for LLM extractors)
- **R-Learner vs DragonNet**: R-Learner for heterogeneous treatment effects; DragonNet for general use
- **TF-IDF Forest baseline**: `model_type="tfidf_forest"` -- no neural network, pure TF-IDF + CausalForestDML
- **Causal head dims**: `causal_head_representation_dim`, `causal_head_hidden_outcome_dim`, `causal_head_dropout` apply to all neural causal heads
