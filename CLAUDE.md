# CLAUDE.md - OCI (Oncology Causal Inference)

## Overview

OCI estimates treatment effects from clinical text by combining frozen LLM feature extraction with causal inference heads. It extracts confounders from unstructured EHR narratives to estimate individual (ITE) and average (ATE) treatment effects.

## Repository Structure

```
oci/
├── cli.py                 # CLI: `oci init`, `oci run`
├── config.py              # Dataclass configs
├── data/
│   ├── dataset.py                    # ClinicalTextDataset
│   ├── cached_hidden_state_dataset.py  # Dataset for pre-cached hidden states
│   └── collators.py                  # Collator utilities (returns None for frozen LLM)
├── experiments/runner.py  # Orchestrates inference & plasmode
├── extraction/
│   ├── explicit_confounders.py   # LLM-based confounder extraction via vLLM
│   └── cache.py                  # Extraction result caching
├── inference/
│   ├── applied.py             # Applied inference (CV or fixed split)
│   ├── applied_forest.py      # Causal forest inference pipeline
│   └── applied_tfidf_forest.py  # TF-IDF forest baseline pipeline
├── models/
│   ├── causal_text.py                # Main model (extractor + causal head)
│   ├── causal_text_forest.py         # Two-stage neural + causal forest model
│   ├── causal_forest_head.py         # CausalForestDML wrapper
│   ├── frozen_llm_pooler_extractor.py  # Frozen LLM + gated attention pooling
│   ├── gated_attention_pooling.py    # GatedAttentionPooling module
│   ├── hidden_state_cache.py         # Disk-based hidden state cache
│   ├── gpu_hidden_state_store.py     # GPU-resident hidden state store
│   ├── extractor_factory.py          # Factory for creating feature extractors
│   ├── dragonnet.py                  # DragonNet causal head
│   ├── rlearner.py                   # R-Learner causal head
│   ├── numeric_features.py           # Numeric value featurization (magnitude + type)
│   ├── explicit_confounder_featurizer.py  # MLP featurization of extracted confounders
│   ├── intra_batch_contrastive.py    # Intra-batch contrastive learning
│   ├── propensity_model.py           # Propensity-only model for trimming
│   └── outcome_model.py              # Outcome-only model for assessment
├── training/
│   ├── plasmode.py            # Plasmode simulation
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

### Feature Extractor

OCI uses a single feature extractor: **Frozen LLM Pooler**.

| Stage | Component | Description |
|-------|-----------|-------------|
| Tokenization | Pretrained HF tokenizer | Right-padded (all tokens used with mask) |
| Backbone | Decoder-only LLM (frozen, autocast float16) | All token hidden states from final layer |
| Downprojection | `nn.Linear(hidden_size, downprojection_dim)` (trainable, optional) | Reduces per-token dim before pooling |
| Pooling | GatedAttentionPooling | tanh x sigmoid gating + softmax attention |
| Projection | 2-layer MLP | Linear->LN->GELU->Dropout->Linear->LN |

The frozen LLM processes full documents (no chunking) up to the configured `flp_max_length`. No `fit_tokenizer()` is required -- it uses the pretrained HuggingFace tokenizer.

### Causal Heads

| Type | Description | Key output |
|------|-------------|------------|
| `dragonnet` | Propensity + Y0/Y1 potential outcomes | ITE = sigma(y1) - sigma(y0) |
| `rlearner` | Direct tau(X) optimization, detached nuisance functions | tau directly predicts ITE |
| `causal_forest` | Two-stage: neural features + econml CausalForestDML | tau with confidence intervals |
| `tfidf_forest` | TF-IDF features + econml CausalForestDML (no neural network) | tau with confidence intervals |

**R-Learner advantage**: Nuisance functions (e, m) are detached in R-loss, providing stronger gradient signal for treatment effect modifiers.

### R-Learner Dual Extractor Mode

When `rlearner_dual_extractors=True`, the R-Learner uses two independent frozen LLM pooler extractors:

| Component | Purpose | Training Signal |
|-----------|---------|-----------------|
| Nuisance Extractor | e(X), m(X) | Propensity BCE + Outcome BCE |
| Effect Extractor | tau(X) | R-learner loss only |

This separation prevents gradient interference between confounder learning (nuisance) and effect modifier learning (tau). The effect extractor learns representations optimized specifically for treatment effect heterogeneity.

**Memory Note**: Dual mode approximately doubles feature extraction memory/compute.

**Config:**
```json
{
  "architecture": {
    "model_type": "rlearner",
    "feature_extractor_type": "frozen_llm_pooler",
    "rlearner_dual_extractors": true
  }
}
```

## CLI

```bash
oci init --output config.json
oci run --config config.json --device cuda:0 --workers 4 [--skip-plasmode] [--skip-pretraining] [--verbose]

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

## Numeric Feature Extraction

Clinical text contains numbers critical for causal inference (lab values, vitals, scores, doses, ages) that receive no special treatment from standard tokenizers. The numeric features module adds magnitude-aware numeric featurization as a parallel channel.

### How It Works

1. **Regex extraction**: Detects integers, decimals, and fractions (e.g., BP 120/80) in raw text
2. **Log-scale magnitude binning**: Maps values into 8 bins: `[0, 0.1, 1, 10, 100, 1000, 10000, 100000]`
3. **Context-based type detection**: Classifies numbers by preceding keywords into 10 categories (vitals, labs, scores, demographics, doses, etc.)
4. **Document-level injection**: Aggregate histogram (`NumericFeatureVector`) merged before output projection

### Config Parameters

| Param | Description | Default |
|-------|-------------|---------|
| `numeric_features_enabled` | Enable numeric feature extraction | `False` |
| `numeric_embedding_dim` | Output dimension of numeric feature vectors | `32` |
| `numeric_magnitude_bins` | Number of log-scale magnitude bins | `8` |
| `numeric_type_categories` | Number of numeric type categories | `10` |

When `numeric_features_enabled` is `False` (default), there is no behavior change.

## Explicit Confounder Extraction

Researchers can specify explicit confounder variables to be extracted from clinical text using an LLM (via vLLM). The extracted confounders are featurized and concatenated to text embeddings before the causal heads.

### How It Works

```
1. Config specifies explicit confounders (name, type, categories)
2. vLLM extracts confounders from clinical text (preprocessing step)
3. Generates structured values per patient with missingness flags
4. ExplicitConfounderFeaturizer MLP encodes confounders
5. Concatenated to frozen LLM pooler output
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

For **neural models** (DragonNet, R-Learner):
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
- Cache location: `{dataset_dir}/.oci_cache/extraction_{hash}.parquet`
- Invalidated automatically if config changes

## Causal Forest Mode

When `model_type="causal_forest"`, OCI uses a two-stage approach combining neural feature extraction with econml's CausalForestDML for treatment effect estimation.

### Architecture

```
Stage 1: Representation Learning (Neural Network)
+-- Frozen LLM Pooler -> Features
+-- Propensity Head: P(T=1|X) -> BCE loss
+-- Outcome Head: E[Y|X] -> BCE loss
+-- [Optional] Effect Head: tau(X) -> R-loss (when use_rlearner_representation=True)

Stage 1 with Dual Extractors (when rlearner_dual_extractors=True):
+-- Nuisance Extractor (feature_extractor)
|   +-- Text -> Features_nuisance
|   +-- Propensity Head -> e(X) [BCE loss]
|   +-- Outcome Head -> m(X) [BCE loss]
+-- Effect Extractor (effect_feature_extractor)
    +-- Text -> Features_effect -> effect_mlp -> tau(X) [R-loss]

Stage 2: Effect Estimation (Causal Forest)
+-- Extract learned representations from Stage 1
|   (In dual mode: uses Effect Extractor features, optimized for tau)
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
| `use_rlearner_representation` | Add tau head and R-loss to Stage 1 training | `False` |
| `gamma_rlearner` | Weight for R-learner loss during representation training | `1.0` |
| `rlearner_dual_extractors` | Use separate extractors for nuisance vs effect | `False` |

### R-Learner Representation Training

When `use_rlearner_representation=True`, Stage 1 adds a treatment effect head (tau) and trains with the R-learner loss in addition to propensity and outcome losses.

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

## Intra-Batch Contrastive Learning

Supervised contrastive loss (SupCon, Khosla et al. 2020) within similarity clusters improves confounder detection by encouraging the model to learn representations that discriminate treatment/outcome status among otherwise similar patients.

### How It Works

```
Text -> Extractor -> Z (features) -> [K-means on detached Z] -> Within-cluster SupCon Loss
                                   -> Causal Head -> Standard Losses

Total Loss = Standard Loss + contrastive_weight x SupCon Loss
```

1. **Feature projection**: 2-layer MLP projects features to a contrastive space
2. **Clustering**: K-means on detached features groups similar patients
3. **Label construction**: Treatment x outcome creates 4-class labels (joint mode)
4. **SupCon within clusters**: Contrastive loss computed independently per cluster, averaged

### Why Cluster-Then-Contrast?

Global SupCon would push ALL treated patients' representations together, destroying heterogeneity. Intra-cluster contrastive learning targets exactly the subtle confounders: "among clinically similar patients, the model should still distinguish treatment/outcome status."

### Config Parameters

**Architecture config** (`ModelArchitectureConfig`):

| Param | Description | Default |
|-------|-------------|---------|
| `contrastive_enabled` | Enable contrastive learning | `False` |
| `contrastive_num_clusters` | Number of K-means clusters (K) | `4` |
| `contrastive_temperature` | SupCon temperature (lower = sharper) | `0.1` |
| `contrastive_label_mode` | Label construction: "treatment", "outcome", or "joint" | `"joint"` |
| `contrastive_projection_dim` | Projection head output dimension | `64` |
| `contrastive_min_cluster_size` | Minimum samples per cluster | `2` |
| `contrastive_clustering_method` | "kmeans" or "random" | `"kmeans"` |

**Training config** (`TrainingConfig`):

| Param | Description | Default |
|-------|-------------|---------|
| `contrastive_weight` | Weight for contrastive loss in total loss | `0.1` |

In dual extractor mode (R-Learner), contrastive loss targets the **nuisance extractor** features. The effect extractor is not affected.

### Edge Cases

Graceful degradation (contrastive loss = 0, standard losses carry training):
- Batch too small (< 4 samples)
- All-same-label clusters (no negative pairs)
- No valid clusters in batch

## Training Options for tau Learning

| Option | Effect |
|--------|--------|
| `stop_grad_propensity=True` | Prevents propensity from dominating representation |
| `attention_entropy_weight>0` | Encourages focused attention (low entropy) |
| `gamma_rlearner>1.0` | Stronger treatment effect signal |
| `numeric_features_enabled=True` | Adds magnitude-aware numeric featurization from clinical text |
| `rlearner_dual_extractors=True` | Uses separate extractors for nuisance (e,m) and effect (tau) |
| `contrastive_enabled=True` | Enables intra-batch contrastive learning for confounder detection |
| `contrastive_weight>0` | Weight for contrastive loss term (default 0.1) |

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
2. **Plasmode Simulation**: Synthetic outcomes with known ATE for validation
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
+-- plasmode_experiments/       # If enabled
```

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
| Feature extractor | `oci/models/frozen_llm_pooler_extractor.py` |
| Extractor factory | `oci/models/extractor_factory.py` |
| Gated attention | `oci/models/gated_attention_pooling.py` |
| Hidden state cache | `oci/models/hidden_state_cache.py`, `oci/models/gpu_hidden_state_store.py` |
| Numeric features | `oci/models/numeric_features.py` |
| Contrastive learning | `oci/models/intra_batch_contrastive.py` |
| Explicit confounders | `oci/extraction/explicit_confounders.py`, `oci/extraction/cache.py`, `oci/models/explicit_confounder_featurizer.py` |
| Propensity/Outcome models | `oci/models/propensity_model.py`, `oci/models/outcome_model.py` |
| Training | `oci/inference/applied.py`, `oci/inference/applied_forest.py`, `oci/inference/applied_tfidf_forest.py` |
| Plasmode | `oci/training/plasmode.py` |
| Config | `oci/config.py` |
| PSM | `oci/analysis/psm_analysis.py`, `oci/analysis/statistical_analysis.py`, `oci/matching/propensity_matcher.py` |
| Utilities | `oci/utils/io.py`, `oci/utils/system.py` |
| Data | `oci/data/dataset.py`, `oci/data/cached_hidden_state_dataset.py`, `oci/data/collators.py` |
| Synthetic data | `synthetic_data/generator.py`, `synthetic_data/config.py`, `synthetic_data/prompts.py`, `synthetic_data/structured_data.py` |

## Dependencies

**Core**: torch, transformers, pandas, numpy, scikit-learn, tqdm, pyarrow, joblib, accelerate, econml

**Optional**: openai (explicit confounder extraction via vLLM server)

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
| `oci/training/plasmode.py` | Add any head-specific plasmode logic |

## Modifying the Feature Extractor

Since there is only one extractor (`frozen_llm_pooler`), changes are centralized:

| File | Purpose |
|------|---------|
| `oci/models/frozen_llm_pooler_extractor.py` | Core extractor implementation |
| `oci/models/extractor_factory.py` | Factory function (delegates to FrozenLLMPoolerExtractor) |
| `oci/config.py` | `flp_*` config parameters in `ModelArchitectureConfig` |
| `oci/models/causal_text.py` | Extractor instantiation and usage |
| `oci/models/propensity_model.py` | Uses extractor factory for propensity-only models |
| `oci/models/outcome_model.py` | Uses extractor factory for outcome-only models |

## Quick Reference

- **ITE**: `preds['y1_prob'] - preds['y0_prob']` (probability scale for binary, raw values for continuous)
- **Outcome type**: `outcome_type="binary"` (BCE + sigmoid) or `"continuous"` (MSE, no sigmoid). Treatment always binary.
- **No fit_tokenizer**: Uses pretrained HF tokenizer from the frozen LLM
- **Interpretability**: `interpret_attention()`, `get_attention_weights()` (not available in cached mode)
- **R-Learner vs DragonNet**: R-Learner for heterogeneous treatment effects; DragonNet for general use
- **TF-IDF Forest baseline**: `model_type="tfidf_forest"` -- no neural network, pure TF-IDF + CausalForestDML
- **Causal head dims**: `causal_head_representation_dim`, `causal_head_hidden_outcome_dim`, `causal_head_dropout` apply to all neural causal heads
