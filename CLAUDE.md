# CLAUDE.md - CDT (Causal DragonNet Text)

## Project Overview

CDT is a framework for **clinical causal inference using electronic health record (EHR) text**. It estimates treatment effects from unstructured clinical narratives by combining text feature extraction with DragonNet causal inference heads.

**Key purpose**: Extract confounders from clinical text to estimate individual treatment effects (ITEs) and average treatment effects (ATEs) in observational studies where critical confounders exist only in unstructured notes.

## Repository Structure

```
cdt/                          # Main package
├── cli.py                    # CLI entry point (`cdt init`, `cdt run`)
├── config.py                 # Dataclass configurations (ExperimentConfig, ModelArchitectureConfig, etc.)
├── data/
│   └── dataset.py            # ClinicalTextDataset, data loading/validation
├── experiments/
│   └── runner.py             # ExperimentRunner - orchestrates applied inference & plasmode
├── inference/
│   └── applied.py            # Run applied causal inference (CV or fixed split)
├── models/
│   ├── causal_text.py        # CausalText - main model combining extractor + DragonNet
│   ├── cnn_extractor.py      # CNNFeatureExtractor - 1D CNN with semantic filter init
│   ├── bert_extractor.py     # BertFeatureExtractor - HuggingFace transformer CLS token
│   ├── gru_extractor.py      # GRUFeatureExtractor - BiGRU with attention pooling
│   ├── confounder_extractor.py # ConfounderExtractor, HierarchicalConfounderExtractor, GRUHierarchicalConfounderExtractor
│   ├── hierarchical_transformer_extractor.py # HierarchicalTransformerExtractor - sentence BERT + transformer pooling
│   ├── dragonnet.py          # DragonNet head (propensity + potential outcomes)
│   ├── uplift.py             # UpliftNet head (alternative parametrization)
│   ├── rlearner.py           # RLearnerNet head (direct tau optimization)
│   ├── outcome_heads.py      # Outcome prediction heads
│   └── propensity_model.py   # Standalone propensity model
├── training/
│   ├── plasmode.py           # Plasmode simulation experiments
│   ├── propensity_trimming.py# Propensity-based dataset trimming
│   └── outcome_training.py   # Standalone outcome model training
├── matching/                 # Propensity score matching
│   ├── propensity_matcher.py # PropensityMatcher, MatchResult, balance utilities
│   └── __init__.py           # Exports matching classes and functions
├── analysis/                 # Statistical analysis for causal inference
│   ├── statistical_analysis.py # ATT/ATE estimation, McNemar's test, Rosenbaum bounds
│   ├── psm_analysis.py       # run_psm_analysis() - main PSM workflow
│   └── __init__.py           # Exports analysis functions
└── utils/                    # Utilities (io, system, etc.)

synthetic_data/               # LLM-based synthetic data generation
├── generator.py              # Main pipeline for generating synthetic datasets
├── config.py                 # SyntheticDataConfig
├── prompts.py                # Clinical prompts for LLM
├── llm_client.py             # OpenAI API client
└── vllm_batch_client.py      # vLLM batch inference client

examples/                     # Example config files
├── semantic_cnn_config.json  # CNN with explicit clinical concepts
├── bert_config.json          # BERT feature extractor config
├── modernbert_config.json    # ModernBERT config
├── rlearner_config.json      # R-Learner with direct tau optimization
├── confounder_config.json    # Confounder extractor with sparse cross-attention
├── gru_confounder_config.json # GRU-based confounder extractor (learns from scratch)
└── hierarchical_transformer_config.json # Hierarchical transformer (sentence BERT + pooling)
```

## Architecture

### Core Model: CausalText (`cdt/models/causal_text.py`)

The main model combines:
1. **Feature Extractor** (one of five types):
   - `cnn`: 1D CNN with word-level tokenization (default, fastest)
   - `bert`: HuggingFace transformer (Bio_ClinicalBERT, ModernBERT, etc.)
   - `gru`: Bidirectional GRU with attention (O(N) for long sequences)
   - `confounder`: Perceiver-style cross-attention with sparse attention for long documents
   - `hierarchical_transformer`: Sentence-level BERT + transformer pooling (simple, effective for long docs)

2. **Causal Inference Head** (one of three types):
   - `dragonnet`: Classic DragonNet (propensity + Y0/Y1 potential outcomes)
   - `uplift`: UpliftNet (base outcome + treatment effect parametrization)
   - `rlearner`: R-Learner (direct tau optimization with detached nuisance functions)

### CNN Feature Extractor (`cdt/models/cnn_extractor.py`)

Key features:
- **Semantic filter initialization**: CNN filters can be initialized from explicit clinical concepts (e.g., "stage iv cancer", "performance status poor")
- **K-means filter initialization**: Additional filters derived from clustering training n-grams
- **BERT embedding initialization**: Word embeddings initialized from Bio_ClinicalBERT
- **Filter interpretability**: `interpret_filters()` method shows which n-grams activate each filter

**Important**: Call `fit_tokenizer(texts)` before training with CNN/GRU extractors.

### DragonNet (`cdt/models/dragonnet.py`)

Outputs:
- `y0_logit`: Predicted outcome under control (T=0)
- `y1_logit`: Predicted outcome under treatment (T=1)
- `t_logit`: Treatment propensity logit
- Individual Treatment Effect (ITE) = sigmoid(y1_logit) - sigmoid(y0_logit)

Loss function components:
- Outcome loss (factual only)
- Propensity loss (binary cross-entropy)
- Targeted regularization (R-loss)

### RLearnerNet (`cdt/models/rlearner.py`)

R-Learner architecture for direct treatment effect optimization. Uses three heads:
- `e(X)`: Propensity head - P(T=1|X)
- `m(X)`: Marginal outcome head - E[Y|X]
- `tau(X)`: Treatment effect head - E[Y(1)-Y(0)|X] (unbounded, can be negative)

**Key advantage**: The nuisance functions (e, m) are **detached** in the R-loss, so gradients flow directly through tau(X). This provides stronger gradient signal for learning treatment effect modifiers from text.

Loss function:
```
L = L_outcome(m) + alpha * L_propensity(e) + gamma * L_rlearner(tau)

L_rlearner = E[((Y - m(X)) - tau(X) * (T - e(X)))^2]  # e, m detached
```

Outputs (predict method):
- `tau_pred`: Direct treatment effect tau(X)
- `m_prob`: Marginal outcome probability E[Y|X]
- `y0_prob`, `y1_prob`: Derived for backward compatibility

Reference: Nie & Wager (2021). Quasi-oracle estimation of heterogeneous treatment effects. Biometrika.

### ConfounderExtractor (`cdt/models/confounder_extractor.py`)

Perceiver-style feature extractor designed for extracting confounder signals from long clinical text. Uses sentence-level processing with sparse cross-attention.

**Standard Architecture (sentence-level):**
```
Long Clinical Text
        ↓
Split into Sentences (S sentences)
        ↓
Sentence Encoder (SentenceTransformer, e.g., all-MiniLM-L6-v2)
        ↓
Sentence Embeddings (S × d)
        ↓
Latent Queries (K learnable confounder vectors)
        ↓
Sparse Cross-Attention (entmax/top-k) with Iterative Refinement
        ↓
K Latent Representations (K × d)
        ↓
MLP Projection → Causal Head (DragonNet/RLearner)
```

**Hierarchical Architecture (token-level):**
With `confounder_hierarchical=True`, preserves fine-grained token signal:
```
Long Clinical Text
        ↓
Split into Sentences (S sentences)
        ↓
Encode EACH sentence with BERT → S × (L_s tokens × 768)
        ↓
Mean-pool each sentence → Sentence Embeddings (S × 768)
        ↓
Sentence-Level Sparse Attention (entmax) → Sentence Weights (K × S)
        ↓
Token-Level Cross-Attention (within each sentence, gated by sentence weights)
        ↓
K Confounder Representations (K × D)
        ↓
Task-Specific Multi-Head Aggregation (3-way):
  - Propensity Query → weighted sum → propensity_repr (D,)
  - For DragonNet:
    - Y0 Query → weighted sum → y0_repr (D,)
    - Y1 Query → weighted sum → y1_repr (D,)
  - For R-Learner:
    - Outcome Query → weighted sum → outcome_repr (D,)
    - Tau Query → weighted sum → tau_repr (D,)
        ↓
Concatenate: (3*D,) → MLP → Causal Head
```

**Key features:**
- **Sparse attention** via entmax (forces exact zeros on irrelevant sentences)
- **Iterative refinement**: Multiple cross-attention passes for progressive focusing
- **Explicit confounder initialization**: Optional concept phrases (e.g., "metastatic sites")
- **Self-attention between latents**: Allows confounders to share information
- **Attention visualization**: `interpret_attention()` method shows top-attended sentences
- **Hierarchical mode**: Preserves token-level distinctions (e.g., "ECOG PS 0" vs "ECOG PS 2")
- **3-way task-specific aggregation**: Different learnable queries for each prediction head (propensity + Y0/Y1 for DragonNet, propensity + outcome/tau for R-Learner) allow distinct confounder weighting per task

**Key configuration:**
```python
confounder_num_latents: int = 4        # Learnable latent vectors
confounder_explicit_texts: List[str]   # Explicit concept phrases
confounder_num_iterations: int = 2     # Refinement passes
confounder_sparse_attention: bool = True
confounder_sparse_alpha: float = 1.5   # 1.5=entmax15, 2.0=sparsemax
confounder_sparse_method: str = "entmax"  # "entmax", "topk", "softmax"

# Hierarchical mode (BERT-based token-level attention)
confounder_hierarchical: bool = False   # Enable token-level attention
confounder_token_encoder: str = "distilbert-base-uncased"  # BERT for token encoding
confounder_freeze_token_encoder: bool = True
confounder_max_sentence_tokens: int = 128

# GRU-based mode (learns from scratch)
confounder_use_gru: bool = False          # Enable GRU-based extraction
confounder_gru_embedding_dim: int = 128   # Word embedding dimension
confounder_gru_hidden_dim: int = 128      # GRU hidden state per direction
confounder_gru_num_layers: int = 1        # Stacked GRU layers
confounder_gru_bidirectional: bool = True # Bidirectional GRU
confounder_gru_dropout: float = 0.1
confounder_gru_max_vocab: int = 50000
confounder_gru_min_word_freq: int = 2
confounder_gru_max_sentence_length: int = 128
```

**GRU-based Hierarchical Architecture (learns from scratch):**
With `confounder_use_gru=True`, all parameters learn together via the causal objective:
```
Long Clinical Text
        ↓
Split into Sentences (S sentences)
        ↓
Tokenize each sentence (word-level vocabulary)
        ↓
Embed tokens with learnable embeddings
        ↓
Encode each sentence with BiGRU + attention pooling → (S × encoder_dim)
        ↓
Latent Queries (K learnable confounder vectors)
        ↓
Sentence-Level Sparse Attention (entmax) → Sentence Weights (K × S)
        ↓
Token-Level Cross-Attention (within each sentence, gated by sentence weights)
        ↓
K Confounder Representations (K × D)
        ↓
Task-Specific Multi-Head Aggregation (3-way):
  - Propensity Query → weighted sum → propensity_repr (D,)
  - For DragonNet:
    - Y0 Query → weighted sum → y0_repr (D,)
    - Y1 Query → weighted sum → y1_repr (D,)
  - For R-Learner:
    - Outcome Query → weighted sum → outcome_repr (D,)
    - Tau Query → weighted sum → tau_repr (D,)
        ↓
Concatenate: (3*D,) → MLP → Causal Head
```

**Key advantages of GRU mode:**
- All parameters (embeddings, GRU, attention, latent confounders, task queries) optimized together
- No domain mismatch from pretrained encoder
- Lighter weight than BERT-based approaches
- Better adaptation to specific clinical vocabulary
- 3-way task-specific aggregation reduces dimensionality from K*D to 3*D (e.g., 8*256=2048 → 768)
- DragonNet can learn different confounder weights for Y0 vs Y1 (treatment effect modifiers)
- R-Learner can learn which confounders are prognostic (m) vs effect-modifying (τ)
- **Important**: Requires `fit_tokenizer(texts)` before training (like CNN/GRU extractors)

**Why this helps for long documents:**
- Sentence-level attention reduces search space from 2048 tokens to ~50-100 sentences
- Sparse attention forces each latent to focus on few relevant sentences
- Iterative refinement allows progressive "zooming in" on confounder mentions
- Hierarchical mode preserves fine-grained signal that sentence embeddings may lose
- Works with standard causal loss - no concept labels needed

### HierarchicalTransformerExtractor (`cdt/models/hierarchical_transformer_extractor.py`)

Simple hierarchical feature extractor using sentence-level BERT encoding + transformer pooling.

**Architecture:**
```
Long Clinical Text
        ↓
Split into Sentences (S sentences)
        ↓
Tiny BERT per Sentence → [CLS] token (S × hidden_dim)
        ↓
Transformer Layer(s) with learnable [POOL] token
        ↓
Final Representation (D,) → DragonNet/RLearner
```

**Key features:**
- **Simple design**: No latent confounders or sparse attention - just straightforward hierarchical encoding
- **Sentence encoder**: Uses lightweight BERT models (e.g., `prajjwal1/bert-tiny` with 4.4M params)
- **Transformer pooling**: Learnable [POOL] token aggregates sentence embeddings via self-attention
- **Interpretability**: `interpret_attention()` method shows which sentences the [POOL] token attends to
- **No fit_tokenizer needed**: Uses pretrained BERT tokenizer

**Key configuration:**
```python
hier_transformer_sentence_model: str = "prajjwal1/bert-tiny"  # Sentence encoder
hier_transformer_freeze_sentence_encoder: bool = True  # Freeze encoder weights
hier_transformer_max_sentences: int = 100  # Max sentences per document
hier_transformer_max_sentence_length: int = 128  # Max tokens per sentence
hier_transformer_num_layers: int = 2  # Transformer pooling layers
hier_transformer_num_heads: int = 4  # Attention heads
hier_transformer_dim: int = 256  # Transformer hidden dimension
hier_transformer_dropout: float = 0.1
hier_transformer_projection_dim: int = 128  # Final output dimension
```

**When to use:**
- Long documents where sentence-level processing is appropriate
- When you want a simpler alternative to ConfounderExtractor
- When interpretability via sentence attention is desired
- When you don't need explicit confounder queries

### Sparse Attention Utilities (`cdt/models/sparse_attention.py`)

Provides sparse attention mechanisms that produce exact zeros for irrelevant positions:
- `sparse_softmax()`: Unified interface for entmax/sparsemax with fallback implementations
- `top_k_attention()`: Hard top-k selection with straight-through gradients
- `SparseCrossAttention`: Multi-head cross-attention with configurable sparsity

## Key Commands

```bash
# Initialize default config
cdt init --output config.json

# Run experiment
cdt run --config config.json --device cuda:0 --workers 4

# CLI options
cdt run --config config.json \
    --device cuda:1 \
    --workers 4 \
    --output-dir ./results \
    --skip-plasmode \
    --verbose
```

## Configuration (`cdt/config.py`)

Main config classes:
- `ExperimentConfig`: Top-level config
- `AppliedInferenceConfig`: Dataset paths, column names, CV folds
- `ModelArchitectureConfig`: Feature extractor type, CNN/BERT/GRU params, DragonNet dims
- `TrainingConfig`: Learning rate, epochs, batch size, loss weights (alpha_propensity, beta_targreg, gamma_rlearner)
- `PropensityTrimmingConfig`: Pre-trimming by propensity scores
- `MatchingAnalysisConfig`: Post-hoc PSM analysis using DragonNet's propensity scores
- `PlasmodeConfig`: Plasmode simulation parameters

Key architecture settings:
```python
# Feature extractor type
feature_extractor_type: str = "cnn"  # "cnn", "bert", or "gru"

# CNN-specific
cnn_embedding_dim: int = 128
cnn_kernel_sizes: List[int] = [3, 4, 5, 7]
cnn_explicit_filter_concepts: Dict[str, List[str]]  # kernel_size -> concepts
cnn_num_kmeans_filters: int = 64
cnn_init_embeddings_from: str = "emilyalsentzer/Bio_ClinicalBERT"

# BERT-specific
bert_model_name: str = "bert-base-uncased"
bert_max_length: int = 512
bert_freeze_encoder: bool = False

# GRU-specific
gru_hidden_dim: int = 256
gru_num_layers: int = 2
gru_max_length: int = 8192  # Efficient for long sequences
```

## Dataset Format

Expected columns in Parquet/CSV:
| Column | Type | Description |
|--------|------|-------------|
| `clinical_text` | string | Clinical narrative text |
| `treatment_indicator` | int/float | Binary treatment (0 or 1) |
| `outcome_indicator` | int/float | Binary outcome (0 or 1) |
| `split` | string | Optional: "train", "val", "test" |

## Workflow Modes

### 1. Applied Inference
Estimates treatment effects on real data:
- K-fold cross-validation (default: 5 folds)
- Fixed train/val/test splits
- Outputs: `predictions.parquet` with `pred_ite_prob`, `pred_propensity_prob`, etc.

### 2. Plasmode Simulation
Generates synthetic outcomes with known ground truth for method validation:
1. Train "generator" model on real data
2. Generate synthetic outcomes with known ATE
3. Train "evaluator" on synthetic data
4. Compare estimated vs. true treatment effects

### 3. Propensity Score Matching Analysis
Post-hoc traditional PSM analysis using DragonNet's learned propensity scores:
- ATT estimation from matched pairs with bootstrap CIs
- ATE via IPW (inverse probability weighting)
- ATE via stratification (propensity subclassification)
- Balance diagnostics (SMD before/after matching)
- Rosenbaum sensitivity analysis for hidden bias

## Matching Module (`cdt/matching/`)

**PropensityMatcher** - Main matching class:
```python
from cdt.matching import PropensityMatcher, assess_overlap

matcher = PropensityMatcher(
    method='nearest',      # 'nearest', 'optimal', or 'caliper'
    caliper=0.2,           # Max distance for valid match
    caliper_scale='std',   # 'propensity', 'logit', or 'std'
    ratio=1,               # 1:k matching
    replacement=False
)

match_result = matcher.match(propensity_scores, treatment)
# match_result.matched_pairs: (n_matches, 2) array of indices
# match_result.distances: distance for each pair
```

**Helper functions**:
- `compute_balance_statistics(covariates, treatment, match_result)` - SMD before/after
- `assess_overlap(propensity_scores, treatment)` - Overlap coefficient, common support

## Analysis Module (`cdt/analysis/`)

**Treatment effect estimation**:
```python
from cdt.analysis import (
    estimate_att_matched,    # ATT from matched pairs
    estimate_ate_ipw,        # ATE via inverse probability weighting
    estimate_ate_stratified, # ATE via propensity stratification
    run_psm_analysis         # Full PSM analysis workflow
)

# Run full PSM analysis
from cdt.config import MatchingAnalysisConfig
results = run_psm_analysis(
    predictions_df,          # Must have propensity_pred, treatment, outcome columns
    config=MatchingAnalysisConfig(),
    output_dir=Path('./psm_results')
)
# Returns: match_result, overlap, balance_stats, att_matched, ate_ipw, etc.
```

**Statistical tests**:
- `mcnemars_test(outcomes, match_result)` - For binary outcomes
- `paired_t_test(outcomes, match_result)` - For continuous outcomes
- `sensitivity_analysis_rosenbaum(outcomes, match_result)` - Rosenbaum bounds

## Synthetic Data Generation (`synthetic_data/`)

LLM-based pipeline for generating synthetic clinical datasets:
1. Generate confounders from clinical question
2. Generate treatment/outcome regression equations
3. Generate summary statistics
4. Sample patient characteristics and generate clinical histories

Supports both OpenAI API and local vLLM batch inference.

## Key Files for Development

- **Main model**: `cdt/models/causal_text.py` (CausalText)
- **Causal heads**: `cdt/models/dragonnet.py`, `cdt/models/uplift.py`, `cdt/models/rlearner.py`
- **Feature extractors**: `cdt/models/cnn_extractor.py`, `cdt/models/bert_extractor.py`, `cdt/models/gru_extractor.py`, `cdt/models/confounder_extractor.py` (ConfounderExtractor, HierarchicalConfounderExtractor, GRUHierarchicalConfounderExtractor), `cdt/models/hierarchical_transformer_extractor.py` (HierarchicalTransformerExtractor)
- **Sparse attention**: `cdt/models/sparse_attention.py` (entmax, top-k, SparseCrossAttention)
- **Training loop**: `cdt/inference/applied.py` (_train_single_model, _train_epoch)
- **Plasmode**: `cdt/training/plasmode.py` (plasmode simulation experiments)
- **Config**: `cdt/config.py`
- **CLI**: `cdt/cli.py`
- **Dataset**: `cdt/data/dataset.py`
- **PSM Analysis**: `cdt/analysis/psm_analysis.py` (run_psm_analysis)
- **Matching**: `cdt/matching/propensity_matcher.py` (PropensityMatcher, MatchResult)
- **Statistical Tests**: `cdt/analysis/statistical_analysis.py` (ATT, ATE, McNemar's, Rosenbaum)

## Common Patterns

### Training a model manually
```python
from cdt.models import CausalText

model = CausalText(
    feature_extractor_type="cnn",
    embedding_dim=128,
    kernel_sizes=[3, 4, 5, 7],
    num_kmeans_filters=64,
    device="cuda:0"
)

# IMPORTANT: Fit tokenizer before training
model.fit_tokenizer(train_texts)

# Optional: Initialize embeddings from BERT
model.feature_extractor.init_embeddings_from_bert("emilyalsentzer/Bio_ClinicalBERT")

# Optional: Initialize semantic filters
model.feature_extractor.init_filters(train_texts)

# Training loop
for batch in dataloader:
    losses = model.train_step(batch, alpha_propensity=1.0, beta_targreg=0.1)
    losses['loss'].backward()
    optimizer.step()
```

### Training with R-Learner
```python
from cdt.models import CausalText

# Create model with R-Learner architecture
model = CausalText(
    feature_extractor_type="cnn",
    model_type="rlearner",  # Use R-Learner instead of DragonNet
    embedding_dim=128,
    kernel_sizes=[3, 4, 5, 7],
    num_kmeans_filters=64,
    device="cuda:0"
)

model.fit_tokenizer(train_texts)

# Training loop with gamma_rlearner
for batch in dataloader:
    losses = model.train_step(
        batch,
        alpha_propensity=1.0,
        gamma_rlearner=1.0  # Weight for R-learner loss
    )
    losses['loss'].backward()
    optimizer.step()

# R-learner returns additional loss component
print(f"R-loss: {losses['r_loss']}")
```

### Training with ConfounderExtractor
```python
from cdt.models import CausalText

# Create model with ConfounderExtractor for long documents
model = CausalText(
    feature_extractor_type="confounder",
    model_type="rlearner",
    # Confounder extractor settings
    confounder_num_latents=4,
    confounder_explicit_texts=[
        "metastatic disease",
        "performance status",
        "prior treatment"
    ],
    confounder_value_dim=128,
    confounder_num_iterations=2,
    confounder_sparse_attention=True,
    confounder_sparse_alpha=1.5,  # entmax15
    device="cuda:0"
)

# No fit_tokenizer needed - uses pretrained sentence encoder
model.fit_tokenizer(train_texts)  # No-op for confounder extractor

# Training loop
for batch in dataloader:
    losses = model.train_step(
        batch,
        alpha_propensity=1.0,
        gamma_rlearner=1.0
    )
    losses['loss'].backward()
    optimizer.step()

# Interpret attention weights (which sentences each confounder attends to)
interpretations = model.feature_extractor.interpret_attention(texts, top_k=5)
for doc_idx, doc_interp in enumerate(interpretations):
    print(f"Document {doc_idx}:")
    for conf_name, attended in doc_interp.items():
        print(f"  {conf_name}: {[a['sentence'][:50] for a in attended]}")
```

### Training with Hierarchical ConfounderExtractor
```python
from cdt.models import CausalText

# Hierarchical mode preserves token-level signal (e.g., "PS 0" vs "PS 2")
model = CausalText(
    feature_extractor_type="confounder",
    model_type="rlearner",
    # Enable hierarchical mode
    confounder_hierarchical=True,
    confounder_token_encoder="distilbert-base-uncased",  # or "emilyalsentzer/Bio_ClinicalBERT"
    confounder_freeze_token_encoder=True,
    confounder_max_sentence_tokens=128,
    # Other settings
    confounder_num_latents=4,
    confounder_explicit_texts=["metastatic disease", "performance status"],
    confounder_value_dim=128,
    confounder_sparse_attention=True,
    confounder_sparse_alpha=1.5,
    device="cuda:0"
)

# Training is the same
model.fit_tokenizer(train_texts)  # No-op
for batch in dataloader:
    losses = model.train_step(batch, alpha_propensity=1.0, gamma_rlearner=1.0)
    losses['loss'].backward()
    optimizer.step()
```

### Training with GRU Hierarchical ConfounderExtractor
```python
from cdt.models import CausalText

# GRU mode: learns entirely from scratch via causal objective
model = CausalText(
    feature_extractor_type="confounder",
    model_type="rlearner",
    # Enable GRU-based mode (learns from scratch)
    confounder_use_gru=True,
    confounder_gru_embedding_dim=128,
    confounder_gru_hidden_dim=128,
    confounder_gru_num_layers=1,
    confounder_gru_bidirectional=True,
    confounder_gru_min_word_freq=2,
    confounder_gru_max_sentence_length=128,
    # Confounder architecture
    confounder_num_latents=8,
    confounder_value_dim=128,
    confounder_max_sentences=100,
    confounder_num_heads=4,
    # Sparse attention
    confounder_sparse_attention=True,
    confounder_sparse_method="entmax",
    confounder_sparse_alpha=1.5,
    device="cuda:0"
)

# IMPORTANT: GRU mode requires tokenizer fitting (like CNN/GRU extractors)
model.fit_tokenizer(train_texts)

# Training loop
for batch in dataloader:
    losses = model.train_step(batch, alpha_propensity=1.0, gamma_rlearner=1.0)
    losses['loss'].backward()
    optimizer.step()

# Interpret attention weights
interpretations = model.feature_extractor.interpret_attention(texts, top_k=5)
for doc_idx, doc_interp in enumerate(interpretations):
    print(f"Document {doc_idx}:")
    for conf_name, attended in doc_interp.items():
        print(f"  {conf_name}: {[a['sentence'][:50] for a in attended]}")
```

### Training with HierarchicalTransformerExtractor
```python
from cdt.models import CausalText

# Simple hierarchical approach: sentence BERT + transformer pooling
model = CausalText(
    feature_extractor_type="hierarchical_transformer",
    model_type="rlearner",
    # Sentence encoder (prajjwal1/bert-tiny is fast and lightweight)
    hier_transformer_sentence_model="prajjwal1/bert-tiny",
    hier_transformer_freeze_sentence_encoder=True,
    # Transformer pooling
    hier_transformer_num_layers=2,
    hier_transformer_num_heads=4,
    hier_transformer_dim=256,
    hier_transformer_dropout=0.1,
    hier_transformer_projection_dim=128,
    # Document limits
    hier_transformer_max_sentences=100,
    hier_transformer_max_sentence_length=128,
    device="cuda:0"
)

# No fit_tokenizer needed - uses pretrained BERT tokenizer
model.fit_tokenizer(train_texts)  # Triggers lazy initialization

# Training loop
for batch in dataloader:
    losses = model.train_step(batch, alpha_propensity=1.0, gamma_rlearner=1.0)
    losses['loss'].backward()
    optimizer.step()

# Interpret attention weights (which sentences the [POOL] token attends to)
interpretations = model.feature_extractor.interpret_attention(texts, top_k=5)
for doc_idx, interp in enumerate(interpretations):
    print(f"Document {doc_idx}:")
    for sent_info in interp['top_sentences']:
        print(f"  [{sent_info['attention']:.3f}] {sent_info['sentence'][:60]}...")
```

### Getting predictions
```python
preds = model.predict(texts)
# preds contains: y0_prob, y1_prob, propensity, y0_logit, y1_logit, t_logit

# Individual treatment effect (probability scale)
ite = preds['y1_prob'] - preds['y0_prob']
```

## Dependencies

Core: torch, transformers, pandas, numpy, scikit-learn, tqdm, pyarrow
Optional:
- openai (for synthetic data generation)
- sentence-transformers (for ConfounderExtractor)
- entmax (for sparse attention; fallback implementations provided if not installed)

## Output Files

```
output_dir/
├── config.json                 # Experiment configuration
├── applied_inference/
│   ├── predictions.parquet     # Per-sample predictions
│   ├── training_log.csv        # Training metrics
│   ├── filter_interpretations.json          # If save_filter_interpretations=true (CNN only)
│   ├── filter_interpretations_summary.txt   # Human-readable filter summary
│   ├── confounder_interpretations.json      # If save_confounder_interpretations=true
│   ├── confounder_interpretations_summary.txt  # Human-readable confounder summary
│   ├── confounder_task_weights.json         # Task-specific confounder weights (propensity vs outcome)
│   └── psm_analysis/           # If matching_analysis.enabled=true
│       ├── matched_pairs.csv   # Matched treated-control pairs
│       ├── balance_statistics.csv # SMD before/after matching
│       ├── sensitivity_analysis.csv # Rosenbaum bounds
│       └── psm_summary.json    # Treatment effect estimates
└── plasmode_experiments/       # If enabled
    └── results.csv             # Plasmode results
```

## Notes for Development

1. **Feature extractor type**: CNN is fastest, BERT is most expressive, GRU is efficient for long sequences, Confounder is best for extracting specific signals from long documents, Hierarchical Transformer is simple and effective for long documents
2. **Tokenizer fitting**: CNN/GRU require `fit_tokenizer()` before use; BERT/Confounder (pretrained)/Hierarchical Transformer use pretrained tokenizers; GRU Confounder (`confounder_use_gru=True`) requires `fit_tokenizer()`
3. **Filter initialization**: Semantic filters improve interpretability; k-means captures data patterns
4. **Propensity trimming**: Optional preprocessing to enforce positivity assumption
5. **All predictions are on probability scale**: ITE = P(Y=1|T=1,X) - P(Y=1|T=0,X)
6. **PSM analysis**: Post-hoc analysis using `cdt.analysis.run_psm_analysis()` - validates DragonNet estimates with traditional methods
7. **Matching module**: `cdt.matching.PropensityMatcher` supports nearest neighbor, optimal (Hungarian), and caliper matching
8. **R-Learner vs DragonNet**: R-Learner provides stronger gradient signal for tau(X) by detaching nuisance functions; use when treatment effect heterogeneity is the primary focus
9. **Confounder extractor**: Use for long documents (2048+ tokens) where confounders are mentioned in specific sentences. Sparse attention (entmax) forces focus on relevant sentences. Use `interpret_attention()` to visualize what each latent confounder attends to.
10. **Hierarchical confounder mode**: Enable with `confounder_hierarchical=True` when fine-grained token distinctions matter (e.g., "ECOG PS 0" vs "ECOG PS 2"). Uses sentence-level sparse attention to focus on relevant sentences, then token-level attention to preserve specific values.
11. **GRU confounder mode**: Enable with `confounder_use_gru=True` for learning confounder extraction from scratch. All parameters (embeddings, GRU, attention, latent confounders) are optimized together via the causal loss. Requires `fit_tokenizer()` before training. Best when pretrained encoders may have domain mismatch or when you want the model to learn clinical-specific representations.
12. **Hierarchical Transformer extractor**: Use `feature_extractor_type="hierarchical_transformer"` for a simple sentence-level encoding approach. Uses lightweight BERT (e.g., `prajjwal1/bert-tiny`) to encode each sentence, then transformer layers with a learnable [POOL] token to aggregate. Simpler than ConfounderExtractor but still effective for long documents. Use `interpret_attention()` to see which sentences contribute most to the representation.
