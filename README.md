# Causal DragonNet Text (CDT)

A framework for clinical causal inference using electronic health record (EHR) text as the primary input. CDT estimates treatment effects from unstructured clinical narratives using semantically-initialized CNNs with DragonNet causal inference heads.

## Clinical Research Objective

Observational studies using EHR data are essential for comparative effectiveness research when randomized trials are infeasible or unethical. However, standard approaches rely on structured covariates (diagnoses, labs, demographics) which may fail to capture critical confounders documented only in clinical notes—such as functional status, symptom severity, patient preferences, or nuanced disease characteristics.

CDT addresses this gap by:
- **Extracting confounders from clinical text** using 1D CNNs with semantically meaningful filters
- **Estimating treatment effects** using a DragonNet architecture that jointly models propensity scores and potential outcomes
- **Validating methods via plasmode simulation** to assess sensitivity to unmeasured confounding and model misspecification

## How It Works

### Architecture Overview

CDT processes clinical text through a CNN-based pipeline:

1. **Word-Level Tokenization**: Clinical notes are tokenized at the word level with a vocabulary built from training data

2. **Semantic Embeddings**: Word embeddings can be initialized from ClinicalBERT (Bio_ClinicalBERT) by:
   - Tokenizing each vocabulary word with BERT's subword tokenizer
   - Averaging the subword embeddings
   - Projecting to the CNN embedding dimension

3. **CNN Feature Extraction**: 1D convolutions with multiple kernel sizes (e.g., 3, 4, 5, 7 words) extract n-gram patterns. Filters can be initialized from:
   - **Explicit clinical concepts**: User-specified phrases like "stage iv cancer", "performance status poor"
   - **Latent patterns**: Data-driven filters learned via k-means clustering of training n-grams

4. **DragonNet Causal Inference**: The CNN features feed into a DragonNet that jointly predicts:
   - Treatment propensity P(T=1|X)
   - Potential outcomes E[Y|T=0,X] and E[Y|T=1,X]
   - Individual treatment effects (ITE) as the difference in potential outcomes

### Semantic Filter Initialization

The key innovation is initializing CNN filters with clinical meaning:

```
Kernel Size 3 (3-word phrases):
  - "stage iv cancer"
  - "performance status poor"
  - "disease progression noted"
  ...

Kernel Size 5 (5-word phrases):
  - "white blood cell count elevated"
  - "computed tomography scan of chest"
  ...
```

Each explicit concept is converted to an embedding sequence using the BERT-initialized word embeddings, creating a filter that responds strongly to that clinical pattern. Additional "latent" filters are learned by clustering n-grams from the training corpus.

### Confounder Extractors

For long clinical documents where confounders are mentioned in specific sentences, CDT offers specialized confounder extractors:

- **ConfounderExtractor**: Sentence-level attention with Perceiver-style cross-attention. Learnable latent queries attend to sentence embeddings using sparse attention (entmax).

- **HierarchicalConfounderExtractor**: Token-level attention using pretrained BERT. Sentence-level sparse attention identifies relevant sentences, then token-level attention preserves fine-grained signal (e.g., "ECOG PS 0" vs "ECOG PS 2").

- **GRUHierarchicalConfounderExtractor**: Learns entirely from scratch via the causal objective. Uses BiGRU with learnable embeddings instead of pretrained BERT. All parameters (embeddings, GRU, attention, latent confounders) optimize together.

**Task-Specific Multi-Head Aggregation**: The hierarchical confounder extractors use task-specific aggregation to combine the K latent confounder representations. Separate learnable queries for propensity and outcome tasks allow each to weight confounders differently based on patient characteristics. This reduces dimensionality from K×D to 2×D while enabling patient-specific and task-specific weighting.

See `examples/confounder_config.json` and `examples/gru_confounder_config.json` for configuration examples.

### Hierarchical Transformer Extractor

For a simpler alternative to ConfounderExtractor, the **HierarchicalTransformerExtractor** provides straightforward hierarchical encoding without latent confounders or sparse attention:

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
- **Sentence encoding**: Uses `prajjwal1/bert-tiny` (4.4M params) by default - fast and lightweight
- **Learnable [POOL] token**: Like BERT's [CLS], aggregates all sentence embeddings through self-attention
- **Transformer pooling**: 1-2 transformer layers allow sentences to attend to each other
- **Interpretability**: `interpret_attention()` method shows which sentences contribute most to the final representation
- **Optional fine-tuning**: Sentence encoder can be frozen or fine-tuned

**Key configuration:**
```python
hier_transformer_sentence_model: str = "prajjwal1/bert-tiny"  # Sentence encoder
hier_transformer_freeze_sentence_encoder: bool = True  # Freeze or fine-tune
hier_transformer_max_sentences: int = 100  # Max sentences per document
hier_transformer_max_sentence_length: int = 128  # Max tokens per sentence
hier_transformer_num_layers: int = 2  # Transformer layers for pooling
hier_transformer_num_heads: int = 4  # Attention heads
hier_transformer_dim: int = 256  # Transformer hidden dimension
hier_transformer_dropout: float = 0.1
hier_transformer_projection_dim: int = 128  # Final output dimension
```

See `examples/hierarchical_transformer_config.json` for a complete configuration example.

### Gated MIL Hierarchical Extractor

For causally-motivated confounder extraction with task-specific weighting, the **GatedMILHierarchicalExtractor** uses gated MIL (Multiple Instance Learning) attention from pathology AI:

**Key insight**: Confounders are patient characteristics (e.g., performance status, metastatic burden) that affect both treatment assignment and outcomes. The same K confounders should feed into all tasks, but each task can weight them differently:
- Propensity: "Which confounders predict treatment?"
- Tau: "Which confounders modify treatment effect?"
- Outcome: "Which confounders predict baseline outcome?"

**Sentence-level Architecture (default):**
```
Long Clinical Text
        ↓
Split into Sentences (S sentences)
        ↓
Tiny BERT per Sentence → [CLS] token (S × D)
        ↓
Gated MIL Attention: h = tanh(V·sent) ⊙ sigmoid(U·sent)
  For each confounder k: a_k = softmax(q_k · W · h)
        ↓
K Confounder Representations (K × D)
        ↓
Task-Specific Weighting (propensity, tau/y0, outcome/y1)
        ↓
Concatenate + MLP Projection → Final Representation
```

**Token-level Architecture (`gated_mil_hierarchical=True`):**

When fine-grained token distinctions matter (e.g., "ECOG PS 0" vs "ECOG PS 2"), enable token-level gated pooling:

```
Long Clinical Text
        ↓
Split into Sentences (S sentences)
        ↓
Tiny BERT per Sentence → ALL tokens (S × L × D)
        ↓
Token-Level Gated Pooling:
  Each confounder query attends to tokens,
  creating K confounder-specific sentence embeddings
        ↓
S × K × D confounder-specific sentence embeddings
        ↓
Sentence-Level Gated MIL Attention (per confounder view)
        ↓
K Confounder Representations → Task-Specific Weighting
```

**Key features:**
- **Gated attention**: tanh × sigmoid gating suppresses irrelevant sentences (from pathology AI)
- **K confounder queries**: Each learns to extract one type of confounder signal
- **Task-specific weighting**: Same K confounders, but weighted differently per task
- **Token-level option**: Preserves fine-grained distinctions when enabled
- **Interpretability**: `interpret_attention()` shows top-attended sentences per confounder
- **`get_task_weights()`**: Shows how confounders are weighted per task

**Key configuration:**
```python
gated_mil_sentence_model: str = "prajjwal1/bert-tiny"  # Sentence encoder
gated_mil_freeze_sentence_encoder: bool = True  # Freeze or fine-tune
gated_mil_max_sentences: int = 100  # Max sentences per document
gated_mil_max_sentence_length: int = 128  # Max tokens per sentence
gated_mil_hidden_dim: int = 128  # Hidden dim for gated attention
gated_mil_num_confounders: int = 4  # Number of confounder queries (K)
gated_mil_dropout: float = 0.1
gated_mil_projection_dim: int = 128  # Final output dimension
# Token-level mode (optional)
gated_mil_hierarchical: bool = False  # Enable token-level gated pooling
gated_mil_token_hidden_dim: int = 64  # Hidden dim for token-level gating
```

**When to use:**
- Long documents with multiple confounders mentioned in different sentences
- When you want explicit K confounder queries with task-specific weighting
- When interpretability (which sentences each confounder attends to) is important
- Use `gated_mil_hierarchical=True` when fine-grained token distinctions matter

See `examples/gated_mil_config.json` for a complete configuration example.

### Workflow Modes

#### Applied Inference

For estimating treatment effects on real clinical data:

```
Clinical Text → Word Tokens → CNN Features → DragonNet → Treatment Effect Estimates
```

The system supports:
- **K-fold cross-validation**: Out-of-sample predictions across all data
- **Fixed train/val/test splits**: When data comes pre-split

#### Plasmode Simulation

Plasmode simulation generates synthetic outcomes while preserving the real covariate (text) distribution. This enables:

- **Method validation**: Test if your model can recover known treatment effects
- **Sensitivity analysis**: Assess robustness across different outcome-generating processes

The plasmode workflow:
1. Train a "generator" model on real data to learn confounder representations
2. Generate synthetic outcomes with known true treatment effects
3. Train an "evaluator" model on the synthetic data
4. Compare estimated effects to ground truth

### Propensity Score Matching Analysis

CDT includes a traditional propensity score matching (PSM) module that can be run as a post-hoc analysis using DragonNet's learned propensity scores. This provides:

- **Treatment effect estimation via multiple methods**:
  - **ATT (Average Treatment Effect on Treated)**: From matched pairs
  - **ATE via IPW**: Inverse probability weighting
  - **ATE via Stratification**: Propensity score subclassification

- **Balance diagnostics**: Standardized mean differences before/after matching

- **Statistical inference**: Bootstrap confidence intervals, McNemar's test (binary outcomes), paired t-tests (continuous outcomes)

- **Sensitivity analysis**: Rosenbaum bounds to assess robustness to unmeasured confounding

This allows comparison of DragonNet's ITE estimates with traditional PSM estimates, providing validation and enabling traditional statistical inference.

### Matched Pair ITE Estimation

As an alternative to DragonNet's joint modeling approach, CDT provides a **two-stage matched pair ITE estimation** pipeline:

```
Stage 1: Train Propensity Model
        ↓
     (Optional: Joint outcome training for true confounder learning)
        ↓
Stage 2: Match Patients (propensity or embedding similarity)
        ↓
Stage 3: Train Outcome/Tau Model on Matched Pairs
        ↓
ITE Predictions for All Patients
```

**Key features:**

- **Propensity matching**: Match treated/control patients by propensity score (caliper-based)
- **Embedding matching**: Match by representation similarity (cosine distance)
- **Tau head**: Predicts treatment effect from untreated patient's embedding
- **Joint outcome training** (new): Co-train Stage 1 on both propensity AND outcome to learn true confounders
- **Flexible freezing** (new): Optionally fine-tune representation during Stage 2

**Joint Outcome Training:**

By default, Stage 1 trains only on propensity (treatment prediction). However, features optimized only for propensity may miss outcome-relevant confounders. When `joint_outcome_training=True`, Stage 1 also trains on outcome prediction:

```
Loss = alpha_propensity * BCE(T_pred, T) + alpha_outcome * BCE(Y_pred, Y)
```

This encourages learning features that are true confounders (predictive of both T and Y) rather than just instruments (predict T only) or mediators.

**Freeze vs Fine-tune Representation:**

By default, the representation is frozen after Stage 1 to preserve covariate balance. Setting `freeze_representation_stage2=False` allows continued fine-tuning during matched pair training, which may improve outcome/tau prediction at the cost of potentially affecting balance.

**Dynamic Re-matching:**

When the representation is not frozen (`freeze_representation_stage2=False`), the embedding space changes during training. By default, matched pairs are computed once before Stage 2. Enable `dynamic_rematching=True` to periodically recompute matches as the representation evolves:

```json
{
  "matched_pair": {
    "freeze_representation_stage2": false,
    "dynamic_rematching": true,
    "rematching_frequency": 5,
    "rematching_warmup_epochs": 2
  }
}
```

This recomputes propensity scores (or embeddings) and re-runs the matching algorithm every `rematching_frequency` epochs, starting after `rematching_warmup_epochs`.

**Cross-Encoder for Residual Confounder Capture:**

After propensity matching, matched pairs have similar propensity scores but may still differ in subtle ways captured in the clinical text. The optional cross-encoder learns to identify these discriminative features:

- **How it works:**
  1. Extract sentence embeddings for both matched patients
  2. Cross-attention: T sentences attend to U sentences (and vice versa)
  3. Discriminative queries aggregate differences
  4. Residual features enhance tau prediction

- **Enable with:**
```json
{
  "matched_pair": {
    "use_cross_encoder": true,
    "cross_encoder_num_queries": 4,
    "cross_encoder_num_heads": 4,
    "gamma_discrimination": 0.1,
    "delta_consistency": 0.1
  }
}
```

- **Loss function:**
```
L_total = α * L_outcome + β * L_tau + γ * L_disc + δ * L_consistency
```

- **Interpretation:** When `save_cross_encoder_attention=true`, the output includes:
  - Top discriminative sentences for each matched pair
  - Cross-attention matrices showing which sentences relate across patients

See `examples/matched_pair_cross_encoder_config.json` for a complete example.

**Configuration:**
```json
{
  "matched_pair": {
    "matching_method": "propensity",
    "matching_algorithm": "optimal",
    "caliper": 0.2,

    "joint_outcome_training": true,
    "alpha_propensity_stage1": 1.0,
    "alpha_outcome_stage1": 0.5,
    "freeze_representation_stage2": false,
    "dynamic_rematching": false
  }
}
```

See `examples/matched_pair_config.json` for a complete configuration example.

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for practical use)

### Clone the Repository

```bash
git clone https://github.com/kenlkehl/causal-dragonnet-text.git
cd causal-dragonnet-text
```

### Install uv (Recommended Package Manager)

[uv](https://github.com/astral-sh/uv) is a fast Python package manager. Install it via:

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

### Create Environment and Install

```bash
# Create a virtual environment
uv venv --python 3.10
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install CDT in editable mode
uv pip install -e .

# For development (includes testing/linting tools)
uv pip install -e ".[dev]"
```

### Alternative: Standard pip Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Dataset Requirements

CDT expects datasets in Parquet or CSV format with the following columns:

| Column | Description | Type |
|--------|-------------|------|
| `clinical_text` | The clinical narrative text | string |
| `treatment_indicator` | Binary treatment assignment (0 or 1) | int/float |
| `outcome_indicator` | Binary outcome (0 or 1) | int/float |
| `split` | Data split: "train", "val", "test" (optional for CV) | string |

Example:
```
| clinical_text                          | treatment_indicator | outcome_indicator | split |
|----------------------------------------|---------------------|-------------------|-------|
| "58yo male with stage IV NSCLC..."     | 1                   | 0                 | train |
| "Patient presents with dyspnea..."     | 0                   | 1                 | train |
| "History of smoking, 40 pack-years..." | 1                   | 1                 | test  |
```

## Running Experiments

### Basic Usage

```bash
# Generate a default configuration file
cdt init --output my_config.json

# Edit my_config.json to set your dataset paths and parameters

# Run the experiment
cdt run --config my_config.json
```

### Configuration Structure

A configuration file controls all aspects of the experiment:

```json
{
  "output_dir": "./cdt_results",
  "seed": 42,
  "device": "cuda:0",
  "num_workers": 1,

  "applied_inference": {
    "dataset_path": "./data/clinical_notes.parquet",
    "text_column": "clinical_text",
    "outcome_column": "outcome_indicator",
    "treatment_column": "treatment_indicator",
    "cv_folds": 5,

    "architecture": {
      "model_type": "dragonnet",
      "cnn_embedding_dim": 128,
      "cnn_num_filters": 256,
      "cnn_kernel_sizes": [3, 4, 5, 7],
      "cnn_max_length": 2048,
      "cnn_min_word_freq": 2,
      "cnn_max_vocab_size": 50000,

      "cnn_init_embeddings_from": "emilyalsentzer/Bio_ClinicalBERT",
      "cnn_freeze_embeddings": false,

      "cnn_explicit_filter_concepts": {
        "3": ["stage iv cancer", "performance status poor", "disease progression noted"],
        "4": ["no evidence of disease", "complete response to treatment"],
        "5": ["white blood cell count elevated"],
        "7": ["patient was started on first line chemotherapy regimen"]
      },
      "cnn_num_latent_filters": 64,

      "dragonnet_representation_dim": 128,
      "dragonnet_hidden_outcome_dim": 64
    },

    "training": {
      "epochs": 50,
      "batch_size": 8,
      "learning_rate": 0.0001,
      "alpha_propensity": 1.0,
      "beta_targreg": 0.1
    },

    "matching_analysis": {
      "enabled": true,
      "method": "nearest",
      "caliper": 0.2,
      "caliper_scale": "std",
      "ratio": 1,
      "replacement": false,
      "n_bootstrap": 1000,
      "ci_level": 0.95
    }
  },

  "plasmode_experiments": {
    "enabled": false,
    "num_repeats": 3,
    "train_fraction": 0.8,
    "plasmode_scenarios": [
      {
        "generation_mode": "phi_linear",
        "target_ate_prob": 0.10,
        "ite_heterogeneity_scale": 1.0
      }
    ]
  }
}
```

### Key Configuration Options

**CNN Architecture:**
- `cnn_embedding_dim`: Dimension of word embeddings (default: 128)
- `cnn_num_filters`: Number of filters per kernel size (default: 256)
- `cnn_kernel_sizes`: List of kernel sizes for n-gram detection (default: [3, 4, 5, 7])
- `cnn_max_length`: Maximum sequence length in words (default: 2048)

**Semantic Initialization:**
- `cnn_init_embeddings_from`: HuggingFace model for embedding initialization (e.g., "emilyalsentzer/Bio_ClinicalBERT")
- `cnn_freeze_embeddings`: Whether to freeze BERT-initialized embeddings during training
- `cnn_explicit_filter_concepts`: Dict mapping kernel size (as string) to list of concept phrases
- `cnn_num_latent_filters`: Number of k-means derived filters per kernel size

**DragonNet Head:**
- `dragonnet_representation_dim`: Dimension of shared representation layer
- `dragonnet_hidden_outcome_dim`: Hidden dimension for outcome prediction heads

**Training:**
- `alpha_propensity`: Weight for propensity score loss
- `beta_targreg`: Weight for targeted regularization loss
- `cv_folds`: Number of cross-validation folds (0 or 1 for fixed splits)

**Plasmode:**
- `generation_mode`: How synthetic outcomes are generated ("phi_linear")
- `target_ate_prob`: True average treatment effect on probability scale (e.g., 0.10 = 10% increase)
- `ite_heterogeneity_scale`: Scale of individual treatment effect heterogeneity

**Matching Analysis (PSM):**
- `enabled`: Whether to run PSM analysis using DragonNet's propensity scores (default: true)
- `method`: Matching algorithm - "nearest" (greedy), "optimal" (Hungarian), or "caliper" (default: "nearest")
- `caliper`: Maximum allowed distance for a match (default: 0.2)
- `caliper_scale`: Scale for caliper - "propensity", "logit", or "std" (standard deviations of logit propensity)
- `ratio`: Matching ratio (1:k matching, default: 1)
- `replacement`: Whether to match with replacement (default: false)
- `n_bootstrap`: Number of bootstrap iterations for confidence intervals (default: 1000)
- `ci_level`: Confidence level for intervals (default: 0.95)

**Interpretation:**
- `save_filter_interpretations`: Save CNN filter interpretation analysis (default: false, CNN only)
- `filter_interpretation_top_k`: Number of top n-grams per filter to save (default: 10)
- `save_confounder_interpretations`: Save confounder attention analysis (default: false, confounder extractors only)
- `confounder_interpretation_top_k`: Number of top-attended sentences per confounder (default: 5)

### CLI Options

```bash
cdt run --config config.json \
    --device cuda:1 \           # Override GPU device
    --workers 4 \               # Parallel workers for CV folds
    --output-dir ./my_results \ # Override output directory
    --skip-applied \            # Skip applied inference
    --skip-plasmode \           # Skip plasmode experiments
    --verbose                   # Enable debug logging
```

## Output Files

After running, results are saved to the output directory:

```
output_dir/
├── config.json                    # Copy of experiment configuration
├── applied_inference/
│   ├── predictions.parquet        # Per-sample treatment effect estimates
│   ├── training_log.csv           # Training metrics per epoch
│   ├── filter_interpretations.json           # (if save_filter_interpretations=true, CNN only)
│   ├── filter_interpretations_summary.txt    # Human-readable filter summary
│   ├── confounder_interpretations.json       # (if save_confounder_interpretations=true)
│   ├── confounder_interpretations_summary.txt # Human-readable confounder summary
│   ├── confounder_task_weights.json          # Task-specific confounder weights
│   └── psm_analysis/              # (if matching_analysis.enabled=true)
│       ├── matched_pairs.csv      # Matched treated-control pairs with distances
│       ├── balance_statistics.csv # SMD before/after matching
│       ├── sensitivity_analysis.csv # Rosenbaum bounds for hidden bias
│       └── psm_summary.json       # Treatment effect estimates and comparison
└── plasmode_experiments/          # (if enabled)
    ├── results.csv                # Aggregated plasmode metrics
    └── simulated_datasets/        # (if save_datasets=true)
```

The `predictions.parquet` file contains:
- `pred_y0_prob`: Predicted outcome probability under control
- `pred_y1_prob`: Predicted outcome probability under treatment
- `pred_ite_prob`: Predicted individual treatment effect on probability scale (pred_y1_prob - pred_y0_prob)
- `pred_propensity_prob`: Predicted treatment propensity (probability)
- `cv_fold`: Which cross-validation fold (if using CV)

## Example: Semantic CNN for Oncology

See `examples/semantic_cnn_config.json` for a complete configuration with clinical oncology concepts:

```json
{
  "cnn_explicit_filter_concepts": {
    "3": [
      "stage iv cancer",
      "performance status poor",
      "disease progression noted",
      "tumor size increased",
      "lymph node positive",
      "prior chemotherapy received",
      "adverse event reported",
      "patient tolerated well"
    ],
    "4": [
      "no evidence of disease",
      "complete response to treatment",
      "partial response to therapy",
      "stable disease on imaging",
      "progressive disease confirmed today"
    ],
    "5": [
      "white blood cell count elevated",
      "eastern cooperative oncology group performance",
      "computed tomography scan of chest"
    ],
    "7": [
      "patient was started on first line chemotherapy regimen",
      "imaging revealed new metastatic lesions in the liver"
    ]
  }
}
```

These concepts create CNN filters that specifically detect clinically meaningful patterns in the text.

## Citation

If you use CDT in your research, please cite:

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

## Contact

Ken Kehl - kenneth_kehl@dfci.harvard.edu
