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
├── inference/applied.py   # Applied inference (CV or fixed split)
├── models/
│   ├── causal_text.py     # Main model (extractor + causal head)
│   ├── cnn_extractor.py, bert_extractor.py, gru_extractor.py
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

examples/                  # Config files for each extractor type
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

**Note**: Hierarchical extractors use overlapping token-based chunking (`chunk_size`, `chunk_overlap`) instead of sentence splitting for more consistent context windows.

### Causal Heads

| Type | Description | Key output |
|------|-------------|------------|
| `dragonnet` | Propensity + Y0/Y1 potential outcomes | ITE = σ(y1) - σ(y0) |
| `uplift` | Base outcome + treatment effect parametrization | ITE from effect head |
| `rlearner` | Direct τ(X) optimization, detached nuisance functions | τ directly predicts ITE |
| `traditional_logreg` | Traditional logistic regression with treatment as feature | ITE = σ(y\|T=1) - σ(y\|T=0) |

**R-Learner advantage**: Nuisance functions (e, m) are detached in R-loss, providing stronger gradient signal for treatment effect modifiers.

**Traditional LogReg approach**: Models P(Y|X, T) directly with treatment concatenated as a feature input to the outcome head. At inference, computes counterfactuals by running the outcome head twice with T=0 and T=1. Simpler loss function (outcome + propensity, no targeted regularization needed). Supports `stop_grad_propensity` but off by default.

## CLI

```bash
cdt init --output config.json
cdt run --config config.json --device cuda:0 --workers 4 [--skip-plasmode] [--verbose]
```

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

See `examples/` for complete config files for each extractor type.

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

## Training Options for τ Learning

| Option | Effect |
|--------|--------|
| `stop_grad_propensity=True` | Prevents propensity from dominating representation |
| `attention_entropy_weight>0` | Encourages focused attention (low entropy) |
| `gamma_rlearner>1.0` | Stronger treatment effect signal |
| `clam_enabled=True` | Enables CLAM instance-level loss (hierarchical extractors) |
| `clam_instance_weight>0` | Weight for instance-level loss on top-attended chunks |

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
| Causal heads | `dragonnet.py`, `rlearner.py`, `uplift.py`, `traditional_logreg.py` |
| Extractors | `cnn_extractor.py`, `bert_extractor.py`, `gru_extractor.py`, `confounder_extractor.py`, `hierarchical_transformer_extractor.py`, `gated_mil_hierarchical_extractor.py`, `gru_transformer_mil_extractor.py`, `gru_pool_extractor.py` |
| Text chunking | `cdt/models/chunking.py` |
| Training | `cdt/inference/applied.py` |
| Config | `cdt/config.py` |
| PSM | `cdt/analysis/psm_analysis.py`, `cdt/matching/propensity_matcher.py` |

## Dependencies

**Core**: torch, transformers, pandas, numpy, scikit-learn, tqdm, pyarrow

**Optional**: openai (synthetic data), sentence-transformers (confounder), entmax (sparse attention; fallback provided)

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
| `examples/new_config.json` | Create example configuration file |
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
- **Long docs**: Use `confounder`, `hierarchical_transformer`, `gated_mil_hierarchical`, `gru_transformer_mil`, or `gru_pool`
- **Interpretability**: `interpret_filters()` (CNN), `interpret_attention()` (others)
- **R-Learner vs DragonNet**: R-Learner for heterogeneous treatment effects; DragonNet for general use
