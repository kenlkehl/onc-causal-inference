# CLAUDE.md - CDT (Causal DragonNet Text)

## Project Overview

CDT is a framework for **clinical causal inference using EHR text**. It estimates treatment effects from unstructured clinical narratives by combining text feature extraction with causal inference heads (DragonNet, R-Learner).

**Key purpose**: Extract confounders from clinical text to estimate ITEs and ATEs in observational studies.

## Repository Structure

```
cdt/
├── cli.py                    # CLI (`cdt init`, `cdt run`)
├── config.py                 # Dataclass configurations
├── data/dataset.py           # ClinicalTextDataset
├── models/
│   ├── causal_text.py        # Main model (extractor + causal head)
│   ├── cnn_extractor.py      # 1D CNN with semantic filter init
│   ├── bert_extractor.py     # HuggingFace transformer
│   ├── gru_extractor.py      # BiGRU with attention
│   ├── confounder_extractor.py           # Perceiver-style cross-attention
│   ├── hierarchical_transformer_extractor.py  # Sentence BERT + transformer pooling
│   ├── gated_mil_hierarchical_extractor.py    # Gated MIL + task-specific weighting
│   ├── dragonnet.py          # DragonNet head (propensity + Y0/Y1)
│   ├── rlearner.py           # R-Learner head (direct tau)
│   └── uplift.py             # UpliftNet head
├── matching/propensity_matcher.py  # PropensityMatcher, balance utilities
├── analysis/
│   ├── statistical_analysis.py     # ATT/ATE, McNemar's, Rosenbaum
│   └── psm_analysis.py             # run_psm_analysis()
├── training/plasmode.py      # Plasmode simulation
└── inference/applied.py      # Training loop, CV

synthetic_data/               # LLM-based synthetic data generation
examples/                     # Example config files
```

## Key Commands

```bash
cdt init --output config.json
cdt run --config config.json --device cuda:0 --workers 4 --skip-plasmode
```

## Architecture

### CausalText Model

Combines a **feature extractor** + **causal head**:

| Extractor | Description | Tokenizer | Best For |
|-----------|-------------|-----------|----------|
| `cnn` | 1D CNN with semantic/k-means filters | `fit_tokenizer()` required | Fast, interpretable |
| `bert` | HuggingFace transformer [CLS] | Pretrained | Short docs (<512 tokens) |
| `gru` | BiGRU with attention | `fit_tokenizer()` required | Long sequences, O(N) |
| `confounder` | Perceiver cross-attention, sparse | Pretrained* | Long docs, explicit confounders |
| `hierarchical_transformer` | Sentence BERT + [POOL] token | Pretrained | Long docs, simple |
| `gated_mil_hierarchical` | Gated MIL + K confounders | Pretrained | Long docs, task-specific weighting |

\* `confounder_use_gru=True` requires `fit_tokenizer()`

| Causal Head | Description |
|-------------|-------------|
| `dragonnet` | Propensity + Y0/Y1 potential outcomes |
| `rlearner` | Propensity + marginal outcome + tau (detached nuisances for better tau gradients) |
| `uplift` | Base outcome + treatment effect parametrization |

### Feature Extractor Details

**CNN** (`cnn_extractor.py`): Semantic filter init from concepts, k-means filters, BERT embedding init. Use `interpret_filters()` for interpretability.

**Confounder** (`confounder_extractor.py`): K learnable latent queries attend to sentences via sparse attention (entmax). Modes:
- Standard: Sentence embeddings → sparse cross-attention → K latents
- Hierarchical (`confounder_hierarchical=True`): Token-level BERT + sentence gating
- GRU (`confounder_use_gru=True`): Learns from scratch, requires `fit_tokenizer()`

**Gated MIL** (`gated_mil_hierarchical_extractor.py`): tanh × sigmoid gating, K confounder queries with task-specific weighting. Optional token-level mode with `gated_mil_hierarchical=True`.

### Causal Head Details

**DragonNet**: Outputs `y0_logit`, `y1_logit`, `t_logit`. ITE = sigmoid(y1) - sigmoid(y0).

**R-Learner**: Outputs `tau_pred`, `m_prob`, derived `y0_prob`/`y1_prob`. Nuisance functions (e, m) detached in R-loss for stronger tau gradients. Reference: Nie & Wager (2021).

## Dataset Format

| Column | Type | Description |
|--------|------|-------------|
| `clinical_text` | string | Clinical narrative |
| `treatment_indicator` | int | Binary (0/1) |
| `outcome_indicator` | int | Binary (0/1) |
| `split` | string | Optional: "train"/"val"/"test" |

## Training Example

```python
from cdt.models import CausalText

model = CausalText(
    feature_extractor_type="gated_mil_hierarchical",  # or "cnn", "bert", "confounder", etc.
    model_type="rlearner",  # or "dragonnet", "uplift"
    gated_mil_num_confounders=4,
    device="cuda:0"
)

# Required for cnn/gru/confounder_use_gru extractors
model.fit_tokenizer(train_texts)

for batch in dataloader:
    losses = model.train_step(
        batch,
        alpha_propensity=1.0,
        gamma_rlearner=1.0,       # R-learner weight
        stop_grad_propensity=True,  # Prevent propensity dominating features
        attention_entropy_weight=0.1  # Focus attention (gated_mil only)
    )
    losses['loss'].backward()
    optimizer.step()

# Predictions
preds = model.predict(texts)
ite = preds['y1_prob'] - preds['y0_prob']

# Interpretability
interp = model.feature_extractor.interpret_attention(texts, top_k=5)  # confounder/gated_mil
interp = model.feature_extractor.interpret_filters(texts)  # cnn
```

## Matching & Analysis

```python
from cdt.matching import PropensityMatcher
from cdt.analysis import run_psm_analysis

# Manual matching
matcher = PropensityMatcher(method='nearest', caliper=0.2)
result = matcher.match(propensity_scores, treatment)

# Full PSM analysis pipeline
results = run_psm_analysis(predictions_df, config, output_dir)
# Returns: ATT, ATE (IPW/stratified), balance stats, Rosenbaum bounds
```

## Configuration

Main classes in `cdt/config.py`:
- `ExperimentConfig`: Top-level
- `ModelArchitectureConfig`: Extractor type, dimensions
- `TrainingConfig`: LR, epochs, loss weights (alpha_propensity, beta_targreg, gamma_rlearner)
- `MatchingAnalysisConfig`: PSM settings
- `MatchedPairConfig`: Two-stage matched pair ITE estimation

### MatchedPairConfig (Two-Stage ITE Estimation)

Alternative to DragonNet using propensity matching:

| Stage | Description |
|-------|-------------|
| Stage 1 | Train propensity model (optionally with joint outcome training) |
| Stage 2 | Match patients by propensity or embedding similarity |
| Stage 3 | Train outcome/tau model on matched pairs |

**Cross-validation**: When using n-fold CV (`cv_folds > 1`), training uses pure n-fold CV where 100% of the training fold is used for propensity model training (no internal validation split). The training runs for a fixed number of epochs without early stopping within each fold.

Key options:

| Option | Default | Purpose |
|--------|---------|---------|
| `joint_outcome_training` | `False` | Train Stage 1 on both propensity AND outcome |
| `alpha_propensity_stage1` | `1.0` | Weight for propensity loss in joint training |
| `alpha_outcome_stage1` | `1.0` | Weight for outcome loss in joint training |
| `freeze_representation_stage2` | `True` | Freeze representation during Stage 2 training |
| `dynamic_rematching` | `False` | Re-match patients during Stage 2 (when representation not frozen) |
| `rematching_frequency` | `5` | Re-match every N epochs |
| `rematching_warmup_epochs` | `0` | Skip re-matching for first N epochs |

**Joint outcome training**: When enabled, Stage 1 trains on both treatment and outcome prediction. This encourages learning features that are true confounders (predictive of both T and Y) rather than just instruments or mediators.

**Freeze representation**: When `freeze_representation_stage2=False`, the representation continues to be fine-tuned during Stage 2 outcome/tau training.

**Dynamic re-matching**: When the representation is not frozen (`freeze_representation_stage2=False`), the embedding space changes during training. By default, matched pairs are computed once before Stage 2. Enable `dynamic_rematching=True` to periodically recompute matches as the representation evolves. This recomputes propensity scores (or embeddings) and re-runs the matching algorithm every `rematching_frequency` epochs, starting after `rematching_warmup_epochs`.

### Cross-Encoder for Residual Confounder Capture

When `use_cross_encoder=True`, Stage 3 training uses a `ResidualCrossEncoder` to identify discriminative features between matched pairs that may represent residual confounders missed by propensity matching.

**Architecture:**
- Bidirectional cross-attention between treated and untreated sentence embeddings
- Discriminative query aggregation to extract residual features
- Gated attention (tanh × sigmoid) for focused attention
- Enhanced tau head: repr_U + residual_features

**Key options:**

| Option | Default | Purpose |
|--------|---------|---------|
| `use_cross_encoder` | `False` | Enable cross-encoder for Stage 3 |
| `cross_encoder_num_queries` | `4` | Discriminative queries for aggregation |
| `cross_encoder_num_heads` | `4` | Attention heads |
| `cross_encoder_hidden_dim` | `128` | Hidden dimension for cross-encoder |
| `cross_encoder_use_gating` | `True` | Use gated attention (tanh × sigmoid) |
| `gamma_discrimination` | `0.1` | Weight for treatment discrimination loss |
| `delta_consistency` | `0.1` | Weight for tau-outcome consistency loss |
| `save_cross_encoder_attention` | `False` | Save attention weights for analysis |

**Loss function:**
```
L_total = α * L_outcome + β * L_tau + γ * L_disc + δ * L_consistency
```

**Interpretability:**
The cross-encoder provides `interpret_discrimination()` to identify which sentences from the treated patient most distinguish them from the untreated patient in their matched pair.

See `examples/matched_pair_cross_encoder_config.json` for a complete example.

### End-to-End Training Mode

When `end_to_end_training=True`, training skips the separate Stage 1 propensity pre-training and performs joint training from scratch using a single unified `EndToEndMatchedPairModel`.

**Key differences from 3-Stage approach:**
- Single model with shared feature extractor + propensity/outcome/tau heads
- Propensity loss applied throughout training (not just Stage 1)
- Representation always trainable (never frozen)
- Re-matching is mandatory (computed periodically as model improves)

**Key options:**

| Option | Default | Purpose |
|--------|---------|---------|
| `end_to_end_training` | `False` | Enable end-to-end training mode |
| `e2e_epochs` | `100` | Total training epochs |
| `e2e_lr` | `1e-4` | Learning rate |
| `e2e_batch_size` | `32` | Batch size |
| `e2e_alpha_propensity` | `1.0` | Propensity loss weight |
| `e2e_alpha_outcome` | `1.0` | Outcome loss weight |
| `e2e_beta_tau` | `1.0` | Tau loss weight |
| `e2e_rematching_frequency` | `5` | Re-match every N epochs |
| `e2e_rematching_warmup_epochs` | `5` | Skip re-matching for warmup |
| `e2e_initial_matching` | `"propensity"` | Initial matching strategy: "propensity", "embedding", or "random" |
| `e2e_initial_caliper_multiplier` | `2.0` | Relaxed caliper for initial random model |
| `e2e_lr_schedule` | `"cosine"` | LR schedule: "cosine", "linear", or "constant" |
| `e2e_early_stopping_patience` | `20` | Stop if no improvement for N epochs |

**Architecture:**
```
EndToEndMatchedPairModel
├── HierarchicalTransformerExtractor
├── repr_layers (Linear -> ELU -> Linear -> LayerNorm)
├── propensity_head (Linear -> ReLU -> Dropout -> Linear)
├── outcome_head (shared for Y_U and Y_T)
└── tau_head (predicts ITE from untreated repr)
```

**Usage:**
```python
from cdt.models import EndToEndMatchedPairModel
from cdt.training import train_end_to_end_matched_pair

model = EndToEndMatchedPairModel(device="cuda:0")
model.fit_tokenizer(texts)

model, history = train_end_to_end_matched_pair(
    model, train_df, val_df, config, device
)

# Inference
y0, y1, ite = model.predict_potential_outcomes(test_texts)
propensity = model.predict_propensity(test_texts)
```

See `examples/matched_pair_e2e_config.json` for a complete example.

### Mean-Embedding ITE Model

When `use_mean_embedding_ite=True`, Stage 3 uses a symmetric ITE formulation where the ITE head is trainable and the base model can be frozen or unfrozen.

**Prerequisites:**
- Requires `joint_outcome_training=True` in Stage 1 (to have an outcome head)

**Key differences from standard approach:**

| Aspect | Standard (`MatchedPairOutcomeModel`) | Mean-Embedding (`MeanEmbeddingITEModel`) |
|--------|-------------------------------------|------------------------------------------|
| Input to ITE head | Untreated repr only (`repr_U`) | Mean of pair: `(repr_T + repr_U) / 2` |
| Outcome head | Trained in Stage 3 | Frozen (default) or trainable |
| Stage 3 trainable params | outcome_head + tau_head | ITE head only (frozen) or all (unfrozen) |
| Inference | Use repr directly | Use patient's own repr directly |

**Key options:**

| Option | Default | Purpose |
|--------|---------|---------|
| `use_mean_embedding_ite` | `False` | Enable mean-embedding ITE model |
| `mean_ite_hidden_dim` | `128` | Hidden dimension for ITE head |
| `mean_ite_dropout` | `0.2` | Dropout rate for ITE head |
| `freeze_representation_stage2` | `True` | Freeze base model during ITE training |

**Frozen mode (default, `freeze_representation_stage2=True`):**
- Frozen components: feature extractor, propensity head, outcome head
- Trainable: ITE head only (new 3-layer MLP)
- Representations pre-extracted once before training
- Training input: mean of matched pair embeddings `(repr_T + repr_U) / 2`

**Unfrozen mode (`freeze_representation_stage2=False`):**
- Trainable: ITE head + full propensity model (with lower LR 0.1x)
- Representations computed on-the-fly each batch
- Allows continued feature learning during Stage 3
- After training, outcome head is frozen for inference consistency

**Training formulation:**
```
mean_repr = (repr_T + repr_U) / 2
base_logit = outcome_head(mean_repr)  # frozen or trainable
ite_half = ite_head(mean_repr)        # always trainable

Y_T_logit = base_logit + ite_half
Y_U_logit = base_logit - ite_half

Loss = BCE(sigmoid(Y_T_logit), y_T) + BCE(sigmoid(Y_U_logit), y_U)
```

**Inference (single patient):**
```
base_logit = frozen_outcome_head(repr)
ite_half = ite_head(repr)

Y0_logit = base_logit - ite_half
Y1_logit = base_logit + ite_half
ITE_prob = sigmoid(Y1_logit) - sigmoid(Y0_logit)
```

**Usage:**
```python
from cdt.models import PropensityMatchingModel, MeanEmbeddingITEModel
from cdt.training import train_propensity_model, train_mean_embedding_ite_model

# Stage 1: Train propensity model with joint outcome training
propensity_model = PropensityMatchingModel(
    joint_outcome_training=True,  # Required!
    device="cuda:0"
)
propensity_model, _, instance_head = train_propensity_model(propensity_model, train_df, val_df, config, device)

# Stage 2: Match patients (as usual)
# ...

# Stage 3: Train mean-embedding ITE model
ite_model, history = train_mean_embedding_ite_model(
    propensity_model, train_df, matched_pairs, config, device
)

# Inference
repr = propensity_model.get_representation(test_texts)
y0, y1, ite = ite_model.predict_potential_outcomes(repr)
```

**Example config:**
```json
{
  "matched_pair": {
    "joint_outcome_training": true,
    "use_mean_embedding_ite": true,
    "mean_ite_hidden_dim": 128,
    "mean_ite_dropout": 0.2,
    "cv_folds": 5
  }
}
```

See `examples/matched_pair_config.json` for sample configs.

### Chunk Encoder Selection

The `chunk_encoder` option selects the text encoding strategy for `PropensityMatchingModel`:

| `chunk_encoder` | Extractor | Description |
|-----------------|-----------|-------------|
| `"bert"` (default) | `HierarchicalTransformerExtractor` | Sentence boundaries, BERT [CLS] per sentence |
| `"gru"` | `HierarchicalGRUTransformerExtractor` | Overlapping token chunks, BiGRU + attention |

**GRU encoder options:**

| Option | Default | Purpose |
|--------|---------|---------|
| `gru_chunk_size` | `128` | Tokens per chunk |
| `gru_chunk_overlap` | `32` | Overlap between chunks |
| `gru_embedding_dim` | `128` | Word embedding dimension |
| `gru_hidden_dim` | `128` | BiGRU hidden dim per direction |
| `gru_num_layers` | `2` | Number of GRU layers |
| `gru_max_vocab_size` | `50000` | Maximum vocabulary size |
| `gru_min_word_freq` | `2` | Minimum word frequency |

The GRU encoder requires `fit_tokenizer()` to build vocabulary from training text. Both encoders share the same interface (`fit_tokenizer(texts)` or `init_extractor(texts)`).

**Example config:**
```json
{
  "matched_pair": {
    "chunk_encoder": "gru",
    "gru_chunk_size": 128,
    "gru_chunk_overlap": 32,
    "gru_embedding_dim": 128,
    "gru_hidden_dim": 128,
    "gru_num_layers": 2
  }
}
```

See `examples/matched_pair_mean_ite_config.json` for a complete example with GRU encoder.

### Gated Pool Extractors

The `bert_gated_pool` and `gru_pool` chunk encoders provide hierarchical text processing with gated attention pooling (tanh × sigmoid) for final document aggregation:

| `chunk_encoder` | Description |
|-----------------|-------------|
| `"bert_gated_pool"` | BERT [CLS] per chunk → Transformer → Gated Attention Pooling |
| `"gru_pool"` | BiGRU + attention per chunk → Transformer → Gated Attention Pooling |

**Architecture:**
```
Long Clinical Text
        ↓
Split into Overlapping Token Chunks (C chunks)
        ↓
[Per Chunk Encoding]
  bert_gated_pool: BERT → [CLS] or mean pool
  gru_pool: Word embeddings → BiGRU → Attention pooling
        ↓
Project to transformer_dim (C × transformer_dim)
        ↓
Positional Encoding + Transformer Layer(s)
  (chunks attend to each other for cross-chunk context)
        ↓
Gated Attention Pooling: h = tanh(V·x) ⊙ sigmoid(U·x)
        ↓
Output Projection → Final Representation
```

**Gated attention pooling** (from pathology AI) uses element-wise gating to suppress irrelevant chunks. The tanh branch captures content features while the sigmoid branch learns which chunks to attend to.

**`forward_with_instances()` method**: Both extractors expose chunk-level embeddings and attention weights for CLAM instance-level supervision (see below).

**GRU pool options:**

| Option | Default | Purpose |
|--------|---------|---------|
| `gru_pool_transformer_layers` | `2` | Transformer layers for cross-chunk context |
| `gru_pool_transformer_heads` | `4` | Attention heads in transformer |
| `gru_pool_transformer_dim` | `256` | Transformer hidden dimension |
| `gru_pool_gated_attention_dim` | `128` | Hidden dimension for gated pooling |

See `examples/matched_pair_gru_pool_clam_config.json` for a complete example.

### CLAM Instance-Level Supervision

CLAM (Clustering-constrained Attention Multiple Instance Learning) adds instance-level supervision to guide attention toward predictive chunks. When enabled with `bert_gated_pool` or `gru_pool` extractors, the model supervises top-attended chunks with document-level labels.

**How it works:**
1. Extract chunk embeddings and gated attention weights via `forward_with_instances()`
2. Select top-K chunks by attention weight for each document
3. Apply small instance classifier head to each selected chunk
4. Supervise with document-level labels (propensity in Stage 1, outcome in Stage 3)

**Key options:**

| Option | Default | Purpose |
|--------|---------|---------|
| `clam_enabled` | `False` | Enable CLAM instance supervision |
| `clam_num_instances` | `5` | Number of top-attended chunks to supervise |
| `clam_instance_hidden_dim` | `64` | Hidden dimension for instance classifier |
| `clam_instance_weight_stage1` | `0.5` | Weight for CLAM loss in Stage 1 (propensity) |
| `clam_instance_weight_stage3` | `0.5` | Weight for CLAM loss in Stage 3 (outcome) |

**Integration:**
- **Stage 1**: CLAM loss encourages attention on chunks predictive of treatment
- **Stage 3**: CLAM loss encourages attention on chunks predictive of outcome

**Example config:**
```json
{
  "matched_pair": {
    "chunk_encoder": "gru_pool",
    "clam_enabled": true,
    "clam_num_instances": 5,
    "clam_instance_hidden_dim": 64,
    "clam_instance_weight_stage1": 0.5,
    "clam_instance_weight_stage3": 0.5
  }
}
```

See `examples/matched_pair_gru_pool_clam_config.json` for a complete example.

## Key Training Options

| Option | Purpose |
|--------|---------|
| `stop_grad_propensity=True` | Prevent propensity from dominating feature learning |
| `attention_entropy_weight>0` | Encourage focused attention (gated_mil) |
| `gamma_rlearner>1.0` | Stronger tau signal |
| `gated_mil_hierarchical=True` | Token-level gated pooling |
| `gated_mil_use_mean_pooling=True` | Mean pool vs [CLS] |

## Output Files

```
output_dir/
├── applied_inference/
│   ├── predictions.parquet
│   ├── training_log.csv
│   ├── *_interpretations.json
│   └── psm_analysis/
└── plasmode_experiments/
```

## Dependencies

Core: torch, transformers, pandas, numpy, scikit-learn, tqdm, pyarrow
Optional: sentence-transformers, entmax, openai

## Development Notes

1. **Tokenizer fitting**: CNN/GRU/GRU-confounder require `fit_tokenizer()` before training
2. **R-Learner vs DragonNet**: R-Learner for treatment effect heterogeneity focus (detached nuisances)
3. **Long documents**: Use confounder/hierarchical_transformer/gated_mil extractors
4. **Interpretability**: `interpret_attention()` for sentence weights, `interpret_filters()` for CNN, `get_task_weights()` for task-specific confounder weighting
5. **PSM validation**: Use `run_psm_analysis()` to validate DragonNet estimates with traditional methods
