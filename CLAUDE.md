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

See `examples/matched_pair_config.json` for sample configs.

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
