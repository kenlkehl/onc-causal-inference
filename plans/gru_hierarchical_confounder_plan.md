# GRU-Based Hierarchical Confounder Extractor

## Motivation

The current `HierarchicalConfounderExtractor` uses pretrained BERT for token encoding. This has issues:
1. Explicit confounders encoded with BERT may not align semantically with clinical text
2. Latent confounders start random but attend to BERT-encoded sentences - mismatched spaces
3. No training signal flowing back through frozen BERT encoder

**New approach**: Everything learns from scratch via the causal objective.

## Proposed Architecture

```
Long Clinical Text
        ↓
Split into Sentences (S sentences)
        ↓
Custom WordTokenizer (trained on dataset, like CNN extractor)
        ↓
Randomly Initialized Token Embeddings (learnable)
        ↓
Per-Sentence BiGRU with Attention Pooling → S sentence embeddings
        ↓
Latent Queries (K randomly initialized, learnable)
        ↓
Sentence-Level Sparse Cross-Attention (entmax)
        ↓
Token-Level Cross-Attention (within attended sentences)
        ↓
K Confounder Representations → Causal Head
```

## Key Design Decisions

### 1. Custom Tokenizer (like CNN/GRU extractors)
- Use `WordTokenizer` from `cdt/models/cnn_extractor.py`
- Requires `fit_tokenizer(texts)` before training
- Learns vocabulary from training data
- No pretrained anything

### 2. Learnable Token Embeddings
```python
self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
# Randomly initialized, learns during training
```

### 3. Per-Sentence BiGRU Encoder
For each sentence:
```python
# Tokenize sentence
tokens = tokenizer.encode(sentence)  # (L,)

# Embed tokens
token_embs = self.embedding(tokens)  # (L, D)

# BiGRU encoding
gru_out, _ = self.gru(token_embs)  # (L, 2*hidden)

# Attention pooling for sentence embedding
attn_weights = softmax(self.attn_query @ gru_out.T)  # (L,)
sentence_emb = attn_weights @ gru_out  # (2*hidden,)
```

Keep both:
- `token_embeddings`: (L, 2*hidden) for token-level attention
- `sentence_embedding`: (2*hidden,) for sentence-level attention

### 4. Latent Confounders Only
- No explicit confounders (they require pretrained encoder to make sense)
- K latent confounders: `nn.Parameter(torch.randn(K, hidden_dim) * 0.1)`
- All randomly initialized, all learnable
- Learn what to attend to purely from causal loss

### 5. Two-Level Sparse Attention
Same as current hierarchical:
1. Sentence-level: latents attend to sentence embeddings (sparse via entmax)
2. Token-level: within high-weight sentences, latents attend to tokens

## Implementation Plan

### Files to Modify/Create

| File | Action | Description |
|------|--------|-------------|
| `cdt/models/confounder_extractor.py` | Add class | `GRUHierarchicalConfounderExtractor` |
| `cdt/models/causal_text.py` | Modify | Wire up new extractor type |
| `cdt/config.py` | Modify | Add config options |

### New Class: `GRUHierarchicalConfounderExtractor`

```python
class GRUHierarchicalConfounderExtractor(nn.Module):
    """
    Hierarchical confounder extractor with GRU sentence encoding.

    Everything learns from scratch - no pretrained models.
    Uses custom tokenizer trained on dataset.
    """

    def __init__(
        self,
        # Tokenizer (set via fit_tokenizer)
        vocab_size: int = 50000,
        embedding_dim: int = 128,
        min_word_freq: int = 2,
        max_length: int = 128,  # per sentence
        # GRU encoder
        gru_hidden_dim: int = 128,
        gru_num_layers: int = 1,
        gru_bidirectional: bool = True,
        gru_dropout: float = 0.1,
        # Confounders
        num_latent_confounders: int = 8,
        # Attention
        num_attention_heads: int = 4,
        sparse_attention: bool = True,
        sparse_alpha: float = 1.5,
        # Architecture
        max_sentences: int = 100,
        value_dim: int = 128,
        dropout: float = 0.1,
        device: Optional[torch.device] = None
    ):
        super().__init__()

        # Tokenizer (fitted later)
        self.tokenizer = WordTokenizer(
            max_vocab_size=vocab_size,
            min_freq=min_word_freq,
            max_length=max_length
        )
        self._tokenizer_fitted = False

        # Token embeddings (randomly initialized)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Per-sentence BiGRU
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_num_layers,
            bidirectional=gru_bidirectional,
            dropout=gru_dropout if gru_num_layers > 1 else 0,
            batch_first=True
        )

        # Attention pooling for sentences
        gru_output_dim = gru_hidden_dim * (2 if gru_bidirectional else 1)
        self.sentence_attn = nn.Linear(gru_output_dim, 1)

        # Latent confounders (randomly initialized, learnable)
        self.latent_confounders = nn.Parameter(
            torch.randn(num_latent_confounders, gru_output_dim) * 0.1
        )

        # Cross-attention projections
        self.W_q = nn.Linear(gru_output_dim, gru_output_dim, bias=False)
        self.W_k = nn.Linear(gru_output_dim, gru_output_dim, bias=False)
        self.W_v = nn.Linear(gru_output_dim, gru_output_dim, bias=False)

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(gru_output_dim * num_latent_confounders, value_dim * 2),
            nn.LayerNorm(value_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(value_dim * 2, value_dim),
            nn.LayerNorm(value_dim)
        )

    def fit_tokenizer(self, texts: List[str]) -> 'GRUHierarchicalConfounderExtractor':
        """Fit tokenizer on training texts. REQUIRED before training."""
        # Flatten all sentences from all documents
        all_sentences = []
        for text in texts:
            sentences = split_into_sentences(text, self.max_sentences)
            all_sentences.extend(sentences)

        self.tokenizer.fit(all_sentences)
        self._tokenizer_fitted = True
        return self

    def _encode_sentence(self, sentence: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a single sentence with GRU.

        Returns:
            token_embeddings: (L, gru_output_dim)
            sentence_embedding: (gru_output_dim,)
        """
        # Tokenize
        token_ids = self.tokenizer.encode(sentence)  # (L,)
        token_ids = token_ids.to(self._device)

        # Embed
        token_embs = self.embedding(token_ids)  # (L, embedding_dim)

        # GRU encode
        gru_out, _ = self.gru(token_embs.unsqueeze(0))  # (1, L, gru_output_dim)
        gru_out = gru_out.squeeze(0)  # (L, gru_output_dim)

        # Attention pooling for sentence embedding
        attn_scores = self.sentence_attn(gru_out).squeeze(-1)  # (L,)
        attn_weights = F.softmax(attn_scores, dim=0)  # (L,)
        sentence_emb = (attn_weights.unsqueeze(-1) * gru_out).sum(dim=0)  # (gru_output_dim,)

        return gru_out, sentence_emb

    def forward(self, texts: List[str]) -> torch.Tensor:
        """Extract confounder representations."""
        if not self._tokenizer_fitted:
            raise RuntimeError("Must call fit_tokenizer() before forward()")

        batch_results = []

        for text in texts:
            # Split into sentences
            sentences = split_into_sentences(text, self.max_sentences)
            if not sentences:
                sentences = [text[:500]]

            # Encode each sentence
            token_embs_list = []
            sentence_embs = []
            for sent in sentences:
                tok_embs, sent_emb = self._encode_sentence(sent)
                token_embs_list.append(tok_embs)
                sentence_embs.append(sent_emb)

            sentence_embs = torch.stack(sentence_embs)  # (S, D)

            # Sentence-level sparse attention
            sentence_weights = self._compute_sentence_attention(
                self.latent_confounders, sentence_embs
            )  # (K, S)

            # Token-level attention within sentences
            confounder_reprs = []
            for k in range(self.num_latent_confounders):
                weighted_repr = torch.zeros(self.gru_output_dim, device=self._device)

                for s, sent_tokens in enumerate(token_embs_list):
                    weight = sentence_weights[k, s].item()
                    if weight < 1e-6:
                        continue  # Skip zero-weight sentences

                    # Token attention
                    sent_repr = self._compute_token_attention(
                        self.latent_confounders[k], sent_tokens
                    )
                    weighted_repr = weighted_repr + sentence_weights[k, s] * sent_repr

                confounder_reprs.append(weighted_repr)

            doc_confounders = torch.stack(confounder_reprs)  # (K, D)
            batch_results.append(doc_confounders)

        # Stack and project
        batch_confounders = torch.stack(batch_results)  # (B, K, D)
        flat = batch_confounders.reshape(len(texts), -1)  # (B, K*D)
        features = self.output_projection(flat)  # (B, value_dim)

        return features
```

### Config Options

```python
# In ModelArchitectureConfig

# GRU-based hierarchical confounder extractor
confounder_use_gru: bool = False  # Use GRU instead of BERT for sentence encoding
confounder_gru_embedding_dim: int = 128
confounder_gru_hidden_dim: int = 128
confounder_gru_num_layers: int = 1
confounder_gru_bidirectional: bool = True
confounder_gru_max_vocab: int = 50000
confounder_gru_min_word_freq: int = 2
```

### Integration in CausalText

```python
elif self.feature_extractor_type == "confounder":
    if confounder_use_gru:
        # GRU-based hierarchical (learns from scratch)
        self.feature_extractor = GRUHierarchicalConfounderExtractor(
            vocab_size=confounder_gru_max_vocab,
            embedding_dim=confounder_gru_embedding_dim,
            gru_hidden_dim=confounder_gru_hidden_dim,
            num_latent_confounders=confounder_num_latents,
            # ... other params
        )
    elif confounder_hierarchical:
        # BERT-based hierarchical
        self.feature_extractor = HierarchicalConfounderExtractor(...)
    else:
        # Sentence-level only
        self.feature_extractor = ConfounderExtractor(...)
```

## Advantages of This Approach

1. **Everything learns together** - embeddings, GRU, confounders all optimize for causal loss
2. **No pretrained model dependency** - works on any domain
3. **Consistent embedding space** - confounders and tokens live in same learned space
4. **Lighter weight** - GRU is faster than BERT
5. **Requires fit_tokenizer()** - explicit data-dependent initialization step

## Potential Concerns

1. **Slower convergence** - learning from scratch needs more epochs
2. **Need enough data** - vocabulary and embeddings need training signal
3. **No transfer learning** - can't leverage pretrained clinical knowledge

## Testing Plan

1. Verify GRU sentence encoding produces reasonable embeddings
2. Test sparse attention focuses on relevant sentences
3. Compare to BERT-based hierarchical on synthetic data
4. Check training curves - should see gradual improvement

## Implementation Order

1. Add `GRUHierarchicalConfounderExtractor` class
2. Add config options
3. Wire up in `CausalText`
4. Update experiment script with `--use-gru` flag
5. Run comparison experiments
