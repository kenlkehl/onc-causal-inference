# oci/models/hierarchical_llm_extractor.py
"""Hierarchical Frozen LLM feature extractor with two-level gated attention pooling.

Architecture:
    Raw text
      -> Pretrained tokenizer (full text)
      -> Overlapping token-based chunking (chunk_size tokens, chunk_overlap overlap)
      -> For each chunk: frozen LLM -> hidden states -> [optional downprojection]
         -> GatedAttentionPooling -> chunk_vector
      -> All chunk vectors -> GatedAttentionPooling (document-level) -> document_vector
      -> Projection MLP -> (batch, output_dim)

Cached mode (skip_llm=True):
    Pre-computed per-chunk hidden states
      -> Two-level GatedAttentionPooling -> Projection MLP
      (No LLM loaded)

DOES NOT require fit_tokenizer() — uses pretrained HuggingFace tokenizer.
"""

import logging
from typing import Optional, List, Dict, Any, Union

import torch
import torch.nn as nn

from .gated_attention_pooling import GatedAttentionPooling
from .text_chunking import chunk_token_ids
from .gpu_hidden_state_store import _get_hidden_size

logger = logging.getLogger(__name__)


class HierarchicalLLMExtractor(nn.Module):
    """Frozen LLM on chunks with two-level gated attention pooling.

    Same LLM as FrozenLLMPoolerExtractor, but processes overlapping chunks
    instead of the full document. Produces chunk-level vectors via token pooling,
    then aggregates chunks into a document vector via a second pooling layer.

    Args:
        model_name: HuggingFace model name.
        chunk_size: Number of tokens per chunk.
        chunk_overlap: Number of overlapping tokens between consecutive chunks.
        max_chunks: Maximum number of chunks per document.
        freeze_llm: If True, freeze all LLM parameters.
        gated_attention_dim: Hidden dimension for gated attention pooling.
        projection_dim: Output projection dimension.
        dropout: Dropout rate.
        gradient_checkpointing: Enable gradient checkpointing (when not frozen).
        downprojection_dim: Optional trainable linear projection before pooling.
        device: PyTorch device.
        skip_llm: If True, skip loading the LLM (cached mode).
        cached_hidden_size: Hidden size when skip_llm=True.
        chat_template_prompt: Optional chat template prompt for instruct models.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B-Base",
        chunk_size: int = 2048,
        chunk_overlap: int = 256,
        max_chunks: int = 16,
        freeze_llm: bool = True,
        gated_attention_dim: int = 128,
        projection_dim: int = 128,
        dropout: float = 0.1,
        gradient_checkpointing: bool = True,
        downprojection_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
        skip_llm: bool = False,
        cached_hidden_size: int = 0,
        chat_template_prompt: Optional[str] = None,
    ):
        super().__init__()

        self._device = device or torch.device('cpu')
        self._model_name = model_name
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._max_chunks = max_chunks
        self._freeze_llm = freeze_llm
        self._gated_attention_dim = gated_attention_dim
        self._projection_dim = projection_dim
        self._dropout = dropout
        self._gradient_checkpointing = gradient_checkpointing
        self._downprojection_dim = downprojection_dim
        self._skip_llm = skip_llm
        self._chat_template_prompt = chat_template_prompt

        if skip_llm:
            if cached_hidden_size <= 0:
                raise ValueError("cached_hidden_size must be > 0 when skip_llm=True")
            self._hidden_size = cached_hidden_size
            self._compute_dtype = None
            self._model = None
            self._tokenizer = None
            logger.info(
                f"HierarchicalLLMExtractor in CACHED mode "
                f"(skip_llm=True, hidden_size={cached_hidden_size})"
            )
        else:
            try:
                from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
            except ImportError:
                raise ImportError("transformers library is required.")

            logger.info(f"Initializing HierarchicalLLMExtractor with {model_name}")

            self._hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            self._hidden_size = _get_hidden_size(self._hf_config)

            from .gpu_hidden_state_store import _get_model_dtype
            self._compute_dtype = _get_model_dtype(self._hf_config) if freeze_llm else None
            logger.info(f"Loading pretrained weights from {model_name} "
                        f"(hidden_size={self._hidden_size}, dtype={self._compute_dtype})")

            self._model = AutoModelForCausalLM.from_pretrained(
                model_name, config=self._hf_config, trust_remote_code=True,
                torch_dtype=self._compute_dtype,
            )
            self._model = self._model.to(self._device)

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True,
                padding_side="right", truncation_side="left",
            )
            if self._tokenizer.pad_token is None:
                if self._tokenizer.eos_token is not None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
                    self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
                else:
                    self._tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    self._model.resize_token_embeddings(len(self._tokenizer))

            if chat_template_prompt is not None:
                if not hasattr(self._tokenizer, 'chat_template') or self._tokenizer.chat_template is None:
                    logger.warning(f"chat_template_prompt set but {model_name} has no chat_template. Disabling.")
                    self._chat_template_prompt = None
                else:
                    logger.info(f"Chat template prompt enabled ({len(chat_template_prompt)} chars)")

            if freeze_llm:
                for param in self._model.parameters():
                    param.requires_grad = False
                logger.info("LLM parameters frozen")
            else:
                if gradient_checkpointing:
                    self._model.gradient_checkpointing_enable()
                    logger.info("Gradient checkpointing enabled")

        # Optional downprojection
        if downprojection_dim is not None:
            self._downprojection = nn.Linear(self._hidden_size, downprojection_dim)
            effective_dim = downprojection_dim
        else:
            self._downprojection = None
            effective_dim = self._hidden_size
        self._effective_dim = effective_dim

        # Token-level pooling within each chunk
        self._token_pooling = GatedAttentionPooling(
            hidden_dim=effective_dim,
            attention_dim=gated_attention_dim,
        )

        # Document-level pooling across chunks
        self._chunk_pooling = GatedAttentionPooling(
            hidden_dim=effective_dim,
            attention_dim=gated_attention_dim,
        )

        # Projection MLP
        self._output_dim = projection_dim
        self._projection = nn.Sequential(
            nn.Linear(effective_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )

        if not skip_llm:
            logger.info(f"HierarchicalLLMExtractor initialized:")
            logger.info(f"  Model: {model_name} ({'frozen' if freeze_llm else 'trainable'})")
            logger.info(f"  Chunk: size={chunk_size}, overlap={chunk_overlap}, max={max_chunks}")
        if downprojection_dim is not None:
            logger.info(f"  Downprojection: {self._hidden_size} -> {downprojection_dim}")
        logger.info(f"  Effective dim: {effective_dim}, Output dim: {projection_dim}")

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def fit_tokenizer(self, texts: List[str]) -> None:
        """No-op: uses pretrained tokenizer."""
        pass

    def forward(self, texts_or_cached: Union[List[str], dict]) -> torch.Tensor:
        """Extract features from texts or pre-computed hidden states.

        Args:
            texts_or_cached: List[str] texts, or dict with hierarchical cached data.

        Returns:
            Feature tensor of shape (batch_size, output_dim)
        """
        if isinstance(texts_or_cached, dict) and 'hierarchical_cached' in texts_or_cached:
            return self._forward_cached_hierarchical(texts_or_cached)
        if isinstance(texts_or_cached, dict) and 'cached_hidden_states' in texts_or_cached:
            return self._forward_cached_hierarchical(texts_or_cached)
        return self._forward_from_texts(texts_or_cached)

    def _prepare_text(self, text: str) -> str:
        """Wrap text in chat template if configured."""
        if self._chat_template_prompt is None:
            return text
        messages = [{"role": "user", "content": f"{self._chat_template_prompt}{text}"}]
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    def _forward_from_texts(self, texts: List[str]) -> torch.Tensor:
        """Extract features from raw texts."""
        if self._model is None:
            raise RuntimeError("LLM not loaded (skip_llm=True). Provide cached hidden states.")

        # Tokenize full texts (no truncation here; chunking handles length)
        max_total_tokens = self._chunk_size * self._max_chunks
        prepared = [self._prepare_text(t) for t in texts]

        # Tokenize all texts to get token IDs, then chunk
        all_chunks_input_ids = []  # flat list of chunk token ID tensors
        all_chunks_attention_mask = []
        sample_chunk_counts = []  # how many chunks each sample has

        for text in prepared:
            encoding = self._tokenizer(
                text, truncation=True, max_length=max_total_tokens,
                return_tensors="pt", padding=False,
            )
            token_ids = encoding['input_ids'][0].tolist()
            chunks = chunk_token_ids(
                token_ids, self._chunk_size, self._chunk_overlap, self._max_chunks
            )
            sample_chunk_counts.append(len(chunks))

            for chunk_ids in chunks:
                # Pad each chunk to chunk_size for uniform LLM batching
                padded = chunk_ids + [self._tokenizer.pad_token_id] * (self._chunk_size - len(chunk_ids))
                mask = [1] * len(chunk_ids) + [0] * (self._chunk_size - len(chunk_ids))
                all_chunks_input_ids.append(torch.tensor(padded[:self._chunk_size], dtype=torch.long))
                all_chunks_attention_mask.append(torch.tensor(mask[:self._chunk_size], dtype=torch.long))

        # Stack all chunks into a single batch for LLM forward
        total_chunks = len(all_chunks_input_ids)
        if total_chunks == 0:
            # Edge case: empty batch
            return torch.zeros(len(texts), self._output_dim, device=self._device)

        chunk_input_ids = torch.stack(all_chunks_input_ids).to(self._device)  # (total_chunks, chunk_size)
        chunk_attention_mask = torch.stack(all_chunks_attention_mask).to(self._device)

        # LLM forward on all chunks at once
        if self._freeze_llm:
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=self._compute_dtype, enabled=self._device.type == "cuda"):
                    outputs = self._model.model(
                        input_ids=chunk_input_ids,
                        attention_mask=chunk_attention_mask,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    hidden_states = outputs.last_hidden_state
            from .hidden_state_cache import _sanitize_hidden_states
            hidden_states = _sanitize_hidden_states(hidden_states, context="hierarchical_live")
            hidden_states = hidden_states.detach().float()
        else:
            outputs = self._model.model(
                input_ids=chunk_input_ids,
                attention_mask=chunk_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.last_hidden_state
            from .hidden_state_cache import _sanitize_hidden_states
            hidden_states = _sanitize_hidden_states(hidden_states, context="hierarchical_live_trainable")
            if hidden_states.dtype != torch.float32:
                hidden_states = hidden_states.float()

        chunk_attention_mask = chunk_attention_mask.float()

        # Apply two-level pooling
        return self._two_level_pool(
            hidden_states, chunk_attention_mask, sample_chunk_counts
        )

    def _forward_cached_hierarchical(self, batch: dict) -> torch.Tensor:
        """Forward from pre-computed per-chunk hidden states.

        Expects batch dict with:
          - 'chunk_hidden_states': (total_chunks, chunk_len, hidden_size) float32
          - 'chunk_attention_mask': (total_chunks, chunk_len) float32
          - 'sample_chunk_counts': List[int] — chunks per sample
        """
        hidden_states = batch.get('chunk_hidden_states', batch.get('cached_hidden_states'))
        attention_mask = batch.get('chunk_attention_mask', batch.get('cached_attention_mask'))
        sample_chunk_counts = batch.get('sample_chunk_counts', [hidden_states.shape[0]])

        if hidden_states.dtype != torch.float32:
            hidden_states = hidden_states.float()
        if attention_mask.dtype != torch.float32:
            attention_mask = attention_mask.float()

        from .hidden_state_cache import _sanitize_hidden_states
        hidden_states = _sanitize_hidden_states(hidden_states, context="hierarchical_cached")

        return self._two_level_pool(hidden_states, attention_mask, sample_chunk_counts)

    def _two_level_pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        sample_chunk_counts: List[int],
    ) -> torch.Tensor:
        """Two-level pooling: tokens→chunks, chunks→document.

        Args:
            hidden_states: (total_chunks, chunk_len, hidden_size)
            attention_mask: (total_chunks, chunk_len)
            sample_chunk_counts: Number of chunks per sample in the batch.

        Returns:
            (batch_size, output_dim)
        """
        # Optional downprojection
        if self._downprojection is not None:
            hidden_states = self._downprojection(hidden_states)

        # Token-level pooling per chunk: (total_chunks, effective_dim)
        chunk_vectors, self._last_token_weights = self._token_pooling(
            hidden_states, attention_mask=attention_mask
        )

        # Reshape into per-sample groups and pad for batch pooling
        B = len(sample_chunk_counts)
        max_chunks = max(sample_chunk_counts)
        device = chunk_vectors.device

        padded_chunks = torch.zeros(B, max_chunks, self._effective_dim, device=device)
        chunk_mask = torch.zeros(B, max_chunks, device=device)

        offset = 0
        for b, count in enumerate(sample_chunk_counts):
            padded_chunks[b, :count] = chunk_vectors[offset:offset + count]
            chunk_mask[b, :count] = 1.0
            offset += count

        # Document-level pooling: (B, effective_dim)
        doc_vector, self._last_chunk_weights = self._chunk_pooling(
            padded_chunks, attention_mask=chunk_mask
        )

        # Project to output dim: (B, projection_dim)
        features = self._projection(doc_vector)

        return features

    def get_state(self) -> Dict[str, Any]:
        return {
            'extractor_type': 'hierarchical_llm',
            'model_name': self._model_name,
            'chunk_size': self._chunk_size,
            'chunk_overlap': self._chunk_overlap,
            'max_chunks': self._max_chunks,
            'freeze_llm': self._freeze_llm,
            'gated_attention_dim': self._gated_attention_dim,
            'projection_dim': self._projection_dim,
            'dropout': self._dropout,
            'downprojection_dim': self._downprojection_dim,
            'hidden_size': self._hidden_size,
            'output_dim': self._output_dim,
            'skip_llm': self._skip_llm,
            'chat_template_prompt': self._chat_template_prompt,
        }

    def get_num_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable, 'frozen': total - trainable}

    def to(self, device):
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        return super().to(device)

    def interpret_attention(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Get chunk-level and token-level attention weights for interpretability."""
        if self._model is None:
            raise RuntimeError("interpret_attention() not available in cached mode.")

        self.eval()
        with torch.no_grad():
            self._forward_from_texts(texts)

        results = []
        chunk_weights = self._last_chunk_weights  # (B, max_chunks)
        for b in range(len(texts)):
            results.append({
                'chunk_attention_weights': chunk_weights[b].cpu().numpy().tolist(),
            })
        return results
