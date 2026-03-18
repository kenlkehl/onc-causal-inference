# oci/models/frozen_llm_pooler_extractor.py
"""Frozen LLM + Gated Attention Pooler feature extractor.

This module implements a feature extractor that uses a pretrained decoder-only LLM
(e.g., Qwen3-0.6B-Base) with frozen weights and applies GatedAttentionPooling over
ALL token hidden states to create a rich patient-level representation.

Architecture:
    Clinical Text (full document, no chunking)
         |
    Tokenize with pretrained HF tokenizer (right-padded)
         |
    Decoder-only LLM (pretrained, frozen by default, autocast float16)
         |
    ALL token hidden states from final layer: (batch, seq_len, hidden_size)
         |
    [Optional] Trainable downprojection: Linear(hidden_size -> downprojection_dim)
         |
    GatedAttentionPooling (with attention_mask for padding):
      g = tanh(V(h)) * sigmoid(U(h))  ->  scores = v(W(LN(g)))  ->  softmax  ->  weighted sum
      Output: (batch, effective_dim)  where effective_dim = downprojection_dim or hidden_size
         |
    2-layer MLP: Linear->LN->GELU->Dropout->Linear->LN
         |
    Output: (batch, projection_dim)

Cached mode (skip_llm=True):
    Pre-computed hidden states loaded from cache
         |
    GatedAttentionPooling -> projection
    (No LLM loaded, ~200K trainable params only)

DOES NOT require fit_tokenizer() - uses pretrained tokenizer from HuggingFace.
"""

import logging
from typing import Optional, List, Dict, Any, Tuple, Union

import torch
import torch.nn as nn

from .gated_attention_pooling import GatedAttentionPooling
from .gpu_hidden_state_store import _get_hidden_size

logger = logging.getLogger(__name__)


class FrozenLLMPoolerExtractor(nn.Module):
    """
    Frozen LLM + Gated Attention Pooler feature extractor.

    Loads a pretrained decoder-only LLM, extracts all token hidden states,
    and applies GatedAttentionPooling to produce a single document vector.
    The LLM is frozen by default so only the pooling + projection layers train.

    When skip_llm=True (cached mode), the LLM is not loaded at all. Instead,
    pre-computed hidden states are provided directly via forward_cached().

    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen3-0.6B-Base")
        max_length: Maximum sequence length
        freeze_llm: If True, freeze all LLM parameters
        gated_attention_dim: Hidden dimension for gated attention pooling
        projection_dim: Output projection dimension
        dropout: Dropout rate for projection layers
        gradient_checkpointing: Enable gradient checkpointing (only when not frozen)
        downprojection_dim: If set, trainable linear projection from hidden_size to this dim
            before pooling. Reduces memory for trainable layers.
        device: PyTorch device
        skip_llm: If True, skip loading the LLM entirely (cached mode)
        cached_hidden_size: Hidden size to use when skip_llm=True
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B-Base",
        max_length: int = 8192,
        freeze_llm: bool = True,
        gated_attention_dim: int = 128,
        projection_dim: int = 128,
        dropout: float = 0.1,
        gradient_checkpointing: bool = True,
        downprojection_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
        skip_llm: bool = False,
        cached_hidden_size: int = 0,
    ):
        super().__init__()

        self._device = device or torch.device('cpu')
        self._model_name = model_name
        self._max_length = max_length
        self._freeze_llm = freeze_llm
        self._gated_attention_dim = gated_attention_dim
        self._projection_dim = projection_dim
        self._dropout = dropout
        self._gradient_checkpointing = gradient_checkpointing
        self._downprojection_dim = downprojection_dim
        self._skip_llm = skip_llm

        if skip_llm:
            # Cached mode: no LLM loaded, use provided hidden_size
            if cached_hidden_size <= 0:
                raise ValueError(
                    "cached_hidden_size must be > 0 when skip_llm=True"
                )
            self._hidden_size = cached_hidden_size
            self._model = None
            self._tokenizer = None
            logger.info(
                f"FrozenLLMPoolerExtractor in CACHED mode "
                f"(skip_llm=True, hidden_size={cached_hidden_size})"
            )
        else:
            # Standard mode: load LLM
            try:
                from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
            except ImportError:
                raise ImportError(
                    "transformers library is required for FrozenLLMPoolerExtractor. "
                    "Install with: pip install transformers"
                )

            logger.info(f"Initializing FrozenLLMPoolerExtractor with {model_name}")

            # Load config and model (always pretrained)
            self._hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            self._hidden_size = _get_hidden_size(self._hf_config)

            logger.info(f"Loading pretrained weights from {model_name} (hidden_size={self._hidden_size})")
            # Load to CPU first, then move to target device.  Avoids meta
            # tensors that device_map + remove_hook_from_module can leave
            # for models with tied weights (e.g. lm_head tied to embed_tokens).
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name, config=self._hf_config, trust_remote_code=True,
                torch_dtype=torch.float16 if freeze_llm else None,
            )
            self._model = self._model.to(self._device)

            # Load pretrained tokenizer (right padding for pooling over all tokens)
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                padding_side="right",
                truncation_side="left",  # Keep end of long documents
            )

            # Ensure pad token exists
            if self._tokenizer.pad_token is None:
                if self._tokenizer.eos_token is not None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
                    self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
                else:
                    self._tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    self._model.resize_token_embeddings(len(self._tokenizer))

            logger.info(f"Tokenizer vocab size: {len(self._tokenizer)}")
            logger.info(f"Pad token: {self._tokenizer.pad_token} (id={self._tokenizer.pad_token_id})")

            # Freeze LLM parameters if requested
            if freeze_llm:
                for param in self._model.parameters():
                    param.requires_grad = False
                logger.info("LLM parameters frozen")
            else:
                # Enable gradient checkpointing only when LLM is trainable
                if gradient_checkpointing:
                    self._model.gradient_checkpointing_enable()
                    logger.info("Gradient checkpointing enabled")

        # Trainable downprojection from hidden_size to a smaller dim (optional)
        if downprojection_dim is not None:
            self._downprojection = nn.Linear(self._hidden_size, downprojection_dim)
            effective_dim = downprojection_dim
        else:
            self._downprojection = None
            effective_dim = self._hidden_size
        self._effective_dim = effective_dim

        # Gated attention pooling over token hidden states (or their projections)
        self._pooling = GatedAttentionPooling(
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

        if skip_llm:
            logger.info(f"FrozenLLMPoolerExtractor (cached mode) initialized:")
            logger.info(f"  Hidden size: {self._hidden_size}")
        else:
            logger.info(f"FrozenLLMPoolerExtractor initialized:")
            logger.info(f"  Model: {model_name} (pretrained, {'frozen' if freeze_llm else 'trainable'})")
            logger.info(f"  Hidden size: {self._hidden_size}")
            logger.info(f"  Max length: {max_length}")
        if downprojection_dim is not None:
            logger.info(f"  Downprojection: {self._hidden_size} -> {downprojection_dim} (trainable)")
        logger.info(f"  Effective dim (pooling input): {effective_dim}")
        logger.info(f"  Gated attention dim: {gated_attention_dim}")
        logger.info(f"  Output dim: {projection_dim}")

    @property
    def output_dim(self) -> int:
        """Return the output dimension of this feature extractor."""
        return self._output_dim

    @property
    def hidden_size(self) -> int:
        """Return the hidden size of the underlying LLM."""
        return self._hidden_size

    def forward(self, texts_or_cached: Union[List[str], dict]) -> torch.Tensor:
        """
        Extract features from texts or pre-computed hidden states.

        Dispatches to _forward_from_texts() or forward_cached() based on input type.

        Args:
            texts_or_cached: Either a list of text strings, or a dict with
                'cached_hidden_states' and 'cached_attention_mask' keys.

        Returns:
            Feature tensor of shape (batch_size, output_dim)
        """
        if isinstance(texts_or_cached, dict) and 'cached_hidden_states' in texts_or_cached:
            return self.forward_cached(
                hidden_states=texts_or_cached['cached_hidden_states'],
                attention_mask=texts_or_cached['cached_attention_mask'],
            )
        return self._forward_from_texts(texts_or_cached)

    def _forward_from_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Extract features from raw texts using the LLM.

        Args:
            texts: List of document texts

        Returns:
            Feature tensor of shape (batch_size, output_dim)
        """
        if self._model is None:
            raise RuntimeError(
                "LLM is not loaded (skip_llm=True). Use forward_cached() "
                "or provide cached hidden states via a dict input."
            )

        # Tokenize with right padding
        encoding = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )

        input_ids = encoding['input_ids'].to(self._device, non_blocking=True)
        attention_mask = encoding['attention_mask'].to(self._device, non_blocking=True)

        # Forward through the LLM
        if self._freeze_llm:
            with torch.no_grad():
                # Use autocast for memory-efficient frozen LLM forward pass
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=self._device.type == "cuda"):
                    # Use base transformer (model.model) to skip lm_head logits
                    # computation — saves memory and avoids tied-weight issues.
                    outputs = self._model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
            # Detach from LLM graph and cast to float32
            hidden_states = hidden_states.detach().float()
        else:
            # Use base transformer (model.model) to skip lm_head logits
            outputs = self._model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.last_hidden_state
            # Cast BFloat16 -> Float32 if needed (Qwen3 uses BFloat16)
            if hidden_states.dtype != torch.float32:
                hidden_states = hidden_states.float()

        # Trainable downprojection (has gradients even when LLM is frozen)
        if self._downprojection is not None:
            hidden_states = self._downprojection(hidden_states)

        # Gated attention pooling with attention mask
        pooled, self._last_attention_weights = self._pooling(
            hidden_states, attention_mask=attention_mask
        )  # pooled: (batch, effective_dim)

        # Project to output dimension
        features = self._projection(pooled)

        return features

    def forward_cached(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract features from pre-computed hidden states (cached mode).

        Runs only the trainable layers: pooling -> projection.

        Args:
            hidden_states: Pre-computed hidden states (batch, seq_len, hidden_size) float32
            attention_mask: Attention mask (batch, seq_len) float32

        Returns:
            Feature tensor of shape (batch_size, output_dim)
        """
        # Ensure float32 (defensive guard against float16 from cache)
        if hidden_states.dtype != torch.float32:
            hidden_states = hidden_states.float()
        if attention_mask.dtype != torch.float32:
            attention_mask = attention_mask.float()

        assert hidden_states.dtype == torch.float32, (
            f"forward_cached: hidden_states is {hidden_states.dtype} after .float() cast"
        )

        # Validate hidden state dimension matches model expectation
        actual_dim = hidden_states.shape[-1]
        expected_dim = self._hidden_size
        if actual_dim != expected_dim:
            raise ValueError(
                f"Cached hidden state dim ({actual_dim}) does not match "
                f"model's expected hidden_size ({expected_dim}). "
                f"This usually means the cache was built with a different "
                f"downprojection_dim than the model expects."
            )

        # Trainable downprojection (if configured)
        if self._downprojection is not None:
            hidden_states = self._downprojection(hidden_states)

        # Gated attention pooling with attention mask
        pooled, self._last_attention_weights = self._pooling(
            hidden_states, attention_mask=attention_mask
        )  # pooled: (batch, effective_dim)

        # Project to output dimension
        features = self._projection(pooled)

        return features

    def get_state(self) -> Dict[str, Any]:
        """Get extractor state for checkpoint saving."""
        return {
            'model_name': self._model_name,
            'max_length': self._max_length,
            'freeze_llm': self._freeze_llm,
            'gated_attention_dim': self._gated_attention_dim,
            'projection_dim': self._projection_dim,
            'dropout': self._dropout,
            'gradient_checkpointing': self._gradient_checkpointing,
            'downprojection_dim': self._downprojection_dim,
            'hidden_size': self._hidden_size,
            'output_dim': self._output_dim,
            'skip_llm': self._skip_llm,
        }

    def to(self, device):
        """Override to track device properly."""
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        return super().to(device)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        if not self._freeze_llm and self._model is not None:
            self._model.gradient_checkpointing_enable()
            self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        if not self._freeze_llm and self._model is not None:
            self._model.gradient_checkpointing_disable()
            self._gradient_checkpointing = False

    def get_num_parameters(self) -> Dict[str, int]:
        """Get parameter counts for the model."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params,
        }

    def interpret_attention(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Get attention weights for interpretability.

        Args:
            texts: List of document texts

        Returns:
            List of dicts with token-level attention weights per document
        """
        if self._model is None:
            raise RuntimeError(
                "interpret_attention() is not available in cached mode (skip_llm=True). "
                "Use a non-cached model for interpretability analysis."
            )

        self.eval()
        with torch.no_grad():
            self._forward_from_texts(texts)

        results = []
        weights = self._last_attention_weights  # (batch, seq_len)

        for i, text in enumerate(texts):
            encoding = self._tokenizer(
                text,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            )
            tokens = self._tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
            w = weights[i, :len(tokens)].cpu().numpy()

            # Get top attended tokens
            top_indices = w.argsort()[::-1][:20]
            top_tokens = [(tokens[idx], float(w[idx])) for idx in top_indices]

            results.append({
                'tokens': tokens,
                'attention_weights': w.tolist(),
                'top_tokens': top_tokens,
            })

        return results

    def get_attention_weights(self, texts: List[str]) -> torch.Tensor:
        """
        Get raw attention weights from gated attention pooling.

        Args:
            texts: List of document texts

        Returns:
            Attention weight tensor of shape (batch, seq_len)
        """
        if self._model is None:
            raise RuntimeError(
                "get_attention_weights() is not available in cached mode (skip_llm=True). "
                "Use a non-cached model for interpretability analysis."
            )

        self.eval()
        with torch.no_grad():
            self._forward_from_texts(texts)
        return self._last_attention_weights
