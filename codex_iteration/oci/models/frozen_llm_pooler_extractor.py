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
from .token_windowing import tokenize_with_document_window, validate_document_window

logger = logging.getLogger(__name__)


class MultiSlotGatedAttentionPooling(nn.Module):
    """Gated attention pooling with several learned attention slots."""

    def __init__(self, hidden_dim: int, attention_dim: int, num_slots: int):
        super().__init__()
        self.num_slots = num_slots
        self.V = nn.Linear(hidden_dim, attention_dim)
        self.U = nn.Linear(hidden_dim, attention_dim)
        self.W = nn.Linear(attention_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, num_slots, bias=False)
        self.layer_norm = nn.LayerNorm(attention_dim)

    def forward(
        self,
        chunk_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tanh_branch = torch.tanh(self.V(chunk_embeddings))
        sigmoid_branch = torch.sigmoid(self.U(chunk_embeddings))
        gated = self.layer_norm(tanh_branch * sigmoid_branch)
        scores = self.v(self.W(gated))

        if chunk_embeddings.dim() != 3:
            raise ValueError("Multi-slot pooling expects a batched 3D tensor")

        scores = scores.transpose(1, 2)  # (B, slots, seq_len)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.unsqueeze(1) == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.bmm(weights, chunk_embeddings)
        return pooled.flatten(start_dim=1), weights


class TokenCNNGatedPooling(nn.Module):
    """Convolutional token encoder followed by gated attention pooling."""

    def __init__(
        self,
        hidden_dim: int,
        conv_dim: int,
        attention_dim: int,
        kernel_sizes: Tuple[int, ...] = (1, 3, 5, 7),
        dropout: float = 0.1,
    ):
        super().__init__()
        if not kernel_sizes:
            raise ValueError("kernel_sizes must contain at least one value")
        self.kernel_sizes = tuple(int(k) for k in kernel_sizes)
        self.convs = nn.ModuleList(
            nn.Conv1d(hidden_dim, conv_dim, kernel_size=k, padding=k // 2)
            for k in self.kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.output_dim = conv_dim * len(self.kernel_sizes)
        self.V = nn.Linear(self.output_dim, attention_dim)
        self.U = nn.Linear(self.output_dim, attention_dim)
        self.W = nn.Linear(attention_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)
        self.layer_norm = nn.LayerNorm(attention_dim)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if token_embeddings.dim() != 3:
            raise ValueError("TokenCNNGatedPooling expects a batched 3D tensor")

        x = token_embeddings.transpose(1, 2)
        conv_outputs = []
        for conv in self.convs:
            z = torch.nn.functional.gelu(conv(x))
            conv_outputs.append(z)
        token_features = torch.cat(conv_outputs, dim=1).transpose(1, 2)
        token_features = self.dropout(token_features)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(dtype=token_features.dtype)
            token_features = token_features * mask

        tanh_branch = torch.tanh(self.V(token_features))
        sigmoid_branch = torch.sigmoid(self.U(token_features))
        gated = self.layer_norm(tanh_branch * sigmoid_branch)
        scores = self.v(self.W(gated)).squeeze(-1)

        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.bmm(weights.unsqueeze(1), token_features).squeeze(1)
        return pooled, weights


class MaskedStatPooling(nn.Module):
    """Generic masked pooling over frozen neural token states.

    This deliberately does not inspect token strings. It preserves global and
    coarse positional evidence from the hidden-state sequence so the downstream
    forest can discover useful effect-modifier dimensions without a concept
    parser.
    """

    def __init__(self, hidden_dim: int, num_segments: int = 2):
        super().__init__()
        if num_segments < 1:
            raise ValueError("num_segments must be >= 1")
        self.hidden_dim = hidden_dim
        self.num_segments = num_segments
        self.output_dim = hidden_dim * (6 + 2 * num_segments)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if token_embeddings.dim() != 3:
            raise ValueError("MaskedStatPooling expects a batched 3D tensor")

        batch_size, seq_len, hidden_dim = token_embeddings.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Expected hidden_dim={self.hidden_dim}, got {hidden_dim}"
            )

        if attention_mask is None:
            lengths = torch.full(
                (batch_size,),
                seq_len,
                dtype=torch.long,
                device=token_embeddings.device,
            )
            weights = torch.full(
                (batch_size, seq_len),
                1.0 / max(seq_len, 1),
                dtype=token_embeddings.dtype,
                device=token_embeddings.device,
            )
        else:
            lengths = attention_mask.to(dtype=torch.long).sum(dim=1)
            weights = attention_mask.to(dtype=token_embeddings.dtype)
            denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            weights = weights / denom

        pooled_rows = []
        zero = token_embeddings.new_zeros(hidden_dim)
        for row_idx in range(batch_size):
            length = int(lengths[row_idx].item())
            if length <= 0:
                pieces = [zero] * (6 + 2 * self.num_segments)
                pooled_rows.append(torch.cat(pieces, dim=0))
                continue

            valid = token_embeddings[row_idx, :length]
            pieces = [
                valid.mean(dim=0),
                valid.std(dim=0, unbiased=False),
                valid.max(dim=0).values,
                valid.min(dim=0).values,
                valid[0],
                valid[-1],
            ]
            for seg_idx in range(self.num_segments):
                start = (seg_idx * length) // self.num_segments
                end = ((seg_idx + 1) * length) // self.num_segments
                if end <= start:
                    end = min(length, start + 1)
                segment = valid[start:end]
                pieces.extend([
                    segment.mean(dim=0),
                    segment.max(dim=0).values,
                ])
            pooled_rows.append(torch.cat(pieces, dim=0))

        return torch.stack(pooled_rows, dim=0), weights


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
        chat_template_prompt: Optional[str] = None,
        attention_slots: int = 1,
        document_window: str = "tail",
    ):
        super().__init__()
        if attention_slots < 1:
            raise ValueError("attention_slots must be >= 1")
        document_window = validate_document_window(document_window)

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
        self._chat_template_prompt = chat_template_prompt
        self._attention_slots = attention_slots
        self._document_window = document_window

        if skip_llm:
            # Cached mode: no LLM loaded, use provided hidden_size
            if cached_hidden_size <= 0:
                raise ValueError(
                    "cached_hidden_size must be > 0 when skip_llm=True"
                )
            self._hidden_size = cached_hidden_size
            self._compute_dtype = None
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

            # Use model's preferred dtype (bfloat16 for Gemma/MedGemma, float16 for Qwen)
            from .gpu_hidden_state_store import _get_model_dtype
            self._compute_dtype = _get_model_dtype(self._hf_config) if freeze_llm else None
            logger.info(f"Loading pretrained weights from {model_name} (hidden_size={self._hidden_size}, dtype={self._compute_dtype})")
            # Load to CPU first, then move to target device.  Avoids meta
            # tensors that device_map + remove_hook_from_module can leave
            # for models with tied weights (e.g. lm_head tied to embed_tokens).
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name, config=self._hf_config, trust_remote_code=True,
                torch_dtype=self._compute_dtype,
            )
            self._model = self._model.to(self._device)

            # Load pretrained tokenizer (right padding for pooling over all tokens)
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                padding_side="right",
                truncation_side="left",
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

            # Validate chat template availability
            if chat_template_prompt is not None:
                if not hasattr(self._tokenizer, 'chat_template') or self._tokenizer.chat_template is None:
                    logger.warning(
                        f"chat_template_prompt is set but tokenizer for {model_name} "
                        f"has no chat_template. The prompt will be ignored. "
                        f"Use an instruct model for chat template support."
                    )
                    self._chat_template_prompt = None
                else:
                    logger.info(f"Chat template prompt enabled ({len(chat_template_prompt)} chars)")

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

        # Gated attention pooling over token hidden states (or projections).
        if attention_slots == 1:
            self._pooling = GatedAttentionPooling(
                hidden_dim=effective_dim,
                attention_dim=gated_attention_dim,
            )
            pooled_dim = effective_dim
        else:
            self._pooling = MultiSlotGatedAttentionPooling(
                hidden_dim=effective_dim,
                attention_dim=gated_attention_dim,
                num_slots=attention_slots,
            )
            pooled_dim = effective_dim * attention_slots

        # Projection MLP
        self._output_dim = projection_dim
        self._projection = nn.Sequential(
            nn.Linear(pooled_dim, projection_dim),
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
        logger.info(f"  Document window: {document_window}")
        if downprojection_dim is not None:
            logger.info(f"  Downprojection: {self._hidden_size} -> {downprojection_dim} (trainable)")
        logger.info(f"  Effective dim (pooling input): {effective_dim}")
        logger.info(f"  Gated attention dim: {gated_attention_dim}")
        logger.info(f"  Attention slots: {attention_slots}")
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

    def _prepare_texts(self, texts: List[str]) -> List[str]:
        """Wrap texts in the model's chat template if configured.

        When chat_template_prompt is set, each text is formatted as a single
        user message: [{role: "user", content: "{prompt}{text}"}] using the
        tokenizer's apply_chat_template method.

        Args:
            texts: Raw clinical text strings.

        Returns:
            Formatted text strings (or originals if no chat template).
        """
        if self._chat_template_prompt is None:
            return texts
        prepared = []
        for text in texts:
            messages = [{"role": "user", "content": f"{self._chat_template_prompt}{text}"}]
            formatted = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            prepared.append(formatted)
        return prepared

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

        # Apply chat template if configured
        texts = self._prepare_texts(texts)

        # Tokenize with a generic document window and right padding.
        encoding = tokenize_with_document_window(
            self._tokenizer,
            texts,
            max_length=self._max_length,
            document_window=self._document_window,
            return_tensors="pt",
        )

        input_ids = encoding['input_ids'].to(self._device, non_blocking=True)
        attention_mask = encoding['attention_mask'].to(self._device, non_blocking=True)

        # Forward through the LLM
        if self._freeze_llm:
            with torch.no_grad():
                # Use autocast for memory-efficient frozen LLM forward pass
                with torch.amp.autocast("cuda", dtype=self._compute_dtype, enabled=self._device.type == "cuda"):
                    # Use base transformer (model.model) to skip lm_head logits
                    # computation — saves memory and avoids tied-weight issues.
                    outputs = self._model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
            # Sanitize NaN/Inf (some models overflow in float16)
            from .hidden_state_cache import _sanitize_hidden_states
            hidden_states = _sanitize_hidden_states(hidden_states, context="live_forward")
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
            # Sanitize NaN/Inf (some models overflow in float16)
            from .hidden_state_cache import _sanitize_hidden_states
            hidden_states = _sanitize_hidden_states(hidden_states, context="live_forward_trainable")
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
        self._last_pooled_features = pooled.detach()

        # Project to output dimension
        features = self._projection(pooled)

        return features

    def extract_shared_forest_features(
        self,
        texts_or_cached: Union[List[str], dict],
        text_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Expose learned neural features to both X and W forest matrices."""
        if text_features is not None:
            return text_features
        return self.forward(texts_or_cached)

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

        # Sanitize NaN/Inf that may have been stored in cache
        from .hidden_state_cache import _sanitize_hidden_states
        hidden_states = _sanitize_hidden_states(hidden_states, context="forward_cached")

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
        self._last_pooled_features = pooled.detach()

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
            'chat_template_prompt': self._chat_template_prompt,
            'attention_slots': self._attention_slots,
            'document_window': self._document_window,
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

        prepared_texts = self._prepare_texts(texts)
        for i, text in enumerate(prepared_texts):
            encoding = tokenize_with_document_window(
                self._tokenizer,
                text,
                max_length=self._max_length,
                document_window=self._document_window,
                return_tensors="pt",
            )
            tokens = self._tokenizer.convert_ids_to_tokens(
                encoding['input_ids'][0].tolist()
            )
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

    def attention_entropy_loss(self) -> Optional[torch.Tensor]:
        """Return normalized entropy of the latest attention weights.

        Lower values mean the extractor is using a smaller set of token
        positions. This is a generic neural regularizer; it does not inspect
        token strings or clinical concepts.
        """
        weights = getattr(self, "_last_attention_weights", None)
        if weights is None:
            return None

        weights = weights.float()
        if weights.dim() not in {2, 3}:
            return None

        valid = weights > 0
        support = valid.sum(dim=-1).float().clamp_min(2.0)
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1)
        normalized_entropy = entropy / support.log()
        return normalized_entropy.mean()


class FrozenLLMTokenCNNExtractor(FrozenLLMPoolerExtractor):
    """Frozen LLM extractor with trainable local token CNN pooling.

    This uses the same cached/live hidden-state interface as
    FrozenLLMPoolerExtractor, but replaces direct gated pooling with learned
    local convolutions over the hidden-state sequence before attention pooling.
    It is intentionally generic: no token strings, clinical concepts, regexes,
    or role labels are encoded in the architecture.
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
        chat_template_prompt: Optional[str] = None,
        attention_slots: int = 1,
        document_window: str = "tail",
    ):
        if attention_slots != 1:
            logger.info(
                "FrozenLLMTokenCNNExtractor ignores attention_slots=%d; "
                "token CNN pooling uses one learned document attention map.",
                attention_slots,
            )
        super().__init__(
            model_name=model_name,
            max_length=max_length,
            freeze_llm=freeze_llm,
            gated_attention_dim=gated_attention_dim,
            projection_dim=projection_dim,
            dropout=dropout,
            gradient_checkpointing=gradient_checkpointing,
            downprojection_dim=downprojection_dim,
            device=device,
            skip_llm=skip_llm,
            cached_hidden_size=cached_hidden_size,
            chat_template_prompt=chat_template_prompt,
            attention_slots=1,
            document_window=document_window,
        )

        token_cnn_dim = max(16, gated_attention_dim // 4)
        self._pooling = TokenCNNGatedPooling(
            hidden_dim=self._effective_dim,
            conv_dim=token_cnn_dim,
            attention_dim=gated_attention_dim,
            dropout=dropout,
        )
        pooled_dim = self._pooling.output_dim
        self._projection = nn.Sequential(
            nn.Linear(pooled_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )
        self._pooler_type = "token_cnn"
        logger.info(
            "FrozenLLMTokenCNNExtractor initialized: effective_dim=%d, "
            "conv_dim=%d, pooled_dim=%d, output_dim=%d",
            self._effective_dim,
            token_cnn_dim,
            pooled_dim,
            projection_dim,
        )
        self.to(self._device)

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state["pooler_type"] = self._pooler_type
        state["token_cnn_kernel_sizes"] = self._pooling.kernel_sizes
        return state


class FrozenLLMStatPoolerExtractor(FrozenLLMPoolerExtractor):
    """Frozen LLM extractor that exposes generic pooled hidden-state evidence.

    The projection path still trains like the standard frozen LLM pooler, but
    the forest receives the pre-projection statistical bank. This is meant to
    preserve literal local evidence in the neural hidden states without naming
    any clinical concepts or parsing raw text.
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
        chat_template_prompt: Optional[str] = None,
        attention_slots: int = 1,
        document_window: str = "tail",
    ):
        if attention_slots != 1:
            logger.info(
                "FrozenLLMStatPoolerExtractor ignores attention_slots=%d; "
                "stat pooling uses fixed global and segment statistics.",
                attention_slots,
            )
        super().__init__(
            model_name=model_name,
            max_length=max_length,
            freeze_llm=freeze_llm,
            gated_attention_dim=gated_attention_dim,
            projection_dim=projection_dim,
            dropout=dropout,
            gradient_checkpointing=gradient_checkpointing,
            downprojection_dim=downprojection_dim,
            device=device,
            skip_llm=skip_llm,
            cached_hidden_size=cached_hidden_size,
            chat_template_prompt=chat_template_prompt,
            attention_slots=1,
            document_window=document_window,
        )

        self._pooling = MaskedStatPooling(
            hidden_dim=self._effective_dim,
            num_segments=2,
        )
        pooled_dim = self._pooling.output_dim
        self._projection = nn.Sequential(
            nn.Linear(pooled_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )
        self._pooler_type = "stat_pooler"
        self._shared_feature_dim = pooled_dim
        logger.info(
            "FrozenLLMStatPoolerExtractor initialized: effective_dim=%d, "
            "shared_feature_dim=%d, output_dim=%d",
            self._effective_dim,
            pooled_dim,
            projection_dim,
        )
        self.to(self._device)

    def extract_shared_forest_features(
        self,
        texts_or_cached: Union[List[str], dict],
        text_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Expose the generic pooled hidden-state bank to the forest."""
        del text_features
        pooled = getattr(self, "_last_pooled_features", None)
        if pooled is None:
            _ = self.forward(texts_or_cached)
            pooled = self._last_pooled_features
        return pooled

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state["pooler_type"] = self._pooler_type
        state["shared_feature_dim"] = self._shared_feature_dim
        state["stat_pool_segments"] = self._pooling.num_segments
        return state
