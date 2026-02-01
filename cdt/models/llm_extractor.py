# cdt/models/llm_extractor.py
"""LLM (decoder-only) feature extractor using last token embedding.

This module implements a feature extractor using a decoder-only LLM architecture
(e.g., Qwen3-0.6B-Base) that is initialized with RANDOM weights and trained
entirely from scratch via the supervised causal objective.

Key design choices:
1. Random weight initialization (no pretrained weights)
2. Pretrained tokenizer (BBPE tokenization from the model)
3. Last token embedding as document representation (GPT-style)
4. Left padding for consistent last-token extraction
5. Gradient checkpointing for memory efficiency

Architecture:
    Clinical Text
         |
    Tokenize with pretrained BBPE tokenizer (left-padded)
         |
    Randomly-initialized Decoder-only LLM
         |
    Extract last token hidden state from final layer
         |
    Projection layer (2-layer MLP with LayerNorm)
         |
    Output Representation (projection_dim)

DOES NOT require fit_tokenizer() - uses pretrained tokenizer from HuggingFace.
"""

import logging
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LLMFeatureExtractor(nn.Module):
    """
    Decoder-only LLM feature extractor with random weight initialization.

    Uses the architecture of a pretrained model (e.g., Qwen/Qwen3-0.6B-Base)
    but initializes weights randomly. The pretrained tokenizer is used.

    Extracts features by taking the last token's hidden state from the final
    layer, similar to how GPT models are used for classification.

    Args:
        model_name: HuggingFace model name to use architecture/tokenizer from
        max_length: Maximum sequence length (up to 32768 for Qwen3)
        projection_dim: Output projection dimension (None = use raw hidden size)
        dropout: Dropout rate for projection layers
        gradient_checkpointing: Enable gradient checkpointing for memory efficiency
        device: PyTorch device
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B-Base",
        max_length: int = 8192,
        projection_dim: Optional[int] = 128,
        dropout: float = 0.1,
        gradient_checkpointing: bool = True,
        device: Optional[torch.device] = None
    ):
        super().__init__()

        self._device = device or torch.device('cpu')
        self._model_name = model_name
        self._max_length = max_length
        self._projection_dim = projection_dim
        self._dropout = dropout
        self._gradient_checkpointing = gradient_checkpointing

        # Import transformers here to handle import errors gracefully
        try:
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers library is required for LLMFeatureExtractor. "
                "Install with: pip install transformers"
            )

        logger.info(f"Initializing LLMFeatureExtractor with {model_name} architecture (random weights)")

        # Load config from pretrained model
        self._config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self._hidden_size = self._config.hidden_size

        # Initialize model with random weights (not pretrained!)
        # Using from_config instead of from_pretrained gives random initialization
        logger.info(f"Creating model from config with random weights (hidden_size={self._hidden_size})")
        self._model = AutoModelForCausalLM.from_config(self._config)

        # Load pretrained tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"  # Critical for last-token extraction
        )

        # Ensure pad token exists
        if self._tokenizer.pad_token is None:
            if self._tokenizer.eos_token is not None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
            else:
                # Add a new pad token
                self._tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self._model.resize_token_embeddings(len(self._tokenizer))

        logger.info(f"Tokenizer vocab size: {len(self._tokenizer)}")
        logger.info(f"Pad token: {self._tokenizer.pad_token} (id={self._tokenizer.pad_token_id})")

        # Enable gradient checkpointing if requested
        if gradient_checkpointing:
            self._model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        # Projection layer
        if projection_dim is not None:
            self._output_dim = projection_dim
            self._projection = nn.Sequential(
                nn.Linear(self._hidden_size, projection_dim),
                nn.LayerNorm(projection_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(projection_dim, projection_dim),
                nn.LayerNorm(projection_dim),
            )
        else:
            self._output_dim = self._hidden_size
            self._projection = None

        logger.info(f"LLMFeatureExtractor initialized:")
        logger.info(f"  Model: {model_name} (random weights)")
        logger.info(f"  Hidden size: {self._hidden_size}")
        logger.info(f"  Max length: {max_length}")
        logger.info(f"  Output dim: {self._output_dim}")
        logger.info(f"  Gradient checkpointing: {gradient_checkpointing}")

    @property
    def output_dim(self) -> int:
        """Return the output dimension of this feature extractor."""
        return self._output_dim

    @property
    def hidden_size(self) -> int:
        """Return the hidden size of the underlying LLM."""
        return self._hidden_size

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Extract features from texts using last token embedding.

        Args:
            texts: List of document texts

        Returns:
            Feature tensor of shape (batch_size, output_dim)
        """
        # Tokenize with left padding
        encoding = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt"
        )

        input_ids = encoding['input_ids'].to(self._device)
        attention_mask = encoding['attention_mask'].to(self._device)

        # Forward through the model
        outputs = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # Extract last token hidden state from final layer
        # With left padding, the last token is always at position -1
        hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden_size)
        last_token_embedding = hidden_states[:, -1, :]  # (batch, hidden_size)

        # Convert to float32 if needed (Qwen3 uses BFloat16 by default)
        if last_token_embedding.dtype != torch.float32:
            last_token_embedding = last_token_embedding.float()

        # Apply projection if configured
        if self._projection is not None:
            features = self._projection(last_token_embedding)
        else:
            features = last_token_embedding

        return features

    def get_state(self) -> Dict[str, Any]:
        """
        Get extractor state for checkpoint saving.

        Returns:
            Dictionary containing configuration for reconstruction
        """
        return {
            'model_name': self._model_name,
            'max_length': self._max_length,
            'projection_dim': self._projection_dim,
            'dropout': self._dropout,
            'gradient_checkpointing': self._gradient_checkpointing,
            'hidden_size': self._hidden_size,
            'output_dim': self._output_dim,
        }

    def to(self, device):
        """Override to track device properly."""
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        return super().to(device)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        self._model.gradient_checkpointing_enable()
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self._model.gradient_checkpointing_disable()
        self._gradient_checkpointing = False

    def get_num_parameters(self) -> Dict[str, int]:
        """
        Get parameter counts for the model.

        Returns:
            Dictionary with total and trainable parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params
        }
