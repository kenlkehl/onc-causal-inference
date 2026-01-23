# cdt/models/gated_mil_hierarchical_extractor.py
"""Gated MIL Hierarchical feature extractor using sentence-level BERT + gated attention.

This module implements a hierarchical approach for extracting features from long
clinical text using gated MIL (Multiple Instance Learning) attention:

1. Split text into sentences
2. Encode each sentence with a tiny BERT (e.g., prajjwal1/bert-tiny), taking the [CLS] token
3. Apply gated MIL attention with K learnable confounder queries
4. Task-specific weighting of confounders (propensity, tau/y0, outcome/y1)
5. Concatenate and project to output dimension

Key insight: Confounders are patient characteristics (metastatic sites, performance status)
that affect both treatment and outcome. The same K confounders feed into all tasks,
but each task can weight them differently:
- Propensity: "Which confounders predict treatment?"
- Tau: "Which confounders modify treatment effect?"
- Outcome: "Which confounders predict baseline outcome?"

Architecture:
    Long Clinical Text
            |
    Split into Sentences (S sentences)
            |
    Tiny BERT per Sentence -> [CLS] token (S x hidden_dim)
            |
    Gated MIL Attention with K Confounder Queries
            |
    K Confounder Representations (K x D)
            |
    Task-Specific Weighting (propensity, tau, outcome)
            |
    Concatenate: [prop_repr || tau_repr || out_repr] (3 x D)
            |
    MLP Projection
            |
    Final Representation (output_dim,) -> DragonNet/RLearner

References:
- Ilse et al. (2018): "Attention-based Deep Multiple Instance Learning"
- Lu et al. (2021): "Data-efficient and weakly supervised computational pathology"
"""

import logging
import math
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .confounder_extractor import split_into_sentences
from .gated_mil_attention import GatedMILAttention, TaskSpecificConfounderWeighting


logger = logging.getLogger(__name__)


class GatedMILHierarchicalExtractor(nn.Module):
    """
    Hierarchical feature extractor using gated MIL attention.

    Combines:
    - Sentence-level BERT encoding (tiny BERT for efficiency)
    - Gated MIL attention with K learnable confounder queries
    - Task-specific weighting of shared confounders

    Args:
        sentence_encoder_model: HuggingFace model name for sentence encoding
        freeze_sentence_encoder: Whether to freeze the sentence encoder weights
        max_sentences: Maximum number of sentences to process per document
        max_sentence_length: Maximum tokens per sentence for BERT encoding
        mil_hidden_dim: Hidden dimension for gated MIL attention
        num_confounders: Number of confounder queries (K)
        model_type: "rlearner" or "dragonnet"
        projection_dim: Final output dimension
        dropout: Dropout rate
        device: PyTorch device
    """

    def __init__(
        self,
        sentence_encoder_model: str = "prajjwal1/bert-tiny",
        freeze_sentence_encoder: bool = True,
        max_sentences: int = 100,
        max_sentence_length: int = 128,
        mil_hidden_dim: int = 128,
        num_confounders: int = 4,
        model_type: str = "rlearner",
        projection_dim: int = 128,
        dropout: float = 0.1,
        device: Optional[torch.device] = None
    ):
        super().__init__()

        self._device = device or torch.device('cpu')
        self._sentence_encoder_model = sentence_encoder_model
        self._freeze = freeze_sentence_encoder
        self._max_sentences = max_sentences
        self._max_sentence_length = max_sentence_length
        self._mil_hidden_dim = mil_hidden_dim
        self._num_confounders = num_confounders
        self._model_type = model_type
        self._projection_dim = projection_dim
        self._dropout = dropout

        # Lazy initialization
        self._sentence_encoder = None
        self._tokenizer = None
        self._sentence_dim = None
        self._gated_mil_attention = None
        self._task_weighting = None
        self._output_projection = None
        self._initialized = False

        logger.info(f"GatedMILHierarchicalExtractor initialized:")
        logger.info(f"  Sentence encoder: {sentence_encoder_model}")
        logger.info(f"  Freeze encoder: {freeze_sentence_encoder}")
        logger.info(f"  Num confounders: {num_confounders}")
        logger.info(f"  MIL hidden dim: {mil_hidden_dim}")
        logger.info(f"  Model type: {model_type}")
        logger.info(f"  Projection dim: {projection_dim}")

    def _ensure_initialized(self):
        """Lazily initialize components."""
        if self._initialized:
            return

        from transformers import AutoModel, AutoTokenizer

        logger.info(f"Loading sentence encoder: {self._sentence_encoder_model}")
        self._tokenizer = AutoTokenizer.from_pretrained(self._sentence_encoder_model)
        self._sentence_encoder = AutoModel.from_pretrained(self._sentence_encoder_model)
        self._sentence_encoder = self._sentence_encoder.to(self._device)
        self._sentence_dim = self._sentence_encoder.config.hidden_size
        logger.info(f"  Sentence encoder dim: {self._sentence_dim}")

        if self._freeze:
            for param in self._sentence_encoder.parameters():
                param.requires_grad = False
            logger.info("  Sentence encoder frozen")

        # Gated MIL attention
        self._gated_mil_attention = GatedMILAttention(
            input_dim=self._sentence_dim,
            hidden_dim=self._mil_hidden_dim,
            num_confounders=self._num_confounders,
            dropout=self._dropout
        ).to(self._device)

        # Task-specific weighting
        self._task_weighting = TaskSpecificConfounderWeighting(
            confounder_dim=self._sentence_dim,
            num_confounders=self._num_confounders,
            model_type=self._model_type
        ).to(self._device)

        # Output projection: 3 * sentence_dim -> projection_dim
        # (propensity_repr || tau_repr || outcome_repr) -> final output
        self._output_projection = nn.Sequential(
            nn.Linear(3 * self._sentence_dim, self._projection_dim * 2),
            nn.LayerNorm(self._projection_dim * 2),
            nn.GELU(),
            nn.Dropout(self._dropout),
            nn.Linear(self._projection_dim * 2, self._projection_dim),
            nn.LayerNorm(self._projection_dim)
        ).to(self._device)

        self._initialized = True
        logger.info("GatedMILHierarchicalExtractor initialization complete")

    @property
    def output_dim(self) -> int:
        """Return the output dimension of this feature extractor."""
        return self._projection_dim

    def _encode_sentences_batch(self, sentences: List[str]) -> torch.Tensor:
        """
        Encode sentences with BERT, returning [CLS] tokens.

        Args:
            sentences: List of sentence strings

        Returns:
            Tensor of shape (num_sentences, sentence_dim) containing [CLS] embeddings
        """
        if not sentences:
            self._ensure_initialized()
            return torch.zeros(0, self._sentence_dim, device=self._device)

        encoded = self._tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self._max_sentence_length,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].to(self._device)
        attention_mask = encoded['attention_mask'].to(self._device)

        with torch.set_grad_enabled(not self._freeze):
            outputs = self._sentence_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        # [CLS] token at position 0
        return outputs.last_hidden_state[:, 0, :]

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Extract features from texts.

        Args:
            texts: List of document texts

        Returns:
            Feature tensor of shape (batch_size, projection_dim)
        """
        self._ensure_initialized()
        batch_size = len(texts)
        batch_outputs = []

        for text in texts:
            # 1. Split into sentences
            sentences = split_into_sentences(text, self._max_sentences)
            if not sentences:
                sentences = [text[:500]]  # Fallback for short/malformed text

            # 2. Encode sentences with BERT
            sentence_embeddings = self._encode_sentences_batch(sentences)  # (S, sentence_dim)

            # 3. Apply gated MIL attention to get K confounders
            confounders, _ = self._gated_mil_attention(sentence_embeddings)  # (K, sentence_dim)

            # 4. Apply task-specific weighting
            prop_repr, task2_repr, task3_repr = self._task_weighting(confounders)
            # Each is (sentence_dim,)

            # 5. Concatenate task representations
            combined = torch.cat([prop_repr, task2_repr, task3_repr], dim=0)  # (3 * sentence_dim,)

            batch_outputs.append(combined)

        # Stack batch
        batch_outputs = torch.stack(batch_outputs)  # (B, 3 * sentence_dim)

        # 6. Project to output dimension
        features = self._output_projection(batch_outputs)  # (B, projection_dim)

        return features

    def init_extractor(self, texts: List[str]) -> 'GatedMILHierarchicalExtractor':
        """
        Initialize the feature extractor (triggers lazy initialization).

        For GatedMILHierarchicalExtractor, this loads the pretrained sentence
        encoder and initializes the gated MIL attention components. The texts
        argument is not used since we use pretrained tokenizers.

        Args:
            texts: List of training text strings (not used, kept for API compatibility)

        Returns:
            self for method chaining
        """
        self._ensure_initialized()
        return self

    def fit_tokenizer(self, texts: List[str]) -> 'GatedMILHierarchicalExtractor':
        """Alias for init_extractor() for backward compatibility."""
        return self.init_extractor(texts)

    def to(self, device):
        """Override to track device properly."""
        self._device = device if isinstance(device, torch.device) else torch.device(device)

        if self._sentence_encoder is not None:
            self._sentence_encoder = self._sentence_encoder.to(self._device)
        if self._gated_mil_attention is not None:
            self._gated_mil_attention = self._gated_mil_attention.to(self._device)
        if self._task_weighting is not None:
            self._task_weighting = self._task_weighting.to(self._device)
        if self._output_projection is not None:
            self._output_projection = self._output_projection.to(self._device)

        return super().to(device)

    def get_state(self) -> Dict[str, Any]:
        """
        Get extractor state for checkpoint saving.

        Returns:
            Dictionary containing configuration for reconstruction
        """
        return {
            'sentence_encoder_model': self._sentence_encoder_model,
            'freeze_sentence_encoder': self._freeze,
            'max_sentences': self._max_sentences,
            'max_sentence_length': self._max_sentence_length,
            'mil_hidden_dim': self._mil_hidden_dim,
            'num_confounders': self._num_confounders,
            'model_type': self._model_type,
            'projection_dim': self._projection_dim,
            'dropout': self._dropout
        }

    def interpret_attention(
        self,
        texts: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get human-readable interpretation of gated MIL attention.

        This extracts attention weights from each confounder query to sentences,
        showing which sentences each confounder focuses on, plus the task-specific
        weights for each confounder.

        Args:
            texts: List of document texts
            top_k: Number of top-attended sentences to show per confounder

        Returns:
            List of dicts per document with attention interpretations:
            - 'sentences': List of sentence strings
            - 'confounder_attention': Dict mapping confounder index to attention weights
            - 'top_sentences_per_confounder': Top-k sentences per confounder
            - 'task_weights': Task-specific weights for each confounder
        """
        self._ensure_initialized()
        interpretations = []

        # Get task weights (shared across all documents)
        task_weights = self._task_weighting.get_weights()

        with torch.no_grad():
            for text in texts:
                sentences = split_into_sentences(text, self._max_sentences)
                if not sentences:
                    sentences = [text[:500]]

                # Encode sentences
                sentence_embeddings = self._encode_sentences_batch(sentences)

                # Get attention weights
                _, attention_weights = self._gated_mil_attention(
                    sentence_embeddings, return_attention=True
                )  # (K, S)

                if attention_weights is not None and len(sentences) > 0:
                    confounder_attention = {}
                    top_sentences_per_confounder = {}

                    for k in range(self._num_confounders):
                        attn = attention_weights[k].cpu()  # (S,)
                        confounder_attention[f'confounder_{k}'] = attn.tolist()

                        # Get top-k sentences for this confounder
                        k_actual = min(top_k, len(sentences))
                        top_vals, top_indices = torch.topk(attn, k_actual)

                        top_sentences_per_confounder[f'confounder_{k}'] = [
                            {
                                'sentence': sentences[idx],
                                'attention': val.item(),
                                'idx': int(idx)
                            }
                            for val, idx in zip(top_vals, top_indices)
                        ]

                    interpretations.append({
                        'sentences': sentences,
                        'confounder_attention': confounder_attention,
                        'top_sentences_per_confounder': top_sentences_per_confounder,
                        'task_weights': task_weights
                    })
                else:
                    interpretations.append({
                        'sentences': sentences,
                        'confounder_attention': {},
                        'top_sentences_per_confounder': {},
                        'task_weights': task_weights
                    })

        return interpretations

    def get_attention_weights(self, texts: List[str]) -> Dict[str, Any]:
        """
        Get raw attention weights for visualization.

        Args:
            texts: List of document texts

        Returns:
            Dictionary with interpretations and model metadata
        """
        interpretations = self.interpret_attention(texts, top_k=self._max_sentences)
        return {
            'interpretations': interpretations,
            'num_confounders': self._num_confounders,
            'model_type': self._model_type,
            'sentence_model': self._sentence_encoder_model
        }

    def get_task_weights(self) -> Dict[str, List[float]]:
        """
        Get the task-specific confounder weights for interpretability.

        Returns:
            Dictionary with normalized weights per task:
            - 'propensity': [w1, w2, ..., wK] weights for propensity prediction
            - 'tau' or 'y0': [w1, w2, ..., wK] weights for tau/y0 prediction
            - 'outcome' or 'y1': [w1, w2, ..., wK] weights for outcome/y1 prediction
        """
        self._ensure_initialized()
        return self._task_weighting.get_weights()
