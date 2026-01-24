# cdt/models/causal_cnn.py
"""Causal inference model using simple 1D CNN for text representation."""

import logging
from typing import Optional, List, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn_extractor import CNNFeatureExtractor
from .bert_extractor import BertFeatureExtractor
from .gru_extractor import GRUFeatureExtractor
from .confounder_extractor import ConfounderExtractor, HierarchicalConfounderExtractor, GRUHierarchicalConfounderExtractor
from .hierarchical_transformer_extractor import HierarchicalTransformerExtractor
from .gated_mil_hierarchical_extractor import GatedMILHierarchicalExtractor
from .dragonnet import DragonNet
from .uplift import UpliftNet
from .rlearner import RLearnerNet
from ..config import normalize_feature_extractor_type


logger = logging.getLogger(__name__)


class CausalText(nn.Module):
    """
    Causal inference model for text using various feature extraction methods.

    Architecture:
    - Feature extractor (CNN, BERT, GRU, or Confounder) encodes text into feature vector
    - DragonNet/UpliftNet/RLearnerNet predicts outcomes and propensity

    CNN mode (feature_extractor_type="cnn"):
    - 1D CNN with word-level tokenization
    - Much faster to train than transformers
    - IMPORTANT: Call fit_tokenizer(texts) with training data before use

    BERT mode (feature_extractor_type="bert"):
    - HuggingFace transformer with CLS token extraction
    - Fine-tuning or frozen encoder options
    - No fit_tokenizer() needed (uses pretrained tokenizer)
    - O(N^2) attention - may struggle with very long sequences

    GRU mode (feature_extractor_type="gru"):
    - Bidirectional GRU with attention pooling
    - O(N) complexity - efficient for long sequences
    - Attention weights provide interpretability
    - IMPORTANT: Call fit_tokenizer(texts) with training data before use

    Confounder mode (feature_extractor_type="confounder"):
    - Perceiver-style cross-attention with sparse attention
    - Hierarchical option: sentence-level + token-level attention
    - Designed for extracting confounders from long clinical text
    - No fit_tokenizer() needed (uses pretrained encoders)
    """

    def __init__(
        self,
        # Feature extractor type
        feature_extractor_type: str = "cnn",
        # CNN-specific args
        embedding_dim: int = 128,
        kernel_sizes: List[int] = [3, 4, 5, 7],
        explicit_filter_concepts: Optional[Dict[str, List[str]]] = None,
        num_kmeans_filters: int = 64,
        num_random_filters: int = 0,
        cnn_dropout: float = 0.1,
        max_length: int = 2048,
        min_word_freq: int = 2,
        max_vocab_size: Optional[int] = 50000,
        projection_dim: Optional[int] = 128,
        # BERT-specific args
        bert_model_name: str = "bert-base-uncased",
        bert_max_length: int = 512,
        bert_projection_dim: Optional[int] = 128,
        bert_dropout: float = 0.1,
        bert_freeze_encoder: bool = False,
        bert_gradient_checkpointing: bool = False,
        # GRU-specific args
        gru_hidden_dim: int = 256,
        gru_num_layers: int = 2,
        gru_dropout: float = 0.1,
        gru_bidirectional: bool = True,
        gru_attention_dim: Optional[int] = None,
        gru_projection_dim: Optional[int] = 128,
        # Confounder extractor args
        confounder_num_latents: int = 4,
        confounder_explicit_texts: Optional[List[str]] = None,
        confounder_value_dim: int = 128,
        confounder_sentence_model: str = "all-MiniLM-L6-v2",
        confounder_freeze_encoder: bool = True,
        confounder_max_sentences: int = 100,
        confounder_num_heads: int = 4,
        confounder_num_iterations: int = 2,
        confounder_use_self_attention: bool = True,
        confounder_sparse_attention: bool = True,
        confounder_sparse_method: str = "entmax",
        confounder_sparse_alpha: float = 1.5,
        confounder_top_k: int = 5,
        confounder_dropout: float = 0.1,
        # Hierarchical confounder args (token-level attention)
        confounder_hierarchical: bool = False,
        confounder_token_encoder: str = "distilbert-base-uncased",
        confounder_freeze_token_encoder: bool = True,
        confounder_max_sentence_tokens: int = 128,
        # GRU-based hierarchical confounder args (learns from scratch)
        confounder_use_gru: bool = False,
        confounder_gru_embedding_dim: int = 128,
        confounder_gru_hidden_dim: int = 128,
        confounder_gru_num_layers: int = 1,
        confounder_gru_bidirectional: bool = True,
        confounder_gru_dropout: float = 0.1,
        confounder_gru_max_vocab: int = 50000,
        confounder_gru_min_word_freq: int = 2,
        confounder_gru_max_sentence_length: int = 128,
        # Hierarchical Transformer args
        hier_transformer_sentence_model: str = "prajjwal1/bert-tiny",
        hier_transformer_freeze_sentence_encoder: bool = True,
        hier_transformer_max_sentences: int = 100,
        hier_transformer_max_sentence_length: int = 128,
        hier_transformer_num_layers: int = 2,
        hier_transformer_num_heads: int = 4,
        hier_transformer_dim: int = 256,
        hier_transformer_dropout: float = 0.1,
        hier_transformer_projection_dim: int = 128,
        # Gated MIL Hierarchical args
        gated_mil_sentence_model: str = "prajjwal1/bert-tiny",
        gated_mil_freeze_sentence_encoder: bool = True,
        gated_mil_max_sentences: int = 100,
        gated_mil_max_sentence_length: int = 128,
        gated_mil_hidden_dim: int = 128,
        gated_mil_num_confounders: int = 4,
        gated_mil_dropout: float = 0.1,
        gated_mil_projection_dim: int = 128,
        gated_mil_hierarchical: bool = False,
        gated_mil_token_hidden_dim: int = 64,
        gated_mil_use_mean_pooling: bool = False,
        # DragonNet args
        dragonnet_representation_dim: int = 128,
        dragonnet_hidden_outcome_dim: int = 64,
        dragonnet_dropout: float = 0.2,
        device: str = "cuda:0",
        model_type: str = "dragonnet",  # "dragonnet", "uplift", or "rlearner"
        # Auxiliary features (for hybrid text + categorical models)
        auxiliary_dim: int = 0  # Dimension of auxiliary categorical features (0 = no auxiliary)
    ):
        """
        Initialize causal inference model with CNN, BERT, or GRU feature extractor.

        Args:
            feature_extractor_type: "cnn", "bert", or "gru"
            embedding_dim: (CNN/GRU) Dimension of word embeddings
            kernel_sizes: (CNN) List of kernel sizes for n-gram capture
            explicit_filter_concepts: (CNN) Dict mapping kernel_size to concept phrases
            num_kmeans_filters: (CNN) Number of k-means derived filters per kernel size
            num_random_filters: (CNN) Number of randomly initialized filters per kernel size
            cnn_dropout: (CNN) Dropout rate
            max_length: (CNN/GRU) Maximum sequence length in tokens
            min_word_freq: (CNN/GRU) Minimum word frequency for vocabulary inclusion
            max_vocab_size: (CNN/GRU) Maximum vocabulary size
            projection_dim: (CNN) Dimension to project CNN output to
            bert_model_name: (BERT) HuggingFace model name or path
            bert_max_length: (BERT) Maximum sequence length in subword tokens
            bert_projection_dim: (BERT) Projection dimension after CLS token
            bert_dropout: (BERT) Dropout rate for projection layer
            bert_freeze_encoder: (BERT) Whether to freeze transformer weights
            bert_gradient_checkpointing: (BERT) Enable gradient checkpointing
            gru_hidden_dim: (GRU) Hidden state dimension per direction
            gru_num_layers: (GRU) Number of stacked GRU layers
            gru_dropout: (GRU) Dropout rate
            gru_bidirectional: (GRU) Use bidirectional GRU
            gru_attention_dim: (GRU) Attention hidden dimension (default: 2*hidden_dim)
            gru_projection_dim: (GRU) Output projection dimension
            dragonnet_representation_dim: DragonNet representation dimension
            dragonnet_hidden_outcome_dim: DragonNet outcome hidden dimension
            dragonnet_dropout: Dropout rate for DragonNet layers
            device: Device string
            model_type: Architecture type ("dragonnet", "uplift", or "rlearner")
            auxiliary_dim: Dimension of auxiliary categorical features (0 = disabled)
        """
        super().__init__()

        self._device = torch.device(device)
        self.model_type = model_type
        # Normalize feature extractor type (e.g., "modernbert" -> "bert")
        self.feature_extractor_type = normalize_feature_extractor_type(feature_extractor_type)

        # Store config for checkpointing (store original type for reproducibility)
        self.config = {
            'feature_extractor_type': feature_extractor_type,
            'embedding_dim': embedding_dim,
            'kernel_sizes': kernel_sizes,
            'explicit_filter_concepts': explicit_filter_concepts,
            'num_kmeans_filters': num_kmeans_filters,
            'num_random_filters': num_random_filters,
            'cnn_dropout': cnn_dropout,
            'max_length': max_length,
            'min_word_freq': min_word_freq,
            'max_vocab_size': max_vocab_size,
            'projection_dim': projection_dim,
            'bert_model_name': bert_model_name,
            'bert_max_length': bert_max_length,
            'bert_projection_dim': bert_projection_dim,
            'bert_dropout': bert_dropout,
            'bert_freeze_encoder': bert_freeze_encoder,
            'bert_gradient_checkpointing': bert_gradient_checkpointing,
            'gru_hidden_dim': gru_hidden_dim,
            'gru_num_layers': gru_num_layers,
            'gru_dropout': gru_dropout,
            'gru_bidirectional': gru_bidirectional,
            'gru_attention_dim': gru_attention_dim,
            'gru_projection_dim': gru_projection_dim,
            'confounder_num_latents': confounder_num_latents,
            'confounder_explicit_texts': confounder_explicit_texts,
            'confounder_value_dim': confounder_value_dim,
            'confounder_sentence_model': confounder_sentence_model,
            'confounder_freeze_encoder': confounder_freeze_encoder,
            'confounder_max_sentences': confounder_max_sentences,
            'confounder_num_heads': confounder_num_heads,
            'confounder_num_iterations': confounder_num_iterations,
            'confounder_use_self_attention': confounder_use_self_attention,
            'confounder_sparse_attention': confounder_sparse_attention,
            'confounder_sparse_method': confounder_sparse_method,
            'confounder_sparse_alpha': confounder_sparse_alpha,
            'confounder_top_k': confounder_top_k,
            'confounder_dropout': confounder_dropout,
            'confounder_hierarchical': confounder_hierarchical,
            'confounder_token_encoder': confounder_token_encoder,
            'confounder_freeze_token_encoder': confounder_freeze_token_encoder,
            'confounder_max_sentence_tokens': confounder_max_sentence_tokens,
            'confounder_use_gru': confounder_use_gru,
            'confounder_gru_embedding_dim': confounder_gru_embedding_dim,
            'confounder_gru_hidden_dim': confounder_gru_hidden_dim,
            'confounder_gru_num_layers': confounder_gru_num_layers,
            'confounder_gru_bidirectional': confounder_gru_bidirectional,
            'confounder_gru_dropout': confounder_gru_dropout,
            'confounder_gru_max_vocab': confounder_gru_max_vocab,
            'confounder_gru_min_word_freq': confounder_gru_min_word_freq,
            'confounder_gru_max_sentence_length': confounder_gru_max_sentence_length,
            'hier_transformer_sentence_model': hier_transformer_sentence_model,
            'hier_transformer_freeze_sentence_encoder': hier_transformer_freeze_sentence_encoder,
            'hier_transformer_max_sentences': hier_transformer_max_sentences,
            'hier_transformer_max_sentence_length': hier_transformer_max_sentence_length,
            'hier_transformer_num_layers': hier_transformer_num_layers,
            'hier_transformer_num_heads': hier_transformer_num_heads,
            'hier_transformer_dim': hier_transformer_dim,
            'hier_transformer_dropout': hier_transformer_dropout,
            'hier_transformer_projection_dim': hier_transformer_projection_dim,
            'gated_mil_sentence_model': gated_mil_sentence_model,
            'gated_mil_freeze_sentence_encoder': gated_mil_freeze_sentence_encoder,
            'gated_mil_max_sentences': gated_mil_max_sentences,
            'gated_mil_max_sentence_length': gated_mil_max_sentence_length,
            'gated_mil_hidden_dim': gated_mil_hidden_dim,
            'gated_mil_num_confounders': gated_mil_num_confounders,
            'gated_mil_dropout': gated_mil_dropout,
            'gated_mil_projection_dim': gated_mil_projection_dim,
            'gated_mil_hierarchical': gated_mil_hierarchical,
            'gated_mil_token_hidden_dim': gated_mil_token_hidden_dim,
            'gated_mil_use_mean_pooling': gated_mil_use_mean_pooling,
            'dragonnet_representation_dim': dragonnet_representation_dim,
            'dragonnet_hidden_outcome_dim': dragonnet_hidden_outcome_dim,
            'dragonnet_dropout': dragonnet_dropout,
            'model_type': model_type,
            'auxiliary_dim': auxiliary_dim
        }

        # Store auxiliary dimension
        self.auxiliary_dim = auxiliary_dim

        # Initialize feature extractor based on normalized type
        if self.feature_extractor_type == "bert":
            self.feature_extractor = BertFeatureExtractor(
                model_name=bert_model_name,
                projection_dim=bert_projection_dim,
                max_length=bert_max_length,
                dropout=bert_dropout,
                freeze_encoder=bert_freeze_encoder,
                device=self._device
            )
            if bert_gradient_checkpointing:
                self.feature_extractor.gradient_checkpointing_enable()
            logger.info(f"Using BERT feature extractor: {bert_model_name}")
        elif self.feature_extractor_type == "gru":
            self.feature_extractor = GRUFeatureExtractor(
                embedding_dim=embedding_dim,
                hidden_dim=gru_hidden_dim,
                num_layers=gru_num_layers,
                dropout=gru_dropout,
                bidirectional=gru_bidirectional,
                attention_dim=gru_attention_dim,
                projection_dim=gru_projection_dim,
                max_length=max_length,
                min_word_freq=min_word_freq,
                max_vocab_size=max_vocab_size,
                device=self._device
            )
            logger.info(f"Using GRU feature extractor: {gru_num_layers} layers, "
                       f"hidden_dim={gru_hidden_dim}, bidirectional={gru_bidirectional}")
        elif self.feature_extractor_type == "confounder":
            if confounder_use_gru:
                # GRU-based hierarchical extractor (learns from scratch)
                self.feature_extractor = GRUHierarchicalConfounderExtractor(
                    vocab_size=confounder_gru_max_vocab,
                    embedding_dim=confounder_gru_embedding_dim,
                    min_word_freq=confounder_gru_min_word_freq,
                    max_sentence_length=confounder_gru_max_sentence_length,
                    gru_hidden_dim=confounder_gru_hidden_dim,
                    gru_num_layers=confounder_gru_num_layers,
                    gru_bidirectional=confounder_gru_bidirectional,
                    gru_dropout=confounder_gru_dropout,
                    num_latent_confounders=confounder_num_latents,
                    num_attention_heads=confounder_num_heads,
                    sparse_attention=confounder_sparse_attention,
                    sparse_alpha=confounder_sparse_alpha,
                    sparse_method=confounder_sparse_method,
                    top_k=confounder_top_k,
                    max_sentences=confounder_max_sentences,
                    value_dim=confounder_value_dim,
                    dropout=confounder_dropout,
                    model_type=model_type,
                    device=self._device
                )
                logger.info(f"Using GRU Hierarchical Confounder feature extractor: {confounder_num_latents} latents, "
                           f"GRU hidden_dim={confounder_gru_hidden_dim}, sparse={confounder_sparse_attention}")
            elif confounder_hierarchical:
                # Hierarchical extractor with token-level attention (BERT-based)
                self.feature_extractor = HierarchicalConfounderExtractor(
                    num_latent_confounders=confounder_num_latents,
                    explicit_confounder_texts=confounder_explicit_texts,
                    value_dim=confounder_value_dim,
                    token_encoder_model=confounder_token_encoder,
                    freeze_token_encoder=confounder_freeze_token_encoder,
                    max_sentences=confounder_max_sentences,
                    max_sentence_tokens=confounder_max_sentence_tokens,
                    num_attention_heads=confounder_num_heads,
                    sparse_attention=confounder_sparse_attention,
                    sparse_method=confounder_sparse_method,
                    sparse_alpha=confounder_sparse_alpha,
                    top_k=confounder_top_k,
                    dropout=confounder_dropout,
                    model_type=model_type,
                    device=self._device
                )
                logger.info(f"Using Hierarchical Confounder feature extractor: {confounder_num_latents} latents, "
                           f"token_encoder={confounder_token_encoder}, sparse={confounder_sparse_attention}")
            else:
                # Standard sentence-level extractor
                self.feature_extractor = ConfounderExtractor(
                    num_latent_confounders=confounder_num_latents,
                    explicit_confounder_texts=confounder_explicit_texts,
                    value_dim=confounder_value_dim,
                    sentence_transformer_model=confounder_sentence_model,
                    freeze_sentence_encoder=confounder_freeze_encoder,
                    max_sentences=confounder_max_sentences,
                    num_attention_heads=confounder_num_heads,
                    num_iterations=confounder_num_iterations,
                    use_self_attention=confounder_use_self_attention,
                    sparse_attention=confounder_sparse_attention,
                    sparse_method=confounder_sparse_method,
                    sparse_alpha=confounder_sparse_alpha,
                    top_k=confounder_top_k,
                    dropout=confounder_dropout,
                    device=self._device
                )
                logger.info(f"Using Confounder feature extractor: {confounder_num_latents} latents, "
                           f"{confounder_num_iterations} iterations, sparse={confounder_sparse_attention}")
        elif self.feature_extractor_type == "hierarchical_transformer":
            # Hierarchical Transformer: sentence BERT + transformer pooling
            self.feature_extractor = HierarchicalTransformerExtractor(
                sentence_encoder_model=hier_transformer_sentence_model,
                freeze_sentence_encoder=hier_transformer_freeze_sentence_encoder,
                max_sentences=hier_transformer_max_sentences,
                max_sentence_length=hier_transformer_max_sentence_length,
                num_transformer_layers=hier_transformer_num_layers,
                num_attention_heads=hier_transformer_num_heads,
                transformer_dim=hier_transformer_dim,
                transformer_dropout=hier_transformer_dropout,
                projection_dim=hier_transformer_projection_dim,
                device=self._device
            )
            logger.info(f"Using Hierarchical Transformer feature extractor: {hier_transformer_sentence_model}, "
                       f"{hier_transformer_num_layers} layers, projection_dim={hier_transformer_projection_dim}")
        elif self.feature_extractor_type == "gated_mil_hierarchical":
            # Gated MIL Hierarchical: sentence BERT + gated MIL attention with task-specific weighting
            self.feature_extractor = GatedMILHierarchicalExtractor(
                sentence_encoder_model=gated_mil_sentence_model,
                freeze_sentence_encoder=gated_mil_freeze_sentence_encoder,
                max_sentences=gated_mil_max_sentences,
                max_sentence_length=gated_mil_max_sentence_length,
                mil_hidden_dim=gated_mil_hidden_dim,
                num_confounders=gated_mil_num_confounders,
                model_type=model_type,
                projection_dim=gated_mil_projection_dim,
                dropout=gated_mil_dropout,
                hierarchical=gated_mil_hierarchical,
                token_hidden_dim=gated_mil_token_hidden_dim,
                use_mean_pooling=gated_mil_use_mean_pooling,
                device=self._device
            )
            logger.info(f"Using Gated MIL Hierarchical feature extractor: {gated_mil_sentence_model}, "
                       f"{gated_mil_num_confounders} confounders, projection_dim={gated_mil_projection_dim}, "
                       f"hierarchical={gated_mil_hierarchical}, mean_pooling={gated_mil_use_mean_pooling}")
        else:
            # CNN feature extractor (default)
            self.feature_extractor = CNNFeatureExtractor(
                embedding_dim=embedding_dim,
                kernel_sizes=kernel_sizes,
                explicit_filter_concepts=explicit_filter_concepts,
                num_kmeans_filters=num_kmeans_filters,
                num_random_filters=num_random_filters,
                projection_dim=projection_dim,
                dropout=cnn_dropout,
                max_length=max_length,
                min_word_freq=min_word_freq,
                max_vocab_size=max_vocab_size,
                device=self._device
            )
            logger.info("Using CNN feature extractor")

        # Auxiliary feature projection (if enabled)
        if auxiliary_dim > 0:
            self.auxiliary_projection = nn.Sequential(
                nn.Linear(auxiliary_dim, dragonnet_representation_dim // 2),
                nn.LayerNorm(dragonnet_representation_dim // 2),
                nn.ReLU(),
                nn.Dropout(dragonnet_dropout)
            )
            logger.info(f"Auxiliary features enabled: {auxiliary_dim} -> {dragonnet_representation_dim // 2}")
        else:
            self.auxiliary_projection = None

        # Binary treatment Causal Inference Net
        # Input dim = text features + auxiliary features (if any)
        input_dim = self.feature_extractor.output_dim
        if auxiliary_dim > 0:
            input_dim += dragonnet_representation_dim // 2

        if model_type == "uplift":
            self.net = UpliftNet(
                input_dim=input_dim,
                representation_dim=dragonnet_representation_dim,
                hidden_outcome_dim=dragonnet_hidden_outcome_dim,
                dropout=dragonnet_dropout
            )
            logger.info("Using UpliftNet architecture (Base + ITE parametrization)")
        elif model_type == "rlearner":
            self.net = RLearnerNet(
                input_dim=input_dim,
                representation_dim=dragonnet_representation_dim,
                hidden_outcome_dim=dragonnet_hidden_outcome_dim,
                dropout=dragonnet_dropout
            )
            logger.info("Using R-Learner architecture (direct tau optimization)")
        else:
            self.net = DragonNet(
                input_dim=input_dim,
                representation_dim=dragonnet_representation_dim,
                hidden_outcome_dim=dragonnet_hidden_outcome_dim,
                dropout=dragonnet_dropout
            )
            logger.info("Using classic DragonNet architecture")

        # Alias for backward compatibility
        self.dragonnet = self.net

        # Move to device
        self.to(self._device)

        logger.info(f"CausalText initialized:")
        logger.info(f"  Feature extractor: {self.feature_extractor_type}")
        logger.info(f"  Feature extractor output: {input_dim}")
        logger.info(f"  Device: {self._device}")

    def forward(
        self,
        texts: List[str],
        auxiliary_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the complete model.

        Args:
            texts: List of text strings
            auxiliary_features: Optional tensor of auxiliary features (batch, auxiliary_dim)

        Returns:
            y0_logit: (batch, 1) - outcome prediction under control
            y1_logit: (batch, 1) - outcome prediction under treatment
            t_logit: (batch, 1) - treatment propensity logit
            final_common_layer: (batch, representation_dim) - shared representation
        """
        # Extract features from texts using CNN
        features = self.feature_extractor(texts)

        # Concatenate auxiliary features if provided
        if self.auxiliary_projection is not None and auxiliary_features is not None:
            aux_projected = self.auxiliary_projection(auxiliary_features.to(self._device))
            features = torch.cat([features, aux_projected], dim=1)

        if self.model_type == "uplift":
            # UpliftNet returns: y0_logit, tau_logit, t_logit, final_common_layer
            y0_logit, tau_logit, t_logit, final_common_layer = self.net(features)
            # Reconstruct y1_logit = y0_logit + tau_logit
            y1_logit = y0_logit + tau_logit
        elif self.model_type == "rlearner":
            # RLearnerNet returns: m_logit, tau, t_logit, final_common_layer
            # Returns native outputs - caller handles interpretation
            m_logit, tau, t_logit, final_common_layer = self.net(features)
            # For forward() compatibility, return in same tuple format
            # But these are semantically different: m_logit is marginal, tau is effect
            return m_logit, tau, t_logit, final_common_layer
        else:
            # DragonNet returns: y0_logit, y1_logit, t_logit, final_common_layer
            y0_logit, y1_logit, t_logit, final_common_layer = self.net(features)

        return y0_logit, y1_logit, t_logit, final_common_layer

    def train_step(
        self,
        batch: Dict[str, Any],
        alpha_propensity: float = 1.0,
        beta_targreg: float = 0.1,
        gamma_rlearner: float = 1.0,
        label_smoothing: float = 0.0,
        stop_grad_propensity: bool = False,
        attention_entropy_weight: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """
        Perform single training step.

        Args:
            batch: Dictionary with 'texts', 'treatment', 'outcome' keys.
                   Optional 'auxiliary_features' for hybrid models.
            alpha_propensity: Weight for propensity loss
            beta_targreg: Weight for targeted regularization (dragonnet/uplift)
            gamma_rlearner: Weight for R-learner loss (rlearner only)
            label_smoothing: Label smoothing factor (0 = no smoothing)
            stop_grad_propensity: If True, detach features before propensity loss
                so propensity optimization doesn't affect the feature extractor.
                This forces the representation to optimize for tau/outcome.
            attention_entropy_weight: Weight for attention entropy regularization.
                Higher = stronger penalty on diffuse attention. Only applies to
                gated_mil_hierarchical extractor.

        Returns:
            Dictionary with loss components and detached predictions
        """
        # Dispatch to specialized training step for rlearner
        if self.model_type == "rlearner":
            return self._train_step_rlearner(
                batch, alpha_propensity, gamma_rlearner, label_smoothing,
                stop_grad_propensity, attention_entropy_weight
            )

        texts = batch['texts']
        treatments = batch['treatment']  # (batch,)
        outcomes = batch['outcome']  # (batch,)
        auxiliary_features = batch.get('auxiliary_features', None)

        # Apply label smoothing if enabled
        if label_smoothing > 0:
            treatments_smooth = treatments * (1 - label_smoothing) + 0.5 * label_smoothing
            outcomes_smooth = outcomes * (1 - label_smoothing) + 0.5 * label_smoothing
        else:
            treatments_smooth = treatments
            outcomes_smooth = outcomes

        # Forward pass
        y0_logit, y1_logit, t_logit, phi = self.forward(texts, auxiliary_features)

        # Propensity loss
        propensity_loss = F.binary_cross_entropy_with_logits(
            t_logit.squeeze(-1),
            treatments_smooth
        )

        # Outcome loss - factual outcome only
        factual_logit = torch.where(
            treatments.unsqueeze(1) > 0.5,
            y1_logit,
            y0_logit
        )

        outcome_loss = F.binary_cross_entropy_with_logits(
            factual_logit.squeeze(-1),
            outcomes_smooth
        )

        # Targeted regularization (R-loss)
        if beta_targreg > 0:
            with torch.no_grad():
                propensity = torch.sigmoid(t_logit).clamp(1e-3, 1 - 1e-3)
                H = (treatments.unsqueeze(1) / propensity) - \
                    ((1 - treatments.unsqueeze(1)) / (1 - propensity))

            factual_prob = torch.sigmoid(factual_logit)
            moment = torch.mean((outcomes.unsqueeze(1) - factual_prob) * H)
            targreg_loss = moment ** 2
        else:
            targreg_loss = torch.tensor(0.0, device=self._device)

        # Total loss
        total_loss = (
            outcome_loss +
            alpha_propensity * propensity_loss +
            beta_targreg * targreg_loss
        )

        return {
            'loss': total_loss,
            'outcome_loss': outcome_loss.detach(),
            'propensity_loss': propensity_loss.detach(),
            'targreg_loss': targreg_loss.detach() if isinstance(targreg_loss, torch.Tensor) else targreg_loss,
            'y0_logit': y0_logit.detach(),
            'y1_logit': y1_logit.detach(),
            't_logit': t_logit.detach()
        }

    def _train_step_rlearner(
        self,
        batch: Dict[str, Any],
        alpha_propensity: float,
        gamma_rlearner: float,
        label_smoothing: float,
        stop_grad_propensity: bool = False,
        attention_entropy_weight: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """
        Perform R-learner training step with three-headed loss.

        R-learner loss decomposes into:
        1. Propensity loss: BCE for e(X) = P(T=1|X)
        2. Marginal outcome loss: BCE for m(X) = E[Y|X]
        3. R-loss: ((Y - m(X)) - tau(X) * (T - e(X)))^2

        The key insight is that e(X) and m(X) are DETACHED in the R-loss,
        so gradients from effect estimation flow only through tau(X).
        This provides stronger gradient signal for learning treatment
        effect modifiers from text.

        Args:
            batch: Dictionary with 'texts', 'treatment', 'outcome' keys
            alpha_propensity: Weight for propensity loss
            gamma_rlearner: Weight for R-learner loss
            label_smoothing: Label smoothing factor
            stop_grad_propensity: If True, detach features before propensity loss
            attention_entropy_weight: Weight for attention entropy regularization

        Returns:
            Dictionary with loss components and predictions
        """
        texts = batch['texts']
        treatments = batch['treatment']  # (batch,)
        outcomes = batch['outcome']  # (batch,)
        auxiliary_features = batch.get('auxiliary_features', None)

        # Apply label smoothing if enabled
        if label_smoothing > 0:
            treatments_smooth = treatments * (1 - label_smoothing) + 0.5 * label_smoothing
            outcomes_smooth = outcomes * (1 - label_smoothing) + 0.5 * label_smoothing
        else:
            treatments_smooth = treatments
            outcomes_smooth = outcomes

        # Extract features
        features = self.feature_extractor(texts)

        # Compute attention entropy loss if enabled and extractor supports it
        entropy_loss = torch.tensor(0.0, device=self._device)
        if attention_entropy_weight > 0 and hasattr(self.feature_extractor, 'compute_attention_entropy_loss'):
            # Use forward_with_attention to get entropy
            _, attention_info = self.feature_extractor.forward_with_attention(texts)
            entropy_loss = attention_info['attention_entropy']

        # Concatenate auxiliary features if provided
        if self.auxiliary_projection is not None and auxiliary_features is not None:
            aux_projected = self.auxiliary_projection(auxiliary_features.to(self._device))
            features = torch.cat([features, aux_projected], dim=1)

        if stop_grad_propensity:
            # CRITICAL: Detach features for propensity to prevent propensity
            # from dominating the representation learning
            # Create two forward paths: one for propensity (detached), one for outcome/tau
            features_for_propensity = features.detach()

            # Forward pass through R-Learner heads with split features
            # We need to run the propensity head separately with detached features
            m_logit, tau, t_logit, phi = self.net(features)

            # Re-compute propensity with detached features
            # We need direct access to the propensity head
            if hasattr(self.net, 'propensity_fc'):
                # Get shared representation
                shared_rep_detached = self.net.shared_layers(features_for_propensity)
                t_logit_detached = self.net.propensity_fc(shared_rep_detached)

                # Loss 1: Propensity loss with DETACHED features
                propensity_loss = F.binary_cross_entropy_with_logits(
                    t_logit_detached.squeeze(-1),
                    treatments_smooth
                )
            else:
                # Fallback: use detached t_logit
                propensity_loss = F.binary_cross_entropy_with_logits(
                    t_logit.detach().squeeze(-1),
                    treatments_smooth
                )
        else:
            # Standard forward pass
            m_logit, tau, t_logit, phi = self.net(features)

            # Loss 1: Propensity loss - BCE for e(X)
            propensity_loss = F.binary_cross_entropy_with_logits(
                t_logit.squeeze(-1),
                treatments_smooth
            )

        # Loss 2: Marginal outcome loss - BCE for m(X) = E[Y|X]
        outcome_loss = F.binary_cross_entropy_with_logits(
            m_logit.squeeze(-1),
            outcomes_smooth
        )

        # Loss 3: R-learner loss
        # CRITICAL: Detach nuisance functions so gradients flow only through tau
        e_X = torch.sigmoid(t_logit).detach().clamp(0.01, 0.99)
        m_X = torch.sigmoid(m_logit).detach()

        # Compute residuals
        Y_residual = outcomes - m_X.squeeze(-1)  # Y - m(X)
        T_residual = treatments - e_X.squeeze(-1)  # T - e(X)

        # R-loss: E[((Y - m(X)) - tau(X) * (T - e(X)))^2]
        r_loss = ((Y_residual - tau.squeeze(-1) * T_residual) ** 2).mean()

        # Total loss
        total_loss = (
            outcome_loss +
            alpha_propensity * propensity_loss +
            gamma_rlearner * r_loss +
            attention_entropy_weight * entropy_loss
        )

        # Derive y0/y1 for backward-compatible metrics
        # From: m = e*y1 + (1-e)*y0 and tau = y1 - y0
        # Solving: y0 = m - e*tau, y1 = m + (1-e)*tau
        with torch.no_grad():
            m_prob = torch.sigmoid(m_logit)
            prop = torch.sigmoid(t_logit)
            tau_val = tau
            y0_logit_approx = m_logit - prop * tau_val
            y1_logit_approx = m_logit + (1 - prop) * tau_val

        result = {
            'loss': total_loss,
            'outcome_loss': outcome_loss.detach(),
            'propensity_loss': propensity_loss.detach(),
            'r_loss': r_loss.detach(),
            'targreg_loss': r_loss.detach(),  # Alias for compatibility
            'm_logit': m_logit.detach(),
            'tau': tau.detach(),
            't_logit': t_logit.detach(),
            # Backward compatible outputs (derived)
            'y0_logit': y0_logit_approx.detach(),
            'y1_logit': y1_logit_approx.detach()
        }

        # Add entropy loss if computed
        if attention_entropy_weight > 0:
            result['entropy_loss'] = entropy_loss.detach() if isinstance(entropy_loss, torch.Tensor) else entropy_loss

        return result

    def predict(
        self,
        texts: List[str],
        auxiliary_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions for inference.

        Args:
            texts: List of text strings
            auxiliary_features: Optional tensor of auxiliary features (batch, auxiliary_dim)

        Returns:
            Dictionary with prediction outputs
        """
        with torch.no_grad():
            features = self.feature_extractor(texts)

            # Concatenate auxiliary features if provided
            if self.auxiliary_projection is not None and auxiliary_features is not None:
                aux_projected = self.auxiliary_projection(auxiliary_features.to(self._device))
                features = torch.cat([features, aux_projected], dim=1)

            if self.model_type == "uplift":
                y0_logit, tau_logit, t_logit, final_common_layer = self.net(features)
                y1_logit = y0_logit + tau_logit
                tau_pred = tau_logit.squeeze(-1)
            elif self.model_type == "rlearner":
                # RLearnerNet returns: m_logit, tau, t_logit, final_common_layer
                m_logit, tau, t_logit, final_common_layer = self.net(features)

                m_prob = torch.sigmoid(m_logit).squeeze(-1)  # E[Y|X]
                tau_val = tau.squeeze(-1)  # τ(X)
                prop = torch.sigmoid(t_logit).squeeze(-1)  # e(X)

                # Derive Y0/Y1 from m and τ for backward compatibility:
                # From: m = e*y1 + (1-e)*y0 and tau = y1 - y0
                # Solving: y0 = m - e*tau, y1 = m + (1-e)*tau
                y0_prob = (m_prob - prop * tau_val).clamp(0, 1)
                y1_prob = (m_prob + (1 - prop) * tau_val).clamp(0, 1)

                return {
                    'y0_prob': y0_prob,
                    'y1_prob': y1_prob,
                    'propensity': prop,
                    'm_prob': m_prob,  # Native E[Y|X]
                    'tau_pred': tau_val,  # Native τ(X)
                    't_logit': t_logit.squeeze(-1),
                    'm_logit': m_logit.squeeze(-1),
                    'final_common_layer': final_common_layer,
                    # Approximate logits for compatibility
                    'y0_logit': torch.logit(y0_prob.clamp(1e-6, 1 - 1e-6)),
                    'y1_logit': torch.logit(y1_prob.clamp(1e-6, 1 - 1e-6)),
                }
            else:
                y0_logit, y1_logit, t_logit, final_common_layer = self.net(features)
                tau_pred = (y1_logit - y0_logit).squeeze(-1)

            # Convert to probabilities
            y0_prob = torch.sigmoid(y0_logit).squeeze(-1)
            y1_prob = torch.sigmoid(y1_logit).squeeze(-1)
            propensity = torch.sigmoid(t_logit).squeeze(-1)

            return {
                'y0_prob': y0_prob,
                'y1_prob': y1_prob,
                'propensity': propensity,
                'y0_logit': y0_logit.squeeze(-1),
                'y1_logit': y1_logit.squeeze(-1),
                't_logit': t_logit.squeeze(-1),
                'final_common_layer': final_common_layer,
                'tau_pred': tau_pred
            }

    def get_features(
        self,
        texts: List[str]
    ) -> torch.Tensor:
        """
        Extract feature representations from texts.

        Args:
            texts: List of text strings

        Returns:
            Feature tensor: (batch, output_dim)
        """
        with torch.no_grad():
            return self.feature_extractor(texts)

    def init_extractor(self, texts: List[str]) -> 'CausalText':
        """
        Initialize the feature extractor with training texts.

        This method performs different operations depending on the feature extractor type:
        - CNN/GRU: Builds vocabulary from texts and initializes embeddings (required)
        - BERT: No-op (uses pretrained tokenizer)
        - ConfounderExtractor: Triggers lazy initialization of pretrained encoder
        - HierarchicalTransformer: Triggers lazy initialization of sentence encoder

        MUST be called before training for CNN/GRU/GRUConfounder extractors.
        For pretrained extractors (BERT, Confounder, HierarchicalTransformer), this
        triggers lazy initialization but the texts argument is not used.

        Args:
            texts: List of training text strings

        Returns:
            self for method chaining
        """
        if hasattr(self.feature_extractor, 'fit_tokenizer'):
            self.feature_extractor.fit_tokenizer(texts)
        # BERT uses pretrained tokenizer, no fitting needed
        return self

    def fit_tokenizer(self, texts: List[str]) -> 'CausalText':
        """
        Alias for init_extractor() for backward compatibility.

        See init_extractor() for documentation.
        """
        return self.init_extractor(texts)

    def save_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save model checkpoint including tokenizer state.
        """
        checkpoint = {
            'config': self.config,
            'model_state_dict': self.state_dict(),
            'feature_extractor': self.feature_extractor.state_dict(),
            'dragonnet': self.net.state_dict(),
            'feature_extractor_type': self.feature_extractor_type,
        }

        # Save tokenizer state for CNN, or extractor state for BERT
        if self.feature_extractor_type == "cnn":
            checkpoint['tokenizer_state'] = self.feature_extractor.get_tokenizer_state()
        else:
            checkpoint['extractor_state'] = self.feature_extractor.get_state()

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        if epoch is not None:
            checkpoint['epoch'] = epoch

        if metrics is not None:
            checkpoint['metrics'] = metrics

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    @classmethod
    def load_from_checkpoint(
        cls,
        path: str,
        device: Optional[str] = None
    ) -> 'CausalText':
        """
        Load model from checkpoint including tokenizer state.
        """
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        config = checkpoint['config']

        if device is not None:
            config['device'] = device

        # Create model
        model = cls(**config)

        # Load tokenizer state for CNN (rebuilds embedding layer with correct vocab size)
        if model.feature_extractor_type == "cnn" and 'tokenizer_state' in checkpoint:
            model.feature_extractor.load_tokenizer_state(checkpoint['tokenizer_state'])

        # Load state dict (after tokenizer so embedding has correct size)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            if 'feature_extractor' in checkpoint:
                model.feature_extractor.load_state_dict(
                    checkpoint['feature_extractor'],
                    strict=False
                )
            if 'dragonnet' in checkpoint:
                model.net.load_state_dict(
                    checkpoint['dragonnet'],
                    strict=False
                )

        logger.info(f"Model loaded from {path}")
        return model

    def to(self, device):
        """Override to track device properly."""
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        return super().to(device)


# Backward compatibility alias
CausalCNNText = CausalText
