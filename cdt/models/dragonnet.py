# cdt/models/dragonnet.py
"""DragonNet architecture for causal inference from confounders."""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class DragonNet(nn.Module):
    """Binary treatment DragonNet. Can be initialized from pretrained multi-treatment model."""

    def __init__(self, input_dim, representation_dim=200, hidden_outcome_dim=100, dropout=0.2):
        super().__init__()
        self.dropout_rate = dropout

        # Shared representation layers (can be loaded from pretrained)
        self.representation_fc1 = nn.Linear(input_dim, representation_dim)
        self.representation_fc2 = nn.Linear(representation_dim, representation_dim)

        # Dropout for representation layers
        self.rep_dropout = nn.Dropout(dropout)

        # Binary treatment propensity head (matches old_cdt: single linear layer)
        self.propensity_fc1 = nn.Linear(representation_dim, 1)

        # Binary outcome heads (matches old_cdt: 2 hidden layers)
        self.outcome0_fc1 = nn.Linear(representation_dim, hidden_outcome_dim)
        self.outcome0_fc2 = nn.Linear(hidden_outcome_dim, hidden_outcome_dim)
        self.outcome0_fc3 = nn.Linear(hidden_outcome_dim, 1)

        self.outcome1_fc1 = nn.Linear(representation_dim, hidden_outcome_dim)
        self.outcome1_fc2 = nn.Linear(hidden_outcome_dim, hidden_outcome_dim)
        self.outcome1_fc3 = nn.Linear(hidden_outcome_dim, 1)

        # Dropout for outcome heads
        self.outcome_dropout = nn.Dropout(dropout)

        
    def forward(self, confounder_features):
        """
        Args:
            confounder_features: Output from FeatureExtractor
                Shape: (batch, num_confounders * features_per_confounder)

        Returns:
            y0_logit, y1_logit, t_logit, final_common_layer
        """
        h = F.relu(self.representation_fc1(confounder_features))
        h = self.rep_dropout(h)
        final_common_layer = F.elu(self.representation_fc2(h))
        final_common_layer = self.rep_dropout(final_common_layer)

        t_logit = self.propensity_fc1(final_common_layer)

        y0 = F.relu(self.outcome0_fc1(final_common_layer))
        y0 = self.outcome_dropout(y0)
        y0 = F.elu(self.outcome0_fc2(y0))
        y0 = self.outcome_dropout(y0)
        y0_logit = self.outcome0_fc3(y0)

        y1 = F.relu(self.outcome1_fc1(final_common_layer))
        y1 = self.outcome_dropout(y1)
        y1 = F.elu(self.outcome1_fc2(y1))
        y1 = self.outcome_dropout(y1)
        y1_logit = self.outcome1_fc3(y1)

        return y0_logit, y1_logit, t_logit, final_common_layer

    def get_representation(self, features):
        """Compute shared representation from input features."""
        h = F.relu(self.representation_fc1(features))
        h = self.rep_dropout(h)
        final_common_layer = F.elu(self.representation_fc2(h))
        final_common_layer = self.rep_dropout(final_common_layer)
        return final_common_layer

    def propensity_from_representation(self, phi):
        """Compute propensity logit from shared representation."""
        return self.propensity_fc1(phi)

    def load_pretrained_representation(self, pretrained_state_dict):
        """
        Load pretrained representation layers (fc1-fc6) if dimensions match.
        
        Args:
            pretrained_state_dict: State dict from pretrained model (can be full checkpoint or just representation layers)
        
        Returns:
            bool: True if loaded successfully, False if dimension mismatch
        """
        # Handle both full checkpoint and direct state dict

        # Handle full checkpoint format
        if 'dragonnet' in pretrained_state_dict:
            state_dict = pretrained_state_dict['dragonnet']
        elif 'dragonnet_representation' in pretrained_state_dict:
            state_dict = pretrained_state_dict['dragonnet_representation']
        elif 'representation_fc1.weight' in pretrained_state_dict:
            state_dict = pretrained_state_dict
        elif 'representation_fc1' in pretrained_state_dict and isinstance(pretrained_state_dict['representation_fc1'], dict):
            # Nested dict format (legacy)
            state_dict = {}
            for key in ['representation_fc1', 'representation_fc2']:
                if key in pretrained_state_dict:
                    for param_name, param_value in pretrained_state_dict[key].items():
                        state_dict[f'{key}.{param_name}'] = param_value
        else:
            logger.warning("Cannot parse pretrained state dict format")
            return False
        
        # Check if input dimensions match
        pretrained_fc1_weight_shape = state_dict['representation_fc1.weight'].shape
        current_fc1_weight_shape = self.representation_fc1.weight.shape
        
        if pretrained_fc1_weight_shape != current_fc1_weight_shape:
            logger.warning("Cannot load pretrained representation - dimension mismatch!")
            logger.warning(f"  Pretrained input dim: {pretrained_fc1_weight_shape[1]}")
            logger.warning(f"  Current input dim: {current_fc1_weight_shape[1]}")
            logger.warning(f"  This usually means different numbers of confounders between pretrain and current model.")
            logger.warning(f"  Skipping pretrained representation weights. Model will use random initialization.")
            return False
        
        # Dimensions match - load all layers
        try:
            # Create state dict for just the representation layers
            rep_state_dict = {}
            for key in ['representation_fc1', 'representation_fc2']:
                for param_name in ['weight', 'bias']:
                    full_key = f'{key}.{param_name}'
                    if full_key in state_dict:
                        rep_state_dict[full_key] = state_dict[full_key]

            # Load with strict=False to allow missing keys (outcome/propensity heads)
            self.load_state_dict(rep_state_dict, strict=False)
            logger.info("Successfully loaded pretrained representation layers (fc1-fc2)")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load pretrained weights: {e}")
            return False