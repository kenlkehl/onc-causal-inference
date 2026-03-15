# oci/models/explicit_confounder_featurizer.py
"""Featurization of explicitly extracted confounders for neural network models.

This module provides an MLP-based featurizer that encodes extracted confounder
values (categorical and continuous) into a fixed-size vector that can be
concatenated with text embeddings before the causal head.

For categorical confounders: One-hot encoding (k-1 dummy variables for k categories)
For continuous confounders: Normalized value (z-score) with single node
For all confounders: Binary missingness indicator

Example usage:
    from oci.models.explicit_confounder_featurizer import ExplicitConfounderFeaturizer
    from oci.config import ExplicitConfounderSpec

    specs = [
        ExplicitConfounderSpec(name="ps", type="categorical", categories=["0","1","2","3","4"]),
        ExplicitConfounderSpec(name="age", type="continuous"),
    ]

    featurizer = ExplicitConfounderFeaturizer(specs, output_dim=64)

    # confounder_values is a list of dicts per batch sample
    # e.g., [{"ps": "2", "ps_missing": False, "age": 65.0, "age_missing": False}, ...]
    features = featurizer(confounder_values)  # (batch, 64)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..config import ExplicitConfounderSpec

logger = logging.getLogger(__name__)


class ExplicitConfounderFeaturizer(nn.Module):
    """MLP-based featurizer for explicitly extracted confounders.

    Encodes a mix of categorical and continuous confounders into a fixed-size
    vector for use alongside text embeddings in causal inference models.

    Input format:
        List of dicts, one per sample in batch. Each dict contains:
        - "{name}": value (str for categorical, float for continuous, None if missing)
        - "{name}_missing": bool indicating if extraction failed

    Output:
        (batch, output_dim) tensor
    """

    def __init__(
        self,
        specs: List[ExplicitConfounderSpec],
        output_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        device: str = "cuda:0"
    ):
        """Initialize featurizer.

        Args:
            specs: List of confounder specifications
            output_dim: Dimension of output feature vector
            hidden_dim: Hidden dimension for MLP
            dropout: Dropout rate
            device: Device string
        """
        super().__init__()

        self.specs = specs
        self.output_dim = output_dim
        self._device = torch.device(device)

        # Build input encoding scheme
        # For each confounder:
        #   - Categorical: k-1 dummy variables (reference coding)
        #   - Continuous: 1 normalized value
        #   - Both: 1 missingness indicator
        self._category_maps = {}  # name -> {category: index}
        input_dim = 0

        for spec in specs:
            if spec.type == "categorical":
                # k-1 dummy variables + 1 missing indicator
                n_cats = len(spec.categories) if spec.categories else 2
                self._category_maps[spec.name] = {
                    cat: i for i, cat in enumerate(spec.categories or [])
                }
                input_dim += (n_cats - 1) + 1  # k-1 dummies + missing
            else:
                # 1 value + 1 missing indicator
                input_dim += 2

        self.input_dim = input_dim

        # Normalization stats for continuous confounders (computed during fitting)
        self._continuous_means = {}
        self._continuous_stds = {}
        self._fitted = False

        # MLP projection
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )

        # Scale factor for smooth integration with text features
        self.scale = nn.Parameter(torch.tensor(0.1))

        logger.info(f"ExplicitConfounderFeaturizer: {len(specs)} confounders, "
                   f"input_dim={input_dim}, output_dim={output_dim}")

    def fit(self, confounder_values_list: List[Dict[str, Any]]) -> 'ExplicitConfounderFeaturizer':
        """Compute normalization statistics from training data.

        Args:
            confounder_values_list: List of confounder value dicts from training data

        Returns:
            self for method chaining
        """
        # Collect continuous values
        continuous_values = {
            spec.name: [] for spec in self.specs if spec.type == "continuous"
        }

        for values in confounder_values_list:
            for spec in self.specs:
                if spec.type == "continuous":
                    val = values.get(spec.name)
                    missing = values.get(f"{spec.name}_missing", val is None)
                    if not missing and val is not None:
                        continuous_values[spec.name].append(float(val))

        # Compute mean and std for each continuous confounder
        for name, vals in continuous_values.items():
            if vals:
                self._continuous_means[name] = sum(vals) / len(vals)
                variance = sum((v - self._continuous_means[name]) ** 2 for v in vals) / len(vals)
                self._continuous_stds[name] = max(variance ** 0.5, 1e-6)  # Avoid division by zero
            else:
                self._continuous_means[name] = 0.0
                self._continuous_stds[name] = 1.0

        self._fitted = True
        logger.info(f"Fitted featurizer on {len(confounder_values_list)} samples")
        return self

    def _encode_sample(self, values: Dict[str, Any]) -> torch.Tensor:
        """Encode a single sample's confounder values.

        Args:
            values: Dict with confounder values and missing flags

        Returns:
            (input_dim,) tensor
        """
        features = []

        for spec in self.specs:
            name = spec.name
            val = values.get(name)
            missing = values.get(f"{name}_missing", val is None)

            if spec.type == "categorical":
                # k-1 dummy coding
                n_cats = len(spec.categories) if spec.categories else 2
                dummy = torch.zeros(n_cats - 1, device=self._device)

                if not missing and val is not None:
                    cat_idx = self._category_maps.get(name, {}).get(str(val))
                    if cat_idx is not None and cat_idx > 0:
                        # Reference category (idx 0) is all zeros
                        dummy[cat_idx - 1] = 1.0

                features.append(dummy)
                features.append(torch.tensor([1.0 if missing else 0.0], device=self._device))

            else:  # continuous
                if not missing and val is not None:
                    # Z-score normalization
                    mean = self._continuous_means.get(name, 0.0)
                    std = self._continuous_stds.get(name, 1.0)
                    normalized = (float(val) - mean) / std
                    features.append(torch.tensor([normalized], device=self._device))
                else:
                    # Mean imputation (0 after z-score)
                    features.append(torch.tensor([0.0], device=self._device))

                features.append(torch.tensor([1.0 if missing else 0.0], device=self._device))

        return torch.cat(features)

    def forward(self, confounder_values_list: List[Dict[str, Any]]) -> torch.Tensor:
        """Forward pass: encode batch of confounder values.

        Args:
            confounder_values_list: List of dicts, one per sample.
                Each dict should have:
                - "{name}": value or None
                - "{name}_missing": bool

        Returns:
            (batch, output_dim) tensor
        """
        # Encode each sample
        encoded = []
        for values in confounder_values_list:
            encoded.append(self._encode_sample(values))

        # Stack into batch
        batch_encoded = torch.stack(encoded)  # (batch, input_dim)

        # Project through MLP
        output = self.mlp(batch_encoded)

        return output * self.scale

    def get_state(self) -> Dict[str, Any]:
        """Get state for checkpointing."""
        return {
            'specs': [
                {
                    'name': s.name,
                    'type': s.type,
                    'categories': s.categories,
                    'description': s.description
                }
                for s in self.specs
            ],
            'output_dim': self.output_dim,
            'continuous_means': self._continuous_means,
            'continuous_stds': self._continuous_stds,
            'fitted': self._fitted,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load state from checkpoint."""
        self._continuous_means = state.get('continuous_means', {})
        self._continuous_stds = state.get('continuous_stds', {})
        self._fitted = state.get('fitted', False)


def get_raw_confounder_features(
    confounder_values_list: List[Dict[str, Any]],
    specs: List[ExplicitConfounderSpec],
    continuous_means: Optional[Dict[str, float]] = None,
    continuous_stds: Optional[Dict[str, float]] = None
) -> Tuple[List[List[float]], List[str]]:
    """Get raw confounder features for causal forest (no MLP projection).

    Returns one-hot encoded categoricals + normalized continuous + missingness indicators.
    This is used for causal forest models where we want interpretable raw features.

    Args:
        confounder_values_list: List of dicts, one per sample
        specs: List of confounder specifications
        continuous_means: Optional pre-computed means for normalization
        continuous_stds: Optional pre-computed stds for normalization

    Returns:
        Tuple of (features_list, feature_names)
        features_list: List of feature vectors, one per sample
        feature_names: List of feature names for interpretability
    """
    continuous_means = continuous_means or {}
    continuous_stds = continuous_stds or {}

    # Build feature names
    feature_names = []
    for spec in specs:
        if spec.type == "categorical":
            cats = spec.categories or []
            for cat in cats[1:]:  # Skip reference category
                feature_names.append(f"{spec.name}_{cat}")
            feature_names.append(f"{spec.name}_missing")
        else:
            feature_names.append(f"{spec.name}_normalized")
            feature_names.append(f"{spec.name}_missing")

    # Compute means/stds if not provided
    if not continuous_means or not continuous_stds:
        for spec in specs:
            if spec.type == "continuous" and spec.name not in continuous_means:
                vals = []
                for values in confounder_values_list:
                    val = values.get(spec.name)
                    missing = values.get(f"{spec.name}_missing", val is None)
                    if not missing and val is not None:
                        vals.append(float(val))
                if vals:
                    continuous_means[spec.name] = sum(vals) / len(vals)
                    variance = sum((v - continuous_means[spec.name]) ** 2 for v in vals) / len(vals)
                    continuous_stds[spec.name] = max(variance ** 0.5, 1e-6)
                else:
                    continuous_means[spec.name] = 0.0
                    continuous_stds[spec.name] = 1.0

    # Encode samples
    features_list = []
    for values in confounder_values_list:
        features = []

        for spec in specs:
            name = spec.name
            val = values.get(name)
            missing = values.get(f"{name}_missing", val is None)

            if spec.type == "categorical":
                cats = spec.categories or []
                # k-1 dummy coding
                for i, cat in enumerate(cats[1:], 1):
                    if not missing and str(val) == cat:
                        features.append(1.0)
                    else:
                        features.append(0.0)
                features.append(1.0 if missing else 0.0)
            else:
                if not missing and val is not None:
                    mean = continuous_means.get(name, 0.0)
                    std = continuous_stds.get(name, 1.0)
                    features.append((float(val) - mean) / std)
                else:
                    features.append(0.0)
                features.append(1.0 if missing else 0.0)

        features_list.append(features)

    return features_list, feature_names
