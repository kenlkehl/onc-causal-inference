# cdt/models/intra_batch_contrastive.py
"""Intra-batch contrastive learning for improved confounder detection.

Clusters samples by representation similarity, then applies supervised contrastive
loss (SupCon) within each cluster. This encourages the model to learn features that
discriminate treatment/outcome among otherwise similar patients.

Reference: Khosla et al. (2020) "Supervised Contrastive Learning"
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class IntraBatchContrastiveLoss(nn.Module):
    """
    Intra-batch contrastive loss with K-means clustering.

    Architecture:
        features -> projection_head -> L2-normalize -> cluster (detached) -> SupCon within clusters

    The projection head maps features to a separate contrastive space so the loss
    doesn't distort features consumed by causal heads (SimCLR/SupCon convention).

    Args:
        feature_dim: Dimension of input features
        num_clusters: Number of K-means clusters (K)
        temperature: SupCon temperature (lower = sharper similarity)
        label_mode: How to construct labels - "treatment", "outcome", or "joint" (T*2+Y)
        projection_dim: Dimension of projection head output
        min_cluster_size: Minimum samples per cluster to compute loss
        clustering_method: "kmeans" or "random"
    """

    def __init__(
        self,
        feature_dim: int,
        num_clusters: int = 4,
        temperature: float = 0.1,
        label_mode: str = "joint",
        projection_dim: int = 64,
        min_cluster_size: int = 2,
        clustering_method: str = "kmeans",
    ):
        super().__init__()

        self.num_clusters = num_clusters
        self.temperature = temperature
        self.label_mode = label_mode
        self.projection_dim = projection_dim
        self.min_cluster_size = min_cluster_size
        self.clustering_method = clustering_method

        # 2-layer projection head (Linear -> ReLU -> Linear)
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, projection_dim),
        )

        logger.info(
            f"IntraBatchContrastiveLoss: K={num_clusters}, tau={temperature}, "
            f"mode={label_mode}, proj_dim={projection_dim}, "
            f"clustering={clustering_method}"
        )

    def forward(
        self,
        features: torch.Tensor,
        treatment: torch.Tensor,
        outcome: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute intra-batch contrastive loss.

        Args:
            features: (batch, feature_dim) - extracted text features
            treatment: (batch,) - binary treatment indicators
            outcome: (batch,) - binary outcome indicators

        Returns:
            Scalar contrastive loss (0 if no valid clusters)
        """
        batch_size = features.size(0)

        # Edge case: batch too small for any meaningful contrastive learning
        if batch_size < 4:
            return torch.tensor(0.0, device=features.device)

        # Project features to contrastive space and L2-normalize
        z = self.projection_head(features)
        z = F.normalize(z, dim=1)

        # Construct labels based on mode
        labels = self._construct_labels(treatment, outcome)

        # Cluster assignment (no gradient through assignments)
        if self.clustering_method == "random":
            assignments = torch.randint(
                0, max(1, min(self.num_clusters, batch_size // 2)),
                (batch_size,), device=features.device,
            )
        else:
            K = min(self.num_clusters, max(1, batch_size // 2))
            assignments = self._batch_kmeans(features.detach(), K)

        # Compute SupCon loss within each valid cluster
        total_loss = torch.tensor(0.0, device=features.device)
        valid_clusters = 0

        unique_clusters = assignments.unique()
        for c in unique_clusters:
            mask = assignments == c
            cluster_size = mask.sum().item()

            # Skip clusters that are too small
            if cluster_size < self.min_cluster_size:
                continue

            cluster_z = z[mask]
            cluster_labels = labels[mask]

            # Skip clusters with all-same labels (no negative pairs)
            if cluster_labels.unique().size(0) < 2:
                continue

            loss = self._supervised_contrastive_loss(
                cluster_z, cluster_labels, self.temperature
            )

            if loss > 0:
                total_loss = total_loss + loss
                valid_clusters += 1

        if valid_clusters > 0:
            total_loss = total_loss / valid_clusters

        return total_loss

    def _construct_labels(
        self, treatment: torch.Tensor, outcome: torch.Tensor
    ) -> torch.Tensor:
        """Construct contrastive labels from treatment and outcome."""
        treatment_int = (treatment > 0.5).long()
        outcome_int = (outcome > 0.5).long()

        if self.label_mode == "treatment":
            return treatment_int
        elif self.label_mode == "outcome":
            return outcome_int
        else:  # "joint"
            return treatment_int * 2 + outcome_int

    def _batch_kmeans(
        self,
        features: torch.Tensor,
        K: int,
        num_iterations: int = 5,
    ) -> torch.Tensor:
        """
        GPU-friendly K-means clustering.

        Args:
            features: (N, D) feature matrix (detached)
            K: Number of clusters
            num_iterations: Number of Lloyd's iterations

        Returns:
            (N,) cluster assignments
        """
        N = features.size(0)
        if K <= 1:
            return torch.zeros(N, dtype=torch.long, device=features.device)

        # Initialize centroids with K-means++ style: first random, rest by distance
        indices = torch.randperm(N, device=features.device)[:K]
        centroids = features[indices].clone()

        for _ in range(num_iterations):
            # Assign each point to nearest centroid
            # (N, K) distance matrix
            dists = torch.cdist(features, centroids)
            assignments = dists.argmin(dim=1)

            # Update centroids
            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(K, device=features.device)

            for k in range(K):
                mask = assignments == k
                count = mask.sum()
                if count > 0:
                    new_centroids[k] = features[mask].mean(dim=0)
                    counts[k] = count
                else:
                    # Empty cluster: steal farthest point from largest cluster
                    largest = counts.argmax()
                    largest_mask = assignments == largest
                    largest_dists = dists[largest_mask, largest]
                    farthest_idx = largest_dists.argmax()
                    # Get the global index
                    global_indices = torch.where(largest_mask)[0]
                    steal_idx = global_indices[farthest_idx]
                    new_centroids[k] = features[steal_idx]
                    counts[k] = 1

            centroids = new_centroids

        # Final assignment
        dists = torch.cdist(features, centroids)
        assignments = dists.argmin(dim=1)
        return assignments

    @staticmethod
    def _supervised_contrastive_loss(
        projections: torch.Tensor,
        labels: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """
        Supervised contrastive loss (Khosla et al. 2020).

        For each anchor i, positives P(i) = {j != i : label_j == label_i}
        L_i = -(1/|P(i)|) * sum_{p in P(i)} log(
            exp(sim(z_i, z_p) / tau) / sum_{k != i} exp(sim(z_i, z_k) / tau)
        )

        Args:
            projections: (N, D) L2-normalized projections
            labels: (N,) integer labels
            temperature: Temperature scaling

        Returns:
            Scalar SupCon loss
        """
        N = projections.size(0)
        if N < 2:
            return torch.tensor(0.0, device=projections.device)

        # Cosine similarity matrix (already L2-normalized)
        sim_matrix = torch.mm(projections, projections.t()) / temperature

        # Mask out self-similarity (diagonal)
        self_mask = torch.eye(N, dtype=torch.bool, device=projections.device)
        sim_matrix = sim_matrix.masked_fill(self_mask, -1e9)

        # For numerical stability, subtract max
        sim_max, _ = sim_matrix.max(dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_max.detach()

        # Log-sum-exp denominator (over all k != i)
        log_sum_exp = torch.logsumexp(sim_matrix, dim=1)  # (N,)

        # Positive mask: same label, not self
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (N, N)
        positive_mask = labels_eq & ~self_mask

        # Count positives per anchor
        num_positives = positive_mask.sum(dim=1).float()  # (N,)

        # Skip anchors with no positives
        valid_anchors = num_positives > 0
        if not valid_anchors.any():
            return torch.tensor(0.0, device=projections.device)

        # Sum of log-probs for positive pairs
        # For each anchor i: sum_{p in P(i)} (sim(i,p)/tau - log_sum_exp(i))
        log_probs = sim_matrix - log_sum_exp.unsqueeze(1)  # (N, N)
        positive_log_probs = (positive_mask.float() * log_probs).sum(dim=1)  # (N,)

        # Average over positives per anchor, then over valid anchors
        loss_per_anchor = -positive_log_probs / num_positives.clamp(min=1)
        loss = loss_per_anchor[valid_anchors].mean()

        return loss
