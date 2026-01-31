# tests/test_traditional_logreg.py
"""Unit tests for TraditionalLogRegNet causal head."""

import pytest
import torch

from cdt.models.traditional_logreg import TraditionalLogRegNet


class TestTraditionalLogRegNet:
    """Tests for TraditionalLogRegNet module."""

    @pytest.fixture
    def model(self):
        """Create a TraditionalLogRegNet instance."""
        return TraditionalLogRegNet(
            input_dim=128,
            representation_dim=64,
            hidden_outcome_dim=32,
            dropout=0.1
        )

    @pytest.fixture
    def sample_features(self):
        """Create sample input features."""
        batch_size = 8
        return torch.randn(batch_size, 128)

    def test_forward_with_treatment(self, model, sample_features):
        """Test forward pass with observed treatment."""
        batch_size = sample_features.size(0)
        treatment = torch.randint(0, 2, (batch_size,)).float()

        y_logit, t_logit, phi = model(sample_features, treatment=treatment)

        assert y_logit.shape == (batch_size, 1)
        assert t_logit.shape == (batch_size, 1)
        assert phi.shape == (batch_size, 64)  # representation_dim

    def test_forward_counterfactual_mode(self, model, sample_features):
        """Test forward pass in counterfactual mode (treatment=None)."""
        batch_size = sample_features.size(0)

        y0_logit, y1_logit, t_logit, phi = model(sample_features, treatment=None)

        assert y0_logit.shape == (batch_size, 1)
        assert y1_logit.shape == (batch_size, 1)
        assert t_logit.shape == (batch_size, 1)
        assert phi.shape == (batch_size, 64)

    def test_treatment_dimension_handling(self, model, sample_features):
        """Test that treatment tensor with different dims is handled."""
        model.eval()  # Disable dropout for consistent results
        batch_size = sample_features.size(0)

        # 1D treatment tensor
        treatment_1d = torch.randint(0, 2, (batch_size,)).float()
        y_logit_1d, _, _ = model(sample_features, treatment=treatment_1d)

        # 2D treatment tensor
        treatment_2d = treatment_1d.unsqueeze(1)
        y_logit_2d, _, _ = model(sample_features, treatment=treatment_2d)

        # Should produce same output
        assert torch.allclose(y_logit_1d, y_logit_2d)

    def test_get_representation(self, model, sample_features):
        """Test get_representation method."""
        batch_size = sample_features.size(0)

        phi = model.get_representation(sample_features)

        assert phi.shape == (batch_size, 64)

    def test_propensity_from_representation(self, model, sample_features):
        """Test propensity_from_representation method."""
        batch_size = sample_features.size(0)

        phi = model.get_representation(sample_features)
        t_logit = model.propensity_from_representation(phi)

        assert t_logit.shape == (batch_size, 1)

    def test_outcome_from_representation(self, model, sample_features):
        """Test outcome_from_representation method."""
        batch_size = sample_features.size(0)
        treatment = torch.randint(0, 2, (batch_size,)).float()

        phi = model.get_representation(sample_features)
        y_logit = model.outcome_from_representation(phi, treatment)

        assert y_logit.shape == (batch_size, 1)

    def test_counterfactual_consistency(self, model, sample_features):
        """Test that forward with treatment produces same output as counterfactual mode."""
        model.eval()  # Disable dropout for consistent results
        batch_size = sample_features.size(0)

        # Get counterfactual predictions
        y0_logit_cf, y1_logit_cf, _, _ = model(sample_features, treatment=None)

        # Get predictions with treatment=0
        treatment_0 = torch.zeros(batch_size)
        y_logit_0, _, _ = model(sample_features, treatment=treatment_0)

        # Get predictions with treatment=1
        treatment_1 = torch.ones(batch_size)
        y_logit_1, _, _ = model(sample_features, treatment=treatment_1)

        # Should match counterfactual outputs
        assert torch.allclose(y0_logit_cf, y_logit_0)
        assert torch.allclose(y1_logit_cf, y_logit_1)

    def test_gradient_flow(self, model, sample_features):
        """Test that gradients flow properly."""
        batch_size = sample_features.size(0)
        treatment = torch.randint(0, 2, (batch_size,)).float()

        y_logit, t_logit, phi = model(sample_features, treatment=treatment)

        # Compute loss and backprop
        loss = y_logit.sum() + t_logit.sum()
        loss.backward()

        # Check gradients exist
        assert model.representation_fc1.weight.grad is not None
        assert model.outcome_fc1.weight.grad is not None
        assert model.propensity_fc.weight.grad is not None

    def test_batch_size_one(self, model):
        """Test with batch size of 1."""
        features = torch.randn(1, 128)
        treatment = torch.tensor([1.0])

        y_logit, t_logit, phi = model(features, treatment=treatment)

        assert y_logit.shape == (1, 1)
        assert t_logit.shape == (1, 1)
        assert phi.shape == (1, 64)


class TestTraditionalLogRegNetWithCausalText:
    """Integration tests with CausalText."""

    @pytest.fixture
    def model(self):
        """Create a CausalText model with traditional_logreg head."""
        from cdt.models import CausalText

        return CausalText(
            feature_extractor_type="cnn",
            model_type="traditional_logreg",
            embedding_dim=64,
            kernel_sizes=[3],
            num_kmeans_filters=8,
            num_random_filters=8,
            dragonnet_representation_dim=32,
            dragonnet_hidden_outcome_dim=16,
            device="cpu"
        )

    @pytest.fixture
    def sample_texts(self):
        """Sample texts for testing."""
        return [
            "Patient has diabetes and hypertension.",
            "No significant medical history.",
            "Metastatic cancer with poor prognosis.",
            "Healthy individual for routine checkup."
        ]

    def test_fit_tokenizer(self, model, sample_texts):
        """Test that tokenizer can be fit."""
        model.fit_tokenizer(sample_texts)

        # Verify tokenizer was initialized
        assert hasattr(model.feature_extractor, 'tokenizer')

    def test_forward(self, model, sample_texts):
        """Test forward pass through complete model."""
        model.fit_tokenizer(sample_texts)

        y0_logit, y1_logit, t_logit, phi = model(sample_texts)

        assert y0_logit.shape == (4, 1)
        assert y1_logit.shape == (4, 1)
        assert t_logit.shape == (4, 1)

    def test_train_step(self, model, sample_texts):
        """Test training step."""
        model.fit_tokenizer(sample_texts)
        model.train()

        batch = {
            'texts': sample_texts,
            'treatment': torch.tensor([1.0, 0.0, 1.0, 0.0]),
            'outcome': torch.tensor([1.0, 0.0, 1.0, 0.0])
        }

        losses = model.train_step(
            batch,
            alpha_propensity=1.0,
            label_smoothing=0.0
        )

        assert 'loss' in losses
        assert 'outcome_loss' in losses
        assert 'propensity_loss' in losses
        assert 'targreg_loss' in losses
        assert 'y0_logit' in losses
        assert 'y1_logit' in losses
        assert 't_logit' in losses

        # targreg_loss should be 0 for traditional_logreg
        assert losses['targreg_loss'].item() == 0.0

    def test_train_step_with_stop_grad_propensity(self, model, sample_texts):
        """Test training step with stop_grad_propensity=True."""
        model.fit_tokenizer(sample_texts)
        model.train()

        batch = {
            'texts': sample_texts,
            'treatment': torch.tensor([1.0, 0.0, 1.0, 0.0]),
            'outcome': torch.tensor([1.0, 0.0, 1.0, 0.0])
        }

        losses = model.train_step(
            batch,
            alpha_propensity=1.0,
            stop_grad_propensity=True
        )

        assert 'loss' in losses
        # Loss should still be computed even with stop_grad
        assert losses['loss'].item() > 0

    def test_predict(self, model, sample_texts):
        """Test prediction."""
        model.fit_tokenizer(sample_texts)
        model.eval()

        preds = model.predict(sample_texts)

        assert 'y0_prob' in preds
        assert 'y1_prob' in preds
        assert 'propensity' in preds
        assert 'tau_pred' in preds

        # Probabilities should be in [0, 1]
        assert (preds['y0_prob'] >= 0).all() and (preds['y0_prob'] <= 1).all()
        assert (preds['y1_prob'] >= 0).all() and (preds['y1_prob'] <= 1).all()
        assert (preds['propensity'] >= 0).all() and (preds['propensity'] <= 1).all()

        # tau_pred is computed from logit differences (y1_logit - y0_logit)
        # ITE on probability scale is y1_prob - y0_prob
        # These are related but not identical. Verify tau_pred has correct shape.
        assert preds['tau_pred'].shape == (4,)

        # Verify ITE can be computed from y1_prob - y0_prob
        ite_prob = preds['y1_prob'] - preds['y0_prob']
        assert ite_prob.shape == (4,)

    def test_net_type(self, model):
        """Test that correct network type is instantiated."""
        from cdt.models.traditional_logreg import TraditionalLogRegNet

        assert isinstance(model.net, TraditionalLogRegNet)
        assert model.model_type == "traditional_logreg"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
