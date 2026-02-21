"""Tests for model architectures — shape validation and gradient flow."""

import numpy as np
import pytest
import torch

from scalp2.config import ModelConfig
from scalp2.models.gru import GRUEncoder
from scalp2.models.hybrid import HybridEncoder
from scalp2.models.tcn import TCNEncoder


@pytest.fixture
def model_config():
    return ModelConfig()


@pytest.fixture
def batch():
    """Dummy batch: (batch=4, seq_len=64, features=120)."""
    return torch.randn(4, 64, 120)


class TestTCNEncoder:
    def test_output_shape(self, batch):
        tcn = TCNEncoder(input_channels=120, num_channels=[64, 64, 64, 64])
        out = tcn(batch)
        assert out.shape == (4, 64), f"Expected (4, 64), got {out.shape}"

    def test_output_dim_attribute(self):
        tcn = TCNEncoder(input_channels=50, num_channels=[32, 32])
        assert tcn.output_dim == 32

    def test_different_kernel_sizes(self, batch):
        tcn = TCNEncoder(input_channels=120, num_channels=[64, 64], kernel_size=5)
        out = tcn(batch)
        assert out.shape == (4, 64)


class TestGRUEncoder:
    def test_output_shape(self, batch):
        gru = GRUEncoder(input_size=120, hidden_size=128, num_layers=2)
        out = gru(batch)
        assert out.shape == (4, 128), f"Expected (4, 128), got {out.shape}"

    def test_output_dim_attribute(self):
        gru = GRUEncoder(input_size=50, hidden_size=64)
        assert gru.output_dim == 64


class TestHybridEncoder:
    def test_forward_shapes(self, batch, model_config):
        model = HybridEncoder(n_features=120, config=model_config)
        logits, latent = model(batch)
        assert logits.shape == (4, 3), f"Logits: expected (4, 3), got {logits.shape}"
        assert latent.shape == (4, 128), f"Latent: expected (4, 128), got {latent.shape}"

    def test_extract_latent(self, batch, model_config):
        model = HybridEncoder(n_features=120, config=model_config)
        latent = model.extract_latent(batch)
        assert latent.shape == (4, 128)
        assert not latent.requires_grad

    def test_gradient_flow(self, batch, model_config):
        model = HybridEncoder(n_features=120, config=model_config)
        logits, _ = model(batch)
        loss = logits.sum()
        loss.backward()

        # Check all parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.all(param.grad == 0), f"Zero gradient for {name}"

    def test_parameter_count(self, model_config):
        model = HybridEncoder(n_features=120, config=model_config)
        n_params = model.count_parameters()
        # Should be approximately 315K params
        assert 200_000 < n_params < 500_000, f"Unexpected param count: {n_params}"
