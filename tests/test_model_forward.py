"""Tests for NeRF MLP model."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model.nerf_mlp import NeRFMLP


def test_forward_shapes():
    """Verify output shapes for standard config."""
    model = NeRFMLP(pos_enc_dim=63, dir_enc_dim=27)
    pos = torch.randn(32, 63)
    dirs = torch.randn(32, 27)
    rgb, sigma = model(pos, dirs)
    assert rgb.shape == (32, 3)
    assert sigma.shape == (32, 1)


def test_rgb_range():
    """RGB output should be in [0, 1] due to sigmoid."""
    model = NeRFMLP(pos_enc_dim=63, dir_enc_dim=27)
    pos = torch.randn(100, 63)
    dirs = torch.randn(100, 27)
    rgb, _ = model(pos, dirs)
    assert rgb.min() >= 0.0
    assert rgb.max() <= 1.0


def test_single_input():
    """Works with batch size 1."""
    model = NeRFMLP(pos_enc_dim=63, dir_enc_dim=27)
    pos = torch.randn(1, 63)
    dirs = torch.randn(1, 27)
    rgb, sigma = model(pos, dirs)
    assert rgb.shape == (1, 3)
    assert sigma.shape == (1, 1)


def test_large_batch():
    """Works with a large batch."""
    model = NeRFMLP(pos_enc_dim=63, dir_enc_dim=27)
    pos = torch.randn(4096, 63)
    dirs = torch.randn(4096, 27)
    rgb, sigma = model(pos, dirs)
    assert rgb.shape == (4096, 3)
    assert sigma.shape == (4096, 1)


def test_skip_connection():
    """Layer 4 must have input dim = hidden_dim + pos_enc_dim."""
    model = NeRFMLP(pos_enc_dim=63, dir_enc_dim=27, hidden_dim=256, skip_layer=4)
    # Layer 4 should accept 256 + 63 = 319 inputs
    assert model.layers[4].in_features == 319


def test_gradient_flow():
    """Gradients flow through both rgb and sigma outputs."""
    model = NeRFMLP(pos_enc_dim=63, dir_enc_dim=27)
    pos = torch.randn(16, 63, requires_grad=True)
    dirs = torch.randn(16, 27, requires_grad=True)
    rgb, sigma = model(pos, dirs)
    loss = rgb.sum() + sigma.sum()
    loss.backward()
    assert pos.grad is not None
    assert dirs.grad is not None


def test_parameter_count():
    """Approximately 600K parameters per model."""
    model = NeRFMLP(pos_enc_dim=63, dir_enc_dim=27)
    total = sum(p.numel() for p in model.parameters())
    assert 400_000 < total < 800_000, f"Unexpected param count: {total}"
