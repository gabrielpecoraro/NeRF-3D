"""Tests for positional encoding module."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model.positional_encoding import PositionalEncoding


def test_output_dim_xyz():
    """PE with L=10 for 3D input: output = 3 + 3*2*10 = 63."""
    pe = PositionalEncoding(num_frequencies=10)
    assert pe.output_dim == 63


def test_output_dim_dir():
    """PE with L=4 for 3D input: output = 3 + 3*2*4 = 27."""
    pe = PositionalEncoding(num_frequencies=4)
    assert pe.output_dim == 27


def test_output_dim_no_identity():
    """Without identity, L=10: output = 3*2*10 = 60."""
    pe = PositionalEncoding(num_frequencies=10, include_identity=False)
    assert pe.output_dim == 60


def test_output_shape():
    """Verify output tensor shape matches output_dim."""
    pe = PositionalEncoding(num_frequencies=10)
    x = torch.randn(32, 3)
    out = pe(x)
    assert out.shape == (32, 63)


def test_batch_dimensions():
    """Works with arbitrary batch dimensions."""
    pe = PositionalEncoding(num_frequencies=4)
    x = torch.randn(8, 16, 3)
    out = pe(x)
    assert out.shape == (8, 16, 27)


def test_deterministic():
    """Same input produces same output."""
    pe = PositionalEncoding(num_frequencies=10)
    x = torch.randn(10, 3)
    out1 = pe(x)
    out2 = pe(x)
    assert torch.allclose(out1, out2)


def test_frequency_bands():
    """Frequency bands are [1, 2, 4, ..., 512] for L=10."""
    pe = PositionalEncoding(num_frequencies=10)
    expected = 2.0 ** torch.arange(10)
    assert torch.allclose(pe.freq_bands, expected)


def test_identity_included():
    """First 3 values of output should be the raw input when include_identity=True."""
    pe = PositionalEncoding(num_frequencies=2, include_identity=True)
    x = torch.tensor([[1.0, 2.0, 3.0]])
    out = pe(x)
    assert torch.allclose(out[0, :3], x[0])
