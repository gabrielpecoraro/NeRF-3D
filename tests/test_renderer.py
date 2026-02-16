"""Tests for volumetric rendering."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model.renderer import (
    sample_stratified,
    volume_render,
    render_rays,
)
from model.nerf_mlp import NeRFMLP
from model.positional_encoding import PositionalEncoding


def test_stratified_sampling_shape():
    """Verify output shapes of stratified sampling."""
    rays_o = torch.randn(16, 3)
    rays_d = torch.randn(16, 3)
    t_vals, pts = sample_stratified(rays_o, rays_d, near=2.0, far=6.0, n_samples=64)
    assert t_vals.shape == (16, 64)
    assert pts.shape == (16, 64, 3)


def test_stratified_sampling_range():
    """t_vals should be between near and far."""
    rays_o = torch.randn(8, 3)
    rays_d = torch.randn(8, 3)
    t_vals, _ = sample_stratified(
        rays_o, rays_d, near=2.0, far=6.0, n_samples=64, perturb=False
    )
    assert t_vals.min() >= 2.0 - 1e-5
    assert t_vals.max() <= 6.0 + 1e-5


def test_stratified_sampling_monotonic():
    """t_vals should be monotonically increasing per ray."""
    rays_o = torch.randn(8, 3)
    rays_d = torch.randn(8, 3)
    t_vals, _ = sample_stratified(
        rays_o, rays_d, near=2.0, far=6.0, n_samples=64, perturb=True
    )
    diffs = t_vals[:, 1:] - t_vals[:, :-1]
    assert (diffs >= -1e-6).all(), "t_vals should be approximately monotonic"


def test_volume_render_shapes():
    """Volume rendering output shapes."""
    B, N = 16, 64
    raw_rgb = torch.rand(B, N, 3)
    raw_sigma = torch.rand(B, N, 1)
    t_vals = torch.linspace(2.0, 6.0, N).unsqueeze(0).expand(B, -1)
    rays_d = torch.randn(B, 3)

    result = volume_render(raw_rgb, raw_sigma, t_vals, rays_d)
    assert result["rgb_map"].shape == (B, 3)
    assert result["depth_map"].shape == (B,)
    assert result["weights"].shape == (B, N)
    assert result["acc_map"].shape == (B,)


def test_volume_render_rgb_range():
    """RGB output should be in [0, 1] with white background."""
    B, N = 8, 32
    raw_rgb = torch.rand(B, N, 3)
    raw_sigma = torch.rand(B, N, 1) * 10
    t_vals = torch.linspace(2.0, 6.0, N).unsqueeze(0).expand(B, -1)
    rays_d = torch.randn(B, 3)

    result = volume_render(raw_rgb, raw_sigma, t_vals, rays_d, white_background=True)
    assert result["rgb_map"].min() >= -1e-5
    assert result["rgb_map"].max() <= 1.0 + 1e-5


def test_volume_render_depth_range():
    """Depth should be between near and far."""
    B, N = 8, 32
    raw_rgb = torch.rand(B, N, 3)
    raw_sigma = torch.ones(B, N, 1) * 100  # High density everywhere
    t_vals = torch.linspace(2.0, 6.0, N).unsqueeze(0).expand(B, -1)
    rays_d = torch.ones(B, 3) / (3**0.5)  # Unit direction

    result = volume_render(raw_rgb, raw_sigma, t_vals, rays_d)
    # Depth should be close to near since high density at front
    assert result["depth_map"].min() >= 1.5  # Allow some margin


def test_render_rays_full_pipeline():
    """End-to-end render_rays produces correct shapes."""
    coarse = NeRFMLP(pos_enc_dim=63, dir_enc_dim=27)
    fine = NeRFMLP(pos_enc_dim=63, dir_enc_dim=27)
    pos_enc = PositionalEncoding(10)
    dir_enc = PositionalEncoding(4)

    rays_o = torch.randn(4, 3)
    rays_d = torch.randn(4, 3)

    result = render_rays(
        coarse,
        fine,
        rays_o,
        rays_d,
        pos_enc,
        dir_enc,
        near=2.0,
        far=6.0,
        n_coarse=8,
        n_fine=16,
        chunk_size=1024,
    )

    assert result["rgb_coarse"].shape == (4, 3)
    assert result["rgb_fine"].shape == (4, 3)
    assert result["depth_coarse"].shape == (4,)
    assert result["depth_fine"].shape == (4,)
