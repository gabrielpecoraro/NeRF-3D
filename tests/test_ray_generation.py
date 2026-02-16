"""Tests for ray generation utilities."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from data.ray_utils import get_ray_directions, get_rays


def test_ray_directions_shape():
    """Output shape is (H, W, 3)."""
    dirs = get_ray_directions(100, 200, focal=250.0)
    assert dirs.shape == (100, 200, 3)


def test_center_ray_direction():
    """Ray through image center should point along -z in camera space."""
    H, W, focal = 100, 100, 50.0
    dirs = get_ray_directions(H, W, focal)
    # Center pixel
    center = dirs[H // 2, W // 2]
    # Should be approximately (0, 0, -1)
    assert abs(center[0].item()) < 0.02  # x near 0
    assert abs(center[1].item()) < 0.02  # y near 0
    assert center[2].item() < 0  # z negative


def test_get_rays_identity_pose():
    """With identity pose, ray origins are at origin, directions unchanged."""
    dirs = get_ray_directions(10, 10, focal=10.0)
    c2w = torch.eye(4)
    rays_o, rays_d = get_rays(dirs, c2w)

    assert rays_o.shape == (10, 10, 3)
    assert rays_d.shape == (10, 10, 3)

    # Origins should all be (0, 0, 0)
    assert torch.allclose(rays_o, torch.zeros(10, 10, 3))

    # Directions should equal the camera-space directions (identity rotation)
    assert torch.allclose(rays_d, dirs, atol=1e-6)


def test_get_rays_translated_pose():
    """With a translated pose, ray origins should reflect the translation."""
    dirs = get_ray_directions(5, 5, focal=5.0)
    c2w = torch.eye(4)
    c2w[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
    rays_o, rays_d = get_rays(dirs, c2w)

    # All origins should be (1, 2, 3)
    for i in range(5):
        for j in range(5):
            assert torch.allclose(rays_o[i, j], torch.tensor([1.0, 2.0, 3.0]))


def test_get_rays_rotated_pose():
    """Rotation should transform ray directions."""
    dirs = get_ray_directions(5, 5, focal=5.0)
    # 90-degree rotation around y-axis
    c2w = torch.eye(4)
    c2w[0, 0] = 0
    c2w[0, 2] = 1
    c2w[2, 0] = -1
    c2w[2, 2] = 0
    rays_o, rays_d = get_rays(dirs, c2w)

    # Directions should be rotated
    assert rays_d.shape == (5, 5, 3)
    # Verify they are different from unrotated
    assert not torch.allclose(rays_d, dirs)
