"""Tests for the Blender dataset loader.

These tests use mocked data since the actual dataset needs to be downloaded.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image


def _create_mock_scene(tmpdir: str, n_frames: int = 3, img_size: int = 32):
    """Create a minimal mock Blender scene for testing."""
    # Create train directory
    train_dir = os.path.join(tmpdir, "train")
    os.makedirs(train_dir)

    frames = []
    for i in range(n_frames):
        # Create a random RGBA image
        img = np.random.randint(0, 255, (img_size, img_size, 4), dtype=np.uint8)
        img_path = os.path.join(train_dir, f"r_{i:03d}.png")
        Image.fromarray(img).save(img_path)

        # Random camera pose (4x4 identity + jitter)
        pose = np.eye(4)
        pose[:3, 3] = np.random.randn(3) * 2
        frames.append(
            {
                "file_path": f"./train/r_{i:03d}",
                "transform_matrix": pose.tolist(),
            }
        )

    # Write transforms.json
    transforms = {
        "camera_angle_x": 0.6911,
        "frames": frames,
    }
    with open(os.path.join(tmpdir, "transforms_train.json"), "w") as f:
        json.dump(transforms, f)

    return tmpdir


def test_dataset_loads():
    """Dataset loads mock data without errors."""
    from data.dataset import BlenderDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_mock_scene(tmpdir, n_frames=2, img_size=16)
        ds = BlenderDataset(root_dir=tmpdir, split="train", img_wh=(16, 16))
        assert len(ds) == 2 * 16 * 16  # 2 images * 16 * 16 pixels


def test_dataset_item_shapes():
    """Each item has correct shapes."""
    from data.dataset import BlenderDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_mock_scene(tmpdir, n_frames=2, img_size=16)
        ds = BlenderDataset(root_dir=tmpdir, split="train", img_wh=(16, 16))
        item = ds[0]
        assert item["ray_o"].shape == (3,)
        assert item["ray_d"].shape == (3,)
        assert item["rgb"].shape == (3,)


def test_dataset_rgb_range():
    """RGB values should be in [0, 1]."""
    from data.dataset import BlenderDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_mock_scene(tmpdir, n_frames=1, img_size=16)
        ds = BlenderDataset(root_dir=tmpdir, split="train", img_wh=(16, 16))
        assert ds.rgbs.min() >= 0.0
        assert ds.rgbs.max() <= 1.0


def test_focal_length():
    """Focal length computed from FOV."""
    from data.dataset import BlenderDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_mock_scene(tmpdir, n_frames=1, img_size=32)
        ds = BlenderDataset(root_dir=tmpdir, split="train", img_wh=(32, 32))
        # focal = 0.5 * W / tan(0.5 * camera_angle_x)
        expected = 0.5 * 32 / np.tan(0.5 * 0.6911)
        assert abs(ds.focal - expected) < 1e-3


def test_get_image_data():
    """get_image_data returns correct shapes."""
    from data.dataset import BlenderDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_mock_scene(tmpdir, n_frames=3, img_size=16)
        ds = BlenderDataset(root_dir=tmpdir, split="train", img_wh=(16, 16))
        img_data = ds.get_image_data(0)
        assert img_data["rays_o"].shape == (16 * 16, 3)
        assert img_data["rays_d"].shape == (16 * 16, 3)
        assert img_data["rgb"].shape == (16 * 16, 3)
        assert img_data["pose"].shape == (4, 4)


def test_dataset_resize():
    """Dataset resizes images correctly."""
    from data.dataset import BlenderDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_mock_scene(tmpdir, n_frames=1, img_size=64)
        ds = BlenderDataset(root_dir=tmpdir, split="train", img_wh=(32, 32))
        # 1 image at 32x32 = 1024 rays
        assert len(ds) == 1024
