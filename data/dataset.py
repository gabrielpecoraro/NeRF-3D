"""PyTorch Dataset for NeRF Synthetic (Blender) data."""

import json
import os
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.ray_utils import get_ray_directions, get_rays


class BlenderDataset(Dataset):
    """Dataset for NeRF Synthetic (Blender) scenes.

    Loads all images and camera poses for a given split, pre-generates
    all rays for efficient shuffled training.

    The JSON structure (transforms_{split}.json):
        {
            "camera_angle_x": float,  # Horizontal FOV in radians
            "frames": [
                {
                    "file_path": "./train/r_000",
                    "transform_matrix": [[4x4 float]]
                },
                ...
            ]
        }
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        img_wh: Tuple[int, int] = (400, 400),
        white_background: bool = True,
    ):
        """
        Args:
            root_dir: Path to scene directory (e.g., 'datasets/nerf_synthetic/lego').
            split: One of 'train', 'val', 'test'.
            img_wh: (width, height) to resize images to.
            white_background: Composite RGBA over white background.
        """
        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh
        self.white_background = white_background

        self.W, self.H = img_wh

        # Load transforms JSON
        json_path = os.path.join(root_dir, f"transforms_{split}.json")
        with open(json_path, "r") as f:
            meta = json.load(f)

        # Compute focal length from field of view
        camera_angle_x = meta["camera_angle_x"]
        self.focal = 0.5 * self.W / np.tan(0.5 * camera_angle_x)

        # Pre-compute ray directions for the image grid (shared across all frames)
        self.directions = get_ray_directions(self.H, self.W, self.focal)  # (H, W, 3)

        # Load all images and poses, generate rays
        all_rays_o = []
        all_rays_d = []
        all_rgbs = []
        self.poses = []
        self.images = []

        for frame in meta["frames"]:
            # Load image
            file_path = frame["file_path"]
            if not file_path.endswith(".png"):
                file_path += ".png"
            img_path = os.path.join(root_dir, file_path)

            img = Image.open(img_path).convert("RGBA")
            if img.size != (self.W, self.H):
                img = img.resize((self.W, self.H), Image.LANCZOS)

            img = np.array(img, dtype=np.float32) / 255.0  # (H, W, 4), [0, 1]

            # Alpha compositing
            alpha = img[..., 3:]  # (H, W, 1)
            if white_background:
                rgb = img[..., :3] * alpha + (1.0 - alpha)
            else:
                rgb = img[..., :3] * alpha

            rgb = torch.from_numpy(rgb)  # (H, W, 3)
            self.images.append(rgb)

            # Camera pose
            c2w = torch.FloatTensor(frame["transform_matrix"])[:4, :4]
            self.poses.append(c2w)

            # Generate rays for this image
            rays_o, rays_d = get_rays(self.directions, c2w)  # (H, W, 3) each

            all_rays_o.append(rays_o.reshape(-1, 3))
            all_rays_d.append(rays_d.reshape(-1, 3))
            all_rgbs.append(rgb.reshape(-1, 3))

        # Stack all rays across all images
        self.rays_o = torch.cat(all_rays_o, dim=0)  # (N_total, 3)
        self.rays_d = torch.cat(all_rays_d, dim=0)  # (N_total, 3)
        self.rgbs = torch.cat(all_rgbs, dim=0)  # (N_total, 3)

        self.poses = torch.stack(self.poses, dim=0)  # (N_images, 4, 4)

        print(
            f"[{split}] Loaded {len(self.images)} images, "
            f"{self.rays_o.shape[0]} rays, "
            f"focal={self.focal:.1f}"
        )

    def __len__(self) -> int:
        return self.rays_o.shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {
            "ray_o": self.rays_o[idx],  # (3,)
            "ray_d": self.rays_d[idx],  # (3,)
            "rgb": self.rgbs[idx],  # (3,)
        }

    def get_image_data(self, img_idx: int) -> dict:
        """Get all rays and ground truth for a single image.

        Useful for validation (rendering a full image).

        Args:
            img_idx: Image index within the split.

        Returns:
            Dictionary with rays_o, rays_d, rgb (all (H*W, 3)), and pose (4, 4).
        """
        n_pixels = self.H * self.W
        start = img_idx * n_pixels
        end = start + n_pixels
        return {
            "rays_o": self.rays_o[start:end],
            "rays_d": self.rays_d[start:end],
            "rgb": self.rgbs[start:end],
            "pose": self.poses[img_idx],
        }
