"""Rendering utilities for the Gradio app: orbit camera and novel view synthesis."""

import math

import numpy as np
import torch

from model.nerf_mlp import NeRFMLP
from model.positional_encoding import PositionalEncoding
from model.renderer import render_rays
from data.ray_utils import get_ray_directions, get_rays


def generate_orbit_pose(theta: float, phi: float, radius: float = 4.0) -> torch.Tensor:
    """Generate a camera-to-world matrix for an orbit camera.

    The camera orbits around the origin, always looking at [0, 0, 0].

    Args:
        theta: Azimuthal angle in degrees (0-360, horizontal orbit).
        phi: Polar/elevation angle in degrees (-90 to 90).
        radius: Distance from origin.

    Returns:
        (4, 4) camera-to-world transformation matrix.
    """
    theta_rad = math.radians(theta)
    phi_rad = math.radians(phi)

    # Camera position on sphere
    x = radius * math.cos(phi_rad) * math.sin(theta_rad)
    y = radius * math.sin(phi_rad)
    z = radius * math.cos(phi_rad) * math.cos(theta_rad)

    camera_pos = torch.tensor([x, y, z], dtype=torch.float32)

    # Look-at construction
    forward = -camera_pos / camera_pos.norm()  # Points toward origin
    world_up = torch.tensor([0.0, 1.0, 0.0])

    right = torch.cross(forward, world_up)
    if right.norm() < 1e-6:
        # Camera is looking straight up/down; use a fallback up vector
        world_up = torch.tensor([0.0, 0.0, 1.0])
        right = torch.cross(forward, world_up)
    right = right / right.norm()

    up = torch.cross(right, forward)
    up = up / up.norm()

    c2w = torch.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = -forward  # Camera looks along -z in its own frame
    c2w[:3, 3] = camera_pos

    return c2w


@torch.no_grad()
def render_novel_view(
    coarse_model: NeRFMLP,
    fine_model: NeRFMLP,
    pos_encoder: PositionalEncoding,
    dir_encoder: PositionalEncoding,
    pose: torch.Tensor,
    H: int,
    W: int,
    focal: float,
    near: float,
    far: float,
    n_coarse: int,
    n_fine: int,
    white_background: bool,
    chunk_size: int,
    device: torch.device,
    chunk_rays: int = 512,
) -> dict:
    """Render a complete image from a novel camera pose.

    Processes rays in chunks to manage MPS memory.

    Args:
        coarse_model: Coarse NeRF MLP.
        fine_model: Fine NeRF MLP.
        pos_encoder: Positional encoding for xyz.
        dir_encoder: Positional encoding for directions.
        pose: (4, 4) camera-to-world matrix.
        H, W: Image dimensions.
        focal: Focal length in pixels.
        near, far: Near and far planes.
        n_coarse, n_fine: Sample counts.
        white_background: Composite over white.
        chunk_size: Points per network forward pass.
        device: Torch device.
        chunk_rays: Rays per rendering chunk.

    Returns:
        Dictionary with 'rgb' (H, W, 3) and 'depth' (H, W) numpy arrays.
    """
    coarse_model.eval()
    fine_model.eval()

    directions = get_ray_directions(H, W, focal).to(device)
    rays_o, rays_d = get_rays(directions, pose.to(device))
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)

    all_rgb = []
    all_depth = []

    for i in range(0, rays_o.shape[0], chunk_rays):
        ro = rays_o[i : i + chunk_rays]
        rd = rays_d[i : i + chunk_rays]
        results = render_rays(
            coarse_model,
            fine_model,
            ro,
            rd,
            pos_encoder,
            dir_encoder,
            near=near,
            far=far,
            n_coarse=n_coarse,
            n_fine=n_fine,
            perturb=False,
            white_background=white_background,
            chunk_size=chunk_size,
        )
        all_rgb.append(results["rgb_fine"].cpu())
        all_depth.append(results["depth_fine"].cpu())

    rgb = torch.cat(all_rgb, 0).reshape(H, W, 3).clamp(0, 1).numpy()
    depth = torch.cat(all_depth, 0).reshape(H, W).numpy()

    if device.type == "mps":
        torch.mps.empty_cache()

    return {"rgb": rgb, "depth": depth}


def colorize_depth(
    depth: np.ndarray, near: float = 2.0, far: float = 6.0
) -> np.ndarray:
    """Convert a depth map to a colored visualization.

    Args:
        depth: (H, W) depth values.
        near: Near plane for normalization.
        far: Far plane for normalization.

    Returns:
        (H, W, 3) uint8 RGB image using a viridis-like colormap.
    """
    import matplotlib.cm as cm

    normalized = (depth - near) / (far - near + 1e-8)
    normalized = np.clip(normalized, 0, 1)
    colored = cm.viridis(normalized)[..., :3]  # Drop alpha
    return (colored * 255).astype(np.uint8)


def render_turntable_video(
    coarse_model: NeRFMLP,
    fine_model: NeRFMLP,
    pos_encoder: PositionalEncoding,
    dir_encoder: PositionalEncoding,
    H: int,
    W: int,
    focal: float,
    near: float,
    far: float,
    n_coarse: int,
    n_fine: int,
    white_background: bool,
    chunk_size: int,
    device: torch.device,
    output_path: str,
    n_frames: int = 60,
    phi: float = 15.0,
    radius: float = 4.0,
    fps: int = 30,
):
    """Render a 360-degree turntable animation and save as MP4.

    Args:
        output_path: Path to save the MP4 video.
        n_frames: Number of frames in the animation.
        phi: Elevation angle in degrees.
        radius: Distance from origin.
        fps: Frames per second.
    """
    import imageio

    frames = []
    for i in range(n_frames):
        theta = 360.0 * i / n_frames
        pose = generate_orbit_pose(theta, phi, radius)
        result = render_novel_view(
            coarse_model,
            fine_model,
            pos_encoder,
            dir_encoder,
            pose,
            H,
            W,
            focal,
            near,
            far,
            n_coarse,
            n_fine,
            white_background,
            chunk_size,
            device,
        )
        frame = (result["rgb"] * 255).astype(np.uint8)
        frames.append(frame)
        print(f"Frame {i + 1}/{n_frames} rendered")

    imageio.mimwrite(output_path, frames, fps=fps, quality=8)
    print(f"Video saved: {output_path}")
