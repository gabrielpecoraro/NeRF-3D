"""Extract a colored point cloud from a trained NeRF."""

import numpy as np
import torch
from tqdm import tqdm

from model.nerf_mlp import NeRFMLP
from model.positional_encoding import PositionalEncoding


@torch.no_grad()
def extract_point_cloud(
    fine_model: NeRFMLP,
    pos_encoder: PositionalEncoding,
    dir_encoder: PositionalEncoding,
    device: torch.device,
    resolution: int = 256,
    density_threshold: float = 50.0,
    bound: float = 1.5,
    max_points: int = 100_000,
    chunk: int = 65536,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract colored point cloud by thresholding the density field.

    1. Create a 3D grid and query densities.
    2. Keep points above the threshold.
    3. Query colors at those points.
    4. Subsample if too many points.

    Args:
        fine_model: Trained fine NeRF MLP.
        pos_encoder: Positional encoding for xyz.
        dir_encoder: Positional encoding for directions.
        device: Torch device.
        resolution: Grid resolution (N x N x N).
        density_threshold: Minimum density for a point to be included.
        bound: Half-extent of the query volume.
        max_points: Maximum number of points to return.
        chunk: Points per forward pass.

    Returns:
        points (N, 3), colors (N, 3) as numpy arrays.
    """
    fine_model.eval()

    # Create 3D grid
    x = torch.linspace(-bound, bound, resolution)
    y = torch.linspace(-bound, bound, resolution)
    z = torch.linspace(-bound, bound, resolution)
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing="ij")
    pts = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)

    # Query density
    sigmas = []
    for i in tqdm(range(0, pts.shape[0], chunk), desc="Querying density"):
        chunk_pts = pts[i : i + chunk].to(device)
        pos_enc = pos_encoder(chunk_pts)
        dir_enc = torch.zeros(chunk_pts.shape[0], dir_encoder.output_dim, device=device)
        _, sigma = fine_model(pos_enc, dir_enc)
        sigmas.append(sigma.cpu())

    sigmas = torch.cat(sigmas, 0).squeeze(-1)  # (res^3,)

    # Filter by threshold
    mask = sigmas > density_threshold
    valid_pts = pts[mask]

    if len(valid_pts) == 0:
        print("No points above threshold. Try lowering the threshold.")
        return np.zeros((0, 3)), np.zeros((0, 3))

    # Subsample if too many
    if len(valid_pts) > max_points:
        indices = torch.randperm(len(valid_pts))[:max_points]
        valid_pts = valid_pts[indices]

    # Query colors
    view_dir = torch.tensor([0.0, 0.0, -1.0]).expand(valid_pts.shape[0], -1)
    all_rgb = []
    for i in range(0, valid_pts.shape[0], chunk):
        p = valid_pts[i : i + chunk].to(device)
        d = view_dir[i : i + chunk].to(device)
        pos_enc = pos_encoder(p)
        dir_enc = dir_encoder(d)
        rgb, _ = fine_model(pos_enc, dir_enc)
        all_rgb.append(rgb.cpu())

    colors = torch.cat(all_rgb, 0).numpy()
    colors = np.clip(colors, 0, 1)

    return valid_pts.numpy(), colors


def save_point_cloud(points: np.ndarray, colors: np.ndarray, filepath: str):
    """Save point cloud as PLY file.

    Args:
        points: (N, 3) xyz coordinates.
        colors: (N, 3) RGB values in [0, 1].
        filepath: Output .ply path.
    """
    import trimesh

    vertex_colors = (colors * 255).astype(np.uint8)
    alpha = np.full((vertex_colors.shape[0], 1), 255, dtype=np.uint8)
    vertex_colors = np.hstack([vertex_colors, alpha])

    cloud = trimesh.PointCloud(vertices=points, colors=vertex_colors)
    cloud.export(filepath)
    print(f"Point cloud saved: {filepath} ({len(points)} points)")
