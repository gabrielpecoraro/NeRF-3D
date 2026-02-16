"""Extract a triangle mesh from a trained NeRF using marching cubes."""

import numpy as np
import torch
from tqdm import tqdm

from model.nerf_mlp import NeRFMLP
from model.positional_encoding import PositionalEncoding


@torch.no_grad()
def extract_mesh(
    fine_model: NeRFMLP,
    pos_encoder: PositionalEncoding,
    dir_encoder: PositionalEncoding,
    device: torch.device,
    resolution: int = 256,
    density_threshold: float = 50.0,
    bound: float = 1.5,
    chunk: int = 65536,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract mesh via marching cubes on the density field.

    1. Create a 3D grid in [-bound, bound]^3.
    2. Query density at each grid point through the fine network.
    3. Run marching cubes at the threshold isosurface.
    4. Query colors at mesh vertices.

    Args:
        fine_model: Trained fine NeRF MLP.
        pos_encoder: Positional encoding for xyz.
        dir_encoder: Positional encoding for directions.
        device: Torch device.
        resolution: Grid resolution (N x N x N).
        density_threshold: Isosurface level for marching cubes.
        bound: Half-extent of the query volume.
        chunk: Points per forward pass.

    Returns:
        vertices (V, 3), faces (F, 3), colors (V, 3) as numpy arrays.
    """
    from skimage.measure import marching_cubes

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

    sigma_volume = (
        torch.cat(sigmas, 0)
        .squeeze(-1)
        .reshape(resolution, resolution, resolution)
        .numpy()
    )

    # Marching cubes
    spacing = (2 * bound / resolution,) * 3
    try:
        vertices, faces, normals, _ = marching_cubes(
            sigma_volume, level=density_threshold, spacing=spacing
        )
    except ValueError as e:
        print(f"Marching cubes failed: {e}")
        print("Try lowering the density threshold.")
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int), np.zeros((0, 3))

    # Shift vertices to world coordinates (marching_cubes outputs in grid space)
    vertices = vertices - bound

    # Query colors at vertex positions
    colors = _query_vertex_colors(
        fine_model, pos_encoder, dir_encoder, vertices, device, chunk
    )

    return vertices, faces, colors


@torch.no_grad()
def _query_vertex_colors(
    fine_model: NeRFMLP,
    pos_encoder: PositionalEncoding,
    dir_encoder: PositionalEncoding,
    vertices: np.ndarray,
    device: torch.device,
    chunk: int = 65536,
) -> np.ndarray:
    """Query RGB colors at vertex positions using an average viewing direction.

    Args:
        fine_model: Trained fine NeRF MLP.
        pos_encoder: Positional encoding for xyz.
        dir_encoder: Positional encoding for directions.
        vertices: (V, 3) vertex positions.
        device: Torch device.
        chunk: Points per forward pass.

    Returns:
        colors: (V, 3) RGB values in [0, 1].
    """
    pts = torch.from_numpy(vertices).float()
    # Use a fixed viewing direction (looking from +z)
    view_dir = torch.tensor([0.0, 0.0, -1.0]).expand(pts.shape[0], -1)

    all_rgb = []
    for i in range(0, pts.shape[0], chunk):
        p = pts[i : i + chunk].to(device)
        d = view_dir[i : i + chunk].to(device)
        pos_enc = pos_encoder(p)
        dir_enc = dir_encoder(d)
        rgb, _ = fine_model(pos_enc, dir_enc)
        all_rgb.append(rgb.cpu())

    colors = torch.cat(all_rgb, 0).numpy()
    return np.clip(colors, 0, 1)


def save_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    filepath: str,
):
    """Save mesh to file (OBJ, PLY, or GLB).

    Args:
        vertices: (V, 3) vertex positions.
        faces: (F, 3) face indices.
        colors: (V, 3) vertex colors in [0, 1].
        filepath: Output path. Format inferred from extension.
    """
    import trimesh

    vertex_colors = (colors * 255).astype(np.uint8)
    # Add alpha channel
    alpha = np.full((vertex_colors.shape[0], 1), 255, dtype=np.uint8)
    vertex_colors = np.hstack([vertex_colors, alpha])

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors)
    mesh.export(filepath)
    print(f"Mesh saved: {filepath} ({len(vertices)} vertices, {len(faces)} faces)")
