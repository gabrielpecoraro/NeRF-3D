import torch
from typing import Tuple


def get_ray_directions(H: int, W: int, focal: float) -> torch.Tensor:
    """Generate ray directions for all pixels in camera space (pinhole model).

    Uses OpenGL/NeRF convention: x-right, y-up, z-backward (out of screen).

    Args:
        H: Image height.
        W: Image width.
        focal: Focal length in pixels.

    Returns:
        (H, W, 3) ray direction vectors in camera coordinates.
    """
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32),
        torch.arange(H, dtype=torch.float32),
        indexing="xy",
    )
    directions = torch.stack(
        [
            (i - W * 0.5) / focal,
            -(j - H * 0.5) / focal,
            -torch.ones_like(i),
        ],
        dim=-1,
    )
    return directions


def get_rays(
    directions: torch.Tensor, c2w: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Transform ray directions from camera space to world space.

    Args:
        directions: (H, W, 3) ray directions in camera coordinates.
        c2w: (4, 4) camera-to-world transformation matrix.

    Returns:
        rays_o: (H, W, 3) ray origins in world space.
        rays_d: (H, W, 3) ray directions in world space (not normalized).
    """
    rays_d = torch.sum(directions[..., None, :] * c2w[:3, :3], dim=-1)
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    return rays_o, rays_d
