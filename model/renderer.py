import torch
import torch.nn.functional as F
from typing import Dict, Optional

from model.nerf_mlp import NeRFMLP
from model.positional_encoding import PositionalEncoding


def sample_stratified(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float,
    far: float,
    n_samples: int,
    perturb: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stratified sampling along rays.

    Divides [near, far] into n_samples uniform bins and samples one point
    per bin, with optional random jitter within each bin.

    Args:
        rays_o: (B, 3) ray origins.
        rays_d: (B, 3) ray directions.
        near: Near plane distance.
        far: Far plane distance.
        n_samples: Number of sample points per ray.
        perturb: If True, add random offset within each bin.

    Returns:
        t_vals: (B, n_samples) distances along rays.
        pts: (B, n_samples, 3) sampled 3D points.
    """
    batch_size = rays_o.shape[0]
    t_vals = torch.linspace(near, far, n_samples, device=rays_o.device)
    t_vals = t_vals.unsqueeze(0).expand(batch_size, -1)  # (B, n_samples)

    if perturb:
        # Width of each bin
        mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat([mids, t_vals[..., -1:]], dim=-1)
        lower = torch.cat([t_vals[..., :1], mids], dim=-1)
        t_rand = torch.rand_like(t_vals)
        t_vals = lower + (upper - lower) * t_rand

    # pts = o + t * d
    pts = rays_o[:, None, :] + rays_d[:, None, :] * t_vals[:, :, None]  # (B, N, 3)
    return t_vals, pts


def sample_hierarchical(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    t_coarse: torch.Tensor,
    weights_coarse: torch.Tensor,
    n_fine: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hierarchical (importance) sampling based on coarse weights.

    Concentrates additional samples in regions where the coarse network
    predicted high density.

    Args:
        rays_o: (B, 3) ray origins.
        rays_d: (B, 3) ray directions.
        t_coarse: (B, N_coarse) coarse sample distances.
        weights_coarse: (B, N_coarse) volume rendering weights from coarse pass.
        n_fine: Number of additional fine samples.

    Returns:
        t_vals: (B, N_coarse + n_fine) combined sorted sample distances.
        pts: (B, N_coarse + n_fine, 3) sampled 3D points.
    """
    # Build PDF from coarse weights (avoid sampling at boundaries)
    weights = weights_coarse[:, 1:-1] + 1e-5  # (B, N_coarse - 2)
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat(
        [torch.zeros_like(cdf[:, :1]), cdf], dim=-1
    )  # (B, N_coarse - 2 + 1)

    # Sample uniform values for inverse CDF
    u = torch.rand(rays_o.shape[0], n_fine, device=rays_o.device)
    u = u.contiguous()

    # Inverse CDF sampling
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)

    inds_g = torch.stack([below, above], dim=-1)  # (B, n_fine, 2)

    # Gather CDF and bin edges
    cdf_g = torch.gather(cdf, 1, inds_g.reshape(cdf.shape[0], -1)).reshape(
        *inds_g.shape
    )

    mids = 0.5 * (t_coarse[:, 1:] + t_coarse[:, :-1])
    bins = torch.cat([t_coarse[:, :1], mids, t_coarse[:, -1:]], dim=-1)
    bins_g = torch.gather(bins, 1, inds_g.reshape(bins.shape[0], -1)).reshape(
        *inds_g.shape
    )

    # Linear interpolation within bins
    denom = cdf_g[:, :, 1] - cdf_g[:, :, 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t_fine = bins_g[:, :, 0] + (u - cdf_g[:, :, 0]) / denom * (
        bins_g[:, :, 1] - bins_g[:, :, 0]
    )

    # Merge coarse and fine samples and sort
    t_vals, _ = torch.sort(torch.cat([t_coarse, t_fine], dim=-1), dim=-1)
    pts = rays_o[:, None, :] + rays_d[:, None, :] * t_vals[:, :, None]

    return t_vals, pts


def volume_render(
    raw_rgb: torch.Tensor,
    raw_sigma: torch.Tensor,
    t_vals: torch.Tensor,
    rays_d: torch.Tensor,
    white_background: bool = True,
) -> Dict[str, torch.Tensor]:
    """Volume rendering via alpha compositing.

    Implements the classical NeRF rendering equation:
        C(r) = sum_i T_i * alpha_i * c_i
        T_i = prod_{j<i} (1 - alpha_j)
        alpha_i = 1 - exp(-sigma_i * delta_i)

    Args:
        raw_rgb: (B, N, 3) predicted colors at sample points.
        raw_sigma: (B, N, 1) predicted densities at sample points.
        t_vals: (B, N) sample distances along rays.
        rays_d: (B, 3) ray directions (for scaling distances).
        white_background: Composite over white background.

    Returns:
        Dictionary with rgb_map (B, 3), depth_map (B,), weights (B, N), acc_map (B,).
    """
    # Distances between adjacent samples
    dists = t_vals[..., 1:] - t_vals[..., :-1]  # (B, N-1)
    dists = torch.cat([dists, torch.full_like(dists[:, :1], 1e10)], dim=-1)  # (B, N)
    # Scale by ray direction magnitude
    dists = dists * torch.norm(rays_d[:, None, :], dim=-1)

    # Density to alpha
    sigma = F.relu(raw_sigma.squeeze(-1))  # (B, N)
    alpha = 1.0 - torch.exp(-sigma * dists)  # (B, N)

    # Accumulated transmittance
    T = torch.cumprod(
        torch.cat(
            [torch.ones(alpha.shape[0], 1, device=alpha.device), 1.0 - alpha + 1e-10],
            dim=-1,
        ),
        dim=-1,
    )[:, :-1]  # (B, N)

    # Compositing weights
    weights = alpha * T  # (B, N)

    # Final color
    rgb_map = torch.sum(weights[..., None] * raw_rgb, dim=-2)  # (B, 3)

    # Accumulated opacity
    acc_map = torch.sum(weights, dim=-1)  # (B,)

    # White background compositing
    if white_background:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    # Depth map (expected termination distance)
    depth_map = torch.sum(weights * t_vals, dim=-1)  # (B,)

    return {
        "rgb_map": rgb_map,
        "depth_map": depth_map,
        "weights": weights,
        "acc_map": acc_map,
    }


def batchify_forward(
    model: NeRFMLP,
    pos_enc: torch.Tensor,
    dir_enc: torch.Tensor,
    chunk_size: int = 32768,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Process points through the MLP in chunks to manage memory on MPS.

    Args:
        model: NeRF MLP model.
        pos_enc: (N, pos_enc_dim) encoded positions.
        dir_enc: (N, dir_enc_dim) encoded directions.
        chunk_size: Max points per forward pass.

    Returns:
        rgb: (N, 3) colors.
        sigma: (N, 1) densities.
    """
    all_rgb = []
    all_sigma = []
    for i in range(0, pos_enc.shape[0], chunk_size):
        rgb, sigma = model(pos_enc[i : i + chunk_size], dir_enc[i : i + chunk_size])
        all_rgb.append(rgb)
        all_sigma.append(sigma)
    return torch.cat(all_rgb, dim=0), torch.cat(all_sigma, dim=0)


def render_rays(
    coarse_model: NeRFMLP,
    fine_model: NeRFMLP,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    pos_encoder: PositionalEncoding,
    dir_encoder: PositionalEncoding,
    near: float = 2.0,
    far: float = 6.0,
    n_coarse: int = 64,
    n_fine: int = 128,
    perturb: bool = True,
    white_background: bool = True,
    chunk_size: int = 32768,
) -> Dict[str, torch.Tensor]:
    """Full NeRF rendering pipeline for a batch of rays.

    1. Stratified coarse sampling
    2. Coarse network evaluation + volume rendering
    3. Hierarchical fine sampling from coarse weights
    4. Fine network evaluation + volume rendering

    Args:
        coarse_model: Coarse NeRF MLP.
        fine_model: Fine NeRF MLP.
        rays_o: (B, 3) ray origins.
        rays_d: (B, 3) ray directions.
        pos_encoder: Positional encoding for xyz.
        dir_encoder: Positional encoding for view direction.
        near: Near plane.
        far: Far plane.
        n_coarse: Number of coarse samples.
        n_fine: Number of additional fine samples.
        perturb: Apply stratified jitter.
        white_background: Composite over white.
        chunk_size: Points per network forward pass.

    Returns:
        Dictionary with rgb_coarse, rgb_fine, depth_coarse, depth_fine.
    """
    # Normalize directions for encoding (unit vectors)
    viewdirs = F.normalize(rays_d, dim=-1)

    # --- Coarse pass ---
    t_coarse, pts_coarse = sample_stratified(
        rays_o, rays_d, near, far, n_coarse, perturb
    )

    # Encode positions and directions
    B, N_c = pts_coarse.shape[:2]
    pos_enc_coarse = pos_encoder(pts_coarse.reshape(-1, 3))  # (B*N_c, 63)
    dir_enc_coarse = dir_encoder(
        viewdirs[:, None, :].expand(-1, N_c, -1).reshape(-1, 3)
    )  # (B*N_c, 27)

    # Forward through coarse model
    rgb_c, sigma_c = batchify_forward(
        coarse_model, pos_enc_coarse, dir_enc_coarse, chunk_size
    )
    rgb_c = rgb_c.reshape(B, N_c, 3)
    sigma_c = sigma_c.reshape(B, N_c, 1)

    # Volume render coarse
    coarse_results = volume_render(rgb_c, sigma_c, t_coarse, rays_d, white_background)

    # --- Fine pass (hierarchical) ---
    t_fine, pts_fine = sample_hierarchical(
        rays_o, rays_d, t_coarse, coarse_results["weights"].detach(), n_fine
    )

    B, N_f = pts_fine.shape[:2]
    pos_enc_fine = pos_encoder(pts_fine.reshape(-1, 3))
    dir_enc_fine = dir_encoder(viewdirs[:, None, :].expand(-1, N_f, -1).reshape(-1, 3))

    rgb_f, sigma_f = batchify_forward(
        fine_model, pos_enc_fine, dir_enc_fine, chunk_size
    )
    rgb_f = rgb_f.reshape(B, N_f, 3)
    sigma_f = sigma_f.reshape(B, N_f, 1)

    fine_results = volume_render(rgb_f, sigma_f, t_fine, rays_d, white_background)

    return {
        "rgb_coarse": coarse_results["rgb_map"],
        "rgb_fine": fine_results["rgb_map"],
        "depth_coarse": coarse_results["depth_map"],
        "depth_fine": fine_results["depth_map"],
        "weights_coarse": coarse_results["weights"],
        "weights_fine": fine_results["weights"],
        "acc_coarse": coarse_results["acc_map"],
        "acc_fine": fine_results["acc_map"],
    }
