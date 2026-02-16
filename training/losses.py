"""Loss functions for NeRF training."""

import torch
import torch.nn.functional as F


def nerf_loss(
    rgb_coarse: torch.Tensor,
    rgb_fine: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute NeRF training loss.

    Total loss = MSE(coarse, target) + MSE(fine, target).
    Both terms weighted equally per the original paper.

    Args:
        rgb_coarse: (B, 3) coarse network predictions.
        rgb_fine: (B, 3) fine network predictions.
        target: (B, 3) ground truth pixel colors.

    Returns:
        Tuple of (total_loss, loss_coarse, loss_fine).
    """
    loss_coarse = F.mse_loss(rgb_coarse, target)
    loss_fine = F.mse_loss(rgb_fine, target)
    total = loss_coarse + loss_fine
    return total, loss_coarse, loss_fine
