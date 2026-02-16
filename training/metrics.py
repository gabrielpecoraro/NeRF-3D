"""Evaluation metrics for NeRF: PSNR, SSIM, LPIPS."""

import torch
import torch.nn.functional as F


def compute_psnr(predicted: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Peak Signal-to-Noise Ratio.

    PSNR = -10 * log10(MSE)
    For NeRF Synthetic, expected: 25-33 dB depending on scene.

    Args:
        predicted: (..., 3) predicted RGB in [0, 1].
        target: (..., 3) ground truth RGB in [0, 1].

    Returns:
        PSNR value in dB.
    """
    mse = F.mse_loss(predicted, target)
    if mse == 0:
        return float("inf")
    return (-10.0 * torch.log10(mse)).item()


def compute_ssim(predicted: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Structural Similarity Index.

    Expects images in (H, W, 3) format with values in [0, 1].

    Args:
        predicted: (H, W, 3) predicted image.
        target: (H, W, 3) ground truth image.

    Returns:
        SSIM value (higher is better, max 1.0).
    """
    from pytorch_msssim import ssim

    # Reshape to (1, 3, H, W) for the library
    pred = predicted.permute(2, 0, 1).unsqueeze(0)
    tgt = target.permute(2, 0, 1).unsqueeze(0)
    return ssim(pred, tgt, data_range=1.0).item()


def compute_lpips(predicted: torch.Tensor, target: torch.Tensor, lpips_model) -> float:
    """Compute Learned Perceptual Image Patch Similarity.

    Uses a pre-initialized LPIPS model (AlexNet backbone).
    Lower is better. Expected: 0.02-0.08 for NeRF Synthetic.

    Args:
        predicted: (H, W, 3) predicted image in [0, 1].
        target: (H, W, 3) ground truth image in [0, 1].
        lpips_model: Pre-initialized lpips.LPIPS model.

    Returns:
        LPIPS distance (lower is better).
    """
    # Scale from [0, 1] to [-1, 1] as required by LPIPS
    pred = predicted.permute(2, 0, 1).unsqueeze(0) * 2 - 1
    tgt = target.permute(2, 0, 1).unsqueeze(0) * 2 - 1
    return lpips_model(pred, tgt).item()
