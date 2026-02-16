"""Evaluate a trained NeRF model on the test set.

Computes PSNR, SSIM, and LPIPS on all test views.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/ckpt_200000.pt --scene lego
"""

import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.default import NeRFConfig
from data.dataset import BlenderDataset
from model.nerf_mlp import NeRFMLP
from model.positional_encoding import PositionalEncoding
from model.renderer import render_rays
from training.metrics import compute_psnr, compute_ssim, compute_lpips


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained models from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    pos_encoder = PositionalEncoding(config.pos_enc_freqs).to(device)
    dir_encoder = PositionalEncoding(config.dir_enc_freqs).to(device)

    pos_dim = pos_encoder.output_dim
    dir_dim = dir_encoder.output_dim

    coarse_model = NeRFMLP(
        pos_enc_dim=pos_dim,
        dir_enc_dim=dir_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        skip_layer=config.skip_layer,
        color_hidden_dim=config.color_hidden_dim,
    ).to(device)

    fine_model = NeRFMLP(
        pos_enc_dim=pos_dim,
        dir_enc_dim=dir_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        skip_layer=config.skip_layer,
        color_hidden_dim=config.color_hidden_dim,
    ).to(device)

    coarse_model.load_state_dict(checkpoint["coarse_model"])
    fine_model.load_state_dict(checkpoint["fine_model"])
    coarse_model.eval()
    fine_model.eval()

    return coarse_model, fine_model, pos_encoder, dir_encoder, config


@torch.no_grad()
def render_image(
    coarse_model, fine_model, pos_encoder, dir_encoder, rays_o, rays_d, config, device
):
    """Render a full image from rays in chunks."""
    all_rgb = []
    chunk = config.batch_size
    for i in range(0, rays_o.shape[0], chunk):
        ro = rays_o[i : i + chunk].to(device)
        rd = rays_d[i : i + chunk].to(device)
        results = render_rays(
            coarse_model,
            fine_model,
            ro,
            rd,
            pos_encoder,
            dir_encoder,
            near=config.near,
            far=config.far,
            n_coarse=config.n_coarse,
            n_fine=config.n_fine,
            perturb=False,
            white_background=config.white_background,
            chunk_size=config.chunk_size,
        )
        all_rgb.append(results["rgb_fine"].cpu())
    return torch.cat(all_rgb, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Evaluate NeRF")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--scene", type=str, default="lego")
    parser.add_argument("--dataset_dir", type=str, default="datasets/nerf_synthetic")
    parser.add_argument("--output_dir", type=str, default="exports/test_renders")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    coarse_model, fine_model, pos_encoder, dir_encoder, config = load_model(
        args.checkpoint, device
    )

    # Override scene if specified
    config.scene = args.scene
    config.dataset_dir = args.dataset_dir

    test_dataset = BlenderDataset(
        root_dir=os.path.join(config.dataset_dir, config.scene),
        split="test",
        img_wh=(config.img_width, config.img_height),
        white_background=config.white_background,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize LPIPS (may need CPU fallback on MPS)
    try:
        import lpips

        lpips_model = lpips.LPIPS(net="alex").to("cpu")
        use_lpips = True
    except Exception:
        print("LPIPS not available, skipping")
        use_lpips = False

    psnrs, ssims, lpips_vals = [], [], []
    n_images = len(test_dataset.images)

    for idx in tqdm(range(n_images), desc="Evaluating test views"):
        img_data = test_dataset.get_image_data(idx)
        pred_flat = render_image(
            coarse_model,
            fine_model,
            pos_encoder,
            dir_encoder,
            img_data["rays_o"],
            img_data["rays_d"],
            config,
            device,
        )

        H, W = test_dataset.H, test_dataset.W
        pred_img = pred_flat.reshape(H, W, 3).clamp(0, 1)
        target_img = img_data["rgb"].reshape(H, W, 3)

        psnr = compute_psnr(pred_img, target_img)
        psnrs.append(psnr)

        ssim = compute_ssim(pred_img, target_img)
        ssims.append(ssim)

        if use_lpips:
            lp = compute_lpips(pred_img.cpu(), target_img.cpu(), lpips_model)
            lpips_vals.append(lp)

        # Save rendered image
        from PIL import Image

        img_np = (pred_img.numpy() * 255).astype(np.uint8)
        Image.fromarray(img_np).save(
            os.path.join(args.output_dir, f"test_{idx:03d}.png")
        )

        if device.type == "mps":
            torch.mps.empty_cache()

    # Report
    print(f"\n{'=' * 40}")
    print(f"Scene: {config.scene}")
    print(f"Test views: {n_images}")
    print(f"{'=' * 40}")
    print(f"PSNR:  {np.mean(psnrs):.2f} +/- {np.std(psnrs):.2f} dB")
    print(f"SSIM:  {np.mean(ssims):.4f} +/- {np.std(ssims):.4f}")
    if use_lpips:
        print(f"LPIPS: {np.mean(lpips_vals):.4f} +/- {np.std(lpips_vals):.4f}")
    print(f"{'=' * 40}")
    print(f"Rendered images saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
