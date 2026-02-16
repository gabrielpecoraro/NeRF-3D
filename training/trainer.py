"""NeRF training loop with MPS optimization."""

import os
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config.default import NeRFConfig
from data.dataset import BlenderDataset
from model.nerf_mlp import NeRFMLP
from model.positional_encoding import PositionalEncoding
from model.renderer import render_rays
from training.losses import nerf_loss
from training.metrics import compute_psnr


class NeRFTrainer:
    """Orchestrates NeRF training with MPS-optimized settings."""

    def __init__(self, config: NeRFConfig):
        self.config = config
        self.device = self._get_device()
        self.global_step = 0

        # Positional encoders
        self.pos_encoder = PositionalEncoding(config.pos_enc_freqs).to(self.device)
        self.dir_encoder = PositionalEncoding(config.dir_enc_freqs).to(self.device)

        # Coarse and fine models
        pos_dim = self.pos_encoder.output_dim
        dir_dim = self.dir_encoder.output_dim

        self.coarse_model = NeRFMLP(
            pos_enc_dim=pos_dim,
            dir_enc_dim=dir_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            skip_layer=config.skip_layer,
            color_hidden_dim=config.color_hidden_dim,
        ).to(self.device)

        self.fine_model = NeRFMLP(
            pos_enc_dim=pos_dim,
            dir_enc_dim=dir_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            skip_layer=config.skip_layer,
            color_hidden_dim=config.color_hidden_dim,
        ).to(self.device)

        # Optional torch.compile
        if config.use_torch_compile and hasattr(torch, "compile"):
            try:
                self.coarse_model = torch.compile(self.coarse_model)
                self.fine_model = torch.compile(self.fine_model)
                print("torch.compile() applied successfully")
            except Exception as e:
                print(f"torch.compile() failed: {e}, continuing without compilation")

        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.coarse_model.parameters()) + list(self.fine_model.parameters()),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
        )

        # LR scheduler: exponential decay
        # lr(t) = lr_init * decay_rate^(t / decay_steps)
        decay_gamma = config.lr_decay_rate ** (1.0 / config.lr_decay_steps)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=decay_gamma
        )

        param_count = sum(
            p.numel()
            for p in list(self.coarse_model.parameters())
            + list(self.fine_model.parameters())
        )
        print(f"Total parameters: {param_count:,} (coarse + fine)")
        print(f"Device: {self.device}")

    def _get_device(self) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("WARNING: No GPU found, falling back to CPU")
            return torch.device("cpu")

    def train(self):
        """Run the full training loop."""
        config = self.config

        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

        # Load datasets
        scene_dir = os.path.join(config.dataset_dir, config.scene)
        train_dataset = BlenderDataset(
            root_dir=scene_dir,
            split="train",
            img_wh=(config.img_width, config.img_height),
            white_background=config.white_background,
        )
        val_dataset = BlenderDataset(
            root_dir=scene_dir,
            split="val",
            img_wh=(config.img_width, config.img_height),
            white_background=config.white_background,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )

        writer = SummaryWriter(config.log_dir)

        self.coarse_model.train()
        self.fine_model.train()

        pbar = tqdm(
            total=config.num_iterations, initial=self.global_step, desc="Training"
        )
        epoch = 0
        start_time = time.time()

        while self.global_step < config.num_iterations:
            for batch in train_loader:
                if self.global_step >= config.num_iterations:
                    break

                rays_o = batch["ray_o"].to(self.device)
                rays_d = batch["ray_d"].to(self.device)
                target_rgb = batch["rgb"].to(self.device)

                # Forward pass
                results = render_rays(
                    self.coarse_model,
                    self.fine_model,
                    rays_o,
                    rays_d,
                    self.pos_encoder,
                    self.dir_encoder,
                    near=config.near,
                    far=config.far,
                    n_coarse=config.n_coarse,
                    n_fine=config.n_fine,
                    perturb=config.perturb,
                    white_background=config.white_background,
                    chunk_size=config.chunk_size,
                )

                # Compute loss
                total_loss, loss_c, loss_f = nerf_loss(
                    results["rgb_coarse"], results["rgb_fine"], target_rgb
                )

                # Backward
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                # Logging
                if self.global_step % config.log_every == 0:
                    psnr = compute_psnr(results["rgb_fine"], target_rgb)
                    lr = self.optimizer.param_groups[0]["lr"]
                    elapsed = time.time() - start_time
                    writer.add_scalar("train/loss", total_loss.item(), self.global_step)
                    writer.add_scalar(
                        "train/loss_coarse", loss_c.item(), self.global_step
                    )
                    writer.add_scalar(
                        "train/loss_fine", loss_f.item(), self.global_step
                    )
                    writer.add_scalar("train/psnr", psnr, self.global_step)
                    writer.add_scalar("train/lr", lr, self.global_step)
                    pbar.set_postfix(
                        loss=f"{total_loss.item():.4f}",
                        psnr=f"{psnr:.2f}",
                        lr=f"{lr:.2e}",
                    )

                # Validation
                if self.global_step > 0 and self.global_step % config.val_every == 0:
                    val_psnr = self._validate(val_dataset, self.global_step, writer)
                    tqdm.write(f"[Step {self.global_step}] Val PSNR: {val_psnr:.2f} dB")

                # Checkpoint
                if self.global_step > 0 and self.global_step % config.save_every == 0:
                    self._save_checkpoint(self.global_step)

                # MPS memory management
                if self.device.type == "mps" and self.global_step % 1000 == 0:
                    torch.mps.empty_cache()

                self.global_step += 1
                pbar.update(1)

            epoch += 1

        pbar.close()
        self._save_checkpoint(self.global_step)
        writer.close()

        total_time = time.time() - start_time
        print(
            f"Training complete. {self.global_step} steps in {total_time / 3600:.1f} hours"
        )

    @torch.no_grad()
    def _validate(
        self, val_dataset: BlenderDataset, step: int, writer: SummaryWriter
    ) -> float:
        """Render one validation image and compute PSNR."""
        self.coarse_model.eval()
        self.fine_model.eval()

        config = self.config
        img_data = val_dataset.get_image_data(0)
        rays_o = img_data["rays_o"]
        rays_d = img_data["rays_d"]
        target_rgb = img_data["rgb"]

        # Render in chunks
        all_rgb = []
        chunk_rays = config.batch_size
        for i in range(0, rays_o.shape[0], chunk_rays):
            ro = rays_o[i : i + chunk_rays].to(self.device)
            rd = rays_d[i : i + chunk_rays].to(self.device)
            results = render_rays(
                self.coarse_model,
                self.fine_model,
                ro,
                rd,
                self.pos_encoder,
                self.dir_encoder,
                near=config.near,
                far=config.far,
                n_coarse=config.n_coarse,
                n_fine=config.n_fine,
                perturb=False,
                white_background=config.white_background,
                chunk_size=config.chunk_size,
            )
            all_rgb.append(results["rgb_fine"].cpu())

        pred_rgb = torch.cat(all_rgb, dim=0)  # (H*W, 3)
        pred_img = pred_rgb.reshape(val_dataset.H, val_dataset.W, 3)
        target_img = target_rgb.reshape(val_dataset.H, val_dataset.W, 3)

        val_psnr = compute_psnr(pred_img, target_img)
        writer.add_scalar("val/psnr", val_psnr, step)

        # Log image to TensorBoard
        writer.add_image(
            "val/predicted",
            pred_img.permute(2, 0, 1).clamp(0, 1),
            step,
        )
        writer.add_image(
            "val/ground_truth",
            target_img.permute(2, 0, 1).clamp(0, 1),
            step,
        )

        self.coarse_model.train()
        self.fine_model.train()

        if self.device.type == "mps":
            torch.mps.empty_cache()

        return val_psnr

    def _save_checkpoint(self, step: int):
        """Save model weights and optimizer state."""
        path = os.path.join(self.config.checkpoint_dir, f"ckpt_{step:06d}.pt")
        torch.save(
            {
                "global_step": step,
                "coarse_model": self.coarse_model.state_dict(),
                "fine_model": self.fine_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.config,
            },
            path,
        )
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load model weights and optimizer state from a checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.coarse_model.load_state_dict(checkpoint["coarse_model"])
        self.fine_model.load_state_dict(checkpoint["fine_model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.global_step = checkpoint["global_step"]
        print(f"Resumed from step {self.global_step}: {path}")
