"""Train a NeRF model on a NeRF Synthetic scene.

Usage:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/train.py --scene lego
    PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/train.py --scene lego --batch_size 2048 --num_iterations 100000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.default import NeRFConfig
from data.download import download_nerf_synthetic, verify_scene
from training.trainer import NeRFTrainer


def main():
    parser = argparse.ArgumentParser(description="Train NeRF")
    parser.add_argument("--scene", type=str, default="lego", help="Scene name")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_iterations", type=int, default=200_000)
    parser.add_argument(
        "--img_size", type=int, default=400, help="Image width and height"
    )
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--resume", type=str, default=None, help="Checkpoint path to resume from"
    )
    parser.add_argument("--dataset_dir", type=str, default="datasets/nerf_synthetic")
    args = parser.parse_args()

    config = NeRFConfig(
        scene=args.scene,
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        img_width=args.img_size,
        img_height=args.img_size,
        learning_rate=args.lr,
        dataset_dir=args.dataset_dir,
    )

    # Download dataset if needed
    parent_dir = os.path.dirname(config.dataset_dir)
    if not os.path.exists(config.dataset_dir):
        print(f"Dataset not found at {config.dataset_dir}, downloading...")
        download_nerf_synthetic(parent_dir)

    if not verify_scene(config.dataset_dir, config.scene):
        print(f"ERROR: Scene '{config.scene}' is missing required files.")
        sys.exit(1)

    print(f"Training NeRF on scene: {config.scene}")
    print(f"  Image size: {config.img_width}x{config.img_height}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Iterations: {config.num_iterations}")
    print(f"  Learning rate: {config.learning_rate}")

    trainer = NeRFTrainer(config)
    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
