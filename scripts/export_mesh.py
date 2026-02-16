"""Export a 3D mesh or point cloud from a trained NeRF checkpoint.

Usage:
    python scripts/export_mesh.py --checkpoint checkpoints/ckpt_200000.pt --format obj
    python scripts/export_mesh.py --checkpoint checkpoints/ckpt_200000.pt --format ply --type pointcloud
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.nerf_mlp import NeRFMLP
from model.positional_encoding import PositionalEncoding
from export.mesh_export import extract_mesh, save_mesh
from export.pointcloud_export import extract_point_cloud, save_point_cloud


def main():
    parser = argparse.ArgumentParser(description="Export 3D from NeRF")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=256, help="Grid resolution")
    parser.add_argument(
        "--threshold", type=float, default=50.0, help="Density threshold"
    )
    parser.add_argument(
        "--format", type=str, default="obj", choices=["obj", "ply", "glb"]
    )
    parser.add_argument(
        "--type", type=str, default="mesh", choices=["mesh", "pointcloud"]
    )
    parser.add_argument("--output_dir", type=str, default="exports")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]

    pos_encoder = PositionalEncoding(config.pos_enc_freqs).to(device)
    dir_encoder = PositionalEncoding(config.dir_enc_freqs).to(device)

    fine_model = NeRFMLP(
        pos_enc_dim=pos_encoder.output_dim,
        dir_enc_dim=dir_encoder.output_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        skip_layer=config.skip_layer,
        color_hidden_dim=config.color_hidden_dim,
    ).to(device)
    fine_model.load_state_dict(checkpoint["fine_model"])
    fine_model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.type == "mesh":
        verts, faces, colors = extract_mesh(
            fine_model,
            pos_encoder,
            dir_encoder,
            device,
            resolution=args.resolution,
            density_threshold=args.threshold,
        )
        if len(verts) == 0:
            print("No mesh extracted. Try lowering the threshold.")
            sys.exit(1)
        filepath = os.path.join(args.output_dir, f"{config.scene}_mesh.{args.format}")
        save_mesh(verts, faces, colors, filepath)
    else:
        pts, colors = extract_point_cloud(
            fine_model,
            pos_encoder,
            dir_encoder,
            device,
            resolution=args.resolution,
            density_threshold=args.threshold,
        )
        if len(pts) == 0:
            print("No points extracted. Try lowering the threshold.")
            sys.exit(1)
        filepath = os.path.join(args.output_dir, f"{config.scene}_pointcloud.ply")
        save_point_cloud(pts, colors, filepath)


if __name__ == "__main__":
    main()
