"""Gradio web UI for NeRF 3D reconstruction viewer.

Three tabs:
    1. Novel View - Orbit camera sliders + render button -> RGB + depth
    2. 3D Model - Interactive 3D mesh viewer (gr.Model3D)
    3. Export - Download OBJ, PLY point cloud, or 360 turntable MP4
"""

import os
import sys

import gradio as gr
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.default import NeRFConfig
from model.nerf_mlp import NeRFMLP
from model.positional_encoding import PositionalEncoding
from app.render_utils import (
    generate_orbit_pose,
    render_novel_view,
    colorize_depth,
    render_turntable_video,
)
from export.mesh_export import extract_mesh, save_mesh
from export.pointcloud_export import extract_point_cloud, save_point_cloud


def load_checkpoint(checkpoint_path: str, device: torch.device):
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

    # Compute focal length
    focal = 0.5 * config.img_width / np.tan(0.5 * 0.6911)  # Default Blender FOV

    return {
        "coarse": coarse_model,
        "fine": fine_model,
        "pos_enc": pos_encoder,
        "dir_enc": dir_encoder,
        "config": config,
        "focal": focal,
    }


def create_app(checkpoint_path: str, export_dir: str = "exports"):
    """Build and return the Gradio app.

    Args:
        checkpoint_path: Path to a trained checkpoint .pt file.
        export_dir: Directory for exported meshes, point clouds, videos.
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    models = load_checkpoint(checkpoint_path, device)
    config = models["config"]

    os.makedirs(export_dir, exist_ok=True)

    H, W = config.img_height, config.img_width
    focal = models["focal"]

    def render_view(theta, phi, radius):
        pose = generate_orbit_pose(theta, phi, radius)
        result = render_novel_view(
            models["coarse"],
            models["fine"],
            models["pos_enc"],
            models["dir_enc"],
            pose,
            H,
            W,
            focal,
            config.near,
            config.far,
            config.n_coarse,
            config.n_fine,
            config.white_background,
            config.chunk_size,
            device,
        )
        rgb = (result["rgb"] * 255).astype(np.uint8)
        depth = colorize_depth(result["depth"], config.near, config.far)
        return rgb, depth

    def export_mesh_callback(resolution, threshold):
        verts, faces, colors = extract_mesh(
            models["fine"],
            models["pos_enc"],
            models["dir_enc"],
            device,
            resolution=int(resolution),
            density_threshold=threshold,
        )
        if len(verts) == 0:
            return None
        filepath = os.path.join(export_dir, f"{config.scene}_mesh.obj")
        save_mesh(verts, faces, colors, filepath)
        return filepath

    def export_glb_callback(resolution, threshold):
        verts, faces, colors = extract_mesh(
            models["fine"],
            models["pos_enc"],
            models["dir_enc"],
            device,
            resolution=int(resolution),
            density_threshold=threshold,
        )
        if len(verts) == 0:
            return None
        filepath = os.path.join(export_dir, f"{config.scene}_mesh.glb")
        save_mesh(verts, faces, colors, filepath)
        return filepath

    def export_pointcloud_callback(resolution, threshold):
        pts, colors = extract_point_cloud(
            models["fine"],
            models["pos_enc"],
            models["dir_enc"],
            device,
            resolution=int(resolution),
            density_threshold=threshold,
        )
        if len(pts) == 0:
            return None
        filepath = os.path.join(export_dir, f"{config.scene}_pointcloud.ply")
        save_point_cloud(pts, colors, filepath)
        return filepath

    def render_video_callback(n_frames, phi, radius):
        filepath = os.path.join(export_dir, f"{config.scene}_turntable.mp4")
        render_turntable_video(
            models["coarse"],
            models["fine"],
            models["pos_enc"],
            models["dir_enc"],
            H,
            W,
            focal,
            config.near,
            config.far,
            config.n_coarse,
            config.n_fine,
            config.white_background,
            config.chunk_size,
            device,
            filepath,
            n_frames=int(n_frames),
            phi=phi,
            radius=radius,
        )
        return filepath

    # --- Build Gradio interface ---
    with gr.Blocks(title="NeRF 3D Viewer", theme=gr.themes.Soft()) as app:
        gr.Markdown(f"# NeRF 3D Reconstruction Viewer")
        gr.Markdown(
            f"**Scene:** {config.scene} | **Device:** {device} | **Resolution:** {W}x{H}"
        )

        with gr.Tabs():
            # ---- Tab 1: Novel View ----
            with gr.Tab("Novel View"):
                with gr.Row():
                    with gr.Column(scale=1):
                        theta_slider = gr.Slider(
                            0, 360, value=0, step=1, label="Azimuth (degrees)"
                        )
                        phi_slider = gr.Slider(
                            -90, 90, value=0, step=1, label="Elevation (degrees)"
                        )
                        radius_slider = gr.Slider(
                            2.0, 6.0, value=4.0, step=0.1, label="Camera Distance"
                        )
                        render_btn = gr.Button("Render", variant="primary")
                    with gr.Column(scale=2):
                        rgb_output = gr.Image(label="Rendered View", type="numpy")
                        depth_output = gr.Image(label="Depth Map", type="numpy")

                render_btn.click(
                    fn=render_view,
                    inputs=[theta_slider, phi_slider, radius_slider],
                    outputs=[rgb_output, depth_output],
                )

            # ---- Tab 2: 3D Model Viewer ----
            with gr.Tab("3D Model"):
                gr.Markdown("Generate a 3D mesh and view it interactively.")
                model_viewer = gr.Model3D(label="3D Reconstruction", height=600)
                with gr.Row():
                    mesh_res = gr.Slider(
                        64, 512, value=256, step=64, label="Grid Resolution"
                    )
                    mesh_thresh = gr.Slider(
                        1, 100, value=50, step=1, label="Density Threshold"
                    )
                generate_btn = gr.Button("Generate 3D Model", variant="primary")
                generate_btn.click(
                    fn=export_glb_callback,
                    inputs=[mesh_res, mesh_thresh],
                    outputs=[model_viewer],
                )

            # ---- Tab 3: Export & Download ----
            with gr.Tab("Export & Download"):
                gr.Markdown("### Download 3D Assets")
                with gr.Row():
                    export_res = gr.Slider(
                        64, 512, value=256, step=64, label="Grid Resolution"
                    )
                    export_thresh = gr.Slider(
                        1, 100, value=50, step=1, label="Density Threshold"
                    )

                with gr.Row():
                    dl_mesh_btn = gr.Button("Export OBJ Mesh")
                    dl_ply_btn = gr.Button("Export PLY Point Cloud")
                download_file = gr.File(label="Download")

                dl_mesh_btn.click(
                    fn=export_mesh_callback,
                    inputs=[export_res, export_thresh],
                    outputs=[download_file],
                )
                dl_ply_btn.click(
                    fn=export_pointcloud_callback,
                    inputs=[export_res, export_thresh],
                    outputs=[download_file],
                )

                gr.Markdown("### 360 Turntable Video")
                with gr.Row():
                    video_frames = gr.Slider(30, 120, value=60, step=10, label="Frames")
                    video_phi = gr.Slider(-45, 45, value=15, step=5, label="Elevation")
                    video_radius = gr.Slider(
                        2.0, 6.0, value=4.0, step=0.1, label="Distance"
                    )
                dl_video_btn = gr.Button("Render 360 Video")
                video_file = gr.File(label="Download Video")
                dl_video_btn.click(
                    fn=render_video_callback,
                    inputs=[video_frames, video_phi, video_radius],
                    outputs=[video_file],
                )

    return app
