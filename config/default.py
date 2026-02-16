from dataclasses import dataclass


@dataclass
class NeRFConfig:
    # --- Scene ---
    scene: str = "lego"
    dataset_dir: str = "datasets/nerf_synthetic"

    # --- Image ---
    img_width: int = 400
    img_height: int = 400
    white_background: bool = True

    # --- Model ---
    pos_enc_freqs: int = 10
    dir_enc_freqs: int = 4
    hidden_dim: int = 256
    num_layers: int = 8
    skip_layer: int = 4
    color_hidden_dim: int = 128
    use_viewdirs: bool = True

    # --- Sampling ---
    near: float = 2.0
    far: float = 6.0
    n_coarse: int = 64
    n_fine: int = 128
    perturb: bool = True

    # --- Training ---
    batch_size: int = 1024
    num_iterations: int = 200_000
    learning_rate: float = 5e-4
    lr_decay_steps: int = 250_000
    lr_decay_rate: float = 0.1

    # --- MPS Optimization ---
    chunk_size: int = 32768
    use_torch_compile: bool = False
    pin_memory: bool = False
    num_workers: int = 0

    # --- Logging ---
    log_every: int = 100
    val_every: int = 5000
    save_every: int = 10000
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # --- Export ---
    export_dir: str = "exports"
    mesh_resolution: int = 256
    density_threshold: float = 50.0
