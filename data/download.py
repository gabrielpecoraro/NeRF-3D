"""Download the NeRF Synthetic (Blender) dataset."""

import os
import zipfile

import requests
from tqdm import tqdm


# Hugging Face mirror (reliable, no rate limits)
DATASET_URL = "https://huggingface.co/datasets/XayahHina/nerf_synthetic/resolve/main/nerf_synthetic.zip?download=true"
AVAILABLE_SCENES = [
    "chair",
    "drums",
    "ficus",
    "hotdog",
    "lego",
    "materials",
    "mic",
    "ship",
]


def download_nerf_synthetic(target_dir: str = "datasets") -> str:
    """Download and extract the NeRF Synthetic dataset.

    Downloads from a Hugging Face mirror using requests with a progress bar.

    Args:
        target_dir: Parent directory where nerf_synthetic/ will be created.

    Returns:
        Path to the extracted nerf_synthetic directory.
    """
    output_dir = os.path.join(target_dir, "nerf_synthetic")

    # Check if already downloaded
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        print(f"Dataset already exists at {output_dir}")
        return output_dir

    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, "nerf_synthetic.zip")

    print("Downloading NeRF Synthetic dataset (~1.3GB) from Hugging Face...")
    response = requests.get(DATASET_URL, stream=True, timeout=30)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with open(zip_path, "wb") as f, tqdm(
        total=total_size, unit="B", unit_scale=True, desc="Downloading"
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # The zip may extract scenes directly (no nerf_synthetic/ parent).
        # Check the first entry to detect the structure.
        top_entries = {name.split("/")[0] for name in zf.namelist() if "/" in name}
        has_parent = "nerf_synthetic" in top_entries

        if has_parent:
            zf.extractall(target_dir)
        else:
            # Scenes are at the zip root — extract into the nerf_synthetic subdir
            os.makedirs(output_dir, exist_ok=True)
            zf.extractall(output_dir)

    # Clean up zip
    os.remove(zip_path)
    print(f"Dataset extracted to {output_dir}")

    # Verify
    for scene in AVAILABLE_SCENES:
        scene_dir = os.path.join(output_dir, scene)
        if not os.path.exists(scene_dir):
            print(f"  WARNING: Expected scene directory not found: {scene_dir}")

    return output_dir


def verify_scene(dataset_dir: str, scene: str) -> bool:
    """Verify that a scene has the expected files.

    Args:
        dataset_dir: Path to nerf_synthetic directory.
        scene: Scene name (e.g., 'lego').

    Returns:
        True if all expected files are present.
    """
    scene_dir = os.path.join(dataset_dir, scene)
    required = [
        "transforms_train.json",
        "transforms_val.json",
        "transforms_test.json",
    ]
    for f in required:
        if not os.path.exists(os.path.join(scene_dir, f)):
            print(f"Missing: {os.path.join(scene_dir, f)}")
            return False
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download NeRF Synthetic dataset")
    parser.add_argument("--target_dir", type=str, default="datasets")
    args = parser.parse_args()
    download_nerf_synthetic(args.target_dir)
