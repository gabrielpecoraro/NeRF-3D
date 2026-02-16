"""Launch the Gradio NeRF viewer app.

Usage:
    python scripts/launch_app.py --checkpoint checkpoints/ckpt_200000.pt
    python scripts/launch_app.py --checkpoint checkpoints/ckpt_200000.pt --port 7860 --share
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.gradio_app import create_app


def main():
    parser = argparse.ArgumentParser(description="Launch NeRF Viewer")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint .pt file"
    )
    parser.add_argument("--export_dir", type=str, default="exports")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--share", action="store_true", help="Create a public Gradio link"
    )
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    print(f"Loading checkpoint: {args.checkpoint}")
    app = create_app(args.checkpoint, args.export_dir)

    print(f"Launching on http://localhost:{args.port}")
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
