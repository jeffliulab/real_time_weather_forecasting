"""
Upload model checkpoints to HuggingFace Hub.

Usage:
    # Upload best checkpoint for a specific model run
    python scripts/hf_upload.py --model cnn_baseline

    # Upload with a tag/description
    python scripts/hf_upload.py --model cnn_baseline --tag "50ep-lr1e-3" --note "val_loss=0.58 AUC=0.75"

    # Upload norm_stats as well
    python scripts/hf_upload.py --model cnn_baseline --include_norm_stats

HuggingFace repo: https://huggingface.co/jeffliulab/weather-forecasting-v1
Token: read from ~/.hf_token (never commit the token to git)
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

HF_REPO_ID = "jeffliulab/weather-forecasting-v1"
ROOT = Path(__file__).parent.parent


def get_token():
    token_path = Path.home() / ".hf_token"
    if token_path.exists():
        return token_path.read_text().strip()
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    print("ERROR: No HuggingFace token found.")
    print("  Option 1: echo 'hf_xxx' > ~/.hf_token")
    print("  Option 2: export HF_TOKEN=hf_xxx")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Upload checkpoint to HuggingFace Hub")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (e.g. cnn_baseline, cnn_multi_frame, cnn_3d, vit)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Optional tag for this upload (e.g. '50ep-lr1e-3')")
    parser.add_argument("--note", type=str, default=None,
                        help="Optional note (e.g. 'val_loss=0.58 AUC=0.75')")
    parser.add_argument("--include_norm_stats", action="store_true",
                        help="Also upload norm_stats.pt")
    parser.add_argument("--checkpoint", type=str, default="best",
                        choices=["best", "latest"],
                        help="Which checkpoint to upload (default: best)")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: huggingface_hub not installed.")
        print("Run: pip install huggingface_hub --user")
        sys.exit(1)

    token = get_token()
    api = HfApi()

    # Resolve checkpoint path
    ckpt_path = ROOT / "runs" / args.model / "checkpoints" / f"{args.checkpoint}.pt"
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # Destination path in HF repo
    timestamp = datetime.now().strftime("%Y%m%d")
    tag_str = f"_{args.tag}" if args.tag else ""
    dest_name = f"{args.checkpoint}.pt"
    dest_path = f"{args.model}{tag_str}/{dest_name}"

    print(f"Uploading: {ckpt_path}")
    print(f"       To: {HF_REPO_ID}/{dest_path}")
    if args.note:
        print(f"     Note: {args.note}")

    api.upload_file(
        path_or_fileobj=str(ckpt_path),
        path_in_repo=dest_path,
        repo_id=HF_REPO_ID,
        repo_type="model",
        token=token,
        commit_message=f"Upload {args.model}{tag_str} {args.checkpoint}.pt [{timestamp}]"
                       + (f" — {args.note}" if args.note else ""),
    )
    print(f"Done! View at: https://huggingface.co/{HF_REPO_ID}/tree/main/{args.model}{tag_str}/")

    # Optionally upload norm_stats
    if args.include_norm_stats:
        ns_path = ROOT / "norm_stats.pt"
        if ns_path.exists():
            print(f"\nUploading norm_stats.pt...")
            api.upload_file(
                path_or_fileobj=str(ns_path),
                path_in_repo="norm_stats.pt",
                repo_id=HF_REPO_ID,
                repo_type="model",
                token=token,
                commit_message=f"Upload norm_stats.pt [{timestamp}]",
            )
            print("Done!")
        else:
            print(f"WARNING: norm_stats.pt not found at {ns_path}")


if __name__ == "__main__":
    main()
