#!/usr/bin/env python3
"""hf_dataset/push_to_hub.py — upload the packed export (from pack_and_manifest.py)
to a HuggingFace **dataset** repo. Re-runnable: it overwrites changed files, so it
doubles as the "maintain / update" step.

    huggingface-cli login                 # once, stores the token
    <fb>/.venv/bin/python hf_dataset/push_to_hub.py \
        --export   /raid/tongle/rl/fb-data/hf_export \
        --repo-id  <username>/factoredfb-testbed-demos \
        --private                         # optional; start private, share later

Uploads everything under --export (data/*/*.npz, *.meta.json, manifest.json) plus
this folder's README.md as the dataset card.
"""
import argparse
import shutil
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default="/raid/tongle/rl/fb-data/hf_export")
    ap.add_argument("--repo-id", required=True, help="e.g. <username>/factoredfb-testbed-demos")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--token", default=None, help="defaults to the cached login token")
    args = ap.parse_args()

    export = Path(args.export)
    assert (export / "manifest.json").exists(), \
        f"no manifest at {export}; run pack_and_manifest.py first"

    # Ship the dataset card (README.md) alongside the data so the repo renders it.
    card = Path(__file__).with_name("README.md")
    if card.exists():
        shutil.copy(card, export / "README.md")

    api = HfApi(token=args.token)
    create_repo(args.repo_id, repo_type="dataset", private=args.private,
                exist_ok=True, token=args.token)
    api.upload_folder(
        folder_path=str(export),
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Update FactoredFB testbed expert demos",
    )
    print(f"[push] done -> https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
