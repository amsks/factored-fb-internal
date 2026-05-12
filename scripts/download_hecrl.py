"""scripts/download_hecrl.py — Download HECRL datasets from Hugging Face.

Usage:
    python scripts/download_hecrl.py                     # all known datasets
    python scripts/download_hecrl.py ppp-cube            # single dataset
    python scripts/download_hecrl.py ppp-cube --target_dir datasets/hecrl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "DanHrmti/hecrl"

DATASETS = {
    "ppp-cube": "ppp-cube-noisy-v0-mv.npz",
}


def main():
    ap = argparse.ArgumentParser(
        description="Download HECRL's pre-collected datasets from Hugging Face."
    )
    ap.add_argument(
        "datasets",
        nargs="*",
        default=list(DATASETS),
        help=f"Datasets to download. Default: all. Known: {sorted(DATASETS)}",
    )
    ap.add_argument(
        "--target_dir",
        default="datasets/hecrl",
        help="Directory to write the .npz files into (default: datasets/hecrl).",
    )
    args = ap.parse_args()

    target = Path(args.target_dir)
    target.mkdir(parents=True, exist_ok=True)

    for name in args.datasets:
        if name not in DATASETS:
            raise SystemExit(f"Unknown dataset {name!r}. Known: {sorted(DATASETS)}")
        print(f"Downloading {name} -> {target}")
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=DATASETS[name],
            local_dir=str(target),
        )
    print("Done.")


if __name__ == "__main__":
    main()
