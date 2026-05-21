"""scripts/fb_pixel_curves.py — FB (and optionally GCIQL) pixel eval curves.

Loads FB pixel run histories from a wandb_pull cache and produces per-task +
aggregate IQM success curves (95% stratified-bootstrap CI) and a final IQM
success table, reusing the method-generic machinery in fb_gciql_curves.

GCIQL pixel runs are not available yet; pass --gciql <eval.csv root> to add
them when they exist (the curves + table then include both methods).

Usage:
    python scripts/fb_pixel_curves.py \
        --fb analysis/wandb_data_fbpixel \
        --out analysis/fb_pixel
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.fb_gciql_curves import load_fb, load_gciql, render


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fb", default=str(REPO_ROOT / "analysis"
                                        / "wandb_data_fbpixel"))
    ap.add_argument("--gciql", default=None,
                    help="optional GCIQL pixel eval.csv root (reserved slot)")
    ap.add_argument("--out", default=str(REPO_ROOT / "analysis" / "fb_pixel"))
    args = ap.parse_args()

    methods = {"FB": load_fb(args.fb)}
    if args.gciql:
        methods["GCIQL"] = load_gciql(args.gciql)

    print(f"[fb_pixel_curves] methods={list(methods)} "
          f"FB seeds={sorted(methods['FB'])}")
    render(methods, Path(args.out), title_prefix="FB (pixels)",
           n_seeds_note="n=5 seeds")


if __name__ == "__main__":
    main()
