"""scripts/build_paper_fragment.py — assemble PAPER/<subdir>/ from the
phase-probe and representation-profile aggregates (full-arc narrative).

Usage:
    python scripts/build_paper_fragment.py \\
      --phase-root analysis/phase_probe/aggregate \\
      --repr-root analysis/representation_profile/aggregate \\
      --paper-subdir fb-cube-failure
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase-root",
                    default=str(REPO_ROOT / "analysis" / "phase_probe"
                                / "aggregate"))
    ap.add_argument("--repr-root",
                    default=str(REPO_ROOT / "analysis"
                                / "representation_profile" / "aggregate"))
    ap.add_argument("--out", default=str(REPO_ROOT / "PAPER"))
    ap.add_argument("--paper-subdir", default="fb-cube-failure")
    args = ap.parse_args()

    from evals.paper_fragment import build_fragment

    base = build_fragment(args.phase_root, args.repr_root, args.out,
                          args.paper_subdir)
    print(f"[build_paper_fragment] wrote {base}")


if __name__ == "__main__":
    main()
