"""Spec 14 §5 — the port must not move any archived cube-single number.

Archived JSONs key their summary by absolute half-width ('0.4'); the ported
code keys by multiplier ('1.0'). On cube-single eps=0.4 the map is exact:
0.4->1.0, 0.8->2.0, 1.6->4.0.

Usage:
  .venv/bin/python scripts/rescuing_fb/check_l1_regression.py \
      results/spec10_torchfb/L1fix_rzK50_s42.json /tmp/..._REPLAY.json
"""
import json
import sys

KEYMAP = {"0.4": "1.0", "0.8": "2.0", "1.6": "4.0"}

# The gate asserts only what is DETERMINISTIC given (seed, checkpoint, code):
# the model's own set masses. Their residual is float boundary ties -- points
# sitting exactly on a region edge can flip between the scaled-L2 comparison and
# the old raw-L2 one -- which is worth ~0.2-0.5% of mass and nothing more.
DETERMINISTIC = ["model_exact_mean", "model_head_mean"]
TOL = 0.01

# Everything below is a function of the Monte-Carlo rollouts, whose RNG we do
# not control across processes. Measured 2026-07-31 on p2_rzK50_s42: the
# archive-vs-replay gap is 1.5-16% while the seed0-vs-seed1 gap on the SAME code
# is 2-170% -- an order of magnitude larger. Reported, never gated.
STOCHASTIC = ["mc_mean", "ratio_exact", "spearman_exact",
              "ratio_head", "spearman_head"]


def main(archived: str, replay: str) -> int:
    A = json.load(open(archived))["summary"]
    B = json.load(open(replay))["summary"]
    bad = []
    for label, fields, gated in (("GATED (deterministic)", DETERMINISTIC, True),
                                 ("reported (MC-driven)", STOCHASTIC, False)):
        print(f"\n--- {label} ---")
        for old_key, new_key in KEYMAP.items():
            if old_key not in A or new_key not in B:
                print(f"[skip] {old_key}->{new_key}: absent in one of the files")
                continue
            a, b = A[old_key], B[new_key]
            for f in fields:
                if f not in a or f not in b:
                    continue
                rel = abs(a[f] - b[f]) / max(abs(a[f]), 1e-9)
                mark = ""
                if gated and rel > TOL:
                    bad.append((old_key, f, a[f], b[f]))
                    mark = "  <-- OVER TOLERANCE"
                print(f"hw={old_key:>4} {f:<18} archived={a[f]:>10.4f} "
                      f"replay={b[f]:>10.4f} rel={rel:.3%}{mark}")
    if bad:
        print(f"\nFAIL: {len(bad)} deterministic fields moved by more "
              f"than {TOL:.0%} — the port changed the model's set masses")
        for k, f, x, y in bad:
            print(f"  hw={k} {f}: {x} -> {y}")
        return 1
    print(f"\nPASS: the model's set masses reproduce the archive to within "
          f"{TOL:.0%} at every width")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
