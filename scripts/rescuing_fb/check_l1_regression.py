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
FIELDS = ["mc_mean", "model_exact_mean", "ratio_exact", "spearman_exact",
          "model_head_mean", "ratio_head", "spearman_head"]
TOL = 0.02          # MC rollouts are stochastic in the env, not the code


def main(archived: str, replay: str) -> int:
    A = json.load(open(archived))["summary"]
    B = json.load(open(replay))["summary"]
    bad = []
    for old_key, new_key in KEYMAP.items():
        if old_key not in A or new_key not in B:
            print(f"[skip] {old_key}->{new_key}: absent in one of the files")
            continue
        a, b = A[old_key], B[new_key]
        for f in FIELDS:
            if f not in a or f not in b:
                continue
            rel = abs(a[f] - b[f]) / max(abs(a[f]), 1e-9)
            print(f"hw={old_key:>4} {f:<18} archived={a[f]:>10.4f} "
                  f"replay={b[f]:>10.4f} rel={rel:.3%}")
            if rel > TOL:
                bad.append((old_key, f, a[f], b[f]))
    if bad:
        print(f"\nFAIL: {len(bad)} fields moved by more than {TOL:.0%}")
        for k, f, x, y in bad:
            print(f"  hw={k} {f}: {x} -> {y}")
        return 1
    print("\nPASS: the port reproduces the archived cube-single numbers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
