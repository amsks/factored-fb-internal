#!/usr/bin/env python3
"""Compare FB vs UM multiseed runs at synchronized eval milestones."""

from __future__ import annotations

import argparse
import json
import statistics as stats
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from agents.crl_iws.schedules import eval_milestones, legacy_nominal_to_aligned_map

METRIC = "eval/success_once"


def _eval_by_step(run_dir: Path) -> dict[int, float]:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return {}
    out: dict[int, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if METRIC in row:
            out[int(row["step"])] = float(row[METRIC])
    return out


def _lookup_eval(
    eval_by_step: dict[int, float],
    milestone: int,
    *,
    legacy_nominal_map: dict[int, int],
) -> float | None:
    if milestone in eval_by_step:
        return eval_by_step[milestone]
    for nominal, aligned in legacy_nominal_map.items():
        if aligned == milestone and nominal in eval_by_step:
            return eval_by_step[nominal]
    return None


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return stats.mean(values), stats.pstdev(values)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign", choices=["sgcrl", "iws"], default="sgcrl")
    ap.add_argument("--fb-root", type=Path, default=Path("runs/air_hockey"))
    ap.add_argument("--um-root", type=Path, default=Path("../Unsupervised_Manipulation/runs/air_hockey"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 42, 43])
    ap.add_argument("--every-n-steps", type=int, default=500_000)
    ap.add_argument("--align-to", type=int, default=256)
    ap.add_argument("--total-env-steps", type=int, default=20_000_000)
    ap.add_argument("--max-milestone", type=int, default=None, help="Only show milestones <= this step")
    args = ap.parse_args()

    prefix = f"{args.campaign}_seed"
    milestones = eval_milestones(
        total_env_steps=args.total_env_steps,
        every_n_steps=args.every_n_steps,
        align_to=args.align_to,
    )
    if args.max_milestone is not None:
        milestones = [m for m in milestones if m <= args.max_milestone]

    legacy_nominal_map = legacy_nominal_to_aligned_map(
        total_env_steps=args.total_env_steps,
        every_n_steps=args.every_n_steps,
        align_to=args.align_to,
    )

    print(f"Campaign: {args.campaign}")
    print(f"Synced milestones (align_to={args.align_to}): {[f'{m:,}' for m in milestones[:6]]}{'...' if len(milestones) > 6 else ''}")
    print()

    header = f"{'step':>12}  {'FB mean':>8}  {'UM mean':>8}  {'Δ UM-FB':>8}  {'n_FB':>4}  {'n_UM':>4}"
    print(header)
    print("-" * len(header))

    for step in milestones:
        fb_vals: list[float] = []
        um_vals: list[float] = []
        for seed in args.seeds:
            fb_dir = args.fb_root / f"{prefix}{seed}"
            um_dir = args.um_root / f"{prefix}{seed}"
            fb_eval = _eval_by_step(fb_dir)
            um_eval = _eval_by_step(um_dir)
            fb_val = _lookup_eval(fb_eval, step, legacy_nominal_map=legacy_nominal_map)
            um_val = _lookup_eval(um_eval, step, legacy_nominal_map={})
            if fb_val is not None:
                fb_vals.append(fb_val)
            if um_val is not None:
                um_vals.append(um_val)

        if not fb_vals and not um_vals:
            continue

        fb_mean, _ = _mean_std(fb_vals)
        um_mean, _ = _mean_std(um_vals)
        delta = um_mean - fb_mean if fb_vals and um_vals else float("nan")
        fb_str = f"{fb_mean * 100:6.1f}%" if fb_vals else "   n/a"
        um_str = f"{um_mean * 100:6.1f}%" if um_vals else "   n/a"
        delta_str = f"{delta * 100:+6.1f}pp" if fb_vals and um_vals else "     n/a"
        print(
            f"{step:12,}  {fb_str:>8}  {um_str:>8}  {delta_str:>8}  {len(fb_vals):4d}  {len(um_vals):4d}"
        )

    print()
    print("Per-seed detail at latest shared milestone:")
    shared: list[int] = []
    for step in reversed(milestones):
        if all(
            _lookup_eval(
                _eval_by_step(args.fb_root / f"{prefix}{seed}"),
                step,
                legacy_nominal_map=legacy_nominal_map,
            )
            is not None
            and _lookup_eval(
                _eval_by_step(args.um_root / f"{prefix}{seed}"),
                step,
                legacy_nominal_map={},
            )
            is not None
            for seed in args.seeds
        ):
            shared = [step]
            break
    if not shared:
        print("  (no milestone where all seeds have both FB and UM eval)")
        return
    step = shared[0]
    print(f"  milestone={step:,}")
    for seed in args.seeds:
        fb = _lookup_eval(
            _eval_by_step(args.fb_root / f"{prefix}{seed}"),
            step,
            legacy_nominal_map=legacy_nominal_map,
        )
        um = _lookup_eval(
            _eval_by_step(args.um_root / f"{prefix}{seed}"),
            step,
            legacy_nominal_map={},
        )
        if fb is None and um is None:
            continue
        fb_s = f"{fb * 100:.1f}%" if fb is not None else "n/a"
        um_s = f"{um * 100:.1f}%" if um is not None else "n/a"
        print(f"    seed {seed}: FB={fb_s}  UM={um_s}")


if __name__ == "__main__":
    main()
