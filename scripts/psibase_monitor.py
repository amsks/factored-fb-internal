#!/usr/bin/env python
"""psibase_monitor.py — live progress of the psi_base (1b) + corrected-E3 campaign, with each
cell compared against its baseline AT THE SAME STEP AND SEED.

The campaign reuses seeds 0/1/2, and every cell is a one-knob perturbation of a cell we have
already run. So at an early eval point the perturbed cell should track its baseline closely
unless the knob genuinely bites (or something is broken). A wild divergence at 100k on the
cells that should barely differ (bilinear-d28 vs bilinear-d50) is a bug signal, not a result.

Baselines come from the living CSVs (results/{summary_agents,bridge_grafts}.csv, which carry
per_seed_100k..per_seed_400k). Live values come straight from wandb — NOT from
analysis/results_cache/hist/, whose per-run files are frozen at first fetch and would go stale
under a training run.

  .venv/bin/python scripts/psibase_monitor.py               # nearest common eval step
  .venv/bin/python scripts/psibase_monitor.py --step 100000
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ENTITY_PROJECT = 'amsks/factored-fb'
METRIC = 'evaluation/overall_success'
TOL = 10_000

# cell -> (baseline group, what the comparison tests)
CELLS = {
    'GCIQL-State-Flow-Bilinear-d28': (
        'GCIQL-State-Flow-Bilinear-d50',
        'SANITY: same recipe, rank 28 vs 50. Should track closely; a big gap means the launch '
        'or the rank itself is off, not psi_base (this cell does not touch psi_base at all).'),
    'GCIQL-State-Flow-Bilinear-d50-FrozenPsi': (
        'GCIQL-State-Flow-Bilinear-d50',
        '1b: scalar TD, psi frozen. Gap vs d50 == the cost of the frozen readout ALONE.'),
    'GCIQL-State-Flow-Bilinear-IdentityPsi': (
        'GCIQL-State-Flow-Bilinear-d28',
        '1b: scalar TD, psi(g)=g. Baseline is d28 (same rank), NOT d50. Torch prior: collapse.'),
    'GCIQL-State-Flow-SF-FeatureTD-RewReg-Shift-Fix': (
        'GCIQL-State-Flow-SF-FeatureTD-RewReg-Shift',
        'E3 re-run vs the pre-fix (psi(s\')) run. Same config except the reward index; expect '
        'CLOSE-BUT-NOT-EQUAL early. Identical would mean the fix never took effect.'),
    'GCIQL-State-Flow-SF-FeatureTD-RewReg-Shift-Cond': (
        'GCIQL-State-Flow-SF-FeatureTD-RewReg-Shift-Fix',
        'E3 + task-conditioned phi (audit pt3) vs the agnostic corrected cell.'),
}


def load_csv_baselines():
    """group -> {step: [per-seed values]} from the living CSVs."""
    out = {}
    for name in ('summary_agents.csv', 'bridge_grafts.csv'):
        p = REPO / 'results' / name
        if not p.exists():
            continue
        for row in csv.DictReader(open(p)):
            g = row['wandb_group']
            by_step = {}
            for k in (100, 200, 300, 400):
                cell = (row.get(f'per_seed_{k}k') or '').split()
                if cell:
                    by_step[k * 1000] = cell
            if row.get('per_seed'):
                by_step[500_000] = row['per_seed'].split()
            out[g] = by_step
    return out


def live_series(groups):
    """group -> {seed: [[step, success], ...]} pulled fresh from wandb.

    Keyed by the run's CONFIGURED seed, not by name or CSV column order: the d50 baseline has
    13 runs (the 10-seed 1M campaign plus 500k cells), so positional pairing against our 3 seeds
    would silently compare seed 0 to seed 7. Duplicate seeds within a group keep the LONGEST
    history (the 1M run), which is the one whose 100k point we want."""
    import wandb
    api = wandb.Api(timeout=120)
    out = {g: {} for g in groups}
    for g in groups:
        for r in api.runs(ENTITY_PROJECT, {'group': g}, per_page=200):
            pts = [[row['_step'], row[METRIC]]
                   for row in r.history(keys=[METRIC], samples=2000, pandas=False)
                   if row.get(METRIC) is not None]
            if not pts:
                continue
            seed = r.config.get('seed')
            if seed is None:
                continue
            prev = out[g].get(int(seed))
            if prev is None or len(pts) > len(prev):
                out[g][int(seed)] = sorted(pts)
    return out


def at_step(pts, target):
    near = [(s, v) for s, v in pts if abs(s - target) <= TOL]
    if not near:
        return None
    return min(near, key=lambda t: abs(t[0] - target))[1]


def fmt(vals):
    return ' '.join(f'{v:.3f}' for v in vals) if vals else '—'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--step', type=int, default=None, help='eval step to compare at (default: deepest step all live cells have reached)')
    args = ap.parse_args()

    csv_base = load_csv_baselines()
    groups = sorted(set(CELLS) | {b for b, _ in CELLS.values()})
    live = live_series(groups)

    # Deepest 100k-multiple that EVERY cell has at least one seed at. (ogbench also logs an eval
    # at step 1; it is not a checkpoint and must never be picked as the comparison step.)
    if args.step is None:
        reached = [t for t in (500_000, 400_000, 300_000, 200_000, 100_000)
                   if all(any(at_step(pts, t) is not None for pts in live.get(c, {}).values())
                          for c in CELLS)]
        if not reached:
            print('not every cell has reached 100k yet — per-cell progress:')
            for c in CELLS:
                got = {s: max(x for x, _ in pts) for s, pts in live.get(c, {}).items()}
                print(f'  {c:52s} seeds@step {got or "(no evals yet)"}')
            print('\nrerun once all cells have a 100k eval, or pass --step 100000 to compare '
                  'the cells that do.')
            return 0
        args.step = reached[0]

    print(f'\n=== psi_base + corrected-E3 @ {args.step // 1000}k '
          f'(live wandb, paired on seed) ===\n')
    for cell, (base, why) in CELLS.items():
        cur = {s: v for s, pts in live.get(cell, {}).items() if (v := at_step(pts, args.step)) is not None}
        bas = {s: v for s, pts in live.get(base, {}).items() if (v := at_step(pts, args.step)) is not None}

        seeds = sorted(cur)
        vals = [cur[s] for s in seeds]
        bvals = [bas[s] for s in seeds if s in bas]          # ONLY the matching seeds
        csv_mean = (csv_base.get(base, {}) or {}).get(args.step)
        csv_mean = (sum(float(x.split('@')[0]) for x in csv_mean) / len(csv_mean)) if csv_mean else None

        print(f'{cell}')
        print(f'  seeds {seeds}  {fmt(vals)}   (n={len(vals)}/3)'
              + (f'   mean {sum(vals) / len(vals):.3f}' if vals else ''))
        print(f'  baseline      {fmt(bvals)}   [{base}]'
              + (f'   mean {sum(bvals) / len(bvals):.3f}' if bvals else '')
              + (f'   (csv all-seed mean {csv_mean:.3f})' if csv_mean is not None else ''))
        paired = [(s, cur[s] - bas[s]) for s in seeds if s in bas]
        if paired:
            print(f'  per-seed Δ    ' + ' '.join(f's{s}:{d:+.3f}' for s, d in paired)
                  + f'   |Δ|mean {sum(abs(d) for _, d in paired) / len(paired):.3f}')
        print(f'  ? {why}\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
