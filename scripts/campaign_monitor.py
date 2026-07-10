#!/usr/bin/env python
"""campaign_monitor.py — one report for every live bridge campaign, at each 100k eval.

Each cell is a one-knob perturbation of something we have already run, and every campaign reuses
seeds 0/1/2. So a cell is compared against its baseline AT THE SAME STEP AND THE SAME SEED, and a
divergence on a cell that should barely differ is a bug signal, not a result.

Live values come straight from wandb -- NOT from analysis/results_cache/hist/, whose per-run files
are frozen at first fetch and go stale under a training run.

  .venv/bin/python scripts/campaign_monitor.py              # deepest 100k step each cell reached
  .venv/bin/python scripts/campaign_monitor.py --step 200000
  .venv/bin/python scripts/campaign_monitor.py --campaign sfjoint
"""
from __future__ import annotations

import argparse
import sys

ENTITY_PROJECT = 'amsks/factored-fb'
METRIC = 'evaluation/overall_success'
TOL = 10_000
CKPTS = (500_000, 400_000, 300_000, 200_000, 100_000)

# cell -> (campaign, baseline group or None, what the comparison tests)
CELLS = {
    # ---- psi_base (1b): readout-vs-TD crux, scalar IQL TD with a crippled goal leg ----
    'GCIQL-State-Flow-Bilinear-d28': (
        'psibase', 'GCIQL-State-Flow-Bilinear-d50',
        'SANITY: same recipe, rank 28 vs 50. Should track closely; this cell does not touch psi_base.'),
    'GCIQL-State-Flow-Bilinear-d50-FrozenPsi': (
        'psibase', 'GCIQL-State-Flow-Bilinear-d50',
        '1b: scalar TD, psi frozen. Gap vs d50 == the cost of the frozen readout ALONE.'),
    'GCIQL-State-Flow-Bilinear-IdentityPsi': (
        'psibase', 'GCIQL-State-Flow-Bilinear-d28',
        '1b: scalar TD, psi(g)=g. Baseline is d28 (same rank), NOT d50. Torch prior: collapse.'),
    'GCIQL-State-Flow-SF-FeatureTD-RewReg-Shift-Fix': (
        'psibase', 'GCIQL-State-Flow-SF-FeatureTD-RewReg-Shift',
        "E3 re-run vs the pre-fix psi(s') run. Expect CLOSE-BUT-NOT-EQUAL early."),
    'GCIQL-State-Flow-SF-FeatureTD-RewReg-Shift-Cond': (
        'psibase', 'GCIQL-State-Flow-SF-FeatureTD-RewReg-Shift-Fix',
        'E3 + task-conditioned phi (audit pt3) vs the agnostic corrected cell.'),

    # ---- joint-psi SF (1a): psi trained by the SF TD itself + FB ortho + sphere ----
    'GCIQL-State-Flow-SF-JointPsi-Ortho1k': (
        'sfjoint', 'GCIQL-State-Flow-Bilinear-d50-C-FBzProjZ-O1000',
        'CELL A, the prize. Matched to the FB measure cell (0.231) in EVERY respect except the '
        'estimator (per-sample vs BxB). ~0.23 => estimator irrelevant, SF dissolves into FB. '
        '~0.0 => the per-sample vector TD is uniquely destructive.'),
    'GCIQL-State-Flow-SF-JointPsi-Ortho1k-Absorb': (
        'sfjoint', 'GCIQL-State-Flow-SF-JointPsi-Ortho1k',
        'CELL C: absorbing mask => Q = gamma^T (first-hitting) instead of a visitation density.'),
}

# ortho diagnostics, only meaningful for the joint-psi cells
DIAG = ['training/sf/psi_ortho', 'training/sf/psi_norm_mean', 'training/sf/psi_ortho_off',
        'training/sf/psi_ortho_diag', 'training/sf/sf_loss']
JOINT = [c for c, (camp, _, _) in CELLS.items() if camp == 'sfjoint']
D_LATENT = 50        # value_latent_dim
B_BATCH = 1024       # gciql.py:282


def c_minus_i_sq(psi_ortho, d=D_LATENT, B=B_BATCH):
    """||C - I||_F^2 from the logged psi_ortho, EXACTLY for sphere-normed psi (project_z).

    The code computes, per head,
        psi_ortho = o_off + o_diag = 0.5 * mean_{i!=j} <psi_i,psi_j>^2  -  mean_i ||psi_i||^2
    With ||psi_i||^2 = d exactly (project_z) we have tr(C) = d and sum_i ||psi_i||^4 = B d^2, so
    using ||Psi Psi^T||_F^2 = B^2 ||C||_F^2:
        psi_ortho = 0.5 * (B^2||C||_F^2 - B d^2)/(B^2 - B)  -  d
    Invert for ||C||_F^2, then ||C - I||_F^2 = ||C||_F^2 - 2 tr(C) + d = ||C||_F^2 - d.

    Do NOT use the asymptotic 2*(psi_ortho + d/2): at B=1024, d=50 the whitened floor is
    psi_ortho = d(B-d)/(2(B-1)) - d = -26.2, not -25, so that form returns NEGATIVE squared norms.
    """
    c_fro2 = (2.0 * (psi_ortho + d) * (B * B - B) + B * d * d) / (B * B)
    return c_fro2 - d


def live_series(groups, keys):
    """group -> {seed: [[step, value...], ...]} pulled fresh from wandb, keyed by CONFIGURED seed.

    Not by name or column order: the d50 baseline has 13 runs (a 10-seed 1M campaign plus 500k
    cells), so positional pairing would compare seed 0 to seed 7. Duplicate seeds keep the longest
    history (the 1M run), whose 100k point is the one we want."""
    import wandb
    api = wandb.Api(timeout=120)
    out = {g: {} for g in groups}
    for g in groups:
        for r in api.runs(ENTITY_PROJECT, {'group': g}, per_page=200):
            pts = [[row['_step']] + [row.get(k) for k in keys]
                   for row in r.history(keys=keys, samples=2000, pandas=False)
                   if row.get(keys[0]) is not None]
            seed = r.config.get('seed')
            if not pts or seed is None:
                continue
            prev = out[g].get(int(seed))
            if prev is None or len(pts) > len(prev):
                out[g][int(seed)] = sorted(pts)
    return out


def at_step(pts, target, col=1):
    near = [p for p in pts if abs(p[0] - target) <= TOL]
    if not near:
        return None
    return min(near, key=lambda p: abs(p[0] - target))[col]


def fmt(vals):
    return ' '.join(f'{v:.3f}' for v in vals) if vals else '—'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--step', type=int, default=None, help='eval step to compare at (default: per-cell deepest 100k multiple)')
    ap.add_argument('--campaign', default=None, help='psibase | sfjoint (default: both)')
    args = ap.parse_args()

    cells = {c: v for c, v in CELLS.items() if args.campaign in (None, v[0])}
    groups = sorted(set(cells) | {b for _, b, _ in cells.values() if b})
    live = live_series(groups, [METRIC])

    # ortho diagnostics for the joint cells (separate key set -> separate fetch)
    diag = live_series([c for c in JOINT if c in cells], DIAG) if any(c in cells for c in JOINT) else {}

    for cell, (camp, base, why) in cells.items():
        runs = live.get(cell, {})
        if args.step is not None:
            step = args.step
        else:
            step = next((t for t in CKPTS if any(at_step(p, t) is not None for p in runs.values())), None)
        if step is None:
            depth = max((max(p[0] for p in pts) for pts in runs.values()), default=0)
            print(f'{cell}\n  [{camp}] no 100k eval yet (deepest logged step {depth})\n')
            continue

        cur = {s: v for s, pts in runs.items() if (v := at_step(pts, step)) is not None}
        bas = {s: v for s, pts in live.get(base, {}).items() if (v := at_step(pts, step)) is not None} if base else {}
        seeds = sorted(cur)
        vals = [cur[s] for s in seeds]
        bvals = [bas[s] for s in seeds if s in bas]

        print(f'{cell}   [{camp}] @ {step // 1000}k')
        print(f'  seeds {seeds}  {fmt(vals)}   (n={len(vals)}/3)'
              + (f'   mean {sum(vals) / len(vals):.3f}' if vals else ''))
        if base:
            print(f'  baseline      {fmt(bvals)}   [{base}]'
                  + (f'   mean {sum(bvals) / len(bvals):.3f}' if bvals else ''))
            paired = [(s, cur[s] - bas[s]) for s in seeds if s in bas]
            if paired:
                print('  per-seed Δ    ' + ' '.join(f's{s}:{d:+.3f}' for s, d in paired)
                      + f'   |Δ|mean {sum(abs(d) for _, d in paired) / len(paired):.3f}')
        if cell in diag:
            # Do NOT read psi_ortho_off on its own: asymptotically o_off = 0.5*||C||_F^2, so at the
            # WHITENED optimum C=I it equals d/2 = 25 -- the same value random sphere-normed psi
            # gives (random isotropic vectors already have C ~ I). It cannot tell the two apart.
            # The discriminating quantity is the SUM:
            #   psi_ortho = o_off + o_diag = 0.5||C||_F^2 - tr(C) = 0.5||C - I||_F^2 - d/2
            # so psi_ortho -> -d/2 = -25 iff C -> I, and  ||C - I||_F^2 = 2*(psi_ortho + d/2).
            # dev >> 0 means psi has NOT whitened and the peaked-kernel argument never took hold.
            floor = D_LATENT * (B_BATCH - D_LATENT) / (2.0 * (B_BATCH - 1)) - D_LATENT
            for s, pts in sorted(diag[cell].items()):
                step_, orth, norm, _off, _diag, sfl = pts[-1]
                dev = c_minus_i_sq(orth)                     # ||C - I||_F^2, exact under project_z
                print(f'  ortho  s{s} @{step_:>6}  psi_ortho={orth:+.2f} (C=I floor {floor:.1f})  '
                      f'||C-I||²={dev:.2f} (of ||C||²≈{dev + D_LATENT:.0f})  '
                      f'|psi|={norm:.3f} (√d=7.071)  sf_loss={sfl:.3g}')
        print(f'  ? {why}\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
