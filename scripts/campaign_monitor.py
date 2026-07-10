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

    # ---- task-weighted moments (1e): replay norm vs task norm ----
    'GCIQL-State-Flow-Bilinear-d50-C-TaskNorm': (
        'tasknorm', 'GCIQL-State-Flow-Bilinear-d50-C-FBzProjZ-O1000',
        'N1: measure residual tested under nu (value_goals) instead of rho. Matched to the anchor '
        '(0.231) in EVERY respect except the test set. ~0.90 vindicates the note; ~0.23 refutes it.'),
    'GCIQL-State-Flow-Bilinear-d50-C-TaskProj': (
        'tasknorm', 'GCIQL-State-Flow-Bilinear-d50-C-TaskNorm',
        'N2: + rank-1 test set. Fixed point = GCIQL backup with the learned kernel reward. '
        '~0.90 => the RANK was the failure; ~0.23 => the KERNEL REWARD was; ~0.12 => the diagonal '
        'estimator dominates (cell-A effect) and N2 says nothing about the reward.'),

    # ---- GC-SimpleSF (1d): architecture, then reward-grounding of psi ----
    'GCIQL-State-Flow-SimpleSF-Arch': (
        'simplesf', 'GCIQL-State-Flow-Bilinear-d50',
        'rung 2: the SimpleSF trunk split, psi still learned end-to-end. Isolates the ARCHITECTURE '
        'against bilinear GCIQL (0.904). Expect 0.84-0.90 (B1 says z-conditioning costs ~0.06).'),
    'GCIQL-State-Flow-SimpleSF-Grounded': (
        'simplesf', 'GCIQL-State-Flow-SimpleSF-Arch',
        "rung 3 = Chua et al.'s cell: psi's ONLY gradient is the reward regression. Isolates the "
        'GROUNDING. Predicted 0.0-0.2; near 0.90 would falsify "grounding is a repair, not a virtue".'),
}

# SimpleSF reward-model diagnostics. PRE-REGISTERED: read reward_loss BEFORE the success number.
SS_DIAG = ['training/simplesf/reward_loss', 'training/simplesf/b_norm_mean',
           'training/simplesf/z_norm_mean', 'training/simplesf/rhat_mean']
SS_PRIOR = 0.166   # r in {-1,0}, ~79% at -1 => predicting the mean scores 0.79*0.21
GROUNDED = 'GCIQL-State-Flow-SimpleSF-Grounded'

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


def _print_diag(cell, diag, ss):
    """Training-side diagnostics: ortho health (joint-psi cells) and the PRE-REGISTERED
    reward-model check (the grounded SimpleSF cell). Printed whether or not an eval has landed."""
    if cell in diag:
        # Do NOT read psi_ortho_off on its own: asymptotically o_off = 0.5*||C||_F^2, so at the
        # WHITENED optimum C=I it equals d/2 = 25 -- the same value random sphere-normed psi
        # gives. The discriminating quantity is the SUM, psi_ortho -> -d/2 iff C -> I.
        floor = D_LATENT * (B_BATCH - D_LATENT) / (2.0 * (B_BATCH - 1)) - D_LATENT
        for s, pts in sorted(diag[cell].items()):
            step_, orth, norm, _off, _diag, sfl = pts[-1]
            dev = c_minus_i_sq(orth)                     # ||C - I||_F^2, exact under project_z
            print(f'  ortho  s{s} @{step_:>6}  psi_ortho={orth:+.2f} (C=I floor {floor:.1f})  '
                  f'||C-I||²={dev:.2f} (of ||C||²≈{dev + D_LATENT:.0f})  '
                  f'|psi|={norm:.3f} (√d=7.071)  sf_loss={sfl:.3g}')
    if cell in ss:
        # PRE-REGISTERED. reward_loss must fall well below the class prior (0.166) or the reward is
        # NOT linearly readable in b -- the reference's load-bearing assumption failing -- and the
        # success number then says nothing about grounding.
        for s, pts in sorted(ss[cell].items()):
            step_, rl, bn, zn, rm = pts[-1]
            verdict = 'FITS' if rl < 0.8 * SS_PRIOR else ('at prior' if rl < 1.2 * SS_PRIOR else 'ABOVE prior')
            print(f'  reward s{s} @{step_:>6}  reward_loss={rl:.4f} (prior {SS_PRIOR}) -> {verdict}  '
                  f'rhat_mean={rm:+.3f}  |b|={bn:.4f} (1.0)  |z|={zn:.4f} (√d=7.071)')


# Annealed SM finetuning: a dose curve, not a seed-paired comparison. lambda(t) ramps 0->lam_max
# over 300k then holds; each run is success-vs-lambda. PRE-REGISTERED: critic/q_mean must stay in
# [-21, -8] (outside = Q-scale blowup, success uninterpretable); Anneal-Zero must stay ~0.90.
ANNEAL = {
    'GCIQL-State-Flow-Anneal-Zero':   ('lam=0 CONTROL: 400k more IQL, no SM loss. Must stay ~0.90 or restore/drift is broken.', 0.0),
    'GCIQL-State-Flow-Anneal-SgPhi':  ('lambda 0->10, stopgrad-phi. Cold cliff at lambda=3 (0.857->0.176). Survives => path artifact; collapses => bad fixed point.', 10.0),
    'GCIQL-State-Flow-Anneal-Shared': ('lambda 0->100, shared legs. Cold cliff above 10 (l10=0.917), never located.', 100.0),
}
ANNEAL_RAMP = 300_000
Q_BAND = (-21.0, -8.0)


def report_anneal():
    # eval success and the training diagnostics log on DIFFERENT steps, so fetch them separately
    # and pair by "latest of each" -- the reader wants "at ~this depth: success X, lambda Y, q Z".
    succ = live_series(list(ANNEAL), [METRIC])
    trn = live_series(list(ANNEAL), ['training/critic/q_mean', 'training/anneal/lambda'])
    print('\n=== annealed SM finetune (dose curve; lam(t) ramps 0->lam_max over 300k, holds) ===\n')
    for g, (why, lam_max) in ANNEAL.items():
        sr, tr = succ.get(g, {}), trn.get(g, {})
        print(f'{g.split("-")[-1]}   (lam_max={lam_max:g})')
        if not sr and not tr:
            print('  no data logged yet\n'); continue
        for s in sorted(set(sr) | set(tr)):
            e = sr.get(s, [])[-1] if sr.get(s) else None
            t = tr.get(s, [])[-1] if tr.get(s) else None
            estep, ev = (e[0], e[1]) if e else (None, None)
            tstep, q, lam = (t[0], t[1], t[2]) if t else (None, None, None)
            if lam is None and tstep is not None:
                lam = lam_max * min(1.0, tstep / ANNEAL_RAMP)
            qflag = '' if (q is None or Q_BAND[0] <= q <= Q_BAND[1]) else '  *** q OUT OF BAND [-21,-8]: scale blowup, success uninterpretable ***'
            es = f'{ev:.3f}@{estep}' if e else '   -   '
            qs = f'{q:+.1f}' if q is not None else ' - '
            ls = f'{lam:6.2f}' if lam is not None else '  -  '
            print(f'  s{s}  success={es:>12s}  lambda={ls} (@{tstep})  q_mean={qs}{qflag}')
        print(f'  ? {why}\n')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--step', type=int, default=None, help='eval step to compare at (default: per-cell deepest 100k multiple)')
    ap.add_argument('--campaign', default=None, help='psibase | sfjoint | anneal (default: seed-paired campaigns)')
    args = ap.parse_args()

    if args.campaign == 'anneal':
        return report_anneal()

    cells = {c: v for c, v in CELLS.items() if args.campaign in (None, v[0])}
    groups = sorted(set(cells) | {b for _, b, _ in cells.values() if b})
    live = live_series(groups, [METRIC])

    # ortho diagnostics for the joint cells (separate key set -> separate fetch)
    diag = live_series([c for c in JOINT if c in cells], DIAG) if any(c in cells for c in JOINT) else {}
    ss = live_series([GROUNDED], SS_DIAG) if GROUNDED in cells else {}

    for cell, (camp, base, why) in cells.items():
        runs = live.get(cell, {})
        if args.step is not None:
            step = args.step
        else:
            step = next((t for t in CKPTS if any(at_step(p, t) is not None for p in runs.values())), None)
        if step is None:
            depth = max((max(p[0] for p in pts) for pts in runs.values()), default=0)
            print(f'{cell}\n  [{camp}] no 100k eval yet (deepest logged step {depth})')
            # Still print the training-side diagnostics: the PRE-REGISTERED reward_loss check and
            # the ortho health are most useful BEFORE the first eval lands, not after.
            _print_diag(cell, diag, ss)
            print(f'  ? {why}\n')
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
        _print_diag(cell, diag, ss)
        print(f'  ? {why}\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
