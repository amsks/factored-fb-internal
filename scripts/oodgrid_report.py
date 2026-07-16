"""Aggregate the OOD grid into the 2x2 the campaign exists to produce.

                    ID goal (hindsight)      OOD goal (eval task)
  ID start (data)        C1                        C2
  OOD start (reset)      C4                        C3  = the real eval

Measured off-manifold-ness (median distance to the nearest dataset state, standardised;
a held-out dataset state = 0.845):
    ID start  0.887 (1.0x)     OOD start  1.955 (2.3x,  9.0% far)
    ID goal   0.848 (1.0x)     OOD goal   3.047 (3.6x, 83.5% far)   <- the BIGGER shift

Derived effects (all relative to C1, the on-manifold cell):
    dGOAL  = C2 - C1     goal shift alone
    dSTART = C4 - C1     start shift alone
    dBOTH  = C3 - C1     both (the real eval)
    INTER  = dBOTH - (dGOAL + dSTART)     super/sub-additivity of the two shifts

FB rows carry BOTH z-routes and they are NOT interchangeable (0.05 vs 0.40 on C3):
  zR = reward-inference  = FB's DEPLOYED path, comparable to its published number.
  zB = project_z(B(g))   = the route matched to GCIQL's psi(g).
zR is undefined for hindsight goals, so C1/C4 are zB-only. Never mix them in one row.
"""
import glob
import json
import sys

import numpy as np

SP = '/home/adityamohan/git/Factored-FB/analysis/oodgrid'
ANCHOR = {'GCIQL-State-Flow-Bilinear-d50': 0.904, 'GCIQL-State-Flow': 0.717,
          'fb_state_plain': 0.42}

rows = [json.load(open(f)) for f in sorted(glob.glob(f'{SP}/results/*.json'))]
errs = [r for r in rows if 'error' in r]
ok = [r for r in rows if 'error' not in r]
groups = {}
for r in ok:
    groups.setdefault(r['group'], []).append(r)


def cell(rs, *keys):
    """mean success over seeds; tries each key in order (zR before zB for FB)."""
    for k in keys:
        v = [r[k]['success'] for r in rs if k in r]
        if v:
            return np.mean(v), np.std(v), k
    return np.nan, np.nan, None


out = []
for g, rs in groups.items():
    fw = rs[0]['framework']
    if fw == 'jax':
        c1, c2, c4, c3 = (cell(rs, k)[0] for k in ('C1', 'C2', 'C4', 'C3'))
        route = '-'
    else:
        c1 = cell(rs, 'C1_zB')[0]
        c4 = cell(rs, 'C4_zB')[0]
        c2, _, k2 = cell(rs, 'C2_zR', 'C2_zB')
        c3, _, k3 = cell(rs, 'C3_zR', 'C3_zB')
        route = 'zR' if (k3 or '').endswith('zR') else 'zB'
    out.append(dict(group=g, fw=fw[:3], route=route, n=len(rs), c1=c1, c2=c2, c4=c4, c3=c3,
                    dgoal=c2 - c1, dstart=c4 - c1,
                    inter=(c3 - c1) - ((c2 - c1) + (c4 - c1))))

out.sort(key=lambda r: -(r['c3'] if not np.isnan(r['c3']) else -1))
f = lambda x: '  -   ' if (x is None or np.isnan(x)) else f'{x:6.3f}'
sf = lambda x: '  -   ' if (x is None or np.isnan(x)) else f'{x:+6.3f}'

only = sys.argv[1] if len(sys.argv) > 1 else None
print(f'{"group":<42}{"fw":>4}{"rt":>3}{"n":>3} '
      f'{"C1":>7}{"C2":>7}{"C4":>7}{"C3":>7}  {"dGOAL":>7}{"dSTART":>7}{"INTER":>7}  anchor')
print(f'{"":<42}{"":>4}{"":>3}{"":>3} '
      f'{"ID/ID":>7}{"ID/OOD":>7}{"OOD/ID":>7}{"EVAL":>7}')
print('-' * 116)
for r in out:
    if only and only.lower() not in r['group'].lower():
        continue
    a = ANCHOR.get(r['group'])
    print(f"{r['group'][:41]:<42}{r['fw']:>4}{r['route']:>3}{r['n']:>3} "
          f"{f(r['c1'])}{f(r['c2'])}{f(r['c4'])}{f(r['c3'])}  "
          f"{sf(r['dgoal'])}{sf(r['dstart'])}{sf(r['inter'])}  "
          f"{'' if a is None else f'{a:.3f}'}")

print(f'\n{len(out)} groups, {len(ok)} run-dirs.  errored: {len(errs)} '
      f'(out-of-scope agents are recorded, not silently dropped)')
