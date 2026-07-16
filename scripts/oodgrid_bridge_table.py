"""The OOD 2x2 for EVERY cell in the FB<->GCIQL bridge program.

Joins analysis/oodgrid/results/*.json against scripts/results_map.py (csv='bridge'), so
every row carries its real variant/graft label instead of a truncated wandb group name.
Canonical anchors (csv='summary') are printed as a reference block on top.

Cells (byte-identical starts/goals across all four, analysis/oodgrid/cells.npz):

                    ID goal (hindsight)      OOD goal (eval task)
  ID start (data)        C1                        C2
  OOD start (reset)      C4                        C3  = the real eval

  dGOAL = C2-C1   dSTART = C4-C1   INTER = (C3-C1) - (dGOAL+dSTART)

WARNING: for torch FB rows C1/C4 come from the zB route and C2/C3 from zR (zR is undefined
for hindsight goals). Their dGOAL/dSTART are therefore CROSS-ROUTE and meaningless -- they
are printed as '.' rather than a number. Only jax rows decompose.

Usage:  .venv/bin/python scripts/oodgrid_bridge_table.py [--md] [--all]
"""
import glob
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from results_map import MAP  # noqa: E402

SP = '/home/adityamohan/git/Factored-FB/analysis/oodgrid'
ANCHOR = {'GCIQL-State-Flow-Bilinear-d50': 0.904, 'GCIQL-State-Flow': 0.717,
          'fb_state_plain': 0.42}

# --- the `experiment` column -------------------------------------------------------------
# results_map's `graft` field is written in this programme's private vocabulary ("two-tower",
# "fbz", "sphere", "zmix", "dnorm"). Plain-RL restatements, grounded in the source:
#   two-tower   -> Q-feature encoder with separate (s,a) and (s,z) branches merged by a trunk,
#                  instead of one MLP over [s,a]            (gciql_flow.py:119 _TwoTowerPhi)
#   sphere/projz-> L2-normalise the goal embedding to fixed norm sqrt(d)   (_sphere, :27)
#   fbz         -> make the Q-feature task-conditioned (depend on z)       (fb_forward_z, :245)
#   ortho<N>    -> orthonormality penalty on the representation, coefficient N
#   measure     -> replace scalar reward TD with a successor-measure (occupancy) objective
#   SF/featureTD-> replace scalar reward TD with successor-feature VECTOR TD
#   sgphi       -> stop gradients from that objective into the Q-feature
#   zmix/zHER   -> sample task vectors randomly / relabel them from future states
#   z=g, idB    -> no learned goal embedding: the task vector IS the goal
# Any group NOT in this dict falls back to its results_map text (and is counted), so a cell is
# never silently given a description I could not ground.
EXPERIMENT = {
    'GCIQL-State-Flow-Anneal-Zero': 'control: fine-tune the 0.904 baseline 400k longer, no change',
    'gciql_g4_zcond': 'actor conditions on the goal embedding instead of the raw goal',
    'GCIQL-State-Flow-Bilinear-d50-singleq': 'single Q head instead of double-Q',
    'GCIQL-State-Flow-Bilinear-d50-tau05': 'IQL expectile 0.5',
    'GCIQL-State-Flow-Bilinear-d50-tau095': 'IQL expectile 0.95',
    'GCIQL-State-Flow-Bilinear-d50-TwoTower': 'Q-feature: separate (s,a)/(s,z) encoder branches',
    'GCIQL-State-Flow-Bilinear-d50-B-TwoTowerProjZ': 'above + goal embedding L2-normalised to fixed norm',
    'GCIQL-State-Flow-Bilinear-d50-B-TwoTowerOrtho1': 'above + orthonormality penalty (coef 1)',
    'GCIQL-State-Flow-Bilinear-d50-B1-FBzIQL': 'task-conditioned Q-feature, still trained by scalar reward TD',
    'GCIQL-State-Flow-Bilinear-d50-B4-FBstruct': 'all representation changes at once, scalar reward TD kept',
    'GCIQL-State-Flow-Bilinear-d50-Ortho1000': 'orthonormality penalty (coef 1000) on the reward-TD critic',
    'GCIQL-State-Flow-Bilinear-IdentityPsi': 'no learned goal embedding: goal used directly',
    'GCIQL-State-Flow-Bilinear-FrozenPsi-d50': 'goal embedding frozen at random init',
    'GCIQL-State-Flow-SimpleSF-Arch': 'successor features: normalised base features + linear readout',
    'GCIQL-State-Flow-SimpleSF-Grounded': 'above, base features trained only by reward regression',
    'GCIQL-State-Flow-Bilinear-SG-d50': 'goal embedding also sees the state (removes the low-rank goal bottleneck)',

    # --- successor-measure objective grafted onto the reward-TD critic ---
    'GCIQL-Bil-MeasureOnly-O1000': 'successor-measure objective only (no reward TD) + orthonormality penalty 1000',
    'GCIQL-Bil-FBz-O1000': 'successor-measure objective + task-conditioned Q-feature + orthonormality 1000',
    'GCIQL-State-Flow-Bilinear-d50-C-FBz-O1': 'successor-measure + task-conditioned Q-feature + orthonormality 1',
    'GCIQL-State-Flow-Bilinear-d50-C-FBzProjZ-O1000': 'successor-measure + task-conditioned Q-feature + normalised goal embedding + orthonormality 1000',
    'GCIQL-State-Flow-Bilinear-d50-C2-MeasureTwoTower': 'successor-measure + separate (s,a)/(s,z) Q-feature branches',
    'GCIQL-State-Flow-Bilinear-d50-C4-FBrebuild': 'successor-measure + all representation changes at once (full stack)',
    'GCIQL-State-Flow-Bilinear-d50-C5-Zmix': 'above + task vectors sampled as random unit directions during training',
    'GCIQL-State-Flow-Bilinear-d50-C5-Zmix-ActorZ': 'above + actor conditions on the task vector + task-consistent bootstrap',
    'GCIQL-State-Flow-Bilinear-d50-D1-ActorZCritic': 'actor takes its task vector from the critic goal embedding, under the successor-measure objective',
    'GCIQL-State-Flow-Bilinear-d50-Exp4-ActorZInfer': 'actor conditions on a task vector inferred from reward (⚠ artifact: the inferred vector was ~99% noise)',
    'GCIQL-State-Flow-Bilinear-d50-C-TaskNorm': 'successor-measure residual measured under the task-state distribution instead of the replay distribution',
    'GCIQL-State-Flow-Bilinear-d50-C-TaskProj': 'above + each residual projected onto its own goal embedding (rank-1; fixed point = reward TD with a kernel reward)',

    # --- dose curve: mix the successor-measure loss into reward TD at coefficient lambda ---
    'GCIQL-State-Flow-Bilinear-d50-MeasureMix-l001': 'reward TD + successor-measure loss, weight 0.01 (shared representation)',
    'GCIQL-State-Flow-Bilinear-d50-MeasureMix-l01': 'reward TD + successor-measure loss, weight 0.1 (shared representation)',
    'GCIQL-State-Flow-Bilinear-d50-MeasureMix-l1': 'reward TD + successor-measure loss, weight 1 (shared representation)',
    'GCIQL-State-Flow-Bilinear-d50-MeasureMix-l10': 'reward TD + successor-measure loss, weight 10 (shared representation)',
    'GCIQL-State-Flow-Bilinear-d50-MeasureMix-l01-sgphi': 'as above, weight 0.1, measure gradients blocked from the Q-feature',
    'GCIQL-State-Flow-Bilinear-d50-MeasureMix-l1-sgphi': 'as above, weight 1, measure gradients blocked from the Q-feature',
    'GCIQL-State-Flow-Bilinear-d50-MeasureMix-l3-sgphi': 'as above, weight 3, measure gradients blocked from the Q-feature',
    'GCIQL-State-Flow-Bilinear-d50-MeasureMix-l10-sgphi': 'as above, weight 10 (⚠ 1 Q-range diagnostic violation)',
    'GCIQL-State-Flow-Bilinear-d50-MeasureMix-l30-sgphi': 'as above, weight 30 (⚠ 3/3 Q-range violations — uninterpretable)',

    # --- curricula: start from the trained baseline, ramp the measure in ---
    'GCIQL-State-Flow-Anneal-SgPhi': 'warm-start from baseline, ramp measure weight 0→10 (gradients blocked from the Q-feature)',
    'GCIQL-State-Flow-Anneal-Shared': 'warm-start from baseline, ramp measure weight 0→100 (shared representation)',
    'GCIQL-State-Flow-Anneal-SgPhi-IQLfade': 'as above, but also fade reward TD out — endpoint is measure-only: does the good solution survive without it?',
    'GCIQL-State-Flow-SimpleSF-Anneal': 'successor-features curriculum: fade true-reward TD out as reward-regression grounding fades in',

    # --- successor-features vector TD replacing scalar reward TD ---
    'GCIQL-State-Flow-SF-JointPsi-Ortho1k': 'successor-feature vector TD, base features learned jointly; normalised embedding + orthonormality 1000',
    'GCIQL-State-Flow-SF-JointPsi-Ortho1k-Absorb': 'above + absorbing goal (first-hitting Q=γ^T) instead of a visitation density',
    'GCIQL-State-Flow-SF-FeatureTD-RewReg-Shift-Fix': 'successor-feature vector TD with base features learned by reward regression',
    'GCIQL-State-Flow-SF-FeatureTD-RewReg-Shift-Cond': 'above + task-conditioned Q-feature',
    'GCIQL-State-Flow-SF-FeatureTD-IdentityB': 'successor-feature vector TD, base features = raw state (⚠ confounded, see audit)',
    'GCIQL-State-Flow-SF-FeatureTD-Frozen50': 'successor-feature vector TD, base features frozen at random init (⚠ confounded)',
    'GCIQL-State-Flow-SF-FeatureTD-IdentityCond': 'above (raw-state base) + task-conditioned Q-feature (⚠ confounded)',
    'GCIQL-State-Flow-SF-FeatureTD-FrozenCond': 'above (frozen base) + task-conditioned Q-feature (⚠ confounded)',
    'GCIQL-State-Flow-SF-FeatureTD-Identity-Ortho1k': 'raw-state base + orthonormality penalty (kills the critic)',
    'GCIQL-State-Flow-SF-FeatureTD-Frozen-Ortho1k': 'frozen base + orthonormality penalty',
    'GCIQL-State-Flow-SF-FeatureTD-IdentityCond-Ortho1k': 'raw-state base + task-conditioned Q-feature + orthonormality penalty',
    'GCIQL-State-Flow-SF-FeatureTD-FrozenCond-Ortho1k': 'frozen base + task-conditioned Q-feature + orthonormality penalty',
    'GCIQL-State-Flow-SF-FeatureTD-IdentityB-DDPGBC': 'raw-state base, DDPG+BC actor (actor-isolation control) (⚠ confounded)',
    'GCIQL-State-Flow-SF-FeatureTD-Frozen50-DDPGBC': 'frozen base, DDPG+BC actor (actor-isolation control)',
    'GCIQL-State-Flow-SF-FeatureTD-IdentityCond-DDPGBC': 'raw-state base + task-conditioned Q-feature, DDPG+BC actor (control)',
    'GCIQL-State-Flow-SF-FeatureTD-FrozenCond-DDPGBC': 'frozen base + task-conditioned Q-feature, DDPG+BC actor (control)',

    # --- FB critic trained with the IQL objective (the bridge, from the FB side) ---
    'fb_state_iql': 'FB critic trained with the IQL objective instead of the measure loss (first version)',
    'fb_state_iqlcritic': 'as above, earlier hyperparameters',
    'fb_state_iqlcritic_fix': 'as above, corrected learning rate + orthonormality 1000',
    'fb_iql_ontraj_flow': 'FB critic under IQL, goals sampled from the same trajectory',
    'fb_iql_ontraj_ddpgbc': 'as above with a DDPG+BC actor (actor-isolation control)',
    'fb_bridge_r0_base': 'FB critic under IQL + goal-matched sampling + GCIQL-style goal mixture',
    'fb_bridge_r1_decond': 'above + Q-feature no longer conditioned on the task vector',
    'fb_bridge_r2_dnorm': 'above + inner product scaled by 1/sqrt(d)',
    'fb_bridge_r3_allon': 'above + both changes together',
    'fb_bridge_r3_bilinearv': 'above + bilinear value head (tested as a gap-closer; refuted)',
    'fb_bridge_a0_cubexyz': 'FB given a privileged reward: distance of the cube to its target',

    # --- what does the learned goal embedding buy? ---
    'fb_bridge_idB_dnorm': 'no learned goal embedding (task vector = the goal) + 1/sqrt(d) scaling',
    'fb_bridge_idB_decond_dnorm': 'above + Q-feature not task-conditioned (collapses)',
    'fb_cube_zg_iql': 'no learned goal embedding, IQL objective',
    'fb_cube_zg_her': 'no learned goal embedding, hindsight relabelling',
    'fb_cube_iql_learnedB_r1': 'learned goal embedding, IQL objective (matched control)',
    'fb_cube_her_learnedB_r1': 'learned goal embedding, hindsight relabelling (matched control)',
    'fb_cube_her_idnorm_r1': 'unnormalised goal embedding, hindsight relabelling',

    # --- actor/task-vector and data-sampling variants on FB ---
    'fb_cube_decouple': 'actor no longer shares the critic task vector',
    'fb_cube_decouple_iql': 'above + IQL objective',
    'fb_cube_decouple_zher_traj': 'above + task vector relabelled from future states in the trajectory',
    'fb_state_zher': 'task vector relabelled from future states (hindsight over tasks)',
    'fb_state_iwr': 'importance-weighted regression on the actor',
    'fb_state_iwr_wz': 'above, weights also conditioned on the task vector',
    'covbal_prio_wz_a0.5': 'coverage-balanced prioritised replay, task-vector-aware (alpha 0.5)',
    'fb_z128_ortho1000': 'FB with a 128-dim task vector + orthonormality 1000',
    'cube-single-onestep-ortho1000-lrb1e-4': 'one-step (non-bootstrapped) FB + orthonormality 1000',

    # --- scalar-gated FB (ISFB) ---
    'ISFB-Torch-Exp1-fixed': 'scalar-gated FB (ISFB), faithful reference implementation (torch)',
    'ISFB-State-Coord-d50': 'scalar-gated FB with a coordinate readout',
    'ISFB-State-ScalarGateHard-d50': 'scalar-gated FB, hard gate',
    'ISFB-State-ScalarGateAWR-d50': 'scalar-gated FB, AWR extraction',
}

MD = '--md' in sys.argv
ALL = '--all' in sys.argv
CSV = '--csv' in sys.argv

rows = [json.load(open(f)) for f in sorted(glob.glob(f'{SP}/results/*.json'))]
groups = {}
for r in rows:
    if 'error' not in r:
        groups.setdefault(r['group'], []).append(r)


def cell(rs, *keys):
    for k in keys:
        v = [r[k]['success'] for r in rs if k in r]
        if v:
            return float(np.mean(v)), k
    return np.nan, None


def build(g, rs):
    fw = rs[0]['framework']
    if fw == 'jax':
        c1, c2, c4, c3 = (cell(rs, k)[0] for k in ('C1', 'C2', 'C4', 'C3'))
        route, mixed = '-', False
    else:
        c1 = cell(rs, 'C1_zB')[0]
        c4 = cell(rs, 'C4_zB')[0]
        c2, k2 = cell(rs, 'C2_zR', 'C2_zB')
        c3, k3 = cell(rs, 'C3_zR', 'C3_zB')
        route = 'zR' if (k3 or '').endswith('zR') else 'zB'
        # deltas are only meaningful if goal-cells and start-cells share a z-route
        mixed = route == 'zR'
    e = MAP.get(g, {})
    # results_map's 'flowbc' (torch FB-family) and 'fql_flow' (jax) are the SAME actor:
    # both are the FQL one-step-distillation flow actor -- rectified-flow BC matching, a
    # noise-conditioned one-shot head, -Q.mean() normalised by |Q|, distillation to the
    # frozen multi-step rollout, flow_steps=10. Verified term-for-term
    # (agents/fb/flow_bc/agent.py:74-107 vs tools/wandb_mode_shim/gciql_flow.py:545-596).
    # The only difference is where Q comes from (FB's F.z vs GCIQL's IQL min(q1,q2)) --
    # that is the CRITIC, i.e. the variable the bridge is manipulating. So: one label.
    # The framework distinction the two names used to carry now lives in the 'fw' column
    # (ISFB has rows in BOTH torch and jax and would otherwise become ambiguous).
    actor = e.get('actor', '?')
    actor = 'flow' if actor in ('flowbc', 'fql_flow') else actor
    exp = EXPERIMENT.get(g)
    return dict(group=g, fw=fw[:3], route=route, n=len(rs), c1=c1, c2=c2, c4=c4, c3=c3,
                mixed=mixed, bucket=e.get('csv', '?'), agent=e.get('agent', '?'),
                actor=actor, variant=e.get('variant', ''),
                experiment=exp or e.get('graft', ''), translated=exp is not None,
                dgoal=c2 - c1, dstart=c4 - c1,
                inter=(c3 - c1) - ((c2 - c1) + (c4 - c1)))


out = [build(g, rs) for g, rs in groups.items()]
bridge = sorted([r for r in out if r['bucket'] == 'bridge'], key=lambda r: -np.nan_to_num(r['c3'], nan=-1))
anchors = sorted([r for r in out if r['bucket'] == 'summary'], key=lambda r: -np.nan_to_num(r['c3'], nan=-1))
unmapped = [r for r in out if r['bucket'] not in ('bridge', 'summary')]

f = lambda x: '     -' if x is None or np.isnan(x) else f'{x:6.3f}'
# a cross-route delta (C1/C4 from zB, C2/C3 from zR) is not a number we are willing to show
sd = lambda x, mixed: '     .' if mixed else ('     -' if x is None or np.isnan(x) else f'{x:+7.3f}')


def emit(title, rs, show_exp=True):
    if MD:
        print(f'\n### {title}\n')
        print('| agent | fw | actor | variant | n | C1 | C2 | C4 | **C3** | dGOAL | dSTART | INTER |'
              + (' experiment |' if show_exp else ''))
        print('|' + '---|' * (12 + bool(show_exp)))
        for r in rs:
            g = f" {r['experiment']} |" if show_exp else ''
            print(f"| {r['agent']} | {r['fw']} | {r['actor']} | {r['variant']} | {r['n']} | "
                  f"{f(r['c1']).strip()} | {f(r['c2']).strip()} | {f(r['c4']).strip()} | **{f(r['c3']).strip()}** | "
                  f"{sd(r['dgoal'], r['mixed']).strip()} | {sd(r['dstart'], r['mixed']).strip()} | "
                  f"{sd(r['inter'], r['mixed']).strip()} |{g}")
        return
    print(f'\n=== {title} ({len(rs)} cells) ===')
    print(f'{"agent":<6}{"fw":<5}{"actor":<9}{"variant":<28}{"rt":>3}{"n":>3} '
          f'{"C1":>6}{"C2":>6}{"C4":>6}{"C3":>7}  {"dGOAL":>7}{"dSTART":>7}{"INTER":>7}  '
          f'{"experiment" if show_exp else ""}')
    print(f'{"":<6}{"":<5}{"":<9}{"":<28}{"":>3}{"":>3} '
          f'{"ID/ID":>6}{"ID/OO":>6}{"OO/ID":>6}{"EVAL":>7}')
    print('-' * 142)
    for r in rs:
        a = ANCHOR.get(r['group'])
        an = f"   [pub {a:.3f}]" if a else ''
        print(f"{r['agent']:<6}{r['fw']:<5}{r['actor']:<9}{r['variant'][:27]:<28}{r['route']:>3}{r['n']:>3} "
              f"{f(r['c1'])}{f(r['c2'])}{f(r['c4'])}{f(r['c3']):>7}  "
              f"{sd(r['dgoal'], r['mixed'])}{sd(r['dstart'], r['mixed'])}{sd(r['inter'], r['mixed'])}  "
              f"{(r['experiment'][:44] if show_exp else '')}{an}")


def write_csv(path, rs):
    """One row per cell. Cross-route deltas are written EMPTY, never as a number."""
    import csv as _csv
    cols = ['block', 'agent', 'framework', 'actor', 'variant', 'experiment', 'n_seeds',
            'z_route', 'C1_id_start_id_goal', 'C2_id_start_ood_goal', 'C4_ood_start_id_goal',
            'C3_eval_ood_start_ood_goal', 'dGOAL', 'dSTART', 'INTER',
            'deltas_cross_route', 'published_anchor', 'wandb_group']
    nn = lambda x: '' if x is None or np.isnan(x) else f'{x:.3f}'
    with open(path, 'w', newline='') as fh:
        w = _csv.writer(fh)
        w.writerow(cols)
        for r in rs:
            dl = ['', '', ''] if r['mixed'] else [nn(r['dgoal']), nn(r['dstart']), nn(r['inter'])]
            a = ANCHOR.get(r['group'])
            w.writerow([r['bucket'], r['agent'], 'torch' if r['fw'] == 'tor' else 'jax',
                        r['actor'], r['variant'], r['experiment'], r['n'],
                        '' if r['route'] == '-' else r['route'],
                        nn(r['c1']), nn(r['c2']), nn(r['c4']), nn(r['c3']),
                        *dl, 'yes' if r['mixed'] else 'no',
                        '' if a is None else f'{a:.3f}', r['group']])
    print(f'[csv] {path}: {len(rs)} rows')


if CSV:
    write_csv(f'{SP}/oodgrid_bridge_table.csv', anchors + bridge)
    sys.exit(0)

emit('ANCHORS (canonical agents, csv=summary)', anchors, show_exp=False)
emit('BRIDGE PROGRAM — every experiment (csv=bridge)', bridge)

n_raw = sum(1 for r in bridge if not r['translated'])
print(f"\n{n_raw}/{len(bridge)} bridge cells still show their raw results_map text in "
      f"`experiment` (no grounded plain-RL restatement written yet).")
n_mixed = sum(r['mixed'] for r in bridge)
print(f"\n{len(bridge)} bridge cells with OOD results ({len(MAP)} groups in results_map, "
      f"{sum(1 for v in MAP.values() if v.get('csv') == 'bridge')} bridge total).")
print(f"{n_mixed} torch-FB rows have CROSS-ROUTE deltas (C1/C4=zB, C2/C3=zR) -> shown as '.', not a number.")
if unmapped and ALL:
    print(f"\nunmapped/excluded groups with results ({len(unmapped)}):")
    for r in unmapped:
        print(f"  {r['group'][:60]:<62}{f(r['c1'])}{f(r['c2'])}{f(r['c4'])}{f(r['c3'])}")
elif unmapped:
    print(f"{len(unmapped)} groups excluded/unmapped in results_map (--all to list).")
