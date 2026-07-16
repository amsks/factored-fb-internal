"""Build the manifest for the OOD-generalization grid over the FB<->GCIQL bridge.

Scans the checkpoint roots and emits ONE row per (group, seed) that is actually
evaluable, with the config READ FROM THE RECORDED CONFIG FILE -- never inferred from
the directory name (this repo has documented name-vs-config traps: a group called
`fb_state_L256` has L_dim=50; `GCIQL-State-Flow-Bilinear` is d=512, not d=50).

Fail-loud: every candidate dir is either emitted with a verified config or recorded in
`skipped` with a reason. Nothing is silently dropped.

Usage: .venv/bin/python scripts/oodgrid_manifest.py --out analysis/oodgrid/manifest.json
"""
import argparse
import glob
import json
import os
from pathlib import Path

import yaml

JAX_ROOTS = ['/dev/shm/gciql_outputs', '/home/adityamohan/bridge_ckpt_backup/gciql_outputs']
TORCH_ROOTS = ['/dev/shm/factored-fb/runs', '/dev/shm/factored-fb_outputs',
               '/dev/shm/bridge_knobs_out', '/dev/shm/phase2_out', '/dev/shm/decpl2_out',
               '/dev/shm/zg_cube_out', '/dev/shm/idB_dnorm_out', '/dev/shm/r3bilinv_out',
               '/home/adityamohan/bridge_ckpt_backup/factored-fb/runs',
               '/home/adityamohan/bridge_ckpt_backup/factored-fb_outputs']
DOMAIN = 'cube-single-play-v0'
MIN_STEP = 100_000          # anything below this is a smoke stub, not a run


def jax_rows():
    rows, skipped = [], []
    seen = set()
    for root in JAX_ROOTS:
        for fj in glob.glob(f'{root}/**/flags.json', recursive=True):
            d = os.path.dirname(fj)
            try:
                f = json.load(open(fj))
            except Exception as e:
                skipped.append({'dir': d, 'reason': f'unreadable flags.json: {e}'})
                continue
            if f.get('env_name') != DOMAIN:
                continue
            steps = sorted(int(Path(p).stem.split('_')[1])
                           for p in glob.glob(f'{d}/params_*.pkl'))
            steps = [s for s in steps if s >= MIN_STEP]
            if not steps:
                skipped.append({'dir': d, 'reason': 'no params_*.pkl >= 100k (stub)'})
                continue
            ag = f.get('agent', {})
            group = f.get('run_group') or Path(d).parent.name
            # Seed key MUST be the full basename. Seed dirs are named inconsistently
            # (`sd000_<ts>` in some groups, `cube_single_state_..._s0` in others); keying
            # on name.split('_')[0] collapsed all three anneal seeds to "cube" and silently
            # dropped 2 of 3. The basename also dedups the backup mirror against /dev/shm,
            # since the mirror preserves dir names.
            key = (group, Path(d).name)
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                'framework': 'jax', 'group': group, 'dir': d, 'step': steps[-1],
                'seed': Path(d).name,
                'agent_name': ag.get('agent_name'),
                'value_arch': ag.get('value_arch', 'mlp'),
                'value_latent_dim': ag.get('value_latent_dim'),
                'actor_kind': ag.get('actor_kind', 'flow'),
                'actor_cond': ag.get('actor_cond', 'goal'),
                'cfg': ag,
            })
    return rows, skipped


def torch_rows():
    rows, skipped = [], []
    seen = set()
    for root in TORCH_ROOTS:
        for cy in glob.glob(f'{root}/**/.hydra/config.yaml', recursive=True):
            d = os.path.dirname(os.path.dirname(cy))
            try:
                c = yaml.safe_load(open(cy))
                ov = yaml.safe_load(open(f'{d}/.hydra/overrides.yaml')) or []
            except Exception as e:
                skipped.append({'dir': d, 'reason': f'unreadable hydra cfg: {e}'})
                continue
            if c.get('domain') != DOMAIN or c.get('obs_type') != 'state':
                continue
            cks = glob.glob(f'{d}/checkpoints/*.pt')
            if not cks:
                skipped.append({'dir': d, 'reason': 'no checkpoints/*.pt'})
                continue
            ck = f'{d}/checkpoints/final.pt' if os.path.exists(f'{d}/checkpoints/final.pt') \
                else sorted(cks)[-1]
            group = c.get('wandb_group') or Path(d).name.split('__s')[0]
            key = (group, Path(d).name)          # full basename; see jax_rows() note
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                'framework': 'torch', 'group': group, 'dir': d, 'ckpt': ck,
                'seed': Path(d).name,
                'agent_name': c.get('agent'), 'z_dim': c.get('z_dim'),
                'cfg': {'overrides': ov},
            })
    return rows, skipped


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='analysis/oodgrid/manifest.json')
    ap.add_argument('--max-seeds', type=int, default=3)
    a = ap.parse_args()
    jr, js = jax_rows()
    tr, ts = torch_rows()
    rows = jr + tr

    # cap seeds per group (deterministically, by dir name) so the sweep is bounded
    by_group = {}
    for r in sorted(rows, key=lambda x: (x['group'], x['seed'])):
        by_group.setdefault(r['group'], []).append(r)
    capped, dropped = [], 0
    for g, rs in by_group.items():
        capped.extend(rs[:a.max_seeds])
        dropped += max(0, len(rs) - a.max_seeds)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({'rows': capped, 'skipped': js + ts,
               'seeds_dropped_by_cap': dropped, 'max_seeds': a.max_seeds},
              open(a.out, 'w'), indent=1)
    nj = sum(r['framework'] == 'jax' for r in capped)
    print(f'evaluable run-dirs: {len(rows)}  ->  after {a.max_seeds}-seed cap: {len(capped)}'
          f'  (jax {nj}, torch {len(capped) - nj})')
    print(f'distinct groups:    {len(by_group)}')
    print(f'seeds dropped by cap (LOGGED, not silent): {dropped}')
    print(f'skipped (recorded with reason): {len(js) + len(ts)}')
    ns = {g: len(rs) for g, rs in by_group.items()}
    print(f'groups with only 1 seed: {sum(v == 1 for v in ns.values())}')
    print(f'\nwrote {a.out}')
