"""Walk the persistent checkpoint corpus and emit an auditable MANIFEST.csv:
one row per real run, mapped to its canonical (env, agent, actor, variant) identity and
joined to its success number from the living results CSVs.

Read-only: parses flags.json (jax) / .hydra config (torch), never copies or moves anything.

jax  : <root>/factored-fb/<GROUP>/<seed_dir>/{params_*.pkl, flags.json}
       identity via flags.json['run_group'] -> results_map.MAP (direct).
torch: <root>/<ts>__<env>__<agent>__<seed>/{checkpoints/*.pt, .hydra/*.yaml}
       wandb_group is NOT in the checkpoint config, so variant is inferred from the
       distinguishing override flags; unmatched torch runs are flagged mapped=no.

Smoke filter: num_train_steps / max saved step below SMOKE_STEPS => is_smoke=yes.

Usage: .venv/bin/python scripts/hf_manifest.py [--out analysis/atlas/MANIFEST.csv]
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from results_map import MAP  # noqa: E402

ROOT = os.path.expanduser('~/bridge_ckpt_backup')
REPO = '/home/adityamohan/git/Factored-FB'
SMOKE_STEPS = 100_000

# distinguishing torch override keys that separate variants (drop boilerplate like seed,
# data_path, eval_*). Used to build a variant_hint for unmapped torch runs.
_BORING = {'seed', 'data_path', 'num_train_steps', 'eval_every', 'save_every', 'log_every',
           'eval_n_episodes', 'eval_relabel_size', 'load_n_episodes', 'save_eval_videos',
           'use_wandb', 'domain', 'device', 'eval_shift_reward'}


def _load_success():
    """wandb_group -> (mean, ci95, env, agent, actor, variant) from the living CSVs."""
    out = {}
    for f in ('results/summary_agents.csv', 'results/bridge_grafts.csv'):
        p = os.path.join(REPO, f)
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            g = r.get('wandb_group')
            if g:
                out[g] = (r.get('mean', ''), r.get('ci95', ''), r.get('env', ''),
                          r.get('agent', ''), r.get('actor', ''), r.get('variant', ''))
    return out


SUCCESS = _load_success()


def _dir_size_mb(d):
    tot = 0
    for dp, _, fs in os.walk(d):
        for f in fs:
            try:
                tot += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return round(tot / 1e6, 1)


def _jax_rows(root):
    rows = []
    for flags in glob.glob(f'{root}/**/flags.json', recursive=True):
        d = os.path.dirname(flags)
        pkls = sorted(glob.glob(f'{d}/params_*.pkl'))
        if not pkls:
            continue
        try:
            cfg = json.load(open(flags))
        except Exception:
            continue
        ag = cfg.get('agent', {}) if isinstance(cfg.get('agent'), dict) else {}
        group = cfg.get('run_group') or ''
        steps = sorted(int(re.search(r'params_(\d+)', p).group(1)) for p in pkls)
        m = MAP.get(group, {})
        s = SUCCESS.get(group, ('', '', '', '', '', ''))
        rows.append(dict(
            framework='jax', root=os.path.basename(root), path=os.path.relpath(d, ROOT),
            wandb_group=group, mapped='yes' if m else 'no',
            env=m.get('env', s[2]), agent=m.get('agent', s[3]),
            actor=m.get('actor', ag.get('actor_kind') or 'flow'),
            variant=m.get('variant', ag.get('value_arch') or 'mlp'),
            seed=cfg.get('seed', ''), steps=';'.join(map(str, steps)),
            max_step=steps[-1], is_smoke='yes' if steps[-1] < SMOKE_STEPS else 'no',
            success=s[0], ci95=s[1], size_mb=_dir_size_mb(d)))
    return rows


def _torch_rows(root):
    rows = []
    for cfgp in glob.glob(f'{root}/**/.hydra/config.yaml', recursive=True):
        d = os.path.dirname(os.path.dirname(cfgp))
        pts = glob.glob(f'{d}/checkpoints/*.pt')
        if not pts:
            continue
        try:
            cfg = yaml.safe_load(open(cfgp))
            ov = yaml.safe_load(open(os.path.join(d, '.hydra', 'overrides.yaml'))) or []
        except Exception:
            continue
        ovd = dict(x.split('=', 1) for x in ov if '=' in x)
        nts = int(ovd.get('num_train_steps', cfg.get('num_train_steps', 0) or 0))
        name = os.path.basename(d)
        parts = name.split('__')
        env = parts[1] if len(parts) > 1 else cfg.get('domain', '')
        agent = parts[2] if len(parts) > 2 else ''
        seed = ovd.get('seed', '')
        # wandb_group lives in the OVERRIDES for most torch runs (config.yaml has it null).
        group = ovd.get('wandb_group') or cfg.get('wandb_group') or ''
        m = MAP.get(group, {})
        s = SUCCESS.get(group, ('', '', '', '', '', ''))
        hint = ','.join(f'{k}={v}' for k, v in sorted(ovd.items()) if k not in _BORING)
        rows.append(dict(
            framework='torch', root=os.path.basename(root), path=os.path.relpath(d, ROOT),
            wandb_group=group, mapped='yes' if m else 'no',
            env=m.get('env', s[2]) or env, agent=m.get('agent', s[3]) or agent,
            actor=m.get('actor', 'flowbc' if 'flowbc' in agent else agent),
            variant=m.get('variant', hint[:60]), seed=seed, steps=str(len(pts)) + ' .pt',
            max_step=nts, is_smoke='yes' if nts < SMOKE_STEPS else 'no',
            success=s[0], ci95=s[1], size_mb=_dir_size_mb(d)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(REPO, 'analysis/atlas/MANIFEST.csv'))
    a = ap.parse_args()

    jax_roots = ['gciql_outputs', 'gciql_outputs2', 'gciql_outputs3', 'gciql_outputs_vis',
                 'GCIQL-State-Flow-Bilinear-SG-d50']
    torch_roots = ['factored-fb_outputs', 'factored-fb', 'fb_outputs', 'phase2_out',
                   'bridge_knobs_out', 'zg_cube_out', 'decpl2_out']
    rows = []
    for r in jax_roots:
        p = os.path.join(ROOT, r)
        if os.path.isdir(p):
            rows += _jax_rows(p)
    for r in torch_roots:
        p = os.path.join(ROOT, r)
        if os.path.isdir(p):
            rows += _torch_rows(p)

    cols = ['framework', 'root', 'env', 'agent', 'actor', 'variant', 'seed', 'max_step',
            'is_smoke', 'mapped', 'success', 'ci95', 'steps', 'size_mb', 'wandb_group', 'path']
    with open(a.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x['framework'], x['env'], x['agent'],
                                             str(x['variant']), str(x['seed']))):
            w.writerow({c: r.get(c, '') for c in cols})

    real = [r for r in rows if r['is_smoke'] == 'no']
    smoke = [r for r in rows if r['is_smoke'] == 'yes']
    mapped = [r for r in real if r['mapped'] == 'yes']
    gb = sum(r['size_mb'] for r in rows) / 1000
    real_gb = sum(r['size_mb'] for r in real) / 1000
    print(f'{a.out}: {len(rows)} runs  ({len(real)} real, {len(smoke)} smoke)')
    print(f'  real mapped to a paper cell: {len(mapped)}/{len(real)}')
    print(f'  jax: {sum(1 for r in rows if r["framework"]=="jax")}  '
          f'torch: {sum(1 for r in rows if r["framework"]=="torch")}')
    print(f'  size: {gb:.0f} GB total, {real_gb:.0f} GB real (upload target)')


if __name__ == '__main__':
    main()
