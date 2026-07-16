"""Exp 5: assemble the training trajectory of the bilinear mechanism.

For each saved checkpoint (params_<step>.pkl) of the exp5 bilinear run, compute:
  - rerank success  (deploy argmax_a Q over mu candidates as the policy; rerank_eval.py)
  - advantage effective rank, off-reference peak gap, off-ref win-frac (ckpt_static_metrics.py)
and emit a step-indexed trajectory. Distinguishes stable advantage learning (metrics rise then
hold) from teacher-then-drift (advantage geometry / rerank quality peak then degrade while the
off-support peak keeps sharpening).

Shells out to the two metric scripts per step (each loads jax fresh) so it is restartable and
reuses tested code. Run with the repo python; it invokes the jax venv per call.

Usage:
  python scripts/atlas/exp5_trajectory.py \
      --bil-dir /home/adityamohan/exp5_retrain/factored-fb/GCIQL-State-Flow-Bilinear-d50/exp5_bilinear_sd0 \
      [--episodes 15] [--steps 50000,100000,...]
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
JAX_PY = '/dev/shm/.venv-jax/bin/python'
ANALYSIS = REPO / 'analysis/atlas'


def discover_steps(run_dir):
    steps = sorted(int(m.group(1)) for p in Path(run_dir).glob('params_*.pkl')
                   if (m := re.match(r'params_(\d+)\.pkl', p.name)))
    return steps


def run(cmd, gpu):
    env_prefix = f'CUDA_VISIBLE_DEVICES={gpu} XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl'
    subprocess.run(f'{env_prefix} {JAX_PY} {cmd}', shell=True, cwd=str(REPO),
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bil-dir', required=True, help='exp5 bilinear run dir (holds params_*.pkl)')
    ap.add_argument('--episodes', type=int, default=15, help='rerank episodes/task (speed vs noise)')
    ap.add_argument('--n-samples', type=int, default=32)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--steps', default=None, help='comma steps; default = all discovered')
    ap.add_argument('--out-tag', default='')
    args = ap.parse_args()

    steps = ([int(s) for s in args.steps.split(',')] if args.steps
             else discover_steps(args.bil_dir))
    if not steps:
        sys.exit(f'no params_*.pkl in {args.bil_dir}')
    print(f'[exp5-traj] {len(steps)} checkpoints: {steps}', flush=True)

    traj = []
    for st in steps:
        # static metrics -> analysis/atlas/ckpt_static_bilinear_flow_<st>_traj.json
        run(f'scripts/atlas/ckpt_static_metrics.py --step {st} --critic bilinear_flow '
            f'--run-dir {args.bil_dir} --n-states 8 --out-tag _traj', args.gpu)
        stat = json.loads((ANALYSIS / f'ckpt_static_bilinear_flow_{st}_traj.json').read_text())
        # rerank success -> analysis/atlas/rerank_eval_traj<st>.json
        run(f'scripts/atlas/rerank_eval.py --n-samples {args.n_samples} --episodes {args.episodes} '
            f'--max-steps 200 --mu bilinear_flow --bil-dir {args.bil_dir} --bil-step {st} '
            f'--out-tag _traj{st}', args.gpu)
        rr = json.loads((ANALYSIS / f'rerank_eval_traj{st}.json').read_text())['success']
        row = dict(step=st, rerank_success=rr['bilinear_flow']['overall'],
                   floor=rr['floor']['overall'],
                   effective_rank=stat['effective_rank'],
                   offref_peak_gap=stat['offref_peak_gap'],
                   offref_win_frac=stat['offref_win_frac'])
        traj.append(row)
        print(f"  step {st:>7}: rerank {row['rerank_success']:.3f}  eff_rank {row['effective_rank']:.2f}  "
              f"peak_gap {row['offref_peak_gap']:+.2f}  floor {row['floor']:.3f}", flush=True)

    out = ANALYSIS / f'exp5_trajectory{args.out_tag}.json'
    out.write_text(json.dumps({'bil_dir': args.bil_dir, 'episodes': args.episodes,
                               'trajectory': traj}, indent=2))
    print(f'[exp5-traj] -> {out}', flush=True)


if __name__ == '__main__':
    main()
