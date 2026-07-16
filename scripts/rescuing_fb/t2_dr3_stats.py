"""T2 (rescuing-fb probes spec): DR3 feature co-adaptation across training.

DR3 (Kumar et al., 2112.04716): TD implicitly grows the dot product between the
last-layer features of bootstrap-adjacent pairs (s,a) and (s',a'). Hypothesis
(01-synthesis-deep-audit §2.3): removing the psi(g) goal bottleneck
(value_arch='bilinear_sg', psi(s,g)) opens a shared-s co-adaptation channel on
Q = phi.psi; its runaway growth is what the sg divergence (q_mean crossing
positive) looks like. Stock bilinear psi(g) and the MLP critic should stay flat.

Per checkpoint, on ONE fixed dataset batch (B=4096, fixed RNG, raw ogbench npz):

    coad = E[phi(s,a) . phi(s',a'_ds)] / sqrt(E[||phi(s,a)||^2] E[||phi(s',a'_ds)||^2])

(a'_ds = the dataset's next action) plus E[||phi||^2], E[||psi||^2], q stats.
phi = the explicit phi tower for bilinear/bilinear_sg; the penultimate-layer
activations (last pre-output LayerNorm, via flax capture_intermediates) for mlp.
Per ensemble head (head 0, head 1) + head mean. Goals are sampled once with the
training value-goal mixture (0.2 cur / 0.5 traj-geometric(1-discount) / 0.3
random) so q_mean is comparable to the training logs. NOTE: train.csv's
training/critic/q_mean is the TD *target* r + gamma*mask*V(s'); here q_* are the
critic's own predictions Q(s,a,g) (min over heads for q_mean/q_min/q_max).

Run under /dev/shm/.venv-jax/bin/python (CPU is fine):
    JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python scripts/rescuing_fb/t2_dr3_stats.py
Output: analysis/rescuing-fb/t2_dr3_stats.json (incremental; restartable).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from scripts.atlas.jax_load import load_agent  # noqa: E402  (sets up jax paths)

DATA = os.path.expanduser('~/.ogbench/data/cube-single-play-v0.npz')
OUT = REPO / 'analysis/rescuing-fb/t2_dr3_stats.json'
SEED = 20260716
B = 4096
DISCOUNT = 0.99
# training value-goal mixture (all runs here share these; verified in flags.json)
P_CUR, P_TRAJ, P_RAND = 0.2, 0.5, 0.3

SG = os.path.expanduser('~/bridge_ckpt_backup/GCIQL-State-Flow-Bilinear-SG-d50')
E5 = '/home/adityamohan/exp5_retrain/factored-fb'
REF = os.path.expanduser(
    '~/bridge_ckpt_backup/gciql_outputs/factored-fb/GCIQL-State-Flow-Bilinear-d50/sd000_20260615_011444')

E5_STEPS = list(range(50000, 500001, 50000))
CKPTS = (
    [(f'{E5}/GCIQL-State-Flow-Bilinear-d50/exp5_bilinear_sd0', st, 'bilinear', 'exp5_bilinear_sd0')
     for st in E5_STEPS]
    + [(f'{E5}/GCIQL-State-Flow/exp5_mlp_sd0', st, 'mlp', 'exp5_mlp_sd0') for st in E5_STEPS]
    + [(REF, 500000, 'bilinear', 'ref_bilinear_sd000'),
       (REF, 1000000, 'bilinear', 'ref_bilinear_sd000')]
    + [(f'{SG}/cube_single_state_gciql_flow_bilinear_sg_d50_s{s}', 500000, 'bilinear_sg', f'sg_s{s}')
       for s in range(3)]
)


def build_batch():
    """Fixed (s, a, s', a'_ds, g) batch from the raw ogbench npz.

    Raw npz convention (ogbench.utils.load_dataset docstring): terminals[i]=1 marks the
    FINAL state of a trajectory (its action has no next state). Valid transition indices
    are i with terminals[i]==0: s'=obs[i+1], a'_ds=actions[i+1]. We additionally require
    terminals[i+1]==0 so a'_ds is an action that was actually followed by a state.
    Goals replicate GCDataset.sample_goals for the value goals (cur/traj-geom/random).
    """
    d = np.load(DATA)
    obs, act, term = d['observations'], d['actions'], d['terminals'].astype(bool)
    rng = np.random.default_rng(SEED)
    valid = np.nonzero(~term[:-1] & ~term[1:])[0]
    idx = rng.choice(valid, size=B, replace=False)

    terminal_locs = np.nonzero(term)[0]
    final = terminal_locs[np.searchsorted(terminal_locs, idx)]
    offsets = rng.geometric(p=1 - DISCOUNT, size=B)
    traj_g = np.minimum(idx + offsets, final)
    rand_g = rng.choice(len(obs), size=B)
    g_idx = np.where(rng.random(B) < P_TRAJ / (1 - P_CUR), traj_g, rand_g)
    g_idx = np.where(rng.random(B) < P_CUR, idx, g_idx)

    return dict(s=obs[idx], a=act[idx], sn=obs[idx + 1], an=act[idx + 1], g=obs[g_idx])


def mlp_penult(agent, s, g, a, check_q=None):
    """Penultimate features of the ensembled GCValue critic, re-applied manually.

    flax capture_intermediates does NOT propagate through the nn.vmap ensemble lift
    (only 'params' is in variable_axes), so we re-run the MLP from the extracted
    params: Dense_i -> gelu -> LayerNorm_i for i<3; phi = LayerNorm_2 output (the
    input to the final Dense_3). Verified against the critic's own Q when check_q
    is passed (max |q_recon - q| < 1e-3).
    """
    import jax
    import jax.numpy as jnp
    p = agent.network.params['modules_critic']['value_net']
    n_dense = len([k for k in p if k.startswith('Dense')])
    x = jnp.asarray(np.concatenate([s, g, a], axis=-1))[None]        # [1,B,in]
    for i in range(n_dense - 1):
        w, b = p[f'Dense_{i}']['kernel'], p[f'Dense_{i}']['bias']    # [2,in,out],[2,out]
        x = jnp.einsum('hbi,hio->hbo', jnp.broadcast_to(x, (w.shape[0], *x.shape[1:])), w) \
            + b[:, None, :]
        x = jax.nn.gelu(x)
        ln = p[f'LayerNorm_{i}']
        mu = x.mean(-1, keepdims=True)
        var = ((x - mu) ** 2).mean(-1, keepdims=True)
        x = (x - mu) / jnp.sqrt(var + 1e-6) * ln['scale'][:, None, :] + ln['bias'][:, None, :]
    phi = np.asarray(x)                                              # [2,B,512]
    if check_q is not None:
        w, b = p[f'Dense_{n_dense - 1}']['kernel'], p[f'Dense_{n_dense - 1}']['bias']
        q_recon = np.asarray(jnp.einsum('hbi,hio->hbo', x, w) + b[:, None, :])[..., 0]
        err = np.abs(q_recon - check_q).max()
        assert err < 1e-3, f'MLP feature re-application mismatch: max|dq|={err}'
    return phi


def head_stats(phi, phin, psi, q):
    """phi, phin, psi: [2,B,d]; q: [2,B]. Returns per-head + head-mean stats."""
    res = {}
    coads, dots = [], []
    for h in range(phi.shape[0]):
        dot = float(np.mean(np.sum(phi[h] * phin[h], axis=-1)))
        n1 = float(np.mean(np.sum(phi[h] ** 2, axis=-1)))
        n2 = float(np.mean(np.sum(phin[h] ** 2, axis=-1)))
        coad = dot / np.sqrt(n1 * n2)
        coads.append(coad); dots.append(dot)
        res[f'head{h}'] = dict(
            coad=coad, dot_raw=dot, phi_sq=n1, phi_next_sq=n2,
            psi_sq=float(np.mean(np.sum(psi[h] ** 2, axis=-1))) if psi is not None else None,
            q_mean=float(q[h].mean()), q_min=float(q[h].min()), q_max=float(q[h].max()))
    res['coad_mean'] = float(np.mean(coads))
    res['dot_raw_mean'] = float(np.mean(dots))
    qmin = q.min(axis=0)
    res['q_mean'] = float(qmin.mean()); res['q_min'] = float(qmin.min())
    res['q_max'] = float(qmin.max())
    return res


def eval_ckpt(run_dir, step, arch, batch):
    import jax.numpy as jnp
    agent = load_agent(run_dir, step)
    s, a, sn, an, g = (batch[k] for k in ('s', 'a', 'sn', 'an', 'g'))
    if arch in ('bilinear', 'bilinear_sg'):
        crit = agent.network.select('critic')
        q, phi, psi = crit(jnp.asarray(s), jnp.asarray(g), jnp.asarray(a), info=True)
        _, phin, psin = crit(jnp.asarray(sn), jnp.asarray(g), jnp.asarray(an), info=True)
        q, phi, psi, phin = (np.asarray(x) for x in (q, phi, psi, phin))
        res = head_stats(phi, phin, np.asarray(psi), q)
        if arch == 'bilinear_sg':
            # extra: psi([s,g]) vs psi([s',g]) co-adaptation (the shared-s channel is
            # open on BOTH legs for sg; for stock bilinear psi(g)=psi(g) trivially)
            psin = np.asarray(psin)
            pc = [float(np.mean(np.sum(psi[h] * psin[h], -1))
                        / np.sqrt(np.mean(np.sum(psi[h] ** 2, -1))
                                  * np.mean(np.sum(psin[h] ** 2, -1)))) for h in range(2)]
            res['psi_coad_head'] = pc
            res['psi_coad_mean'] = float(np.mean(pc))
    else:  # mlp
        q12 = agent.network.select('critic')(jnp.asarray(s), jnp.asarray(g), jnp.asarray(a))
        q = np.asarray(q12)
        phi = mlp_penult(agent, s, g, a, check_q=q)
        phin = mlp_penult(agent, sn, g, an)
        res = head_stats(phi, phin, None, q)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None, help='substring filter on run tag')
    args = ap.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    batch = build_batch()
    print(f'[t2] batch B={B} seed={SEED} from {DATA}', flush=True)

    for run_dir, step, arch, tag in CKPTS:
        key = f'{tag}@{step}'
        if args.only and args.only not in key:
            continue
        if key in results:
            print(f'[t2] skip {key} (done)', flush=True)
            continue
        if not os.path.exists(os.path.join(run_dir, f'params_{step}.pkl')):
            print(f'[t2] MISSING {key}: {run_dir}/params_{step}.pkl', flush=True)
            continue
        res = eval_ckpt(run_dir, step, arch, batch)
        res.update(arch=arch, tag=tag, step=step, run_dir=run_dir)
        results[key] = res
        OUT.write_text(json.dumps(results, indent=2))
        print(f"[t2] {key:>28} ({arch:11s}): coad h0={res['head0']['coad']:+.4f} "
              f"mean={res['coad_mean']:+.4f}  phi_sq={res['head0']['phi_sq']:9.2f}  "
              f"q_mean={res['q_mean']:+8.3f} q_max={res['q_max']:+9.2f}", flush=True)

    print(f'[t2] -> {OUT} ({len(results)} entries)', flush=True)


if __name__ == '__main__':
    main()
