"""Direct critic-side confirmation of the actor reversal (no simulator).

Phase 0A showed the DDPG+BC action departs the mu-reference cloud far more under the bilinear
critic. This asks WHETHER that departure is explained by an attractive off-reference critic
PEAK, and distinguishes it from a mere STEEPER LANDSCAPE:

  peak      : Q(s, a_DDPG, g) - max_{a in B_mu} Q(s, a, g)   (>0 => off-ref peak above best ref)
  path rise : Q along the segment a_ref* -> a_DDPG rises monotonically toward a_DDPG
  gradient  : ||grad_a Q(s,a,g)|| over B_mu (bilinear >> mlp => steeper landscape, an
              alternative that shifts the DDPG+BC BC/RL tradeoff even without a separate peak)

a_DDPG and Q come from the SAME co-trained agent (bilinear_ddpgbc / mlp_ddpgbc). Q = min(q1,q2).
Run under the jax venv, CPU.
"""
import json
import sys

import numpy as np

sys.path.insert(0, '/home/adityamohan/git/Factored-FB')

from scripts.atlas.banks import load_banks
from scripts.atlas.critics import load_critic

N_STATES = 8
PATH_STEPS = 11


def _q_min(agent, obs, goal, acts):
    """min(q1,q2) for [n,5] actions at one (obs,goal). Returns [n]."""
    import jax.numpy as jnp
    n = acts.shape[0]
    o = jnp.asarray(np.repeat(obs[None], n, 0))
    g = jnp.asarray(np.repeat(goal[None], n, 0))
    q1, q2 = agent.network.select('critic')(o, g, jnp.asarray(acts))
    return np.asarray(jnp.minimum(q1, q2))


def _grad_norm(agent, obs, goal, acts):
    """mean ||grad_a min(q1,q2)|| over the given actions."""
    import jax
    import jax.numpy as jnp

    def qscalar(a):
        q1, q2 = agent.network.select('critic')(obs[None], goal[None], a[None])
        return jnp.minimum(q1, q2)[0]
    g = jax.vmap(jax.grad(qscalar))(jnp.asarray(acts))
    return float(np.mean(np.linalg.norm(np.asarray(g), axis=1)))


def run_critic(spec, banks):
    crit = load_critic(spec)
    agent = crit.agent
    goals = banks['goals_eval']
    peaks, path_rise, gnorms = [], [], []
    import jax
    for p in (0, 2):
        for k in range(N_STATES):
            obs = banks['state_obs'][p, k]
            Bmu = banks['A_mu'][p, k]                      # [256,5]
            for gi in range(len(goals)):
                g = goals[gi]
                a_ddpg = np.clip(np.asarray(agent.sample_actions(
                    observations=obs[None], goals=g[None],
                    seed=jax.random.PRNGKey(0), temperature=0.0))[0], -1, 1)
                q_ref = _q_min(agent, obs, g, Bmu)
                a_best = Bmu[int(np.argmax(q_ref))]
                q_ddpg = float(_q_min(agent, obs, g, a_ddpg[None])[0])
                peaks.append(q_ddpg - float(q_ref.max()))
                # value along a_best -> a_ddpg
                lam = np.linspace(0, 1, PATH_STEPS)[:, None]
                seg = (1 - lam) * a_best[None] + lam * a_ddpg[None]
                qseg = _q_min(agent, obs, g, seg.astype(np.float32))
                path_rise.append(float(qseg[-1] - qseg[0]))    # ddpg end minus ref end
                gnorms.append(_grad_norm(agent, obs, g, Bmu[:64]))
    peaks = np.array(peaks)
    return dict(
        n=len(peaks),
        peak_median=round(float(np.median(peaks)), 3),
        peak_frac_positive=round(float((peaks > 0).mean()), 3),
        path_rise_median=round(float(np.median(path_rise)), 3),
        path_rise_frac_positive=round(float((np.array(path_rise) > 0).mean()), 3),
        grad_norm_mean=round(float(np.mean(gnorms)), 4))


def main():
    b = load_banks('/home/adityamohan/git/Factored-FB/analysis/atlas/banks.npz')
    out = {}
    for spec in ('bilinear_ddpgbc', 'mlp_ddpgbc'):
        out[spec] = run_critic(spec, b)
        r = out[spec]
        print(f"{spec}: peak(Q_ddpg - max_ref) median={r['peak_median']} "
              f"frac>0={r['peak_frac_positive']} | path_rise median={r['path_rise_median']} "
              f"frac>0={r['path_rise_frac_positive']} | ||grad_a Q||={r['grad_norm_mean']}",
              flush=True)
    p = '/home/adityamohan/git/Factored-FB/analysis/atlas/off_reference_results.json'
    json.dump(out, open(p, 'w'), indent=1)
    print(f'wrote {p}', flush=True)


if __name__ == '__main__':
    main()
