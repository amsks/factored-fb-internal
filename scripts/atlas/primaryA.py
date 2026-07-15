"""Primary A: does the bilinear critic rank mu-reference actions better than the MLP under
recombined goals, AND does its flow actor consume that advantage?

Ranking is measured as Spearman correlation of the centered atlas A^m_s(a,g) with the
behavior-constrained oracle J^MPC_mu(s,a,g) over the mu-reference set B_mu.

Factorial cells (start x goal), using prefix-replayable states so the oracle is exact:
  ID start  = mu-rollout on-manifold states (banks phases P1..P3)
  OOD start = reset states (banks phase P0)
  ID goal   = hindsight goals            OOD goal = eval-task goals
    C1 ID/ID   C2 ID/OOD   C4 OOD/ID   C3 OOD/OOD(=benchmark)
Pre-registered: rho_MLP>=rho_bil (C1); ~tie (C4); rho_bil>rho_MLP (C2, C3, C3 largest).
Runs under the jax venv.
"""
import numpy as np
from scipy.stats import spearmanr

from scripts.atlas import config as C
from scripts.atlas.build_atlas import center_and_standardize
from scripts.atlas.oracle import oracle_J
from scripts.atlas.prefix_replay import PrefixState

CELLS = {                        # name -> (phase, goal_source)
    'C1': (2, 'hindsight'),      # ID start (on-manifold) + ID goal (reachable hindsight)
    'C2': (2, 'eval'),          # ID start + OOD goal (eval task)
    'C4': (0, 'hindsight'),     # OOD start (reset) + ID goal
    'C3': (0, 'eval'),          # OOD start + OOD goal = the benchmark
}


def _cell_goals(banks, phase, k, source, n=5):
    """Goals for one (phase, state). Eval/grid goals are shared [n,28]; hindsight goals are
    PER-STATE and mu-reachable (banks['goals_hindsight_ps'][phase,k]) so the oracle can rank."""
    if source == 'eval':
        return banks['goals_eval'][:n]
    if source == 'grid':
        return banks['goals_grid'][:n]
    if source == 'hindsight':
        return banks['goals_hindsight_ps'][phase, k][:n]
    raise ValueError(source)


def spearman_gap(a_vals, j_vals):
    r = spearmanr(a_vals, j_vals).correlation
    return float(r) if np.isfinite(r) else 0.0


def equivalence_verdict(rho_bil, rho_mlp, delta=None):
    delta = C.EQUIV_MARGIN_DELTA if delta is None else delta
    if abs(rho_bil - rho_mlp) <= delta:
        return 'tie'
    return 'bilinear' if rho_bil > rho_mlp else 'mlp'


def _ps(banks, p, k):
    L = int(banks['state_prefix_len'][p, k])
    return PrefixState(seed=k % 5000, task_id=1, prefix=banks['state_prefix_actions'][p, k, :L])


def _atlas_col(critic, obs, acts, mu_center, goal):
    """Centered+standardized A^m_s(.,g) for one state and one goal: [n_acts]."""
    Q = critic.Q(obs[None], acts[None], goal[None])[0]           # [n_acts, 1]
    Qmu = critic.Q(obs[None], mu_center[None], goal[None])[0]    # [N_MU, 1]
    return center_and_standardize(Q, Qmu)[:, 0]


def primary_a_cell(critics, banks, pool, mu, cell, n_states=3, n_act=16, H=20, replicates=2):
    """One factorial cell. Returns per-critic mean Spearman rho over (state, goal) pairs."""
    phase, source = CELLS[cell]
    rng = np.random.RandomState(C.SEED)
    rhos = {name: [] for name in critics}
    for k in range(n_states):
        obs = banks['state_obs'][phase, k]
        A_mu = banks['A_mu'][phase, k]
        mu_center = banks['mu_center_samples'][phase, k]
        idx = rng.choice(len(A_mu), n_act, replace=False)
        acts = A_mu[idx]
        ps = _ps(banks, phase, k)
        for g in _cell_goals(banks, phase, k, source):
            J = np.array([oracle_J(pool, mu, ps, a, g, replicates=replicates, H=H)
                          for a in acts])
            for name, crit in critics.items():
                A = _atlas_col(crit, obs, acts, mu_center, g)
                rhos[name].append(spearman_gap(A, J))
    return {name: float(np.mean(v)) for name, v in rhos.items()}


def primary_a(critics, banks, pool, mu, cells=('C1', 'C2'), **kw):
    out = {}
    for cell in cells:
        r = primary_a_cell(critics, banks, pool, mu, cell, **kw)
        rb, rm = r.get('bilinear_flow'), r.get('mlp_flow')
        out[cell] = dict(r, verdict=equivalence_verdict(rb, rm) if rb is not None else None)
    return out
