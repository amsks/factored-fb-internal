"""Uniform Q(s, A, G) -> [S, n_a, n_g] across the jax GCIQL cells and the torch FB cell.

jax GCIQL (bilinear/mlp x flow/ddpgbc, simplesf; both gciql and gciql_flow agent classes):
    Q via network.select('critic')(obs, goals, actions) -> (q1, q2); use min.  [jax venv]
torch FB: reward-inferred z_g per goal (infer_z(obs, reward)); Q = <F(s,a,z_g), z_g>. [repo .venv]

The two stacks run under different venvs and are loaded lazily so importing this module
never requires both.
"""
import os

import numpy as np

from scripts.atlas.jax_load import CRITICS, load_agent

_JAX_CRITICS = {'bilinear_flow', 'bilinear_ddpgbc', 'mlp_flow', 'mlp_ddpgbc', 'simplesf'}


class _JaxGCIQL:
    """Uniform for both gciql and gciql_flow: critic(obs, goals, actions) -> (q1, q2)."""

    def __init__(self, agent):
        self.agent = agent

    def Q(self, obs, actions, goals):
        import jax.numpy as jnp
        S, na, _ = actions.shape
        ng = goals.shape[0]
        out = np.zeros((S, na, ng), np.float32)
        obs_rep = np.repeat(obs[:, None], na, axis=1).reshape(-1, 28)   # [S*na, 28]
        acts = actions.reshape(-1, 5)                                    # [S*na, 5]
        for gi in range(ng):
            g = np.repeat(goals[gi][None], S * na, axis=0)
            q1, q2 = self.agent.network.select('critic')(
                jnp.asarray(obs_rep), jnp.asarray(g), jnp.asarray(acts))
            out[:, :, gi] = np.asarray(jnp.minimum(q1, q2)).reshape(S, na)
        return out


class _TorchFB:
    """FB critic with reward-inferred z per goal. Runs under the repo .venv (torch)."""

    def __init__(self, agent, ref_obs):
        self.agent = agent
        self.ref = ref_obs                      # fixed reference batch for infer_z [R, 28]

    def _reward_for_goal(self, goal):
        """Sparse success indicator over the reference batch: cube within 4cm of the goal
        cube, matching the dataset's exact-match reward semantics (datasets.py:242)."""
        import torch
        ref_cube = self.ref[:, 14:17]
        goal_cube = np.asarray(goal)[14:17]
        hit = (np.linalg.norm(ref_cube - goal_cube[None], axis=1) <= 0.04).astype(np.float32)
        return torch.as_tensor(hit)

    def z_identifiable(self, goals):
        return np.array([int(self._reward_for_goal(g).sum()) > 0 for g in goals], dtype=bool)

    def _z(self, goal):
        return self.agent.infer_z(_totorch(self.ref), self._reward_for_goal(goal))

    def Q(self, obs, actions, goals):
        import torch
        S, na, _ = actions.shape
        ng = goals.shape[0]
        out = np.zeros((S, na, ng), np.float32)
        m = self.agent.model
        left = m._left_encoder(_totorch(np.repeat(obs[:, None], na, 0).reshape(-1, 28)))
        act_t = _totorch(actions.reshape(-1, 5))
        for gi in range(ng):
            z = self._z(goals[gi])                                  # [z_dim]
            z_b = z[None].expand(S * na, -1)
            q = self.agent._fb_Q_heads(left, z_b, act_t, m._forward_map)  # [n_parallel, S*na]
            q = q.min(0).values if q.dim() == 2 else q             # pessimistic over heads
            out[:, :, gi] = q.detach().cpu().numpy().reshape(S, na)
        return out


def _totorch(x):
    import torch
    return torch.as_tensor(np.asarray(x, np.float32))


def load_critic(spec):
    if spec in _JAX_CRITICS:
        ckpt, step = CRITICS[spec]
        return _JaxGCIQL(load_agent(ckpt, step))
    if spec == 'fb':
        return _load_fb()
    raise ValueError(f'unknown critic spec: {spec!r}')


def _load_fb():
    """Torch FB via the same loader oodgrid_run_torch uses. Reference batch = a fixed
    slice of the offline dataset for infer_z."""
    import sys
    sys.path.insert(0, '/home/adityamohan/git/Factored-FB')
    from data.ogbench import load_ogbench_dataset
    from evals.analysis import build_env_and_agent, load_cfg, load_checkpoint
    run = ('/home/adityamohan/bridge_ckpt_backup/factored-fb_outputs/'
           '2026-05-14_21-34-52__cube-single-play-v0__fb_flowbc__s0')
    cfg = load_cfg(f'{run}/.hydra/config.yaml', device='cpu')
    _, agent, _, _ = build_env_and_agent(cfg)
    load_checkpoint(agent, f'{run}/checkpoints/final.pt', map_location='cpu')
    ds = load_ogbench_dataset(domain='cube-single-play-v0',
                              # FB per-episode buffer, moved off volatile /dev/shm to persistent
                              data_path=os.path.expanduser('~/bridge_ckpt_backup/datasets'),
                              load_n_episodes=50, device='cpu')
    ref = ds.sample(2048)['observation'].cpu().numpy().astype(np.float32)
    return _TorchFB(agent, ref)
