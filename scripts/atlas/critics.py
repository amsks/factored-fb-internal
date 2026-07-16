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
    """FB critic with reward-inferred z per goal. Runs under the repo .venv (torch).

    z uses the *exact* FB-eval reward-relabel inference (OGBenchEvaluator._infer_z): relabel a
    large offline sub-batch with the task's cube_reward_fn, shift by +1, project. Precomputed once
    per eval task in `_load_fb` and looked up here by nearest eval goal. The OLD sparse hit-indicator
    z (cube-within-4cm over a ref batch) returned z=0 -> Q=0 on the eval goals (sparse-z fails)."""

    def __init__(self, agent, eval_goals, eval_z):
        self.agent = agent
        self.eval_goals = np.asarray(eval_goals, np.float32)   # [T, 28] task goal observations
        self.eval_z = list(eval_z)                             # [T] each 1-D z tensor (relabel-z)

    def z_identifiable(self, goals):
        # relabel-z with shift_reward=+1 is always non-degenerate
        return np.ones(len(goals), dtype=bool)

    def _z(self, goal):
        """Match the incoming goal observation to its eval task by nearest cube position and
        return that task's precomputed relabel-z."""
        gc = np.asarray(goal, np.float32)[14:17]
        d = np.linalg.norm(self.eval_goals[:, 14:17] - gc[None], axis=1)
        i = int(d.argmin())
        assert d[i] < 1e-3, (f'goal cube {gc} not among the {len(self.eval_z)} eval goals '
                             f'(nearest dist {d[i]:.4g}); relabel-z is only precomputed for those')
        return self.eval_z[i]

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
    """Torch FB via the same loader oodgrid_run_torch uses. Per-task relabel-z (the exact
    FB-eval inference) is precomputed here via OGBenchEvaluator, mirroring rerank_audit_fb.py."""
    import sys
    sys.path.insert(0, '/home/adityamohan/git/Factored-FB')
    from data.ogbench import load_ogbench_dataset
    from evals.analysis import build_env_and_agent, load_cfg, load_checkpoint
    from evals.ogbench import OGBenchEvaluator
    # Plain FB, state, cube: agent=fb_flowbc, z_dim=50, ortho_coef=1000, onestep=false, 1.5M steps
    # (inventory row fb_state_plain, load-verified). The OLD default here was a 100-step SMOKE
    # STUB (2026-05-14_21-34-52...) that scores ~0 -- any prior FB atlas result on it is invalid.
    domain = 'cube-single-play-v0'
    run = ('/home/adityamohan/bridge_ckpt_backup/factored-fb/runs/'
           '20260607_070225__fb_state_plain__s0')
    cfg = load_cfg(f'{run}/.hydra/config.yaml', device='cpu')
    _, agent, _, _ = build_env_and_agent(cfg)
    load_checkpoint(agent, f'{run}/checkpoints/step_1000000.pt', map_location='cpu')
    agent.model.eval()
    ds = load_ogbench_dataset(domain=domain,
                              # FB per-episode buffer, moved off volatile /dev/shm to persistent
                              data_path=os.path.expanduser('~/bridge_ckpt_backup/datasets'),
                              load_n_episodes=400, device='cpu')

    # relabel-z per eval task (task_id 1..5), same order as banks['goals_eval'] (built from
    # env.reset(task_id=t) in banks.py::_eval_goals).
    evaluator = OGBenchEvaluator(domain=domain, agent=agent, offline_buffer=ds,
                                 relabel_size=10000, shift_reward=1.0, device='cpu')
    tasks = evaluator.tasks
    assert len(tasks) == 5, tasks
    eval_z = []
    for t in tasks:
        z, rmet = evaluator._infer_z(t)
        z = z.detach().float().reshape(-1)
        eval_z.append(z)
        print(f"[fb-z] {t}: |z|={z.norm().item():.3f} "
              f"relabel_nonzero={rmet.get('relabel_reward#nonzero')}", flush=True)

    banks = np.load('/home/adityamohan/git/Factored-FB/analysis/atlas/banks.npz')
    eval_goals = banks['goals_eval']                          # [5, 28], task_id 1..5 order
    return _TorchFB(agent, eval_goals, eval_z)
