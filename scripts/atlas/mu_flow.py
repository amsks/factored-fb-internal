"""The frozen behavioral reference flow mu(a|s): actor_vf integrated from noise.

Verified goal-independent (actor_vf takes obs,x_t,t -- no goal). mu provides both the
centering E_mu[Q] and the operational behavioral-reference region for the atlas. NOT the
flow actor's Q-guided head -- purely the BC velocity field.
MUST run under /dev/shm/.venv-jax/bin/python.
"""
import numpy as np

from scripts.atlas.jax_load import load_agent


class MuFlow:
    def __init__(self, agent, flow_steps=10):
        self.agent = agent
        self.fs = flow_steps

    def sample(self, obs, n, seed):
        """Integrate the BC velocity field from noise -> action.
        obs [S,28] -> actions [S, n, 5], deterministic given seed."""
        import jax
        import jax.numpy as jnp
        obs = jnp.asarray(obs)
        S = obs.shape[0]
        a = jax.random.normal(jax.random.PRNGKey(seed), (S, n, 5))
        obs_rep = jnp.repeat(obs[:, None], n, axis=1).reshape(-1, 28)
        for i in range(self.fs):
            t = jnp.ones((S * n, 1)) * (i / self.fs)
            v = self.agent.network.select('actor_vf')(obs_rep, a.reshape(-1, 5), t)
            a = a + v.reshape(S, n, 5) / self.fs
        return np.clip(np.asarray(a), -1, 1)


def load_mu(ckpt_dir, step):
    return MuFlow(load_agent(ckpt_dir, step))


def assert_q_independent(ckpt_dir, step):
    """actor_vf must be trainable only by bc_flow_loss (the -q.mean() term flows through
    the one-shot 'actor' head, and the distillation target is stop_gradient'd). Structural
    proxy: both an 'actor_vf' subtree and a 'critic'/'value' subtree exist and are distinct
    parameter groups. Mirrors the fb_grad_decomp.py disjointness finding for the torch stack.
    """
    import jax
    agent = load_agent(ckpt_dir, step)
    leaves = jax.tree_util.tree_leaves_with_path(agent.network.params)
    names = {'/'.join(str(k) for k in path) for path, _ in leaves}
    has_vf = any('actor_vf' in n for n in names)
    has_critic = any(('critic' in n) or ('value' in n) for n in names)
    assert has_vf, "no actor_vf parameter subtree found"
    assert has_critic, "no critic/value parameter subtree found"
    return True
