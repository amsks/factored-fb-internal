import jax.numpy as jnp
import numpy as np
from impls.agents import Agent
import impls.main as m


def _fb_cfg():
    return dict(batch_size=8, z_dim=50, L_dim=50, num_parallel=2, discount=0.99,
                f_target_tau=0.005, b_target_tau=0.005, ortho_coef=1.0,
                train_goal_ratio=0.5, fb_pessimism_penalty=0.0, norm_z=True,
                lr_f=1e-4, lr_b=1e-4, lr_actor=1e-4,
                forward=dict(hidden_dim=32, hidden_layers=2, embedding_layers=2),
                backward=dict(hidden_dim=32, hidden_layers=2, norm=True),
                left_encoder=dict(hidden_dim=32, hidden_layers=2, norm=True),
                actor=dict(type="flow", hidden_dim=32, hidden_layers=2, embedding_layers=2,
                           bc_coeff=3.0, flow_steps=3, lr_actor_vf=3e-4,
                           flow_vf_hidden_dim=32, flow_vf_hidden_layers=2))


def test_infer_eval_cond_changes_z():
    agent = Agent.create("fb", "flowbc", 0, jnp.zeros((2, 8)), jnp.zeros((2, 5)), _fb_cfg())
    before = np.asarray(agent.cond_eval["z"])
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((32, 8)).astype(np.float32)
    rew = rng.standard_normal((32, 1)).astype(np.float32)
    after = np.asarray(agent.infer_eval_cond(obs, rew).cond_eval["z"])
    assert not np.allclose(before, after), "infer_eval_cond did not change cond_eval"


def test_main_exposes_infer_cond_flag():
    from absl import flags
    assert "infer_cond_episodes" in flags.FLAGS
