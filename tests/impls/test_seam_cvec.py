import jax.numpy as jnp
from impls.critics.fb import FBCritic


def _fb_cfg():
    return dict(z_dim=8, L_dim=8, num_parallel=2, discount=0.99, seed=0,
                f_target_tau=0.005, b_target_tau=0.005, ortho_coef=1.0, train_goal_ratio=0.5,
                fb_pessimism_penalty=0.0, norm_z=True,
                forward=dict(hidden_dim=32, hidden_layers=2, embedding_layers=2),
                backward=dict(hidden_dim=32, hidden_layers=4, norm=True),
                left_encoder=dict(hidden_dim=32, hidden_layers=4, norm=True))


def test_fb_cvec_dim_and_eval_cond0():
    critic, _, _ = FBCritic.build(jnp.zeros((2, 8)), jnp.zeros((2, 5)), _fb_cfg())
    assert critic.cvec_dim == 8
    c0 = critic.eval_cond0(jnp.zeros((2, 8)), _fb_cfg())
    assert set(c0.keys()) == {"z"} and c0["z"].shape == (8,)
