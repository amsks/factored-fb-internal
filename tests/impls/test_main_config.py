import jax.numpy as jnp
import pytest

from impls.utils.config_utils import load_config
from impls.critics.base import ComboError
import impls.main as m


def test_main_has_no_get_config():
    """get_config() is replaced by the config groups; a leftover would silently shadow them."""
    assert not hasattr(m, "get_config")


def test_fb_flowbc_real_yaml_composes():
    """The default launch pair still composes end-to-end from the real config groups."""
    cfg = load_config("fb", "flowbc", "cube_single")
    assert "env_name" in cfg
    assert cfg["actor"]


# --- minimal hand-rolled configs, one per (critic, actor) group, mirroring the pattern
# already used by test_pr2_actors.py / test_pr3_gciql.py / test_combo_validation.py. These
# sidestep real-yaml completeness gaps that are orthogonal to combo validity (e.g. gciql.yaml
# carries no z_dim because only FB-family critics use that concept, so gciql x flowbc needs
# z_dim supplied by hand here to reach check_combo instead of dying earlier on a missing key)
# so the test exercises exactly what it's meant to: does Agent.create (and therefore
# check_combo) accept or structurally reject each registered pair.

def _actor_block(actor_kind):
    if actor_kind in ("flowbc", "bcflow"):
        # bcflow is the same FlowBCActor with the value term off (q_coeff=0), so it needs
        # the same block; only q_coeff differs, exactly as the two yaml configs differ.
        return dict(type="flow", hidden_dim=32, hidden_layers=2, embedding_layers=2,
                    bc_coeff=3.0, q_coeff=0.0 if actor_kind == "bcflow" else 1.0,
                    flow_steps=5, lr_actor_vf=3e-4,
                    flow_vf_hidden_dim=32, flow_vf_hidden_layers=4)
    if actor_kind in ("ddpgbc", "awr"):
        return dict(hidden_dims=[32, 32], alpha=3.0, const_std=True, layer_norm=False)
    return dict(hidden_dims=[32, 32], alpha=10.0, flow_steps=5,        # fql
                normalize_q_loss=False, layer_norm=False)


def _min_cfg_for(critic_kind, actor_kind):
    if critic_kind == "fb":
        base = dict(
            z_dim=8, L_dim=8, num_parallel=2, discount=0.99,
            f_target_tau=0.005, b_target_tau=0.005, ortho_coef=1.0, train_goal_ratio=0.5,
            fb_pessimism_penalty=0.0, norm_z=True, lr_f=1e-4, lr_b=1e-4, lr_actor=1e-4,
            forward=dict(hidden_dim=32, hidden_layers=2, embedding_layers=2),
            backward=dict(hidden_dim=32, hidden_layers=4, norm=True),
            left_encoder=dict(hidden_dim=32, hidden_layers=4, norm=True),
        )
    elif critic_kind == "rldp":
        # fb's block plus the two rldp-only pieces: `horizon` and the `predictor` group.
        base = dict(
            z_dim=8, L_dim=8, num_parallel=2, discount=0.99,
            f_target_tau=0.005, b_target_tau=0.005, ortho_coef=0.01, train_goal_ratio=0.5,
            fb_pessimism_penalty=0.0, norm_z=True, lr_f=1e-4, lr_b=1e-4, lr_actor=1e-4,
            horizon=4,
            forward=dict(hidden_dim=32, hidden_layers=2, embedding_layers=2),
            backward=dict(hidden_dim=32, hidden_layers=4, norm=True),
            left_encoder=dict(hidden_dim=32, hidden_layers=4, norm=True),
            predictor=dict(hidden_dim=32, hidden_layers=2, embedding_layers=2),
        )
    elif critic_kind == "isfb":
        base = dict(
            z_dim=8, L_dim=8, num_parallel=2, discount=0.99,
            f_target_tau=0.005, b_target_tau=0.005, ortho_coef=1.0, train_goal_ratio=0.5,
            fb_pessimism_penalty=0.0, norm_z=True, lr_f=1e-4, lr_b=1e-4, lr_actor=1e-4,
            forward=dict(hidden_dim=32, hidden_layers=2, embedding_layers=2),
            backward=dict(hidden_dim=32, hidden_layers=4, norm=True),
            left_encoder=dict(hidden_dim=32, hidden_layers=4, norm=True),
            measure_insample="scalar_gate_hard", iql_expectile=0.9,
            alpha_n=1.0, alpha_m=1.0, alpha_v=1.0, awr_beta=3.0, awr_wmax=100.0,
            awr_normalize_adv=True, goal_mask=True,
        )
    elif critic_kind == "iql_bilinear":
        base = dict(
            value_hidden_dims=[64, 64], layer_norm=True, discount=0.99, expectile=0.9, tau=0.005,
            lr=3e-4, lr_actor=3e-4, gc_negative=True,
            value_latent_dim=8, psi_base="learned", project_z=False, double_q=True,
            psi_layer_norm=True,
            iql_ortho_coef=0.0, actor_cond="goal", actor_z_from_critic=False,
            actor_z_infer=False,
            z_dim=8,  # only read if paired with an actor (flowbc) that needs a latent width
        )
    elif critic_kind == "hiql":
        # HIQL owns goal_rep + the IVL value head + the HIGH-level policy, so it needs the
        # critic-side actor width too. Without this branch Agent.create died with a bare
        # KeyError('rep_dim') from impls/critics/hiql.py, which is exactly the failure mode
        # this test exists to forbid.
        base = dict(
            value_hidden_dims=[64, 64], layer_norm=True, discount=0.99, expectile=0.7,
            tau=0.005, lr=3e-4, lr_actor=3e-4, gc_negative=True,
            high_alpha=3.0, rep_dim=10, subgoal_steps=25,
            high_actor_hidden_dims=[64, 64], high_const_std=True, low_actor_rep_grad=False,
        )
    elif critic_kind == "simple_sf":
        base = dict(
            value_hidden_dims=[64, 64], layer_norm=True, discount=0.99, expectile=0.9, tau=0.005,
            lr=3e-4, lr_actor=3e-4, gc_negative=True,
            value_latent_dim=8, ground_psi=False, reward_coef=0.0, project_z=False,
            double_q=True, actor_cond="goal", ground_fade=False,
            z_dim=8,  # only read if paired with an actor (flowbc) that needs a latent width
        )
    elif critic_kind == "sf_feature":
        base = dict(
            value_hidden_dims=[64, 64], layer_norm=True, discount=0.99, tau=0.005,
            lr=3e-4, lr_actor=3e-4, gc_negative=True,
            value_latent_dim=16, sf_base="frozen", sf_mask="goal", sf_base_index="current",
            sf_joint_psi=False, sf_reward_regress=False, sf_reward_coef=1.0, sf_reward_shift=0.0,
            sf_ortho_coef=0.0, sf_psi_ortho_coef=0.0, sf_coef=1.0, fb_forward_z=False,
            actor_cond="goal",
            z_dim=8,  # only read if paired with an actor (flowbc) that needs a latent width
        )
    elif critic_kind == "centered_bilinear":
        base = dict(
            value_hidden_dims=[64, 64], layer_norm=True, discount=0.99, expectile=0.9, tau=0.005,
            lr=3e-4, lr_actor=3e-4, gc_negative=True,
            value_latent_dim=8, double_q=True, mode="dueling",
            dueling_b_coef=1.0, centered_measure_coef=1.0, center_b_coef=1.0, center_b_samples=1,
            centered_split_bootstrap=True, centered_scale_anchors=False, use_next_actions=False,
            z_dim=8,  # only read if paired with an actor (flowbc) that needs a latent width
        )
    elif critic_kind == "fb_forward_bilinear":
        base = dict(
            value_hidden_dims=[64, 64], layer_norm=True, discount=0.99, expectile=0.9, tau=0.005,
            lr=3e-4, lr_actor=3e-4, gc_negative=True,
            value_latent_dim=8, phi_module="flat", objective="iql", project_z=False, double_q=True,
            measure_coef=0.0, measure_legs="critic", measure_scale_norm=False,
            measure_stopgrad_phi=False, measure_ortho_coef=0.0, measure_test_states="rho",
            measure_test_rank="full", iql_ortho_coef=0.0,
            measure_insample="scalar_gate_hard", alpha_n=1.0, alpha_m=1.0, alpha_v=1.0,
            awr_beta=3.0, awr_wmax=100.0, awr_normalize_adv=True, fb_pessimism_penalty=0.0,
            goal_mask=True,
            actor_cond="goal", actor_z_from_critic=False, actor_z_infer=False,
            z_dim=8,  # only read if paired with an actor (flowbc) that needs a latent width
        )
    elif critic_kind == "density_fb":
        base = dict(
            value_hidden_dims=[32, 32], layer_norm=True, discount=0.99, tau=0.005,
            lr=3e-4, lr_actor=3e-4,
            measure_hit_spec="0:2:0.4", measure_density_dim=8, measure_density_temp=1.0,
            actor_q_center=True,
            z_dim=8,  # only read if paired with an actor (flowbc) that needs a latent width
        )
    elif critic_kind == "td_fb":
        base = dict(
            value_hidden_dims=[32, 32], layer_norm=True, value_latent_dim=8,
            discount=0.99, tau=0.005, lr=3e-4, lr_actor=3e-4,
            measure_hit_spec="0:2:0.4", measure_scale_norm=True,
            measure_first_passage=True, measure_fp_absorbing=True,
            measure_stopgrad_psi=False, measure_testpoint_dist="hindsight",
            measure_ortho_coef=0.0, actor_q_center=False, warm_start=False,
            z_dim=8,  # only read if paired with an actor (flowbc) that needs a latent width
        )
    elif critic_kind == "psm":
        base = dict(
            z_dim=8, max_log_seed=4, num_parallel=2, discount=0.99, ortho_coef=1.0,
            pessimism_penalty=0.0, actor_pessimism_penalty=0.5, mix_ratio=0.5,
            norm_z=True, target_tau=0.01, proto_seed=0,
            lr_phi=1e-4, lr_sf=1e-4, lr_actor=1e-4,
            phi=dict(hidden_dim=16, hidden_layers=2, norm=True),
            sf=dict(hidden_dim=32, hidden_layers=1, embedding_layers=2),
        )
    elif critic_kind in ("crl", "vip"):
        base = dict(
            value_hidden_dims=[64, 64], layer_norm=True, discount=0.99, tau=0.005,
            lr=3e-4, lr_actor=3e-4, gc_negative=True, latent_dim=8,
            z_dim=8,  # only read if paired with an actor (flowbc) that needs a latent width
        )
    elif critic_kind == "qrl":
        base = dict(
            value_hidden_dims=[64, 64], layer_norm=True, discount=0.99, tau=0.005,
            lr=3e-4, lr_actor=3e-4, gc_negative=True,
            qrl_components=4, qrl_dim_per_component=8, qrl_step_cost=1.0,
            qrl_eps=0.25, qrl_transform_t=100.0,
            z_dim=8,  # only read if paired with an actor (flowbc) that needs a latent width
        )
    else:  # gciql
        base = dict(
            value_hidden_dims=[64, 64], layer_norm=True, discount=0.99, expectile=0.9, tau=0.005,
            lr=3e-4, lr_actor=3e-4, gc_negative=True, q_bound=0.0,
            z_dim=8,  # only read if paired with an actor (flowbc) that needs a latent width
        )
    base["actor"] = _actor_block(actor_kind)
    return base


def test_every_registered_pair_composes():
    """Every registered (critic, actor) pair either builds a working Agent, or is a
    structurally invalid combo that check_combo rejects with a named ComboError — never
    a bare KeyError/AttributeError deep inside construction."""
    from impls.agents import CRITICS, ACTORS, Agent

    ex_obs, ex_act = jnp.zeros((2, 8)), jnp.zeros((2, 5))
    built, skipped = [], []
    for c in CRITICS:
        for a in ACTORS:
            cfg = _min_cfg_for(c, a)
            try:
                agent = Agent.create(c, a, 0, ex_obs, ex_act, cfg)
            except ComboError:
                skipped.append((c, a))  # structurally invalid pair — expected, not a failure
                continue
            assert agent is not None
            built.append((c, a))

    # FlowBCActor now conditions on the critic's cond_vec (z for FB/ISFB, goal otherwise) and
    # AWRActor falls back to a policy-baseline advantage when the critic has no V, so EVERY
    # registered pair is legitimate.
    assert skipped == [], f"no pair should be rejected any more, but these were: {skipped}"
    # Derived from the registries so that adding an actor/critic does not require editing a
    # magic number, but floored at the 32 pairs that existed when this gate was written so
    # that *removing* one still fails loudly.
    assert len(built) == len(CRITICS) * len(ACTORS)
    assert len(built) >= 32


def test_gciql_flowbc_builds():
    """gciql×flowbc is now legitimate: FlowBCActor conditions on GCIQLCritic.cond_vec (the
    goal), so it builds a goal-conditioned flow-BC actor rather than raising ComboError."""
    from impls.agents import Agent

    ex_obs, ex_act = jnp.zeros((2, 8)), jnp.zeros((2, 5))
    cfg = _min_cfg_for("gciql", "flowbc")
    agent = Agent.create("gciql", "flowbc", 0, ex_obs, ex_act, cfg)
    assert agent is not None
