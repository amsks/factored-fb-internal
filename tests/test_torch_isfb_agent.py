import numpy as np
import torch


def _tiny_isfb(**extra):
    import gymnasium
    from agents.fb.isfb.agent import ISFBAgent
    from nn_models import IdentityNNConfig, NoiseConditionedActorArchiConfig, SimpleVectorFieldArchiConfig
    from normalizers import IdentityNormalizerConfig
    return ISFBAgent(
        obs_space=gymnasium.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
        action_dim=2, z_dim=8, L_dim=8, batch_size=16,
        actor_cfg=NoiseConditionedActorArchiConfig(),
        actor_vf_cfg=SimpleVectorFieldArchiConfig(),
        obs_normalizer_cfg=IdentityNormalizerConfig(),
        rgb_encoder_cfg=IdentityNNConfig(),
        augmentator_cfg=IdentityNNConfig(),
        device="cpu", **extra,
    )


def test_defaults_and_model_type():
    from agents.fb.isfb.model import ISFBModel
    a = _tiny_isfb()
    assert isinstance(a.model, ISFBModel)
    assert a.measure_insample == "scalar_gate_hard"
    assert a.alpha_n == 1.0 and a.alpha_m == 1.0
    assert a.critic_mode == "measure"          # ISFB uses the measure-branch update() path


def test_value_forward_params_in_forward_optimizer():
    a = _tiny_isfb()
    opt_params = {id(p) for g in a.forward_optimizer.param_groups for p in g["params"]}
    fv_params = {id(p) for p in a.model._value_forward_map.parameters()}
    assert fv_params.issubset(opt_params)
    assert len(a._target_value_forward_map_paramlist) == len(a._value_forward_map_paramlist) > 0


def _batch(B=16):
    return {
        "observation": torch.randn(B, 4),
        "action": torch.rand(B, 2) * 2 - 1,
        "next": {"observation": torch.randn(B, 4), "terminated": torch.zeros(B, 1)},
    }


def _run(a, B=16, steps=1):
    m = None
    for s in range(steps):
        m = a.update(_batch(B), step=s)
    return m


def test_update_reports_isfb_losses_all_variants():
    for mode in ("coord", "scalar_gate_hard", "scalar_gate_awr"):
        a = _tiny_isfb(measure_insample=mode)
        m = _run(a)
        for k in ("L_N", "L_M", "orth_loss", "actor_loss"):
            assert k in m and np.isfinite(m[k]), (mode, k)
        if mode != "coord":
            assert "L_V" in m and np.isfinite(m["L_V"])


def test_coord_variant_has_no_scalar_value_loss():
    a = _tiny_isfb(measure_insample="coord")
    m = _run(a)
    assert m.get("L_V", 0.0) == 0.0            # coord omits L_V (isfb.tex Eq. coord_total)


def test_bootstrap_uses_target_fv_not_policy_action():
    # The SG-ISFB measure TD bootstraps through N^- = <target_F_V(s'), B>, NOT through a
    # policy next-action. Poison the policy-action sampler; the critic update must not touch it.
    a = _tiny_isfb(measure_insample="scalar_gate_hard")

    def _boom(*args, **kwargs):
        raise AssertionError("critic update sampled a policy action (a'~pi)")

    a.sample_action_from_norm_obs = _boom      # used only by the standard-FB measure bootstrap
    m = _run(a)                                # must succeed => no a'~pi in ISFB critic
    assert np.isfinite(m["L_M"])


def test_hard_weights_split_by_advantage_sign():
    # tau-hard weights: w = tau for A>0, (1-tau) for A<=0. Verify the helper directly.
    from agents.fb.isfb.agent import isfb_hard_weights
    A = torch.tensor([1.0, -1.0, 2.0, -0.5])
    tau = 0.9
    w = isfb_hard_weights(A, tau)
    assert torch.allclose(w, torch.tensor([tau, 1 - tau, tau, 1 - tau]))


def test_value_forward_params_change_after_update():
    a = _tiny_isfb(measure_insample="scalar_gate_hard")
    before = [p.detach().clone() for p in a.model._value_forward_map.parameters()]
    _run(a, steps=3)
    after = list(a.model._value_forward_map.parameters())
    assert any(not torch.allclose(b, x) for b, x in zip(before, after))


# --- review fixes: goal-success mask, adv logging, asserts -------------------------

def _batch_traj(B=16):
    """traj-mode batch: adds future + physics (cube xyz at [14:17]) the goal mask needs."""
    nphys = torch.zeros(B, 20)
    fphys = torch.zeros(B, 20)
    fphys[B // 2:, 14] = 1.0   # first half reached (dist 0), second half far (dist 1)
    return {
        "observation": torch.randn(B, 4),
        "action": torch.rand(B, 2) * 2 - 1,
        "next": {"observation": torch.randn(B, 4), "terminated": torch.zeros(B, 1), "physics": nphys},
        "future": {"observation": torch.randn(B, 4), "physics": fphys},
    }


def test_goal_success_mask_helper():
    from agents.fb.isfb.agent import isfb_goal_success
    nphys = torch.zeros(4, 20)
    fphys = torch.zeros(4, 20)
    fphys[2:, 14] = 1.0                          # rows 0,1 reached (dist 0); rows 2,3 far (dist 1)
    s = isfb_goal_success(nphys, fphys, thresh=0.04)
    assert torch.allclose(s, torch.tensor([1., 1., 0., 0.]))


def test_goal_mask_update_runs_and_reports():
    a = _tiny_isfb(measure_insample="scalar_gate_hard", isfb_goal_mask=True, train_goal_mode="traj")
    m = a.update(_batch_traj(), step=0)
    for k in ("L_M", "adv_std", "adv_min", "adv_max"):
        assert k in m and np.isfinite(m[k]), k


def test_update_reports_adv_stats():
    a = _tiny_isfb()
    m = _run(a)
    for k in ("adv_std", "adv_min", "adv_max"):
        assert k in m and np.isfinite(m[k])


def test_asserts_block_ignored_fb_options():
    # ISFB ignores reweighting / goal_cond / onestep bootstrap; passing them must raise
    # (prevents silent mixed experiments). Asserts sit at the top of update_fb.
    import pytest
    a = _tiny_isfb()
    B = 16
    kw = dict(obs=torch.randn(B, 4), action=torch.rand(B, 2), discount=torch.ones(B, 1),
              next_obs=torch.randn(B, 4), goal=torch.randn(B, 4), z=torch.randn(B, 8),
              q_loss_coef=None, clip_grad_norm=None)
    with pytest.raises(AssertionError):
        a.update_fb(w=torch.ones(B), **kw)
    with pytest.raises(AssertionError):
        a.update_fb(goal_obs=torch.randn(B, 4), **kw)
    with pytest.raises(AssertionError):
        a.update_fb(next_action_override=torch.rand(B, 2), **kw)
