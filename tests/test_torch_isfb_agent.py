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
