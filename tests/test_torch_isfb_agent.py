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
