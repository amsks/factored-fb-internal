import numpy as np
import torch


def _tiny_model():
    import gymnasium
    from agents.fb.isfb.model import ISFBModel
    from nn_models import IdentityNNConfig, NoiseConditionedActorArchiConfig, SimpleVectorFieldArchiConfig
    from normalizers import IdentityNormalizerConfig
    return ISFBModel(
        obs_space=gymnasium.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
        action_dim=2, z_dim=8, L_dim=8,
        actor_cfg=NoiseConditionedActorArchiConfig(),
        actor_vf_cfg=SimpleVectorFieldArchiConfig(),
        obs_normalizer_cfg=IdentityNormalizerConfig(),
        rgb_encoder_cfg=IdentityNNConfig(),
        augmentator_cfg=IdentityNNConfig(),
        device="cpu",
    )


def test_value_forward_map_exists_and_shapes_match_forward():
    m = _tiny_model()
    B = 5
    # Derive the forward-map input exactly as update_fb does: left = left_encoder(fw_encoder(obs)).
    obs = torch.randn(B, 4)
    left = m._left_encoder(m._fw_encoder(obs))
    z = torch.randn(B, m.z_dim)
    a = torch.zeros(B, m.action_dim)
    fv = m._value_forward_map(left, z, a)
    f = m._forward_map(left, z, a)
    assert fv.shape == f.shape                      # [num_parallel, B, z_dim]
    assert fv.shape[-1] == m.z_dim


def test_prepare_for_train_builds_target_value_map():
    m = _tiny_model()
    m._prepare_for_train()
    assert hasattr(m, "_target_value_forward_map")
    # target starts as a copy of the online value map
    for p, tp in zip(m._value_forward_map.parameters(), m._target_value_forward_map.parameters()):
        assert torch.allclose(p, tp)
