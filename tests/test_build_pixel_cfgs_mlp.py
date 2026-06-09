"""train.build_pixel_cfgs must support an MLP state encoder (name == 'mlp')
without changing the existing identity/drq behavior."""
from omegaconf import OmegaConf

from train import build_pixel_cfgs
from nn_models import MLPEncoderArchiConfig, DrQEncoderArchiConfig, IdentityNNConfig


def test_mlp_branch_returns_mlp_encoder_cfg():
    cfg = OmegaConf.create({
        "rgb_encoder": {"name": "mlp", "feature_dim": 256, "hidden_dim": 1100, "hidden_layers": 5},
    })
    _norm, rgb, _aug = build_pixel_cfgs(cfg)
    assert isinstance(rgb, MLPEncoderArchiConfig)
    assert rgb.feature_dim == 256
    assert rgb.hidden_dim == 1100
    assert rgb.hidden_layers == 5


def test_no_block_is_identity_regression():
    _norm, rgb, _aug = build_pixel_cfgs(OmegaConf.create({}))
    assert isinstance(rgb, IdentityNNConfig)


def test_drq_branch_unchanged_regression():
    cfg = OmegaConf.create({"rgb_encoder": {"name": "drq", "feature_dim": 256}})
    _norm, rgb, _aug = build_pixel_cfgs(cfg)
    assert isinstance(rgb, DrQEncoderArchiConfig)
    assert rgb.feature_dim == 256
