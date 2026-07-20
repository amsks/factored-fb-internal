import pytest
from impls.utils.config_utils import load_config, ConfigError


def test_load_fb_flowbc_matches_legacy_values():
    cfg = load_config("fb", "flowbc", "cube_single")
    assert cfg["z_dim"] == 50
    assert cfg["batch_size"] == 256
    assert cfg["lr_f"] == pytest.approx(1e-4)
    assert cfg["actor"]["bc_coeff"] == pytest.approx(3.0)
    assert cfg["actor"]["flow_steps"] == 10
    assert cfg["env_name"] == "cube-single-play-v0"


def test_load_gciql_fql_has_lr_and_dataset_keys():
    cfg = load_config("gciql", "fql", "cube_single")
    assert cfg["lr"] == pytest.approx(3e-4)          # the key _LR_KEY needs
    assert cfg["expectile"] == pytest.approx(0.9)
    assert cfg["tau"] == pytest.approx(0.005)
    assert cfg["dataset_class"] == "GCDataset"
    assert cfg["value_p_trajgoal"] == pytest.approx(0.5)
    assert cfg["actor"]["flow_steps"] == 10


def test_overrides_apply():
    cfg = load_config("gciql", "fql", "cube_single", overrides=["batch_size=64", "actor.alpha=1.5"])
    assert cfg["batch_size"] == 64
    assert cfg["actor"]["alpha"] == pytest.approx(1.5)


def test_missing_required_key_raises_named_error():
    with pytest.raises(ConfigError) as e:
        load_config("gciql", "fql", "cube_single", overrides=["~lr"])
    assert "lr" in str(e.value)


def test_returns_plain_dict_not_dictconfig():
    cfg = load_config("fb", "flowbc", "cube_single")
    assert type(cfg) is dict
    assert type(cfg["actor"]) is dict
