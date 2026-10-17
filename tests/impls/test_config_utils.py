import pytest
from impls.utils.config_utils import load_config, ConfigError, REQUIRED_KEYS


def test_load_fb_flowbc_matches_legacy_values():
    cfg = load_config("fb", "flowbc", "cube_single")
    assert cfg["z_dim"] == 50
    assert cfg["batch_size"] == 256
    assert cfg["lr_f"] == pytest.approx(1e-4)
    assert cfg["actor"]["bc_coeff"] == pytest.approx(3.0)
    assert cfg["actor"]["flow_steps"] == 10
    assert cfg["env_name"] == "cube-single-play-v0"


def test_load_density_fb_has_lr_and_dataset_keys():
    cfg = load_config("density_fb", "flowbc", "cube_single")
    assert cfg["lr"] == pytest.approx(3e-4)          # key read by _LR_KEY
    assert cfg["dataset_class"] == "GCDataset"
    assert cfg["value_p_trajgoal"] == pytest.approx(0.5)
    assert cfg["actor"]["flow_steps"] == 10


def test_overrides_apply():
    cfg = load_config("fb", "ddpgbc", "cube_single",
                      overrides=["batch_size=64", "actor.q_noise_std=1.5"])
    assert cfg["batch_size"] == 64
    assert cfg["actor"]["q_noise_std"] == pytest.approx(1.5)


def test_missing_required_key_raises_named_error():
    with pytest.raises(ConfigError) as e:
        load_config("rldp", "flowbc", "cube_single", overrides=["~horizon"])
    assert "horizon" in str(e.value)


def test_returns_plain_dict_not_dictconfig():
    cfg = load_config("fb", "flowbc", "cube_single")
    assert type(cfg) is dict
    assert type(cfg["actor"]) is dict


# REQUIRED_KEYS covers what Agent.create and make_dataset read.

def test_fb_agent_create_key_error_reproduction():
    """Removing L_dim fails in load_config, not inside FBCritic.build()."""
    with pytest.raises(ConfigError) as e:
        load_config("fb", "flowbc", "cube_single", overrides=["~L_dim"])
    assert "L_dim" in str(e.value)


@pytest.mark.parametrize("key", REQUIRED_KEYS["fb"])
def test_fb_required_key_missing_raises_named_error(key):
    with pytest.raises(ConfigError) as e:
        load_config("fb", "flowbc", "cube_single", overrides=[f"~{key}"])
    assert key in str(e.value)


@pytest.mark.parametrize("key", REQUIRED_KEYS["rldp"])
def test_rldp_required_key_missing_raises_named_error(key):
    with pytest.raises(ConfigError) as e:
        load_config("rldp", "flowbc", "cube_single", overrides=[f"~{key}"])
    assert key in str(e.value)


# `~key` overrides must remove the key, not leave it present with value None.

def test_top_level_delete_override_removes_key():
    # A synthetic key, unrelated to REQUIRED_KEYS.
    cfg = load_config("fb", "flowbc", "cube_single",
                      overrides=["scratch_probe=1", "~scratch_probe"])
    assert "scratch_probe" not in cfg
    # Sibling top-level keys survive.
    assert "env_name" in cfg


def test_nested_delete_override_removes_key():
    cfg = load_config("fb", "ddpgbc", "cube_single", overrides=["~actor.q_noise_std"])
    assert "q_noise_std" not in cfg["actor"]
    # Sibling nested keys survive.
    assert "hidden_dims" in cfg["actor"]
