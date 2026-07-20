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


# --- FINDING 1 regression: REQUIRED_KEYS must actually cover everything Agent.create /
# make_dataset(GCDataset) read off the config, so a missing key surfaces here as a named
# ConfigError rather than a bare KeyError deep inside them.

def test_fb_agent_create_key_error_reproduction():
    """Before the fix, `~L_dim` composed fine and only blew up as a bare KeyError inside
    Agent.create's FBCritic.build(). REQUIRED_KEYS must catch it at load_config time."""
    with pytest.raises(ConfigError) as e:
        load_config("fb", "flowbc", "cube_single", overrides=["~L_dim"])
    assert "L_dim" in str(e.value)


@pytest.mark.parametrize("key", REQUIRED_KEYS["fb"])
def test_fb_required_key_missing_raises_named_error(key):
    with pytest.raises(ConfigError) as e:
        load_config("fb", "flowbc", "cube_single", overrides=[f"~{key}"])
    assert key in str(e.value)


@pytest.mark.parametrize("key", REQUIRED_KEYS["gciql"])
def test_gciql_required_key_missing_raises_named_error(key):
    with pytest.raises(ConfigError) as e:
        load_config("gciql", "fql", "cube_single", overrides=[f"~{key}"])
    assert key in str(e.value)


# --- FINDING 2 regression: `~key` overrides must actually remove the key, not leave it
# present with value None (OmegaConf.merge with a bare `key:` dotlist would do the latter).

def test_top_level_delete_override_removes_key():
    # obs_type is a real, present, top-level key that no REQUIRED_KEYS list needs, so this
    # isolates the delete mechanism itself from the required-key gate exercised above.
    cfg = load_config("fb", "flowbc", "cube_single", overrides=["~obs_type"])
    assert "obs_type" not in cfg
    # sibling top-level keys must survive the delete.
    assert "env_name" in cfg


def test_nested_delete_override_removes_key():
    cfg = load_config("gciql", "fql", "cube_single", overrides=["~actor.alpha"])
    assert "alpha" not in cfg["actor"]
    # sibling keys in the same nested dict must survive the delete.
    assert "hidden_dims" in cfg["actor"]
