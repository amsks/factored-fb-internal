from impls.utils.config_utils import load_config
import impls.main as m


def test_main_has_no_get_config():
    """get_config() is replaced by the config groups; a leftover would silently shadow them."""
    assert not hasattr(m, "get_config")


def test_every_registered_pair_composes():
    from impls.agents import CRITICS, ACTORS
    from impls.critics.base import check_combo
    for c in CRITICS:
        for a in ACTORS:
            cfg = load_config(c, a, "cube_single")
            assert "env_name" in cfg
            assert cfg["actor"], f"{c}x{a} has no actor block"
