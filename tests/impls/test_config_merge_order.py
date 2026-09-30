"""Pin the config merge order: load_config merges (domain, critic, actor) with the actor
winning ties, so lr_actor is owned by the actor yaml for every pair and swapping the critic
never changes the actor's learning rate."""
import pytest
from omegaconf import OmegaConf

from impls.utils.config_utils import CONFIG_ROOT, load_config

CRITICS = ("fb", "density_fb", "hilp_tdjc", "rldp")
ACTORS = ("flowbc", "ddpgbc")


def _actor_yaml_lr(actor):
    return OmegaConf.load(f"{CONFIG_ROOT}/actor/{actor}.yaml")["lr_actor"]


@pytest.mark.parametrize("critic", CRITICS)
@pytest.mark.parametrize("actor", ACTORS)
def test_actor_yaml_owns_lr_actor(critic, actor):
    cfg = load_config(critic, actor, "cube_single")
    assert cfg["lr_actor"] == pytest.approx(_actor_yaml_lr(actor))


def test_critic_lr_does_not_leak_into_actor_lr():
    """fb.yaml sets lr_f/lr_b; density_fb.yaml sets lr. Neither may be mistaken for the other."""
    fb = load_config("fb", "flowbc", "cube_single")
    dfb = load_config("density_fb", "flowbc", "cube_single")
    assert fb["lr_f"] == pytest.approx(1.0e-4)
    assert fb["lr_b"] == pytest.approx(1.0e-4)
    assert dfb["lr"] == pytest.approx(3.0e-4)
    assert "lr" not in fb
