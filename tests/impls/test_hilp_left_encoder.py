"""HILP left encoder: an observation embedding for the successor features, with its own target."""
import jax.numpy as jnp
import pytest
import yaml

from impls.critics.hilp import HILPCritic

EX_OBS = jnp.zeros((4, 28))
EX_ACT = jnp.zeros((4, 5))


def _cfg(**over):
    base = yaml.safe_load(open("impls/configs/critic/hilp_tdjc.yaml"))
    return {**base, "seed": 0, **over}


def _build(cfg):
    return HILPCritic.build(EX_OBS, EX_ACT, cfg)


def test_encoder_group_target_and_output_width():
    critic, params, targets = _build(_cfg(L_dim=50))
    assert "hilp_le" in critic.GROUPS
    assert "hilp_le" in params and "hilp_le" in targets
    assert critic._le(params, EX_OBS).shape == (4, 50)      # obs_dim -> L_dim


def test_encoder_target_rides_sf_tau_not_phi_tau():
    critic, _p, _t = _build(_cfg(sf_target_tau=0.005, phi_target_tau=0.5))
    assert critic.TARGETS["hilp_le"] == pytest.approx(0.005)
    assert critic.TARGETS["hilp_le"] != critic.TARGETS["hilp_phi1"]


def test_encoder_is_trained_by_the_sf_stage_only():
    critic, _p, _t = _build(_cfg())
    assert "hilp_le" in critic.stage_groups("sf")
    assert "hilp_le" not in critic.stage_groups("phi")


def test_sf_tower_reads_the_encoder_output():
    cfg = _cfg()
    critic, params, _t = _build(cfg)
    assert type(critic.hilp_sf).__name__ == "ForwardMap"
    enc = critic._le(params, EX_OBS)
    out = critic._sf(params["hilp_sf"], enc, jnp.zeros((4, cfg["z_dim"])), EX_ACT)
    assert out.shape[:2] == (2, 4)
