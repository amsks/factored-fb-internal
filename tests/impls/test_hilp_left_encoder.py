"""The td_jepa left encoder + SF tower family knobs on HILPCritic.

Reference: td_jepa `metamotivo/agents/sf/{model,agent}.py`. The SF net is built on
`self._left_encoder.output_space` (model.py:82), so it always consumes the ENCODED
observation; `actor_encode_obs` gates only the actor (:84, :117) and is False on OGBench.
The encoder sits in the SF optimizer's parameter list at lr_sf (agent.py:91-92) and its
target polyaks at sf_target_tau (:266-267).

The default-OFF path is what every pre-existing HILP arm runs, so it is asserted to be
structurally untouched here as well as by the boot-contract fixture.
"""
import jax
import jax.numpy as jnp
import pytest
import yaml

from impls.critics.hilp import HILPCritic

EX_OBS = jnp.zeros((4, 28))
EX_ACT = jnp.zeros((4, 5))


def _cfg(**over):
    base = yaml.safe_load(open("impls/configs/critic/hilp_cube.yaml"))
    return {**base, "seed": 0, **over}


def _build(cfg):
    return HILPCritic.build(EX_OBS, EX_ACT, cfg)


def _nparams(tree):
    return sum(x.size for x in jax.tree_util.tree_leaves(tree))


def test_off_by_default_no_group_no_target_no_encoder():
    """A config that predates the knob must produce exactly the old param set."""
    critic, params, targets = _build(_cfg())
    assert critic.le_on is False
    assert critic.GROUPS == ("hilp_phi1", "hilp_phi2", "hilp_sf", "hilp_stats")
    assert "hilp_le" not in params and "hilp_le" not in targets
    assert critic.stage_groups("sf") == ("hilp_sf",)


def test_off_le_is_the_identity():
    """With the knob off `_le` must not touch the observation -- and must not index a
    missing key, which is what would break every existing arm."""
    critic, params, _t = _build(_cfg())
    out = critic._le(params, EX_OBS)
    assert out.shape == EX_OBS.shape
    assert jnp.array_equal(out, EX_OBS)


def test_on_adds_group_target_and_maps_to_L_dim():
    cfg = _cfg(L_dim=50, hilp_left_encoder={"hidden_dim": 512, "hidden_layers": 4,
                                            "norm": True})
    critic, params, targets = _build(cfg)
    assert critic.le_on is True
    assert critic.GROUPS[-1] == "hilp_le"
    assert "hilp_le" in params and "hilp_le" in targets
    assert critic._le(params, EX_OBS).shape == (4, 50)      # obs_dim -> L_dim


def test_encoder_target_rides_sf_tau_not_phi_tau():
    """sf/agent.py:266-267 soft-updates the left encoder with cfg.train.sf_target_tau."""
    cfg = _cfg(L_dim=50, sf_target_tau=0.005, phi_target_tau=0.5,
               hilp_left_encoder={"hidden_dim": 64, "hidden_layers": 2, "norm": True})
    critic, _p, _t = _build(cfg)
    assert critic.TARGETS["hilp_le"] == pytest.approx(0.005)
    assert critic.TARGETS["hilp_le"] != critic.TARGETS["hilp_phi1"]


def test_encoder_is_trained_by_the_sf_stage_only():
    """It is in sf_optimizer's param list, so the SF stage steps it and the phi stage
    must not -- the phi stage's isolation is what makes a density pair comparable."""
    cfg = _cfg(L_dim=50, hilp_left_encoder={"hidden_dim": 64, "hidden_layers": 2})
    critic, _p, _t = _build(cfg)
    assert "hilp_le" in critic.stage_groups("sf")
    assert "hilp_le" not in critic.stage_groups("phi")


def test_sf_archi_tdjepa_is_a_different_tower_than_official_hilp():
    """`tdjepa` selects the generic ForwardArchi (the seam's ForwardMap, which FB uses),
    not official HILP's shared-embedding ForwardMap."""
    common = dict(sf={"hidden_dim": 512, "hidden_layers": 2, "embedding_layers": 2})
    hilp_c, hilp_p, _ = _build(_cfg(hilp_sf_archi="hilp", **common))
    tdj_c, tdj_p, _ = _build(_cfg(hilp_sf_archi="tdjepa", **common))
    assert type(hilp_c.hilp_sf).__name__ != type(tdj_c.hilp_sf).__name__
    assert _nparams(hilp_p["hilp_sf"]) != _nparams(tdj_p["hilp_sf"])


def test_sf_archi_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="hilp_sf_archi"):
        _build(_cfg(hilp_sf_archi="nope"))


def test_shipped_tdjc_yaml_turns_both_knobs_on():
    """Guard the config itself: the whole point of hilp_tdjc is these two being live."""
    cfg = yaml.safe_load(open("impls/configs/critic/hilp_tdjc.yaml"))
    cfg["seed"] = 0
    critic, params, _t = _build(cfg)
    assert critic.le_on is True, "hilp_tdjc must carry td_jepa's left encoder"
    assert type(critic.hilp_sf).__name__ == "ForwardMap"
    assert critic.TARGETS["hilp_le"] == pytest.approx(0.005)
    # and the calibration deltas that motivated the file
    assert cfg["batch_size"] == 256
    assert cfg["sf_target_tau"] == pytest.approx(0.005)
    assert cfg["group_lr"]["hilp_phi1"] == pytest.approx(1.0e-4)
    assert cfg["phi"]["hidden_layers"] == 4
    assert cfg["hilp_independent_target_init"] is False


def test_density_tdjc_is_the_substrate_plus_exactly_three_knobs():
    a = yaml.safe_load(open("impls/configs/critic/hilp_tdjc.yaml"))
    b = yaml.safe_load(open("impls/configs/critic/hilp_density_tdjc.yaml"))
    extra = {k: v for k, v in b.items() if k not in a}
    assert set(extra) == {"measure_density_temp", "measure_density_actor_temp",
                          "actor_q_center"}, extra
    assert all(a[k] == b[k] for k in a), "the substrate must be carried over verbatim"


def test_density_subclass_routes_its_own_sf_calls_through_the_encoder():
    """REGRESSION. HILPDensityCritic overrides the "sf" stage and owns three of its own
    `_sf` call sites. When the left encoder went in, those still passed the RAW observation
    while the tower had been built on the encoder's output, which blew up as

        ScopeParamShapeError: "embed_z_0" expected (78, 512), received (100, 512)

    i.e. obs_dim(28)+z(50) against L_dim(50)+z(50). Caught by a 20-step smoke run, not by
    the unit suite, because every earlier test built the critic without the encoder. This
    pins that the subclass and the parent agree about what the SF tower eats.
    """
    import inspect

    from impls.critics import hilp_density

    src = inspect.getsource(hilp_density)
    # Every _sf call in the subclass must hand it a _le(...) result, never a bare obs row.
    bad = [ln.strip() for ln in src.splitlines()
           if "self._sf(" in ln and "_le(" not in ln and "self._le" not in ln]
    # a call may be split across lines; allow a continuation that carries the _le
    lines = src.splitlines()
    for i, ln in enumerate(lines):
        if "self._sf(" in ln and "_le(" not in ln:
            window = " ".join(lines[i:i + 3])
            if "_le(" in window and ln.strip() in bad:
                bad.remove(ln.strip())
    assert not bad, f"HILPDensityCritic calls _sf on a raw observation: {bad}"


def test_density_tdjc_builds_and_its_sf_tower_matches_the_encoder_width():
    """The end-to-end shape agreement the smoke run failed on."""
    cfg = yaml.safe_load(open("impls/configs/critic/hilp_density_tdjc.yaml"))
    cfg["seed"] = 0
    from impls.critics.hilp_density import HILPDensityCritic
    critic, params, _t = HILPDensityCritic.build(EX_OBS, EX_ACT, cfg)
    assert critic.le_on is True
    enc = critic._le(params, EX_OBS)
    assert enc.shape == (4, cfg["L_dim"])
    # the tower must accept exactly that, not the raw row
    out = critic._sf(params["hilp_sf"], enc, jnp.zeros((4, cfg["z_dim"])), EX_ACT)
    assert out.shape[0] == 2 and out.shape[1] == 4
