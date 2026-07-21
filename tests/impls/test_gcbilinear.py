"""Axis-2 lock: in-house GCBilinearValue vs VENDORED OGBench GCBilinearValue.

Reference: third_party/ogbench/impls/utils/networks.py:327 (upstream OGBench). This is a
real upstream reference, so the bar is a hard 1e-10 on v, phi and psi, across two configs
(latent_dim 8/ensemble and 32/no-ensemble) and both call paths (actions=None -> V(s,g);
actions given -> Q(s,a,g)). Fixture baked by fixtures/_gen_bilinear_fixture.py.
"""
import os
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import pytest
from flax.traverse_util import unflatten_dict

from impls.utils.networks import GCBilinearValue

ATOL = 1e-10
_FX = np.load(os.path.join(os.path.dirname(__file__), "fixtures", "bilinear_reference.npz"))
_CONFIGS = ["c0", "c1"]


def _params(name, tag):
    pref = f"param/{name}/{tag}/"
    flat = {tuple(k[len(pref):].split("/")): jnp.asarray(_FX[k], jnp.float64)
            for k in _FX.files if k.startswith(pref)}
    return unflatten_dict(flat)


def _batch():
    return {k[len("batch/"):]: jnp.asarray(_FX[k], jnp.float64)
            for k in _FX.files if k.startswith("batch/")}


def _net(name):
    hd = tuple(int(x) for x in _FX["meta/hidden_dims"])
    return GCBilinearValue(hidden_dims=hd,
                           latent_dim=int(_FX[f"meta/{name}/latent_dim"]),
                           layer_norm=True,
                           ensemble=bool(_FX[f"meta/{name}/ensemble"]))


@pytest.mark.parametrize("name", _CONFIGS)
@pytest.mark.parametrize("tag", ["value", "critic"])
def test_matches_vendored(name, tag):
    b = _batch()
    net = _net(name)
    params = _params(name, tag)
    act = b["actions"] if tag == "critic" else None
    v, phi, psi = net.apply({"params": params}, b["observations"], b["goals"], act, info=True)
    np.testing.assert_allclose(np.asarray(v), _FX[f"exp/{name}/{tag}/v"], atol=ATOL)
    np.testing.assert_allclose(np.asarray(phi), _FX[f"exp/{name}/{tag}/phi"], atol=ATOL)
    np.testing.assert_allclose(np.asarray(psi), _FX[f"exp/{name}/{tag}/psi"], atol=ATOL)


@pytest.mark.parametrize("name", _CONFIGS)
@pytest.mark.parametrize("tag", ["value", "critic"])
def test_info_false_return_path(name, tag):
    """info=False must return exactly v (not the (v, phi, psi) tuple)."""
    b = _batch()
    net = _net(name)
    params = _params(name, tag)
    act = b["actions"] if tag == "critic" else None
    v = net.apply({"params": params}, b["observations"], b["goals"], act)
    assert not isinstance(v, tuple)
    np.testing.assert_allclose(np.asarray(v), _FX[f"exp/{name}/{tag}/v_noinfo"], atol=ATOL)


def test_param_structure_is_phi_psi():
    """Submodule names must be `phi`/`psi` to match vendored (checkpoint compatibility)."""
    p = _params("c0", "critic")
    assert set(p.keys()) == {"phi", "psi"}


def test_sqrt_latent_dim_normalisation_present():
    """Structural guard: v must be sum(phi*psi)/sqrt(d), not sum(phi*psi).

    Recomputes v from the returned phi/psi at two different latent_dim values, so a dropped
    1/sqrt(d) cannot pass by coincidence at a single d.
    """
    b = _batch()
    hd = tuple(int(x) for x in _FX["meta/hidden_dims"])
    for d in (4, 64):
        net = GCBilinearValue(hidden_dims=hd, latent_dim=d, layer_norm=True, ensemble=False)
        params = net.init(jr.PRNGKey(7), b["observations"], b["goals"], b["actions"])["params"]
        v, phi, psi = net.apply({"params": params}, b["observations"], b["goals"],
                                b["actions"], info=True)
        assert phi.shape[-1] == d and psi.shape[-1] == d
        expected = (phi * psi).sum(-1) / np.sqrt(d)
        np.testing.assert_allclose(np.asarray(v), np.asarray(expected), atol=ATOL)
        # and the unnormalised form must NOT match (guards against d==1-style degeneracy)
        assert not np.allclose(np.asarray(v), np.asarray((phi * psi).sum(-1)), atol=ATOL)


def test_value_exp_exponentiates():
    b = _batch()
    hd = tuple(int(x) for x in _FX["meta/hidden_dims"])
    kw = dict(hidden_dims=hd, latent_dim=8, layer_norm=True, ensemble=False)
    lin, exp = GCBilinearValue(**kw), GCBilinearValue(value_exp=True, **kw)
    params = lin.init(jr.PRNGKey(7), b["observations"], b["goals"])["params"]
    v = lin.apply({"params": params}, b["observations"], b["goals"])
    ve = exp.apply({"params": params}, b["observations"], b["goals"])
    np.testing.assert_allclose(np.asarray(ve), np.exp(np.asarray(v)), atol=ATOL)
