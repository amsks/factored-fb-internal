import jax, jax.numpy as jnp, numpy as np
from impls.utils.networks import (
    PhiMap, ForwardMap, NoiseConditionedActor, FlowVectorField,
)
from impls.utils import torch_to_flax as t2f

ATOL = 1e-10
# EQUIV fixture dims (see PSMFlows export_fb_fixture.py): obs=8, act=5, batch=16, z=8, L=8, hidden=32, P=2.
H, Z, L, A = 32, 8, 8, 5


def _in(fx, name):
    return jnp.asarray(fx["in__" + name], jnp.float64)


def test_backward_equiv(fb_fixture):
    fx = fb_fixture
    net = PhiMap(z_dim=Z, hidden_dim=H, hidden_layers=4, norm=True)
    params = t2f.load_backward_params(fx)
    # static per-module export applies B to `observations`.
    out = net.apply({"params": params}, _in(fx, "observations"))
    np.testing.assert_allclose(np.asarray(out), fx["out__B"], atol=ATOL)


def test_left_encoder_equiv(fb_fixture):
    fx = fb_fixture
    net = PhiMap(z_dim=L, hidden_dim=H, hidden_layers=4, norm=True)
    params = t2f.load_left_encoder_params(fx)
    out = net.apply({"params": params}, _in(fx, "observations"))
    np.testing.assert_allclose(np.asarray(out), fx["out__left_enc"], atol=ATOL)


def test_forward_equiv(fb_fixture):
    fx = fb_fixture
    net = ForwardMap(z_dim=Z, hidden_dim=H, num_parallel=2, hidden_layers=2, embedding_layers=2)
    params = t2f.load_forward_params(fx)
    # F consumes left_enc(obs) as obs_feat; static export uses z_gauss (pre-mix) and actions.
    out = net.apply({"params": params}, jnp.asarray(fx["out__left_enc"], jnp.float64),
                    _in(fx, "z_gauss"), _in(fx, "actions"))
    np.testing.assert_allclose(np.asarray(out), fx["out__F"], atol=ATOL)  # [P, B, z]


def test_flow_actor_equiv(fb_fixture):
    fx = fb_fixture
    net = NoiseConditionedActor(action_dim=A, hidden_dim=H, hidden_layers=2, embedding_layers=2)
    params = t2f.load_flow_actor(fx)
    out = net.apply({"params": params}, _in(fx, "observations"), _in(fx, "z_gauss"), _in(fx, "actor_noise"))
    np.testing.assert_allclose(np.asarray(out), fx["out__actor_mu"], atol=ATOL)


def test_flow_vf_equiv(fb_fixture):
    fx = fb_fixture
    net = FlowVectorField(action_dim=A, hidden_dim=H, hidden_layers=4)
    params = t2f.load_flow_vf(fx)
    # static export evaluates the VF at x_t = actions.
    out = net.apply({"params": params}, _in(fx, "observations"), _in(fx, "actions"), _in(fx, "flow_t"))
    np.testing.assert_allclose(np.asarray(out), fx["out__vf"], atol=ATOL)
