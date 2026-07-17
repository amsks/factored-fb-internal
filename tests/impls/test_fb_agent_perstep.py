"""10-step weight-trace parity: the whole per-group param tree (online + targets) must
match the PyTorch reference fixture after each update, at atol=1e-8.

Instead of spot-checking hand-built param paths, we reconstruct each step's reference flax
tree by renaming the fixture's `step__{i}__<state_dict_key>` snapshots to `w__<...>` and
feeding them through the same torch_to_flax loaders used for the initial params. This
compares every leaf, not a sample."""
import jax.numpy as jnp, numpy as np
from flax.traverse_util import flatten_dict
from impls.agents import Agent
from impls.utils import torch_to_flax as t2f
from tests.impls.test_combo_validation import _min_cfg

ATOL = 1e-8

_ONLINE_LOADER = {
    "forward": lambda v: t2f.load_forward_params(v),
    "backward": lambda v: t2f.load_backward_params(v),
    "left_encoder": lambda v: t2f.load_left_encoder_params(v),
    "actor": lambda v: t2f.load_flow_actor(v),
    "actor_vf": lambda v: t2f.load_flow_vf(v),
}
_TARGET_LOADER = {
    "forward": lambda v: t2f.load_forward_params(v, prefix="_target_forward_map"),
    "backward": lambda v: t2f.load_backward_params(v, prefix="_target_backward_map"),
    "left_encoder": lambda v: t2f.load_backward_params(v, prefix="_target_left_encoder"),
}


def _load_all_online(fx):
    return {g: _ONLINE_LOADER[g](fx) for g in _ONLINE_LOADER}


def _mapped_agent(fx):
    """Create an FB×flowbc agent, then override every online+target group from the fixture."""
    agent = Agent.create("fb", "flowbc", 0, jnp.asarray(fx["in__observations"], jnp.float64),
                         jnp.asarray(fx["in__actions"], jnp.float64), _min_cfg())
    net = dict(agent.net)
    for g, p in _load_all_online(fx).items():
        net[g] = net[g].replace(params=p)
    target = {g: _TARGET_LOADER[g](fx) for g in _TARGET_LOADER}
    return agent.replace(net=net, target=target)


def _step_view(fx, i):
    """Rename step__{i}__<key> snapshots to w__<key> so the loaders can read them."""
    pre = f"step__{i}__"
    return {f"w__{k[len(pre):]}": fx[k] for k in fx if k.startswith(pre)}


def _assert_tree_close(got, ref, tag):
    fg, fr = flatten_dict(got), flatten_dict(ref)
    assert fg.keys() == fr.keys(), f"{tag}: key mismatch {set(fg) ^ set(fr)}"
    for k in fg:
        np.testing.assert_allclose(np.asarray(fg[k]), np.asarray(fr[k]), atol=ATOL,
                                   err_msg=f"{tag} leaf {k}")


def test_fb_perstep_weight_trace(fb_fixture):
    fx = fb_fixture
    agent = _mapped_agent(fx)
    batch = {k: jnp.asarray(fx["in__" + k], jnp.float64)
             for k in ("observations", "actions", "next_observations")}
    for i in range(10):
        cond = {"z": jnp.asarray(fx[f"step_in__{i}__z"], jnp.float64)}
        boot = {
            "next_action": jnp.asarray(fx[f"step_in__{i}__next_action"], jnp.float64),
            "flow_x0": jnp.asarray(fx[f"step_in__{i}__flow_x0"], jnp.float64),
            "flow_t": jnp.asarray(fx[f"step_in__{i}__flow_t"], jnp.float64),
            "actor_noise": jnp.asarray(fx[f"step_in__{i}__actor_noise"], jnp.float64),
        }
        agent, _ = agent.apply_update(batch, cond, boot)
        view = _step_view(fx, i)
        for g in _ONLINE_LOADER:
            _assert_tree_close(agent.net[g].params, _ONLINE_LOADER[g](view), f"step {i} online {g}")
        for g in _TARGET_LOADER:
            _assert_tree_close(agent.target[g], _TARGET_LOADER[g](view), f"step {i} target {g}")
