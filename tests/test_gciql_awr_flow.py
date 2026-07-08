"""TDD for actor_kind='awr_flow' in gciql_flow: IQL's canonical AWR extraction on a
goal-conditioned flow-BC policy. Weight exp(awr_alpha*(minQ(s,a,g)-V(s,g))); actions
sampled by Euler rollout of the vf; NO one-step actor / distillation networks. The
missing cell in the readout matrix {AWR, Q-max} x {GCIQL, GCIVL, HIQL}."""
from tests.jax_subprocess import run_jax, OGBENCH_IMPLS


def _run(code):
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


_SETUP = """
import jax, jax.numpy as jnp
from gciql_flow import GCIQLFlowAgent, get_config
ex_obs = jnp.zeros((4, 11)); ex_act = jnp.zeros((4, 5))
obs = jax.random.normal(jax.random.PRNGKey(1), (4, 11))
act = jnp.tanh(jax.random.normal(jax.random.PRNGKey(3), (4, 5)))
batch = dict(observations=obs, next_observations=obs, actions=act,
             rewards=jnp.zeros((4,)), masks=jnp.ones((4,)),
             value_goals=obs, actor_goals=obs)
def make(**over):
    cfg = dict(get_config()); cfg["encoder"] = None; cfg["frame_stack"] = None
    cfg["actor_kind"] = "awr_flow"; cfg.update(over)
    return GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
def moved(a, b):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return any(bool(jnp.any(x != y)) for x, y in zip(la, lb))
"""


def test_awr_flow_builds_trains_samples_mlp_and_bilinear():
    _run(_SETUP + """
for over in ({}, {"value_arch": "bilinear", "value_latent_dim": 16}):
    agent = make(**over)
    assert "modules_actor_vf" in agent.network.params
    assert "modules_actor" not in agent.network.params, "awr_flow must not build a one-step actor"
    vf0 = agent.network.params["modules_actor_vf"]
    for _ in range(2):
        agent, info = agent.update(batch)
    assert jnp.isfinite(info["actor/actor_loss"]) and jnp.isfinite(info["actor/exp_a"]), sorted(info)
    assert jnp.isfinite(info["critic/critic_loss"]) and jnp.isfinite(info["value/value_loss"])
    assert moved(vf0, agent.network.params["modules_actor_vf"]), "vf must train"
    a = agent.sample_actions(ex_obs, ex_obs, seed=jax.random.PRNGKey(0))
    assert a.shape == (4, 5) and bool(jnp.all(jnp.abs(a) <= 1.0 + 1e-5))
    a1 = agent.sample_actions(ex_obs[0], ex_obs[0], seed=jax.random.PRNGKey(0))
    assert a1.shape == (5,), "unbatched eval path must work"
print("OK")
""")


def test_awr_flow_actor_does_not_train_critic_or_value():
    """The AWR weight reads Q/V without grad_params: only critic_loss/value_loss may
    train them. Zero their loss contribution via a frozen check: actor step gradient
    must leave critic+value params untouched when we call actor_loss alone."""
    _run(_SETUP + """
import optax
agent = make()
params = agent.network.params
loss, info = agent.actor_loss(batch, agent.network.params, rng=jax.random.PRNGKey(0))
g = jax.grad(lambda p: agent.actor_loss(batch, p, rng=jax.random.PRNGKey(0))[0])(agent.network.params)
crit_g = jax.tree_util.tree_leaves(g["modules_critic"]) + jax.tree_util.tree_leaves(g["modules_value"])
assert all(bool(jnp.all(x == 0)) for x in crit_g), "AWR actor loss must not backprop into Q/V"
vf_g = jax.tree_util.tree_leaves(g["modules_actor_vf"])
assert any(bool(jnp.any(x != 0)) for x in vf_g), "vf must receive gradient"
print("OK")
""")


def test_awr_flow_default_config_unchanged():
    _run("""
import jax, jax.numpy as jnp
from gciql_flow import GCIQLFlowAgent, get_config
cfg = dict(get_config()); cfg["encoder"] = None; cfg["frame_stack"] = None
ex_obs = jnp.zeros((2, 11)); ex_act = jnp.zeros((2, 5))
agent = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
assert "modules_actor" in agent.network.params and "modules_actor_vf" in agent.network.params
print("OK")
""")
