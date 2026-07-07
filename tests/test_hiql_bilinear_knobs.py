"""TDD for the HIQL bilinear mechanism knobs (spec 2026-07-07-hiql-bilinear-mechanism):

  bilinear_form: 'rep' (option A, psi(goal_rep([s;g]))) | 'goal' (Schaul, psi(raw g))
  value_split:   'shared' | 'high_bilinear' (separate TD-trained bilinear head that
                 ONLY the high-level actor advantage reads; MLP V keeps the TD/low
                 path and remains goal_rep's trainer)

Both default-OFF. Mirrors tests/test_hiql_flow_injection.py; runs in the jax venv.
"""
from tests.jax_subprocess import run_jax, OGBENCH_IMPLS

_BATCH = """
import jax, jax.numpy as jnp
ex_obs = jnp.zeros((4, 11)); ex_act = jnp.zeros((4, 5))
obs = jax.random.normal(jax.random.PRNGKey(1), (4, 11))
nobs = jax.random.normal(jax.random.PRNGKey(2), (4, 11))
act = jnp.tanh(jax.random.normal(jax.random.PRNGKey(3), (4, 5)))
batch = dict(observations=obs, next_observations=nobs, actions=act,
             rewards=jnp.zeros((4,)), masks=jnp.ones((4,)), value_goals=nobs,
             low_actor_goals=nobs, high_actor_goals=nobs, high_actor_targets=nobs)
def moved(a, b):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return any(bool(jnp.any(x != y)) for x, y in zip(la, lb))
"""


def _run(code):
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_bilinear_form_goal_rep_gradient_routing():
    """Schaul form must cut the value->goal_rep gradient; low_actor_rep_grad=True
    must restore a trainer; option-A 'rep' must keep training goal_rep."""
    _run(_BATCH + """
from agents.hiql import HIQLAgent, get_config
def make(form, low_rep_grad):
    cfg = dict(get_config()); cfg["encoder"] = None; cfg["frame_stack"] = None
    cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 16
    cfg["bilinear_form"] = form; cfg["low_actor_rep_grad"] = low_rep_grad
    return HIQLAgent.create(0, ex_obs, ex_act, cfg)
def rep(agent): return agent.network.params["modules_goal_rep"]
ag = make("goal", False); p0 = rep(ag); ag2, info = ag.update(batch)
assert jnp.isfinite(info["value/value_loss"])
assert not moved(p0, rep(ag2)), "form='goal'+no low rep grad: goal_rep must be FROZEN"
ag = make("goal", True); p0 = rep(ag); ag2, _ = ag.update(batch)
assert moved(p0, rep(ag2)), "low_actor_rep_grad=True must train goal_rep"
ag = make("rep", False); p0 = rep(ag); ag2, _ = ag.update(batch)
assert moved(p0, rep(ag2)), "option-A must keep training goal_rep through psi"
print("OK")
""")


def test_bilinear_form_default_rep_unchanged_tree():
    """Default config must not change the param tree (checkpoint compat)."""
    _run("""
import jax.numpy as jnp
from agents.hiql import HIQLAgent, get_config
cfg = dict(get_config()); cfg["encoder"] = None; cfg["frame_stack"] = None
assert cfg["bilinear_form"] == "rep" and cfg["value_split"] == "shared"
ex_obs = jnp.zeros((4, 11)); ex_act = jnp.zeros((4, 5))
agent = HIQLAgent.create(0, ex_obs, ex_act, cfg)
keys = set(agent.network.params.keys())
assert keys == {"modules_goal_rep", "modules_value", "modules_target_value",
                "modules_low_actor", "modules_high_actor"}, keys
print("OK")
""")


def test_value_split_high_bilinear_builds_trains_and_keeps_rep_trainer():
    _run(_BATCH + """
from agents.hiql import HIQLAgent, get_config
cfg = dict(get_config()); cfg["encoder"] = None; cfg["frame_stack"] = None
cfg["value_split"] = "high_bilinear"; cfg["value_latent_dim"] = 16
cfg["bilinear_form"] = "goal"   # E-D cell: Schaul high head, zero rep coupling
agent = HIQLAgent.create(0, ex_obs, ex_act, cfg)
assert "modules_high_value" in agent.network.params
assert "modules_target_high_value" in agent.network.params
p0 = agent.network.params["modules_goal_rep"]
hv0 = agent.network.params["modules_high_value"]
for _ in range(2):
    agent, info = agent.update(batch)
assert jnp.isfinite(info["value/value_loss"])
assert jnp.isfinite(info["high_value/value_loss"]), sorted(info)
assert jnp.isfinite(info["high_actor/actor_loss"])
assert moved(p0, agent.network.params["modules_goal_rep"]), "MLP V must still train goal_rep"
assert moved(hv0, agent.network.params["modules_high_value"]), "high head must train by TD"
print("OK")
""")


def test_value_split_default_shared_no_extra_modules():
    _run(_BATCH + """
from agents.hiql import HIQLAgent, get_config
cfg = dict(get_config()); cfg["encoder"] = None; cfg["frame_stack"] = None
agent = HIQLAgent.create(0, ex_obs, ex_act, cfg)
assert "modules_high_value" not in agent.network.params
ag2, info = agent.update(batch)
assert not any(k.startswith("high_value/") for k in info), sorted(info)
print("OK")
""")
