"""HIQLCritic — hierarchical IQL on the factored seam (workstream 2).

Transcribed from vendored third_party/ogbench/impls/agents/hiql.py; nothing vendored is
imported. Mapping (per docs/specs/2026-08-21-density-fb-agent-branch.md):

  critic owns  goal_rep, the IVL value head, and the HIGH-level actor
  seam actor IS HIQL's low-level actor, conditioned on the rep the critic hands it

so HIQL keeps the every-actor-composes property. The low-level AWR advantage rides the
adv_fn hook added for GCIVL.
"""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

B, OBS, ACT, T = 12, 6, 3, 20


def _base_dataset():
    from impls.utils.datasets import Dataset
    rng = np.random.default_rng(0)
    size = T * 5
    terminals = np.zeros(size, np.float32)
    terminals[np.arange(1, 6) * T - 1] = 1.0
    return Dataset.create(
        observations=rng.standard_normal((size, OBS)).astype(np.float32),
        next_observations=rng.standard_normal((size, OBS)).astype(np.float32),
        actions=np.tanh(rng.standard_normal((size, ACT))).astype(np.float32),
        terminals=terminals,
    )


def _cfg(actor="awr"):
    from impls.utils.config_utils import load_config
    c = load_config("hiql", actor, "cube_single")
    c["batch_size"] = B
    return c


def _agent(actor="awr"):
    from impls.agents import Agent
    from impls.utils.datasets import make_dataset
    cfg = _cfg(actor)
    ds = make_dataset(_base_dataset(), cfg)
    batch = ds.sample(B)
    agent = Agent.create("hiql", actor, 0, jnp.asarray(batch["observations"]),
                         jnp.asarray(batch["actions"]), cfg)
    return agent, batch, cfg


def test_hiql_is_registered_and_uses_the_hierarchical_dataset():
    from impls.agents import CRITICS
    from impls.critics.hiql import HIQLCritic
    assert CRITICS["hiql"] is HIQLCritic
    assert _cfg()["dataset_class"] == "HGCDataset"


def test_cond_carries_both_the_rep_and_the_raw_low_goal():
    """cond_vec hands the actor the REP; adv_fn needs the RAW low goal, because
    vendored evaluates V(s', low_goal) whose rep is goal_rep([s'; g]) -- a different
    rep from goal_rep([s; g]). Collapsing the two silently changes the advantage."""
    agent, batch, cfg = _agent()
    cond = agent.critic.sample_cond(agent.net, batch, jax.random.PRNGKey(0))
    assert "low_goal" in cond and "goal" in cond
    assert cond["low_goal"].shape == (B, OBS)
    assert cond["goal"].shape == (B, int(cfg["rep_dim"]))
    assert agent.critic.cond_vec(cond).shape == (B, int(cfg["rep_dim"]))


def test_goal_rep_is_length_normalised():
    """vendored appends LengthNormalize to goal_rep_seq: ||rep|| == sqrt(rep_dim)."""
    agent, batch, cfg = _agent()
    cond = agent.critic.sample_cond(agent.net, batch, jax.random.PRNGKey(0))
    norms = np.linalg.norm(np.asarray(cond["goal"]), axis=-1)
    assert np.allclose(norms, np.sqrt(int(cfg["rep_dim"])), rtol=1e-6)


def test_low_level_advantage_is_the_next_state_one_at_the_low_goal():
    """vendored hiql.py:104-108: adv = mean(V(s',g_low)) - mean(V(s,g_low))."""
    agent, batch, _ = _agent()
    critic = agent.critic
    params = {g: agent.net[g].params for g in agent.net}
    cond = critic.sample_cond(agent.net, batch, jax.random.PRNGKey(0))
    got = np.asarray(critic.adv_fn(params, batch, cond))
    v_s = np.asarray(critic._V_raw(params["goal_rep"], params["value"], batch["observations"], cond["low_goal"]))
    v_ns = np.asarray(critic._V_raw(params["goal_rep"], params["value"], batch["next_observations"], cond["low_goal"]))
    assert np.allclose(got, v_ns - v_s, rtol=1e-10, atol=1e-12)
    assert float(np.abs(got).max()) > 0.0, "advantage is identically zero (BC collapse)"


def test_high_actor_predicts_a_subgoal_rep_at_eval():
    """map_eval_cond must turn the raw env goal into the high policy's subgoal REP, using
    the observations -- HIQL's conditioning is goal_rep([s; g]), a joint function of both,
    which is why the hook needed widening."""
    agent, batch, cfg = _agent()
    raw_goal = jnp.asarray(batch["observations"][:1])
    obs = jnp.asarray(batch["observations"][:1])
    cond = agent.critic.map_eval_cond(agent.net, {"goal": raw_goal[0]}, observations=obs)
    assert cond["goal"].shape[-1] == int(cfg["rep_dim"])


@pytest.mark.parametrize("actor", ["flowbc", "ddpgbc", "awr", "fql"])
def test_hiql_builds_and_updates_with_every_actor(actor):
    agent, batch, _ = _agent(actor)
    updated, info = agent.update(batch)
    assert np.isfinite(np.asarray(info["value_loss"])).all()
    assert np.isfinite(np.asarray(info["high_actor_loss"])).all()
    assert np.isfinite(np.asarray(info["actor_loss"])).all()
    before = jax.tree_util.tree_leaves(agent.net["value"].params)
    after = jax.tree_util.tree_leaves(updated.net["value"].params)
    assert any(not np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(before, after))
