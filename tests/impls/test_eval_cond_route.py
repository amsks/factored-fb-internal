"""The reward-inferred conditioning: is it REACHABLE, is it OBSERVABLY used, and is it
solved against the right marginal?

Run this file on its own, on CPU (running all of tests/impls in one process core-dumps):

    PYTHONPATH=$PWD JAX_PLATFORMS=cpu python -m pytest tests/impls/test_eval_cond_route.py -q \
        -p no:cacheprovider

Two separate defects are covered here, and tests/impls/test_infer_eval_cond.py covers
neither of them (it tests `infer_eval_cond` in isolation, which always worked):

  GAP 1 -- unreachable. `Agent.sample_actions` merges a supplied goal over `cond_eval`, and
  both evaluators supply one for every goal-capable critic. For the critics with a
  `map_eval_cond` hook (fb, psm, rldp, density_fb) that goal is then re-encoded INTO the
  conditioning, so `--infer_cond_episodes` produced a vector that the very next action draw
  discarded. `--eval_on_inferred_cond` routes eval through `InferredCondAgent`, which never
  forwards a goal, and publishes counters that PROVE which conditioning was rolled.

  GAP 2 -- wrong marginal. impls/main.py fed `rb["observations"]` to every critic's
  `infer_cond`. The FB family's backward/index tower is trained on `next_observations`
  (`goal = next_obs` in stage_loss; density_fb's rho_phi is "the next-state query
  marginal"), so the readout was being solved against a different measure than the one it
  integrates -- silently, since nothing downstream can tell.
"""
import os
import types

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from impls.agents import Agent, CRITICS
from impls.critics.base import (ComboError, INFER_COND_MARGINALS, infer_cond_marginal,
                                infer_cond_rows)

SEED = 0
OBS_DIM, ACT_DIM = 6, 2


def _fb_cfg():
    return dict(
        env_name="tiny", task="b2d-tiny", adapter="unified", horizon=8, batch_size=16,
        z_dim=8, L_dim=8, num_parallel=2, discount=0.99,
        f_target_tau=0.005, b_target_tau=0.005, ortho_coef=1.0, train_goal_ratio=0.5,
        fb_pessimism_penalty=0.0, norm_z=True, lr_f=1e-4, lr_b=1e-4, lr_actor=1e-4,
        forward=dict(hidden_dim=32, hidden_layers=2, embedding_layers=2),
        backward=dict(hidden_dim=32, hidden_layers=4, norm=True),
        left_encoder=dict(hidden_dim=32, hidden_layers=4, norm=True),
        actor=dict(type="flow", hidden_dim=32, hidden_layers=2, embedding_layers=2,
                   bc_coeff=3.0, q_coeff=1.0, flow_steps=5, lr_actor_vf=3e-4,
                   flow_vf_hidden_dim=32, flow_vf_hidden_layers=4),
    )


def _fb_agent(cfg=None):
    cfg = cfg or _fb_cfg()
    return Agent.create("fb", "flowbc", SEED, jnp.zeros((2, OBS_DIM)),
                        jnp.zeros((2, ACT_DIM)), cfg)


def _inferred(agent, n=16, seed=5):
    """An agent whose cond_eval came from real (non-degenerate) reward samples."""
    rng = np.random.default_rng(seed)
    obs = jnp.asarray(rng.standard_normal((n, OBS_DIM)), jnp.float32)
    rew = jnp.asarray(rng.standard_normal((n,)), jnp.float32)
    return agent.infer_eval_cond(obs, rew)


# ======================================================================================
# GAP 1a. The wrapper itself: a goal handed to it is dropped, not merged.
# ======================================================================================
def test_the_wrapper_drops_a_supplied_goal_and_counts_it():
    import impls.main as m

    agent = _inferred(_fb_agent())
    w = m.InferredCondAgent(agent)
    obs = jnp.zeros((OBS_DIM,))
    goal = jnp.arange(OBS_DIM, dtype=jnp.float32)
    key = jax.random.PRNGKey(3)

    with_goal = np.asarray(w.sample_actions(obs, goals=goal, seed=key, temperature=0))
    want = np.asarray(agent.sample_actions(obs, goals=None, seed=key, temperature=0))
    np.testing.assert_array_equal(with_goal, want)

    # ... and it is NOT what the unwrapped agent would have done with that goal: fb's
    # map_eval_cond re-derives z from the goal frame, which is the whole defect.
    goal_driven = np.asarray(agent.sample_actions(obs, goals=goal, seed=key, temperature=0))
    assert not np.allclose(goal_driven, want), \
        "test premise: for fb, a supplied goal must change the action (map_eval_cond)"

    assert w.action_calls == 1 and w.goals_dropped == 1
    wit = w.witness()
    assert wit["inferred_cond"] == 1.0 and wit["inferred_cond_goals_dropped"] == 1.0
    assert np.isfinite(wit["inferred_cond_norm"]) and wit["inferred_cond_norm"] > 0


def test_witness_cos_neutral_separates_the_constant_reward_fallback():
    """`inferred_cond_norm` cannot tell density_fb's neutral fallback from a real index --
    both are projected onto the sqrt(d) sphere -- so the witness carries cos(cond, 1_d),
    which is 1.0 for the fallback and strictly below it for anything else."""
    import impls.main as m

    agent = _inferred(_fb_agent())
    w = m.InferredCondAgent(agent)
    d = int(jnp.asarray(w.cond_eval["z"]).ravel().shape[0])

    # the fallback density_fb returns at ||c|| < z_const_eps: 1_d on the sqrt(d) sphere.
    w.cond_eval = {"z": jnp.ones((d,))}
    assert w.witness()["inferred_cond_cos_neutral"] == pytest.approx(1.0, abs=1e-6)

    # a reward-informed index is not parallel to 1_d.
    v = jnp.asarray(np.linspace(-1.0, 1.0, d), dtype=jnp.float32)
    w.cond_eval = {"z": jnp.sqrt(jnp.array(float(d))) * v / jnp.linalg.norm(v)}
    cos = w.witness()["inferred_cond_cos_neutral"]
    assert abs(cos) < 0.99, f"a non-neutral index must not read as the fallback (cos={cos})"

    # norm alone cannot make the distinction -- the point of the extra scalar. Both vectors
    # below sit on the sqrt(d) sphere; only one of them is the fallback.
    root_d = float(np.sqrt(d))
    for z in (jnp.ones((d,)), w.cond_eval["z"]):
        w.cond_eval = {"z": jnp.sqrt(jnp.array(float(d))) * z / jnp.linalg.norm(z)}
        assert w.witness()["inferred_cond_norm"] == pytest.approx(root_d, rel=1e-5)


def test_the_wrapper_carries_the_inferred_cond_not_the_placeholder():
    import impls.main as m

    base = _fb_agent()
    agent = _inferred(base)
    w = m.InferredCondAgent(agent)
    assert not np.allclose(np.asarray(w.cond_eval["z"]), np.asarray(base.cond_eval["z"])), \
        "the wrapper must expose the INFERRED cond, not eval_cond0"


# ======================================================================================
# GAP 1b. The evaluators stop building a goal for it.
# ======================================================================================
def test_needs_goal_frame_is_false_for_an_agent_that_rolls_an_inferred_cond():
    import impls.main as m
    from env.base import needs_goal_frame

    agent = _fb_agent()
    assert needs_goal_frame(agent), "fb exposes map_eval_cond, so normally it needs a goal"
    assert not needs_goal_frame(m.InferredCondAgent(agent))


def test_unified_evaluate_does_not_even_synthesise_a_goal_for_the_wrapper():
    """The strongest available proof that the goal path is skipped rather than merely
    overridden: `eval_goal_mode=synth` on a task with no goal-in-obs layout RAISES when a
    goal frame is required. Under the wrapper it must not, because no frame is built."""
    import impls.main as m
    from env.unified import UnifiedAdapter

    class _Env:
        def reset(self, seed=None, options=None):
            return np.zeros(OBS_DIM, np.float32), {"success": False}

        def step(self, a):
            return np.zeros(OBS_DIM, np.float32), 0.0, False, True, {"success": False}

        def render(self):
            return np.zeros((2, 2, 3), np.uint8)

    cfg = {**_fb_cfg(), "eval_goal_mode": "synth"}
    ad = UnifiedAdapter()
    ad._horizon = 2
    agent = _inferred(_fb_agent())

    with pytest.raises(ValueError, match="no goal-in-obs layout"):
        ad.evaluate(agent, _Env(), None, cfg, 1, 0)

    row, _ = ad.evaluate(m.InferredCondAgent(agent), _Env(), None, cfg, 1, 0)
    assert "success" in row


# ======================================================================================
# GAP 1c. End to end through impls/main.py, and OBSERVABLE in the run's own outputs.
# ======================================================================================
class _RecordingAdapter:
    """Stub adapter whose `evaluate` drives sample_actions exactly as the real ones do --
    with a goal -- and records what came back."""

    name = "unified"

    def __init__(self):
        self._horizon = 8
        self._goal = np.arange(OBS_DIM, dtype=np.float32)
        self.saw_goal_kwarg = []
        self.env = None

    def make_env_and_datasets(self, cfg):
        from impls.utils.datasets import FBDataset
        n_ep, ep_len = 6, 8
        n = n_ep * ep_len
        term = np.zeros((n_ep, ep_len), np.float32)
        term[:, -1] = 1.0
        rng = np.random.default_rng(0)
        # DISJOINT ranges: observations are strictly negative, next_observations strictly
        # positive, so "which column did infer_cond receive" is decidable from the values
        # alone, without knowing which row indices were drawn.
        data = dict(
            observations=(-1.0 - np.abs(rng.standard_normal((n, OBS_DIM)))).astype(np.float32),
            actions=np.tanh(rng.standard_normal((n, ACT_DIM))).astype(np.float32),
            next_observations=(1.0 + np.abs(rng.standard_normal((n, OBS_DIM)))).astype(np.float32),
            rewards=rng.standard_normal(n).astype(np.float32),
            terminals=term.reshape(-1),
        )
        return None, FBDataset.create(**data), FBDataset.create(**data)

    def eval_tasks(self, env, cfg):
        return [None]

    def evaluate(self, agent, env, task, cfg, num_episodes, num_video):
        goal = jnp.arange(OBS_DIM, dtype=jnp.float32)
        a = agent.sample_actions(jnp.zeros((OBS_DIM,)), goals=goal,
                                 seed=jax.random.PRNGKey(0), temperature=0)
        self.saw_goal_kwarg.append(np.asarray(a))
        return {"success": 0.0}, []


def _run_main(tmp_path, monkeypatch, cfg, adapter, critic, actor, steps, **over):
    import impls.main as m

    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(m, "make_config", lambda *a, **k: dict(cfg))
    monkeypatch.setattr(m, "get_adapter", lambda c: adapter)
    monkeypatch.setattr(m, "save_agent", lambda agent, d, i: None)

    m.FLAGS(["test"], known_only=True)
    defaults = dict(critic=critic, actor=actor, domain="tiny", override=[], seed=SEED,
                    train_steps=steps, log_interval=10 ** 9, eval_interval=10 ** 9,
                    save_interval=10 ** 9, eval_episodes=1, num_video=0,
                    save_dir=str(tmp_path), restore_path="", warm_start_path="",
                    agg_dir=None, infer_cond_episodes=0, online=False,
                    eval_on_inferred_cond=False)
    for k, v in {**defaults, **over}.items():
        monkeypatch.setattr(m.FLAGS, k, v)
    m.main(None)
    return os.path.join(tmp_path, "tiny_fb_flowbc_seed_0")


def test_eval_on_inferred_cond_reaches_the_evaluator_and_is_logged(tmp_path, monkeypatch):
    """The whole point: with the flag ON, the action the evaluator gets back is the one the
    INFERRED cond produces, even though the evaluator passed a goal -- and eval.csv says so."""
    adapter = _RecordingAdapter()
    run_dir = _run_main(tmp_path, monkeypatch, _fb_cfg(), adapter, "fb", "flowbc", 2,
                        infer_cond_episodes=16, eval_on_inferred_cond=True)

    assert adapter.saw_goal_kwarg, "the stub evaluator never ran"

    header = open(os.path.join(run_dir, "eval.csv")).readline()
    for col in ("inferred_cond", "inferred_cond_goals_dropped", "inferred_cond_norm",
                "inferred_cond_action_calls"):
        assert col in header, f"{col} missing from eval.csv -- the run cannot prove anything"
    rows = open(os.path.join(run_dir, "eval.csv")).read().strip().split("\n")[1:]
    cols = header.strip().split(",")
    rec = dict(zip(cols, rows[0].split(",")))
    assert float(rec["inferred_cond"]) == 1.0
    # POSITIVE evidence, not just an absence: the goal path was live and was intercepted.
    assert float(rec["inferred_cond_goals_dropped"]) == 1.0
    assert float(rec["inferred_cond_action_calls"]) == 1.0
    assert float(rec["inferred_cond_norm"]) > 0.0


def test_without_the_flag_the_env_goal_still_wins(tmp_path, monkeypatch):
    """The regression guard on the other side: --infer_cond_episodes alone must NOT change
    goal-conditioned evaluation, and the action must be the goal-driven one."""
    a_off = _RecordingAdapter()
    run_dir = _run_main(tmp_path, monkeypatch, _fb_cfg(), a_off, "fb", "flowbc", 2,
                        infer_cond_episodes=16, eval_on_inferred_cond=False)
    a_on = _RecordingAdapter()
    _run_main(tmp_path / "on", monkeypatch, _fb_cfg(), a_on, "fb", "flowbc", 2,
              infer_cond_episodes=16, eval_on_inferred_cond=True)

    assert not np.allclose(a_off.saw_goal_kwarg[0], a_on.saw_goal_kwarg[0]), \
        "the two routes produced the same action; the flag did nothing"
    assert "inferred_cond" not in open(os.path.join(run_dir, "eval.csv")).readline(), \
        "the witness columns must not appear in a run that did not take the route"


def test_the_flag_is_refused_without_anything_to_infer(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="infer_cond_episodes"):
        _run_main(tmp_path, monkeypatch, _fb_cfg(), _RecordingAdapter(), "fb", "flowbc", 1,
                  infer_cond_episodes=0, eval_on_inferred_cond=True)


def test_the_noop_detector_now_names_the_goal_override():
    """Before --eval_on_inferred_cond existed, this case was invisible: inference changed
    cond_eval, so the detector said "fine", and the goal then discarded it anyway."""
    import impls.main as m

    agent = _inferred(_fb_agent())
    pre = {"z": jnp.zeros((8,))}
    reason = m.infer_cond_noop_reason(pre, agent.cond_eval, jnp.ones((8,)), agent=agent)
    assert reason is not None and "eval_on_inferred_cond" in reason

    wrapped = m.InferredCondAgent(agent)
    assert m.infer_cond_noop_reason(pre, wrapped.cond_eval, jnp.ones((8,)),
                                    agent=wrapped) is None
    # The pre-existing 3-argument call must keep working (test_infer_eval_cond.py uses it).
    assert m.infer_cond_noop_reason(pre, agent.cond_eval, jnp.ones((8,))) is None


# ======================================================================================
# GAP 2. infer_cond is fed the marginal the critic declares.
# ======================================================================================
def test_every_registered_critic_declares_a_valid_marginal():
    for name, critic in sorted(CRITICS.items()):
        m = infer_cond_marginal(critic)
        assert m in INFER_COND_MARGINALS, f"{name} declares {m!r}"


@pytest.mark.parametrize("name", ["fb", "rldp", "psm", "density_fb"])
def test_the_fb_family_is_solved_against_the_next_state_marginal(name):
    """Each of these builds its backward/index tower on next_observations -- `goal =
    next_obs` in stage_loss, `B(next_observations)` in sample_cond, density_fb's pinned
    rho_phi. Asserted PER CRITIC rather than as one global rule."""
    assert infer_cond_marginal(CRITICS[name]) == "next_observations"
    assert infer_cond_rows(CRITICS[name]) == "next_observations"


@pytest.mark.parametrize("name", ["gciql", "gcivl", "hiql", "iql_bilinear", "crl", "qrl"])
def test_the_goal_conditioned_placeholders_declare_that_they_ignore_the_rows(name):
    """Their infer_cond returns a fixed zero placeholder. Declaring "unused" is the honest
    statement; it is not the same claim as "next_observations is the right marginal"."""
    assert infer_cond_marginal(CRITICS[name]) == "unused"


def test_an_undeclared_marginal_is_refused_by_name():
    class _Bogus:
        pass

    with pytest.raises(ComboError, match="INFER_COND_MARGINAL"):
        infer_cond_rows(_Bogus)

    class _Wrong:
        INFER_COND_MARGINAL = "value_goals"

    with pytest.raises(ComboError, match="INFER_COND_MARGINAL"):
        infer_cond_rows(_Wrong)


def test_main_feeds_infer_cond_the_next_state_column(tmp_path, monkeypatch):
    """End to end: patch FBCritic.infer_cond to record its `obs`, and check the recorded
    rows against the DISJOINT value ranges the stub dataset uses (observations < 0,
    next_observations > 0). Before the fix this recorded the negative column."""
    from impls.critics.fb import FBCritic

    seen = []
    real = FBCritic.infer_cond

    def spy(self, net, obs, rewards):
        seen.append(np.asarray(obs))
        return real(self, net, obs, rewards)

    monkeypatch.setattr(FBCritic, "infer_cond", spy)
    _run_main(tmp_path, monkeypatch, _fb_cfg(), _RecordingAdapter(), "fb", "flowbc", 2,
              infer_cond_episodes=16, eval_on_inferred_cond=True)

    assert seen, "infer_cond was never called"
    rows = seen[0]
    assert rows.shape == (16, OBS_DIM)
    assert (rows > 0).all(), (
        "infer_cond got the CURRENT-state column (all-negative in this fixture); the FB "
        "family's readout must be solved on next_observations")
