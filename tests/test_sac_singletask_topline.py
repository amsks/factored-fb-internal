"""Spec 27 Q4 / spec 29 section 2 -- the offline supervised topline.

Covers the pieces added for this cell (tools/wandb_mode_shim/sac_singletask.py
+ sitecustomize.py):
  - `SACSingleTaskAgent(SACAgent)`: extends the vendored, non-goal-conditioned
    `agents.sac.get_config()` with the keys `main.py`/`GCDataset` require, and
    overrides ONLY `sample_actions` to drop the `goals` argument before
    calling the actor -- `evaluate()` (utils/evaluation.py) always calls
    `sample_actions(..., goals=<real goal>)`, but the actor was initialized
    and trained with `goals=None` (critic_loss/actor_loss never pass a goal),
    so passing the real goal through concatenates an extra obs_dim onto the
    actor's input and raises a ScopeParamShapeError. Registered as
    `agents['sac_singletask']` by sitecustomize.py; `agents['sac']` itself is
    untouched.
  - `sitecustomize.py`'s `GCDataset.sample` wrap: opt-in (`native_reward_
    singletask=True`) substitution of the dataset's OWN `rewards`/`masks`
    (written by OGBench's `relabel_dataset` for `<env>-play-singletask-
    taskN-v0`) in place of GCDataset's synthetic hindsight reward -- and,
    symmetrically, a default-off proof that every *other* GCDataset config
    (which never sets this key) gets the exact original batch back,
    bit-identical.

Runs in the jax venv subprocess (mirrors tests/test_gciql_measure_hit_spec.py).
"""
import textwrap

from tests.jax_subprocess import run_jax, OGBENCH_IMPLS

COMMON = """
    import numpy as np
    import jax.numpy as jnp
    from agents import agents
    from sac_singletask import get_config
"""


def _run(code):
    code = textwrap.dedent(COMMON) + textwrap.dedent(code)
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_get_config_has_the_keys_main_py_and_gcdataset_require():
    _run("""
        cfg = get_config()
        assert cfg.agent_name == 'sac_singletask'
        for k in ('dataset_class', 'discrete', 'encoder', 'frame_stack', 'p_aug'):
            assert k in cfg, k
        assert cfg.dataset_class == 'GCDataset'
        assert cfg.discrete is False
        assert (cfg.value_p_curgoal + cfg.value_p_trajgoal + cfg.value_p_randomgoal) == 1.0
        assert (cfg.actor_p_curgoal + cfg.actor_p_trajgoal + cfg.actor_p_randomgoal) == 1.0
        assert cfg.native_reward_singletask is True
        print("OK")
    """)


def test_sac_singletask_agent_is_registered_and_distinct_from_sac():
    _run("""
        assert 'sac_singletask' in agents
        assert agents['sac_singletask'] is not agents['sac']
        assert issubclass(agents['sac_singletask'], agents['sac'])
        print("OK")
    """)


def test_sample_actions_with_a_real_goal_does_not_shape_error():
    """The regression this agent exists to fix: main.py's evaluate() always
    passes the env's real goal to sample_actions. A plain agents['sac']
    trained/init'd with goals=None raises ScopeParamShapeError the moment a
    non-None goal reaches the actor; agents['sac_singletask'] must not."""
    _run("""
        import jax

        cfg = dict(get_config())
        cfg['actor_hidden_dims'] = (32, 32)
        cfg['value_hidden_dims'] = (32, 32)

        ex_obs = jnp.zeros((1, 10), jnp.float32)
        ex_act = jnp.zeros((1, 4), jnp.float32)
        agent = agents['sac_singletask'].create(0, ex_obs, ex_act, cfg)

        batch = dict(
            observations=jnp.zeros((8, 10), jnp.float32),
            actions=jnp.zeros((8, 4), jnp.float32),
            next_observations=jnp.zeros((8, 10), jnp.float32),
            rewards=jnp.full((8,), -1.0, jnp.float32),
            masks=jnp.ones((8,), jnp.float32),
        )
        new_agent, info = agent.update(batch)
        assert 'critic/critic_loss' in info
        assert 'actor/actor_loss' in info

        # A REAL (non-None) goal, exactly what utils/evaluation.py passes.
        real_goal = jnp.ones((1, 10), jnp.float32)
        acts = new_agent.sample_actions(
            observations=ex_obs, goals=real_goal, seed=jax.random.PRNGKey(0)
        )
        assert acts.shape == (1, 4)
        print("OK")
    """)


def _fake_gcdataset_body(native_reward: bool) -> str:
    return f"""
        import utils.datasets as ds

        n = 16
        real_rewards = np.arange(n, dtype=np.float32) - 100.0  # far from -1/0
        real_masks = np.zeros(n, dtype=np.float32)

        raw = ds.Dataset.create(
            observations=np.zeros((n, 3), np.float32),
            actions=np.zeros((n, 2), np.float32),
            next_observations=np.zeros((n, 3), np.float32),
            terminals=np.array([0.0] * (n - 1) + [1.0], np.float32),
            rewards=real_rewards,
            masks=real_masks,
        )

        cfg = dict(get_config())
        cfg['native_reward_singletask'] = {native_reward}
        gcd = ds.GCDataset(raw, cfg)

        idxs = np.arange(n)
        batch = gcd.sample(n, idxs=idxs)
        real = {native_reward}
        if real:
            assert np.allclose(batch['rewards'], real_rewards), batch['rewards']
            assert np.allclose(batch['masks'], real_masks), batch['masks']
        else:
            # original hindsight formula: gc_negative=True -> rewards in {{-1, 0}}
            assert set(np.unique(batch['rewards']).tolist()) <= {{-1.0, 0.0}}, batch['rewards']
        print("OK")
    """


def test_native_reward_singletask_true_substitutes_the_real_dataset_reward():
    _run(_fake_gcdataset_body(True))


def test_native_reward_singletask_false_is_bit_identical_to_stock_gcdataset():
    """Default-off: every OTHER GCDataset config (this repo's existing GCIQL/
    measure cells) never sets `native_reward_singletask`, so the wrap must be
    a pure pass-through -- the synthetic hindsight reward, untouched."""
    _run(_fake_gcdataset_body(False))
