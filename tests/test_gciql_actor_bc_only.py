"""`actor_bc_only` -- spec 19 section 1 P0.2: the BC-only control (PAPER/Multistep-FB/
specs/19-ors-program.md). Zeroes the flow actor's q-term (q1=q2=None) REGARDLESS of
which q-term branch the rest of the config would otherwise select, reusing the SAME
"pure BC flow" code path spec 19a's `measure_density_behavior` already established
(`tools/wandb_mode_shim/gciql_flow.py`, `actor_loss`'s `if q1 is None:` guard) instead of
duplicating that logic. Isolates whether a run's success is attributable to the
critic/measure Q or just the bc_coeff cloning term -- an open debt the program notes: "no
cube-single number ... is currently attributable to the measure over the cloning term."

Same subprocess-jax harness as tests/test_gciql_measure_density.py: OBS_D=40, action
dim 5, d=8, B=16, hit_spec '19:22:0.4'.

Runner: `.venv/bin/python -m pytest tests/test_gciql_actor_bc_only.py -x -q`
(NOT uv; the jax subprocess harness resolves the jax venv itself).
"""
import textwrap

import numpy as np

from tests.jax_subprocess import run_jax, REPO

OBS_D = 40
D = 8

COMMON = f"""
    import jax, jax.numpy as jnp, numpy as np, flax
    from gciql_flow import GCIQLFlowAgent, get_config

    OBS_D = {OBS_D}
    D = {D}
    EX_OBS = jnp.zeros((1, OBS_D), jnp.float32)
    EX_ACT = jnp.zeros((1, 5), jnp.float32)

    def density_cfg(**kw):
        # the density-arm recipe the BC-only config is byte-equal to (bconly config
        # header), plus whatever the caller overrides.
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = D
        cfg["measure_only"] = True
        cfg["measure_density"] = True
        cfg["measure_density_dim"] = D
        cfg["measure_density_temp"] = 1.0
        cfg["measure_hit_spec"] = "19:22:0.4"
        cfg["actor_q_center"] = True
        cfg.update(kw)
        return cfg

    def make_batch(B=16, seed=0):
        rng = np.random.default_rng(seed)
        obs = rng.normal(size=(B, OBS_D)).astype(np.float32)
        nobs = rng.normal(size=(B, OBS_D)).astype(np.float32)
        batch = dict(
            observations=obs,
            next_observations=nobs,
            actions=np.clip(rng.normal(size=(B, 5)), -1, 1).astype(np.float32),
            value_goals=rng.normal(size=(B, OBS_D)).astype(np.float32),
            actor_goals=rng.normal(size=(B, OBS_D)).astype(np.float32),
            rewards=np.zeros(B, np.float32),
            masks=np.ones(B, np.float32),
        )
        return batch

    def agent_for(cfg, seed=0):
        return GCIQLFlowAgent.create(seed, EX_OBS, EX_ACT, cfg)
"""


def _run(body, **kw):
    code = textwrap.dedent(COMMON) + textwrap.dedent(body)
    res = run_jax(code, **kw)
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    return res


def test_default_off_bit_identical():
    """actor_bc_only=False (the default, unset or explicit): total_loss bit-identical to
    a config that never mentions the knob at all -- a brand new knob with no pre-existing
    history to diff against (there is no commit predating it to pin, unlike the
    measure_density family's tests/test_gciql_measure_density.py -- see that file's
    DENSITY_PRE_SPEC18_SHA docstring for when a git-diff baseline becomes necessary:
    once this knob has shipped and later work might regress its off-path)."""
    _run("""
    cfg_nokey = density_cfg()
    cfg_off = density_cfg(actor_bc_only=False)
    a_nokey = agent_for(cfg_nokey)
    a_off = agent_for(cfg_off)
    t0 = jax.tree_util.tree_leaves(a_nokey.network.params)
    t1 = jax.tree_util.tree_leaves(a_off.network.params)
    assert len(t0) == len(t1)
    assert all((x == y).all() for x, y in zip(t0, t1))

    b = make_batch()
    bj = {k: jnp.asarray(v) for k, v in b.items()}
    l0, info0 = a_nokey.total_loss(bj, a_nokey.network.params, jax.random.PRNGKey(7))
    l1, info1 = a_off.total_loss(bj, a_off.network.params, jax.random.PRNGKey(7))
    assert float(l0) == float(l1), (float(l0), float(l1))
    assert float(info0['actor/q_mean']) == float(info1['actor/q_mean']) != 0.0, \\
        "sanity: the density q-term should be ACTIVE (nonzero) when actor_bc_only is off"
    print("OK")
    """)


def test_actor_bc_only_zeroes_q():
    """actor_bc_only=True on the density recipe: q_mean/q_abs_mean report exactly 0 (the
    SAME signature measure_density_behavior's pure-BC mode reports), jax.grad(actor_loss)
    is zero over modules_density_f/modules_density_b/modules_critic (no q-term reads
    them), and total_loss's actor_loss reduces to the pure BC-flow terms -- while the
    density CRITIC still trains normally (measure/density_ce is finite and nonzero-
    gradient over modules_density_f), confirming the knob affects ONLY the actor, not the
    critic's own objective."""
    _run("""
    cfg_on = density_cfg(actor_bc_only=True)
    cfg_off = density_cfg(actor_bc_only=False)
    a_on = agent_for(cfg_on)
    b = make_batch(seed=3)
    bj = {k: jnp.asarray(v) for k, v in b.items()}
    rng = jax.random.PRNGKey(5)

    loss, info = a_on.actor_loss(bj, a_on.network.params, rng)
    assert float(info['q_mean']) == 0.0 and float(info['q_abs_mean']) == 0.0

    def gnorm(tree, key):
        leaves = jax.tree_util.tree_leaves(tree[key])
        return float(sum(float(jnp.sum(jnp.abs(l))) for l in leaves)) if leaves else 0.0

    def actor_loss_fn(params):
        l, _ = a_on.actor_loss(bj, params, rng)
        return l
    grads = jax.grad(actor_loss_fn)(a_on.network.params)
    for key in ("modules_density_f", "modules_density_b", "modules_critic"):
        assert gnorm(grads, key) < 1e-9, (key, gnorm(grads, key))
    assert gnorm(grads, "modules_actor_vf") > 1e-8
    assert gnorm(grads, "modules_actor") > 1e-8

    # the density CRITIC still trains: total_loss's measure/density_ce has a real
    # gradient into modules_density_f regardless of the actor-only override.
    def total_loss_fn(params):
        l, _ = a_on.total_loss(bj, params, rng)
        return l
    tgrads = jax.grad(total_loss_fn)(a_on.network.params)
    assert gnorm(tgrads, "modules_density_f") > 1e-8, \\
        "density critic should still train even though the actor ignores its q-term"

    l_off, info_off = a_on.replace(config=flax.core.FrozenDict(**cfg_off)).actor_loss(
        bj, a_on.network.params, rng)
    assert float(info_off['q_mean']) != 0.0, "control: off should have a nonzero q_mean"
    print("OK")
    """)


def test_lattice():
    """Refuses: actor_kind != 'flow' (ddpgbc/awr_flow return before the q1/q2 chain is
    ever reached, so the knob would be silently inert there); composition with
    measure_density_behavior (already q-term-free by construction -- redundant, provably
    inert). Accepts the density base config."""
    _run("""
    def refuses(**kw):
        try:
            agent_for(density_cfg(actor_bc_only=True, **kw))
        except AssertionError:
            return True
        return False
    assert refuses(actor_kind="ddpgbc")
    assert refuses(measure_density_behavior=True, actor_q_center=False)
    assert not refuses()
    print("OK")
    """)
