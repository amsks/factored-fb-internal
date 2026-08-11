"""Spec 22 -- `measure_density` (spec 18) on the ddpgbc/awr_flow actor extractions.
Contract: `PAPER/Multistep-FB/specs/22-antmaze-actor-ablation.md` sections 4/5.

Spec 18's density q-term (`_density_actor_q`, extracted from the flow branch's
inline block) lived only in the flow branch of `actor_loss`, hard-pinned by the
assert-lattice (`actor_kind == 'flow'`). This spec relaxes the lattice to
`actor_kind in ('flow', 'ddpgbc', 'awr_flow')` and plugs the SAME density q-readout
into the stock gaussian DDPG+BC extraction and the AWR-flow extraction, mirroring
the agent's three existing actor paths.

THE TRAP (spec section 0): under `measure_only` the bilinear critic is registered
dead weight and never trained. The stock ddpgbc branch normally DELEGATES to
`GCIQLAgent.actor_loss`, which would maximize that never-trained critic while the
density CE trains beside it -- a fake-NULL of exactly the 17b `measure_coef`
species. So ddpgbc/awr_flow get explicit density branches here that NEVER
delegate; the tests below prove it by construction (never-reads-critic) and by
gradient (grad-hygiene), not just by convention.

Implementation note beyond the literal spec snippet: `density_loss`'s bootstrap
action a+ ~ pi(s',g) (the TD target's policy sample) hard-coded the noise-driven
flow-actor call signature, which silently breaks under ddpgbc (GCActor returns a
distribution, not an action array -- and its `goal_encoded` positional slot would
eat the noise arg) and KeyErrors under awr_flow (no one-shot 'actor' module is
registered in that mode). Relaxing the lattice makes this reachable, so
`density_loss` now branches by `actor_kind` too, mirroring `measure_loss`'s own
pre-existing ddpgbc branch (gaussian dist.mode()/sample()) and adding an awr_flow
branch (Euler-roll actor_vf at frozen current params, the same construction as
the awr actor-loss branch's own baseline). The flow branch is untouched.

Same subprocess-jax harness as tests/test_gciql_measure_density.py: OBS_D=40,
action dim 5, d=8, B=16, hit_spec '19:22:0.4' for the cube-shaped cases; OBS_D=29
with hit_spec '0:2:0.5' for the antmaze-shape test (test 9).

Runner: `.venv/bin/python -m pytest tests/test_gciql_density_actors.py -x -q`
(NOT uv; the jax subprocess harness resolves the jax venv itself).
"""
import subprocess
import textwrap

import numpy as np
import pytest

from tests.jax_subprocess import run_jax, REPO

OBS_D = 40
D = 8   # value_latent_dim AND measure_density_dim, matches the spec-18 harness
ACT_D = 5

# Pinned pre-spec-22 baseline commit (same rationale as
# tests/test_gciql_measure_density.py's DENSITY_PRE_SPEC18_SHA -- HEAD moves forward
# on this shared branch as other sessions/tracks land commits, so a `git show HEAD`
# diff eventually always finds spec 22 already merged and false-fails forever). This
# SHA is the commit right before the spec-22 session started editing gciql_flow.py:
# it has measure_density (spec 18) and everything through spec 21, but not the
# `_density_actor_q` helper or the ddpgbc/awr_flow density branches.
PRE_SPEC22_SHA = "0332ba0b8f99dc65585098e781655e16996561a2"


def _baseline_reachable(sha):
    r = subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"],
                        cwd=str(REPO), capture_output=True, text=True)
    return r.returncode == 0

COMMON = f"""
    import jax, jax.numpy as jnp, numpy as np, flax
    from gciql_flow import GCIQLFlowAgent, get_config, _parse_hit_spec

    OBS_D = {OBS_D}
    D = {D}
    ACT_D = {ACT_D}
    EX_OBS = jnp.zeros((1, OBS_D), jnp.float32)
    EX_ACT = jnp.zeros((1, ACT_D), jnp.float32)

    def base_cfg(**kw):
        # the spec's base cell: measure-only, bilinear, density. Default actor_kind is
        # 'flow' (actor_q_center required True); tests override actor_kind (and
        # actor_q_center, for awr_flow) explicitly via kw.
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = D
        cfg["measure_only"] = True
        cfg["measure_density"] = True
        cfg["measure_density_dim"] = D
        cfg["measure_hit_spec"] = "19:22:0.4"
        cfg["actor_kind"] = "flow"
        cfg["actor_q_center"] = True
        cfg.update(kw)
        return cfg

    def make_batch(B=16, seed=0, obs_d=OBS_D, act_d=ACT_D):
        rng = np.random.default_rng(seed)
        obs = rng.normal(size=(B, obs_d)).astype(np.float32)
        nobs = rng.normal(size=(B, obs_d)).astype(np.float32)
        batch = dict(
            observations=obs,
            next_observations=nobs,
            actions=np.clip(rng.normal(size=(B, act_d)), -1, 1).astype(np.float32),
            value_goals=rng.normal(size=(B, obs_d)).astype(np.float32),
            actor_goals=rng.normal(size=(B, obs_d)).astype(np.float32),
            rewards=np.zeros(B, np.float32),
            masks=np.ones(B, np.float32),
        )
        return batch

    def agent_for(cfg, seed=0):
        return GCIQLFlowAgent.create(seed, EX_OBS, EX_ACT, cfg)

    def total_loss_of(agent, batch, seed=7):
        loss, info = agent.total_loss(
            {{k: jnp.asarray(v) for k, v in batch.items()}},
            agent.network.params, jax.random.PRNGKey(seed))
        return float(loss), info

    def gnorm(tree, key):
        leaves = jax.tree_util.tree_leaves(tree[key])
        return float(sum(float(jnp.sum(jnp.abs(l))) for l in leaves)) if leaves else 0.0

    def density_actor_q_ref(agent, cfg, nobs, obs, actions, goals):
        # numpy/direct-module reference for _density_actor_q -- rebuilt from DIRECT
        # module calls (density_f/density_b, _raw_block_cols/_density_queries/
        # _delta_phi), NOT by calling the helper itself.
        blocks = _parse_hit_spec(cfg['measure_hit_spec'])
        u_raw = GCIQLFlowAgent._raw_block_cols(nobs, blocks)
        u_sc = GCIQLFlowAgent._density_queries(nobs, blocks)
        kg = GCIQLFlowAgent._delta_phi(goals, u_raw, blocks)
        b_emb = agent.network.select('density_b')(u_sc)
        f_emb = agent.network.select('density_f')(jnp.concatenate([obs, actions, goals], -1))
        w = jax.nn.softmax(jnp.einsum('hbd,md->hbm', f_emb, b_emb) / cfg['measure_density_temp'], -1)
        q1, q2 = (w * kg[None]).sum(-1)
        return q1, q2
"""


def _run(body, **kw):
    code = textwrap.dedent(COMMON) + textwrap.dedent(body)
    res = run_jax(code, **kw)
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    return res


def _load_baseline_snippet(sha):
    """Snippet (to be embedded in a _run body) that loads the pinned pre-spec-22
    gciql_flow.py as module `baseline`, asserting it predates the spec-22 helper."""
    return f"""
    import subprocess, importlib.util, tempfile, os

    baseline_src = subprocess.run(
        ["git", "show", "{sha}:tools/wandb_mode_shim/gciql_flow.py"],
        cwd={str(REPO)!r}, capture_output=True, text=True, check=True).stdout
    assert "_density_actor_q" not in baseline_src, (
        "pinned baseline {sha} already has the spec-22 helper -- the pin has gone "
        "stale (e.g. history was rewritten); re-pin PRE_SPEC22_SHA to a commit that "
        "genuinely predates spec 22 (see `git log --oneline -- "
        "tools/wandb_mode_shim/gciql_flow.py`)")
    assert "measure_density" in baseline_src, (
        "pinned baseline {sha} predates spec 18 entirely -- re-pin to a commit "
        "between spec 18 and spec 22")
    baseline_path = os.path.join(tempfile.mkdtemp(), "gciql_flow_baseline.py")
    with open(baseline_path, "w") as f:
        f.write(baseline_src)
    spec = importlib.util.spec_from_file_location("gciql_flow_baseline", baseline_path)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    """


# --------------------------------------------------------------------------
# 1. flow path bit-identical after the _density_actor_q extraction
# --------------------------------------------------------------------------
def test_flow_path_bit_identical_after_refactor():
    """density+flow: param tree AND total_loss bit-identical, on a fixed batch, to
    the pinned pre-spec-22 commit (the flow branch's inline density block extracted
    verbatim into `_density_actor_q`, called from the SAME `else:` arm -- see
    gciql_flow.py's actor_loss). This is the load-bearing half of the refactor's
    correctness proof: the ddpgbc/awr_flow branches are NEW code with their own
    numpy-reference tests, but the flow path must not move at all."""
    if not _baseline_reachable(PRE_SPEC22_SHA):
        pytest.skip(f"pinned baseline commit {PRE_SPEC22_SHA} is not reachable in "
                    "this checkout (e.g. a shallow clone) -- cannot run the "
                    "bit-identity diff against it")
    _run(_load_baseline_snippet(PRE_SPEC22_SHA) + """
    def dens_cfg(get_config_fn):
        cfg = dict(get_config_fn())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = D
        cfg["measure_only"] = True
        cfg["actor_q_center"] = True
        cfg["measure_density"] = True
        cfg["measure_density_dim"] = D
        cfg["measure_hit_spec"] = "19:22:0.4"
        return cfg

    a_base = baseline.GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, dens_cfg(baseline.get_config))
    a1 = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, dens_cfg(get_config))

    t_base = jax.tree_util.tree_leaves(a_base.network.params)
    t1 = jax.tree_util.tree_leaves(a1.network.params)
    assert len(t_base) == len(t1), (len(t_base), len(t1))
    assert all((x == y).all() for x, y in zip(t_base, t1)), \\
        "flow+density param tree moved across the _density_actor_q extraction"

    b = make_batch()
    bj = {k: jnp.asarray(v) for k, v in b.items()}
    l_base, info_base = a_base.total_loss(bj, a_base.network.params, jax.random.PRNGKey(7))
    l1, info1 = a1.total_loss(bj, a1.network.params, jax.random.PRNGKey(7))
    # EXACT equality, not np.allclose -- the extraction changed no math and no op
    # order, so the compiled computation is bit-identical, not merely close.
    # Checked both as raw jax float32 scalars (== ) and at the bit-pattern level
    # (view as uint32) -- the latter would catch a same-printed-value-different-
    # bits false pass that a naive float() comparison could theoretically hide.
    assert float(l_base) == float(l1), (float(l_base), float(l1))
    assert int(np.asarray(l_base).view(np.uint32)) == int(np.asarray(l1).view(np.uint32)), \\
        "loss differs at the bit-pattern level"
    for key in ('actor/q_mean', 'actor/bc_flow_loss', 'actor/distill_loss',
                'measure/density_ce', 'measure/density_ess'):
        assert float(info_base[key]) == float(info1[key]), \\
            (key, float(info_base[key]), float(info1[key]))
        assert int(np.asarray(info_base[key]).view(np.uint32)) == int(np.asarray(info1[key]).view(np.uint32)), \\
            (key, "differs at the bit-pattern level")
    print("OK")
    """)


# --------------------------------------------------------------------------
# 2. stock ddpgbc/awr_flow (density OFF) bit-identical -- delegation/rng untouched
# --------------------------------------------------------------------------
def test_stock_kinds_bit_identical_without_density():
    """Plain GCIQL-flow configs with actor_kind='ddpgbc' / 'awr_flow' and NO density
    knob: param tree AND total_loss bit-identical to the pinned pre-spec-22 commit
    for BOTH kinds -- proves the new `if measure_density:` guards ahead of the
    ddpgbc delegation and inside the awr_flow branch are true no-ops off-knob (the
    delegation and rng draw order are untouched)."""
    if not _baseline_reachable(PRE_SPEC22_SHA):
        pytest.skip(f"pinned baseline commit {PRE_SPEC22_SHA} is not reachable in "
                    "this checkout (e.g. a shallow clone) -- cannot run the "
                    "bit-identity diff against it")
    _run(_load_baseline_snippet(PRE_SPEC22_SHA) + """
    def stock_cfg(get_config_fn, kind):
        cfg = dict(get_config_fn())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = D
        cfg["actor_kind"] = kind
        return cfg

    b = make_batch()
    bj = {k: jnp.asarray(v) for k, v in b.items()}
    for kind in ("ddpgbc", "awr_flow"):
        a_base = baseline.GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, stock_cfg(baseline.get_config, kind))
        a1 = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, stock_cfg(get_config, kind))

        t_base = jax.tree_util.tree_leaves(a_base.network.params)
        t1 = jax.tree_util.tree_leaves(a1.network.params)
        assert len(t_base) == len(t1), (kind, len(t_base), len(t1))
        assert all((x == y).all() for x, y in zip(t_base, t1)), (kind, "param tree moved")

        l_base, _ = a_base.total_loss(bj, a_base.network.params, jax.random.PRNGKey(7))
        l1, _ = a1.total_loss(bj, a1.network.params, jax.random.PRNGKey(7))
        # EXACT equality (both value and bit pattern), not np.allclose.
        assert float(l_base) == float(l1), (kind, float(l_base), float(l1))
        assert int(np.asarray(l_base).view(np.uint32)) == int(np.asarray(l1).view(np.uint32)), \\
            (kind, "loss differs at the bit-pattern level")
    print("OK")
    """)


# --------------------------------------------------------------------------
# 3. ddpgbc density numpy reference
# --------------------------------------------------------------------------
def test_ddpgbc_density_numpy_reference():
    """Rebuild dist-mode actions, w, kg, q, the CENTERED scale, q_loss and
    alpha*BC from DIRECT module calls (atol 1e-5) -- proves actor_loss's ddpgbc
    density branch computes exactly the contract's formula, not the stock
    GCIQLAgent.actor_loss delegation (which would read a different, untrained
    critic entirely and produce a different number)."""
    _run("""
    cfg = base_cfg(actor_kind='ddpgbc', actor_q_center=True)
    a = agent_for(cfg)
    b = make_batch(B=6, seed=11)
    bj = {k: jnp.asarray(v) for k, v in b.items()}
    rng = jax.random.PRNGKey(5)

    loss, info = a.actor_loss(bj, a.network.params, rng)

    obs, goals, actions = bj['observations'], bj['actor_goals'], bj['actions']
    dist = a.network.select('actor')(obs, goals, params=a.network.params)
    q_actions = jnp.clip(dist.mode() if cfg['const_std'] else dist.sample(seed=rng), -1, 1)
    q1, q2 = density_actor_q_ref(a, cfg, bj['next_observations'], obs, q_actions, goals)
    q = jnp.minimum(q1, q2) if cfg['double_q'] else q1
    q_scale = jnp.abs(q - q.mean()).mean()
    q_loss_ref = -q.mean() / (q_scale + 1e-6)
    log_prob = dist.log_prob(actions)
    bc_loss_ref = -(cfg['alpha'] * log_prob).mean()
    actor_loss_ref = q_loss_ref + bc_loss_ref

    assert np.allclose(float(actor_loss_ref), float(loss), atol=1e-5), (float(actor_loss_ref), float(loss))
    assert np.allclose(float(q_loss_ref), float(info['q_loss']), atol=1e-5)
    assert np.allclose(float(bc_loss_ref), float(info['bc_loss']), atol=1e-5)
    assert np.allclose(float(q.mean()), float(info['q_mean']), atol=1e-5)
    assert np.allclose(float(log_prob.mean()), float(info['bc_log_prob']), atol=1e-5)
    print("OK")
    """)


# --------------------------------------------------------------------------
# 4. ddpgbc density grad hygiene
# --------------------------------------------------------------------------
def test_ddpgbc_density_grad_hygiene():
    """jax.grad(actor_loss): ~=0 over density_f/density_b/target_density_f/
    target_density_b/critic/value leaves (the density branch never touches the
    critic or value networks); >0 over the gaussian actor. jax.grad(density_loss):
    ~=0 over the actor (ddpgbc's bootstrap action is stop-gradded)."""
    _run("""
    cfg = base_cfg(actor_kind='ddpgbc', actor_q_center=True)
    a = agent_for(cfg)
    b = {k: jnp.asarray(v) for k, v in make_batch().items()}
    rng = jax.random.PRNGKey(3)

    def actor_loss_fn(params):
        loss, _ = a.actor_loss(b, params, rng)
        return loss
    grads_a = jax.grad(actor_loss_fn)(a.network.params)
    for key in ('modules_density_f', 'modules_density_b', 'modules_target_density_f',
                'modules_target_density_b', 'modules_critic', 'modules_value'):
        assert gnorm(grads_a, key) < 1e-9, (key, gnorm(grads_a, key))
    assert gnorm(grads_a, 'modules_actor') > 1e-8, gnorm(grads_a, 'modules_actor')

    def density_loss_fn(params):
        loss, _ = a.density_loss(b, params, rng)
        return loss
    grads_d = jax.grad(density_loss_fn)(a.network.params)
    assert gnorm(grads_d, 'modules_actor') < 1e-9, gnorm(grads_d, 'modules_actor')
    assert gnorm(grads_d, 'modules_density_f') > 1e-8
    assert gnorm(grads_d, 'modules_density_b') > 1e-8
    print("OK")
    """)


# --------------------------------------------------------------------------
# 5. ddpgbc density NEVER reads the dead critic
# --------------------------------------------------------------------------
def test_ddpgbc_density_never_reads_critic():
    """THE delegation-fake-NULL discriminator (spec 22 sec 0/2).

    Every module call in actor_loss -- ddpgbc's included -- reads the critic via
    `self.network.select('critic')(...)` with NO `params=` kwarg, so it ALWAYS
    resolves to `self.network.params` (TrainState.__call__'s own default), never
    to the `grad_params` argument actor_loss receives. That means perturbing
    `grad_params['modules_critic']` (a weaker version of this test used
    initially) is a NULL test: it would pass identically whether or not
    delegation to GCIQLAgent.actor_loss were still active, because delegation's
    own critic read (third_party/ogbench/impls/agents/gciql.py:101,
    `self.network.select('critic')(obs, goals, q_actions)`) ALSO ignores
    grad_params. It proves nothing about which code path ran.

    The test that actually discriminates has to perturb the critic params the
    agent's network itself resolves to -- i.e. rebuild the agent with a
    perturbed `network.params['modules_critic']` (via `network.replace`, not
    `grad_params`) and recompute actor_loss on THAT agent. If delegation were
    still in effect, this WOULD move q1/q2/q_mean/actor_loss (the delegated
    branch reads exactly this critic). Under the shipped density branch, it
    provably does not -- to the last bit, not just zero-gradient (a
    dead-but-read critic passed through a stop_gradient would still pass the
    grad-hygiene test but move the LOSS here; this test would catch that,
    the old grad_params version could not).

    Counterfactual half: separately compute what `GCIQLAgent.actor_loss` (the
    stock delegation this branch replaces) would have returned for the exact
    same dist/q_actions on this agent -- it reads the SAME perturbed critic and
    is therefore provably different from our (unperturbed) q. This nails down
    that "unaffected by the critic" is not a coincidence of the numbers
    involved, but because the density branch computes q from a disjoint set of
    modules entirely."""
    _run("""
    from agents.gciql import GCIQLAgent

    cfg = base_cfg(actor_kind='ddpgbc', actor_q_center=True)
    a = agent_for(cfg)
    b = {k: jnp.asarray(v) for k, v in make_batch().items()}
    rng = jax.random.PRNGKey(9)

    loss1, info1 = a.actor_loss(b, a.network.params, rng)

    # Perturb the CRITIC AS STORED ON THE AGENT (what any code path -- delegated
    # or not -- actually reads for it), not grad_params.
    perturbed_params = dict(a.network.params)
    perturbed_params['modules_critic'] = jax.tree_util.tree_map(
        lambda x: x + 1e3, perturbed_params['modules_critic'])
    a2 = a.replace(network=a.network.replace(params=perturbed_params))
    loss2, info2 = a2.actor_loss(b, a2.network.params, rng)

    assert float(loss1) == float(loss2), (float(loss1), float(loss2))
    assert float(info1['q_mean']) == float(info2['q_mean'])
    assert float(info1['q_loss']) == float(info2['q_loss'])

    # Counterfactual: the STOCK delegation this branch replaces DOES move under
    # the same critic perturbation (proving the perturbation is actually live
    # and would have been visible had delegation still been in effect).
    delegated_loss_orig, delegated_info_orig = GCIQLAgent.actor_loss(a, b, a.network.params, rng)
    delegated_loss_pert, delegated_info_pert = GCIQLAgent.actor_loss(a2, b, a2.network.params, rng)
    assert float(delegated_info_orig['q_mean']) != float(delegated_info_pert['q_mean']), \\
        "test setup bug: the +1e3 critic perturbation must move the DELEGATED q_mean " \\
        "(else it cannot serve as a discriminator against delegation)"
    assert float(delegated_loss_orig) != float(delegated_loss_pert)

    # And the shipped density branch's own q_mean does not coincide with either
    # delegated value -- it is reading an entirely different set of modules.
    assert float(info1['q_mean']) != float(delegated_info_orig['q_mean'])
    print("OK")
    """)


# --------------------------------------------------------------------------
# 6. awr density numpy reference
# --------------------------------------------------------------------------
def test_awr_density_numpy_reference():
    """adv = q(a_data) - q(a_bar) with a_bar from a HAND-ROLLED Euler integration
    of actor_vf (mirrors _flow_rollout exactly, at the SAME frozen params used as
    grad_params in this test); exp_a cap at 100; the weighted flow-matching loss
    matches numpy, atol 1e-5. Proves the stock 'value' network is never read (the
    reference never calls it either) and the baseline really is the one-sample
    E_pi[q] the contract specifies."""
    _run("""
    cfg = base_cfg(actor_kind='awr_flow', actor_q_center=False)
    a = agent_for(cfg)
    b = make_batch(B=6, seed=21)
    bj = {k: jnp.asarray(v) for k, v in b.items()}
    rng = jax.random.PRNGKey(4)

    loss, info = a.actor_loss(bj, a.network.params, rng)

    obs, goals, actions = bj['observations'], bj['actor_goals'], bj['actions']
    nobs = bj['next_observations']
    rng2 = rng
    q1, q2 = density_actor_q_ref(a, cfg, nobs, obs, actions, goals)
    q = jnp.minimum(q1, q2)
    vf_obs = jnp.concatenate([obs, goals], axis=-1)
    rng2, v_rng = jax.random.split(rng2)
    noise_v = jax.random.normal(v_rng, actions.shape)
    fs = cfg['flow_steps']
    a_bar = noise_v
    for i in range(fs):
        t_i = jnp.ones((actions.shape[0], 1)) * (i / fs)
        vel = a.network.select('actor_vf')(vf_obs, a_bar, t_i)   # frozen params (default)
        a_bar = a_bar + vel / fs
    a_bar = jnp.clip(a_bar, -1.0, 1.0)
    v1, v2 = density_actor_q_ref(a, cfg, nobs, obs, a_bar, goals)
    adv_ref = q - jnp.minimum(v1, v2)
    exp_a_ref = jnp.minimum(jnp.exp(adv_ref * cfg['awr_alpha']), 100.0)

    rng2, x0_rng, t_rng = jax.random.split(rng2, 3)
    x_0 = jax.random.normal(x0_rng, actions.shape)
    t = jax.random.uniform(t_rng, (actions.shape[0], 1))
    x_t = (1 - t) * x_0 + t * actions
    vel = actions - x_0
    pred = a.network.select('actor_vf')(vf_obs, x_t, t, params=a.network.params)
    fm_ref = jnp.mean((pred - vel) ** 2, axis=-1)
    actor_loss_ref = jnp.mean(exp_a_ref * fm_ref)

    assert np.allclose(float(actor_loss_ref), float(loss), atol=1e-5), (float(actor_loss_ref), float(loss))
    assert np.allclose(float(adv_ref.mean()), float(info['adv']), atol=1e-5)
    assert np.allclose(float(fm_ref.mean()), float(info['flow_bc']), atol=1e-5)
    assert np.allclose(float(exp_a_ref.mean()), float(info['exp_a']), atol=1e-5)
    print("OK")
    """)


# --------------------------------------------------------------------------
# 7. awr density grad hygiene
# --------------------------------------------------------------------------
def test_awr_density_grad_hygiene():
    """jax.grad(actor_loss): ~=0 over density_f/density_b/target towers/critic/
    value (the q/baseline/adv chain reads density towers at frozen params AND is
    itself stop-gradded via the sg'd rollout params -- no gradient reaches
    grad_params through q, v, or adv at all); >0 over actor_vf ONLY (the
    flow-matching BC term, weighted by the detached exp_a)."""
    _run("""
    cfg = base_cfg(actor_kind='awr_flow', actor_q_center=False)
    a = agent_for(cfg)
    b = {k: jnp.asarray(v) for k, v in make_batch().items()}
    rng = jax.random.PRNGKey(6)

    def actor_loss_fn(params):
        loss, _ = a.actor_loss(b, params, rng)
        return loss
    grads_a = jax.grad(actor_loss_fn)(a.network.params)
    for key in ('modules_density_f', 'modules_density_b', 'modules_target_density_f',
                'modules_target_density_b', 'modules_critic', 'modules_value'):
        assert gnorm(grads_a, key) < 1e-9, (key, gnorm(grads_a, key))
    assert gnorm(grads_a, 'modules_actor_vf') > 1e-8, gnorm(grads_a, 'modules_actor_vf')
    print("OK")
    """)


# --------------------------------------------------------------------------
# 8. assert lattice
# --------------------------------------------------------------------------
def test_lattice_spec22():
    """Accepts flow(+center), ddpgbc(+center), awr_flow(-center). Refuses:
    ddpgbc without actor_q_center; awr_flow WITH actor_q_center; behavior+ddpgbc;
    actor_cond='z' with any kind; actor_bc_only+ddpgbc. All pre-existing spec-18
    refusals still fire (unaffected by the relaxation)."""
    _run("""
    def refuses(**kw):
        try:
            agent_for(base_cfg(**kw))
        except AssertionError:
            return True
        return False

    # accepted lattice points
    assert not refuses()                                                       # flow, center=True (default)
    assert not refuses(actor_kind='ddpgbc', actor_q_center=True)
    assert not refuses(actor_kind='awr_flow', actor_q_center=False)

    # spec 22 refusals
    assert refuses(actor_kind='ddpgbc', actor_q_center=False)
    assert refuses(actor_kind='awr_flow', actor_q_center=True)
    assert refuses(actor_kind='ddpgbc', actor_q_center=True, measure_density_behavior=True)
    assert refuses(actor_kind='ddpgbc', actor_q_center=True, actor_cond='z')
    assert refuses(actor_kind='awr_flow', actor_q_center=False, actor_cond='z')
    assert refuses(actor_cond='z')                                              # flow + actor_cond='z'
    assert refuses(actor_kind='ddpgbc', actor_q_center=True, actor_bc_only=True)
    assert refuses(actor_kind='bogus_kind')

    # pre-existing spec-18 refusals, unaffected by the relaxation
    assert refuses(measure_only=False)
    assert refuses(measure_delta=True)
    assert refuses(measure_delta_partition=True)
    assert refuses(measure_first_passage=True)
    assert refuses(measure_density_temp=0.0)
    assert refuses(measure_delta_protos='/tmp/does_not_matter.npy')
    assert refuses(warm_start_ckpt='/nonexistent/does/not/exist*')
    assert refuses(measure_ortho_coef=1.0)
    assert refuses(value_arch='dueling')
    assert refuses(task_prior_critic_mix=True)
    print("OK")
    """)


# --------------------------------------------------------------------------
# 9. antmaze hit-spec shape ('0:2:0.5' on OBS_D=29)
# --------------------------------------------------------------------------
def test_antmaze_hit_spec_shape():
    """The antmaze hit spec '0:2:0.5' (ant obs = [qpos(15); qvel(14)], xy =
    qpos[:2] = obs cols 0-1; eps = OGBench's own ant goal tolerance 0.5) on
    OBS_D=29: `_density_queries`/`_raw_block_cols`/`_delta_phi` slice cols 0:2 and
    scale by 0.5; the kernel row i peaks at its own diagonal j=i (self-coverage) on
    synthetic xy data -- exercised at the static-method level (no agent needed),
    the same primitives `_density_actor_q`/`density_loss` consume for any hit_spec."""
    _run("""
    OBS_D_AM = 29
    blocks = _parse_hit_spec('0:2:0.5')
    assert blocks == ((0, 2, 0.5),), blocks

    rng = np.random.default_rng(5)
    B = 7
    nobs = rng.normal(size=(B, OBS_D_AM)).astype(np.float32)
    nobs_j = jnp.asarray(nobs)

    u_sc = np.asarray(GCIQLFlowAgent._density_queries(nobs_j, blocks))
    assert u_sc.shape == (B, 2)
    assert np.allclose(u_sc, nobs[:, 0:2] / 0.5, atol=1e-6)

    u_raw = GCIQLFlowAgent._raw_block_cols(nobs_j, blocks)
    assert np.asarray(u_raw).shape == (B, 2)
    assert np.allclose(np.asarray(u_raw), nobs[:, 0:2], atol=1e-6)

    k = np.asarray(GCIQLFlowAgent._delta_phi(nobs_j, u_raw, blocks))
    assert k.shape == (B, B)
    assert np.allclose(k.sum(-1), 1.0, atol=1e-5)
    assert (np.argmax(k, axis=-1) == np.arange(B)).all(), np.argmax(k, axis=-1)
    print("OK")
    """)


# --------------------------------------------------------------------------
# 10. end-to-end updates, both new actor kinds
# --------------------------------------------------------------------------
def test_end_to_end_updates_both_kinds():
    """30 update() steps for density+ddpgbc and density+awr_flow: finite losses
    each step (actor and measure), w on the simplex, no NaN in params, target
    towers move (tau soft-update, not frozen at init)."""
    _run("""
    for kind, center in (('ddpgbc', True), ('awr_flow', False)):
        cfg = base_cfg(actor_kind=kind, actor_q_center=center)
        a = agent_for(cfg)
        p0 = a.network.params
        for step in range(30):
            b = {k: jnp.asarray(v) for k, v in make_batch(seed=step).items()}
            a, info = a.update(b)
            assert np.isfinite(float(info['actor/actor_loss'])), (kind, step, info['actor/actor_loss'])
            assert np.isfinite(float(info['measure/density_ce'])), (kind, step)

        leaves = jax.tree_util.tree_leaves(a.network.params)
        assert all(np.isfinite(np.asarray(l)).all() for l in leaves), (kind, "NaN in params")

        b = {k: jnp.asarray(v) for k, v in make_batch(seed=99).items()}
        blocks = _parse_hit_spec(cfg['measure_hit_spec'])
        u_sc = GCIQLFlowAgent._density_queries(b['next_observations'], blocks)
        b_emb = a.network.select('density_b')(u_sc)
        f_emb = a.network.select('density_f')(
            jnp.concatenate([b['observations'], b['actions'], b['value_goals']], -1))
        logits = jnp.einsum('hbd,md->hbm', f_emb, b_emb) / cfg['measure_density_temp']
        w = np.asarray(jax.nn.softmax(logits, -1))
        assert np.allclose(w.sum(-1), 1.0, atol=1e-4), kind
        assert (w >= -1e-6).all(), kind

        p_final = a.network.params
        for tower in ('density_f', 'density_b'):
            onl0 = jax.tree_util.tree_leaves(p0[f'modules_{tower}'])
            tgt_final = jax.tree_util.tree_leaves(p_final[f'modules_target_{tower}'])
            assert any(not np.allclose(np.asarray(o), np.asarray(t), atol=0)
                       for o, t in zip(onl0, tgt_final)), \\
                (kind, tower, "target tower never moved from init after 30 updates")
    print("OK")
    """)
