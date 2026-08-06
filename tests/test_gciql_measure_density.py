"""`measure_density` -- spec 18: continuous normalized successor density. Contract:
`PAPER/Multistep-FB/specs/18-density-measure.md` section 4/5.

Removes 17b's fixed K-prototype simplex head and replaces `MLP([s,a,g]) -> K logits`
by a two-tower score `f(s,a,g,u) = F(s,a,g).B(u)/tau` over arbitrary object-coordinate
queries `u`, softmax-normalized over a query set RESAMPLED FROM THE BATCH every
update (round 1: the batch's own next-state object coords, M == B, no extra iid
draws -- spec section 3, anti-scope section 8):

    w_ij   = softmax_j f(s_i, a_i, g_i, u_j)                        (online, [2,B,M])
    k_ij   = softmax_j( -0.5 sum_b ||(x'_i,b - u_j,b)/eps_b||^2 )  = _delta_phi(nobs, u_raw)
    wbar_ij = softmax_j f_bar(s'_i, a+_i, g_i, u_j)                  (target towers)
    t_ij   = sg[ (1-gamma)*k_ij + gamma*wbar_ij ]                    (rows on the simplex)
    L      = -E_i[ sum_j t_ij * log_softmax(logits)_ij ]

No prototype .npy, no protos field -- `_density_queries`/`_raw_block_cols` build the
query support fresh from `next_observations` every call. `measure_density_hard`
ablates the immediate term to a one-hot at the row's own successor (k = I).

Same subprocess-jax harness as tests/test_gciql_measure_delta_partition.py: OBS_D=40,
action dim 5, d=8, B=16, hit_spec '19:22:0.4'. Both towers (density_f/density_b) get
their own target towers + soft-update branch in `update()` (mirrors delta_head's).

The spec's 10 tests plus `test_target_stopgrad_is_load_bearing` (round-2 review
item 3: makes the target-tower stop_gradient invariant testable via a
deliberately-live reconstruction, since density_loss's own target-tower calls
already never touch grad_params regardless of that wrapper -- see its
docstring). `test_default_off_bit_identical` and `test_hard_immediate_ablation`
were also strengthened per round-2 review (a `git show HEAD` baseline diff and
a hard-vs-soft cross-check, respectively) -- see their docstrings for exactly
what each does and does not prove.

Runner: `.venv/bin/python -m pytest tests/test_gciql_measure_density.py -x -q`
(NOT uv; the jax subprocess harness resolves the jax venv itself).
"""
import subprocess
import textwrap

import numpy as np
import pytest

from tests.jax_subprocess import run_jax, REPO

OBS_D = 40
D = 8   # value_latent_dim AND measure_density_dim, matches the spec's harness

# Pinned pre-measure_density baseline commit for test_default_off_bit_identical --
# NOT "HEAD". HEAD moves forward on this shared branch as other sessions/tracks land
# commits (verified in practice: it moved from c5348cf to bf70d20 mid-way through the
# spec-19b session that added this pin), so a `git show HEAD` baseline diff eventually
# always finds measure_density already merged and false-fails forever once spec 18
# itself is committed -- not just transiently while a PR is in flight. A pinned SHA is
# immune to that: it names a specific, permanent point in history that is known (and
# re-verified below) to predate measure_density, regardless of where HEAD is today.
# Re-pin only if this SHA becomes unreachable (see _baseline_reachable / the skip path).
DENSITY_PRE_SPEC18_SHA = "370938f"


def _baseline_reachable(sha):
    r = subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"],
                        cwd=str(REPO), capture_output=True, text=True)
    return r.returncode == 0

COMMON = f"""
    import jax, jax.numpy as jnp, numpy as np, flax
    from gciql_flow import GCIQLFlowAgent, get_config, _parse_hit_spec

    OBS_D = {OBS_D}
    D = {D}
    EX_OBS = jnp.zeros((1, OBS_D), jnp.float32)
    EX_ACT = jnp.zeros((1, 5), jnp.float32)

    def base_cfg(**kw):
        # the spec's base cell: measure-only, bilinear, density, actor_q_center required.
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = D
        cfg["measure_only"] = True
        cfg["actor_q_center"] = True
        cfg["measure_density"] = True
        cfg["measure_density_dim"] = D
        cfg["measure_hit_spec"] = "19:22:0.4"
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

    def total_loss_of(agent, batch, seed=7):
        loss, info = agent.total_loss(
            {{k: jnp.asarray(v) for k, v in batch.items()}},
            agent.network.params, jax.random.PRNGKey(seed))
        return float(loss), info

    def density_ref(agent, cfg, bj, rng, hard):
        # numpy/direct-module reference for density_loss's own computation, rebuilt
        # from DIRECT module calls at the agent's own params (mirrors density_loss's
        # code exactly, including the bootstrap noise drawn from the SAME rng key).
        obs, nobs = bj['observations'], bj['next_observations']
        acts, g = bj['actions'], bj['value_goals']
        gamma, tau = cfg['discount'], cfg['measure_density_temp']
        blocks = _parse_hit_spec(cfg['measure_hit_spec'])
        u_raw = GCIQLFlowAgent._raw_block_cols(nobs, blocks)
        u_sc = GCIQLFlowAgent._density_queries(nobs, blocks)
        b_emb = agent.network.select('density_b')(u_sc, params=agent.network.params)
        f_emb = agent.network.select('density_f')(
            jnp.concatenate([obs, acts, g], -1), params=agent.network.params)
        logits = jnp.einsum('hbd,md->hbm', f_emb, b_emb) / tau
        logp = np.asarray(jax.nn.log_softmax(logits, -1))
        noise = jax.random.normal(rng, acts.shape)
        a_plus = jax.lax.stop_gradient(agent.network.select('actor')(nobs, g, noise))
        tb = agent.network.select('target_density_b')(u_sc)
        tf = agent.network.select('target_density_f')(jnp.concatenate([nobs, a_plus, g], -1))
        w_bar = np.asarray(jax.nn.softmax(jnp.einsum('hbd,md->hbm', tf, tb) / tau, -1))
        if hard:
            k = np.eye(nobs.shape[0])
        else:
            k = np.asarray(GCIQLFlowAgent._delta_phi(nobs, u_raw, blocks))
        t = (1.0 - gamma) * k[None] + gamma * w_bar
        ce_ref = -np.mean(np.sum(t * logp, axis=-1))
        return ce_ref, t, np.asarray(logits)

    def ess_ref_stats(logits):
        # numpy reference for the G2 gate instruments (round-2 review item 1):
        # mean/median/per-head ESS over the SAME online w the loss computes.
        w = np.exp(logits - logits.max(axis=-1, keepdims=True))
        w = w / w.sum(axis=-1, keepdims=True)
        ess_rows = 1.0 / np.sum(w ** 2, axis=-1)          # [2, B]
        return dict(mean=float(np.mean(ess_rows)), median=float(np.median(ess_rows)),
                    h0=float(np.mean(ess_rows[0])), h1=float(np.mean(ess_rows[1])))
"""


def _run(body, **kw):
    code = textwrap.dedent(COMMON) + textwrap.dedent(body)
    res = run_jax(code, **kw)
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    return res


# --------------------------------------------------------------------------
# 1. default-off bit-identical
# --------------------------------------------------------------------------
def test_default_off_bit_identical():
    """measure_density=False (the default, whether unset or explicit): param tree
    byte-identical to a no-knob config, no density_f/density_b/target_density_f/
    target_density_b modules registered, and total_loss bit-identical on a fixed
    batch.

    WHAT THIS PROVES: turning measure_density off today reproduces, byte-for-byte
    (param tree AND total_loss), the code as it stood at the pinned commit
    DENSITY_PRE_SPEC18_SHA -- a real historical baseline that predates spec 18 (and
    therefore every later spec 19a/19b/etc. change too). That catches a genuine
    off-path regression (wrong param count, a changed default, reordered rng
    splits) introduced by ANY change since that commit, because the two sides run
    different code -- comparing two knob-off configs built from the SAME current
    file (kept below as a cheap redundant sanity check, not the load-bearing half)
    cannot detect that: if a later edit broke the default path, both configs would
    break identically and stay green.

    WHAT THIS DOES NOT PROVE: that the off-path matches "whatever commit most
    recently touched this file" -- it deliberately does NOT track HEAD. Round-2
    review originally pinned this diff to `git show HEAD`, reasoning that HEAD was
    the pre-spec-18 baseline at the time the test was written; that assumption
    broke the very first time another commit landed on this shared branch (this
    file's own git log moved past it while spec 19b was being implemented in the
    SAME session), so the test started failing on code that was never touched by
    spec 18 or by whatever the "off" path is only exercised. A HEAD-relative diff
    is not a fix for the off-path being wrong; it is a fix that expires with the
    next commit. Pinning a SHA instead means: (1) it never expires as HEAD moves,
    and (2) if the pinned SHA itself is ever unreachable (a shallow clone, a
    history rewrite), the test SKIPS with a clear reason instead of either
    silently passing on a stale assumption or failing on an infrastructure
    problem unrelated to this feature -- see _baseline_reachable above."""
    if not _baseline_reachable(DENSITY_PRE_SPEC18_SHA):
        pytest.skip(f"pinned baseline commit {DENSITY_PRE_SPEC18_SHA} is not reachable in "
                    "this checkout (e.g. a shallow clone) -- cannot run the off-path "
                    "regression diff against it")
    _run("""
    import subprocess, importlib.util, tempfile, os

    baseline_src = subprocess.run(
        ["git", "show", "%s:tools/wandb_mode_shim/gciql_flow.py"],
        cwd=%r, capture_output=True, text=True, check=True).stdout
    assert "measure_density" not in baseline_src, (
        "pinned baseline %s no longer predates measure_density -- the pin has gone "
        "stale (e.g. history was rewritten); re-pin DENSITY_PRE_SPEC18_SHA to a commit "
        "that genuinely predates spec 18 (see `git log --oneline -- "
        "tools/wandb_mode_shim/gciql_flow.py`)")
    baseline_path = os.path.join(tempfile.mkdtemp(), "gciql_flow_baseline.py")
    with open(baseline_path, "w") as f:
        f.write(baseline_src)
    spec = importlib.util.spec_from_file_location("gciql_flow_baseline", baseline_path)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)

    cfg_base = dict(baseline.get_config())
    cfg_base["encoder"] = None; cfg_base["frame_stack"] = None
    a_base = baseline.GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg_base)

    cfg1 = dict(get_config()); cfg1["encoder"] = None; cfg1["frame_stack"] = None
    cfg1["measure_density"] = False
    a1 = agent_for(cfg1)
    for mod in ("modules_density_f", "modules_density_b",
                "modules_target_density_f", "modules_target_density_b"):
        assert mod not in a1.network.params, mod

    t_base = jax.tree_util.tree_leaves(a_base.network.params)
    t1 = jax.tree_util.tree_leaves(a1.network.params)
    assert len(t_base) == len(t1), (len(t_base), len(t1))
    assert all((x == y).all() for x, y in zip(t_base, t1)), \\
        "param tree differs from the pre-spec-18 HEAD baseline with the knob off"

    b = make_batch()
    bj = {kk: jnp.asarray(vv) for kk, vv in b.items()}
    l_base, _ = a_base.total_loss(bj, a_base.network.params, jax.random.PRNGKey(7))
    l1, _ = total_loss_of(a1, b)
    assert np.allclose(float(l_base), float(l1), atol=1e-6), (float(l_base), float(l1))

    # same-code sanity check kept as a cheap redundant signal (NOT the load-bearing
    # half of this test -- see docstring): a second current-code knob-off config
    # must also match bit-for-bit.
    cfg0 = dict(get_config()); cfg0["encoder"] = None; cfg0["frame_stack"] = None
    a0 = agent_for(cfg0)
    t0 = jax.tree_util.tree_leaves(a0.network.params)
    assert all((x == y).all() for x, y in zip(t0, t1))
    l0, _ = total_loss_of(a0, b)
    assert l0 == l1, (l0, l1)
    print("OK")
    """ % (DENSITY_PRE_SPEC18_SHA, str(REPO), DENSITY_PRE_SPEC18_SHA))


# --------------------------------------------------------------------------
# 2. query scaling agrees with _delta_phi's internal scaling; self-coverage
# --------------------------------------------------------------------------
def test_query_scaling_matches_delta_phi():
    """2-block spec '19:22:0.4,28:31:0.8': `_density_queries`'s eps-scaled coords
    equal `_delta_phi`'s own internal eps-scaled `xc` construction applied to the
    same rows (equal-logit construction: the two must literally use the same
    per-block divide-then-concat), and the kernel term `k = _delta_phi(nobs, u_raw,
    blocks)` (u_raw the RAW twin of the same rows) peaks on its own diagonal --
    row i's nearest query is query i itself, the free coverage guarantee `U = batch
    next-obs` gives (spec section 1)."""
    _run("""
    blocks = _parse_hit_spec("19:22:0.4,28:31:0.8")
    rng = np.random.default_rng(3)
    B = 9
    nobs = rng.normal(size=(B, OBS_D)).astype(np.float32)
    nobs_j = jnp.asarray(nobs)

    def phi_np_xc(obs, blocks):
        parts = []
        for (a, b, e) in blocks:
            parts.append(obs[:, a:b] / e)
        return np.concatenate(parts, axis=1)

    u_sc = np.asarray(GCIQLFlowAgent._density_queries(nobs_j, blocks))
    xc_ref = phi_np_xc(nobs, blocks)
    assert np.allclose(u_sc, xc_ref, atol=1e-6), np.abs(u_sc - xc_ref).max()

    u_raw = GCIQLFlowAgent._raw_block_cols(nobs_j, blocks)
    # the raw twin must NOT already be eps-scaled -- it should equal the plain
    # block slice, not u_sc.
    raw_ref = np.concatenate([nobs[:, a:b] for (a, b, e) in blocks], axis=1)
    assert np.allclose(np.asarray(u_raw), raw_ref, atol=1e-6)
    assert not np.allclose(np.asarray(u_raw), u_sc, atol=1e-3), \\
        "u_raw must be RAW obs units, not eps-scaled (the double-scale trap)"

    k = np.asarray(GCIQLFlowAgent._delta_phi(nobs_j, u_raw, blocks))
    assert k.shape == (B, B)
    assert np.allclose(k.sum(-1), 1.0, atol=1e-5)
    assert (np.argmax(k, axis=-1) == np.arange(B)).all(), np.argmax(k, axis=-1)
    print("OK")
    """)


# --------------------------------------------------------------------------
# 3. CE loss matches a numpy/direct-module reference (soft AND hard)
# --------------------------------------------------------------------------
def test_ce_matches_numpy_reference():
    """Rebuild w, w_bar, k, t, CE from DIRECT module calls at the agent's own params
    (density_ref helper), atol 1e-5, for both the soft (eps-kernel) and hard
    (one-hot) immediate term; rows of t sum to 1 in both cases. Also (round-2
    review item 1): rebuild the G2 gate instruments -- mean/median/per-head ESS
    (ess_ref_stats, over the SAME online w) -- from the reference logits and
    check them against density_loss's own info keys, atol 1e-5."""
    _run("""
    cfg = base_cfg()
    a = agent_for(cfg)
    b = make_batch(B=6, seed=11)
    bj = {k: jnp.asarray(v) for k, v in b.items()}
    rng = jax.random.PRNGKey(7)

    loss, info = a.density_loss(bj, a.network.params, rng)
    ce_ref, t, logits = density_ref(a, cfg, bj, rng, hard=False)
    assert np.allclose(ce_ref, float(loss), atol=1e-5), (ce_ref, float(loss))
    assert np.allclose(t.sum(-1), 1.0, atol=1e-5)

    ess_ref = ess_ref_stats(logits)
    assert np.allclose(ess_ref['mean'], float(info['density_ess']), atol=1e-5), \\
        (ess_ref['mean'], float(info['density_ess']))
    assert np.allclose(ess_ref['median'], float(info['density_ess_median']), atol=1e-5), \\
        (ess_ref['median'], float(info['density_ess_median']))
    assert np.allclose(ess_ref['h0'], float(info['density_ess_h0']), atol=1e-5), \\
        (ess_ref['h0'], float(info['density_ess_h0']))
    assert np.allclose(ess_ref['h1'], float(info['density_ess_h1']), atol=1e-5), \\
        (ess_ref['h1'], float(info['density_ess_h1']))
    # the median statistic must be a REAL median, not a relabeled mean: on this
    # batch the two must differ (guards against a copy-paste that ships
    # jnp.mean under the density_ess_median key).
    assert not np.allclose(ess_ref['mean'], ess_ref['median'], atol=1e-9), \\
        "test setup: mean and median coincide, this batch cannot discriminate them"

    cfg_hard = base_cfg(measure_density_hard=True)
    a_hard = agent_for(cfg_hard)
    loss_h, info_h = a_hard.density_loss(bj, a_hard.network.params, rng)
    ce_ref_h, t_h, _ = density_ref(a_hard, cfg_hard, bj, rng, hard=True)
    assert np.allclose(ce_ref_h, float(loss_h), atol=1e-5), (ce_ref_h, float(loss_h))
    assert np.allclose(t_h.sum(-1), 1.0, atol=1e-5)
    print("OK")
    """)


# --------------------------------------------------------------------------
# 4. target towers: init mirror + soft update, BOTH towers
# --------------------------------------------------------------------------
def test_target_towers_init_and_soft_update():
    """At create, target_density_f == density_f and target_density_b == density_b
    (the init mirror). Two update() steps; the SECOND target equals
    tau*online_pre + (1-tau)*target_pre elementwise for BOTH towers, where
    online_pre/target_pre are the params AS THEY WERE going into that second
    update() call (agents/gciql.py's target_update formula reads the PRE-step
    online params) -- catches a missing override branch or init mirror for either
    tower."""
    _run("""
    cfg = base_cfg()
    a = agent_for(cfg)
    p0 = a.network.params
    for tower in ("density_f", "density_b"):
        onl0 = jax.tree_util.tree_leaves(p0[f"modules_{tower}"])
        tgt0 = jax.tree_util.tree_leaves(p0[f"modules_target_{tower}"])
        assert len(onl0) == len(tgt0) and len(onl0) > 0
        assert all((np.asarray(x) == np.asarray(y)).all() for x, y in zip(onl0, tgt0)), \\
            f"target_{tower} != {tower} at init"

    b1 = {k: jnp.asarray(v) for k, v in make_batch(seed=0).items()}
    a1, info1 = a.update(b1)
    p1 = a1.network.params
    tau = cfg["tau"]

    for tower in ("density_f", "density_b"):
        onl1 = jax.tree_util.tree_leaves(p1[f"modules_{tower}"])
        tgt1 = jax.tree_util.tree_leaves(p1[f"modules_target_{tower}"])
        onl0 = jax.tree_util.tree_leaves(p0[f"modules_{tower}"])
        assert any(not np.allclose(np.asarray(o0), np.asarray(o1), atol=0)
                   for o0, o1 in zip(onl0, onl1)), \\
            f"{tower} did not move after one update() -- CE gradient is zero"

        b2 = {k: jnp.asarray(v) for k, v in make_batch(seed=1).items()}
        a2, info2 = a1.update(b2)
        p2 = a2.network.params
        tgt2 = jax.tree_util.tree_leaves(p2[f"modules_target_{tower}"])
        for onl_pre, old_pre, new in zip(onl1, tgt1, tgt2):
            expected = tau * np.asarray(onl_pre) + (1 - tau) * np.asarray(old_pre)
            assert np.allclose(np.asarray(new), expected, atol=1e-7), \\
                (tower, np.abs(np.asarray(new) - expected).max())
    print("OK")
    """)


# --------------------------------------------------------------------------
# 5. measure_only wiring in total_loss
# --------------------------------------------------------------------------
def test_measure_only_wiring():
    """With the knob on, total_loss's info dict contains a density_ce-derived key
    (measure_only calls density_loss INSTEAD of measure_loss/delta_partition_loss,
    prefixed exactly as the stock measure info is), no measure_loss/delta_ce keys,
    and no IQL value/critic keys (measure_only semantics preserved)."""
    _run("""
    cfg = base_cfg()
    a = agent_for(cfg)
    b = make_batch()
    loss, info = total_loss_of(a, b)
    assert np.isfinite(loss)
    keys = sorted(info.keys())
    assert any('density_ce' in k for k in info), keys
    assert not any('delta_ce' in k for k in info), keys
    assert not any(k.endswith('measure_loss') for k in info), keys
    assert not any(k.startswith('value/') for k in info), keys
    assert not any(k.startswith('critic/') for k in info), keys
    print("OK")
    """)


# --------------------------------------------------------------------------
# 6. grad hygiene
# --------------------------------------------------------------------------
def test_actor_grad_hygiene():
    """jax.grad(actor_loss): grad-norm over density_f/density_b/target_density_f/
    target_density_b/critic leaves ~= 0 (the q-term reads the towers at FROZEN
    params and never touches the critic); over modules_actor > 0.
    jax.grad(density_loss): grad over modules_actor ~= 0 (bootstrap action is
    stop-gradded); over modules_density_f AND modules_density_b > 0 (both are
    trained); over both target towers == 0 (soft-updated outside the loss)."""
    _run("""
    cfg = base_cfg()
    a = agent_for(cfg)
    b = {k: jnp.asarray(v) for k, v in make_batch().items()}
    rng = jax.random.PRNGKey(3)

    def gnorm(tree, key):
        leaves = jax.tree_util.tree_leaves(tree[key])
        return float(sum(float(jnp.sum(jnp.abs(l))) for l in leaves)) if leaves else 0.0

    def actor_loss_fn(params):
        loss, _ = a.actor_loss(b, params, rng)
        return loss
    grads_a = jax.grad(actor_loss_fn)(a.network.params)
    for key in ("modules_density_f", "modules_density_b",
                "modules_target_density_f", "modules_target_density_b", "modules_critic"):
        assert gnorm(grads_a, key) < 1e-9, (key, gnorm(grads_a, key))
    assert gnorm(grads_a, 'modules_actor') > 1e-8, gnorm(grads_a, 'modules_actor')

    def density_loss_fn(params):
        loss, _ = a.density_loss(b, params, rng)
        return loss
    grads_d = jax.grad(density_loss_fn)(a.network.params)
    assert gnorm(grads_d, 'modules_actor') < 1e-9, gnorm(grads_d, 'modules_actor')
    assert gnorm(grads_d, 'modules_density_f') > 1e-8, gnorm(grads_d, 'modules_density_f')
    assert gnorm(grads_d, 'modules_density_b') > 1e-8, gnorm(grads_d, 'modules_density_b')
    assert gnorm(grads_d, 'modules_target_density_f') < 1e-9, gnorm(grads_d, 'modules_target_density_f')
    assert gnorm(grads_d, 'modules_target_density_b') < 1e-9, gnorm(grads_d, 'modules_target_density_b')
    print("OK")
    """)


def test_target_stopgrad_is_load_bearing():
    """Round-2 review item 3 (bullet 1): the reviewer showed that deleting
    density_loss's `stop_gradient(t)` leaves all 10 shipped tests green. Root
    cause: `TrainState.__call__`'s OWN default (utils/flax_utils.py -- `params
    is None -> use self.params, i.e. no grad; the typical use case is to set
    params=None when you want to *stop* the gradients`) already severs
    target_density_f/target_density_b from `grad_params` in density_loss's
    actual target-tower calls, since neither call passes `params=grad_params`.
    So `test_actor_grad_hygiene`'s `grad(target)==0` checks are true whether or
    not the explicit `stop_gradient(t)` wrapper is there -- they can't detect
    its removal, because the graph never reaches grad_params through those
    calls in the first place.

    This test makes the invariant the wrapper documents actually testable by
    reconstructing t exactly as density_loss does but DELIBERATELY calling the
    target towers WITH params=grad_params (the live path stop_gradient(t) is
    defense-in-depth against -- e.g. a future refactor that shares params or
    otherwise wires target-tower gradients live). With stop_gradient(t) kept,
    grad w.r.t. those params is exactly 0, same as today; WITHOUT it (same
    reconstruction, sg removed), the grad is genuinely nonzero -- a path that
    WOULD be nonzero without the wrapper, proving it is load-bearing on the
    path it exists to guard, even though it is currently inert on
    density_loss's own (already-severed) target-tower calls."""
    _run("""
    cfg = base_cfg()
    a = agent_for(cfg)
    b = {k: jnp.asarray(v) for k, v in make_batch().items()}
    rng = jax.random.PRNGKey(3)
    obs, nobs = b['observations'], b['next_observations']
    acts, g = b['actions'], b['value_goals']
    gamma, tau = cfg['discount'], cfg['measure_density_temp']
    blocks = _parse_hit_spec(cfg['measure_hit_spec'])
    u_raw = GCIQLFlowAgent._raw_block_cols(nobs, blocks)
    u_sc = GCIQLFlowAgent._density_queries(nobs, blocks)

    def gnorm(tree, key):
        leaves = jax.tree_util.tree_leaves(tree[key])
        return float(sum(float(jnp.sum(jnp.abs(l))) for l in leaves)) if leaves else 0.0

    def loss_with_live_target(params, apply_sg):
        b_emb = a.network.select('density_b')(u_sc, params=params)
        f_emb = a.network.select('density_f')(jnp.concatenate([obs, acts, g], -1), params=params)
        logits = jnp.einsum('hbd,md->hbm', f_emb, b_emb) / tau
        logp = jax.nn.log_softmax(logits, -1)
        noise = jax.random.normal(rng, acts.shape)
        a_plus = jax.lax.stop_gradient(a.network.select('actor')(nobs, g, noise))
        # THE DELIBERATE MISTAKE: target towers called WITH params=grad_params
        # (the live path stop_gradient(t) exists to guard against). `params`
        # must be the FULL tree here, not a submodule slice -- TrainState's
        # __call__ forwards it straight to ModuleDict's apply_fn, which
        # dispatches on module name internally (flax_utils.py:90-114).
        tb = a.network.select('target_density_b')(u_sc, params=params)
        tf = a.network.select('target_density_f')(
            jnp.concatenate([nobs, a_plus, g], -1), params=params)
        w_bar = jax.nn.softmax(jnp.einsum('hbd,md->hbm', tf, tb) / tau, -1)
        k = GCIQLFlowAgent._delta_phi(nobs, u_raw, blocks)
        raw_t = (1.0 - gamma) * k[None] + gamma * w_bar
        t = jax.lax.stop_gradient(raw_t) if apply_sg else raw_t
        return -jnp.mean(jnp.sum(t * logp, axis=-1))

    grads_sg = jax.grad(lambda p: loss_with_live_target(p, True))(a.network.params)
    grads_nosg = jax.grad(lambda p: loss_with_live_target(p, False))(a.network.params)

    for key in ('modules_target_density_f', 'modules_target_density_b'):
        assert gnorm(grads_sg, key) < 1e-9, \\
            (key, gnorm(grads_sg, key), "stop_gradient(t) failed to zero a LIVE target-tower path")
        assert gnorm(grads_nosg, key) > 1e-8, \\
            f"test setup bug: {key} grad should be NONZERO without stop_gradient(t)"
    print("OK")
    """)


# --------------------------------------------------------------------------
# 7. actor q-term reads the goal's kernel row
# --------------------------------------------------------------------------
def test_actor_q_reads_goal_kernel():
    """Goals placed EXACTLY on one query row (the batch's own next_observations
    row j*): kg = _delta_phi(goals, u_raw, blocks) must peak at j*. The actor's q
    equals sum_j w_j * kg_j for a hand-computed w (direct density_f/density_b
    module calls at the SAME actor_actions actor_loss itself samples,
    reconstructed deterministically from the same rng key -- the pre-existing
    rng-splitting/one-shot-actor code this reads is untouched by spec 18)."""
    _run("""
    cfg = base_cfg()
    a = agent_for(cfg)
    blocks = _parse_hit_spec(cfg['measure_hit_spec'])
    B = 5
    jstar = 2
    b = make_batch(B=B, seed=6)
    goals = np.array(b['actor_goals'])
    goals[:, 19:22] = b['next_observations'][jstar, 19:22]
    b['actor_goals'] = goals
    bj = {k: jnp.asarray(v) for k, v in b.items()}
    rng = jax.random.PRNGKey(4)

    u_raw = GCIQLFlowAgent._raw_block_cols(bj['next_observations'], blocks)
    kg = GCIQLFlowAgent._delta_phi(jnp.asarray(goals), u_raw, blocks)
    assert (np.asarray(jnp.argmax(kg, axis=-1)) == jstar).all(), np.asarray(kg)

    _, info = a.actor_loss(bj, a.network.params, rng)

    obs, actions = bj['observations'], bj['actions']
    _, x0_rng, t_rng, n_rng, z_rng = jax.random.split(rng, 5)
    noise = jax.random.normal(n_rng, actions.shape)
    actor_actions = a.network.select('actor')(obs, jnp.asarray(goals), noise,
                                              params=a.network.params)

    u_sc = GCIQLFlowAgent._density_queries(bj['next_observations'], blocks)
    b_emb = a.network.select('density_b')(u_sc)
    f_emb = a.network.select('density_f')(
        jnp.concatenate([obs, actor_actions, jnp.asarray(goals)], -1))
    w_hand = jax.nn.softmax(
        jnp.einsum('hbd,md->hbm', f_emb, b_emb) / cfg['measure_density_temp'], -1)
    q1_hand, q2_hand = (w_hand * kg[None]).sum(-1)
    q_hand = jnp.minimum(q1_hand, q2_hand) if cfg['double_q'] else q1_hand
    assert np.isfinite(np.asarray(q_hand)).all()
    assert np.allclose(float(jnp.mean(q_hand)), float(info['q_mean']), atol=1e-4), \\
        (float(jnp.mean(q_hand)), float(info['q_mean']))
    print("OK")
    """)


# --------------------------------------------------------------------------
# 8. assert lattice
# --------------------------------------------------------------------------
def test_assert_lattice():
    """Refusals: no measure_only; with measure_delta; with measure_delta_partition;
    with measure_first_passage; actor_q_center=False; temp=0; with
    measure_delta_protos set; with warm_start_ckpt; measure_ortho_coef>0;
    value_arch='dueling' (round-2 review MEDIUM: actor_loss's dueling branch at
    ~:1027 precedes the measure_density branch, so this composition would
    silently train the density CE while the actor maximizes a never-trained
    dueling critic -- refused explicitly, though it is also already caught
    transitively by the pre-existing generic dueling lattice's `not measure_only`
    assert); task_prior_critic_mix=True (adds a scalar FP-TD leg with no
    task-conditioned critic for density to mix with). Accepts the base config."""
    _run("""
    def refuses(**kw):
        try:
            agent_for(base_cfg(**kw))
        except AssertionError:
            return True
        return False
    assert refuses(measure_only=False)
    assert refuses(measure_delta=True)
    assert refuses(measure_delta_partition=True)
    assert refuses(measure_first_passage=True)
    assert refuses(actor_q_center=False)
    assert refuses(measure_density_temp=0.0)
    assert refuses(measure_delta_protos='/tmp/does_not_matter.npy')
    assert refuses(warm_start_ckpt='/nonexistent/does/not/exist*')
    assert refuses(measure_ortho_coef=1.0)
    assert refuses(value_arch='dueling')
    assert refuses(task_prior_critic_mix=True)
    assert not refuses()
    print("OK")
    """)


# --------------------------------------------------------------------------
# 9. hard immediate-term ablation
# --------------------------------------------------------------------------
def test_hard_immediate_ablation():
    """measure_density_hard=True: k == I (one-hot at the row's own successor), t
    rows still on the simplex, CE finite, and info's density_k_self == 1.0 (the
    coverage diagnostic collapses to its ceiling by construction when hard).

    Round-2 review item 3 (bullet 2): `k == eye(B) == eye(B).T` on its own is a
    tautology about `np.eye`, not about the agent -- it can never fail and
    proves nothing was actually swapped in by the knob. This version instead
    builds a SOFT-immediate agent with the SAME seed (architecture is
    identical either way -- the knob only gates the loss, not module
    registration/init, checked explicitly) and the SAME batch/rng, and shows
    the hard and soft immediate terms actually DIFFER: k_self collapses to
    the soft kernel's own value only under the eps-kernel path, the two CE
    values disagree, and the raw [B,M] k matrices disagree elementwise --
    distinguishing the ablation from a no-op."""
    _run("""
    cfg_hard = base_cfg(measure_density_hard=True)
    cfg_soft = base_cfg(measure_density_hard=False)
    a_hard = agent_for(cfg_hard)
    a_soft = agent_for(cfg_soft)
    leaves_h = jax.tree_util.tree_leaves(a_hard.network.params)
    leaves_s = jax.tree_util.tree_leaves(a_soft.network.params)
    assert all((np.asarray(x) == np.asarray(y)).all() for x, y in zip(leaves_h, leaves_s)), \\
        "measure_density_hard must not change init/architecture -- only the loss"

    # tighten the hit-spec block (cols 19:22) so next-obs queries cluster well
    # within a few eps=0.4 of each other -- on the DEFAULT N(0,1) spread the
    # soft kernel is already near one-hot (points are typically several eps
    # apart), so k_self ~= 0.90 and the hard/soft contrast is too weak to be a
    # useful regression guard. A tighter cluster forces real mass-spreading.
    b_np = make_batch()
    b_np['next_observations'][:, 19:22] *= 0.1
    b = {k: jnp.asarray(v) for k, v in b_np.items()}
    rng = jax.random.PRNGKey(2)
    loss_h, info_h = a_hard.density_loss(b, a_hard.network.params, rng)
    loss_s, info_s = a_soft.density_loss(b, a_soft.network.params, rng)
    assert np.isfinite(float(loss_h)) and np.isfinite(float(loss_s))
    assert np.allclose(float(info_h['density_k_self']), 1.0, atol=1e-6), info_h['density_k_self']
    # on this clustered batch the soft kernel's own diagonal mass must sit well
    # below the hard ceiling -- proves the ablation swaps the immediate term
    # rather than relabeling the same computation.
    assert float(info_s['density_k_self']) < 0.5, info_s['density_k_self']
    assert not np.allclose(float(loss_h), float(loss_s), atol=1e-4), (float(loss_h), float(loss_s))

    blocks = _parse_hit_spec(cfg_hard['measure_hit_spec'])
    nobs = b['next_observations']
    B = nobs.shape[0]
    k_hard = np.eye(B)
    k_soft = np.asarray(GCIQLFlowAgent._delta_phi(
        nobs, GCIQLFlowAgent._raw_block_cols(nobs, blocks), blocks))
    assert not np.allclose(k_hard, k_soft, atol=1e-3), \\
        "hard and soft immediate terms coincide on this batch -- ablation has no effect"
    print("OK")
    """)


# --------------------------------------------------------------------------
# 10. end-to-end updates
# --------------------------------------------------------------------------
def test_end_to_end_updates():
    """30 update() steps on synthetic data: finite density_ce/entropy info each
    step (INCLUDING the round-2 review G2 instruments -- density_ess_median and
    the per-head density_ess_h0/h1, gated exactly like the pre-existing keys:
    only present because measure_density's own info dict emits them, nothing
    extra on the off path), w stays on the simplex, all four ESS statistics in
    [1, M], no NaN in params at the end."""
    _run("""
    cfg = base_cfg()
    a = agent_for(cfg)
    M = None
    for step in range(30):
        b = {k: jnp.asarray(v) for k, v in make_batch(seed=step).items()}
        M = b['next_observations'].shape[0]
        a, info = a.update(b)
        assert np.isfinite(float(info['measure/density_ce'])), (step, info['measure/density_ce'])
        for key in ('measure/density_w_entropy', 'measure/density_tgt_entropy',
                    'measure/density_ess', 'measure/density_ess_median',
                    'measure/density_ess_h0', 'measure/density_ess_h1',
                    'measure/density_logit_max', 'measure/density_k_self'):
            assert np.isfinite(float(info[key])), (step, key, info[key])
        for ess_key in ('measure/density_ess', 'measure/density_ess_median',
                        'measure/density_ess_h0', 'measure/density_ess_h1'):
            ess = float(info[ess_key])
            assert 1.0 - 1e-4 <= ess <= M + 1e-4, (step, ess_key, ess, M)
    leaves = jax.tree_util.tree_leaves(a.network.params)
    assert all(np.isfinite(np.asarray(l)).all() for l in leaves), "NaN in params after 30 updates"

    b = {k: jnp.asarray(v) for k, v in make_batch(seed=99).items()}
    blocks = _parse_hit_spec(cfg['measure_hit_spec'])
    u_sc = GCIQLFlowAgent._density_queries(b['next_observations'], blocks)
    b_emb = a.network.select('density_b')(u_sc)
    f_emb = a.network.select('density_f')(
        jnp.concatenate([b['observations'], b['actions'], b['value_goals']], -1))
    logits = jnp.einsum('hbd,md->hbm', f_emb, b_emb) / cfg['measure_density_temp']
    w = np.asarray(jax.nn.softmax(logits, -1))
    assert np.allclose(w.sum(-1), 1.0, atol=1e-4)
    assert (w >= -1e-6).all()
    print("OK")
    """)
