"""`measure_delta_partition` -- spec 17b variant B: a one-stage simplex successor
measure over a FIXED partition of the hit-spec object-configuration space (K
dataset-drawn prototypes). Contract:
`PAPER/rescuing-fb/specs/17b-delta-partition-impl.md`; science context
`PAPER/rescuing-fb/specs/17-delta-measure.md` section 1 variant B.

    phi_k(x)      = softmax_k( -0.5 * sum_blocks ||(x_blk - proto_k,blk)/eps_blk||^2 )  in Delta^K
    p(s,a,g)      = softmax( delta_head([s, a, g]) )                                     in Delta^K
    p_tgt         = sg[ (1-gamma)*phi(x') + gamma*p_bar(s', a+~pi, g) ]
    L             = CE(p_tgt, p) = -E[ sum_k p_tgt_k * log(p_k + 1e-8) ]
    actor q-term  = sum_k p_h(s, a_actor, g)_k * phi(goal)_k    (frozen delta_head params)

No goal-region queries, no hindsight columns, no phase switch, no bilinear
measure matrix, no ortho, no mass/nonneg penalties (validity is free -- the
softmax simplex conserves mass by construction), no scale_norm, no
z-conditioning, no hindsight, no warm start anywhere in this path (spec
section 4 anti-scope).

Originally written TEST-FIRST against the spec's contract, in the same
subprocess-jax harness as tests/test_gciql_measure_nstep.py, before
`GCIQLFlowAgent` implemented `_delta_phi` / `delta_partition_loss` / the
`delta_head` module / the `measure_delta_partition` knob
(tools/wandb_mode_shim/gciql_flow.py section 2 of the spec). All 10 tests are
green now that the agent-side implementation has landed, INCLUDING the round-1
review fixes: total_loss no longer double-gates the partition CE by
measure_coef (CRITICAL 1), and delta_protos is a static/nonpytree field so
pre-17b checkpoints keep restoring (CRITICAL 2, tested in
tests/test_gciql_checkpoint_restore.py). Test 9 (test_prototype_script) tests
only scripts/rescuing_fb/make_delta_prototypes.py (spec section 1.1, this
repo's own deliverable) and needs no jax.

Runner: `.venv/bin/python -m pytest tests/test_gciql_measure_delta_partition.py -x -q`
(NOT uv; the jax subprocess harness resolves the jax venv itself for tests
1-8/10 -- test 9 runs in the plain repo .venv, no jax needed).
"""
import os
import subprocess
import sys
import tempfile
import textwrap

import numpy as np

from tests.jax_subprocess import run_jax, REPO

OBS_D = 40
D = 8   # value_latent_dim, matches the spec's harness
K = 8   # toy prototype count for the base config

COMMON = f"""
    import jax, jax.numpy as jnp, numpy as np, flax, tempfile, os
    from gciql_flow import GCIQLFlowAgent, get_config, _parse_hit_spec

    OBS_D = {OBS_D}
    D = {D}
    K = {K}
    EX_OBS = jnp.zeros((1, OBS_D), jnp.float32)
    EX_ACT = jnp.zeros((1, 5), jnp.float32)
    TMPDIR = tempfile.mkdtemp()

    def make_protos(k=K, d_c=3, seed=0):
        # protos live in RAW obs units, columns = the hit-spec blocks concatenated in
        # _parse_hit_spec order (spec 1.1/2.3) -- written to a tmp .npy, never a real dataset.
        rng = np.random.default_rng(seed)
        protos = (rng.normal(size=(k, d_c)).astype(np.float32)) * 0.5
        path = os.path.join(TMPDIR, f"protos_{{seed}}_{{k}}_{{d_c}}.npy")
        np.save(path, protos)
        return path, protos

    def base_cfg(protos_path=None, **kw):
        # the spec's base cell: measure-only, bilinear, delta_partition, no first-passage
        # (the 17b path does not ride the hit-region matrix), actor_q_center required.
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = D
        cfg["measure_only"] = True
        cfg["measure_first_passage"] = False
        cfg["actor_q_center"] = True
        cfg["measure_delta_partition"] = True
        cfg["measure_hit_spec"] = "19:22:0.4"
        if protos_path is None:
            protos_path, _ = make_protos(k=K, d_c=3, seed=0)
        cfg["measure_delta_protos"] = protos_path
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
    """measure_delta_partition=False (the default, whether unset or explicit): param
    tree byte-identical to a no-knob config, no delta_head/target_delta_head modules
    registered, and total_loss bit-identical on a fixed batch."""
    _run("""
    cfg0 = dict(get_config()); cfg0["encoder"] = None; cfg0["frame_stack"] = None
    cfg1 = dict(cfg0)
    cfg1["measure_delta_partition"] = False
    cfg1["measure_delta_protos"] = ""
    a0, a1 = agent_for(cfg0), agent_for(cfg1)
    assert "modules_delta_head" not in a0.network.params
    assert "modules_target_delta_head" not in a0.network.params
    t0 = jax.tree_util.tree_leaves(a0.network.params)
    t1 = jax.tree_util.tree_leaves(a1.network.params)
    assert len(t0) == len(t1), (len(t0), len(t1))
    assert all((x == y).all() for x, y in zip(t0, t1)), "param tree differs with default knobs"
    b = make_batch()
    l0, _ = total_loss_of(a0, b)
    l1, _ = total_loss_of(a1, b)
    assert l0 == l1, (l0, l1)
    print("OK")
    """)


# --------------------------------------------------------------------------
# 2. phi: simplex + eps scaling
# --------------------------------------------------------------------------
def test_phi_simplex_and_scaling():
    """_delta_phi: rows sum to 1, >= 0 (Delta^K); argmax = nearest prototype on
    constructed points; a 2-block spec '19:22:0.4,28:31:0.8' with a point off by
    exactly eps_1=0.4 in block 1 vs a point off by exactly eps_2=0.8 in block 2 (from
    the SAME reference prototype) carries EQUAL per-block eps-scaled contributions --
    verified against a numpy reference that divides by each block's own eps BEFORE
    squaring (an unscaled reference would disagree)."""
    _run("""
    def phi_np(obs, protos, blocks):
        parts_x, parts_p, off = [], [], 0
        for (a, b, e) in blocks:
            w = b - a
            parts_x.append(obs[:, a:b] / e)
            parts_p.append(protos[:, off:off + w] / e)
            off += w
        xc = np.concatenate(parts_x, axis=1)
        pc = np.concatenate(parts_p, axis=1)
        diffs = xc[:, None, :] - pc[None, :, :]
        logits = -0.5 * np.sum(diffs ** 2, axis=-1)
        logits = logits - logits.max(axis=-1, keepdims=True)
        ex = np.exp(logits)
        return ex / ex.sum(axis=-1, keepdims=True)

    blocks = ((19, 22, 0.4),)
    _, protos = make_protos(k=5, d_c=3, seed=1)
    B = 6
    obs = np.zeros((B, OBS_D), np.float32)
    for i in range(5):
        obs[i, 19:22] = protos[i]
    obs[5, 19:22] = protos[0] + 5.0
    phi = np.asarray(GCIQLFlowAgent._delta_phi(jnp.asarray(obs), jnp.asarray(protos), blocks))
    assert phi.shape == (B, 5)
    assert np.allclose(phi.sum(-1), 1.0, atol=1e-5), phi.sum(-1)
    assert (phi >= -1e-6).all(), phi.min()
    for i in range(5):
        assert int(np.argmax(phi[i])) == i, (i, phi[i])
    assert int(np.argmax(phi[5])) == 0
    ref = phi_np(obs, protos, blocks)
    assert np.allclose(phi, ref, atol=1e-5), np.abs(phi - ref).max()

    blocks2 = ((19, 22, 0.4), (28, 31, 0.8))
    rng = np.random.default_rng(2)
    protos2 = rng.normal(size=(4, 6)).astype(np.float32)
    x = np.zeros((2, OBS_D), np.float32)
    x[0, 19:22] = [0.4, 0.0, 0.0]
    x[1, 28:31] = [0.8, 0.0, 0.0]
    phi2 = np.asarray(GCIQLFlowAgent._delta_phi(jnp.asarray(x), jnp.asarray(protos2), blocks2))
    ref2 = phi_np(x, protos2, blocks2)
    assert np.allclose(phi2, ref2, atol=1e-5), np.abs(phi2 - ref2).max()
    assert np.allclose(phi2.sum(-1), 1.0, atol=1e-5)
    print("OK")
    """)


# --------------------------------------------------------------------------
# 3. CE loss matches a numpy/direct-module reference
# --------------------------------------------------------------------------
def test_ce_matches_numpy_reference():
    """Small batch: rebuild p, p_bar, phi(x') by DIRECT module calls at the agent's own
    params (mirroring delta_partition_loss's own code exactly, including the bootstrap
    noise drawn from the SAME rng key passed to the loss), rebuild p_tgt/CE in numpy,
    and compare to delta_partition_loss's returned loss (atol 1e-5). Also pins: (a) the
    (1-gamma)/gamma convex combination leaves p_tgt on the simplex, and (b) the
    NEXT-state counting convention -- swapping phi(x') for phi(x) must NOT reproduce
    the same loss (an off-by-one regression guard)."""
    _run("""
    cfg = base_cfg()
    a = agent_for(cfg)
    b = make_batch(B=6, seed=11)
    bj = {k: jnp.asarray(v) for k, v in b.items()}
    rng = jax.random.PRNGKey(7)
    loss, info = a.delta_partition_loss(bj, a.network.params, rng)

    obs, nobs = bj['observations'], bj['next_observations']
    acts, g = bj['actions'], bj['value_goals']
    gamma = cfg['discount']
    blocks = _parse_hit_spec(cfg['measure_hit_spec'])

    logits = a.network.select('delta_head')(
        jnp.concatenate([obs, acts, g], -1), params=a.network.params)
    p = jax.nn.softmax(logits, -1)
    noise = jax.random.normal(rng, acts.shape)
    a_plus = jax.lax.stop_gradient(a.network.select('actor')(nobs, g, noise))
    t_logits = a.network.select('target_delta_head')(jnp.concatenate([nobs, a_plus, g], -1))
    p_bar = jax.nn.softmax(t_logits, -1)
    phi_n = GCIQLFlowAgent._delta_phi(nobs, a.delta_protos, blocks)
    p_tgt = jax.lax.stop_gradient((1.0 - gamma) * phi_n[None] + gamma * p_bar)
    ce_ref = -jnp.mean(jnp.sum(p_tgt * jnp.log(p + 1e-8), axis=-1))
    assert np.allclose(np.asarray(ce_ref), np.asarray(loss), atol=1e-5), \\
        (float(ce_ref), float(loss))
    assert np.allclose(np.asarray(jnp.sum(p_tgt, axis=-1)), 1.0, atol=1e-5)

    phi_c = GCIQLFlowAgent._delta_phi(obs, a.delta_protos, blocks)
    p_tgt_wrong = jax.lax.stop_gradient((1.0 - gamma) * phi_c[None] + gamma * p_bar)
    ce_wrong = -jnp.mean(jnp.sum(p_tgt_wrong * jnp.log(p + 1e-8), axis=-1))
    assert not np.allclose(np.asarray(ce_wrong), np.asarray(loss), atol=1e-5), \\
        "delta_partition_loss must count phi at the NEXT state, not the current one"
    print("OK")
    """)


# --------------------------------------------------------------------------
# 4. target_delta_head: init mirror + soft update
# --------------------------------------------------------------------------
def test_target_head_init_and_soft_update():
    """At create, target_delta_head params == delta_head params (the init mirror,
    spec section 2.7). Two update() steps on synthetic batches; the SECOND target
    equals tau*online_pre + (1-tau)*target_pre elementwise, where online_pre/target_pre
    are the online/target params AS THEY WERE going into that second update() call --
    the repo's target_update formula (agents/gciql.py) reads `self.network.params`
    (i.e. the PRE-step online params), not the just-updated ones. Two steps are
    required to make this discriminating: at init target==online exactly, so a
    single-step check degenerates to "target unchanged" whether or not target_update
    actually runs (both the correct formula and "frozen forever" agree on step 1).
    By step 2 the online/target params have diverged (round-1 review HIGH finding --
    verified by hand: temporarily deleting the `update` override makes this test fail
    on the discrepancy between the still-frozen target and the moved online params;
    restoring the override makes it pass again, atol=0 exactly -- see
    task-17b-agent-report.md)."""
    _run("""
    cfg = base_cfg()
    a = agent_for(cfg)
    p0 = a.network.params
    dh0_leaves = jax.tree_util.tree_leaves(p0["modules_delta_head"])
    tdh0_leaves = jax.tree_util.tree_leaves(p0["modules_target_delta_head"])
    assert len(dh0_leaves) == len(tdh0_leaves) and len(dh0_leaves) > 0
    assert all((np.asarray(x) == np.asarray(y)).all() for x, y in zip(dh0_leaves, tdh0_leaves)), \\
        "target_delta_head != delta_head at init"

    b1 = {k: jnp.asarray(v) for k, v in make_batch(seed=0).items()}
    a1, info1 = a.update(b1)
    p1 = a1.network.params
    dh1_leaves = jax.tree_util.tree_leaves(p1["modules_delta_head"])          # online, pre-step-2
    tdh1_leaves = jax.tree_util.tree_leaves(p1["modules_target_delta_head"])  # target, pre-step-2
    tau = cfg["tau"]
    # step 1 genuinely moved the online delta_head (the CRITICAL-1-fixed gradient path)
    assert any(not np.allclose(np.asarray(o0), np.asarray(o1), atol=0) for o0, o1 in
               zip(dh0_leaves, dh1_leaves)), "delta_head did not move after one update() -- CE gradient is zero"

    b2 = {k: jnp.asarray(v) for k, v in make_batch(seed=1).items()}
    a2, info2 = a1.update(b2)
    p2 = a2.network.params
    tdh2_leaves = jax.tree_util.tree_leaves(p2["modules_target_delta_head"])
    for onl_pre, old_pre, new in zip(dh1_leaves, tdh1_leaves, tdh2_leaves):
        expected = tau * np.asarray(onl_pre) + (1 - tau) * np.asarray(old_pre)
        assert np.allclose(np.asarray(new), expected, atol=1e-7), \\
            np.abs(np.asarray(new) - expected).max()
    print("OK")
    """)


# --------------------------------------------------------------------------
# 5. measure_only wiring in total_loss
# --------------------------------------------------------------------------
def test_measure_only_wiring():
    """With the knob on, total_loss's info dict contains a delta_ce-derived key (the
    measure_only branch calls delta_partition_loss INSTEAD of measure_loss, prefixed
    exactly as the stock measure info is) and NO measure_loss-derived keys; IQL
    value/critic loss keys are absent (measure_only semantics preserved)."""
    _run("""
    cfg = base_cfg()
    a = agent_for(cfg)
    b = make_batch()
    loss, info = total_loss_of(a, b)
    assert np.isfinite(loss)
    keys = sorted(info.keys())
    assert any('delta_ce' in k for k in info), keys
    assert not any('measure_loss' in k for k in info), keys
    assert not any(k.startswith('value/') for k in info), keys
    assert not any(k.startswith('critic/') for k in info), keys
    print("OK")
    """)


# --------------------------------------------------------------------------
# 6. grad hygiene
# --------------------------------------------------------------------------
def test_actor_grad_hygiene():
    """jax.grad(actor_loss): grad-norm over modules_delta_head + modules_target_delta_head
    + modules_critic leaves ~= 0 (the delta-partition q-term reads delta_head at FROZEN
    params and never touches the critic); over modules_actor > 0.
    jax.grad(delta_partition_loss): grad over modules_actor ~= 0 (bootstrap action is
    stop-gradded); over modules_delta_head > 0 (it's the thing being trained); over
    modules_target_delta_head == 0 (target network, no gradient path -- soft-updated
    outside the loss)."""
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
    assert gnorm(grads_a, 'modules_delta_head') < 1e-9, gnorm(grads_a, 'modules_delta_head')
    assert gnorm(grads_a, 'modules_target_delta_head') < 1e-9, gnorm(grads_a, 'modules_target_delta_head')
    assert gnorm(grads_a, 'modules_critic') < 1e-9, gnorm(grads_a, 'modules_critic')
    assert gnorm(grads_a, 'modules_actor') > 1e-8, gnorm(grads_a, 'modules_actor')

    def delta_loss_fn(params):
        loss, _ = a.delta_partition_loss(b, params, rng)
        return loss
    grads_d = jax.grad(delta_loss_fn)(a.network.params)
    assert gnorm(grads_d, 'modules_actor') < 1e-9, gnorm(grads_d, 'modules_actor')
    assert gnorm(grads_d, 'modules_delta_head') > 1e-8, gnorm(grads_d, 'modules_delta_head')
    assert gnorm(grads_d, 'modules_target_delta_head') < 1e-9, gnorm(grads_d, 'modules_target_delta_head')
    print("OK")
    """)


# --------------------------------------------------------------------------
# 7. actor q-term reads the goal's cell
# --------------------------------------------------------------------------
def test_actor_q_reads_goal_cell():
    """Goals constructed sitting EXACTLY on prototype j*: z_g=phi(goal) must be peaked
    at j* (its max component). The actor's q equals sum_k p_k * z_g,k for a
    hand-computed p (direct delta_head module call at the SAME actor_actions actor_loss
    itself samples -- reconstructed deterministically from the same rng key, since the
    pre-existing rng-splitting/one-shot-actor code this reads is untouched by 17b)."""
    _run("""
    cfg = base_cfg()
    a = agent_for(cfg)
    blocks = _parse_hit_spec(cfg['measure_hit_spec'])
    protos = np.asarray(a.delta_protos)
    jstar = 2
    B = 5
    rng_np = np.random.default_rng(5)
    goals = np.zeros((B, OBS_D), np.float32)
    goals[:, 19:22] = protos[jstar]
    obs = rng_np.normal(size=(B, OBS_D)).astype(np.float32)

    z_g = GCIQLFlowAgent._delta_phi(jnp.asarray(goals), a.delta_protos, blocks)
    assert (np.asarray(jnp.argmax(z_g, axis=-1)) == jstar).all(), np.asarray(z_g)

    b = make_batch(B=B, seed=6)
    b['actor_goals'] = goals
    b['observations'] = obs
    bj = {k: jnp.asarray(v) for k, v in b.items()}
    rng = jax.random.PRNGKey(4)
    _, info = a.actor_loss(bj, a.network.params, rng)

    # reproduce actor_actions exactly as actor_loss computes it (pre-existing code,
    # untouched by 17b): rng, x0_rng, t_rng, n_rng, z_rng = split(rng, 5); noise =
    # normal(n_rng, acts.shape); actor_cond_input = goals (actor_cond='goal', default).
    _, x0_rng, t_rng, n_rng, z_rng = jax.random.split(rng, 5)
    noise = jax.random.normal(n_rng, bj['actions'].shape)
    actor_actions = a.network.select('actor')(bj['observations'], jnp.asarray(goals), noise,
                                              params=a.network.params)

    logits_a = a.network.select('delta_head')(
        jnp.concatenate([bj['observations'], actor_actions, jnp.asarray(goals)], -1),
        params=a.network.params)
    p_hand = jax.nn.softmax(logits_a, -1)
    q1_hand, q2_hand = (p_hand * z_g[None]).sum(-1)
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
    """Refusals: no measure_only; with measure_delta (17a); with measure_first_passage;
    empty/missing protos path; wrong-width protos file; actor_q_center=False; with
    warm_start_ckpt; with measure_ortho_coef>0. Accepts the base config."""
    _run("""
    def refuses(**kw):
        try:
            agent_for(base_cfg(**kw))
        except AssertionError:
            return True
        return False
    assert refuses(measure_only=False)
    assert refuses(measure_delta=True)
    assert refuses(measure_first_passage=True)
    assert refuses(measure_delta_protos='')
    wrong_path = os.path.join(TMPDIR, 'wrong_protos.npy')
    np.save(wrong_path, np.random.default_rng(9).normal(size=(8, 5)).astype(np.float32))
    assert refuses(measure_delta_protos=wrong_path)
    assert refuses(actor_q_center=False)
    assert refuses(warm_start_ckpt='/nonexistent/does/not/exist*')
    assert refuses(measure_ortho_coef=1.0)
    assert not refuses()
    print("OK")
    """)


# --------------------------------------------------------------------------
# 9. prototype script (this repo's own deliverable -- must be green)
# --------------------------------------------------------------------------
def test_prototype_script():
    """Runs scripts/rescuing_fb/make_delta_prototypes.py end-to-end, as a subprocess, on
    a synthetic 200-row npz dataset (no real dataset, no jax): output shape [16, d_c],
    float32, and every prototype row appears among the dataset's own block rows (the
    script draws distinct dataset rows, it does not cluster/average)."""
    tmpdir = tempfile.mkdtemp()
    ds_path = os.path.join(tmpdir, "synthetic.npz")
    out_path = os.path.join(tmpdir, "protos.npy")
    rng = np.random.default_rng(0)
    n, obs_d = 200, OBS_D
    obs = rng.normal(size=(n, obs_d)).astype(np.float32)
    np.savez(ds_path, observations=obs)

    script = REPO / "scripts" / "rescuing_fb" / "make_delta_prototypes.py"
    res = subprocess.run(
        [sys.executable, str(script),
         "--dataset", ds_path, "--hit_spec", "19:22:0.4",
         "--k", "16", "--seed", "0", "--out", out_path],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"

    protos = np.load(out_path)
    assert protos.shape == (16, 3), protos.shape
    assert protos.dtype == np.float32, protos.dtype
    block_cols = obs[:, 19:22]
    for row in protos:
        assert any(np.allclose(row, r) for r in block_cols), \
            "prototype row not found among the dataset's own block rows"
    print(res.stdout)


# --------------------------------------------------------------------------
# 10. end-to-end updates
# --------------------------------------------------------------------------
def test_end_to_end_updates():
    """30 update() steps on synthetic data: finite delta_ce / entropy info each step, no
    NaN in params, and p stays on the simplex at the final params."""
    _run("""
    cfg = base_cfg()
    a = agent_for(cfg)
    for step in range(30):
        b = {k: jnp.asarray(v) for k, v in make_batch(seed=step).items()}
        a, info = a.update(b)
        ce_keys = [k for k in info if 'delta_ce' in k]
        assert ce_keys, (step, sorted(info.keys()))
        for k in ce_keys:
            assert np.isfinite(float(info[k])), (step, k, info[k])
        ent_keys = [k for k in info if 'delta_p_entropy' in k or 'delta_tgt_entropy' in k]
        for k in ent_keys:
            assert np.isfinite(float(info[k])), (step, k, info[k])
    leaves = jax.tree_util.tree_leaves(a.network.params)
    assert all(np.isfinite(np.asarray(l)).all() for l in leaves), "NaN in params after 30 updates"

    b = {k: jnp.asarray(v) for k, v in make_batch(seed=99).items()}
    logits = a.network.select('delta_head')(
        jnp.concatenate([b['observations'], b['actions'], b['value_goals']], -1))
    p = np.asarray(jax.nn.softmax(logits, -1))
    assert np.allclose(p.sum(-1), 1.0, atol=1e-4)
    assert (p >= -1e-6).all()
    print("OK")
    """)


# --------------------------------------------------------------------------
# 11. round-1 review CRITICAL 1 regression: the shipped config actually trains
#     delta_head (measure_coef must not gate the partition CE a second time)
# --------------------------------------------------------------------------
def test_shipped_config_trains_delta_head():
    """CRITICAL 1 (round-1 review): measure_coef defaults to 0.0, and
    configs/gciql/cube_single_state_measure_delta_partition_d50.yaml deliberately
    DROPS it (spec 17b section 6) -- so a config built from the SHIPPED knob set
    (measure_coef absent, i.e. left at its 0.0 default) must still train delta_head.
    Before the fix, total_loss's measure_only branch multiplied the WHOLE m_loss
    (including the 17b CE) by measure_coef, so this config trained nothing: zero
    gradient into delta_head, target frozen at random init, while total_loss still
    returned a finite loss and a plausible-looking delta_ce/entropy info dict -- a
    silent false negative that would have burned a real training wave. Directly
    checks (a) jax.grad(total_loss) has nonzero norm over modules_delta_head, and
    (b) delta_head params actually move over 30 update() steps."""
    _run("""
    cfg = base_cfg()
    assert "measure_coef" not in cfg or cfg["measure_coef"] == 0.0, \\
        "this test must exercise the SHIPPED config's knob set: measure_coef absent/0.0"
    a = agent_for(cfg)
    b = {k: jnp.asarray(v) for k, v in make_batch().items()}
    rng = jax.random.PRNGKey(0)

    def gnorm(tree, key):
        leaves = jax.tree_util.tree_leaves(tree[key])
        return float(sum(float(jnp.sum(jnp.abs(l))) for l in leaves)) if leaves else 0.0

    def total_loss_fn(params):
        loss, _ = a.total_loss(b, params, rng)
        return loss
    grads = jax.grad(total_loss_fn)(a.network.params)
    g = gnorm(grads, 'modules_delta_head')
    assert g > 1e-8, f"delta_head grad norm under the SHIPPED config is ~0 ({g}) -- CRITICAL 1 regressed"

    dh0 = jax.tree_util.tree_leaves(a.network.params["modules_delta_head"])
    for step in range(30):
        bs = {k: jnp.asarray(v) for k, v in make_batch(seed=step).items()}
        a, info = a.update(bs)
    dh30 = jax.tree_util.tree_leaves(a.network.params["modules_delta_head"])
    moved = any(not np.allclose(np.asarray(x), np.asarray(y), atol=0) for x, y in zip(dh0, dh30))
    assert moved, "delta_head params did not move after 30 update() steps under the shipped config"
    print("OK grad_norm=%r" % g)
    """)


# --------------------------------------------------------------------------
# 12. round-1 review CRITICAL 2 regression: delta_protos must not break restore
#     of a pre-17b checkpoint (must not be a serialized pytree leaf)
# --------------------------------------------------------------------------
def test_delta_protos_not_serialized_and_old_checkpoint_restores():
    """CRITICAL 2 (round-1 review): restore_agent (flax_utils.py) calls
    flax.serialization.from_state_dict on the WHOLE agent, not just the network.
    If delta_protos were a pytree (data) field, EVERY pre-17b checkpoint's agent
    dict (which only ever has {'rng','network'}) would fail restore with
    "Missing field delta_protos". Proves the fix two ways: (a) to_state_dict of a
    live agent (knob OFF, matching every existing checkpoint's provenance) never
    emits a 'delta_protos' key -- it must not be part of the serialized pytree at
    all; (b) from_state_dict against a state dict containing ONLY {'rng','network'}
    (the literal shape of a pre-17b checkpoint's agent dict) succeeds without
    raising, and the restored agent still has SOME delta_protos value (its own
    field, untouched by the restore -- static fields are not overwritten by
    from_state_dict). See task-17b-agent-report.md for a hand-verification against
    the real anchor checkpoint cd_fp_s0/params_500000.pkl."""
    _run("""
    cfg_off = dict(get_config()); cfg_off["encoder"] = None; cfg_off["frame_stack"] = None
    agent_off = agent_for(cfg_off)

    full_sd = flax.serialization.to_state_dict(agent_off)
    assert set(full_sd.keys()) <= {"rng", "network"}, \\
        f"delta_protos leaked into the serialized state dict: {sorted(full_sd.keys())}"
    assert "delta_protos" not in full_sd

    # also prove it for an ON-knob agent: the field must be excluded from
    # serialization regardless of the knob (it is the DATACLASS FIELD, not the
    # knob, that determines pytree-ness) -- an on-knob agent's own to_state_dict
    # must equally never mention delta_protos.
    agent_on = agent_for(base_cfg())
    assert agent_on.delta_protos is not None
    full_sd_on = flax.serialization.to_state_dict(agent_on)
    assert "delta_protos" not in full_sd_on
    assert set(full_sd_on.keys()) <= {"rng", "network"}

    # the LITERAL shape of a pre-17b checkpoint's agent dict (pickle['agent']),
    # restored into an agent whose CONFIG (hence network architecture) matches --
    # exactly what restore_agent does in practice (the checkpoint's own recorded
    # config always matches its own architecture; 17b only adds a NEW top-level
    # agent field, it doesn't change any existing checkpoint's shape):
    old_style_sd = {"rng": full_sd["rng"], "network": full_sd["network"]}
    restored = flax.serialization.from_state_dict(agent_off, old_style_sd)
    assert restored is not None
    # static field: untouched by restore, still whatever the constructing agent had
    assert restored.delta_protos == agent_off.delta_protos
    print("OK")
    """)
