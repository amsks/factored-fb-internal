"""Spec 28 P1 -- the conditioning x actor-reward 2x2.

Contract: `PAPER/Multistep-FB/specs/28-z-interface-canonicalization.md` section P1.

Spec 24 changed the policy's CONDITIONING (g -> z) and the actor's REWARD READOUT
(exact kernel -> B̃^T z/sqrt(d)) at the same time, so the .736 -> .017 cube-double
collapse is attributable to neither. Two default-off enum knobs cut the 2x2 apart:

  cell (ii)  `measure_density_z_actor_reward = 'kernel'`   [z path]
        w_j(s,a,z) keeps conditioning on z ALONE; the reward vector reverts to the
        g-path's EXACT object kg = _delta_phi(goals, u_raw) -- query-set softmax-
        normalized, rows summing to 1 over the M queries, NOT a raw kappa (parity
        with cell i is what makes the cell readable at all). Random-sphere z rows
        have no coherent goal, so the q term is masked by the existing z_hindsight
        indicator and renormalized by the surviving count, exactly as C3's BC mask.

  cell (iii) `measure_density_actor_reward = 'latent'`     [g path]
        Everything stays g-indexed; only the reward becomes B̃(u)^T z(g)/sqrt(d),
        with z(g) = _density_z_encode(goals) computed but NEVER fed to F or the
        actor. Readout validity in isolation.

Neither knob is a coefficient: there is no dose that could silently zero the new
term (the 17b `measure_coef` fake-NULL species). Both are enums whose only other
value is the shipped default, and every combination in which they would be inert
or contradictory is refused loudly at create().

The suite drives the REAL entrypoints -- `update()`, `actor_loss()` and
`sample_actions()` -- not just `create()`: spec 24's own launch bug (a z build
whose `sample_actions` had no z branch) passed eight construct-only unit tests.

Same subprocess-jax harness as tests/test_gciql_measure_density_z_coupling.py:
OBS_D=40, action dim 5, d=8, hit_spec '19:22:0.4'.

Runner: `.venv/bin/python -m pytest tests/test_gciql_spec28_actor_reward.py -x -q`
(NOT uv; the jax subprocess harness resolves the jax venv itself).
"""
import subprocess
import textwrap

import numpy as np
import pytest

from tests.jax_subprocess import run_jax, REPO

OBS_D = 40
D = 8    # value_latent_dim AND measure_density_dim, matches the spec-18/24/25 harness
ACT_D = 5

# The last commit BEFORE spec 28's implementation. It has spec 24 + spec 25 and none
# of the spec-28 knobs, so it permanently marks the arm this spec must reproduce
# bit-for-bit at defaults -- pinned, not HEAD, because HEAD moves forward on this
# shared branch as other sessions land commits (same discipline as
# tests/test_gciql_measure_density_z_coupling.py's SPEC24_SHA).
PRE_SPEC28_SHA = "f145c2d48a0d2b88c6fb31f3d887ab7bd45d54ed"

NEW_KNOBS = ("measure_density_z_actor_reward", "measure_density_actor_reward")


def _baseline_reachable(sha):
    r = subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"],
                       cwd=str(REPO), capture_output=True, text=True)
    return r.returncode == 0


COMMON = f"""
    import jax, jax.numpy as jnp, numpy as np
    from gciql_flow import GCIQLFlowAgent, get_config, _parse_hit_spec, _sphere

    OBS_D = {OBS_D}
    D = {D}
    ACT_D = {ACT_D}
    EX_OBS = jnp.zeros((1, OBS_D), jnp.float32)
    EX_ACT = jnp.zeros((1, ACT_D), jnp.float32)

    def g_cfg(**kw):
        # cell (i): the g-indexed density arm (spec 18) -- measure-only, bilinear,
        # density, flow actor, every spec-28 knob at its default.
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

    def z_cfg(**kw):
        # cell (iv): spec 24's z-indexed arm at its shipped defaults.
        return g_cfg(measure_density_z=True, **kw)

    def make_batch(B=16, seed=0):
        rng = np.random.default_rng(seed)
        obs = rng.normal(size=(B, OBS_D)).astype(np.float32)
        nobs = rng.normal(size=(B, OBS_D)).astype(np.float32)
        return dict(
            observations=obs,
            next_observations=nobs,
            actions=np.clip(rng.normal(size=(B, ACT_D)), -1, 1).astype(np.float32),
            value_goals=rng.normal(size=(B, OBS_D)).astype(np.float32),
            actor_goals=rng.normal(size=(B, OBS_D)).astype(np.float32),
            rewards=np.zeros(B, np.float32),
            masks=np.ones(B, np.float32),
        )

    def agent_for(cfg, seed=0):
        return GCIQLFlowAgent.create(seed, EX_OBS, EX_ACT, cfg)

    def jb(b):
        return {{k: jnp.asarray(v) for k, v in b.items()}}

    def blocks_of(cfg):
        return _parse_hit_spec(cfg["measure_hit_spec"])

    def kg_ref(goals, nobs, blocks):
        # the g-path reward object, verbatim: query-set softmax-normalized kernel rows.
        u_raw = GCIQLFlowAgent._raw_block_cols(nobs, blocks)
        return np.asarray(GCIQLFlowAgent._delta_phi(goals, u_raw, blocks))

    def z_enc_ref(agent, goal_obs, blocks, d):
        # deterministic half of the z mix (no rng): sqrt(d)*normalize(target_B(x~)).
        x_sc = GCIQLFlowAgent._density_queries(goal_obs, blocks)
        return np.asarray(_sphere(agent.network.select('target_density_b')(x_sc), float(d)))

    def z_mix_ref(agent, goal_obs, blocks, d, rand_frac, rng):
        # full mix + hindsight indicator, SAME rng-split order as _density_z_masked.
        z_g = jnp.asarray(z_enc_ref(agent, goal_obs, blocks, d))
        rng_sphere, rng_mix = jax.random.split(rng)
        B = goal_obs.shape[0]
        z_rand = _sphere(jax.random.normal(rng_sphere, (B, d)), float(d))
        use_rand = jax.random.bernoulli(rng_mix, rand_frac, (B, 1))
        z = jnp.where(use_rand, z_rand, z_g)
        return z, (1.0 - use_rand[:, 0].astype(jnp.float32))

    def q_z_ref(agent, cfg, bj, z, reward):
        # reference for _density_actor_q_z's returned (q1, q2): w from F(s,a,z) at the
        # ONLINE towers, dotted into whatever `reward` row vector is supplied.
        blocks = blocks_of(cfg)
        u_sc = GCIQLFlowAgent._density_queries(bj['next_observations'], blocks)
        b_emb = agent.network.select('density_b')(u_sc)
        f_emb = agent.network.select('density_f')(
            jnp.concatenate([bj['observations'], bj['actions'], z], -1))
        w = np.asarray(jax.nn.softmax(
            jnp.einsum('hbd,md->hbm', f_emb, b_emb) / cfg['measure_density_temp'], -1))
        return (w * np.asarray(reward)[None]).sum(-1)

    def q_g_ref(agent, cfg, bj, goals, reward):
        # reference for _density_actor_q's returned (q1, q2): w from F(s,a,g).
        blocks = blocks_of(cfg)
        u_sc = GCIQLFlowAgent._density_queries(bj['next_observations'], blocks)
        b_emb = agent.network.select('density_b')(u_sc)
        f_emb = agent.network.select('density_f')(
            jnp.concatenate([bj['observations'], bj['actions'], goals], -1))
        w = np.asarray(jax.nn.softmax(
            jnp.einsum('hbd,md->hbm', f_emb, b_emb) / cfg['measure_density_temp'], -1))
        return (w * np.asarray(reward)[None]).sum(-1)
"""


def _run(body, **kw):
    code = textwrap.dedent(COMMON) + textwrap.dedent(body)
    res = run_jax(code, **kw)
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    return res


# --------------------------------------------------------------------------
# 1. all-defaults bit-identity to the pre-spec-28 agent
# --------------------------------------------------------------------------
def test_default_off_bit_identical_to_pre_spec28():
    """Both spec-28 knobs at their defaults: param tree, total_loss, THREE REAL
    update() steps and sample_actions must all be bit-identical to the agent loaded
    from the pinned pre-spec-28 commit -- on the z arm (cell iv), the g arm (cell i)
    and a config with no measure knobs at all.

    Real update steps, not a construct-only check: a previous z build passed 8/8 unit
    tests while its eval path was broken because nothing drove sample_actions. Both
    entrypoints are exercised here on both sides of the diff."""
    if not _baseline_reachable(PRE_SPEC28_SHA):
        pytest.skip(f"pinned baseline {PRE_SPEC28_SHA} is not reachable in this "
                    "checkout (e.g. a shallow clone) -- cannot run the bit-identity diff")
    _run(f"""
    PRE_SPEC28_SHA = {PRE_SPEC28_SHA!r}
    NEW_KNOBS = {NEW_KNOBS!r}
    REPO_DIR = {str(REPO)!r}
    """ + """
    import subprocess, importlib.util, tempfile, os

    baseline_src = subprocess.run(
        ["git", "show", PRE_SPEC28_SHA + ":tools/wandb_mode_shim/gciql_flow.py"],
        cwd=REPO_DIR, capture_output=True, text=True, check=True).stdout
    assert "measure_density_z" in baseline_src, \\
        "pinned baseline predates spec 24 entirely -- re-pin PRE_SPEC28_SHA"
    for k28 in NEW_KNOBS:
        assert k28 not in baseline_src, (
            "pinned baseline already contains " + k28 + " -- the pin has gone stale; "
            "re-pin PRE_SPEC28_SHA to the last commit before the spec-28 implementation")
    baseline_path = os.path.join(tempfile.mkdtemp(), "gciql_flow_pre28.py")
    with open(baseline_path, "w") as f:
        f.write(baseline_src)
    spec = importlib.util.spec_from_file_location("gciql_flow_pre28", baseline_path)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)

    def cfg_of(get_cfg_fn, **kw):
        cfg = dict(get_cfg_fn())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = D
        cfg["measure_only"] = True
        cfg["actor_q_center"] = True
        cfg["measure_density"] = True
        cfg["measure_density_dim"] = D
        cfg["measure_hit_spec"] = "19:22:0.4"
        cfg.update(kw)
        return cfg

    bj = jb(make_batch())
    obs_e = jnp.asarray(np.random.default_rng(1).normal(size=(4, OBS_D)).astype(np.float32))
    g_e = jnp.asarray(np.random.default_rng(2).normal(size=(4, OBS_D)).astype(np.float32))

    def same(a_base, a_new, label):
        t_base = jax.tree_util.tree_leaves(a_base.network.params)
        t_new = jax.tree_util.tree_leaves(a_new.network.params)
        assert len(t_base) == len(t_new), (label, len(t_base), len(t_new))
        assert all((x == y).all() for x, y in zip(t_base, t_new)), label + ": param tree"
        assert set(a_base.network.params.keys()) == set(a_new.network.params.keys()), \\
            (label, sorted(a_base.network.params), sorted(a_new.network.params))
        l_b, i_b = a_base.total_loss(bj, a_base.network.params, jax.random.PRNGKey(7))
        l_n, i_n = a_new.total_loss(bj, a_new.network.params, jax.random.PRNGKey(7))
        assert float(l_b) == float(l_n), (label, float(l_b), float(l_n))
        assert sorted(i_b) == sorted(i_n), (label, "info keys drifted",
                                           set(i_b) ^ set(i_n))

    def same_after_updates(cfg_kw, label, n=3):
        a_b = baseline.GCIQLFlowAgent.create(0, EX_OBS, EX_ACT,
                                             cfg_of(baseline.get_config, **cfg_kw))
        a_n = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg_of(get_config, **cfg_kw))
        same(a_b, a_n, label)
        for step in range(n):
            bb = jb(make_batch(seed=step))
            a_b, i_b = a_b.update(bb)
            a_n, i_n = a_n.update(bb)
            for k in ('measure/density_ce', 'actor/actor_loss', 'actor/q_mean'):
                assert float(i_b[k]) == float(i_n[k]), (label, step, k,
                                                        float(i_b[k]), float(i_n[k]))
        lb = jax.tree_util.tree_leaves(a_b.network.params)
        ln = jax.tree_util.tree_leaves(a_n.network.params)
        assert all((x == y).all() for x, y in zip(lb, ln)), \\
            label + ": param tree diverged after " + str(n) + " real update() steps"
        # and the EVAL entrypoint (the one a construct-only suite would miss)
        ac_b = np.asarray(a_b.sample_actions(obs_e, g_e, seed=jax.random.PRNGKey(0)))
        ac_n = np.asarray(a_n.sample_actions(obs_e, g_e, seed=jax.random.PRNGKey(0)))
        assert (ac_b == ac_n).all(), label + ": sample_actions diverged"

    same_after_updates(dict(measure_density_z=True), "cell (iv): spec-24 z arm")
    same_after_updates(dict(), "cell (i): g-indexed density arm")

    cnb = dict(baseline.get_config()); cnb["encoder"] = None; cnb["frame_stack"] = None
    cnn = dict(get_config()); cnn["encoder"] = None; cnn["frame_stack"] = None
    same(baseline.GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cnb),
         GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cnn), "no-knob config")
    print("OK")
    """)


# --------------------------------------------------------------------------
# 2. cell (ii): the reward object is the g-path's EXACT kg
# --------------------------------------------------------------------------
def test_cell_ii_reward_is_the_normalized_kernel():
    """The parity requirement: cell (ii)'s reward vector must be `_delta_phi(goals,
    u_raw)` -- the SAME object `_density_actor_q` uses -- not a raw kappa. Checked
    three ways: (a) rows sum to 1 over the M queries and are non-negative (the
    signature of the query-set softmax normalization); (b) the q term equals a
    direct-module reference w(s,a,z) . kg; (c) it is byte-equal to the reward the
    g-path builds from the same goals and the same batch."""
    _run("""
    cfg_k = z_cfg(measure_density_z_actor_reward='kernel')
    a = agent_for(cfg_k)
    b = make_batch(B=12, seed=5); bj = jb(b)
    blocks = blocks_of(cfg_k)
    goals = bj['actor_goals']

    kg = kg_ref(goals, bj['next_observations'], blocks)
    assert np.allclose(kg.sum(-1), 1.0, atol=1e-5), \\
        ("cell ii reward rows must sum to 1 over the M queries (query-set softmax "
         "normalized); a raw kappa would not", kg.sum(-1)[:4])
    assert (kg >= -1e-8).all()

    # (b) q term against the direct-module reference, at a FIXED z (so the check is on
    # the reward object alone).
    z = jnp.asarray(z_enc_ref(a, goals, blocks, D))
    q1, q2 = a._density_actor_q_z(bj, bj['observations'], bj['actions'], z, goals=goals)
    ref = q_z_ref(a, cfg_k, bj, z, kg)
    assert np.allclose(np.asarray(q1), ref[0], atol=1e-5), (np.asarray(q1)[:3], ref[0][:3])
    assert np.allclose(np.asarray(q2), ref[1], atol=1e-5)

    # (c) byte-equality with the g-path's own reward object on the same inputs.
    a_g = agent_for(g_cfg())
    kg_g = kg_ref(goals, bj['next_observations'], blocks)
    assert (kg == kg_g).all(), "cell (ii) and cell (i) must share the reward object"

    # and it is genuinely DIFFERENT from the spec-24 latent reward (no accidental no-op)
    a_l = agent_for(z_cfg())
    ql1, _ = a_l._density_actor_q_z(bj, bj['observations'], bj['actions'], z)
    assert not np.allclose(np.asarray(q1), np.asarray(ql1), atol=1e-4), \\
        "kernel and latent readouts coincide -- the knob is not doing anything"
    print("OK")
    """)


# --------------------------------------------------------------------------
# 3. cell (ii): conditioning stays z-only -- the goal touches ONLY the reward
# --------------------------------------------------------------------------
def test_cell_ii_goal_enters_only_the_reward():
    """The decisive cell is only decisive if the goal reaches nothing but the frozen
    reward vector. Held z fixed and swapping the goals must leave the occupancy w
    completely unchanged, so that q(goals_a) - q(goals_b) = sum_j w_j (kg_a - kg_b)_j
    EXACTLY. Conversely, holding goals fixed and swapping z must move q -- otherwise
    the cell would not be z-conditioned at all."""
    _run("""
    cfg_k = z_cfg(measure_density_z_actor_reward='kernel')
    a = agent_for(cfg_k)
    b = make_batch(B=10, seed=11); bj = jb(b)
    blocks = blocks_of(cfg_k)
    ga, gb = bj['actor_goals'], bj['value_goals']
    z = jnp.asarray(z_enc_ref(a, ga, blocks, D))

    qa = np.asarray(a._density_actor_q_z(bj, bj['observations'], bj['actions'], z, goals=ga))
    qb = np.asarray(a._density_actor_q_z(bj, bj['observations'], bj['actions'], z, goals=gb))
    dk = kg_ref(ga, bj['next_observations'], blocks) - kg_ref(gb, bj['next_observations'], blocks)
    # w is unchanged across the two calls, so the difference is exactly w . (kg_a - kg_b)
    pred = q_z_ref(a, cfg_k, bj, z, dk)
    assert np.allclose(qa - qb, pred, atol=1e-5), \\
        "swapping the goal moved something other than the reward vector"
    assert np.abs(qa - qb).max() > 1e-6, "the goals were effectively identical -- vacuous test"

    # z still conditions everything: a different index moves q at a FIXED goal.
    z2 = jnp.asarray(_sphere(jax.random.normal(jax.random.PRNGKey(3), (10, D)), float(D)))
    q2 = np.asarray(a._density_actor_q_z(bj, bj['observations'], bj['actions'], z2, goals=ga))
    assert np.abs(qa - q2).max() > 1e-6, \\
        "q is insensitive to z under the kernel reward -- the cell is not z-conditioned"
    print("OK")
    """)


# --------------------------------------------------------------------------
# 4. cell (ii): the hindsight mask on the q term
# --------------------------------------------------------------------------
def test_cell_ii_q_mask_and_renormalization():
    """The q term is masked to hindsight rows and renormalized by the surviving count,
    the SAME treatment (and the SAME indicator, from the SAME draw) as C3's BC mask.

    (a) rand_frac=0 -> the mask is all-ones and the masked mean reproduces the plain
        mean EXACTLY (the renormalization is not a rescaling);
    (b) at rand_frac=0.5 the actor_loss delta against the spec-24 arm equals
        -maskedmean(q_kernel) + mean(q_latent) computed independently -- i.e. the
        reduction really is the masked mean, not q.mean();
    (c) actor/q_mask_frac is logged and equals the surviving fraction;
    (d) the default ('latent') arm logs NO mask keys at all."""
    _run("""
    blocks = blocks_of(z_cfg())
    rngA = jax.random.PRNGKey(4)
    # actor_loss splits its incoming key 5 ways; the LAST is the z stream.
    z_rng = jax.random.split(rngA, 5)[4]

    def actor_pieces(cfg, rand_frac):
        a = agent_for(cfg)
        b = make_batch(B=24, seed=7); bj = jb(b)
        z, hs = z_mix_ref(a, bj['actor_goals'], blocks, D, rand_frac, z_rng)
        return a, bj, np.asarray(z), np.asarray(hs)

    # ---- (a) rand_frac = 0: mask all-ones, exact reproduction of the plain mean ----
    cfg0 = z_cfg(measure_density_z_actor_reward='kernel',
                 measure_density_z_rand_frac=0.0, bc_coeff=0.0)
    a0, bj0, z0, hs0 = actor_pieces(cfg0, 0.0)
    assert (hs0 == 1.0).all(), "rand_frac=0 must give an all-ones hindsight indicator"
    l_k, info_k = a0.actor_loss(bj0, a0.network.params, rngA)
    noise0 = jax.random.normal(jax.random.split(rngA, 5)[3], bj0['actions'].shape)
    acts0 = a0.network.select('actor')(bj0['observations'], jnp.asarray(z0), noise0)
    q1, q2 = a0._density_actor_q_z(bj0, bj0['observations'], acts0, jnp.asarray(z0),
                                   goals=bj0['actor_goals'])
    q = np.minimum(np.asarray(q1), np.asarray(q2))
    # bc_coeff=0 -> actor_loss = -(q term) + bc_flow_loss
    assert np.allclose(float(l_k) - float(info_k['bc_flow_loss']), -q.mean(), atol=1e-5), \\
        (float(l_k), float(info_k['bc_flow_loss']), -q.mean())
    assert abs(float(info_k['q_mask_frac']) - 1.0) < 1e-6

    # ---- (b) rand_frac = 0.5: the reduction is the MASKED mean ----
    cfg5 = z_cfg(measure_density_z_actor_reward='kernel',
                 measure_density_z_rand_frac=0.5, bc_coeff=0.0)
    a5, bj5, z5, hs5 = actor_pieces(cfg5, 0.5)
    assert 0.0 < hs5.mean() < 1.0, "degenerate draw -- mask test would be vacuous"
    l5, info5 = a5.actor_loss(bj5, a5.network.params, rngA)
    noise = jax.random.normal(jax.random.split(rngA, 5)[3], bj5['actions'].shape)
    acts = a5.network.select('actor')(bj5['observations'], jnp.asarray(z5), noise)
    qk1, qk2 = a5._density_actor_q_z(bj5, bj5['observations'], acts, jnp.asarray(z5),
                                     goals=bj5['actor_goals'])
    qk = np.minimum(np.asarray(qk1), np.asarray(qk2))
    masked = float((qk * hs5).sum() / hs5.sum())
    assert np.allclose(float(l5) - float(info5['bc_flow_loss']), -masked, atol=1e-5), \\
        ("the q reduction is not the masked mean", float(l5),
         float(info5['bc_flow_loss']), -masked, -qk.mean())
    assert not np.allclose(-masked, -qk.mean(), atol=1e-6), \\
        "masked and unmasked means coincide -- the check cannot discriminate"

    # ---- (c)/(d) logging ----
    assert abs(float(info5['q_mask_frac']) - float(hs5.mean())) < 1e-6
    assert np.isclose(float(info5['q_mask_mean']), masked, atol=1e-5)
    a_def = agent_for(z_cfg(bc_coeff=0.0))
    _, info_def = a_def.actor_loss(bj5, a_def.network.params, rngA)
    assert not any(k.startswith('q_mask') for k in info_def), \\
        ("the default latent arm must not log mask keys", sorted(info_def))

    # ---- the masked path still trains with bc_coeff > 0 (q_scale branch) ----
    a_bc = agent_for(z_cfg(measure_density_z_actor_reward='kernel'))
    for step in range(3):
        a_bc, inf = a_bc.update(jb(make_batch(seed=step)))
        for k in ('actor/actor_loss', 'actor/q_mask_mean', 'actor/q_mask_frac'):
            assert np.isfinite(float(inf[k])), (step, k, float(inf[k]))
    print("OK")
    """)


# --------------------------------------------------------------------------
# 5. cell (iii): the g path with the latent reward
# --------------------------------------------------------------------------
def test_cell_iii_latent_reward_on_the_g_path():
    """Cell (iii): reward = B̃(u)^T z(g)/sqrt(d) with z(g) = _density_z_encode(goals)
    (norm sqrt(d), rng-free, TARGET tower), while F and the actor still condition on
    the RAW goal. Checked against a direct-module reference; plus the negative
    controls that the arm is neither a no-op (differs from kg) nor secretly
    z-conditioned (the F tower's input width is still the goal's)."""
    _run("""
    cfg_l = g_cfg(measure_density_actor_reward='latent')
    a = agent_for(cfg_l)
    b = make_batch(B=12, seed=13); bj = jb(b)
    blocks = blocks_of(cfg_l)
    goals = bj['actor_goals']

    zg = z_enc_ref(a, goals, blocks, D)
    assert np.allclose(np.linalg.norm(zg, axis=-1), np.sqrt(D), atol=1e-4), \\
        ("z(g) must carry ||z||=sqrt(d)", np.linalg.norm(zg, axis=-1)[:4])

    u_sc = GCIQLFlowAgent._density_queries(bj['next_observations'], blocks)
    tb = np.asarray(a.network.select('target_density_b')(u_sc))
    r_lat = np.einsum('md,bd->bm', tb, zg) / np.sqrt(D)
    ref = q_g_ref(a, cfg_l, bj, goals, r_lat)
    q1, q2 = a._density_actor_q(bj, bj['observations'], bj['actions'], goals)
    assert np.allclose(np.asarray(q1), ref[0], atol=1e-5), (np.asarray(q1)[:3], ref[0][:3])
    assert np.allclose(np.asarray(q2), ref[1], atol=1e-5)

    # not a no-op: the shipped kernel readout on the same weights is different.
    a_k = agent_for(g_cfg())
    qk1, _ = a_k._density_actor_q(bj, bj['observations'], bj['actions'], goals)
    assert not np.allclose(np.asarray(q1), np.asarray(qk1), atol=1e-4), \\
        "latent and kernel readouts coincide -- the knob is not doing anything"

    # z(g) is NOT fed to F/actor: the g-path modules keep goal-width inputs, so an
    # agent built with this knob has the SAME param shapes as cell (i).
    sa = jax.tree_util.tree_map(lambda x: x.shape, a.network.params['modules_density_f'])
    sk = jax.tree_util.tree_map(lambda x: x.shape, a_k.network.params['modules_density_f'])
    assert sa == sk, "cell (iii) changed the F tower's input width -- it must stay g-indexed"
    assert (jax.tree_util.tree_leaves(a.network.params['modules_actor'])[0].shape
            == jax.tree_util.tree_leaves(a_k.network.params['modules_actor'])[0].shape), \\
        "cell (iii) changed the actor's conditioning width -- it must stay g-indexed"
    print("OK")
    """)


# --------------------------------------------------------------------------
# 6. lattice: contradictory / inert combinations refused loudly
# --------------------------------------------------------------------------
def test_lattice_refuses_contradictory_combinations():
    """Every combination in which a spec-28 knob would be provably inert, or would
    collide with the other cell's knob, is refused at create(). This is the
    fake-NULL guard: a cell that trains as something else while logging as itself has
    burned this program before (17b's measure_coef)."""
    _run("""
    def refuses(cfg):
        try:
            agent_for(cfg)
        except AssertionError:
            return True
        return False

    # -- cell (ii) knob --
    assert refuses(g_cfg(measure_density_z_actor_reward='kernel')), \\
        "z knob outside the z lattice must be refused (it would be inert)"
    cfg_nd = dict(get_config(), encoder=None, frame_stack=None)
    cfg_nd['measure_density_z_actor_reward'] = 'kernel'
    assert refuses(cfg_nd), "z knob with no measure_density at all"
    assert refuses(z_cfg(measure_density_z_actor_reward='kernel', actor_bc_only=True)), \\
        "actor_bc_only deletes the q-term whose reward this knob changes"
    assert refuses(z_cfg(measure_density_z_actor_reward='kernel',
                         measure_density_z_rand_frac=1.0)), \\
        "rand_frac=1.0 masks every row -- the q term would be identically zero"
    assert refuses(z_cfg(measure_density_z_actor_reward='kappa')), "unknown enum value"
    assert refuses(z_cfg(measure_density_z_actor_reward=True)), "non-enum value"

    # -- cell (iii) knob --
    assert refuses(dict(get_config(), encoder=None, frame_stack=None,
                        measure_density_actor_reward='latent')), \\
        "g knob without measure_density"
    assert refuses(z_cfg(measure_density_actor_reward='latent')), \\
        "the two cells' knobs are contradictory -- the z path never calls _density_actor_q"
    assert refuses(g_cfg(measure_density_actor_reward='latent',
                         measure_density_behavior=True, next_actions=True,
                         actor_q_center=False)), \\
        "behavior mode has no actor q-term at all"
    assert refuses(g_cfg(measure_density_actor_reward='latent', actor_bc_only=True))
    assert refuses(g_cfg(measure_density_actor_reward='latent', actor_kind='ddpgbc'))
    assert refuses(g_cfg(measure_density_actor_reward='B'))

    # -- accepted --
    assert not refuses(g_cfg()), "cell (i)"
    assert not refuses(z_cfg()), "cell (iv)"
    assert not refuses(z_cfg(measure_density_z_actor_reward='kernel')), "cell (ii)"
    assert not refuses(z_cfg(measure_density_z_actor_reward='kernel',
                             measure_density_z_rand_frac=0.0)), "cell (ii') tie-break"
    assert not refuses(g_cfg(measure_density_actor_reward='latent')), "cell (iii)"
    print("OK")
    """)


# --------------------------------------------------------------------------
# 7. both new cells: real training + the eval entrypoint
# --------------------------------------------------------------------------
def test_cells_ii_and_iii_end_to_end():
    """30 real update() steps for each new cell: finite metrics throughout, no NaNs in
    the param tree, BOTH density towers actually moved (the term is live, not silently
    zeroed), the actor moved, and sample_actions still returns finite in-range actions
    from the trained agent -- the entrypoint a construct-only suite would miss."""
    _run("""
    obs_e = jnp.asarray(np.random.default_rng(1).normal(size=(4, OBS_D)).astype(np.float32))
    g_e = jnp.asarray(np.random.default_rng(2).normal(size=(4, OBS_D)).astype(np.float32))

    for label, cfg, keys in (
        ('cell ii', z_cfg(measure_density_z_actor_reward='kernel'),
         ('measure/density_ce', 'actor/actor_loss', 'actor/q_mean',
          'actor/q_mask_mean', 'actor/q_mask_frac', 'actor/distill_loss')),
        ('cell iii', g_cfg(measure_density_actor_reward='latent'),
         ('measure/density_ce', 'actor/actor_loss', 'actor/q_mean',
          'actor/distill_loss')),
    ):
        a = agent_for(cfg)
        p0 = a.network.params
        for step in range(30):
            a, info = a.update(jb(make_batch(seed=step)))
            for k in keys:
                assert np.isfinite(float(info[k])), (label, step, k, float(info[k]))
        assert all(np.isfinite(np.asarray(l)).all()
                   for l in jax.tree_util.tree_leaves(a.network.params)), \\
            label + ": NaN after 30 updates"
        for mod in ('density_f', 'density_b', 'actor'):
            assert any(not np.allclose(np.asarray(x), np.asarray(y), atol=0)
                       for x, y in zip(
                           jax.tree_util.tree_leaves(p0['modules_' + mod]),
                           jax.tree_util.tree_leaves(a.network.params['modules_' + mod]))), \\
                label + ": " + mod + " did not move after 30 updates"
        ac = np.asarray(a.sample_actions(obs_e, g_e, seed=jax.random.PRNGKey(0)))
        assert ac.shape == (4, ACT_D) and np.isfinite(ac).all() \\
            and (np.abs(ac) <= 1.0 + 1e-6).all(), (label, ac.shape)
        a1 = np.asarray(a.sample_actions(obs_e[0], g_e[0], seed=jax.random.PRNGKey(1)))
        assert a1.shape == (ACT_D,), (label, "unbatched eval step", a1.shape)
    print("OK")
    """)


# --------------------------------------------------------------------------
# 8. the new terms cannot be silently zeroed, and they move gradients
# --------------------------------------------------------------------------
def test_new_terms_are_live_and_have_no_zeroing_coefficient():
    """Neither knob is a coefficient -- there is no dose that could make the new term
    vanish while the run still logs as the cell (the 17b measure_coef fake-NULL
    species). Positively: switching either knob changes the ACTOR GRADIENT, so the new
    reward is actually driving the extraction rather than sitting inert next to it."""
    _run("""
    import inspect
    from gciql_flow import get_config
    cfg = dict(get_config())
    for k, default, allowed in (('measure_density_z_actor_reward', 'latent',
                                 ('latent', 'kernel')),
                                ('measure_density_actor_reward', 'kernel',
                                 ('kernel', 'latent'))):
        assert k in cfg, k + " missing from get_config()"
        assert cfg[k] == default, (k, cfg[k], default)
        assert isinstance(cfg[k], str) and cfg[k] in allowed, \\
            (k, "must be an enum, not a numeric coefficient", cfg[k])

    bj = jb(make_batch(B=16, seed=21))
    rng = jax.random.PRNGKey(9)

    def actor_grad(cfg_d):
        a = agent_for(cfg_d)
        g = jax.grad(lambda p: a.actor_loss(bj, p, rng)[0])(a.network.params)
        return np.concatenate([np.asarray(x).ravel()
                               for x in jax.tree_util.tree_leaves(g['modules_actor'])])

    gz_def = actor_grad(z_cfg())
    gz_ker = actor_grad(z_cfg(measure_density_z_actor_reward='kernel'))
    assert np.abs(gz_ker).max() > 0, "cell (ii) actor gradient is identically zero"
    assert not np.allclose(gz_def, gz_ker, atol=1e-7), \\
        "cell (ii) did not change the actor gradient -- the reward swap is inert"

    gg_def = actor_grad(g_cfg())
    gg_lat = actor_grad(g_cfg(measure_density_actor_reward='latent'))
    assert np.abs(gg_lat).max() > 0, "cell (iii) actor gradient is identically zero"
    assert not np.allclose(gg_def, gg_lat, atol=1e-7), \\
        "cell (iii) did not change the actor gradient -- the reward swap is inert"
    print("OK")
    """)
