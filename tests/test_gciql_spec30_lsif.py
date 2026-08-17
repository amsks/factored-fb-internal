"""Spec 30a Item 3 -- goal-conditioned outcome-space LSIF, the PRIVILEGED-COORDINATES
CONFOUND CONTROL.

Contract: `PAPER/Multistep-FB/specs/30a-impl-wave1.md` Item 3, authority
`specs/30-review-response.md` P2.2 (Rev 1).

    m(s,a,g,x) = F(s,a,g)^T B(phi(x)) / tau          (no softmax anywhere)
    L = 0.5*E_{i != j}[(m_ij - gamma*mbar_ij)^2] - E_i[sum_j k_ij m_ij]
    q = sum_j m(s,a,g,u_j) * kg_j                    (external kernel reward integration)

The arm exists to answer one question -- do Density-FB's gains come from the NORMALIZED
measure, or merely from predicting in phi coordinates? -- and that inference dies the
moment anything OTHER than the two intended axes (LSIF-vs-CE, unnormalized-vs-normalized)
differs from the Density-FB arm. So the suite is organized around MATCHING, not around the
loss formula alone:

  test 2  the loss is exactly the LSIF closed form (numpy float64, from the agent's own
          towers, including the flow bootstrap action reconstructed from the same rng);
  test 3  MATCHING: the query support, the immediate kernel k, the bootstrap action and
          the target towers are the SAME objects density_loss uses on the same batch and
          the same rng -- checked term by term, not by inspection;
  test 4  the two intended axes are real: m is not a probability vector, and the actor's
          readout is the kernel integral of m (which differs from the softmax readout);
  test 6  MUTATION: normalization creeping back in, the diagonal entering the quadratic,
          and the discount dropped from the bootstrap are each compiled and must be caught.

House rules: default-OFF and bit-identical when off over THREE REAL update() steps +
sample_actions against the pinned pre-implementation commit (test 1); lattice refuses every
inert or match-breaking combination (test 5).

Runner: `.venv/bin/python -m pytest tests/test_gciql_spec30_lsif.py -x -q`
"""
import subprocess
import textwrap

import pytest

from tests.jax_subprocess import run_jax, REPO

OBS_D = 40
D = 8
ACT_D = 5

# The last commit BEFORE the spec-30a Item 2/3 implementation.
PRE_SHA = "7c096f33b197515437ededc41cffb2929af40a0d"

NEW_KNOBS = ("measure_lsif", "measure_lsif_immediate", "measure_tdinfonce")


def _baseline_reachable(sha):
    r = subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"],
                       cwd=str(REPO), capture_output=True, text=True)
    return r.returncode == 0


COMMON = f"""
    import jax, jax.numpy as jnp, numpy as np
    from gciql_flow import GCIQLFlowAgent, get_config, _parse_hit_spec

    OBS_D = {OBS_D}
    D = {D}
    ACT_D = {ACT_D}
    REPO_DIR = {str(REPO)!r}
    EX_OBS = jnp.zeros((1, OBS_D), jnp.float32)
    EX_ACT = jnp.zeros((1, ACT_D), jnp.float32)

    def g_cfg(**kw):
        # THE DENSITY-FB ARM this control must match in everything but two axes.
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

    def lsif_cfg(**kw):
        return g_cfg(measure_lsif=True, **kw)

    def make_batch(B=16, seed=0):
        rng = np.random.default_rng(seed)
        def n(*s):
            return rng.normal(size=s).astype(np.float32)
        return dict(
            observations=n(B, OBS_D),
            next_observations=n(B, OBS_D),
            actions=np.clip(rng.normal(size=(B, ACT_D)), -1, 1).astype(np.float32),
            value_goals=n(B, OBS_D),
            actor_goals=n(B, OBS_D),
            rewards=np.zeros(B, np.float32),
            masks=np.ones(B, np.float32),
        )

    def agent_for(cfg, seed=0):
        return GCIQLFlowAgent.create(seed, EX_OBS, EX_ACT, cfg)

    def jb(b):
        return {{k: jnp.asarray(v) for k, v in b.items()}}

    def blocks_of(cfg):
        return _parse_hit_spec(cfg["measure_hit_spec"])

    def parts(agent, cfg, b, key):
        \"\"\"Every ingredient of the loss, recomputed independently: the query support, the
        online/target bilinear matrices, the flow bootstrap action off the SAME rng, and
        the immediate kernel. Returned so the matching test can compare them against the
        objects density_loss builds from the same batch.\"\"\"
        blocks = blocks_of(cfg)
        bj = jb(b)
        u_raw = GCIQLFlowAgent._raw_block_cols(bj['next_observations'], blocks)
        u_sc = GCIQLFlowAgent._density_queries(bj['next_observations'], blocks)
        tau = float(cfg['measure_density_temp'])
        b_emb = np.asarray(agent.network.select('density_b')(u_sc), np.float64)
        f_emb = np.asarray(agent.network.select('density_f')(jnp.concatenate(
            [bj['observations'], bj['actions'], bj['value_goals']], -1)), np.float64)
        m = np.einsum('hbd,md->hbm', f_emb, b_emb) / tau
        noise = jax.random.normal(key, bj['actions'].shape)
        a_plus = np.asarray(agent.network.select('actor')(
            bj['next_observations'], bj['value_goals'], noise), np.float64)
        tf = np.asarray(agent.network.select('target_density_f')(jnp.concatenate(
            [bj['next_observations'], jnp.asarray(a_plus, jnp.float32),
             bj['value_goals']], -1)), np.float64)
        tb = np.asarray(agent.network.select('target_density_b')(u_sc), np.float64)
        m_bar = np.einsum('hbd,md->hbm', tf, tb) / tau
        k = np.asarray(GCIQLFlowAgent._delta_phi(bj['next_observations'], u_raw, blocks),
                       np.float64)
        return dict(u_sc=np.asarray(u_sc, np.float64), m=m, m_bar=m_bar, k=k,
                    a_plus=a_plus, b_emb=b_emb, f_emb=f_emb)

    def lsif_ref(agent, cfg, b, key, immediate=None, exclude_diag=True, softmax_m=False):
        \"\"\"The LSIF closed form. The three flags are the CONTROLS: `immediate` picks the
        attraction object, `exclude_diag` is FB's split, `softmax_m` is the normalization
        this control exists to NOT have.\"\"\"
        p = parts(agent, cfg, b, key)
        gamma = float(cfg['discount'])
        m, m_bar = p['m'], p['m_bar']
        if softmax_m:
            # the ONLINE matrix only -- mirroring mutant m1, which normalizes the online
            # score and leaves the target matrix as the raw bilinear one.
            e = np.exp(m - m.max(-1, keepdims=True)); m = e / e.sum(-1, keepdims=True)
        imm = immediate or str(cfg.get('measure_lsif_immediate', 'kernel'))
        Bn = m.shape[1]
        k = np.eye(Bn) if imm == 'dirac' else p['k']
        off = 1.0 - np.eye(Bn)
        diff = m - gamma * m_bar
        w = off if exclude_diag else np.ones_like(off)
        offdiag = 0.5 * ((diff * w) ** 2).sum((1, 2)) / off.sum()
        attract = -(k[None] * m).sum(-1).mean(-1)
        return float((offdiag + attract).mean())
"""


def _run(body, **kw):
    code = textwrap.dedent(COMMON) + textwrap.dedent(body)
    res = run_jax(code, **kw)
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    return res


# --------------------------------------------------------------------------
# 1. all-defaults bit-identity to the pre-implementation agent
# --------------------------------------------------------------------------
def test_default_off_bit_identical_to_pre_item23():
    """measure_lsif off: param tree, total_loss, three real update() steps and
    sample_actions bit-identical to the pinned commit, on the Density-FB arm this control
    is matched to, the hard ablation, the ddpgbc/awr actors and a bare config."""
    if not _baseline_reachable(PRE_SHA):
        pytest.skip(f"pinned baseline {PRE_SHA} is not reachable in this checkout")
    _run(f"""
    PRE_SHA = {PRE_SHA!r}
    NEW_KNOBS = {NEW_KNOBS!r}
    """ + """
    import subprocess, importlib.util, tempfile, os

    src = subprocess.run(["git", "show", PRE_SHA + ":tools/wandb_mode_shim/gciql_flow.py"],
                         cwd=REPO_DIR, capture_output=True, text=True, check=True).stdout
    for k in NEW_KNOBS:
        assert k not in src, ("pinned baseline already contains " + k + " -- re-pin PRE_SHA")
    path = os.path.join(tempfile.mkdtemp(), "gciql_flow_pre.py")
    open(path, "w").write(src)
    spec = importlib.util.spec_from_file_location("gciql_flow_pre", path)
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

    def same(cfg_kw, label, n=3):
        a_b = baseline.GCIQLFlowAgent.create(0, EX_OBS, EX_ACT,
                                             cfg_of(baseline.get_config, **cfg_kw))
        a_n = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg_of(get_config, **cfg_kw))
        assert set(a_b.network.params) == set(a_n.network.params), label
        l_b, i_b = a_b.total_loss(bj, a_b.network.params, jax.random.PRNGKey(7))
        l_n, i_n = a_n.total_loss(bj, a_n.network.params, jax.random.PRNGKey(7))
        assert float(l_b) == float(l_n), (label, float(l_b), float(l_n))
        assert sorted(i_b) == sorted(i_n), (label, set(i_b) ^ set(i_n))
        for step in range(n):
            bb = jb(make_batch(seed=step))
            a_b, ib = a_b.update(bb)
            a_n, inn = a_n.update(bb)
            assert float(ib['measure/density_ce']) == float(inn['measure/density_ce'])
        assert all((x == y).all() for x, y in zip(
            jax.tree_util.tree_leaves(a_b.network.params),
            jax.tree_util.tree_leaves(a_n.network.params))), \\
            label + ": param tree diverged after " + str(n) + " real update() steps"
        ac_b = np.asarray(a_b.sample_actions(obs_e, g_e, seed=jax.random.PRNGKey(0)))
        ac_n = np.asarray(a_n.sample_actions(obs_e, g_e, seed=jax.random.PRNGKey(0)))
        assert (ac_b == ac_n).all(), label + ": sample_actions diverged"

    same(dict(), "Density-FB arm (the matched sibling)")
    same(dict(measure_density_hard=True), "density HARD ablation")
    same(dict(actor_kind='ddpgbc'), "density + ddpgbc")
    print("OK")
    """)


# --------------------------------------------------------------------------
# 2. the loss is the LSIF closed form
# --------------------------------------------------------------------------
def test_loss_matches_the_lsif_closed_form():
    """0.5*E_{i != j}[(m - gamma*mbar)^2] - E_i[sum_j k_ij m_ij], both immediate objects."""
    _run("""
    for imm in ('kernel', 'dirac'):
        cfg = lsif_cfg(measure_lsif_immediate=imm)
        a = agent_for(cfg)
        for s in range(3):
            a, _ = a.update(jb(make_batch(B=32, seed=s)))
        b = make_batch(B=32, seed=11)
        key = jax.random.PRNGKey(4)
        loss, info = a.lsif_loss(jb(b), a.network.params, key)
        ref = lsif_ref(a, cfg, b, key)
        rel = abs(float(loss) - ref) / max(1.0, abs(ref))
        assert rel < 1e-5, (imm, float(loss), ref)
        assert abs(float(info['lsif_offdiag']) + float(info['lsif_attract'])
                   - float(loss)) < 1e-3 * max(1.0, abs(ref))
        # the two immediate objects are genuinely different targets
        other = 'dirac' if imm == 'kernel' else 'kernel'
        assert abs(ref - lsif_ref(a, cfg, b, key, immediate=other)) > 1e-3
    print("OK")
    """)


# --------------------------------------------------------------------------
# 3. MATCHING to Density-FB, term by term
# --------------------------------------------------------------------------
def test_matched_to_density_fb_in_everything_but_the_two_axes():
    """The confound control is only a control if nothing else moved. Same batch, same rng,
    same seed: the query support, the immediate kernel k, the bootstrap action a+ and the
    target towers must be the SAME objects the Density-FB arm builds -- and the two arms'
    param trees must be structurally identical, so the only differences are the loss and
    the readout normalization."""
    _run("""
    cfg_d, cfg_l = g_cfg(), lsif_cfg()
    ad, al = agent_for(cfg_d), agent_for(cfg_l)
    assert set(ad.network.params) == set(al.network.params), \\
        "the LSIF arm registers a different module set than Density-FB"
    assert all((x == y).all() for x, y in zip(
        jax.tree_util.tree_leaves(ad.network.params),
        jax.tree_util.tree_leaves(al.network.params))), \\
        "same seed, same init required -- the arms would not be comparable otherwise"
    for k in ('measure_hit_spec', 'measure_density_dim', 'measure_density_temp',
              'discount', 'actor_kind', 'bc_coeff', 'actor_q_center', 'batch_size',
              'value_p_curgoal', 'value_p_trajgoal', 'value_p_randomgoal',
              'actor_p_trajgoal', 'actor_geom_sample', 'lr', 'tau'):
        assert cfg_d[k] == cfg_l[k], ("unmatched config axis", k, cfg_d[k], cfg_l[k])
    diff = {k for k in cfg_d if cfg_d[k] != cfg_l[k]}
    assert diff == {'measure_lsif'}, ("the arms differ in more than the objective", diff)

    b = make_batch(B=32, seed=3)
    key = jax.random.PRNGKey(9)
    p = parts(al, cfg_l, b, key)
    # the query support IS the batch's own next-state block coords (Density-FB's object)
    blocks = blocks_of(cfg_l)
    assert np.allclose(p['u_sc'], np.asarray(
        GCIQLFlowAgent._density_queries(jb(b)['next_observations'], blocks), np.float64))
    # k IS _delta_phi(nobs, u_raw) -- rows on the simplex, the same immediate object
    assert np.allclose(p['k'].sum(-1), 1.0, atol=1e-5)
    # the bootstrap action is the SAME one-shot flow draw density_loss makes off this rng
    dl, _ = ad.density_loss(jb(b), ad.network.params, key)
    noise = jax.random.normal(key, jb(b)['actions'].shape)
    a_dens = np.asarray(ad.network.select('actor')(
        jb(b)['next_observations'], jb(b)['value_goals'], noise), np.float64)
    assert np.allclose(p['a_plus'], a_dens), \\
        "the LSIF bootstrap action differs from Density-FB's -- unmatched"
    assert np.isfinite(float(dl))
    print("OK")
    """)


# --------------------------------------------------------------------------
# 4. the two intended axes
# --------------------------------------------------------------------------
def test_unnormalized_measure_and_external_kernel_readout():
    """Axis 2: m is an UNNORMALIZED ratio (its rows are not a probability vector) and the
    actor integrates it externally, q = sum_j m_j kg_j -- `_density_actor_q` with m in
    place of w. Both are checked in closed form, and against the softmax readout they must
    NOT reproduce."""
    _run("""
    cfg = lsif_cfg()
    a = agent_for(cfg)
    for s in range(5):
        a, _ = a.update(jb(make_batch(B=32, seed=s)))
    b = make_batch(B=32, seed=17)
    bj = jb(b)
    key = jax.random.PRNGKey(1)
    p = parts(a, cfg, b, key)
    rows = p['m'].sum(-1)
    assert np.abs(rows - 1.0).min() > 1e-2, "m looks normalized -- the axis is not real"
    assert (p['m'] < 0).mean() > 0.01, \\
        "no negative entries at all: m is behaving like a softmax output"

    blocks = blocks_of(cfg)
    kg = np.asarray(GCIQLFlowAgent._delta_phi(
        bj['actor_goals'],
        GCIQLFlowAgent._raw_block_cols(bj['next_observations'], blocks), blocks), np.float64)
    q = np.asarray(a._lsif_actor_q(bj, bj['observations'], bj['actions'], bj['actor_goals']),
                   np.float64)
    f2 = np.asarray(a.network.select('density_f')(jnp.concatenate(
        [bj['observations'], bj['actions'], bj['actor_goals']], -1)), np.float64)
    m2 = np.einsum('hbd,md->hbm', f2, p['b_emb']) / float(cfg['measure_density_temp'])
    assert np.abs(q - (m2 * kg[None]).sum(-1)).max() < 1e-4, "the readout is not E_x[m*kg]"

    # the softmax readout (Density-FB's) is a different number on the same towers
    e = np.exp(m2 - m2.max(-1, keepdims=True)); w = e / e.sum(-1, keepdims=True)
    assert np.abs(q - (w * kg[None]).sum(-1)).max() > 1e-3, \\
        "the LSIF readout coincides with the softmax readout -- axis 2 is absent"
    q_dens = np.asarray(a._density_actor_q(bj, bj['observations'], bj['actions'],
                                           bj['actor_goals']), np.float64)
    assert np.abs(q_dens - (w * kg[None]).sum(-1)).max() < 1e-5, \\
        "_density_actor_q is not the softmax readout it is supposed to be"
    print("OK")
    """)


# --------------------------------------------------------------------------
# 5. lattice
# --------------------------------------------------------------------------
def test_lattice_refuses_inert_and_match_breaking_combinations():
    _run("""
    def refuses(cfg):
        try:
            agent_for(cfg)
        except AssertionError:
            return True
        return False

    assert refuses(dict(get_config(), encoder=None, frame_stack=None,
                        measure_lsif_immediate='dirac')), "knob without its owner"
    assert refuses(dict(get_config(), encoder=None, frame_stack=None, measure_lsif=True)), \\
        "no density towers to train"
    assert refuses(g_cfg(measure_lsif=True, measure_only=False))
    assert refuses(g_cfg(measure_lsif=True, measure_tdinfonce=True))
    assert refuses(lsif_cfg(measure_density_hard=True)), "read only in density_loss"
    assert refuses(lsif_cfg(measure_density_dup_instrument=True))
    assert refuses(lsif_cfg(measure_density_z=True))
    assert refuses(lsif_cfg(measure_density_behavior=True))
    assert refuses(lsif_cfg(actor_bc_only=True))
    assert refuses(lsif_cfg(measure_density_actor_reward='latent'))
    assert refuses(lsif_cfg(measure_lsif_immediate='soft')), "enum hygiene"
    assert refuses(lsif_cfg(measure_scale_norm=True)), \\
        "a third unmatched difference from Density-FB"
    bad_kind = lsif_cfg(); bad_kind['actor_kind'] = 'ddpgbc'
    assert refuses(bad_kind), "the control must carry Density-FB's flow extraction"
    bad_q = lsif_cfg(); bad_q['actor_q_center'] = False
    assert refuses(bad_q)

    assert not refuses(lsif_cfg()), "lsif_cd_phi_s0"
    assert not refuses(lsif_cfg(measure_lsif_immediate='dirac'))
    assert not refuses(g_cfg()), "Density-FB arm untouched"
    print("OK")
    """)


# --------------------------------------------------------------------------
# 6. MUTATION
# --------------------------------------------------------------------------
def test_mutants_are_caught():
    """m1 the softmax creeping back onto m (the axis this control exists to remove);
    m2 the diagonal entering the quadratic (FB's split deleted);
    m3 the discount dropped from the bootstrap term.

    Note on the stop-gradient over m_bar: it is BELT-AND-BRACES, not load-bearing, because
    the target towers are called at frozen params (never params=grad_params) and are
    already outside jax.grad's path -- removing it is provably a no-op, so it is not
    mutated. What guarantees the property (zero gradient into modules_target_density_*) is
    asserted directly in test 7."""
    _run("""
    import importlib.util, tempfile, os
    real_path = os.path.join(REPO_DIR, "tools/wandb_mode_shim/gciql_flow.py")
    real = open(real_path).read()

    def mutate(old, new, tag):
        assert real.count(old) == 1, (tag, "anchor not unique: " + old)
        p = os.path.join(tempfile.mkdtemp(), "gciql_flow_" + tag + ".py")
        open(p, "w").write(real.replace(old, new))
        spec = importlib.util.spec_from_file_location("gf_" + tag, p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    cfg = lsif_cfg()
    b = make_batch(B=32, seed=13)
    key = jax.random.PRNGKey(2)

    def scored(mod):
        a = mod.GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, dict(cfg))
        for s in range(3):
            a, _ = a.update(jb(make_batch(B=32, seed=s)))
        return a, float(a.lsif_loss(jb(b), a.network.params, key)[0])

    import gciql_flow as real_mod
    a_real, l_real = scored(real_mod)
    ref = lsif_ref(a_real, cfg, b, key)
    assert abs(l_real - ref) / max(1.0, abs(ref)) < 1e-5

    m1 = mutate("        m = jnp.einsum('hbd,md->hbm', f_emb, b_emb) / tau"
                "                     # [2, B, M]",
                "        m = jax.nn.softmax(jnp.einsum('hbd,md->hbm', f_emb, b_emb) / tau, -1)"
                "  # [2, B, M]", "m1")
    m2 = mutate("offdiag = 0.5 * jnp.sum((diff * off[None]) ** 2, axis=(1, 2)) / jnp.sum(off)   # [2]",
                "offdiag = 0.5 * jnp.sum(diff ** 2, axis=(1, 2)) / jnp.sum(off)   # [2]", "m2")
    m3 = mutate("diff = m - gamma * m_bar                                              # [2, B, M]",
                "diff = m - m_bar                                              # [2, B, M]", "m3")

    for tag, mod, ctrl in (("m1", m1, dict(softmax_m=True)),
                           ("m2", m2, dict(exclude_diag=False))):
        a_m, l_m = scored(mod)
        assert abs(l_m - ref) > 1e-3 * max(1.0, abs(ref)), \\
            (tag, "mutant scores the reference loss -- the closed form has no teeth")
        assert abs(l_m - lsif_ref(a_m, cfg, b, key, **ctrl)) / max(1.0, abs(ref)) < 1e-4, \\
            (tag, "the mutant does not match its own control formula -- the control is wrong")

    a3, l3 = scored(m3)
    assert abs(l3 - ref) > 1e-3 * max(1.0, abs(ref)), \\
        ("m3", "dropping the discount from the bootstrap term scores the reference loss")
    print("OK")
    """)


# --------------------------------------------------------------------------
# 7. end to end
# --------------------------------------------------------------------------
def test_arm_end_to_end():
    """30 real update() steps of lsif_cd_phi_s0: finite everywhere, towers and actor move,
    the validity instruments are logged, the actor's q-term is the LSIF readout, and no
    gradient reaches the target towers."""
    _run("""
    cfg = lsif_cfg()
    a = agent_for(cfg)
    p0 = jax.tree_util.tree_map(lambda x: np.asarray(x).copy(), a.network.params)
    for s in range(30):
        a, info = a.update(jb(make_batch(B=32, seed=s)))
        for k, v in info.items():
            assert np.isfinite(float(v)), (s, k, float(v))
    for mod in ('modules_density_f', 'modules_density_b', 'modules_actor',
                'modules_actor_vf'):
        assert any(not np.allclose(np.asarray(x), y) for x, y in zip(
            jax.tree_util.tree_leaves(a.network.params[mod]),
            jax.tree_util.tree_leaves(p0[mod]))), mod + " never moved"
    for k in ('measure/lsif_loss', 'measure/lsif_offdiag', 'measure/lsif_attract',
              'measure/lsif_m_msq', 'measure/lsif_neg_frac', 'measure/lsif_diag_mean',
              'actor/q_mean', 'actor/bc_flow_loss', 'actor/distill_loss'):
        assert k in info, k
    assert 'measure/density_ce' not in info, "density_loss must not also run"

    def loss_fn(params):
        return a.total_loss(jb(make_batch(B=32, seed=1)), params, jax.random.PRNGKey(0))[0]
    grads = jax.grad(loss_fn)(a.network.params)
    for mod in ('modules_target_density_f', 'modules_target_density_b'):
        assert all(np.allclose(np.asarray(g), 0.0)
                   for g in jax.tree_util.tree_leaves(grads[mod])), mod + " got a gradient"

    ac = np.asarray(a.sample_actions(
        jnp.asarray(np.random.default_rng(0).normal(size=(4, OBS_D)).astype(np.float32)),
        jnp.asarray(np.random.default_rng(1).normal(size=(4, OBS_D)).astype(np.float32)),
        seed=jax.random.PRNGKey(0)))
    assert np.isfinite(ac).all() and np.abs(ac).max() <= 1.0
    print("OK")
    """)
