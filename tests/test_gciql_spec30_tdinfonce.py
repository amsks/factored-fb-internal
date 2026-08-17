"""Spec 30a Item 2 -- the PAPER-EXACT TD InfoNCE control.

Contract: `PAPER/Multistep-FB/specs/30a-impl-wave1.md` Item 2, authority
`specs/30-review-response.md` P2.1 (Rev 1). This arm is not a variant of ours: it answers
"how does the published method (Zheng, Salakhutdinov & Eysenbach, ICLR 2024,
arXiv:2310.20141) do in our setting?", so faithfulness beats performance and every place
their conventions and ours disagree is resolved in THEIR favour and recorded.

The spec-30 audit isolated three implementation deltas between their Eq. 10 and our
density_loss, and this suite tests each one against a control that would silently pass if
it had been implemented our way instead:

  1. the IMMEDIATE term's positive slot is the dynamics next state s', the BOOTSTRAP
     term's is a MARGINAL draw s^(1) soft-labelled by the stop-grad target softmax
     (test 2 closed form; test 3's slot-swap control; test 6's mutants);
  2. ONE shared negative pool s^(2:N) drawn iid from the dataset FUTURE-STATE marginal --
     a new query source, since our standing support is the batch's own next states
     (test 4, plus the GCDataset test 8);
  3. both towers l2-NORMALIZED, temperature implicit (test 5).

Their extraction is tested too (test 7): Eq. 11's goal-contrastive CE with the COMMANDED
GOAL as the positive outcome, mixed convexly with App. D.3's MSE goal-BC -- no q-scale
normalization, no flow distillation, and their geometric future-state relabeling, all
lattice-pinned (test 9) so the arm cannot silently drift back to our recipe.

House rules: every knob default-OFF and bit-identical when off, proven over THREE REAL
update() steps + sample_actions against the pinned pre-implementation commit (test 1),
and MUTATION-TESTED (test 6: the closed-form check is re-run against deliberately broken
copies of the agent source, and must fail on each).

Same subprocess-jax harness as tests/test_gciql_spec30_centered_index.py: OBS_D=40,
action dim 5, d=8, hit_spec '19:22:0.4'.

Runner: `.venv/bin/python -m pytest tests/test_gciql_spec30_tdinfonce.py -x -q`
"""
import subprocess
import textwrap

import pytest

from tests.jax_subprocess import run_jax, REPO

OBS_D = 40
D = 8
ACT_D = 5

# The last commit BEFORE the spec-30a Item 2/3 implementation (item 1's watcher commit).
PRE_SHA = "7c096f33b197515437ededc41cffb2929af40a0d"

NEW_KNOBS = ("measure_tdinfonce", "measure_tdinfonce_actor", "measure_tdinfonce_bc_lambda",
             "measure_tdinfonce_negatives", "marginal_states", "measure_lsif",
             "measure_lsif_immediate")


def _baseline_reachable(sha):
    r = subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"],
                       cwd=str(REPO), capture_output=True, text=True)
    return r.returncode == 0


COMMON = f"""
    import jax, jax.numpy as jnp, numpy as np
    from gciql_flow import GCIQLFlowAgent, get_config, _parse_hit_spec, _l2n

    OBS_D = {OBS_D}
    D = {D}
    ACT_D = {ACT_D}
    REPO_DIR = {str(REPO)!r}
    EX_OBS = jnp.zeros((1, OBS_D), jnp.float32)
    EX_ACT = jnp.zeros((1, ACT_D), jnp.float32)

    def g_cfg(**kw):
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

    def tdi_paper_cfg(**kw):
        # the wave-1 arm: THEIR critic AND THEIR extraction.
        cfg = g_cfg(measure_tdinfonce=True, actor_kind="ddpgbc", **kw)
        cfg["actor_q_center"] = False
        cfg["actor_p_curgoal"] = 0.0
        cfg["actor_p_trajgoal"] = 1.0
        cfg["actor_p_randomgoal"] = 0.0
        cfg["actor_geom_sample"] = True      # THEIR relabeling distribution
        return cfg

    def tdi_matched_cfg(**kw):
        # the promotion-wave arm: their critic, OUR readout + flow actor. Not launched.
        return g_cfg(measure_tdinfonce=True, measure_tdinfonce_actor="matched", **kw)

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
            marginal_states=n(B, OBS_D),
            marginal_positives=n(B, OBS_D),
            rewards=np.zeros(B, np.float32),
            masks=np.ones(B, np.float32),
        )

    def agent_for(cfg, seed=0):
        return GCIQLFlowAgent.create(seed, EX_OBS, EX_ACT, cfg)

    def jb(b):
        return {{k: jnp.asarray(v) for k, v in b.items()}}

    def blocks_of(cfg):
        return _parse_hit_spec(cfg["measure_hit_spec"])

    def l2(x):
        x = np.asarray(x, np.float64)
        return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)

    def logsm(x):
        x = np.asarray(x, np.float64)
        m = x.max(-1, keepdims=True)
        return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))

    def tdi_ref(agent, cfg, b, pool_key='marginal_states', pos_key='marginal_positives',
                normalize=True, boot_slot='marginal'):
        \"\"\"The paper-exact Eq. 10, recomputed in numpy float64 from the agent's OWN
        towers. `normalize`/`boot_slot`/`pool_key` are the CONTROLS: flipping any one of
        them is exactly one of the three audited deltas, and must change the number.\"\"\"
        blocks = blocks_of(cfg)
        gamma = float(cfg['discount'])
        q = GCIQLFlowAgent._density_queries
        pool_sc = q(jnp.asarray(b[pool_key]), blocks)
        pos_sc = q(jnp.asarray(b[pos_key]), blocks)
        nxt_sc = q(jnp.asarray(b['next_observations']), blocks)
        db = lambda x: np.asarray(agent.network.select('density_b')(x), np.float64)
        tdb = lambda x: np.asarray(agent.network.select('target_density_b')(x), np.float64)
        nrm = l2 if normalize else (lambda x: np.asarray(x, np.float64))
        b_pool, b_pos, b_nxt = nrm(db(pool_sc)), nrm(db(pos_sc)), nrm(db(nxt_sc))
        f = nrm(np.asarray(agent.network.select('density_f')(jnp.concatenate(
            [jnp.asarray(b['observations']), jnp.asarray(b['actions']),
             jnp.asarray(b['value_goals'])], -1)), np.float64))              # [2,B,d]
        neg = np.einsum('hbd,pd->hbp', f, b_pool)
        boot_b = b_pos if boot_slot == 'marginal' else b_nxt
        l_imm = np.concatenate([np.einsum('hbd,bd->hb', f, b_nxt)[..., None], neg], -1)
        l_boot = np.concatenate([np.einsum('hbd,bd->hb', f, boot_b)[..., None], neg], -1)
        # a' ~ pi(s', g): the deterministic gaussian actor on the paper arm.
        if cfg['actor_kind'] == 'ddpgbc':
            dist = agent.network.select('actor')(jnp.asarray(b['next_observations']),
                                                 jnp.asarray(b['value_goals']))
            a_plus = np.clip(np.asarray(dist.mode(), np.float64), -1, 1)
        else:
            return None    # the flow bootstrap consumes the loss's own rng; not modelled
        tf = nrm(np.asarray(agent.network.select('target_density_f')(jnp.concatenate(
            [jnp.asarray(b['next_observations']), jnp.asarray(a_plus, jnp.float32),
             jnp.asarray(b['value_goals'])], -1)), np.float64))
        t_pool, t_pos, t_nxt = nrm(tdb(pool_sc)), nrm(tdb(pos_sc)), nrm(tdb(nxt_sc))
        t_boot = t_pos if boot_slot == 'marginal' else t_nxt
        t_logits = np.concatenate([np.einsum('hbd,bd->hb', tf, t_boot)[..., None],
                                   np.einsum('hbd,pd->hbp', tf, t_pool)], -1)
        w = np.exp(logsm(t_logits))
        ce_imm = -logsm(l_imm)[..., 0].mean()
        ce_boot = -(w * logsm(l_boot)).sum(-1).mean()
        return (1.0 - gamma) * ce_imm + gamma * ce_boot
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
    """Every Item-2/3 knob at its default: param tree, total_loss, THREE REAL update()
    steps and sample_actions bit-identical to the agent loaded from the pinned commit --
    on the g-indexed density arm, the HARD ablation, the spec-24 z arm, the spec-30
    centered arm, the ddpgbc density arm (whose actor_loss branch chain this spec edits)
    and a config with no measure knobs at all."""
    if not _baseline_reachable(PRE_SHA):
        pytest.skip(f"pinned baseline {PRE_SHA} is not reachable in this checkout")
    _run(f"""
    PRE_SHA = {PRE_SHA!r}
    NEW_KNOBS = {NEW_KNOBS!r}
    """ + """
    import subprocess, importlib.util, tempfile, os

    src = subprocess.run(["git", "show", PRE_SHA + ":tools/wandb_mode_shim/gciql_flow.py"],
                         cwd=REPO_DIR, capture_output=True, text=True, check=True).stdout
    assert "measure_density_z_prompt" in src, \\
        "pinned baseline predates spec 30 P1 -- re-pin PRE_SHA"
    for k in NEW_KNOBS:
        assert k not in src, (
            "pinned baseline already contains " + k + " -- the pin has gone stale; re-pin "
            "PRE_SHA to the last commit before the spec-30a Item 2/3 implementation")
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
        assert all((x == y).all() for x, y in zip(
            jax.tree_util.tree_leaves(a_b.network.params),
            jax.tree_util.tree_leaves(a_n.network.params))), label + ": param tree"
        l_b, i_b = a_b.total_loss(bj, a_b.network.params, jax.random.PRNGKey(7))
        l_n, i_n = a_n.total_loss(bj, a_n.network.params, jax.random.PRNGKey(7))
        assert float(l_b) == float(l_n), (label, float(l_b), float(l_n))
        assert sorted(i_b) == sorted(i_n), (label, set(i_b) ^ set(i_n))
        for step in range(n):
            bb = jb(make_batch(seed=step))
            a_b, ib = a_b.update(bb)
            a_n, inn = a_n.update(bb)
            assert sorted(ib) == sorted(inn), (label, step, set(ib) ^ set(inn))
            for k in ('actor/actor_loss',):
                assert float(ib[k]) == float(inn[k]), (label, step, k)
        assert all((x == y).all() for x, y in zip(
            jax.tree_util.tree_leaves(a_b.network.params),
            jax.tree_util.tree_leaves(a_n.network.params))), \\
            label + ": param tree diverged after " + str(n) + " real update() steps"
        ac_b = np.asarray(a_b.sample_actions(obs_e, g_e, seed=jax.random.PRNGKey(0)))
        ac_n = np.asarray(a_n.sample_actions(obs_e, g_e, seed=jax.random.PRNGKey(0)))
        assert (ac_b == ac_n).all(), label + ": sample_actions diverged"

    same(dict(), "g-indexed density arm")
    same(dict(measure_density_hard=True), "density HARD ablation")
    same(dict(actor_kind='ddpgbc'), "density + ddpgbc (the edited branch chain)")
    same(dict(measure_density_z=True), "spec-24 z arm")
    same(dict(measure_density_z=True, measure_density_z_encode='dual_centered',
              measure_density_z_enc_tau=0.0005, measure_density_z_actor_reward='kernel',
              measure_density_z_eval_pool_m=8),   # the 16-row test batch draws the pool
         "spec-30 P1 centered arm")

    cb = dict(baseline.get_config()); cb["encoder"] = None; cb["frame_stack"] = None
    cn = dict(get_config()); cn["encoder"] = None; cn["frame_stack"] = None
    ab = baseline.GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cb)
    an = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cn)
    assert set(ab.network.params) == set(an.network.params)
    assert all((x == y).all() for x, y in zip(
        jax.tree_util.tree_leaves(ab.network.params),
        jax.tree_util.tree_leaves(an.network.params))), "no-knob config"
    print("OK")
    """)


# --------------------------------------------------------------------------
# 2. the critic IS their Eq. 10 (closed form)
# --------------------------------------------------------------------------
def test_critic_matches_eq10_closed_form():
    """tdinfonce_loss == (1-gamma)*CE(e_1; softmax over [s', pool]) +
    gamma*CE(sg[w]; softmax over [s^(1), pool]), w = target softmax over the SAME
    candidates -- checked against an independent numpy float64 recomputation from the
    agent's own towers, after real training steps (not at init, where every tower is
    interchangeable)."""
    _run("""
    cfg = tdi_paper_cfg()
    a = agent_for(cfg)
    for s in range(3):
        a, _ = a.update(jb(make_batch(B=32, seed=s)))
    b = make_batch(B=32, seed=11)
    loss, info = a.tdinfonce_loss(jb(b), a.network.params, jax.random.PRNGKey(3))
    ref = tdi_ref(a, cfg, b)
    assert abs(float(loss) - ref) < 1e-5, (float(loss), ref)
    assert abs(float(info['tdi_loss']) - ref) < 1e-5
    # the two terms are separately identified (a bug that swapped their weights would
    # still match the sum only by coincidence)
    gamma = float(cfg['discount'])
    assert abs((1 - gamma) * float(info['tdi_ce_immediate'])
               + gamma * float(info['tdi_ce_bootstrap']) - float(loss)) < 1e-6
    # N = 1 + P: the shared pool plus the row's own positive slot
    assert float(info['tdi_n_candidates']) == 33.0
    print("OK")
    """)


# --------------------------------------------------------------------------
# 3. delta 1: the two terms label DIFFERENT slots
# --------------------------------------------------------------------------
def test_bootstrap_slot_is_a_marginal_draw_not_the_next_state():
    """Their immediate term labels s' but their bootstrap term labels a MARGINAL draw
    (that is the whole point of the "one shared pool + two positive slots" structure).

    The check is an EQUALITY against two references, not a magnitude test: the agent must
    score the boot_slot='marginal' reference, and substituting s' into that slot in the
    BATCH must reproduce the boot_slot='next' reference exactly. Magnitudes are the wrong
    instrument here -- a near-uniform classifier over N=33 candidates moves the scalar loss
    by only ~1e-4 when one candidate is swapped, while the closed-form agreement is ~5e-7,
    so the two references are separated by ~300x the agreement noise but nowhere near 1e-3.
    """
    _run("""
    cfg = tdi_paper_cfg()
    a = agent_for(cfg)
    for s in range(3):
        a, _ = a.update(jb(make_batch(B=32, seed=s)))
    b = make_batch(B=32, seed=5)
    loss = float(a.tdinfonce_loss(jb(b), a.network.params, jax.random.PRNGKey(0))[0])
    theirs = tdi_ref(a, cfg, b, boot_slot='marginal')
    ours = tdi_ref(a, cfg, b, boot_slot='next')
    assert abs(loss - theirs) < 1e-5, (loss, theirs)
    assert abs(theirs - ours) > 5e-5, \\
        "the bootstrap positive slot is indistinguishable from the dynamics next state " \\
        "-- delta (a) is not implemented"
    assert abs(loss - ours) > 5e-5, "the agent scores OUR slot convention"

    # substituting s' into the slot the agent reads must land exactly on the other
    # reference: that pins WHICH batch key the bootstrap positive comes from.
    b_sub = dict(b); b_sub['marginal_positives'] = b['next_observations']
    l_sub = float(a.tdinfonce_loss(jb(b_sub), a.network.params, jax.random.PRNGKey(0))[0])
    assert abs(l_sub - ours) < 1e-5, (l_sub, ours)
    print("OK")
    """)


# --------------------------------------------------------------------------
# 4. delta 2: ONE shared pool from the dataset future-state marginal
# --------------------------------------------------------------------------
def test_negatives_come_from_the_marginal_pool_not_the_batch_next_states():
    """The negatives are batch['marginal_states'] (iid dataset future-state draws), shared
    by every row. Two checks with teeth: perturbing ONLY that key moves the loss (so it is
    genuinely read), and the 'batch_next' ablation -- our standing support -- gives a
    different number through the SAME code path."""
    _run("""
    cfg = tdi_paper_cfg()
    a = agent_for(cfg)
    for s in range(3):
        a, _ = a.update(jb(make_batch(B=32, seed=s)))
    b = make_batch(B=32, seed=6)
    base = float(a.tdinfonce_loss(jb(b), a.network.params, jax.random.PRNGKey(0))[0])
    b2 = dict(b); b2['marginal_states'] = make_batch(B=32, seed=99)['marginal_states']
    moved = float(a.tdinfonce_loss(jb(b2), a.network.params, jax.random.PRNGKey(0))[0])
    assert abs(base - moved) > 5e-5, "the shared negative pool is not read at all"
    assert abs(moved - tdi_ref(a, cfg, b2)) < 1e-5, \\
        "the loss under a swapped pool is not the closed form under that pool"
    b3 = dict(b); b3['marginal_positives'] = make_batch(B=32, seed=98)['marginal_positives']
    moved2 = float(a.tdinfonce_loss(jb(b3), a.network.params, jax.random.PRNGKey(0))[0])
    assert abs(base - moved2) > 1e-6, "the bootstrap positive slot is not read at all"
    assert abs(moved2 - tdi_ref(a, cfg, b3)) < 1e-5

    cfg_ab = tdi_paper_cfg(measure_tdinfonce_negatives='batch_next')
    a2 = agent_for(cfg_ab)
    for s in range(3):
        a2, _ = a2.update(jb(make_batch(B=32, seed=s)))
    l_ab = float(a2.tdinfonce_loss(jb(b), a2.network.params, jax.random.PRNGKey(0))[0])
    ref_ab = tdi_ref(a2, cfg_ab, b, pool_key='next_observations',
                     pos_key='next_observations')
    assert abs(l_ab - ref_ab) < 1e-5
    # same params (same seed, same 3 batches) -> the difference is the query source alone
    l_marg = float(a2.tdinfonce_loss(jb(b), a2.network.params, jax.random.PRNGKey(0))[0])
    assert abs(l_marg - tdi_ref(a2, cfg_ab, b, pool_key='marginal_states')) > 5e-5, \\
        "'batch_next' and 'dataset_marginal' select the same distribution -- the new " \\
        "query-source knob configures nothing"
    print("OK")
    """)


# --------------------------------------------------------------------------
# 5. delta 3: l2-normalized towers, implicit temperature
# --------------------------------------------------------------------------
def test_towers_are_l2_normalized():
    """Their sec 3.1 convention. Checked twice: the closed form with normalization OFF is
    a different number, and every logit is a cosine similarity in [-1, 1] (which the raw
    inner products of these towers are NOT)."""
    _run("""
    cfg = tdi_paper_cfg()
    a = agent_for(cfg)
    for s in range(3):
        a, _ = a.update(jb(make_batch(B=32, seed=s)))
    b = make_batch(B=32, seed=7)
    loss = float(a.tdinfonce_loss(jb(b), a.network.params, jax.random.PRNGKey(0))[0])
    assert abs(loss - tdi_ref(a, cfg, b, normalize=True)) < 1e-5
    assert abs(loss - tdi_ref(a, cfg, b, normalize=False)) > 1e-3, \\
        "the loss is unchanged by l2 normalization -- delta (c) is not implemented"
    _, info = a.tdinfonce_loss(jb(b), a.network.params, jax.random.PRNGKey(0))
    assert abs(float(info['tdi_pos_logit'])) <= 1.0 + 1e-5
    assert abs(float(info['tdi_neg_logit'])) <= 1.0 + 1e-5
    blocks = blocks_of(cfg)
    raw = np.asarray(a.network.select('density_b')(
        GCIQLFlowAgent._density_queries(jnp.asarray(b['next_observations']), blocks)))
    assert np.abs(np.linalg.norm(raw, axis=-1) - 1.0).max() > 1e-2, \\
        "the towers are already unit-norm by accident -- this test would be vacuous"
    print("OK")
    """)


# --------------------------------------------------------------------------
# 6. MUTATION TEST: break the implementation, the closed form must catch it
# --------------------------------------------------------------------------
def test_mutants_are_caught():
    """The checks above only have teeth if they fail on a broken implementation. Four
    mutants are compiled from the real source and re-scored:
      m1 bootstrap slot -> the dynamics next state (delta a deleted);
      m2 l2 normalization removed (delta c deleted);
      m3 the shared pool replaced by the batch next states (delta b deleted);
      m4 the (1-gamma)/gamma weights of the two terms swapped.

    Note on the stop-gradient over the target softmax w: it is BELT-AND-BRACES, not
    load-bearing, because the target towers are called at frozen params (never
    params=grad_params) and are therefore already outside jax.grad's path. Removing it is
    provably a no-op, so it is not mutated here; what actually guarantees the property --
    zero gradient into modules_target_density_* -- is asserted directly in test 10.
    """
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

    cfg = tdi_paper_cfg()
    b = make_batch(B=32, seed=13)

    def scored(mod):
        a = mod.GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, dict(cfg))
        for s in range(3):
            a, _ = a.update(jb(make_batch(B=32, seed=s)))
        return a, float(a.tdinfonce_loss(jb(b), a.network.params, jax.random.PRNGKey(0))[0])

    import gciql_flow as real_mod
    a_real, l_real = scored(real_mod)
    ref = tdi_ref(a_real, cfg, b)
    assert abs(l_real - ref) < 1e-5

    m1 = mutate("[jnp.einsum('hbd,bd->hb', f_emb, b_pos)[..., None], neg], -1)",
                "[jnp.einsum('hbd,bd->hb', f_emb, b_nxt)[..., None], neg], -1)", "m1")
    m2 = mutate("        f_emb = _l2n(self.network.select('density_f')(\\n"
                "            jnp.concatenate([obs, acts, g], -1), params=grad_params))"
                "          # [2, B, d]",
                "        f_emb = (self.network.select('density_f')(\\n"
                "            jnp.concatenate([obs, acts, g], -1), params=grad_params))"
                "          # [2, B, d]", "m2")
    m3 = mutate("pool_obs, pos_obs = batch['marginal_states'], batch['marginal_positives']",
                "pool_obs, pos_obs = nobs, batch['marginal_positives']", "m3")
    m4 = mutate("loss = (1.0 - gamma) * ce_imm + gamma * ce_boot",
                "loss = gamma * ce_imm + (1.0 - gamma) * ce_boot", "m4")

    for tag, mod in (("m1", m1), ("m2", m2), ("m3", m3), ("m4", m4)):
        a_m, l_m = scored(mod)
        # 5e-5 is ~100x the measured closed-form agreement (~5e-7) and ~3x below
        # the smallest real mutant gap (1.4e-4); see test 3 on why magnitudes here are
        # small even though the structural change is total.
        assert abs(l_m - ref) > 5e-5, (tag, "mutant scores the reference loss -- the "
                                            "closed-form check has no teeth")

    print("OK")
    """)


# --------------------------------------------------------------------------
# 7. their extraction (Eq. 11 + App. D.3), and the matched-extractor arm
# --------------------------------------------------------------------------
def test_paper_actor_is_eq11_plus_mse_bc():
    """Eq. 11: the positive candidate is the COMMANDED GOAL as an outcome (phi(g)), the
    negatives are the same marginal pool, and App. D.3 mixes it convexly with the MSE
    goal-BC term. Checked against a numpy recomputation; plus the two controls that would
    pass if we had quietly kept our own actor: lambda must be load-bearing, and the
    'matched' arm (our kernel readout + flow actor) must NOT take this branch."""
    _run("""
    cfg = tdi_paper_cfg()
    a = agent_for(cfg)
    for s in range(3):
        a, _ = a.update(jb(make_batch(B=32, seed=s)))
    b = make_batch(B=32, seed=21)
    bj = jb(b)
    loss, info = a.actor_loss(bj, a.network.params, jax.random.PRNGKey(0))

    blocks = blocks_of(cfg)
    q = GCIQLFlowAgent._density_queries
    dist = a.network.select('actor')(bj['observations'], bj['actor_goals'])
    a_pi = np.clip(np.asarray(dist.mode(), np.float64), -1, 1)
    b_pool = l2(np.asarray(a.network.select('density_b')(q(bj['marginal_states'], blocks))))
    b_goal = l2(np.asarray(a.network.select('density_b')(q(bj['actor_goals'], blocks))))
    f = l2(np.asarray(a.network.select('density_f')(jnp.concatenate(
        [bj['observations'], jnp.asarray(a_pi, jnp.float32), bj['actor_goals']], -1))))
    logits = np.concatenate([np.einsum('hbd,bd->hb', f, b_goal)[..., None],
                             np.einsum('hbd,pd->hbp', f, b_pool)], -1)
    ce = logsm(logits)[..., 0]
    ce = np.minimum(ce[0], ce[1])
    mse = float(np.mean((np.asarray(dist.mode(), np.float64) - b['actions']) ** 2))
    lam = float(cfg['measure_tdinfonce_bc_lambda'])
    ref = -(1 - lam) * ce.mean() + lam * mse
    assert abs(float(loss) - ref) < 1e-5, (float(loss), ref)
    assert abs(float(info['mse']) - mse) < 1e-5

    # the POSITIVE is the goal itself: perturbing only the goals moves the CE term
    b2 = dict(b); b2['actor_goals'] = make_batch(B=32, seed=77)['actor_goals']
    l2_ = float(a.actor_loss(jb(b2), a.network.params, jax.random.PRNGKey(0))[0])
    assert abs(l2_ - float(loss)) > 1e-4

    # lambda is load-bearing (a convex mix, not a decorative knob)
    a_hi = agent_for(tdi_paper_cfg(measure_tdinfonce_bc_lambda=0.9))
    for s in range(3):
        a_hi, _ = a_hi.update(jb(make_batch(B=32, seed=s)))
    assert abs(float(a_hi.actor_loss(bj, a_hi.network.params, jax.random.PRNGKey(0))[0])
               - float(loss)) > 1e-3

    # ... and NO q-scale normalization anywhere on this path
    assert 'q_loss' not in info and 'bc_log_prob' not in info, \\
        "the paper actor must not pick up our DDPG+BC q_loss/log-prob machinery"

    # the matched-extractor arm takes OUR branch instead (kernel readout + flow actor)
    cfgm = tdi_matched_cfg()
    am = agent_for(cfgm)
    for s in range(3):
        am, im = am.update(jb(make_batch(B=32, seed=s)))
    assert 'actor/q_mean' in im and 'measure/tdi_loss' in im
    # its readout reads the NORMALIZED classifier (their critic), not the raw one
    qa = np.asarray(am._density_actor_q(bj, bj['observations'], bj['actions'],
                                        bj['actor_goals']), np.float64)
    b_emb = np.asarray(am.network.select('density_b')(q(bj['next_observations'], blocks)))
    f_emb = np.asarray(am.network.select('density_f')(jnp.concatenate(
        [bj['observations'], bj['actions'], bj['actor_goals']], -1)))
    kg = np.asarray(GCIQLFlowAgent._delta_phi(
        bj['actor_goals'], GCIQLFlowAgent._raw_block_cols(bj['next_observations'], blocks),
        blocks), np.float64)
    def qref(fx, bx):
        lg = np.einsum('hbd,md->hbm', fx, bx)
        w = np.exp(logsm(lg))
        return (w * kg[None]).sum(-1)
    assert np.abs(qa - qref(l2(f_emb), l2(b_emb))).max() < 1e-5
    assert np.abs(qa - qref(np.asarray(f_emb, np.float64),
                            np.asarray(b_emb, np.float64))).max() > 1e-6, \\
        "the matched readout is indifferent to the l2 normalization their critic trains with"
    print("OK")
    """)


# --------------------------------------------------------------------------
# 8. GCDataset supplies the marginal keys, and only when asked
# --------------------------------------------------------------------------
def test_dataset_marginal_keys():
    """batch['marginal_states'] / ['marginal_positives'] are TWO independent iid draws from
    the dataset future-state marginal (fresh uniform rows -> geometric future), present
    only under config.marginal_states, and byte-identical batches otherwise."""
    _run("""
    import numpy as np
    from utils.datasets import Dataset, GCDataset

    T, N = 20, 40          # 40 trajectories of length 20
    size = T * N
    obs = np.arange(size, dtype=np.float32)[:, None].repeat(OBS_D, 1)
    terminals = np.zeros(size, np.float32); terminals[T - 1::T] = 1.0
    ds = Dataset.create(observations=obs, actions=np.zeros((size, ACT_D), np.float32),
                        terminals=terminals,
                        next_observations=np.roll(obs, -1, axis=0))
    base = dict(discount=0.99, value_p_curgoal=0.2, value_p_trajgoal=0.5,
                value_p_randomgoal=0.3, value_geom_sample=True, actor_p_curgoal=0.0,
                actor_p_trajgoal=1.0, actor_p_randomgoal=0.0, actor_geom_sample=True,
                gc_negative=False, p_aug=None, frame_stack=None)

    np.random.seed(0)
    off = GCDataset(ds, dict(base)).sample(64)
    assert 'marginal_states' not in off and 'marginal_positives' not in off
    np.random.seed(0)
    off2 = GCDataset(ds, dict(base)).sample(64)
    assert all(np.array_equal(off[k], off2[k]) for k in off), "knob-off batch is not stable"

    np.random.seed(0)
    on = GCDataset(ds, dict(base, marginal_states=True)).sample(64)
    for k in ('marginal_states', 'marginal_positives'):
        assert on[k].shape == (64, OBS_D)
    assert not np.array_equal(on['marginal_states'], on['marginal_positives']), \\
        "the two draws must be INDEPENDENT (a shared candidate may not appear twice)"
    assert not np.array_equal(on['marginal_states'], on['next_observations']), \\
        "the pool must not be the batch's own next states"
    # the drawn rows are real dataset states, and they are FUTURE states of their own
    # (uniformly drawn) rows -- i.e. spread over the whole dataset, not over `idxs`
    m = on['marginal_states'][:, 0]
    assert np.all((m >= 0) & (m < size))
    assert m.std() > size / 8, "the marginal draw is not spread over the dataset"
    print("OK")
    """)


# --------------------------------------------------------------------------
# 9. lattice
# --------------------------------------------------------------------------
def test_lattice_refuses_inert_and_unsupported_combinations():
    """No knob where it is read nowhere; no composition that would let the arm log as
    paper-exact while running something else."""
    _run("""
    def refuses(cfg):
        try:
            agent_for(cfg)
        except AssertionError:
            return True
        return False

    # scope: the knobs need their owner
    assert refuses(dict(get_config(), encoder=None, frame_stack=None,
                        measure_tdinfonce_actor='matched'))
    assert refuses(dict(get_config(), encoder=None, frame_stack=None,
                        measure_tdinfonce_bc_lambda=0.2))
    assert refuses(dict(get_config(), encoder=None, frame_stack=None,
                        measure_tdinfonce_negatives='batch_next'))
    assert refuses(dict(get_config(), encoder=None, frame_stack=None, marginal_states=True))
    # the towers and the objective slot
    cfg_no_density = dict(get_config(), encoder=None, frame_stack=None, measure_only=True,
                          measure_tdinfonce=True)
    assert refuses(cfg_no_density), "no density towers to train"
    assert refuses(g_cfg(measure_tdinfonce=True, measure_only=False))
    assert refuses(g_cfg(measure_tdinfonce=True, measure_lsif=True))
    # every knob read only inside density_loss is inert here
    assert refuses(tdi_paper_cfg(measure_density_hard=True))
    assert refuses(tdi_paper_cfg(measure_density_dup_instrument=True))
    assert refuses(tdi_paper_cfg(measure_density_z=True))
    assert refuses(tdi_paper_cfg(measure_density_behavior=True))
    assert refuses(tdi_paper_cfg(actor_bc_only=True))
    assert refuses(tdi_paper_cfg(measure_density_actor_reward='latent'))
    # their conventions are pinned
    assert refuses(tdi_paper_cfg(measure_density_temp=0.5)), "temperature is implicit"
    assert refuses(tdi_paper_cfg(measure_tdinfonce_bc_lambda=0.0))
    assert refuses(tdi_paper_cfg(measure_tdinfonce_bc_lambda=1.0))
    assert refuses(tdi_paper_cfg(measure_tdinfonce_negatives='batch_nxt'))
    assert refuses(tdi_paper_cfg(measure_tdinfonce_actor='hybrid'))
    bad_relabel = tdi_paper_cfg(); bad_relabel['actor_geom_sample'] = False
    assert refuses(bad_relabel), "their goal-relabeling distribution is not optional"
    bad_q = tdi_paper_cfg(); bad_q['actor_q_center'] = True
    assert refuses(bad_q), "their Eq. 11 has no q-scale normalizer"
    bad_kind = tdi_paper_cfg(); bad_kind['actor_kind'] = 'flow'
    assert refuses(bad_kind), "the paper actor is the deterministic policy"
    bad_m = tdi_matched_cfg(); bad_m['actor_kind'] = 'ddpgbc'
    assert refuses(bad_m), "the matched arm is OUR flow extraction"
    assert refuses(tdi_matched_cfg(measure_tdinfonce_bc_lambda=0.2)), \\
        "lambda is read only by the paper actor"

    # accepted: the wave-1 arm, the (unlaunched) matched arm, the query-source ablation,
    # and the neighbours this spec must not disturb
    assert not refuses(tdi_paper_cfg()), "tdi_cd_exact_s0"
    assert not refuses(tdi_matched_cfg()), "the promotion-wave arm"
    assert not refuses(tdi_paper_cfg(measure_tdinfonce_negatives='batch_next'))
    assert not refuses(g_cfg()), "g-indexed density arm untouched"
    assert not refuses(g_cfg(measure_density_hard=True)), "hard ablation untouched"
    print("OK")
    """)


# --------------------------------------------------------------------------
# 10. end to end
# --------------------------------------------------------------------------
def test_arm_end_to_end():
    """30 real update() steps of tdi_cd_exact_s0: finite everywhere, both towers and the
    actor move, the classifier's top-1 lifts off chance, the dataset key is forced on, and
    NO gradient reaches the target towers."""
    _run("""
    cfg = tdi_paper_cfg()
    assert cfg['marginal_states'] is not True, "the config starts without the dataset key"
    a = agent_for(cfg)
    assert a.config['marginal_states'] is True, \\
        "create() must force the GCDataset marginal key on (the next_actions precedent)"
    p0 = jax.tree_util.tree_map(lambda x: np.asarray(x).copy(), a.network.params)
    for s in range(30):
        a, info = a.update(jb(make_batch(B=32, seed=s)))
        for k, v in info.items():
            assert np.isfinite(float(v)), (s, k, float(v))
    for mod in ('modules_density_f', 'modules_density_b', 'modules_actor'):
        assert any(not np.allclose(np.asarray(x), y) for x, y in zip(
            jax.tree_util.tree_leaves(a.network.params[mod]),
            jax.tree_util.tree_leaves(p0[mod]))), mod + " never moved"
    for k in ('measure/tdi_loss', 'measure/tdi_ce_immediate', 'measure/tdi_ce_bootstrap',
              'measure/tdi_top1', 'measure/tdi_w_ess', 'actor/tdi_actor_ce', 'actor/mse'):
        assert k in info, k

    # gradient hygiene: the target towers are EMA-only
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
