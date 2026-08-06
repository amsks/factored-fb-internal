"""`ors_reward_ckpt` -- spec 19b section 2: the ORS (occupancy reward shaping) extractor.
Contract: `PAPER/Multistep-FB/specs/19b-ors-extractor-impl.md` section 2/3, transcription
note `PAPER/Multistep-FB/specs/19b-ors-transcription.md`.

A stock g-conditioned GCIQL-flow run (IQL value/critic + flow actor -- NO measure_only,
NO hindsight columns, NO phase switch) whose critic TD target replaces `batch['rewards']`
by a shaping reward `r_W`, read off a FROZEN `measure_density_behavior` checkpoint's
(F_D, B_D) towers (spliced in as `ors_f`/`ors_b`, never trained, never called with
`params=grad_params`):

    logits_j = F_D(s,a).B_D(u_j) / tau                        [2, B, P]
    w        = mean_h softmax_j(logits)_hj                     [B, P]   (MEAN of the 2
               heads' softmax weight distributions -- r_W is a reward substituted into a
               TD target, not a value estimate the actor mins over, so there is no
               min/max asymmetry to preserve; measure_density's OWN actor q-term keeps its
               2 heads separate for exactly that (different) reason)
    r_W      = ors_reward_scale * (1 - gamma_frozen) * sum_j w_j * (-d^2(u_j, g))

r_W REPLACES `batch['rewards']` for every non-terminal (mask==1) row; the success-index
row (mask==0, GCDataset's `masks = 1 - successes` convention) is FORCE-SET to 1.0
regardless of r_W -- the paper's own convention (Appendix B.3's TD target + the released
code's `jnp.where(successes, 1.0, batch['rewards'])`), not a pure replace-everywhere
scheme (transcription note (a)).

The pool U_P (the fixed query support r_W is read against) cannot be drawn inside
`create()` as spec 19b section 2.2 originally suggested -- `create()` only ever sees an
example batch (`ex_observations`/`ex_actions`, shape [1, ...]), never the training
dataset -- so it is generated OUT-OF-BAND with the EXISTING
`scripts/rescuing_fb/make_delta_prototypes.py` (same object as 17b's fixed prototypes: K
distinct dataset-row draws of the hit-spec block columns; P is just a bigger K) and loaded
by path via the new `ors_pool_path` knob, 17b-protos-style. This test file writes its own
tiny synthetic pool files with `np.save` directly (no need to invoke the script, which is
dataset-format-specific and orthogonal to what's being tested here).

No 19a checkpoint exists yet (the real behavior-density training runs are hours out), so
every test here builds a SYNTHETIC `measure_density_behavior` checkpoint (a real
`GCIQLFlowAgent` instance, saved via the exact `flax.serialization.to_state_dict` +
`pickle` format `utils/flax_utils.save_agent` uses, plus a `flags.json` matching what
`main.py`'s `get_flag_dict()` writes) -- this proves the extractor's restore/consistency/
reward-computation machinery is correct in isolation; it does NOT prove anything about
what a REAL 19a-trained behavior-density checkpoint's r_W looks like (that needs the P1.2
calibration probe once 19a actually lands -- spec 19 section 1).

Same subprocess-jax harness as tests/test_gciql_measure_density_behavior.py: OBS_D=40,
action dim 5, hit_spec '19:22:0.4' (d_c=3), B=16. Pool size P=256 (the lattice's own
floor).

Runner: `.venv/bin/python -m pytest tests/test_gciql_ors_extractor.py -x -q`
(NOT uv; the jax subprocess harness resolves the jax venv itself).
"""
import textwrap

import numpy as np

from tests.jax_subprocess import run_jax, REPO

OBS_D = 40
ACT_D = 5
D = 8            # measure_density_dim of the frozen behavior towers (ors_f/ors_b's own dim)
P = 256           # ORS pool size -- the lattice's own >=256 floor
HIT_SPEC = "19:22:0.4"

COMMON = f"""
    import jax, jax.numpy as jnp, numpy as np, flax, json, os, pickle, tempfile
    from gciql_flow import GCIQLFlowAgent, get_config, _parse_hit_spec

    OBS_D = {OBS_D}
    ACT_D = {ACT_D}
    D = {D}
    P = {P}
    HIT_SPEC = {HIT_SPEC!r}
    BLOCKS = _parse_hit_spec(HIT_SPEC)
    EX_OBS = jnp.zeros((1, OBS_D), jnp.float32)
    EX_ACT = jnp.zeros((1, ACT_D), jnp.float32)
    TMPDIR = tempfile.mkdtemp()

    def make_pool(p=P, seed=0, blocks=BLOCKS):
        # RAW obs units, columns = the hit-spec blocks concatenated in _parse_hit_spec
        # order -- the SAME convention as measure_delta_protos/delta_protos (spec 19b
        # section 2.2's "17b protos pattern"). make_delta_prototypes.py draws these from
        # a real dataset npz; this is the same object, drawn synthetically.
        d_c = sum(b - a for (a, b, e) in blocks)
        rng = np.random.default_rng(seed)
        pool = (rng.normal(size=(p, d_c)).astype(np.float32)) * 0.5
        path = os.path.join(TMPDIR, f"pool_{{seed}}_{{p}}_{{d_c}}.npy")
        np.save(path, pool)
        return path

    def behavior_cfg(discount=0.99, temp=1.0, behavior=True, density=True, seed=0, **kw):
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = D
        cfg["measure_only"] = True
        cfg["measure_density"] = density
        cfg["measure_density_behavior"] = behavior
        cfg["measure_density_dim"] = D
        cfg["measure_density_temp"] = temp
        cfg["measure_hit_spec"] = HIT_SPEC
        cfg["discount"] = discount
        cfg.update(kw)
        return cfg

    def make_behavior_ckpt(discount=0.99, temp=1.0, behavior=True, density=True,
                           epoch=0, seed=0, **kw):
        # A REAL GCIQLFlowAgent (measure_density_behavior, spec 19a), saved in the EXACT
        # format utils/flax_utils.save_agent/main.py use: params_<epoch>.pkl holding
        # {{'agent': flax.serialization.to_state_dict(agent)}}, plus a flags.json holding
        # {{'agent': <cfg dict>}} (main.py's get_flag_dict()['agent'], the SAME file
        # scripts/rescuing_fb/eval_rgfb_zeroshot.py / eval_measure_g4_spearman.py already
        # read a run's architecture from).
        cfg = behavior_cfg(discount=discount, temp=temp, behavior=behavior,
                           density=density, seed=seed, **kw)
        agent = GCIQLFlowAgent.create(seed, EX_OBS, EX_ACT, cfg)
        run_dir = tempfile.mkdtemp(dir=TMPDIR)
        with open(os.path.join(run_dir, "flags.json"), "w") as f:
            json.dump({{"agent": cfg}}, f)
        save_dict = dict(agent=flax.serialization.to_state_dict(agent))
        with open(os.path.join(run_dir, f"params_{{epoch}}.pkl"), "wb") as f:
            pickle.dump(save_dict, f)
        return run_dir, cfg, agent

    def base_cfg(behavior_dir, pool_path, **kw):
        # the extractor's OWN config: plain GCIQL-flow (no measure_only, no measure knobs
        # of its own -- ORS is a critic-reward substitution, not a different critic
        # architecture).
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["measure_hit_spec"] = HIT_SPEC
        cfg["ors_reward_ckpt"] = behavior_dir
        cfg["ors_reward_epoch"] = 0
        cfg["ors_pool_path"] = pool_path
        cfg["ors_pool_size"] = P
        cfg.update(kw)
        return cfg

    def make_extractor(behavior_dir=None, pool_path=None, discount=0.99, temp=1.0,
                       behavior=True, density=True, epoch=0, **kw):
        if behavior_dir is None:
            behavior_dir, _, _ = make_behavior_ckpt(discount=discount, temp=temp,
                                                     behavior=behavior, density=density,
                                                     epoch=epoch)
        if pool_path is None:
            pool_path = make_pool()
        cfg = base_cfg(behavior_dir, pool_path, discount=discount, **kw)
        return GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg), cfg, behavior_dir, pool_path

    def make_batch(B=16, seed=0, n_success=0):
        rng = np.random.default_rng(seed)
        obs = rng.normal(size=(B, OBS_D)).astype(np.float32)
        nobs = rng.normal(size=(B, OBS_D)).astype(np.float32)
        actions = np.clip(rng.normal(size=(B, ACT_D)), -1, 1).astype(np.float32)
        masks = np.ones(B, np.float32)
        if n_success:
            masks[:n_success] = 0.0    # mask==0 <=> GCDataset's success-index row
        batch = dict(
            observations=obs,
            next_observations=nobs,
            actions=actions,
            value_goals=rng.normal(size=(B, OBS_D)).astype(np.float32),
            actor_goals=rng.normal(size=(B, OBS_D)).astype(np.float32),
            rewards=(rng.random(B) * -1.0).astype(np.float32),   # arbitrary sparse-style rewards
            masks=masks,
        )
        return batch

    def r_w_ref(agent, cfg, pool_path, goals, obs, acts):
        # numpy/direct-module reference for _ors_reward: softmax-MEAN of the 2 heads,
        # (1-gamma), scale, eps-scaled d^2 -- rebuilt from DIRECT module calls at the
        # agent's own (frozen, spliced) ors_f/ors_b params, matching gciql_flow.py's
        # _ors_reward exactly.
        pool = np.load(pool_path)
        pool_sc = np.asarray(GCIQLFlowAgent._scale_block_cols(jnp.asarray(pool), BLOCKS))
        f_emb = agent.network.select('ors_f')(jnp.concatenate([obs, acts], -1))     # [2,B,D]
        b_emb = agent.network.select('ors_b')(jnp.asarray(pool_sc, jnp.float32))    # [P,D]
        tau = agent.ors_tau
        logits = np.einsum('hbd,pd->hbp', np.asarray(f_emb), np.asarray(b_emb)) / tau
        w = np.exp(logits - logits.max(axis=-1, keepdims=True))
        w = w / w.sum(axis=-1, keepdims=True)
        w = w.mean(axis=0)                                                          # [B,P]
        d2 = np.asarray(GCIQLFlowAgent._eps_sqdist(jnp.asarray(goals), jnp.asarray(pool), BLOCKS))
        gamma = cfg['discount']
        r_w = cfg['ors_reward_scale'] * (1.0 - gamma) * np.sum(w * (-d2), axis=-1)
        return r_w
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
    """ors_reward_ckpt='' (the default): no ors_f/ors_b modules registered, param tree
    byte-identical to a no-knob config, and total_loss bit-identical on a fixed batch.

    This is a BRAND NEW knob with no pre-existing history to diff against (unlike
    tests/test_gciql_measure_density.py's fragility -- there is no historical commit
    predating ors_reward_ckpt to pin, because this test and the knob it tests are being
    introduced together). Two current-code knob-off configs (one with the key entirely
    absent, one with it explicitly '') are compared -- adequate here because there is no
    "off-path regression relative to history" question yet; that question becomes
    meaningful (and gets the git-diff treatment) only once this knob has shipped and
    later work might regress its off-path."""
    _run("""
    cfg_nokey = dict(get_config()); cfg_nokey["encoder"] = None; cfg_nokey["frame_stack"] = None
    a_nokey = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg_nokey)
    for mod in ("modules_ors_f", "modules_ors_b"):
        assert mod not in a_nokey.network.params, mod
    assert a_nokey.ors_pool is None and a_nokey.ors_pool_b_emb is None and a_nokey.ors_tau is None

    cfg_off = dict(get_config()); cfg_off["encoder"] = None; cfg_off["frame_stack"] = None
    cfg_off["ors_reward_ckpt"] = ""
    a_off = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg_off)

    t0 = jax.tree_util.tree_leaves(a_nokey.network.params)
    t1 = jax.tree_util.tree_leaves(a_off.network.params)
    assert len(t0) == len(t1)
    assert all((x == y).all() for x, y in zip(t0, t1))

    b = make_batch()
    bj = {k: jnp.asarray(v) for k, v in b.items()}
    l0, info0 = a_nokey.total_loss(bj, a_nokey.network.params, jax.random.PRNGKey(7))
    l1, info1 = a_off.total_loss(bj, a_off.network.params, jax.random.PRNGKey(7))
    assert float(l0) == float(l1), (float(l0), float(l1))
    assert 'critic/ors_r_mean' not in info0 and 'critic/ors_r_mean' not in info1
    print("OK")
    """)


# --------------------------------------------------------------------------
# 2. r_W matches a numpy reference
# --------------------------------------------------------------------------
def test_rw_matches_numpy():
    """_ors_reward matches r_w_ref (softmax-mean of 2 heads, (1-gamma), ors_reward_scale,
    eps-scaled d^2) atol 1e-4, for two different ors_reward_scale values (checks the
    scale knob is actually multiplied in, not merely recorded)."""
    _run("""
    agent, cfg, behavior_dir, pool_path = make_extractor()
    b = make_batch(B=10, seed=5)
    bj = {k: jnp.asarray(v) for k, v in b.items()}
    r_w = agent._ors_reward(bj['observations'], bj['actions'], bj['value_goals'])
    ref = r_w_ref(agent, cfg, pool_path, b['value_goals'], b['observations'], b['actions'])
    assert np.allclose(np.asarray(r_w), ref, atol=1e-4), (np.asarray(r_w), ref)

    agent2, cfg2, _, _ = make_extractor(behavior_dir=behavior_dir, pool_path=pool_path,
                                        ors_reward_scale=3.0)
    r_w2 = agent2._ors_reward(bj['observations'], bj['actions'], bj['value_goals'])
    ref2 = r_w_ref(agent2, cfg2, pool_path, b['value_goals'], b['observations'], b['actions'])
    assert np.allclose(np.asarray(r_w2), ref2, atol=1e-4)
    assert np.allclose(np.asarray(r_w2), 3.0 * np.asarray(r_w), atol=1e-4), \\
        "ors_reward_scale is not linearly multiplied into r_W"
    print("OK")
    """)


# --------------------------------------------------------------------------
# 3. reward substitution (NaN-poison + success-index override)
# --------------------------------------------------------------------------
def test_reward_substitution():
    """With ORS on, critic_loss's TD target is built from r_W, NOT batch['rewards']:
    poisoning batch['rewards'] with NaN leaves the ORS-on loss FINITE (the substitution
    is total -- batch['rewards'] is never referenced on that path) while the SAME batch
    under ORS-off (a config with ors_reward_ckpt='') produces NaN (the control that
    proves this test can actually detect a missing substitution, not just that NaN never
    propagates for unrelated reasons). Separately, info['q_mean'] is cross-checked
    against a numpy reconstruction of q = where(mask==0, 1.0, r_W) + gamma*mask*next_v --
    pinning BOTH the replace-not-add convention AND the success-index override to a fixed
    1.0 regardless of r_W (transcription note (a))."""
    _run("""
    agent, cfg, behavior_dir, pool_path = make_extractor()
    b = make_batch(B=12, seed=2, n_success=3)
    b['rewards'] = np.full(12, np.nan, np.float32)
    bj = {k: jnp.asarray(v) for k, v in b.items()}

    loss_on, info_on = agent.critic_loss(bj, agent.network.params)
    assert np.isfinite(float(loss_on)), float(loss_on)
    assert np.isfinite(float(info_on['q_mean']))
    for key in ('ors_r_mean', 'ors_r_std', 'ors_r_min', 'ors_r_max', 'ors_reward_mean'):
        assert np.isfinite(float(info_on[key])), (key, info_on[key])

    off_cfg = dict(cfg); off_cfg['ors_reward_ckpt'] = ''
    agent_off = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, off_cfg)
    loss_off, info_off = agent_off.critic_loss(bj, agent_off.network.params)
    assert not np.isfinite(float(loss_off)), \\
        "control: NaN-poisoned rewards should propagate to NaN when ORS is OFF"

    r_w = np.asarray(agent._ors_reward(bj['observations'], bj['actions'], bj['value_goals']))
    next_v = np.asarray(agent.network.select('value')(bj['next_observations'], bj['value_goals']))
    success = (b['masks'] == 0.0)
    q_ref = np.where(success, 1.0, r_w) + cfg['discount'] * b['masks'] * next_v
    assert np.allclose(float(info_on['q_mean']), q_ref.mean(), atol=1e-4), \\
        (float(info_on['q_mean']), q_ref.mean())
    assert success.sum() == 3
    print("OK")
    """)


# --------------------------------------------------------------------------
# 4. frozen-checkpoint consistency refusals
# --------------------------------------------------------------------------
def test_frozen_consistency_refusals():
    """Refuses: a non-behavior measure_density checkpoint; hit_spec mismatch; discount
    mismatch; ors_pool_size < 256. Accepts the base config."""
    _run("""
    def refuses(**kw):
        try:
            make_extractor(**kw)
        except AssertionError:
            return True
        return False

    assert refuses(behavior=False)   # measure_density_behavior=False in the ckpt's flags.json

    behavior_dir, _, _ = make_behavior_ckpt(discount=0.99)
    pool_path = make_pool()

    def refuses_fixed(**kw):
        try:
            make_extractor(behavior_dir=behavior_dir, pool_path=pool_path, **kw)
        except AssertionError:
            return True
        return False

    assert refuses_fixed(measure_hit_spec="19:22:0.8")   # hit_spec mismatch
    assert refuses_fixed(discount=0.95)                   # discount mismatch (ckpt trained at 0.99)
    assert refuses_fixed(ors_pool_size=10)                 # below the >=256 lattice floor
    assert not refuses_fixed()
    print("OK")
    """)


# --------------------------------------------------------------------------
# 5. grad hygiene
# --------------------------------------------------------------------------
def test_grad_hygiene():
    """jax.grad(total_loss): EXACTLY zero over modules_ors_f/modules_ors_b leaves
    (structural -- they are never called with params=grad_params anywhere, so their
    gradient is zero regardless of any stop_gradient discipline), and NONZERO over
    modules_critic/modules_value (the ORS-shaped TD target still trains the critic)."""
    _run("""
    agent, cfg, behavior_dir, pool_path = make_extractor()
    b = make_batch(B=10, seed=9, n_success=1)
    bj = {k: jnp.asarray(v) for k, v in b.items()}
    rng = jax.random.PRNGKey(4)

    def gnorm(tree, key):
        leaves = jax.tree_util.tree_leaves(tree[key])
        return float(sum(float(jnp.sum(jnp.abs(l))) for l in leaves)) if leaves else 0.0

    def loss_fn(params):
        loss, _ = agent.total_loss(bj, params, rng)
        return loss
    grads = jax.grad(loss_fn)(agent.network.params)
    assert gnorm(grads, 'modules_ors_f') == 0.0, gnorm(grads, 'modules_ors_f')
    assert gnorm(grads, 'modules_ors_b') == 0.0, gnorm(grads, 'modules_ors_b')
    assert gnorm(grads, 'modules_critic') > 1e-8, gnorm(grads, 'modules_critic')
    assert gnorm(grads, 'modules_value') > 1e-8, gnorm(grads, 'modules_value')
    print("OK")
    """)


# --------------------------------------------------------------------------
# 6. linearity in the pool rewards
# --------------------------------------------------------------------------
def test_linearity():
    """r_W's readout is sum_j w_j * reward_j, LINEAR in whatever the per-pool-point
    "reward" is (here -d^2(u_j,g), but the identity holds for arbitrary substituted
    reward vectors): r_W(alpha*r1 + beta*r2) == alpha*r_W(r1) + beta*r_W(r2), using the
    SAME softmax weights w the agent's own frozen towers actually produce (grounds the
    check in the real implementation's weight computation, not just an abstract
    identity) -- catches a future change that clips/renormalizes the sum post-hoc."""
    _run("""
    agent, cfg, behavior_dir, pool_path = make_extractor()
    b = make_batch(B=8, seed=13)
    obs, acts = jnp.asarray(b['observations']), jnp.asarray(b['actions'])
    f_emb = np.asarray(agent.network.select('ors_f')(jnp.concatenate([obs, acts], -1)))  # [2,B,D]
    b_emb = np.asarray(agent.ors_pool_b_emb, dtype=np.float32)   # cached static field -> array
    logits = np.einsum('hbd,pd->hbp', f_emb, b_emb) / agent.ors_tau
    w = np.exp(logits - logits.max(axis=-1, keepdims=True))
    w = w / w.sum(axis=-1, keepdims=True)
    w = w.mean(axis=0)                                                                    # [B,P]

    rng = np.random.default_rng(0)
    P_ = w.shape[-1]
    r1 = rng.normal(size=(8, P_)).astype(np.float64)
    r2 = rng.normal(size=(8, P_)).astype(np.float64)
    alpha, beta = 1.7, -0.4
    lhs = np.sum(w * (alpha * r1 + beta * r2), axis=-1)
    rhs = alpha * np.sum(w * r1, axis=-1) + beta * np.sum(w * r2, axis=-1)
    assert np.allclose(lhs, rhs, atol=1e-8), (lhs, rhs)
    print("OK")
    """)


# --------------------------------------------------------------------------
# 7. end-to-end updates
# --------------------------------------------------------------------------
def test_end_to_end_updates():
    """30 update() steps on synthetic data with a synthetic frozen behavior-density
    checkpoint: finite critic/ors_r_mean|std|min|max|reward_mean each step, no NaN in
    params at the end, and ors_f/ors_b params are UNCHANGED across the whole run (the
    optimizer mask actually holds, not just the structural zero-grad)."""
    _run("""
    agent, cfg, behavior_dir, pool_path = make_extractor()
    f0 = jax.tree_util.tree_leaves(agent.network.params['modules_ors_f'])
    b0 = jax.tree_util.tree_leaves(agent.network.params['modules_ors_b'])
    for step in range(30):
        b = {k: jnp.asarray(v) for k, v in make_batch(seed=step, n_success=(step % 5 == 0)).items()}
        agent, info = agent.update(b)
        for key in ('critic/ors_r_mean', 'critic/ors_r_std', 'critic/ors_r_min',
                    'critic/ors_r_max', 'critic/ors_reward_mean', 'critic/critic_loss',
                    'value/value_loss', 'actor/actor_loss'):
            assert np.isfinite(float(info[key])), (step, key, info[key])
    leaves = jax.tree_util.tree_leaves(agent.network.params)
    assert all(np.isfinite(np.asarray(l)).all() for l in leaves), "NaN in params after 30 updates"

    f1 = jax.tree_util.tree_leaves(agent.network.params['modules_ors_f'])
    b1 = jax.tree_util.tree_leaves(agent.network.params['modules_ors_b'])
    assert all((x == y).all() for x, y in zip(f0, f1)), "ors_f params moved despite the optimizer freeze"
    assert all((x == y).all() for x, y in zip(b0, b1)), "ors_b params moved despite the optimizer freeze"
    print("OK")
    """)
