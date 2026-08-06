"""`measure_density_behavior` -- spec 19a: behavior-policy variant of `measure_density`
(spec 18). Contract: `PAPER/Multistep-FB/specs/19a-behavior-density-impl.md`.

The occupancy model ORS needs is d^{pi_D}(.|s,a) -- the DATASET behavior policy's
discounted future object-configuration distribution, not the goal-conditioned
learned-actor occupancy spec 18 trains. Two changes on top of the shipped spec-18
density_loss:

    1. No goal conditioning: F(s,a) instead of F(s,a,g) (the behavior policy
       ignores goals).
    2. SARSA bootstrap: the bootstrap action is the DATASET's own next action
       a'_D = batch['next_actions'], not a one-shot draw from the learned
       (goal-conditioned) flow actor.

The actor in this mode is PURE BC flow -- its q-term is disabled (there is no
goal-conditioned measure for it to read); actor_q_center becomes an inert knob
and is refused by the lattice (the 17b measure_coef fake-NULL lesson). The
dataset-side half of this feature (`GCDataset`'s `next_actions` batch key, in
`third_party/ogbench/impls/utils/datasets.py`) is exercised directly (no jax
needed there) by `test_next_actions_boundary`.

Same subprocess-jax harness as tests/test_gciql_measure_density.py: OBS_D=40,
action dim 5, d=8, B=16, hit_spec '19:22:0.4'.

Runner: `.venv/bin/python -m pytest tests/test_gciql_measure_density_behavior.py -x -q`
(NOT uv; the jax subprocess harness resolves the jax venv itself), then the
spec-18/17a/17b suites (shared code touched).
"""
import textwrap

import numpy as np

from tests.jax_subprocess import run_jax, REPO

OBS_D = 40
D = 8   # value_latent_dim AND measure_density_dim, matches the spec-18 harness
ACT_D = 5

COMMON = f"""
    import jax, jax.numpy as jnp, numpy as np, flax
    from gciql_flow import GCIQLFlowAgent, get_config, _parse_hit_spec

    OBS_D = {OBS_D}
    D = {D}
    ACT_D = {ACT_D}
    EX_OBS = jnp.zeros((1, OBS_D), jnp.float32)
    EX_ACT = jnp.zeros((1, ACT_D), jnp.float32)

    def base_cfg(**kw):
        # spec 19a's base cell: measure-only, bilinear, density, BEHAVIOR on.
        # actor_q_center stays default False -- behavior mode has no actor q-term
        # and the lattice refuses actor_q_center=True here (17b lesson).
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = D
        cfg["measure_only"] = True
        cfg["measure_density"] = True
        cfg["measure_density_behavior"] = True
        cfg["measure_density_dim"] = D
        cfg["measure_hit_spec"] = "19:22:0.4"
        cfg.update(kw)
        return cfg

    def make_batch(B=16, seed=0, with_next_actions=False):
        rng = np.random.default_rng(seed)
        obs = rng.normal(size=(B, OBS_D)).astype(np.float32)
        nobs = rng.normal(size=(B, OBS_D)).astype(np.float32)
        actions = np.clip(rng.normal(size=(B, ACT_D)), -1, 1).astype(np.float32)
        batch = dict(
            observations=obs,
            next_observations=nobs,
            actions=actions,
            value_goals=rng.normal(size=(B, OBS_D)).astype(np.float32),
            actor_goals=rng.normal(size=(B, OBS_D)).astype(np.float32),
            rewards=np.zeros(B, np.float32),
            masks=np.ones(B, np.float32),
        )
        if with_next_actions:
            next_actions = np.clip(rng.normal(size=(B, ACT_D)), -1, 1).astype(np.float32)
            subst = (rng.random(B) < 0.1).astype(np.float32)
            next_actions = np.where(subst[:, None] > 0, actions, next_actions)
            batch["next_actions"] = next_actions
            batch["next_actions_subst"] = subst
        return batch

    def agent_for(cfg, seed=0):
        return GCIQLFlowAgent.create(seed, EX_OBS, EX_ACT, cfg)

    def total_loss_of(agent, batch, seed=7):
        loss, info = agent.total_loss(
            {{k: jnp.asarray(v) for k, v in batch.items()}},
            agent.network.params, jax.random.PRNGKey(seed))
        return float(loss), info

    def density_behavior_ref(agent, cfg, bj):
        # numpy/direct-module reference for density_loss's behavior-mode computation:
        # goal-free F, bootstrap action = batch['next_actions'] (no actor call).
        obs, nobs = bj['observations'], bj['next_observations']
        acts, g = bj['actions'], bj['value_goals']
        gamma, tau = cfg['discount'], cfg['measure_density_temp']
        blocks = _parse_hit_spec(cfg['measure_hit_spec'])
        u_raw = GCIQLFlowAgent._raw_block_cols(nobs, blocks)
        u_sc = GCIQLFlowAgent._density_queries(nobs, blocks)
        b_emb = agent.network.select('density_b')(u_sc, params=agent.network.params)
        f_emb = agent.network.select('density_f')(
            jnp.concatenate([obs, acts], -1), params=agent.network.params)
        logits = jnp.einsum('hbd,md->hbm', f_emb, b_emb) / tau
        logp = np.asarray(jax.nn.log_softmax(logits, -1))
        a_plus = bj['next_actions']
        tb = agent.network.select('target_density_b')(u_sc)
        tf = agent.network.select('target_density_f')(jnp.concatenate([nobs, a_plus], -1))
        w_bar = np.asarray(jax.nn.softmax(jnp.einsum('hbd,md->hbm', tf, tb) / tau, -1))
        k = np.asarray(GCIQLFlowAgent._delta_phi(nobs, u_raw, blocks))
        t = (1.0 - gamma) * k[None] + gamma * w_bar
        ce_ref = -np.mean(np.sum(t * logp, axis=-1))
        return ce_ref, t
"""


def _run(body, **kw):
    code = textwrap.dedent(COMMON) + textwrap.dedent(body)
    res = run_jax(code, **kw)
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    return res


# --------------------------------------------------------------------------
# 1. default-off bit-identical (the gate)
# --------------------------------------------------------------------------
def test_default_off_bit_identical():
    """behavior=False (measure_density=True): param tree AND total_loss bit-identical
    to the shipped spec-18 arm loaded from `git show HEAD` (a genuine off-path
    regression -- wrong param count, changed default, reordered rng splits -- would
    show up here even though both configs have behavior off, because the two sides
    run different code). BOTH knobs off: bit-identical to a no-knob config, on both
    sides."""
    _run("""
    import subprocess, importlib.util, tempfile, os

    baseline_src = subprocess.run(
        ["git", "show", "HEAD:tools/wandb_mode_shim/gciql_flow.py"],
        cwd=%r, capture_output=True, text=True, check=True).stdout
    assert "measure_density_behavior" not in baseline_src, \\
        "HEAD already contains measure_density_behavior -- this repo's baseline moved; " \\
        "re-point the diff target (e.g. to the commit before spec 19a landed)"
    assert "measure_density" in baseline_src, \\
        "HEAD is missing measure_density entirely -- spec 18 must already be shipped there"
    baseline_path = os.path.join(tempfile.mkdtemp(), "gciql_flow_baseline.py")
    with open(baseline_path, "w") as f:
        f.write(baseline_src)
    spec = importlib.util.spec_from_file_location("gciql_flow_baseline", baseline_path)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)

    def cfg18(get_cfg_fn, **kw):
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

    b = make_batch()
    bj = {kk: jnp.asarray(vv) for kk, vv in b.items()}

    # (a) behavior flag OFF, measure_density ON: bit-identical to the shipped spec-18 arm.
    a_base = baseline.GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg18(baseline.get_config))
    a_off = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT,
                                   cfg18(get_config, measure_density_behavior=False))
    for mod in ("modules_density_f", "modules_density_b",
                "modules_target_density_f", "modules_target_density_b"):
        assert mod in a_off.network.params, mod
    t_base = jax.tree_util.tree_leaves(a_base.network.params)
    t_off = jax.tree_util.tree_leaves(a_off.network.params)
    assert len(t_base) == len(t_off), (len(t_base), len(t_off))
    assert all((x == y).all() for x, y in zip(t_base, t_off)), \\
        "param tree differs from the shipped spec-18 HEAD baseline with behavior=False"
    l_base, _ = a_base.total_loss(bj, a_base.network.params, jax.random.PRNGKey(7))
    l_off, _ = a_off.total_loss(bj, a_off.network.params, jax.random.PRNGKey(7))
    assert np.allclose(float(l_base), float(l_off), atol=1e-6), (float(l_base), float(l_off))

    # (b) BOTH off (no measure_density knob at all): bit-identical to a no-knob config,
    # on both sides (HEAD and current code).
    cfg_nk_base = dict(baseline.get_config())
    cfg_nk_base["encoder"] = None; cfg_nk_base["frame_stack"] = None
    a_nk_base = baseline.GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg_nk_base)
    cfg_nk = dict(get_config())
    cfg_nk["encoder"] = None; cfg_nk["frame_stack"] = None
    a_nk = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg_nk)
    for mod in ("modules_density_f", "modules_density_b",
                "modules_target_density_f", "modules_target_density_b"):
        assert mod not in a_nk.network.params, mod
    tn_base = jax.tree_util.tree_leaves(a_nk_base.network.params)
    tn = jax.tree_util.tree_leaves(a_nk.network.params)
    assert len(tn_base) == len(tn)
    assert all((x == y).all() for x, y in zip(tn_base, tn))
    ln_base, _ = a_nk_base.total_loss(bj, a_nk_base.network.params, jax.random.PRNGKey(7))
    ln, _ = a_nk.total_loss(bj, a_nk.network.params, jax.random.PRNGKey(7))
    assert np.allclose(float(ln_base), float(ln), atol=1e-6)
    print("OK")
    """ % str(REPO))


# --------------------------------------------------------------------------
# 2. F tower is goal-free
# --------------------------------------------------------------------------
def test_f_input_width():
    """density_f's first dense layer expects obs_dim + action_dim (no goal width)
    in behavior mode: it accepts a [s,a] input and rejects a [s,a,g] one (kernel
    shape mismatch), proving the goal leg was actually dropped at registration,
    not just at the call site."""
    _run("""
    cfg = base_cfg()
    a = agent_for(cfg)
    ok_input = jnp.zeros((1, OBS_D + ACT_D), jnp.float32)
    out = a.network.select('density_f')(ok_input)
    assert out.shape[-2:] == (1, D), out.shape

    bad_input = jnp.zeros((1, OBS_D + ACT_D + OBS_D), jnp.float32)  # [s,a,g] width
    raised = False
    try:
        a.network.select('density_f')(bad_input)
    except Exception:
        raised = True
    assert raised, "density_f accepted a goal-width input in behavior mode -- F is not goal-free"
    print("OK")
    """)


# --------------------------------------------------------------------------
# 3. SARSA bootstrap matches a numpy reference and is actor-invariant
# --------------------------------------------------------------------------
def test_sarsa_bootstrap_matches_numpy():
    """density_loss's CE matches density_behavior_ref (bootstrap features at
    [nobs, batch['next_actions']], no actor call) atol 1e-5. Actor-invariance:
    rebinding the SAME agent to a params tree whose modules_actor leaves are
    perturbed (density_f/density_b/target towers held fixed) leaves the loss
    EXACTLY unchanged -- the bootstrap literally never reads the actor module in
    this mode."""
    _run("""
    cfg = base_cfg()
    a = agent_for(cfg)
    b = make_batch(B=6, seed=11, with_next_actions=True)
    bj = {k: jnp.asarray(v) for k, v in b.items()}
    rng = jax.random.PRNGKey(7)

    loss, info = a.density_loss(bj, a.network.params, rng)
    ce_ref, t = density_behavior_ref(a, cfg, bj)
    assert np.allclose(ce_ref, float(loss), atol=1e-5), (ce_ref, float(loss))
    assert np.allclose(t.sum(-1), 1.0, atol=1e-5)

    p2 = dict(a.network.params)
    p2['modules_actor'] = jax.tree_util.tree_map(lambda x: x + 5.0, p2['modules_actor'])
    a2 = a.replace(network=a.network.replace(params=p2))
    loss2, _ = a2.density_loss(bj, a2.network.params, rng)
    assert float(loss2) == float(loss), \\
        (float(loss), float(loss2), "density_loss's SARSA bootstrap is NOT actor-param-invariant")
    print("OK")
    """)


# --------------------------------------------------------------------------
# 4. actor is pure BC flow
# --------------------------------------------------------------------------
def test_actor_is_pure_bc():
    """jax.grad(actor_loss): zero over density_f/density_b/target_density_f/
    target_density_b/critic leaves (no q-term reads any of them). The loss value
    equals the pure BC-flow terms alone -- bc_flow_loss (term 1, trains actor_vf)
    plus bc_coeff*distill_error (term 3, trains the one-shot actor against the
    frozen flow rollout) -- hand-reconstructed via the SAME rng-split order and
    helper methods (_vf_obs/_flow_rollout) actor_loss itself uses, with NO q term
    anywhere. q_mean/q_abs_mean report as 0."""
    _run("""
    cfg = base_cfg()
    a = agent_for(cfg)
    b = make_batch(with_next_actions=True)
    bj = {k: jnp.asarray(v) for k, v in b.items()}
    rng = jax.random.PRNGKey(3)

    def gnorm(tree, key):
        leaves = jax.tree_util.tree_leaves(tree[key])
        return float(sum(float(jnp.sum(jnp.abs(l))) for l in leaves)) if leaves else 0.0

    def actor_loss_fn(params):
        loss, _ = a.actor_loss(bj, params, rng)
        return loss
    grads = jax.grad(actor_loss_fn)(a.network.params)
    for key in ("modules_density_f", "modules_density_b",
                "modules_target_density_f", "modules_target_density_b", "modules_critic"):
        assert gnorm(grads, key) < 1e-9, (key, gnorm(grads, key))
    assert gnorm(grads, "modules_actor_vf") > 1e-8, gnorm(grads, "modules_actor_vf")
    assert gnorm(grads, "modules_actor") > 1e-8, gnorm(grads, "modules_actor")

    loss, info = a.actor_loss(bj, a.network.params, rng)
    assert float(info['q_mean']) == 0.0 and float(info['q_abs_mean']) == 0.0

    obs, goals, actions = bj['observations'], bj['actor_goals'], bj['actions']
    rng2, x0_rng, t_rng, n_rng, z_rng = jax.random.split(rng, 5)
    vf_obs = a._vf_obs(obs, goals, a.network.params)
    x_1 = actions
    x_0 = jax.random.normal(x0_rng, actions.shape)
    tt = jax.random.uniform(t_rng, (actions.shape[0], 1))
    x_t = (1 - tt) * x_0 + tt * x_1
    vel = x_1 - x_0
    pred = a.network.select('actor_vf')(vf_obs, x_t, tt, params=a.network.params)
    bc_flow_loss_ref = jnp.mean((pred - vel) ** 2)

    noise = jax.random.normal(n_rng, actions.shape)
    actor_actions = a.network.select('actor')(obs, goals, noise, params=a.network.params)
    target = a._flow_rollout(vf_obs, noise, jax.lax.stop_gradient(a.network.params))
    distill_error_ref = jnp.mean((actor_actions - target) ** 2)
    bc_loss_ref = cfg['bc_coeff'] * distill_error_ref
    loss_ref = bc_flow_loss_ref + bc_loss_ref

    assert np.allclose(float(loss), float(loss_ref), atol=1e-5), (float(loss), float(loss_ref))
    assert np.allclose(float(info['bc_flow_loss']), float(bc_flow_loss_ref), atol=1e-5)
    assert np.allclose(float(info['distill_loss']), float(distill_error_ref), atol=1e-5)
    print("OK")
    """)


# --------------------------------------------------------------------------
# 5. GCDataset next_actions boundary arithmetic (no jax needed, but same harness)
# --------------------------------------------------------------------------
def test_next_actions_boundary():
    """Synthetic ogbench-style trajectory layout (explicit next_observations, i.e. a
    REGULAR/non-compact dataset -- the format cube/scene datasets actually use):
    interior rows get actions[idx+1]; the trajectory's final row gets the CURRENT
    action; next_actions_subst carries the exact per-row indicator. A second,
    larger synthetic dataset with many short trajectories checks the substitution
    fraction lands at ~1/traj_len and stays comfortably under the spec's 5% bar.
    The off (next_actions=False) path adds no batch keys at all."""
    _run("""
    import ml_collections
    from utils.datasets import Dataset, GCDataset

    T1, T2, DIM = 10, 6, 4
    N = T1 + T2
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(N, DIM)).astype(np.float32)
    nobs = rng.normal(size=(N, DIM)).astype(np.float32)
    acts = rng.normal(size=(N, ACT_D)).astype(np.float32)
    terminals = np.zeros(N, np.float32); terminals[T1 - 1] = 1; terminals[N - 1] = 1
    ds = Dataset.create(observations=obs, next_observations=nobs, actions=acts, terminals=terminals)

    def cfg(next_actions):
        return ml_collections.ConfigDict(dict(
            discount=0.99, value_p_curgoal=0.0, value_p_trajgoal=1.0, value_p_randomgoal=0.0,
            value_geom_sample=True, actor_p_curgoal=0.0, actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0, actor_geom_sample=False, gc_negative=True,
            p_aug=None, frame_stack=None, next_actions=next_actions))

    idxs = np.array([0, 7, 9, 10, 13, 15])   # both trajectory tails included
    gcd_off = GCDataset(ds, cfg(False))
    b_off = gcd_off.sample(len(idxs), idxs=idxs)
    assert "next_actions" not in b_off and "next_actions_subst" not in b_off

    gcd = GCDataset(ds, cfg(True))
    b = gcd.sample(len(idxs), idxs=idxs)
    is_final_expected = np.array([False, False, True, False, False, True])
    for r, i in enumerate(idxs):
        if is_final_expected[r]:
            assert np.allclose(b["next_actions"][r], acts[i]), (i, b["next_actions"][r], acts[i])
        else:
            assert np.allclose(b["next_actions"][r], acts[i + 1]), \\
                (i, b["next_actions"][r], acts[i + 1])
    assert np.allclose(b["next_actions_subst"], is_final_expected.astype(np.float32))
    assert np.isclose(float(b["next_actions_subst"].mean()), float(is_final_expected.mean()))

    # substitution fraction on OGBench-play-like data: many short trajectories, one
    # final row each -> subst frac = 1/traj_len, comfortably < 5%.
    Ttraj, ntraj = 25, 40
    Np = Ttraj * ntraj
    obsp = rng.normal(size=(Np, DIM)).astype(np.float32)
    nobsp = rng.normal(size=(Np, DIM)).astype(np.float32)
    actsp = rng.normal(size=(Np, ACT_D)).astype(np.float32)
    termp = np.zeros(Np, np.float32)
    termp[Ttraj - 1::Ttraj] = 1
    dsp = Dataset.create(observations=obsp, next_observations=nobsp, actions=actsp, terminals=termp)
    gcdp = GCDataset(dsp, cfg(True))
    bp = gcdp.sample(Np, idxs=np.arange(Np))
    frac = float(bp["next_actions_subst"].mean())
    assert abs(frac - 1.0 / Ttraj) < 1e-6, frac
    assert frac < 0.05, frac
    print("OK")
    """)


# --------------------------------------------------------------------------
# 6. assert lattice
# --------------------------------------------------------------------------
def test_lattice():
    """Refuses: behavior without measure_density; with measure_delta/
    measure_delta_partition/measure_first_passage (transitively via the spec-18
    block's own refusals, still on the behavior config); with warm_start_ckpt;
    actor_q_center=True (the inert-knob refusal this spec adds). Accepts the base
    config and forces config['next_actions']=True on it."""
    _run("""
    def refuses(**kw):
        try:
            agent_for(base_cfg(**kw))
        except AssertionError:
            return True
        return False
    assert refuses(measure_density=False)
    assert refuses(measure_delta=True)
    assert refuses(measure_delta_partition=True)
    assert refuses(measure_first_passage=True)
    assert refuses(warm_start_ckpt='/nonexistent/does/not/exist*')
    assert refuses(actor_q_center=True)
    assert not refuses()

    cfg = base_cfg()
    a = agent_for(cfg)
    assert cfg['next_actions'] is True, \\
        "measure_density_behavior must force config['next_actions']=True (spec 19a S2.4)"
    print("OK")
    """)


# --------------------------------------------------------------------------
# 7. end-to-end updates
# --------------------------------------------------------------------------
def test_end_to_end_updates():
    """30 update() steps on synthetic data (with next_actions supplied, mirroring
    what a forced-on GCDataset would produce): finite density_ce/entropy/
    density_sarsa_subst info each step, w stays on the simplex, actor q_mean/
    q_abs_mean report 0 every step, no NaN in params at the end."""
    _run("""
    cfg = base_cfg()
    a = agent_for(cfg)
    for step in range(30):
        b = {k: jnp.asarray(v) for k, v in make_batch(seed=step, with_next_actions=True).items()}
        a, info = a.update(b)
        assert np.isfinite(float(info['measure/density_ce'])), (step, info['measure/density_ce'])
        assert np.isfinite(float(info['measure/density_sarsa_subst'])), \\
            (step, info['measure/density_sarsa_subst'])
        assert 0.0 <= float(info['measure/density_sarsa_subst']) <= 1.0
        for key in ('measure/density_w_entropy', 'measure/density_tgt_entropy',
                    'measure/density_ess', 'measure/density_ess_median',
                    'measure/density_ess_h0', 'measure/density_ess_h1',
                    'measure/density_logit_max', 'measure/density_k_self',
                    'actor/actor_loss', 'actor/bc_flow_loss', 'actor/distill_loss'):
            assert np.isfinite(float(info[key])), (step, key, info[key])
        assert float(info['actor/q_mean']) == 0.0 and float(info['actor/q_abs_mean']) == 0.0
    leaves = jax.tree_util.tree_leaves(a.network.params)
    assert all(np.isfinite(np.asarray(l)).all() for l in leaves), "NaN in params after 30 updates"

    b = {k: jnp.asarray(v) for k, v in make_batch(seed=99, with_next_actions=True).items()}
    blocks = _parse_hit_spec(cfg['measure_hit_spec'])
    u_sc = GCIQLFlowAgent._density_queries(b['next_observations'], blocks)
    b_emb = a.network.select('density_b')(u_sc)
    f_emb = a.network.select('density_f')(jnp.concatenate([b['observations'], b['actions']], -1))
    logits = jnp.einsum('hbd,md->hbm', f_emb, b_emb) / cfg['measure_density_temp']
    w = np.asarray(jax.nn.softmax(logits, -1))
    assert np.allclose(w.sum(-1), 1.0, atol=1e-4)
    assert (w >= -1e-6).all()
    print("OK")
    """)
