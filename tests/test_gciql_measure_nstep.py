"""`measure_nstep` — spec 16 k-step / sampled-λ-return region-measure TD.

The one-step region target `M - hit - γ·cont·tM` rebuilds the dwell mass of the
goal ball by bootstrap iteration; on play data that mass is 40-140x the arrival
signal and the sparse-region fixed point collapses to the dwell constant
(PHASE2-LEDGER §9.16). The k-step target moves the observed window mass
`Σ_{u≤k_i} γ^{u-1} h(s_{t+u})` into the data term and bootstraps at s_{t+k_i}
with weight γ^{k_i} (derivation: PAPER/Multistep-FB/main.tex — fixed point is
k-invariant; off-policy bias gated by γ^{T_int}).

Knobs (both default-off, bit-identical):
    measure_nstep         int K>0 = window cap; 0 = stock one-step path.
    measure_nstep_lambda  λ∈(0,1): k_i ~ Geometric(1-λ) truncated at K.

Discipline (mirrors test_gciql_measure_hit_spec.py): defaults byte-identical in
the param tree AND bit-identical in measure_loss; k=1 with a window built from
the batch's own next_observations must reproduce the one-step loss bit-exactly
(the nstep branch reuses the same rng-derived bootstrap noise for this reason).

Runs in the jax venv subprocess.
"""
import textwrap

from tests.jax_subprocess import run_jax, OGBENCH_IMPLS

OBS_D = 40
COMMON = f"""
    import jax, jax.numpy as jnp, numpy as np, flax
    from gciql_flow import GCIQLFlowAgent, get_config

    OBS_D = {OBS_D}
    EX_OBS = jnp.zeros((1, OBS_D), jnp.float32)
    EX_ACT = jnp.zeros((1, 5), jnp.float32)

    def base_cfg(**kw):
        # the phase-1 winner's cell: measure-only, bilinear, first-passage, no ortho.
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 8
        cfg["measure_only"] = True; cfg["measure_legs"] = "critic"
        cfg["measure_ortho_coef"] = 0.0; cfg["measure_scale_norm"] = True
        cfg["measure_first_passage"] = True
        cfg.update(kw)
        return cfg

    def make_batch(B=16, K=0, seed=0):
        rng = np.random.default_rng(seed)
        obs = rng.normal(size=(B, OBS_D)).astype(np.float32)
        nobs = rng.normal(size=(B, OBS_D)).astype(np.float32)
        nobs[:, 19:] *= 0.25   # so the 0.4-ball fires sometimes
        batch = dict(
            observations=obs,
            next_observations=nobs,
            actions=np.clip(rng.normal(size=(B, 5)), -1, 1).astype(np.float32),
            value_goals=rng.normal(size=(B, OBS_D)).astype(np.float32),
            actor_goals=rng.normal(size=(B, OBS_D)).astype(np.float32),
            rewards=np.zeros(B, np.float32),
            masks=np.ones(B, np.float32),
        )
        if K > 0:
            win = rng.normal(size=(B, K, OBS_D)).astype(np.float32)
            win[..., 19:] *= 0.25
            batch["mstep_next_obs"] = win
            batch["mstep_k"] = np.full(B, K, np.int32)
        return batch

    def agent_for(cfg, seed=0):
        return GCIQLFlowAgent.create(seed, EX_OBS, EX_ACT, cfg)

    def measure_loss_of(agent, batch, seed=7):
        loss, info = agent.measure_loss(
            {{k: jnp.asarray(v) for k, v in batch.items()}},
            agent.network.params, jax.random.PRNGKey(seed))
        return float(loss), info
"""


def _run(body, **kw):
    code = textwrap.dedent(COMMON) + textwrap.dedent(body)
    res = run_jax(code, **kw)
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    return res


def test_default_off_ignores_mstep_fields():
    """nstep=0: params identical to a no-knob config, and the loss neither reads
    nor requires the mstep batch fields."""
    _run("""
    cfg0 = base_cfg()
    cfg1 = base_cfg(measure_nstep=0, measure_nstep_lambda=0.0)
    a0, a1 = agent_for(cfg0), agent_for(cfg1)
    t0 = jax.tree_util.tree_leaves(a0.network.params)
    t1 = jax.tree_util.tree_leaves(a1.network.params)
    assert all((x == y).all() for x, y in zip(t0, t1)), "param tree differs with default knobs"
    b_plain = make_batch()
    b_extra = make_batch(); dummy = make_batch(K=3)
    b_extra["mstep_next_obs"] = dummy["mstep_next_obs"]; b_extra["mstep_k"] = dummy["mstep_k"]
    l0, _ = measure_loss_of(a0, b_plain)
    l1, _ = measure_loss_of(a1, b_extra)
    assert l0 == l1, (l0, l1)
    print("OK")
    """)


def test_k1_bit_identical_to_one_step():
    """K=1 window built from the batch's own nobs reproduces the one-step loss
    bit-exactly, for absorbing (first-passage) AND visitation targets, and for
    tpd batch AND hindsight."""
    _run("""
    for absorbing in (True, False):
        for tpd in ("batch", "hindsight"):
            cfg = base_cfg(measure_fp_absorbing=absorbing, measure_testpoint_dist=tpd)
            a_one = agent_for(cfg)
            a_nst = agent_for(base_cfg(measure_fp_absorbing=absorbing,
                                       measure_testpoint_dist=tpd, measure_nstep=1))
            b = make_batch()
            b1 = dict(b)
            b1["mstep_next_obs"] = b["next_observations"][:, None, :]
            b1["mstep_k"] = np.ones(len(b["observations"]), np.int32)
            l_one, _ = measure_loss_of(a_one, b)
            l_nst, _ = measure_loss_of(a_nst, b1)
            assert l_one == l_nst, (absorbing, tpd, l_one, l_nst)
    print("OK")
    """)


def test_window_terms_match_numpy_reference():
    """_nstep_window_terms against a brute-force numpy reference on random
    windows, both absorbing and visitation, with ragged k_i."""
    _run("""
    rng = np.random.default_rng(3)
    K, B, gamma = 5, 7, 0.97
    hits = (rng.random((K, B, B)) < 0.35).astype(np.float32)
    ks = rng.integers(1, K + 1, size=B)
    umask = (np.arange(K)[:, None] < ks[None, :]).astype(np.float32)
    for absorbing in (True, False):
        O, cont = GCIQLFlowAgent._nstep_window_terms(
            jnp.asarray(hits), jnp.asarray(umask), gamma, absorbing)
        O, cont = np.asarray(O), np.asarray(cont)
        O_ref = np.zeros((B, B)); cont_ref = np.ones((B, B))
        for i in range(B):
            for j in range(B):
                notyet = 1.0
                for u in range(ks[i]):
                    h = hits[u, i, j]
                    if absorbing:
                        O_ref[i, j] += gamma**u * h * notyet
                        notyet *= (1.0 - h)
                    else:
                        O_ref[i, j] += gamma**u * h
                cont_ref[i, j] = notyet if absorbing else 1.0
        assert np.allclose(O, O_ref, atol=1e-5), (absorbing, np.abs(O - O_ref).max())
        assert np.allclose(cont, cont_ref, atol=1e-5), absorbing
    print("OK")
    """)


def test_gcdataset_window_fields():
    """GCDataset returns clamped windows + horizons; K=0 adds nothing; the
    geometric mode draws k_i in [1, K]; u=1 equals next_observations."""
    _run("""
    import ml_collections
    from utils.datasets import Dataset, GCDataset
    T1, T2, D = 10, 6, 4
    N = T1 + T2
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(N, D)).astype(np.float32)
    nobs = rng.normal(size=(N, D)).astype(np.float32)
    terminals = np.zeros(N, np.float32); terminals[T1 - 1] = 1; terminals[N - 1] = 1
    ds = Dataset.create(observations=obs, next_observations=nobs, actions=rng.normal(size=(N, 2)).astype(np.float32), terminals=terminals)
    def cfg(K, lam=0.0):
        return ml_collections.ConfigDict(dict(
            discount=0.99, value_p_curgoal=0.0, value_p_trajgoal=1.0, value_p_randomgoal=0.0,
            value_geom_sample=True, actor_p_curgoal=0.0, actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0, actor_geom_sample=False, gc_negative=True,
            p_aug=None, frame_stack=None, measure_nstep=K, measure_nstep_lambda=lam))
    gcd0 = GCDataset(ds, cfg(0))
    b0 = gcd0.sample(8, idxs=np.arange(8))
    assert "mstep_next_obs" not in b0 and "mstep_k" not in b0
    K = 4
    gcd = GCDataset(ds, cfg(K))
    idxs = np.array([0, 7, 9, 10, 13, 15])   # includes both trajectory tails
    b = gcd.sample(len(idxs), idxs=idxs)
    win, ks = b["mstep_next_obs"], b["mstep_k"]
    assert win.shape == (len(idxs), K, D) and ks.shape == (len(idxs),)
    finals = np.array([T1 - 1] * 2 + [T1 - 1] + [N - 1] * 3)
    for r, (i, f) in enumerate(zip(idxs, finals)):
        assert 1 <= ks[r] <= min(K, f - i + 1), (i, f, ks[r])
        assert np.allclose(win[r, 0], nobs[i]), "u=1 must equal next_observations[idx]"
        for u in range(K):
            assert np.allclose(win[r, u], nobs[min(i + u, f)])
    np.random.seed(11)
    bg = GCDataset(ds, cfg(4, lam=0.6)).sample(64, idxs=np.zeros(64, np.int64))
    kg = bg["mstep_k"]
    assert kg.min() >= 1 and kg.max() <= 4 and len(np.unique(kg)) > 1, kg
    print("OK")
    """)


def test_assert_lattice():
    """nstep requires the first-passage region path; refuses the exotic axes."""
    _run("""
    def refuses(**kw):
        try:
            agent_for(base_cfg(**kw))
        except AssertionError:
            return True
        return False
    assert refuses(measure_nstep=4, measure_first_passage=False)
    assert refuses(measure_nstep=4, measure_expectile=True)
    assert refuses(measure_nstep=4, measure_nhat_head=True)
    assert refuses(measure_nstep=4, measure_eps_multiscale='0.2,0.4')
    assert refuses(measure_nstep=4, fb_forward_z=True)
    assert refuses(measure_nstep=0, measure_nstep_lambda=0.5)
    assert not refuses(measure_nstep=4)
    assert not refuses(measure_nstep=4, measure_testpoint_dist='hindsight')
    assert not refuses(measure_nstep=4, measure_fp_absorbing=False)
    print("OK")
    """)


def test_multistep_loss_runs_and_reports_k():
    """K=3 end-to-end: finite loss, k-mean reported, and the target actually
    differs from one-step on a window with later hits."""
    _run("""
    cfg = base_cfg(measure_nstep=3)
    a = agent_for(cfg)
    B = 16
    b = make_batch(B=B, K=3)
    # plant guaranteed later-step hits: window step 3 sits exactly on the columns' cube xyz
    b["mstep_next_obs"][:, 2, 19:22] = b["next_observations"][:, 19:22]
    l, info = measure_loss_of(a, b)
    assert np.isfinite(l)
    assert abs(float(info["measure_nstep_k_mean"]) - 3.0) < 1e-6
    a1 = agent_for(base_cfg(measure_nstep=1))
    b1 = dict(b); b1["mstep_next_obs"] = b["next_observations"][:, None, :]
    b1["mstep_k"] = np.ones(B, np.int32)
    l1, _ = measure_loss_of(a1, b1)
    assert l != l1
    print("OK")
    """)
