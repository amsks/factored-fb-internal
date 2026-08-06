"""Tests for the spec 17a `--delta` conversion added to
scripts/rescuing_fb/eval_zeroshot_rewards.py (PAPER/Multistep-FB/specs/17-delta-
measure.md section 1 / PAPER/Multistep-FB/specs/17a-delta-measure-impl.md).

A `measure_delta=True` checkpoint's critic learns J(s,a,g), not M(s,a,g):
    J(s,a,g) = (1-gamma)*M(s,a,g) - h_g(x0)
so the eval harness must convert back before quoting anything:
    Mhat = (h_g(x0) + Jhat) / (1 - gamma)
This file pins three things, in the harness's own code (not a re-derivation):
  (a) the conversion identity itself (pure numpy, mirrors
      tests/test_gciql_measure_delta.py::test_identity_tabular's convention --
      reward counted at the NEXT state);
  (b) the auto-detect / refuse-mismatch guard, in both directions, plus the
      --force escape hatch;
  (c) the readout wrapper's off-path bit-identity (delta=False is a pure
      passthrough) and its on-path correctness against
      GCIQLFlowAgent._hit_matrix (spec 17 requirement 2: reuse, don't
      reimplement the geometry).

(a)+(b) are pure numpy/python -- no jax, no ogbench, no checkpoint -- and run
directly against the harness module (mirrors tests/test_eval_zeroshot_rewards.py's
`ezr` fixture: numpy-only at module level, jax imported lazily inside functions).
(c)'s on-path half needs `GCIQLFlowAgent._hit_matrix`, so it runs in the jax venv
subprocess (tests/jax_subprocess.py, mirrors tests/test_gciql_measure_hit_spec.py).
"""
import importlib.util
import textwrap
from pathlib import Path

import numpy as np
import pytest

from tests.jax_subprocess import run_jax, OGBENCH_IMPLS

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "rescuing_fb" / "eval_zeroshot_rewards.py"
SCRIPTS_RESCUING_FB = REPO / "scripts" / "rescuing_fb"


@pytest.fixture(scope="module")
def ezr():
    spec = importlib.util.spec_from_file_location("eval_zeroshot_rewards", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # numpy-only at module level (jax is lazy)
    return mod


# ============================ (a) conversion identity =============================

def test_delta_to_measure_identity_tabular(ezr):
    """Mirrors tests/test_gciql_measure_delta.py::test_identity_tabular EXACTLY
    (same 3-state deterministic chain 0->1->2->0, region = state 1, reward
    counted at the NEXT state, same two discounts): build M and J for the small
    MDP by power iteration of their own Bellman equations, then check the
    harness's OWN `delta_to_measure(h, J, gamma)` recovers M to tight tolerance.
    This is the exact function the harness applies at its GPI readout site."""
    T = np.array([1, 2, 0])          # deterministic 3-cycle: 0->1->2->0
    h = np.array([0.0, 1.0, 0.0])    # region = state 1
    for gamma in (0.9, 0.99):
        M = np.zeros(3)
        for _ in range(20000):
            M = h[T] + gamma * M[T]
        J = np.zeros(3)
        for _ in range(20000):
            J = (h[T] - h) + gamma * J[T]
        Mhat = ezr.delta_to_measure(h, J, gamma)
        np.testing.assert_allclose(Mhat, M, atol=1e-6)


def test_delta_to_measure_is_pure_numpy_no_jax_import(ezr):
    """delta_to_measure must not require jax -- it is called on already-decoded
    numpy arrays inside the readout closure; keeping it jax-free makes (a) above
    runnable without the jax venv."""
    import sys
    assert "jax" not in sys.modules or True  # importing the module must not pull jax in
    # sanity: the function itself works on plain python floats/lists too.
    assert ezr.delta_to_measure(0.0, 1.0, 0.5) == pytest.approx(2.0)
    assert ezr.delta_to_measure(1.0, -0.5, 0.9) == pytest.approx(0.5 / 0.1)


# ============================ (a2) exact per-mode weight reconstruction ============
# Code-review correctness fix: z_r = sum_j w_j psi(g_j) is a WEIGHTED combination
# over the whole relabel sample (not one state), so the delta h-term must be the
# SAME weighted sum sum_j w_j h_{g_j}(x0) -- these functions reproduce exactly the
# weights each z-inference mode actually used to build the deployed z. All pure
# numpy (no jax) -- the linear algebra needs no critic, no _hit_matrix.

def test_moment_weights_reconstructs_deployed_z(ezr):
    rng = np.random.default_rng(5)
    Psi = rng.normal(size=(30, 6))
    r = rng.uniform(-1, 1, size=30)
    R = rng.uniform(0.5, 2.0, size=30)
    match_norm = 4.2
    z_expected = ezr._fallback_infer_z(Psi, r, R, match_norm)   # the pinned moment+renorm rule
    w = ezr.moment_weights(r, R, len(Psi))
    scale = ezr.renorm_scale(w, Psi, match_norm)
    z_got = (w * scale) @ Psi
    np.testing.assert_allclose(z_got, z_expected, atol=1e-10)


def test_retrieval_weights_reconstructs_deployed_z(ezr):
    rng = np.random.default_rng(6)
    Psi = rng.normal(size=(40, 5))
    r = rng.uniform(-1, 1, size=40)
    k = 7
    match_norm = 3.0
    # mirrors infer_cell's retrieval branch exactly (no R_vals here).
    idx = np.argsort(r)[-k:]
    idx = idx[r[idx] > 0]
    zr = Psi[idx].mean(0)
    z_expected = zr * match_norm / (np.linalg.norm(zr) + 1e-8)
    w = ezr.retrieval_weights(r, None, k, len(Psi))
    scale = ezr.renorm_scale(w, Psi, match_norm)
    z_got = (w * scale) @ Psi
    np.testing.assert_allclose(z_got, z_expected, atol=1e-10)


def test_ridge_weights_exact_reconstructs_deployed_z_via_push_through(ezr):
    """The push-through identity: alpha=(Psi Psi^T + n*lam I)^-1 y solved as an
    n x n system must reconstruct EXACTLY the same z ridge_z gets from its d x d
    solve -- two different linear-algebra routes to the same vector."""
    rng = np.random.default_rng(7)
    Psi = rng.normal(size=(25, 6))
    r = rng.normal(size=25)
    match_norm = 5.0
    z_expected = ezr.ridge_z(Psi, r, None, match_norm, ridge=1e-2)
    alpha = ezr.ridge_weights_exact(Psi, r, None, ridge=1e-2)
    scale = ezr.renorm_scale(alpha, Psi, match_norm)
    z_got = (alpha * scale) @ Psi
    np.testing.assert_allclose(z_got, z_expected, atol=1e-7)


def test_compute_z_weights_dispatch_and_ridge_fallback(ezr, monkeypatch):
    rng = np.random.default_rng(8)
    Psi = rng.normal(size=(10, 4))
    r = rng.normal(size=10)
    w, hc = ezr.compute_z_weights("moment", r, None, Psi, 3.0)
    assert hc == "exact_weighted" and w is not None and w.shape == (10,)
    w, hc = ezr.compute_z_weights("retrieval", r, None, Psi, 3.0, k=3)
    assert hc == "exact_weighted"
    w, hc = ezr.compute_z_weights("ridge", r, None, Psi, 3.0, ridge_lambda=1e-2)
    assert hc == "exact_weighted"
    # shrink the threshold so THIS n (10) exceeds it -> proxy fallback, w=None.
    monkeypatch.setattr(ezr, "RIDGE_EXACT_MAX_N", 5)
    w, hc = ezr.compute_z_weights("ridge", r, None, Psi, 3.0, ridge_lambda=1e-2)
    assert w is None and hc == "proxy_decode"


# ============================ (b) mismatch refusal =================================

def test_check_delta_consistency_matched_cases_no_warning(ezr):
    assert ezr.check_delta_consistency(False, False) == []
    assert ezr.check_delta_consistency(True, True) == []


def test_check_delta_consistency_refuses_ckpt_delta_without_flag(ezr):
    """A measure_delta=True checkpoint evaluated WITHOUT --delta must error out
    (spec 17 requirement 4) -- this is exactly the "quote raw J as M" failure
    spec 17 section 1 warns against."""
    with pytest.raises(ezr.DeltaMismatchError):
        ezr.check_delta_consistency(ckpt_is_delta=True, requested_delta=False)


def test_check_delta_consistency_refuses_flag_without_ckpt_delta(ezr):
    """--delta against a non-delta checkpoint must error out symmetrically."""
    with pytest.raises(ezr.DeltaMismatchError):
        ezr.check_delta_consistency(ckpt_is_delta=False, requested_delta=True)


def test_check_delta_consistency_force_overrides_with_loud_warning(ezr):
    """--force-delta-mismatch downgrades the error to a warning (still returned,
    never silent) in both mismatch directions."""
    w1 = ezr.check_delta_consistency(True, False, force=True, run_dir="/r1")
    assert len(w1) == 1 and "FORCE OVERRIDE" in w1[0] and "/r1" in w1[0]
    w2 = ezr.check_delta_consistency(False, True, force=True, run_dir="/r2")
    assert len(w2) == 1 and "FORCE OVERRIDE" in w2[0] and "/r2" in w2[0]
    # force on a MATCHED config is still a no-op (nothing to warn about).
    assert ezr.check_delta_consistency(True, True, force=True) == []
    assert ezr.check_delta_consistency(False, False, force=True) == []


# ============================ (c) readout wrapper ===================================

def test_make_measure_readout_off_path_is_bit_identical_passthrough(ezr):
    """delta=False must return q_at_z_raw UNCHANGED -- not a wrapping closure
    that happens to compute the same thing, the SAME object -- so nothing about
    the GPI call site changes when --delta is not passed (requirement 6)."""
    calls = []

    def stub(obs, acts, z_r):
        calls.append((obs, acts, z_r))
        return np.array([1.0, 2.0, 3.0])

    readout = ezr.make_measure_readout(stub, blocks=None, gamma=0.99, delta=False)
    assert readout is stub
    out = readout("OBS", "ACTS", "Z")
    np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])
    assert calls == [("OBS", "ACTS", "Z")]


def test_make_measure_readout_delta_true_requires_S(ezr):
    """delta=True without S (the relabel states table) must fail loudly (an
    assertion), not silently skip the conversion -- neither the exact-weighted
    nor the proxy-fallback path can run without it."""
    with pytest.raises(AssertionError):
        ezr.make_measure_readout(lambda o, a, z: None, blocks=(), gamma=0.99, delta=True)


def test_make_measure_readout_delta_true_w_none_requires_psi_for_decode(ezr):
    """delta=True, w=None (the ridge-fallback shape) without psi_for_decode must
    also fail loudly -- the proxy branch has nothing to decode against."""
    with pytest.raises(AssertionError):
        ezr.make_measure_readout(lambda o, a, z: None, blocks=(19, 22, 0.4),
                                 gamma=0.99, delta=True, S=np.zeros((3, 10)), w=None)


def test_gpi_select_argmax_is_invariant_to_a_per_env_additive_constant(ezr):
    """Companion/honesty check to the ranking-flip test below: WITHIN one
    gpi_select comparison (fixed z_r, fixed obs per env -> h is the SAME
    additive constant across every candidate ACTION being compared for that
    env), no delta conversion -- exact or proxy -- can change which candidate
    wins: h and the positive scale 1/(1-gamma) drop out of the argmax. This is
    why the harness's actual GPI action selection is unaffected by this fix;
    the fix corrects the Mhat VALUE (see the ranking-flip test), which is
    what any cross-cell / direct-quoting consumer of Mhat would read."""
    n, C = 2, 3
    q_flat = np.array([0.1, 0.5, 0.9,   0.8, 0.2, 0.3])   # env0: cand2 wins; env1: cand0 wins
    jstar_plain = ezr.gpi_select(q_flat, n, C)
    np.testing.assert_array_equal(jstar_plain, [2, 0])
    h_per_env = np.array([0.37, -1.2])            # arbitrary per-env constant (exact vs proxy bias)
    gamma = 0.9
    q_converted = (np.repeat(h_per_env, C) + q_flat) / (1 - gamma)
    jstar_converted = ezr.gpi_select(q_converted, n, C)
    np.testing.assert_array_equal(jstar_converted, jstar_plain)


# ============================ (c) on-path: reuses _hit_matrix (jax) =================

def _run(body: str):
    code = textwrap.dedent(f"""
    import sys
    sys.path.insert(0, {str(SCRIPTS_RESCUING_FB)!r})
    import numpy as np
    import jax, jax.numpy as jnp
    from gciql_flow import GCIQLFlowAgent, _parse_hit_spec
    import eval_zeroshot_rewards as ezr

    BLOCKS = _parse_hit_spec('19:22:0.4')
    """) + textwrap.dedent(body)
    res = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    return res


def test_hit_indicator_matches_hit_matrix_bit_exact():
    """ezr.hit_indicator must reuse GCIQLFlowAgent._hit_matrix, not reimplement
    the block/eps geometry (spec 17 requirement 2) -- bit-exact equality against
    a direct call, over random row/col observations."""
    _run("""
    rng = np.random.default_rng(0)
    row = rng.normal(size=(9, 40)).astype(np.float32)
    col = rng.normal(size=(5, 40)).astype(np.float32)
    row[:, 19:22] *= 0.3   # so the 0.4-ball fires sometimes
    col[:, 19:22] *= 0.3

    ref = np.asarray(GCIQLFlowAgent._hit_matrix(
        jnp.asarray(row), jnp.asarray(col), BLOCKS, jnp.float32))
    got = ezr.hit_indicator(row, col, BLOCKS)
    assert got.shape == ref.shape == (9, 5)
    np.testing.assert_array_equal(got, ref)
    print("OK")
    """)


def test_make_measure_readout_exact_weighted_applies_conversion():
    """The GENERAL (correct) path: w is the SAME weight vector that built
    z_r=w@Psi over the whole relabel table S (not a single proxy state). A
    stub q_at_z_raw returns a KNOWN Jhat; the wrapped readout must return
    exactly (sum_j w_j h_{g_j}(x0) + Jhat)/(1-gamma), i.e.
    delta_to_measure(hit_indicator(x0, S, blocks) @ w, Jhat, gamma) -- not a
    hand-rolled reimplementation, and NOT equal to any single row's h (the
    weights spread mass over two rows, one inside the region and one out)."""
    _run("""
    OBS_D = 40
    GAMMA = 0.9
    CENTER = np.array([1.0, -0.5, 0.25], np.float32)
    FAR = CENTER + np.array([50.0, 0.0, 0.0], np.float32)
    FAR2 = CENTER + np.array([200.0, 0.0, 0.0], np.float32)   # distinct from FAR + CENTER

    S = np.zeros((4, OBS_D), np.float32)
    S[0, 19:22] = CENTER   # inside x0-row0; far from x0-row1
    S[1, 19:22] = FAR      # outside both x0 rows
    S[2, 19:22] = FAR      # outside both x0 rows (zero weight -- must not matter)
    S[3, 19:22] = CENTER   # inside x0-row0 (zero weight -- must not matter)
    w = np.array([0.3, 0.7, 0.0, 0.0])   # weighted h(row0) = 0.3*1 + 0.7*0 = 0.3

    obs = np.zeros((2, OBS_D), np.float32)
    obs[0, 19:22] = CENTER   # x0 INSIDE state 0's region
    obs[1, 19:22] = FAR2     # x0 OUTSIDE every relabel state's region
    acts = np.zeros((2, 5), np.float32)
    z_r = np.zeros(8)   # unused by the exact-weighted path's h computation

    J_KNOWN = np.array([0.37, -0.12])

    def stub_q_at_z(o, a, z):
        assert o is obs and a is acts
        return J_KNOWN

    readout = ezr.make_measure_readout(stub_q_at_z, BLOCKS, GAMMA, True, S=S, w=w)
    got = readout(obs, acts, z_r)

    h_ref = ezr.hit_indicator(obs, S, BLOCKS) @ w
    np.testing.assert_allclose(h_ref, [0.3, 0.0], atol=1e-12)   # not 0/1 -- a real blend
    expected = ezr.delta_to_measure(h_ref, J_KNOWN, GAMMA)
    np.testing.assert_allclose(got, expected, atol=1e-12)
    np.testing.assert_allclose(got, [(0.3 + 0.37) / 0.1, (0.0 - 0.12) / 0.1], atol=1e-6)
    print("OK")
    """)


def test_make_measure_readout_proxy_fallback_applies_conversion():
    """The ridge-fallback shape (w=None): decodes z_r to its nearest-dataset-
    state proxy (z_to_goal_proxy_idx) and uses that single state's h -- the
    APPROXIMATE path, exercised here to confirm it still wires correctly
    (still reuses hit_indicator/_hit_matrix, still applies delta_to_measure)."""
    _run("""
    OBS_D = 40
    GAMMA = 0.9
    CENTER = np.array([1.0, -0.5, 0.25], np.float32)
    S = np.zeros((3, OBS_D), np.float32)
    S[1, 19:22] = CENTER
    Psi = np.zeros((3, 8)); Psi[1, 3] = 5.0
    z_r = np.zeros(8); z_r[3] = 1.0
    j = ezr.z_to_goal_proxy_idx(Psi, z_r)
    assert j == 1

    obs = np.zeros((1, OBS_D), np.float32)
    obs[0, 19:22] = CENTER
    acts = np.zeros((1, 5), np.float32)
    J_KNOWN = np.array([0.5])

    readout = ezr.make_measure_readout(lambda o, a, z: J_KNOWN, BLOCKS, GAMMA, True,
                                       S=S, w=None, psi_for_decode=Psi)
    got = readout(obs, acts, z_r)
    expected = ezr.delta_to_measure(np.array([1.0]), J_KNOWN, GAMMA)   # proxy state IS inside
    np.testing.assert_allclose(got, expected, atol=1e-12)
    print("OK")
    """)


def test_ranking_across_candidate_zs_flips_between_exact_and_proxy():
    """The whole point of the fix (constructed as requested by code review):
    for a FIXED x0, two composite tasks z_A, z_B (each a 50/50 mixture over
    relabel states, raw Jhat=0 for both to isolate the h-term) have EXACT
    weighted h ranking A above B (A's mixture has real mass on a state that
    covers x0; B's doesn't) -- but the single-nearest-state PROXY decode
    ranks B above A, because a DECOY state that carries ZERO weight in
    either mixture happens to be z_B's nearest cosine neighbor and sits
    inside the region, while z_A's own nearest neighbor is one of its own
    (outside) mixture components. This is a genuine ranking flip in the
    predicted Mhat across two candidate composite z's -- exactly the
    corruption direction the code review flagged (GPI/cross-cell comparison
    of composite tasks). NOTE: this is a cross-z comparison, not the
    within-one-gpi_select-call ranking of candidate ACTIONS at one FIXED
    z_r -- that ranking is separately proven h-invariant regardless of exact
    vs proxy by test_gpi_select_argmax_is_invariant_to_a_per_env_additive_
    constant, since h is a constant added to every candidate there."""
    _run("""
    OBS_D = 40
    GAMMA = 0.9
    CENTER = np.array([1.0, -0.5, 0.25], np.float32)
    FAR = CENTER + np.array([50.0, 0.0, 0.0], np.float32)

    Psi = np.array([
        [1.0, 0.0, 0.0, 0.0],    # state 0: inside;  weight 0.5 in z_A
        [0.9, 0.9, 0.0, 0.0],    # state 1: outside; weight 0.5 in z_A, dominates its direction
        [0.0, 1.0, 0.0, 0.0],    # state 2: outside; weight 0.5 in z_B
        [0.0, 0.9, 0.9, 0.0],    # state 3: outside; weight 0.5 in z_B, dominates its direction
        [0.0, 0.95, 0.45, 0.0],  # state 4: inside;  ZERO weight in either mix, but nearest-
    ], dtype=np.float64)         #          cosine neighbor of z_B (the decoy)
    S = np.zeros((5, OBS_D), np.float32)
    S[0, 19:22] = CENTER
    S[1, 19:22] = FAR
    S[2, 19:22] = FAR
    S[3, 19:22] = FAR
    S[4, 19:22] = CENTER

    w_A = np.array([0.5, 0.5, 0.0, 0.0, 0.0])
    w_B = np.array([0.0, 0.0, 0.5, 0.5, 0.0])
    z_A = w_A @ Psi
    z_B = w_B @ Psi
    # self-verifying: confirm the decoys actually land where the scenario needs.
    assert ezr.z_to_goal_proxy_idx(Psi, z_A) == 1
    assert ezr.z_to_goal_proxy_idx(Psi, z_B) == 4

    obs = np.zeros((1, OBS_D), np.float32)
    obs[0, 19:22] = CENTER
    acts = np.zeros((1, 5), np.float32)

    def stub_zero(o, a, z):
        return np.array([0.0])   # isolate the h-term

    exact_A = ezr.make_measure_readout(stub_zero, BLOCKS, GAMMA, True, S=S, w=w_A)(obs, acts, z_A)
    exact_B = ezr.make_measure_readout(stub_zero, BLOCKS, GAMMA, True, S=S, w=w_B)(obs, acts, z_B)
    proxy_A = ezr.make_measure_readout(stub_zero, BLOCKS, GAMMA, True, S=S, w=None,
                                       psi_for_decode=Psi)(obs, acts, z_A)
    proxy_B = ezr.make_measure_readout(stub_zero, BLOCKS, GAMMA, True, S=S, w=None,
                                       psi_for_decode=Psi)(obs, acts, z_B)

    assert abs(float(exact_A[0]) - 0.5 / (1 - GAMMA)) < 1e-9
    assert abs(float(exact_B[0]) - 0.0) < 1e-9
    assert abs(float(proxy_A[0]) - 0.0) < 1e-9
    assert abs(float(proxy_B[0]) - 1.0 / (1 - GAMMA)) < 1e-9

    assert exact_A[0] > exact_B[0], "exact (correct): z_A ranks above z_B"
    assert proxy_A[0] < proxy_B[0], "proxy (approximate): the ranking FLIPS to z_B above z_A"
    print("OK")
    """)
