"""Tests for scripts/rescuing_fb/eval_zeroshot_rewards.py (spec 07 item 3).

Structure + math only — no env, no checkpoint, no jax:
  (a) reward-sample construction per family on synthetic obs arrays
      (dims/ranges, the [0,1] FP-corner convention, reachavoid signs);
  (b) GPI selection logic: stubbed critic + candidates -> argmax picks the
      right candidate (env-major layout of t6_gpi_eval's batched call);
  (c) the script imports and --help runs (subprocess).
renewal_head.py may not exist yet (spec 07 item 2 runs in parallel): the
infer_z resolution test covers both the present and absent cases, plus a
sys.modules stub.
"""
import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "rescuing_fb" / "eval_zeroshot_rewards.py"


@pytest.fixture(scope="module")
def ezr():
    spec = importlib.util.spec_from_file_location("eval_zeroshot_rewards", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # numpy-only at module level (jax is lazy)
    return mod


def synth_obs(cube_xyz_obs_units, obs_dim=29):
    """Synthetic obs batch with the cube xyz planted at obs[19:22]."""
    cube = np.asarray(cube_xyz_obs_units, np.float64)
    obs = np.zeros((len(cube), obs_dim))
    obs[:, 19:22] = cube
    return obs


# ============================ (a) reward construction ============================

def test_cube_obs_units_slice(ezr):
    obs = synth_obs([[1.0, 2.0, 3.0], [-0.5, 0.0, 0.25]])
    np.testing.assert_allclose(ezr.cube_obs_units(obs),
                               [[1.0, 2.0, 3.0], [-0.5, 0.0, 0.25]])


def test_reward_dirac_indicator_and_hit_convention(ezr):
    goal = np.array([0.0, 0.0, 0.0])
    cube = np.array([[0.0, 0.0, 0.0],      # at goal        -> 1
                     [0.4, 0.0, 0.0],      # exactly eps    -> 1 (<=, FP hit convention)
                     [0.41, 0.0, 0.0],     # just outside   -> 0
                     [5.0, 5.0, 5.0]])     # far            -> 0
    r = ezr.reward_dirac(cube, goal)
    assert r.shape == (4,)
    assert set(np.unique(r)) <= {0.0, 1.0}          # [0,1] convention, NOT gc_negative
    np.testing.assert_allclose(r, [1.0, 1.0, 0.0, 0.0])
    assert ezr.EPS_OBS == 0.4                        # 4 cm in obs units


def test_reward_mixture_is_max_of_diracs(ezr):
    g1, g2 = np.array([0.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])
    cube = np.array([[0.0, 0.0, 0.0],      # near g1 only -> 1
                     [2.0, 0.0, 0.0],      # near g2 only -> 1
                     [1.0, 0.0, 0.0],      # near neither -> 0
                     [0.1, 0.0, 0.0]])     # near g1      -> 1
    r = ezr.reward_mixture(cube, [g1, g2])
    expected = np.maximum(ezr.reward_dirac(cube, g1), ezr.reward_dirac(cube, g2))
    np.testing.assert_allclose(r, expected)
    np.testing.assert_allclose(r, [1.0, 1.0, 0.0, 1.0])
    assert set(np.unique(r)) <= {0.0, 1.0}


def test_reward_region_box_or_ball(ezr):
    c = np.array([0.4, 0.0, 0.02])
    bh, th = ezr.BOX_HALF, ezr.THRESH                # 0.03 box half, 0.04 ball (meters)
    corner = 0.99 * bh                               # just inside the box corner
    cube = np.array([
        c,                                           # center           -> 1
        c + [corner, corner, corner],                # box corner (dist ~0.051 > ball) -> 1 via box
        c + [th - 1e-4, 0.0, 0.0],                   # inside ball, outside box -> 1 via ball
        c + [0.1, 0.0, 0.0],                         # outside both     -> 0
    ])
    r = ezr.reward_region(cube, c)
    np.testing.assert_allclose(r, [1.0, 1.0, 1.0, 0.0])
    assert set(np.unique(r)) <= {0.0, 1.0}
    # sanity of the constructed cases
    assert np.linalg.norm([corner] * 3) > th and not (np.abs([th - 1e-4, 0, 0]) <= bh).all()


def test_reward_dense_range_and_scale(ezr):
    cstar = np.array([0.0, 0.0, 0.0])
    cube = np.array([[0.0, 0.0, 0.0],
                     [ezr.SIGMA_OBS, 0.0, 0.0],      # distance = sigma -> exp(-1)
                     [10.0, 0.0, 0.0]])
    r = ezr.reward_dense(cube, cstar)
    assert r.shape == (3,)
    assert (r > 0).all() and (r <= 1).all()          # (0, 1]
    assert r[0] == 1.0
    np.testing.assert_allclose(r[1], np.exp(-1.0))
    assert r[0] > r[1] > r[2]                        # monotone in distance
    assert ezr.SIGMA_OBS == pytest.approx(2 * ezr.EPS_OBS)   # sigma = 2*eps (spec)


def test_reward_linear_matches_readout_form(ezr):
    rng = np.random.default_rng(0)
    Psi = rng.normal(size=(7, 16))
    z = rng.normal(size=16)
    r = ezr.reward_linear(Psi, z)
    assert r.shape == (7,)
    np.testing.assert_allclose(r, Psi @ z / np.sqrt(16))


def test_reward_reachavoid_signs(ezr):
    cA = np.array([0.3, 0.1, 0.02])
    cB = np.array([0.5, -0.1, 0.02])
    cube = np.array([cA,                              # in A only  -> +1
                     cB,                              # in B only  -> -1
                     [0.4, 0.0, 0.5]])                # in neither -> 0
    r = ezr.reward_reachavoid(cube, cA, cB)
    np.testing.assert_allclose(r, [1.0, -1.0, 0.0])
    assert set(np.unique(r)) <= {-1.0, 0.0, 1.0}
    # overlap cancels to 0 (A == B)
    np.testing.assert_allclose(ezr.reward_reachavoid(cube[:1], cA, cA), [0.0])


def test_sample_sphere_z_norm(ezr):
    rng = np.random.default_rng(3)
    z = ezr.sample_sphere_z(rng, 50, 7.5)
    assert z.shape == (50,)
    assert np.linalg.norm(z) == pytest.approx(7.5, rel=1e-9)


# ============================ z inference (pinned interface) =====================

def test_moment_and_fallback_infer_z_math(ezr):
    rng = np.random.default_rng(1)
    Psi = rng.normal(size=(20, 8))
    r = rng.uniform(0, 1, size=20)
    match_norm = 3.7
    # plain: z direction = mean_j r_j psi_j, norm = match_norm
    z = ezr._fallback_infer_z(Psi, r, None, match_norm)
    z_ref = (r[:, None] * Psi).mean(0)
    assert np.linalg.norm(z) == pytest.approx(match_norm, rel=1e-6)
    np.testing.assert_allclose(z / np.linalg.norm(z), z_ref / np.linalg.norm(z_ref),
                               atol=1e-9)
    # renewal: each sample additionally weighted by R(g_j)
    R = rng.uniform(0.5, 2.0, size=20)
    zR = ezr._fallback_infer_z(Psi, r, R, match_norm)
    zR_ref = ((r * R)[:, None] * Psi).mean(0)
    assert np.linalg.norm(zR) == pytest.approx(match_norm, rel=1e-6)
    np.testing.assert_allclose(zR / np.linalg.norm(zR),
                               zR_ref / np.linalg.norm(zR_ref), atol=1e-9)
    # the raw-moment helper agrees in direction
    np.testing.assert_allclose(ezr.moment_z(Psi, r, R), zR_ref, atol=1e-12)


def test_get_infer_z_resolution_and_stub(ezr):
    # 1) current tree: matches whatever renewal_head provides (or the fallback).
    try:
        import renewal_head  # noqa: F401
        assert ezr.get_infer_z() is renewal_head.infer_z
    except ImportError:
        assert ezr.get_infer_z() is ezr._fallback_infer_z
    # 2) with a stub module injected, the module's infer_z wins (mocked import).
    stub = types.ModuleType("renewal_head")
    sentinel = lambda psi, rewards, R_vals, match_norm: "SENTINEL"  # noqa: E731
    stub.infer_z = sentinel
    saved = sys.modules.get("renewal_head")
    sys.modules["renewal_head"] = stub
    try:
        assert ezr.get_infer_z() is sentinel
    finally:
        if saved is None:
            del sys.modules["renewal_head"]
        else:
            sys.modules["renewal_head"] = saved


# ============================ actor-mode deployment ==============================

def test_z_to_goal_proxy_idx(ezr):
    """actor_cond='goal' decode: argmax_j cos(psi(s'_j), z_r) over the relabel
    sample — norm-INVARIANT (a large-||psi|| misaligned state must not win)."""
    Psi = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.6]])
    assert ezr.z_to_goal_proxy_idx(Psi, np.array([1.0, 0.1])) == 0
    assert ezr.z_to_goal_proxy_idx(Psi, np.array([0.1, 1.0])) == 1
    assert ezr.z_to_goal_proxy_idx(Psi, np.array([1.0, 1.0])) == 2
    # norm invariance: row 0 scaled 100x still loses to the aligned row 1
    Psi_big = np.array([[100.0, 0.0], [0.0, 1.0]])
    assert ezr.z_to_goal_proxy_idx(Psi_big, np.array([0.1, 1.0])) == 1
    # scaling z_r changes nothing
    assert ezr.z_to_goal_proxy_idx(Psi_big, 50.0 * np.array([0.1, 1.0])) == 1


def test_policy_call_mode_dispatch(ezr):
    """mode='z' -> sample_actions_with_z; mode='goal' -> sample_actions(goals=...)."""
    class StubAgent:
        def sample_actions_with_z(self, observations, z, seed, temperature):
            return np.full((len(observations), 5), 1.0)

        def sample_actions(self, observations, goals, seed, temperature):
            return np.full((len(observations), 5), 2.0)

    a = StubAgent()
    obs = np.zeros((3, 28))
    np.testing.assert_allclose(
        ezr._policy_call(a, "z", obs, np.zeros((3, 50)), None), 1.0)
    np.testing.assert_allclose(
        ezr._policy_call(a, "goal", obs, np.zeros((3, 28)), None), 2.0)


# ============================ (b) GPI selection ==================================

def test_gpi_select_stubbed_critic(ezr):
    """Stubbed critic + candidates: argmax must pick the right candidate per env,
    with the env-major layout (row i*C + c) of the batched policy/critic call."""
    n, C, act_dim = 2, 3, 5
    # candidate actions: candidate c of env i = constant (10*i + c)
    acts_big = np.stack([np.full(act_dim, 10 * i + c)
                         for i in range(n) for c in range(C)])

    def stub_critic(obs_big, acts_big_, z_r):
        # env 0: candidate 2 best; env 1: candidate 0 best
        q = {(0, 0): 0.1, (0, 1): 0.5, (0, 2): 0.9,
             (1, 0): 0.8, (1, 1): 0.2, (1, 2): 0.3}
        return np.array([q[(i, c)] for i in range(n) for c in range(C)])

    q_flat = stub_critic(None, acts_big, None)
    jstar = ezr.gpi_select(q_flat, n, C)
    np.testing.assert_array_equal(jstar, [2, 0])
    acts_exec = ezr.gpi_pick_actions(acts_big, jstar, n, C)
    # the executed action is exactly the winning candidate's action
    np.testing.assert_allclose(acts_exec[0], np.full(act_dim, 2.0))    # env0 cand2
    np.testing.assert_allclose(acts_exec[1], np.full(act_dim, 10.0))   # env1 cand0


def test_gpi_select_tie_and_single_candidate(ezr):
    # tie -> first candidate (np.argmax convention); C=1 -> always 0
    np.testing.assert_array_equal(ezr.gpi_select([1.0, 1.0], 1, 2), [0])
    np.testing.assert_array_equal(ezr.gpi_select([0.4, 0.7, 0.1], 3, 1), [0, 0, 0])


# ============================ misc math ==========================================

def test_discounted_reward(ezr):
    r = np.array([1.0, 0.0, 2.0])
    g = ezr.GAMMA
    assert ezr.discounted_reward(r) == pytest.approx(1.0 + 2.0 * g ** 2)
    assert ezr.discounted_reward(np.array([])) == 0.0
    assert ezr.GAMMA == 0.99


def test_frac_time_in_region(ezr):
    c = np.array([0.4, 0.0, 0.02])
    far = c + np.array([0.5, 0.0, 0.0])
    # episode 0: 2 of 4 real steps inside; episode 1: 0 of 2 real steps inside
    traj = np.zeros((2, 5, 3))
    traj[0, :4] = [c, far, c, far]
    traj[0, 4] = c        # PADDED step (beyond lengths) — must be ignored
    traj[1, :2] = [far, far]
    lengths = np.array([4, 2])
    out = ezr.frac_time_in_region(traj, lengths, c)
    np.testing.assert_allclose(out, [0.5, 0.0])


# ============================ (c) import + --help ================================

def test_script_help_runs():
    proc = subprocess.run([sys.executable, str(SCRIPT), "--help"],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    for flag in ("--run-dir", "--tasks", "--gpi", "--z-weighting",
                 "--renewal-head", "--episodes", "--horizon", "--out"):
        assert flag in proc.stdout, f"{flag} missing from --help"


def test_parser_contract_defaults(ezr):
    ap = ezr.build_parser()
    ns = ap.parse_args(["--run-dir", "/tmp/x"])
    assert ns.episodes == 25 and ns.horizon == 200
    assert ns.z_weighting == "plain" and ns.gpi is False
    assert ns.tasks == "1,2,3,4,5"
    assert set(ns.families.split(",")) == set(ezr.ALL_FAMILIES)
    # renewal weighting without a head path must be refused in main (arg-level check)
    ns2 = ap.parse_args(["--run-dir", "/tmp/x", "--z-weighting", "renewal"])
    assert ns2.renewal_head == ""


# --- spec 08 E3: ridge (GLS / whitened-moment) z-inference ----------------------------
# Mirrors the proven evals/z_inference.infer_l1 properties, in numpy on the harness's own
# ridge_z: reduces to the moment DIRECTION under isotropic features; recovers the true z
# (and beats the moment) under a skewed no-ortho psi covariance -- the dense-split forensic.

def _cos(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def test_ridge_z_reduces_to_moment_direction_when_isotropic(ezr):
    rng = np.random.default_rng(0)
    Psi = rng.standard_normal((200_000, 6))          # E[psi psi^T] ~ I
    r = rng.standard_normal(200_000)
    z_mom = ezr.moment_z(Psi, r, None)
    z_rdg = ezr.ridge_z(Psi, r, None, match_norm=1.0, ridge=1e-3)
    assert _cos(z_mom, z_rdg) > 0.99


def test_ridge_z_recovers_true_z_under_skewed_covariance(ezr):
    rng = np.random.default_rng(1)
    d, n = 6, 100_000
    A = rng.standard_normal((d, d))
    L = np.linalg.cholesky(A @ A.T / d + 0.5 * np.eye(d))
    Psi = rng.standard_normal((n, d)) @ L.T          # skewed covariance C != I
    z_true = rng.standard_normal(d)
    r = Psi @ z_true                                  # reward exactly linear in the features
    z_mom = ezr.moment_z(Psi, r, None)
    z_rdg = ezr.ridge_z(Psi, r, None, match_norm=1.0, ridge=1e-4)
    assert _cos(z_rdg, z_true) > 0.99
    assert _cos(z_rdg, z_true) > _cos(z_mom, z_true) + 0.01   # moment is biased by C


def test_ridge_z_renormalizes_to_match_norm(ezr):
    rng = np.random.default_rng(2)
    Psi = rng.standard_normal((5000, 6)); r = rng.standard_normal(5000)
    z = ezr.ridge_z(Psi, r, None, match_norm=3.5, ridge=1e-3)
    assert abs(np.linalg.norm(z) - 3.5) < 1e-4   # +1e-8 guard in the denominator (as moment/retrieval)
