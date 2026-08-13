"""Tests for scripts/rescuing_fb/eval_density_z_rewards.py -- spec 24 sec.3
E2 zero-shot reward-to-index readout for `measure_density_z` checkpoints
(PAPER/Multistep-FB/specs/24-z-indexed-density.md).

Pure numpy -- no env, no checkpoint, no jax (the script stays jax-free at
module level, same discipline as tests/test_eval_density_zeroshot.py). Covers
exactly what the task brief asked for: (a) a numpy reference for the E2a
z_r construction, (b) a determinism check (bit-identical repeat), (c) the
measure_density_z consistency-refusal test (both directions, force-override),
plus (d) the E2b scoring helper's own numpy reference and (e) import/--help/
parser-default smoke, mirroring test_eval_density_zeroshot.py's own (e)
section.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "rescuing_fb" / "eval_density_z_rewards.py"


@pytest.fixture(scope="module")
def dzr():
    spec = importlib.util.spec_from_file_location("eval_density_z_rewards", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # numpy-only at module level (jax is lazy)
    return mod


# ============================ (a) numpy reference, E2a z_r construction ============

def test_analytic_zr_matches_independent_numpy_rederivation(dzr):
    """d=2, P=3, hand-buildable inputs (mirrors
    test_secondary_matches_handbuilt_numpy_reference's own "independent
    re-derivation, not a by-hand guess" discipline): B_bar=[[1,0],[0,1],[1,1]],
    r=[2,3,4]. z_r must equal sqrt(d)*mean(r[:,None]*B)/(||.||+1e-6) computed
    completely independently of `analytic_zr`'s own implementation."""
    b_pool_target = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    r_pool = np.array([2.0, 3.0, 4.0])
    d = 2
    z = dzr.analytic_zr(r_pool, b_pool_target, d)

    num_ref = np.mean(r_pool[:, None] * b_pool_target, axis=0)
    z_ref = num_ref * np.sqrt(2.0) / (np.linalg.norm(num_ref) + 1e-6)
    np.testing.assert_allclose(z, z_ref, atol=1e-12)
    # sanity: not degenerate, correct shape
    assert z.shape == (2,)
    assert np.linalg.norm(z) == pytest.approx(np.sqrt(2.0), abs=1e-6)


def test_analytic_zr_norm_matches_sphere_convention(dzr):
    """||z_r|| must equal sqrt(d) to the SAME tolerance gciql_flow._sphere's
    own convention gives (x*sqrt(d)/(||x||+1e-6) -- not a bare unit
    normalize), on random inputs, across several d/P -- an apples-to-apples
    comparison against E2b's candidates (built via _density_z_encode, which
    calls _sphere) is the entire point of gate G3, so this convention match
    is load-bearing, not cosmetic."""
    rng = np.random.default_rng(0)
    for d, P in ((2, 5), (8, 50), (50, 200)):
        b_pool_target = rng.normal(size=(P, d))
        r_pool = rng.normal(size=P)
        z = dzr.analytic_zr(r_pool, b_pool_target, d)
        assert z.shape == (d,)
        assert np.linalg.norm(z) == pytest.approx(np.sqrt(float(d)), rel=1e-5)


def test_analytic_zr_rejects_shape_mismatch(dzr):
    with pytest.raises(AssertionError):
        dzr.analytic_zr(np.zeros(3), np.zeros((4, 5)), 5)


# ============================ (b) determinism: bit-identical repeat ================

def test_analytic_zr_deterministic_repeat(dzr):
    """Two calls with IDENTICAL numpy inputs must be bit-for-bit identical --
    analytic_zr has no rng, no jax, no global state; this is the "numpy
    reference reproducible from a restored checkpoint's target tower alone"
    property spec 24 sec.1 requires of E1's transform, checked here at the
    pure-numpy layer this script owns."""
    rng = np.random.default_rng(11)
    b_pool_target = rng.normal(size=(64, 10))
    r_pool = rng.normal(size=64)
    z1 = dzr.analytic_zr(r_pool, b_pool_target, 10)
    z2 = dzr.analytic_zr(r_pool.copy(), b_pool_target.copy(), 10)
    np.testing.assert_array_equal(z1, z2)


def test_z_search_scores_deterministic_repeat(dzr):
    rng = np.random.default_rng(12)
    w = rng.random((20, 30))
    kappa = rng.random(30)
    r_pool = rng.normal(size=30)
    s1 = dzr.z_search_scores(w, kappa, r_pool)
    s2 = dzr.z_search_scores(w.copy(), kappa.copy(), r_pool.copy())
    np.testing.assert_array_equal(s1, s2)


# ============================ (d) E2b scoring: numpy reference =====================

def test_z_search_scores_matches_handbuilt_reference(dzr):
    """P=3, N=2 candidates, hand-computed: w = [[0.5,0.3,0.2],[0.2,0.5,0.3]],
    kappa=[0.2,0.2,0.6], r_pool=[1,0,2].
    cand0: (w-kappa).r = 0.3*1 + 0.1*0 + (-0.4)*2 = -0.5
    cand1: (w-kappa).r = 0.0*1 + 0.3*0 + (-0.3)*2 = -0.6"""
    w = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3]])
    kappa = np.array([0.2, 0.2, 0.6])
    r_pool = np.array([1.0, 0.0, 2.0])
    scores = dzr.z_search_scores(w, kappa, r_pool)
    np.testing.assert_allclose(scores, [-0.5, -0.6], atol=1e-10)


def test_z_search_scores_argmax_matches_reference_via_compute_density_primary(dzr):
    """Cross-check against the REUSED compute_density_primary import (not
    reimplemented): the candidate-independent r(x0)/(1-gamma) offset should
    not change the argmax ranking -- score_j (this module) ranks IDENTICALLY
    to the full Q_r_j (compute_density_primary) for a FIXED r(x0)/gamma."""
    rng = np.random.default_rng(13)
    N, P = 10, 25
    w = rng.random((N, P))
    w /= w.sum(axis=-1, keepdims=True)
    kappa = rng.random(P)
    kappa /= kappa.sum()
    r_pool = rng.normal(size=P)
    scores = dzr.z_search_scores(w, kappa, r_pool)
    full_q = np.array([dzr.compute_density_primary(0.7, w[j], kappa, r_pool, 0.9)
                       for j in range(N)])
    assert int(np.argmax(scores)) == int(np.argmax(full_q))


def test_z_search_scores_rejects_shape_mismatch(dzr):
    with pytest.raises(AssertionError):
        dzr.z_search_scores(np.zeros((3, 4)), np.zeros(5), np.zeros(4))


# ============================ (c) consistency refusals =============================

def test_check_density_z_consistency_refuses_without_density_z(dzr):
    with pytest.raises(dzr.DensityZMismatchError):
        dzr.check_density_z_consistency(True, False, force=False, run_dir="<run>")


def test_check_density_z_consistency_force_warns_not_raises(dzr):
    warnings = dzr.check_density_z_consistency(True, False, force=True, run_dir="<run>")
    assert len(warnings) == 1
    assert "FORCE OVERRIDE" in warnings[0]
    assert "<run>" in warnings[0]


def test_check_density_z_consistency_also_refuses_non_density_base(dzr):
    """Base measure_density=False must ALSO be refused (the wrapped
    check_density_consistency's own branch), independent of the z flag --
    both checks compose, neither one is silently skipped."""
    with pytest.raises(dzr.DensityMismatchError):
        dzr.check_density_z_consistency(False, True, force=False, run_dir="<run>")


def test_check_density_z_consistency_matching_config_is_silent(dzr):
    assert dzr.check_density_z_consistency(True, True, force=False, run_dir="<run>") == []


def test_density_z_mismatch_error_is_a_density_mismatch_error(dzr):
    """Subclass relationship: callers catching the base DensityMismatchError
    (e.g. eval_density_zeroshot.py precedent) still catch this one."""
    assert issubclass(dzr.DensityZMismatchError, dzr.DensityMismatchError)


# ============================ (c') control-arm consistency refusals ================

def test_check_control_consistency_refuses_z_indexed_checkpoint(dzr):
    """The control arm's whole point is a G-INDEXED comparison -- a
    measure_density_z=True checkpoint must be refused, the INVERSE of
    check_density_z_consistency's own z-branch."""
    with pytest.raises(dzr.DensityZMismatchError):
        dzr.check_control_consistency(True, True, force=False, run_dir="<run>")


def test_check_control_consistency_force_warns_not_raises(dzr):
    warnings = dzr.check_control_consistency(True, True, force=True, run_dir="<run>")
    assert len(warnings) == 1
    assert "FORCE OVERRIDE" in warnings[0]
    assert "<run>" in warnings[0]


def test_check_control_consistency_also_refuses_non_density_base(dzr):
    """Base measure_density=False must still be refused (reused
    check_density_consistency branch), independent of the z flag."""
    with pytest.raises(dzr.DensityMismatchError):
        dzr.check_control_consistency(False, False, force=False, run_dir="<run>")


def test_check_control_consistency_matching_gindexed_config_is_silent(dzr):
    """measure_density=True, measure_density_z=False (or absent, passed as
    False here) -- exactly the p0_pmm_dens_s0 shape -- must be silent."""
    assert dzr.check_control_consistency(True, False, force=False, run_dir="<run>") == []


def test_e2_and_control_refusals_are_mutual_inverses_on_the_z_flag(dzr):
    """A checkpoint that check_density_z_consistency ACCEPTS (z=True) must be
    REFUSED by check_control_consistency, and vice versa -- neither arm can
    run the other's checkpoint type by accident."""
    assert dzr.check_density_z_consistency(True, True, force=False, run_dir="<r>") == []
    with pytest.raises(dzr.DensityZMismatchError):
        dzr.check_control_consistency(True, True, force=False, run_dir="<r>")
    assert dzr.check_control_consistency(True, False, force=False, run_dir="<r>") == []
    with pytest.raises(dzr.DensityZMismatchError):
        dzr.check_density_z_consistency(True, False, force=False, run_dir="<r>")


# ============================ (e) import / --help / parser defaults ================

def test_script_help_runs():
    proc = subprocess.run([sys.executable, str(SCRIPT), "--help"],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    for flag in ("--run-dir", "--checkpoint-step", "--arm", "--families", "--episodes",
                 "--horizon", "--seed", "--out", "--pool_size", "--e2b_candidate_size",
                 "--retrieval-k", "--ridge", "--force"):
        assert flag in proc.stdout, f"{flag} missing from --help"
    assert "control" in proc.stdout


def test_script_requires_arm():
    proc = subprocess.run([sys.executable, str(SCRIPT), "--run-dir", "/tmp/x"],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode != 0
    assert "--arm" in proc.stderr


def test_parser_contract_defaults(dzr):
    ap = dzr.build_parser()
    ns = ap.parse_args(["--run-dir", "/tmp/x", "--arm", "e2a"])
    assert ns.episodes == 25 and ns.horizon == 0
    assert ns.tasks == "1,2,3,4,5"
    assert set(ns.families.split(",")) == set(dzr.FAMILIES)
    assert ns.pool_size == 16384
    assert ns.e2b_candidate_size == 4096
    assert ns.ridge == pytest.approx(1e-4)
    assert ns.force is False
    assert ns.retrieval_k == 8
    ns2 = ap.parse_args(["--run-dir", "/tmp/x", "--arm", "e2b"])
    assert ns2.arm == "e2b"
    ns3 = ap.parse_args(["--run-dir", "/tmp/x", "--arm", "control"])
    assert ns3.arm == "control"
    with pytest.raises(SystemExit):
        ap.parse_args(["--run-dir", "/tmp/x", "--arm", "bogus"])


def test_families_constant_matches_partition_harness(dzr):
    assert set(dzr.FAMILIES) == {"dirac", "dense", "region"}


def test_arms_constant(dzr):
    assert set(dzr.ARMS) == {"e2a", "e2b", "control"}
