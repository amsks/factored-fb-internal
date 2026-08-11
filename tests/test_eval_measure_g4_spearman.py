"""Tests for scripts/rescuing_fb/eval_measure_g4_spearman.py (spec 17 sec.6 /
spec 18 sec.7 / spec 19 sec.1 gate G4).

Pure numpy/scipy -- no env, no checkpoint, no jax (the script stays jax-free
at module level; same discipline as tests/test_eval_density_zeroshot.py).

Added 2026-08-06 after an adversarial re-verification
(`.superpowers/sdd/spec19-ors/g4-verification.md`) found the first version of
this script's readout silently injected a spurious term (dropped `r(x0)` but
kept its paired `-kappa(x0)`/`-phi(x0)` subtraction), and that its Spearman
helper broke ties by array order instead of averaging -- both of which
suppressed the gate's own statistic without raising any error. These tests
are the two decisive sanity controls the verification report used to make
the bug diagnosable (`spearman_tie` on the MC estimate fed back as the model
estimate must score ~1.0; on pure noise, ~0), plus a regression pin on the
readout formula itself so the r(x0)/kappa(x0) pairing bug cannot recur
silently.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "rescuing_fb" / "eval_measure_g4_spearman.py"


@pytest.fixture(scope="module")
def g4():
    spec = importlib.util.spec_from_file_location("eval_measure_g4_spearman", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # numpy/scipy-only at module level (jax is lazy)
    return mod


# ============================ spearman_tie sanity floor/ceiling ====================
# The decisive control from g4-verification.md item 4: feeding the MC estimate
# itself back in as "the model estimate" must score ~1.0 (perfect self-agreement);
# pure noise must score ~0 (no spurious structure). A readout bug that injects an
# anti-correlated term, or a Spearman bug that suppresses ties, would fail the
# FIRST of these two controls even though nothing raises an exception -- which is
# exactly how the original bug went undetected.

def test_spearman_tie_mc_vs_itself_is_near_one(g4):
    rng = np.random.default_rng(0)
    # Realistic shape: ~60% exact zeros (degenerate MC, as observed on the real
    # probe), the rest a discounted-sum-like positive tail.
    mc = np.where(rng.random(900) < 0.6, 0.0, rng.exponential(5.0, 900))
    assert g4.spearman_tie(mc, mc) == pytest.approx(1.0, abs=1e-9)


def test_spearman_tie_monotone_transform_is_near_one(g4):
    rng = np.random.default_rng(1)
    mc = np.where(rng.random(900) < 0.6, 0.0, rng.exponential(5.0, 900))
    model = np.log1p(mc) * 3.0 + 7.0     # strictly monotone (log1p is injective on mc>=0)
    assert g4.spearman_tie(mc, model) == pytest.approx(1.0, abs=1e-9)


def test_spearman_tie_pure_noise_is_near_zero(g4):
    rng = np.random.default_rng(2)
    mc = np.where(rng.random(900) < 0.6, 0.0, rng.exponential(5.0, 900))
    noise = rng.normal(size=900)
    assert abs(g4.spearman_tie(mc, noise)) < 0.1


def test_spearman_tie_beats_ordinal_on_heavy_ties(g4):
    """The concrete failure mode (item 3): a model with a PERFECT ordering on
    the non-degenerate points and an ARBITRARY (but fixed, noise-driven)
    ordering within the zero-tie block -- exactly `g4-verification.md` item
    4's own "perfect ordering on mc>0, arbitrary within the mc==0 ties"
    control, which it reports scoring 0.751-0.772 under the OLD ordinal
    function. Note tie-correction does not (and mathematically cannot) push
    this to 1.0: y genuinely varies within the x-tied block, and a
    correctly tie-corrected correlation reports that as unexplained
    variance rather than hiding it -- the point of this test is that
    `spearman_tie` scores MEASURABLY HIGHER than the ordinal function on the
    SAME data (ordinal's tie-breaking-by-array-position is pure noise
    relative to `model`, actively hurting it further), not that either
    reaches 1.0. Values pinned to this exact construction: tie~0.88,
    ordinal~0.76 (seed-independent to within a few hundredths)."""
    rng = np.random.default_rng(3)
    n_zero, n_pos = 540, 360      # ~60% zeros, matching the real probe's degenerate_frac
    mc = np.concatenate([np.zeros(n_zero), np.sort(rng.exponential(5.0, n_pos))])
    model = np.concatenate([rng.normal(size=n_zero), np.arange(n_pos, dtype=float)])
    tie_corrected = g4.spearman_tie(mc, model)
    ordinal = g4.spearman(mc, model)
    assert tie_corrected > 0.85
    assert ordinal < 0.80, f"expected ordinal ranking to be visibly suppressed by ties, got {ordinal}"
    assert tie_corrected - ordinal > 0.08, (tie_corrected, ordinal)


def test_spearman_tie_nan_guards(g4):
    assert np.isnan(g4.spearman_tie([1, 2], [1, 2]))          # n < 3
    assert np.isnan(g4.spearman_tie([1, 1, 1], [1, 2, 3]))    # zero variance


# ============================ readout formula regression pin =======================
# Locks in the fix: the gated readout is w^T r_pool / (1-gamma) (density) /
# p^T z_r / (1-gamma) (partition) -- i.e. `compute_density_primary`/
# `compute_Qr_from_dots` called with the SECOND operand (kappa_x0/phi-term)
# zeroed, NOT with r_x0 zeroed while keeping a nonzero kappa_x0/phi-term. The
# two are numerically different whenever kappa_x0/phi(x0) has mass in the box,
# which is exactly the bug: this test pins the correct call shape by asserting
# it disagrees with the buggy one whenever that mass is nonzero.

def test_density_readout_is_w_dot_rpool_not_the_kappa_x0_variant(g4):
    w = np.array([0.5, 0.3, 0.2])
    kappa_x0 = np.array([0.2, 0.2, 0.6])     # nonzero mass in the box -> bug would show here
    r_pool = np.array([1.0, 0.0, 2.0])       # box indicator (0/1 in the real script; any r here)
    gamma = 0.99

    correct = g4.compute_density_primary(0.0, w, np.zeros_like(kappa_x0), r_pool, gamma)
    assert correct == pytest.approx(np.sum(w * r_pool) / (1 - gamma), abs=1e-12)

    buggy = g4.compute_density_primary(0.0, w, kappa_x0, r_pool, gamma)   # the 2026-08-06 bug
    assert buggy != pytest.approx(correct, abs=1e-6), (
        "the buggy (r_x0=0, kappa_x0 kept) and correct (kappa_x0=0) readouts must "
        "differ whenever kappa_x0 has box mass -- if this ever passes, the fix has "
        "regressed back to the bugged call shape")


def test_partition_readout_is_p_dot_zr_not_the_phi_x0_variant(g4):
    p_dot_zr = 0.42
    phi_dot_zr = 0.35   # nonzero mass in the box -> bug would show here
    gamma = 0.99

    correct = g4.compute_Qr_from_dots(0.0, p_dot_zr, 0.0, gamma)
    assert correct == pytest.approx(p_dot_zr / (1 - gamma), abs=1e-12)

    buggy = g4.compute_Qr_from_dots(0.0, p_dot_zr, phi_dot_zr, gamma)     # the 2026-08-06 bug
    assert buggy != pytest.approx(correct, abs=1e-6)


# ============================ import/parser smoke ==================================

def test_build_parser_defaults(g4):
    ap = g4.build_parser()
    args = ap.parse_args(["--run-dir", "/tmp/x"])
    assert args.checkpoint_step == 500000
    assert args.object == "auto"
    assert args.starts == 24 and args.mc == 2 and args.horizon == 200 and args.boxes == 12
    assert args.pool_size == 16384


# ==================== head aggregation (2026-08-11 rescore fix) ====================
# The deployed partition readout (eval_partition_zeroshot.py) computes the dot with
# z_r PER HEAD and then mins the SCALARS. The probe's first version instead took the
# elementwise min of the two softmax vectors (a sub-normalized vector, density's
# convention) and dotted once -- a different, systematically deflated estimator. The
# G4 statistic must be computed on the estimator the method actually deploys.

def test_partition_head_agg_is_dot_then_min(g4):
    rng = np.random.default_rng(0)
    p_h = rng.dirichlet(np.ones(6), size=(2, 1))          # [2, 1, K] two softmax heads
    z_r = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    got = g4.partition_box_mass(p_h, z_r, double_q=True)   # [1]
    want = min(float(p_h[0, 0] @ z_r), float(p_h[1, 0] @ z_r))
    assert got[0] == pytest.approx(want, abs=1e-12)
    vecmin = float(np.minimum(p_h[0, 0], p_h[1, 0]) @ z_r)
    assert vecmin < want - 1e-9, "construction must separate the two conventions"
    assert got[0] != pytest.approx(vecmin, abs=1e-9), (
        "regressed to elementwise-vector-min-then-dot (the mismatched estimator)")


def test_partition_head_agg_single_head(g4):
    p_h = np.array([[[0.2, 0.3, 0.5]]])                    # [1, 1, K]
    z_r = np.array([1.0, 0.0, 1.0])
    got = g4.partition_box_mass(p_h, z_r, double_q=False)
    assert got[0] == pytest.approx(0.7, abs=1e-12)
