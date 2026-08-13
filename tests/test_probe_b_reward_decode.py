"""Tests for scripts/rescuing_fb/probe_b_reward_decode.py -- spec 25 P0
(coordinator, 2026-08-13): the B-decodability ridge-R^2 probe
(`PAPER/Multistep-FB/specs/25-z-coupling-port.md` sec.1).

Pure numpy -- no jax, no env, no checkpoint (mirrors this repo's usual
jax-free-at-module-level discipline for the algebra layer; see
tests/test_eval_density_zeroshot.py / tests/test_eval_z_identity.py, this
script's closest siblings). Covers:

  (a) `ridge_r2` algebra: perfect linear reconstruction -> R^2~1, unrelated
      features -> R^2 near/below 0, constant-reward -> R^2 undefined (None,
      not a fake 0/1), shape/empty refusals;
  (b) `decode_cell`: all THREE arms present (B, linear raw-coordinate floor,
      RBF/RFF-enriched control), diagnostics (mean/var/n_positive/n_pool)
      correct, shape-mismatch refusal (the exact bug class the mandatory
      control exists to catch -- see script module docstring "v1-E2
      lesson"); `rff_features` itself: shape/determinism, reward-independence
      (the control cannot have been fit to the family it scores), a
      numerical check against the true RBF kernel (Rahimi & Recht 2007's own
      correctness property), and arg refusals;
  (c) the two-arm control's OWN behavior is what the script's docstring
      claims: the LINEAR-raw-coordinate arm is high for a smooth reward over
      an ASYMMETRIC domain (a goal near the edge of the pool's coordinate
      spread) but near-zero for the SAME reward over a domain symmetric
      around the goal (no linear direction is preferred) -- calibrated
      against a real closed-form ridge computed independently in this test
      file, not asserted from thin air; the RBF-ENRICHED arm decodes a sharp
      ball indicator over an ISOTROPIC pool that the linear arm structurally
      cannot -- the concrete gap the coordinator's fix closes;
  (d) reuse-not-reimplement pins: `check_density_consistency`,
      `DensityMismatchError`, `build_pool`, `ridge_fit_wr`,
      `reward_conj_dirac`, `reward_conj_dense`, `draw_region_centers_with_obs`
      are all the SAME objects the reused modules define (mirrors
      test_eval_z_identity.py's `test_check_density_consistency_reused_not_
      reimplemented`);
  (e) import / --help / parser-default smoke (mirrors test_eval_z_identity.py).
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "rescuing_fb" / "probe_b_reward_decode.py"


@pytest.fixture(scope="module")
def probe():
    spec = importlib.util.spec_from_file_location("probe_b_reward_decode", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # numpy-only at module level (jax is lazy)
    return mod


# ============================ (a) ridge_r2 algebra ==================================

def test_ridge_r2_perfect_linear_reconstruction(probe):
    """r EXACTLY a bias-augmented linear function of features -> R^2 ~ 1.0
    at a tiny ridge (near-zero regularization bias)."""
    rng = np.random.default_rng(0)
    P, k = 500, 6
    features = rng.normal(size=(P, k))
    v_true = rng.normal(size=k)
    bias_true = 0.7
    r = features @ v_true + bias_true
    out = probe.ridge_r2(features, r, ridge=1e-8)
    assert out["r2"] == pytest.approx(1.0, abs=1e-4)
    assert out["n"] == P
    assert out["n_features"] == k + 1     # bias column appended


def test_ridge_r2_unrelated_features_near_zero(probe):
    """r independent of features -> R^2 near 0 (well below the perfect case),
    not a large spurious positive value."""
    rng = np.random.default_rng(1)
    P, k = 2000, 6
    features = rng.normal(size=(P, k))
    r = rng.normal(size=P)     # unrelated to features
    out = probe.ridge_r2(features, r, ridge=1e-4)
    assert out["r2"] is not None
    assert out["r2"] < 0.05


def test_ridge_r2_no_bias_option(probe):
    """add_bias=False fits exactly `n_features == k` (no appended column) --
    pins the bias-column convention is opt-in-by-flag, not silently forced."""
    rng = np.random.default_rng(2)
    features = rng.normal(size=(100, 4))
    r = rng.normal(size=100)
    out = probe.ridge_r2(features, r, ridge=1e-4, add_bias=False)
    assert out["n_features"] == 4
    assert out["add_bias"] is False


def test_ridge_r2_constant_reward_is_undefined_not_zero(probe):
    """SS_tot == 0 (constant reward, e.g. a sparse indicator that hit
    NOTHING in the pool) must report r2=None -- NOT silently 0.0 or 1.0,
    which would misread a degenerate reward as either total failure or a
    perfect fit."""
    rng = np.random.default_rng(3)
    features = rng.normal(size=(50, 3))
    r = np.zeros(50)
    out = probe.ridge_r2(features, r, ridge=1e-4)
    assert out["r2"] is None
    assert out["ss_tot"] == pytest.approx(0.0)


def test_ridge_r2_rejects_shape_mismatch(probe):
    with pytest.raises(AssertionError):
        probe.ridge_r2(np.zeros((5, 3)), np.zeros(4), ridge=1e-4)


def test_ridge_r2_rejects_empty(probe):
    with pytest.raises(AssertionError):
        probe.ridge_r2(np.zeros((0, 3)), np.zeros(0), ridge=1e-4)


# ============================ (b) decode_cell ========================================

def test_decode_cell_reports_all_three_arms_and_diagnostics(probe):
    rng = np.random.default_rng(4)
    P, d, d_c, d_rbf = 300, 10, 2, 8
    b_pool = rng.normal(size=(P, d))
    u_raw = rng.normal(size=(P, d_c))
    u_rbf = rng.normal(size=(P, d_rbf))
    r_pool = (rng.normal(size=P) > 1.0).astype(np.float64)     # sparse-like, some 1s
    cell = probe.decode_cell(b_pool, u_raw, u_rbf, r_pool, ridge=1e-4)
    assert set(cell) >= {"b_decode", "raw_coord_control", "raw_coord_rbf_control",
                        "reward_mean", "reward_var", "n_positive", "n_pool"}
    assert cell["n_pool"] == P
    assert cell["n_positive"] == int((r_pool > 0.5).sum())
    assert cell["reward_mean"] == pytest.approx(r_pool.mean())
    assert all("r2" in cell[arm] for arm in
              ("b_decode", "raw_coord_control", "raw_coord_rbf_control"))


def test_decode_cell_rejects_pool_size_mismatch(probe):
    """The exact bug class a mandatory control exists to catch (script
    docstring 'v1-E2 lesson'): B pool and raw-coord pool silently drawn at
    different sizes/orders would score garbage -- must refuse instead."""
    rng = np.random.default_rng(5)
    b_pool = rng.normal(size=(300, 10))
    u_raw = rng.normal(size=(299, 2))          # off-by-one
    u_rbf = rng.normal(size=(300, 8))
    r_pool = rng.normal(size=300)
    with pytest.raises(AssertionError):
        probe.decode_cell(b_pool, u_raw, u_rbf, r_pool, ridge=1e-4)


# ============================ (b2) rff_features ======================================

def test_rff_features_shape_and_determinism(probe):
    rng = np.random.default_rng(6)
    x = rng.normal(size=(50, 3))
    f1 = probe.rff_features(x, dim=20, bandwidth=1.0, seed=0)
    f2 = probe.rff_features(x, dim=20, bandwidth=1.0, seed=0)
    assert f1.shape == (50, 20)
    np.testing.assert_array_equal(f1, f2)          # same seed -> identical bank


def test_rff_features_different_seed_differs(probe):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(50, 3))
    f1 = probe.rff_features(x, dim=20, bandwidth=1.0, seed=0)
    f2 = probe.rff_features(x, dim=20, bandwidth=1.0, seed=1)
    assert not np.allclose(f1, f2)


def test_rff_features_reward_independent(probe):
    """The feature map depends only on (x, dim, bandwidth, seed) -- NEVER on
    a reward vector (an honest control cannot have been fit to the family it
    will be scored against). Calling it twice with the SAME x/seed but used
    to score two DIFFERENT reward vectors downstream must yield the exact
    same feature bank -- this is what makes the B-vs-control comparison in
    `decode_cell` apples-to-apples across families."""
    rng = np.random.default_rng(8)
    x = rng.normal(size=(30, 2))
    f_before_r1 = probe.rff_features(x, dim=10, bandwidth=1.0, seed=0)
    _ = rng.normal(size=30)     # simulate "using" one reward in between
    f_before_r2 = probe.rff_features(x, dim=10, bandwidth=1.0, seed=0)
    np.testing.assert_array_equal(f_before_r1, f_before_r2)


def test_rff_features_approximates_rbf_kernel(probe):
    """Standard RFF correctness property (Rahimi & Recht 2007): for large
    `dim`, phi(x).phi(y) approximates exp(-||x-y||^2 / (2*bandwidth^2)).
    A wrong formula (bad scaling, wrong distribution for W/b) would fail
    this numerically even though shapes/determinism above would still pass."""
    rng = np.random.default_rng(9)
    bandwidth = 1.5
    x = rng.normal(scale=1.0, size=(40, 4))
    phi = probe.rff_features(x, dim=8000, bandwidth=bandwidth, seed=0)
    approx_kernel = phi @ phi.T
    d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    true_kernel = np.exp(-d2 / (2.0 * bandwidth ** 2))
    assert np.max(np.abs(approx_kernel - true_kernel)) < 0.1


def test_rff_features_rejects_bad_args(probe):
    x = np.zeros((10, 3))
    with pytest.raises(AssertionError):
        probe.rff_features(x, dim=0, bandwidth=1.0, seed=0)
    with pytest.raises(AssertionError):
        probe.rff_features(x, dim=10, bandwidth=0.0, seed=0)
    with pytest.raises(AssertionError):
        probe.rff_features(np.zeros(10), dim=10, bandwidth=1.0, seed=0)   # not 2D


# ============================ (b3) rff_features_multiscale ==========================

def test_rff_multiscale_dim_matches_exactly_via_remainder_split(probe):
    """dim not evenly divisible by len(bandwidths) -- the remainder must be
    absorbed (first bands get +1) so the TOTAL feature count is EXACTLY
    `dim`, preserving the 'sized comparably to B' fairness contract even
    when B's own dimension isn't a multiple of the bandwidth count."""
    rng = np.random.default_rng(11)
    x = rng.normal(size=(20, 3))
    out = probe.rff_features_multiscale(x, dim=50, bandwidths=[0.5, 1, 2, 4], seed=0)
    assert out.shape == (20, 50)


def test_rff_multiscale_equals_concat_of_single_scale_calls(probe):
    """Not a black box: the multiscale bank is EXACTLY the concatenation of
    per-bandwidth `rff_features` calls at the documented seed offset
    (seed*1009 + band_index) -- pins the composition, not just the shape."""
    rng = np.random.default_rng(12)
    x = rng.normal(size=(15, 2))
    bandwidths = [1.0, 2.0]
    out = probe.rff_features_multiscale(x, dim=10, bandwidths=bandwidths, seed=3)
    b0 = probe.rff_features(x, dim=5, bandwidth=1.0, seed=3 * 1009 + 0)
    b1 = probe.rff_features(x, dim=5, bandwidth=2.0, seed=3 * 1009 + 1)
    np.testing.assert_array_equal(out, np.concatenate([b0, b1], axis=-1))


def test_rff_multiscale_rejects_too_few_dims_for_bandwidth_count(probe):
    x = np.zeros((10, 3))
    with pytest.raises(AssertionError):
        probe.rff_features_multiscale(x, dim=2, bandwidths=[0.5, 1, 2, 4], seed=0)
    with pytest.raises(AssertionError):
        probe.rff_features_multiscale(x, dim=10, bandwidths=[], seed=0)


def test_rff_control_decodes_a_sharp_indicator_the_linear_control_cannot(probe):
    """The concrete case the coordinator flagged: `reward_conj_dirac` (a hard
    ball indicator) over an ISOTROPIC pool is essentially unreadable by a
    linear-plus-bias fit of raw coordinates (no direction is preferred), but
    IS readable by a nonlinear RBF/RFF feature bank whose bandwidth matches
    the indicator's radius -- this is the concrete gap the enriched control
    exists to close, demonstrated end-to-end (reward builder -> rff_features
    -> ridge_r2), not merely asserted in the module docstring."""
    from env_geometry import HitSpec
    spec = HitSpec.single(0, 2, 0.4)
    rng = np.random.default_rng(10)
    coords = rng.uniform(-2, 2, size=(6000, 2))     # symmetric/isotropic around 0
    center = np.array([0.0, 0.0])
    r = probe.reward_conj_dirac(coords, center, spec)
    assert 0 < r.sum() < len(r)                      # some hits, not degenerate

    linear_r2 = probe.ridge_r2(coords, r, ridge=1e-4)["r2"]
    rbf_feats = probe.rff_features(coords / 0.4, dim=64, bandwidth=1.0, seed=0)
    rbf_r2 = probe.ridge_r2(rbf_feats, r, ridge=1e-4)["r2"]

    assert linear_r2 is not None and rbf_r2 is not None
    assert linear_r2 < 0.05
    assert rbf_r2 > linear_r2 + 0.2


# ============ (c) the raw-coordinate control's own behavior (calibrated) ============

def test_raw_coord_control_high_for_asymmetric_domain_low_for_symmetric(probe):
    """`reward_conj_dense` (reused, not reimplemented) over an ASYMMETRIC
    domain (goal near a corner of the pool's coordinate spread) is
    substantially raw-coordinate-decodable; the SAME reward shape over a
    domain SYMMETRIC around the goal is not (no single linear direction is
    preferred by a radially symmetric target). Calibrated against an
    independently hand-written closed-form ridge (not just re-calling
    `ridge_r2` on itself) so this test cannot pass merely because
    `ridge_r2` and the script agree with themselves."""
    from env_geometry import HitSpec
    spec = HitSpec.single(0, 2, 0.4)
    rng = np.random.default_rng(6)

    def hand_ridge_r2(x, r, ridge):
        x = np.concatenate([x, np.ones((len(x), 1))], axis=-1)
        a = x.T @ x + ridge * np.eye(x.shape[1])
        v = np.linalg.solve(a, x.T @ r)
        r_hat = x @ v
        return 1.0 - np.sum((r - r_hat) ** 2) / np.sum((r - r.mean()) ** 2)

    coords_corner = rng.uniform(0, 4, size=(4000, 2))
    r_corner = probe.reward_conj_dense(coords_corner, np.array([0.0, 0.0]), spec,
                                       sigma_scaled=2.0)
    r2_corner_hand = hand_ridge_r2(coords_corner, r_corner, 1e-4)
    r2_corner_probe = probe.ridge_r2(coords_corner, r_corner, ridge=1e-4)["r2"]

    coords_center = rng.uniform(-2, 2, size=(4000, 2))
    r_center = probe.reward_conj_dense(coords_center, np.array([0.0, 0.0]), spec,
                                       sigma_scaled=2.0)
    r2_center_hand = hand_ridge_r2(coords_center, r_center, 1e-4)
    r2_center_probe = probe.ridge_r2(coords_center, r_center, ridge=1e-4)["r2"]

    assert r2_corner_probe == pytest.approx(r2_corner_hand, abs=1e-8)
    assert r2_center_probe == pytest.approx(r2_center_hand, abs=1e-8)
    assert r2_corner_probe > 0.3        # "known to be high" for the asymmetric case
    assert r2_center_probe < 0.05       # near-zero when no direction is preferred
    assert r2_corner_probe > r2_center_probe + 0.2


# ============================ (d) reuse-not-reimplement pins =========================

def test_check_density_consistency_reused_not_reimplemented(probe):
    with pytest.raises(probe.DensityMismatchError):
        probe.check_density_consistency(False, True, force=False, run_dir="<run>")


def test_ridge_fit_wr_build_pool_are_the_reused_objects(probe):
    """Import the SAME module normally (regular `import`, not a second
    importlib.util file-location load -- that would create a DIFFERENT
    module object with a different identity) and pin object identity: the
    probe script must not have copy-pasted its own `ridge_fit_wr`/`build_pool`."""
    import eval_density_zeroshot as edz     # available: probe's own module-level
                                            # sys.path.insert already ran when
                                            # the `probe` fixture executed
    assert probe.ridge_fit_wr is edz.ridge_fit_wr
    assert probe.build_pool is edz.build_pool
    assert probe.check_density_consistency is edz.check_density_consistency
    assert probe.DensityMismatchError is edz.DensityMismatchError


def test_reward_family_and_region_helpers_are_reused(probe):
    import eval_zeroshot_rewards as ezr
    import eval_partition_zeroshot as epz
    assert probe.reward_conj_dirac is ezr.reward_conj_dirac
    assert probe.reward_conj_dense is ezr.reward_conj_dense
    assert probe.eval_tasks_for is ezr.eval_tasks_for
    assert probe.print_table is ezr.print_table
    assert probe.draw_region_centers_with_obs is epz.draw_region_centers_with_obs


def test_families_tuple_is_sparse_dense_region(probe):
    assert probe.FAMILIES == ("sparse", "dense", "region")


def test_unknown_family_would_be_rejected(probe):
    """Mirrors main()'s own validation logic (not calling main() itself --
    that needs jax/ogbench/a live checkpoint, out of scope for this pure
    module -- see test_eval_density_zeroshot.py's identical scoping note)."""
    families = [f.strip() for f in "sparse,bogus".split(",") if f.strip()]
    unknown = [f for f in families if f not in probe.FAMILIES]
    assert unknown == ["bogus"]


# ============================ (e) import / --help / parser defaults =================

def test_script_help_runs():
    proc = subprocess.run([sys.executable, str(SCRIPT), "--help"],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    for flag in ("--run-dir", "--checkpoint-step", "--tasks", "--families",
                 "--regions", "--region-min-pos", "--relabel-size",
                 "--pool-size", "--ridge", "--rbf-dim", "--rbf-bandwidths",
                 "--seed", "--force", "--out"):
        assert flag in proc.stdout, f"{flag} missing from --help"


def test_parser_contract_defaults(probe):
    ap = probe.build_parser()
    ns = ap.parse_args(["--run-dir", "/tmp/x"])
    assert ns.checkpoint_step == 500000
    assert ns.tasks == "1,2,3,4,5"
    assert ns.families == "sparse,dense,region"
    assert ns.regions == 4
    assert ns.region_min_pos == 5
    assert ns.relabel_size == 50000
    assert ns.pool_size == 16384
    assert ns.ridge == pytest.approx(1e-4)
    assert ns.rbf_dim == 0                  # 0 = auto = match B's dimension
    assert ns.rbf_bandwidths == "0.5,1,2,4"
    assert ns.seed == 0
    assert ns.force is False
