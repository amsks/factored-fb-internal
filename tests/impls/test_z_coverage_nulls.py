"""Pinning tests for impls/utils/z_coverage_nulls.py -- the pure-numpy statistics behind
the reward-class-coverage analysis (docs/notes/2026-08-31-z-coverage-brainstorm.md).

Three things pinned, per the analysis brief:
  (a) the k-hot analytic (CLT) null vs a brute-force simulation, 1e-3 tolerance on the mean
  (b) the matched-null k_eff formula on a hand-computed case
  (c) the permutation noise floor's behavior on pure noise (should not flag it as unusual)
"""
import numpy as np

from impls.utils.z_coverage_nulls import (
    project_sphere, cos_to, pairwise_cos, theta90, resultant_length,
    uniform_cos_cdf, sample_uniform_sphere,
    khot_null_params, sample_khot_null, brute_force_khot_zraw,
    k_eff, permutation_floor_norms, permutation_floor_analytic,
)


# ---------------------------------------------------------------------------------------
# (a) k-hot analytic null vs brute-force simulation
# ---------------------------------------------------------------------------------------

def _synthetic_pool(rng, N=3000, d=6):
    """A finite pool of N rows in R^d with a nontrivial mean and covariance, standing in
    for a real B-row pool without needing a trained critic."""
    mu = rng.normal(size=d) * 0.5
    A = rng.normal(size=(d, d)) * 0.3
    cov = A @ A.T + 0.2 * np.eye(d)                 # PD, not axis-aligned
    L = np.linalg.cholesky(cov)
    rows = mu[None, :] + rng.normal(size=(N, d)) @ L.T
    return rows


def test_khot_analytic_null_matches_brute_force_mean():
    rng = np.random.RandomState(0)
    N, d = 3000, 6
    B_rows = _synthetic_pool(rng, N, d)
    mu_B = B_rows.mean(axis=0)
    Sigma_B = np.cov(B_rows.T, bias=True)           # population covariance (ddof=0)

    for k in (1, 20, 150):
        mean_a, cov_a = khot_null_params(mu_B, Sigma_B, k, N)

        samples = brute_force_khot_zraw(B_rows, k, num_samples=4000,
                                        rng=np.random.RandomState(100 + k))
        mean_bf = samples.mean(axis=0)
        cov_bf = np.cov(samples.T, bias=True)

        mean_err = np.linalg.norm(mean_a - mean_bf)
        assert mean_err < 1e-3, f"k={k}: mean mismatch {mean_err}"

        # Covariance has more Monte Carlo noise than the mean; a looser relative check on
        # total variance (trace) is the honest thing to pin here.
        tr_a, tr_bf = np.trace(cov_a), np.trace(cov_bf)
        rel = abs(tr_a - tr_bf) / (tr_a + 1e-12)
        assert rel < 0.15, f"k={k}: covariance trace mismatch rel={rel}"


def test_khot_null_sampler_matches_its_own_params():
    """sample_khot_null should reproduce the (mean, cov) khot_null_params predicts -- a
    sanity check on the sampler itself, independent of the brute-force comparison above."""
    rng = np.random.RandomState(1)
    N, d = 2000, 5
    B_rows = _synthetic_pool(rng, N, d)
    mu_B, Sigma_B = B_rows.mean(axis=0), np.cov(B_rows.T, bias=True)
    k = 40
    mean_a, cov_a = khot_null_params(mu_B, Sigma_B, k, N)
    samples = sample_khot_null(mu_B, Sigma_B, k, N, num_samples=20000,
                               rng=np.random.RandomState(2))
    np.testing.assert_allclose(samples.mean(axis=0), mean_a, atol=5e-4)
    np.testing.assert_allclose(np.cov(samples.T, bias=True), cov_a, atol=5e-4, rtol=0.1)


def test_khot_null_collapses_onto_mu_B_as_k_grows():
    """H2's 1/sqrt(k) collapse: the null's covariance trace (spread) shrinks with k while
    its mean magnitude grows toward mu_B -- i.e. coverage narrows as k increases."""
    rng = np.random.RandomState(3)
    N, d = 5000, 8
    B_rows = _synthetic_pool(rng, N, d)
    mu_B, Sigma_B = B_rows.mean(axis=0), np.cov(B_rows.T, bias=True)
    traces = []
    for k in (1, 10, 100, 1000):
        mean_a, cov_a = khot_null_params(mu_B, Sigma_B, k, N)
        # normalize the spread by the mean's own magnitude^2 so this compares ANGULAR
        # concentration, not raw scale (both grow with k).
        traces.append(np.trace(cov_a) / (np.linalg.norm(mean_a) ** 2 + 1e-12))
    assert traces[0] > traces[1] > traces[2] > traces[3]


# ---------------------------------------------------------------------------------------
# (b) matched-null k_eff on hand-computed cases
# ---------------------------------------------------------------------------------------

def test_k_eff_zero_one_khot_equals_k():
    r = np.array([1, 1, 1, 0, 0, 0, 0], dtype=np.float64)
    assert np.isclose(k_eff(r), 3.0)


def test_k_eff_weighted_case():
    r = np.array([2.0, 2.0])
    # sum=4, sumsq=8 -> k_eff = 16/8 = 2
    assert np.isclose(k_eff(r), 2.0)


def test_k_eff_single_hot_is_one():
    r = np.zeros(50)
    r[17] = 5.0
    assert np.isclose(k_eff(r), 1.0)


def test_k_eff_all_zero_is_zero():
    assert k_eff(np.zeros(10)) == 0.0


# ---------------------------------------------------------------------------------------
# (c) permutation noise floor behavior
# ---------------------------------------------------------------------------------------

def test_permutation_floor_flags_pure_noise_reward():
    """A reward with NO real association to the B-row pool (white noise, i.e. C2c) must
    land inside the floor its own permutations produce -- the floor should not flag it as
    unusual, because it IS the null case."""
    rng = np.random.RandomState(4)
    N, d = 4000, 10
    B_rows = _synthetic_pool(rng, N, d)
    r_noise = rng.uniform(-1.0, 1.0, size=N)          # C2c: white noise reward

    z_raw = (r_noise @ B_rows) / N
    actual_norm = np.linalg.norm(z_raw)

    floor = permutation_floor_norms(r_noise, B_rows, num_shuffles=300,
                                    rng=np.random.RandomState(5))
    # The floor is built by shuffling the SAME reward against the SAME rows, so if the
    # reward carries no real signal beyond what a random pairing already produces, the
    # unshuffled statistic should sit squarely inside the shuffled distribution -- not in
    # an extreme tail.
    lo, hi = np.percentile(floor, [1.0, 99.0])
    assert lo <= actual_norm <= hi, (
        f"pure-noise reward's raw norm {actual_norm} fell outside its own permutation "
        f"floor [{lo}, {hi}] -- the floor failed to flag it as noise-level")


def test_permutation_floor_does_not_flag_real_signal_as_extreme_by_construction():
    """Sanity check the OTHER direction: a reward that is an exact linear readout of one
    B-row dimension (i.e. carries strong real association) should sit far outside its own
    permutation floor -- otherwise the floor would be trivially uninformative."""
    rng = np.random.RandomState(6)
    N, d = 4000, 10
    B_rows = _synthetic_pool(rng, N, d)
    r_signal = B_rows[:, 0] * 5.0                      # strongly associated with row 0

    z_raw = (r_signal @ B_rows) / N
    actual_norm = np.linalg.norm(z_raw)
    floor = permutation_floor_norms(r_signal, B_rows, num_shuffles=300,
                                    rng=np.random.RandomState(7))
    assert actual_norm > np.percentile(floor, 99.9) or actual_norm > floor.max()


def test_permutation_floor_analytic_same_order_of_magnitude_as_empirical():
    rng = np.random.RandomState(8)
    N, d = 3000, 6
    B_rows = _synthetic_pool(rng, N, d)
    r_noise = rng.uniform(-1.0, 1.0, size=N)
    analytic = permutation_floor_analytic(r_noise, B_rows)
    empirical = permutation_floor_norms(r_noise, B_rows, num_shuffles=200,
                                        rng=np.random.RandomState(9))
    med = np.median(empirical)
    # "analytically ~= ||B||_F sigma_r / N" (doc pitfall 3) -- an order-of-magnitude
    # estimate, not an exact match; pin it within a factor of 3.
    assert med / 3.0 <= analytic <= med * 3.0


# ---------------------------------------------------------------------------------------
# Sphere / angle utilities: light coverage so a regression here is caught too.
# ---------------------------------------------------------------------------------------

def test_project_sphere_norm():
    z = np.array([[3.0, 4.0]])
    out = project_sphere(z)
    np.testing.assert_allclose(np.linalg.norm(out, axis=-1), [np.sqrt(2)], atol=1e-9)


def test_cos_to_self_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert np.isclose(cos_to(v, v), 1.0)


def test_pairwise_cos_diag_is_one():
    rng = np.random.RandomState(0)
    z = rng.normal(size=(5, 4))
    C = pairwise_cos(z)
    np.testing.assert_allclose(np.diag(C), np.ones(5), atol=1e-9)


def test_theta90_tight_for_concentrated_cloud():
    rng = np.random.RandomState(0)
    d = 20
    center = np.zeros(d); center[0] = 1.0
    z = center[None, :] + rng.normal(size=(2000, d)) * 0.01
    t = theta90(z)
    assert t < 10.0   # a tightly concentrated cloud has a small cap angle


def test_theta90_wide_for_uniform_sphere():
    rng = np.random.RandomState(0)
    z = sample_uniform_sphere(20, 4000, rng)
    t = theta90(z)
    assert t > 60.0    # a uniform cloud on a high-d sphere has a wide 90% cap


def test_resultant_length_bounds():
    rng = np.random.RandomState(0)
    z_uniform = sample_uniform_sphere(30, 3000, rng)
    z_point = np.tile(np.array([1.0] + [0.0] * 29), (3000, 1))
    assert resultant_length(z_point) > 0.99
    assert resultant_length(z_uniform) < 0.15


def test_uniform_cos_cdf_matches_empirical_beta():
    rng = np.random.RandomState(0)
    d = 50
    z = sample_uniform_sphere(d, 200000, rng)
    v = np.zeros(d); v[0] = 1.0
    cos = cos_to(z, v)
    for x in (-0.5, -0.1, 0.0, 0.1, 0.5):
        empirical = np.mean(cos <= x)
        analytic = uniform_cos_cdf(x, d)
        assert abs(empirical - analytic) < 0.01, (x, empirical, analytic)


def test_uniform_cos_cdf_symmetric_and_bounded():
    d = 100
    assert np.isclose(uniform_cos_cdf(0.0, d), 0.5)
    assert np.isclose(uniform_cos_cdf(-1.0, d), 0.0, atol=1e-6)
    assert np.isclose(uniform_cos_cdf(1.0, d), 1.0, atol=1e-6)
