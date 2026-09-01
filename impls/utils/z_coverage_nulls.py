"""Pure-numpy statistics for the reward-class-coverage analysis
(docs/notes/2026-08-31-z-coverage-brainstorm.md, docs/notes/2026-08-31-z-coverage-walker-
results.md). No jax, no critic imports: this module is the ANALYSIS side (angular
statistics, the k-hot analytic null, the permutation noise floor) that runs on whatever
vectors a critic's own `infer_cond` / raw-encoder call already produced. It never computes
a z from a reward itself -- that stays inside `impls/critics/{rldp,hilp}.py`.

Vocabulary note: everything here is plain linear algebra on vectors on a sphere (means,
covariances, cosines, percentiles). No measure-theoretic framing.
"""
import numpy as np
from scipy import special


# ---------------------------------------------------------------------------------------
# Sphere / angle utilities
# ---------------------------------------------------------------------------------------

def project_sphere(z, radius=None):
    """sqrt(d)-normalize (or `radius`-normalize) rows of `z`. Same convention as
    `impls/utils/fb_math.py::project_z(z, norm_z=True)`, reimplemented in plain numpy so
    this module has no jax dependency. Only ever applied to vectors already produced by a
    critic's real `infer_cond`/encoder call -- this is normalization bookkeeping, not a
    reimplementation of the inference math."""
    z = np.asarray(z, dtype=np.float64)
    d = z.shape[-1]
    r = np.sqrt(d) if radius is None else radius
    norm = np.linalg.norm(z, axis=-1, keepdims=True)
    return r * z / (norm + 1e-12)


def cos_to(z, v):
    """Signed cosine similarity of every row of `z` [N, d] (or a single vector [d]) to a
    fixed direction `v` [d]. Antipodes are NOT folded (z and -z are different tasks for a
    sign-definite moment encoder -- brainstorm doc pitfall 2). Returns a python float when
    `z` is a single 1-D vector, otherwise an array of shape [N]."""
    z_in = np.asarray(z, dtype=np.float64)
    squeeze = z_in.ndim == 1
    z = np.atleast_2d(z_in)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    zn = np.linalg.norm(z, axis=-1)
    vn = np.linalg.norm(v)
    out = (z @ v) / (zn * vn + 1e-12)
    out = np.clip(out, -1.0, 1.0)
    return float(out[0]) if squeeze else out


def pairwise_cos(z, y=None):
    """Full signed pairwise-cosine matrix between rows of `z` [N,d] and rows of `y` [M,d]
    (or `z` against itself if `y` is None, in which case the diagonal is 1 and is left in
    -- callers that want within-class-excluding-self drop it themselves)."""
    z = np.atleast_2d(np.asarray(z, dtype=np.float64))
    y = z if y is None else np.atleast_2d(np.asarray(y, dtype=np.float64))
    zn = np.linalg.norm(z, axis=-1, keepdims=True)
    yn = np.linalg.norm(y, axis=-1, keepdims=True)
    num = z @ y.T
    den = zn @ yn.T + 1e-12
    return np.clip(num / den, -1.0, 1.0)


def theta90(z, center=None):
    """Angular radius (degrees) of the smallest cap around `center` (default: the class's
    own mean direction) containing 90% of the rows of `z`. This is M1's theta_90: the
    10th-percentile cosine to the center, turned into an angle."""
    z = np.atleast_2d(np.asarray(z, dtype=np.float64))
    if center is None:
        m = z.mean(axis=0)
        nm = np.linalg.norm(m)
        if nm < 1e-12:
            return 180.0
        center = m / nm
    cos = cos_to(z, center)
    cutoff = np.percentile(cos, 10.0)          # 90% of mass has cos >= this
    cutoff = np.clip(cutoff, -1.0, 1.0)
    return float(np.degrees(np.arccos(cutoff)))


def resultant_length(z):
    """R_bar = ||mean(z)|| / sqrt(mean ||z_i||^2) -- 1 for a perfect point mass, ~0 for an
    isotropic cloud. Normalized by the typical row norm so it is comparable whether `z`
    rows sit on a sqrt(d) sphere, a unit sphere, or are un-normalized raw vectors."""
    z = np.atleast_2d(np.asarray(z, dtype=np.float64))
    mean_norm = np.linalg.norm(z.mean(axis=0))
    typical = np.sqrt(np.mean(np.sum(z ** 2, axis=-1))) + 1e-12
    return float(mean_norm / typical)


# ---------------------------------------------------------------------------------------
# M1's analytic uniform-sphere null: cos(u, v) for u uniform on S^{d-1}, v fixed.
# cos^2 ~ Beta(1/2, (d-1)/2); cos is symmetric about 0.
# ---------------------------------------------------------------------------------------

def uniform_cos_cdf(x, d):
    """P(cos <= x) for a uniformly-random direction on S^{d-1} dotted with a fixed unit
    vector, x in [-1, 1]. Derived from cos^2 ~ Beta(1/2, (d-1)/2) and the symmetry of cos:
    P(cos <= x) = 1/2 + 1/2 * sign(x) * I_{x^2}(1/2, (d-1)/2)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.clip(x, -1.0, 1.0) ** 2
    ib = special.betainc(0.5, (d - 1) / 2.0, y)     # regularized incomplete beta I_y(a,b)
    return 0.5 + 0.5 * np.sign(x) * ib


def sample_uniform_sphere(d, n, rng):
    """n iid draws uniform on S^{d-1} (unit norm)."""
    g = rng.normal(size=(n, d))
    return g / (np.linalg.norm(g, axis=-1, keepdims=True) + 1e-12)


# ---------------------------------------------------------------------------------------
# The k-hot analytic (CLT) null for an E[r B] moment encoder (RLDP; doc pitfall 6).
#
# Scattered k-hot: r_i = 1 for a uniformly-random size-k subset of the N inference rows,
# else 0. z_raw = (1/N) * sum_{i in S_k} B(row_i). If the N rows of B are treated as an
# exchangeable finite pool with (population) mean mu_B and covariance Sigma_B, a size-k
# subset drawn WITHOUT replacement has
#     E[sum] = k * mu_B
#     Cov[sum] = k * Sigma_B * (N - k) / (N - 1)        (finite-population correction)
# so z_raw = sum / N has mean (k/N) mu_B and covariance (k/N^2) Sigma_B * (N-k)/(N-1).
# For k << N the fpc term is ~1 and this reduces to the doc's CLT statement
# z_raw ~= (k/N) mu_B + (sqrt(k)/N) Sigma_B^{1/2} g.
# ---------------------------------------------------------------------------------------

def khot_null_params(mu_B, Sigma_B, k, N, finite_pop_correction=True):
    """-> (mean, cov) of the scattered-k-hot z_raw under the CLT/exchangeability
    approximation, exact (given the population mu_B/Sigma_B) up to the CLT itself."""
    mu_B = np.asarray(mu_B, dtype=np.float64)
    Sigma_B = np.asarray(Sigma_B, dtype=np.float64)
    mean = (k / N) * mu_B
    fpc = (N - k) / (N - 1) if (finite_pop_correction and N > 1) else 1.0
    cov = (k / N ** 2) * Sigma_B * fpc
    return mean, cov


def sample_khot_null(mu_B, Sigma_B, k, N, num_samples, rng, finite_pop_correction=True):
    """num_samples draws of z_raw from the analytic (Gaussian) CLT null."""
    mean, cov = khot_null_params(mu_B, Sigma_B, k, N,
                                 finite_pop_correction=finite_pop_correction)
    d = mean.shape[0]
    # eigh rather than cholesky: Sigma_B from a finite empirical sample can be PSD with
    # tiny negative eigenvalues from floating-point round-off.
    w, V = np.linalg.eigh(cov)
    w = np.clip(w, 0.0, None)
    A = V * np.sqrt(w)[None, :]
    g = rng.normal(size=(num_samples, d))
    return mean[None, :] + g @ A.T


def brute_force_khot_zraw(B_rows, k, num_samples, rng):
    """The REAL scattered-k-hot procedure, exactly: draw a uniformly-random size-k subset
    of the N rows (without replacement) `num_samples` times and average. This is what
    `khot_null_params` approximates analytically -- used both by the pinning test and, in
    the main script, to sanity-check the analytic null against the actual B-row pool."""
    B_rows = np.asarray(B_rows, dtype=np.float64)
    N = B_rows.shape[0]
    out = np.empty((num_samples, B_rows.shape[1]), dtype=np.float64)
    for i in range(num_samples):
        idx = rng.choice(N, size=k, replace=False)
        out[i] = B_rows[idx].sum(axis=0) / N
    return out


# ---------------------------------------------------------------------------------------
# Matched-null effective support size (doc pitfall 6): for a general reward vector, the
# "k" a scattered k-hot class would need to match its second-moment concentration.
# ---------------------------------------------------------------------------------------

def k_eff(rewards):
    """k_eff = (sum r_i)^2 / sum(r_i^2). For a 0/1 k-hot vector this is exactly k. For a
    general nonnegative-or-signed reward it is the standard "effective sample size" /
    inverse participation ratio of the weight vector."""
    r = np.asarray(rewards, dtype=np.float64).reshape(-1)
    num = float(np.sum(r) ** 2)
    den = float(np.sum(r ** 2))
    if den == 0.0:
        return 0.0
    return num / den


# ---------------------------------------------------------------------------------------
# Permutation noise floor (doc pitfall 3): shuffle the reward vector against the B-row
# pool -- destroying any real association between reward and dynamics -- and see how large
# a raw-norm that produces. A class whose true raw norm sits inside this floor is
# indistinguishable from noise; its downstream angular spread must be discounted.
# ---------------------------------------------------------------------------------------

def permutation_floor_norms(rewards, B_rows, num_shuffles, rng):
    """-> array [num_shuffles] of ||z_raw|| = ||(1/N) sum_i r_shuffled_i * B_row_i|| under
    independent shuffles of `rewards` against `B_rows` (both length N)."""
    r = np.asarray(rewards, dtype=np.float64).reshape(-1)
    B_rows = np.asarray(B_rows, dtype=np.float64)
    N = B_rows.shape[0]
    assert r.shape[0] == N, f"rewards ({r.shape[0]}) and B_rows ({N}) length mismatch"
    out = np.empty(num_shuffles, dtype=np.float64)
    for i in range(num_shuffles):
        rp = rng.permutation(r)
        z = (rp @ B_rows) / N
        out[i] = np.linalg.norm(z)
    return out


def permutation_floor_analytic(rewards, B_rows):
    """Cheap closed-form estimate of the same floor's scale, per the doc:
    ||z_raw|| ~= ||B||_F * std(r) / N (a per-component-budget approximation; the empirical
    permutation floor above is the number to actually gate on)."""
    r = np.asarray(rewards, dtype=np.float64).reshape(-1)
    B_rows = np.asarray(B_rows, dtype=np.float64)
    N = B_rows.shape[0]
    return float(np.linalg.norm(B_rows) * np.std(r) / N)
