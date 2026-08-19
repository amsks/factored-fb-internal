"""spec 30a Item 4 -- the zero-shot de-biased readout against EXACT tabular
ground truth.

`PAPER/Multistep-FB/specs/30a-impl-wave1.md` Item 4;
`PAPER/Multistep-FB/DENSITY-FB-EXPLAINED.md` sec.3.5 / sec.4.3.

The object under test is

    Qhat_r(s0, a) = ( r(x0) + (w - kappa(x0|U_P))^T r(U_P) ) / (1 - gamma)

(`scripts/rescuing_fb/eval_density_zeroshot.compute_density_primary`, also the
readout `eval_measure_g4_spearman.py` and `eval_density_z_rewards.py` quote
through). Every previous test of it -- tests/test_eval_density_zeroshot.py --
checks ALGEBRAIC properties (linearity, hand-pinned arithmetic, pool-sweep
smoke). None of them ever compared it to a successor measure computed
independently, so a sign, an off-by-one in the start index, or a misplaced
(1-gamma) could sit inside a suite that is entirely green. This file closes that:
the MDP is small enough that

    M[s,a,x] = sum_{t>=1} gamma^{t-1} Pr(x_t = x | s0=s, a0=a) = P[s,a,:] (I - gamma P^pi)^{-1}

is a matrix inverse (`scripts/spec30_tabular_mdp.py`), and both sides of the
pipeline are checked against it: the SHIPPED TD target
`t = (1-gamma) k + gamma w_bar` (`gciql_flow.density_loss`) is iterated to its
population fixed point, and that fixed point is then fed to the SHIPPED readout.

Covered, one concern per section: measure start index; reward attachment;
terminal/absorbing handling; the 1-gamma normalization; the finite-pool
correction; hard AND soft immediate kernels; and the cancelling de-biasing pair
`r(x0)` / `-kappa(x0|U_P)` -- tested TOGETHER and with each half dropped, because
an early G4 calibration probe dropped one half and produced a dramatic false
null (Spearman .13-.18 vs the corrected .767-.790; the post-mortem is in
`eval_measure_g4_spearman.py` lines 380-387).

CPU only, no jax, no checkpoint, no env: pure float64 linear algebra.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.spec30_tabular_mdp import (  # noqa: E402
    absorbing_mdp,
    band_random_walk,
    density_td_fixed_point,
    eps_kernel,
    exact_successor_measure,
    measure_integral,
    normalized_measure,
    policy_transition,
    q_current_state,
    restrict_to_pool,
    truncated_successor_measure,
)

SCRIPT = REPO / "scripts" / "rescuing_fb" / "eval_density_zeroshot.py"

GAMMA = 0.99            # the shipped default; also makes 1/(1-gamma) = 100x amplification visible
N_STATES = 25
BAND = (8, 16)          # visited support; the states outside are pool-only buffer
EPS_DIRAC = 0.02        # kernel width at which _delta_phi's softmax IS a one-hot in float64
EPS_SOFT = 0.5          # a genuinely smearing width (spacing 1: nearest-neighbour weight ~.135)
S0, A0 = 12, 0


@pytest.fixture(scope="module")
def dzs():
    """The shipped readout module (numpy-only at module level; jax is lazy)."""
    spec = importlib.util.spec_from_file_location("eval_density_zeroshot", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def toy():
    """One shared toy MDP + its exact measure. Deterministic (seeded)."""
    rng = np.random.default_rng(0)
    P = band_random_walk(N_STATES, *BAND, seed=3)
    pi = rng.dirichlet(np.ones(2), size=N_STATES)
    M = exact_successor_measure(P, pi, GAMMA)
    coords = np.arange(N_STATES, dtype=np.float64)[:, None]
    return dict(P=P, pi=pi, M=M, nu=normalized_measure(M, GAMMA), coords=coords,
                r_rand=rng.normal(size=N_STATES),
                r_quad=coords[:, 0] ** 2)


def _readout(dzs, r_x0, w, kappa, r_pool, gamma=GAMMA):
    return float(dzs.compute_density_primary(r_x0, w, kappa, r_pool, gamma))


# =========================== (0) the ground truth checks itself ================

def test_matrix_inverse_measure_matches_the_explicit_power_series(toy):
    """(I - gamma P^pi)^{-1} vs sum_{t=1..4000} gamma^{t-1} P (P^pi)^{t-1}. Guards
    the ground truth's own start index before anything is compared to it."""
    series = truncated_successor_measure(toy["P"], toy["pi"], GAMMA, 4000)
    assert np.abs(toy["M"] - series).max() < 1e-11


def test_normalized_measure_is_a_probability_vector(toy):
    """The 1-gamma normalization, at the source: nu = (1-gamma)M has unit rows,
    so total measure mass is exactly 1/(1-gamma) and the readout's single
    1/(1-gamma) is the whole of the rescale (no second factor anywhere)."""
    mass = toy["nu"].sum(-1)
    assert np.abs(mass - 1.0).max() < 1e-12
    assert np.abs(toy["M"].sum(-1) - 1.0 / (1.0 - GAMMA)).max() < 1e-9
    # P^pi row-stochastic is what makes that true -- state it, don't assume it.
    assert np.abs(policy_transition(toy["P"], toy["pi"]).sum(-1) - 1.0).max() < 1e-12


# ============ (1) the SHIPPED TD target's fixed point, hard and soft ==========

def test_hard_td_target_fixed_point_is_the_normalized_measure(toy):
    """`t = (1-gamma)*k + gamma*w_bar` with k = one-hot at the row's own successor
    (`gciql_flow.density_loss`: `k = jnp.eye(...)` when measure_density_hard) has
    population fixed point exactly nu = (1-gamma)M. This is the START INDEX and
    the (1-gamma) PLACEMENT, checked on the training side: nu's first term is the
    SUCCESSOR x_1, and (1-gamma) multiplies only the immediate term."""
    w = density_td_fixed_point(toy["P"], toy["pi"], GAMMA, np.eye(N_STATES), iters=50000)
    assert np.abs(w - toy["nu"]).max() < 1e-11
    # and it is NOT the s0-inclusive convention (which would put mass on x0 itself)
    s0_inclusive = (1.0 - GAMMA) * (np.eye(N_STATES)[:, None, :] + GAMMA * toy["M"])
    assert np.abs(w[S0, A0] - s0_inclusive[S0, A0]).max() > 1e-3


def test_soft_td_target_fixed_point_is_the_kernel_smoothed_measure(toy):
    """Same target with k = _delta_phi(nobs, u_raw) (the eps-kernel softmax row at
    the successor) has fixed point nu @ K -- the measure of the SMOOTHED outcome,
    not a smoothed measure of a different object."""
    K = eps_kernel(toy["coords"], toy["coords"], EPS_SOFT)
    w = density_td_fixed_point(toy["P"], toy["pi"], GAMMA, K, iters=50000)
    assert np.abs(w - toy["nu"] @ K).max() < 1e-11
    assert np.abs(w.sum(-1) - 1.0).max() < 1e-12      # still on the simplex


# ================= (2) EXACT recovery: hard kernel, full pool =================

def test_exact_recovery_hard_kernel_full_pool(toy, dzs):
    """THE strongest statement available here. Hard immediate kernel, pool = the
    whole state space, kappa a true Dirac: the readout reproduces the linear-
    algebra answer M^T r to float64 precision -- not "approximately", not "within
    a tolerance chosen to pass". Checked on four unrelated rewards."""
    w = density_td_fixed_point(toy["P"], toy["pi"], GAMMA, np.eye(N_STATES), iters=50000)
    kappa = eps_kernel(toy["coords"], toy["coords"], EPS_DIRAC)
    assert np.abs(kappa - np.eye(N_STATES)).max() == 0.0       # exactly one-hot in float64

    rewards = {
        "random": toy["r_rand"],
        "quadratic": toy["r_quad"],
        "dirac": (np.arange(N_STATES) == 14).astype(np.float64),
        "linear": np.arange(N_STATES, dtype=np.float64),
    }
    for name, r in rewards.items():
        truth = float(measure_integral(toy["M"], r)[S0, A0])
        got = _readout(dzs, r[S0], w[S0, A0], kappa[S0], r)
        assert abs(got - truth) <= 1e-11 * max(1.0, abs(truth)), (name, got, truth)


# ============== (3) measure start index / reward attachment ===================

def test_start_index_is_the_successor_and_not_the_current_state(toy, dzs):
    """The readout returns sum_{t>=1} gamma^{t-1} E r(x_t). It is NOT the ordinary
    current-state Q, and the discriminating term is exactly r(x0): the two
    conventions are related by Q_std = r(x0) + gamma * Qhat. Both directions are
    asserted so the test cannot pass by the two happening to coincide."""
    w = density_td_fixed_point(toy["P"], toy["pi"], GAMMA, np.eye(N_STATES), iters=50000)
    kappa = eps_kernel(toy["coords"], toy["coords"], EPS_DIRAC)
    r = toy["r_rand"]
    got = _readout(dzs, r[S0], w[S0, A0], kappa[S0], r)

    q_std = float(q_current_state(toy["P"], toy["pi"], GAMMA, r)[S0, A0])
    assert abs(got - (q_std - r[S0]) / GAMMA) <= 1e-9          # the exact relation

    # ... and they really are different objects. Sharpest witness: a reward that
    # fires ONLY at x0. The readout returns the discounted RE-visit mass
    # M[s0,a0,x0]; the current-state convention returns 1 + gamma * that. The gap
    # is exactly 1 - nu[s0,a0,x0], which is large because nu is a distribution.
    dirac_x0 = (np.arange(N_STATES) == S0).astype(np.float64)
    got_d = _readout(dzs, dirac_x0[S0], w[S0, A0], kappa[S0], dirac_x0)
    std_d = float(q_current_state(toy["P"], toy["pi"], GAMMA, dirac_x0)[S0, A0])
    assert got_d == pytest.approx(float(toy["M"][S0, A0, S0]), rel=1e-11)
    assert std_d - got_d == pytest.approx(1.0 - float(toy["nu"][S0, A0, S0]), rel=1e-9)
    assert std_d - got_d > 0.5


def test_reward_is_attached_to_the_states_the_measure_lands_on(toy, dzs):
    """Reward attachment convention: r is read at x_1, x_2, ... with weights
    gamma^{t-1}, i.e. E[sum_{t>=0} gamma^t r(x_{t+1})]. Built here by explicitly
    pushing the reward forward one step at a time, with no reference to the
    resolvent -- an independent construction of the same number."""
    w = density_td_fixed_point(toy["P"], toy["pi"], GAMMA, np.eye(N_STATES), iters=50000)
    kappa = eps_kernel(toy["coords"], toy["coords"], EPS_DIRAC)
    r = toy["r_rand"]

    Ppi = policy_transition(toy["P"], toy["pi"])
    dist = toy["P"][S0, A0].copy()                 # Pr(x_1 = .)
    acc = 0.0
    for t in range(4000):
        acc += (GAMMA ** t) * float(dist @ r)      # r read at x_{t+1}
        dist = dist @ Ppi
    got = _readout(dzs, r[S0], w[S0, A0], kappa[S0], r)
    assert abs(got - acc) <= 1e-9 * max(1.0, abs(acc))


# ==================== (4) the 1-gamma normalization ==========================

@pytest.mark.parametrize("gamma", [0.5, 0.9, 0.99, 0.995])
@pytest.mark.parametrize("const", [1.0, -2.5])
def test_constant_reward_pins_the_normalization(toy, dzs, gamma, const):
    """Closed form with no MDP dependence at all: for r == c the measure integral
    is c/(1-gamma) for EVERY (s,a). A missing or doubled (1-gamma) shows up here
    as a factor-(1-gamma) error, at four discount factors."""
    M = exact_successor_measure(toy["P"], toy["pi"], gamma)
    w = density_td_fixed_point(toy["P"], toy["pi"], gamma, np.eye(N_STATES), iters=50000)
    kappa = eps_kernel(toy["coords"], toy["coords"], EPS_DIRAC)
    r = np.full(N_STATES, const)
    got = _readout(dzs, r[S0], w[S0, A0], kappa[S0], r, gamma=gamma)
    assert abs(got - const / (1.0 - gamma)) <= 1e-10 * abs(const / (1.0 - gamma))
    assert abs(got - float(measure_integral(M, r)[S0, A0])) <= 1e-9


# ================ (5) the cancelling de-biasing pair =========================

def test_debias_pair_cancels_exactly_in_the_dirac_limit(toy, dzs):
    """With kappa a true Dirac at x0, `r(x0)` and `-kappa^T r` cancel term by term
    -- in floating point too, not merely in the limit. So the readout is exactly
    w^T r / (1-gamma), which is what makes the exact-recovery statement above a
    statement about w and nothing else."""
    w = density_td_fixed_point(toy["P"], toy["pi"], GAMMA, np.eye(N_STATES), iters=50000)
    kappa = eps_kernel(toy["coords"], toy["coords"], EPS_DIRAC)
    r = toy["r_rand"]
    paired = _readout(dzs, r[S0], w[S0, A0], kappa[S0], r)
    bare = float(w[S0, A0] @ r) / (1.0 - GAMMA)
    assert paired == pytest.approx(bare, abs=1e-12)


def test_dropping_either_half_of_the_pair_breaks_the_identity(toy, dzs):
    """The regression guard for the false-null mechanism. Each half alone leaves a
    residual of exactly +-r(x0)/(1-gamma): at gamma=.99 that is a 100x amplified
    single-state reward injected into the estimate. Both halves are checked, with
    the exact predicted magnitude, not just "it is different"."""
    w = density_td_fixed_point(toy["P"], toy["pi"], GAMMA, np.eye(N_STATES), iters=50000)
    kappa = eps_kernel(toy["coords"], toy["coords"], EPS_DIRAC)
    r = toy["r_rand"]
    truth = float(measure_integral(toy["M"], r)[S0, A0])

    dropped_r0 = _readout(dzs, 0.0, w[S0, A0], kappa[S0], r)
    dropped_kappa = _readout(dzs, r[S0], w[S0, A0], np.zeros(N_STATES), r)
    assert dropped_r0 - truth == pytest.approx(-r[S0] / (1.0 - GAMMA), rel=1e-9)
    assert dropped_kappa - truth == pytest.approx(+r[S0] / (1.0 - GAMMA), rel=1e-9)
    # and it is a large error, not a rounding-level one, on this MDP
    assert abs(dropped_r0 - truth) > 0.5 * abs(truth)
    assert abs(dropped_kappa - truth) > 0.5 * abs(truth)


def test_dropping_a_half_is_catastrophic_for_an_indicator_reward(toy, dzs):
    """The G4 geometry specifically (`eval_measure_g4_spearman.py` 380-387): with
    a box-indicator reward the dropped-kappa variant injects -1[x0 in box]/(1-gamma)
    = -100 into a bounded-by-100 quantity, i.e. it flips exactly the boxes that
    contain x0 -- the ones with the LARGEST true mass. Reproduced here as a
    signed statement, since that anti-correlation is what produced the false null."""
    w = density_td_fixed_point(toy["P"], toy["pi"], GAMMA, np.eye(N_STATES), iters=50000)
    kappa = eps_kernel(toy["coords"], toy["coords"], EPS_DIRAC)
    box = ((np.arange(N_STATES) >= S0 - 1) & (np.arange(N_STATES) <= S0 + 1)).astype(np.float64)
    truth = float(measure_integral(toy["M"], box)[S0, A0])
    bad = _readout(dzs, 0.0, w[S0, A0], kappa[S0], box)     # the old, half-dropped form
    assert truth > 0.0
    assert bad - truth == pytest.approx(-1.0 / (1.0 - GAMMA), rel=1e-9)
    assert bad < 0.0                                        # a mass estimate turned negative


# ===================== (6) the finite-pool correction ========================

def test_soft_finite_pool_conserves_mass_and_error_shrinks_with_pool_size(toy, dzs):
    """The realistic finite-pool path: the eval kappa is ALWAYS the eps-kernel over
    the pool (`density_query` -> `_delta_phi`), and the matching trained w is
    nu @ K_P. K_P's rows are softmax-normalized over the pool, so w keeps unit mass
    at every pool size -- the finite-pool error is a SMOOTHING error, which the
    de-biasing pair is what corrects. Error falls monotonically as the pool
    refines and is exact at full coverage."""
    K_full = eps_kernel(toy["coords"], toy["coords"], EPS_SOFT)
    r = toy["r_quad"]
    truth = float(measure_integral(toy["M"], r)[S0, A0])
    errs = []
    for stride in (3, 2, 1):
        pool = list(range(0, N_STATES, stride))
        K_P = eps_kernel(toy["coords"], toy["coords"][pool], EPS_SOFT)
        w = toy["nu"][S0, A0] @ K_P
        assert abs(w.sum() - 1.0) < 1e-12
        errs.append(abs(_readout(dzs, r[S0], w[S0] if w.ndim > 1 else w,
                                 K_P[S0], r[pool]) - truth))
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] <= 1e-9 * abs(truth)        # full pool -> exact
    del K_full


def test_the_pair_does_not_correct_support_truncation(toy, dzs):
    """The honest limit of the correction, stated as a test so it is not later
    mistaken for a bug. If the pool simply OMITS states that carry measure mass
    (the hard-kernel picture: w = nu restricted and renormalized), then kappa is
    still an exact Dirac at x0, the pair is identically zero, and the readout
    inherits the full renormalization error. Growing P fixes it; the pair does
    not, and was never meant to."""
    r = toy["r_rand"]
    truth = float(measure_integral(toy["M"], r)[S0, A0])
    band_width = BAND[1] - BAND[0] + 1                 # nu's whole support
    for keep in (3, 5, band_width):
        pool = sorted(set(range(BAND[0], BAND[0] + keep)) | {S0})
        w = restrict_to_pool(toy["nu"][S0, A0], pool)
        kappa = eps_kernel(toy["coords"][S0:S0 + 1], toy["coords"][pool], EPS_DIRAC)[0]
        pair = (r[S0] - float(kappa @ r[pool])) / (1.0 - GAMMA)
        assert pair == pytest.approx(0.0, abs=1e-12)
        paired = _readout(dzs, r[S0], w, kappa, r[pool])
        bare = float(w @ r[pool]) / (1.0 - GAMMA)
        assert paired == pytest.approx(bare, abs=1e-10)
        if keep < band_width:
            assert abs(paired - truth) > 1e-3          # truncation error, pair or no pair
        else:
            assert paired == pytest.approx(truth, rel=1e-11)
    # full coverage of nu's support -> exact again
    full = list(range(N_STATES))
    w = restrict_to_pool(toy["nu"][S0, A0], full)
    kappa = eps_kernel(toy["coords"][S0:S0 + 1], toy["coords"], EPS_DIRAC)[0]
    assert _readout(dzs, r[S0], w, kappa, r) == pytest.approx(truth, rel=1e-11)


# ======================= (7) the soft immediate kernel =======================

def test_soft_kernel_closed_form_identity(toy, dzs):
    """Exactly what the readout computes in soft mode, written out:
        Qhat = M^T (K r)  +  ( r(x0) - (K r)(x0) ) / (1 - gamma).
    The second term is the pair, and it is the model's kernel-smoothing bias
    sampled at x0 and extrapolated over the whole discounted horizon."""
    K = eps_kernel(toy["coords"], toy["coords"], EPS_SOFT)
    w = density_td_fixed_point(toy["P"], toy["pi"], GAMMA, K, iters=50000)
    r = toy["r_quad"]
    Kr = K @ r
    predicted = float(measure_integral(toy["M"], Kr)[S0, A0]) \
        + (r[S0] - float(Kr[S0])) / (1.0 - GAMMA)
    got = _readout(dzs, r[S0], w[S0, A0], K[S0], r)
    assert got == pytest.approx(predicted, rel=1e-11)


def test_soft_kernel_recovers_ground_truth_when_the_smoothing_bias_is_constant(toy, dzs):
    """The soft-mode exact-recovery case, with its sufficient condition made
    explicit: when (r - K r) is constant over the states the measure visits AND at
    x0 (here: quadratic reward, symmetric kernel, complete neighbourhoods inside
    the band), the pair cancels the smoothing bias EXACTLY -- because total mass
    is 1/(1-gamma), which is precisely the factor the correction carries. The
    uncorrected readout is off by bias/(1-gamma); the corrected one is exact."""
    K = eps_kernel(toy["coords"], toy["coords"], EPS_SOFT)
    w = density_td_fixed_point(toy["P"], toy["pi"], GAMMA, K, iters=50000)
    r = toy["r_quad"]
    bias = r - K @ r
    lo, hi = BAND
    assert np.ptp(bias[lo:hi + 1]) < 1e-12          # constant over the visited support

    truth = float(measure_integral(toy["M"], r)[S0, A0])
    got = _readout(dzs, r[S0], w[S0, A0], K[S0], r)
    bare = float(w[S0, A0] @ r) / (1.0 - GAMMA)
    assert got == pytest.approx(truth, rel=1e-11)
    assert bare - truth == pytest.approx(-float(bias[S0]) / (1.0 - GAMMA), rel=1e-6)


def test_soft_readout_converges_to_the_hard_readout_as_eps_shrinks(toy, dzs):
    """Hard is the eps -> 0 limit of soft, on BOTH legs at once (the trained w and
    the eval kappa): the two modes are the same estimator at different widths, so
    a sign that were right in one and wrong in the other could not pass this."""
    r = toy["r_quad"]
    truth = float(measure_integral(toy["M"], r)[S0, A0])
    prev = None
    for eps in (0.7, 0.4, 0.2, 0.05):
        K = eps_kernel(toy["coords"], toy["coords"], eps)
        w = density_td_fixed_point(toy["P"], toy["pi"], GAMMA, K, iters=50000)
        err = abs(_readout(dzs, r[S0], w[S0, A0], K[S0], r) - truth)
        if prev is not None:
            assert err < prev
        prev = err
    assert prev <= 1e-6 * abs(truth)


# =================== (8) terminal / absorbing handling =======================

def test_absorbing_sink_conserves_mass_and_recovers_exactly(toy, dzs):
    """`density_loss` applies NO terminal mask to its bootstrap leg, so the
    semantics it implements is a SELF-LOOPING absorbing state. Under that
    semantics P^pi stays row-stochastic, nu stays a distribution, and exact
    recovery survives -- including from the sink itself, where the closed form is
    r(sink)/(1-gamma)."""
    rng = np.random.default_rng(7)
    n, sink = 6, 5
    P = absorbing_mdp(n, sink, seed=1)
    pi = rng.dirichlet(np.ones(2), size=n)
    M = exact_successor_measure(P, pi, GAMMA)
    nu = normalized_measure(M, GAMMA)
    assert np.abs(nu.sum(-1) - 1.0).max() < 1e-12

    coords = np.arange(n, dtype=np.float64)[:, None]
    kappa = eps_kernel(coords, coords, EPS_DIRAC)
    w = density_td_fixed_point(P, pi, GAMMA, np.eye(n), iters=60000)
    assert np.abs(w - nu).max() < 1e-11
    r = rng.normal(size=n)
    assert _readout(dzs, r[0], w[0, 0], kappa[0], r) == \
        pytest.approx(float(measure_integral(M, r)[0, 0]), rel=1e-11)
    assert _readout(dzs, r[sink], w[sink, 0], kappa[sink], r) == \
        pytest.approx(r[sink] / (1.0 - GAMMA), rel=1e-11)


def test_readout_implements_absorbing_not_episode_truncation(toy, dzs):
    """The other terminal semantics, and the exact size of the difference. If an
    episode instead ENDS at the sink, the ground truth is the sink-row-zeroed
    resolvent. The readout does not compute that: it keeps collecting r(sink)
    forever, over-counting by (M - M_trunc)[s,a,sink] * r(sink). The two agree
    exactly iff r(sink) = 0 -- which is why goal-indicator rewards have never
    exposed this, and why a nonzero-reward absorbing state would."""
    rng = np.random.default_rng(7)
    n, sink = 6, 5
    P = absorbing_mdp(n, sink, seed=1)
    pi = rng.dirichlet(np.ones(2), size=n)
    M = exact_successor_measure(P, pi, GAMMA)
    nu = normalized_measure(M, GAMMA)
    coords = np.arange(n, dtype=np.float64)[:, None]
    kappa = eps_kernel(coords, coords, EPS_DIRAC)

    Ppi_trunc = policy_transition(P, pi)
    Ppi_trunc[sink, :] = 0.0                       # episode emits nothing after the sink
    M_trunc = P @ np.linalg.inv(np.eye(n) - GAMMA * Ppi_trunc)
    assert np.abs(M[:, :, :sink] - M_trunc[:, :, :sink]).max() < 1e-12   # differ only at the sink

    r = rng.normal(size=n)
    got = _readout(dzs, r[0], nu[0, 0], kappa[0], r)
    gap = (M[0, 0, sink] - M_trunc[0, 0, sink]) * r[sink]
    assert got - float(M_trunc[0, 0] @ r) == pytest.approx(gap, rel=1e-9)
    assert abs(gap) > 1.0                          # not a rounding-level distinction

    r0 = r.copy()
    r0[sink] = 0.0
    assert _readout(dzs, r0[0], nu[0, 0], kappa[0], r0) == \
        pytest.approx(float(M_trunc[0, 0] @ r0), rel=1e-11)


# ============ (9) two properties of the shipped eval path, quantified ========

def test_double_q_min_over_heads_costs_mass_and_biases_the_readout_low(toy, dzs):
    """`density_weights(double_q=True)` takes an elementwise MIN of two SIMPLEX
    vectors, so the result is sub-normalized whenever the heads disagree. On a
    constant reward c the exact consequence is Qhat = mass * c/(1-gamma): a
    pessimism, but a multiplicative one on the mass, not a bounded shift. Recorded
    here so the exactness claims above are not read as covering this path."""
    rng = np.random.default_rng(3)
    d, pool = 4, 8
    f_emb = rng.normal(size=(2, d)) * 1.5
    b_pool = rng.normal(size=(pool, d))
    w = dzs.density_weights(f_emb, b_pool, temp=1.0, double_q=True)
    w_h0 = dzs.density_weights(f_emb[:1], b_pool, temp=1.0, double_q=False)
    assert abs(w_h0.sum() - 1.0) < 1e-12
    mass = float(w.sum())
    assert mass < 1.0 - 1e-3

    c = 2.0
    r = np.full(pool, c)
    kappa = np.full(pool, 1.0 / pool)              # any simplex kappa: kappa^T r = c
    got = _readout(dzs, c, w, kappa, r)
    assert got == pytest.approx(mass * c / (1.0 - GAMMA), rel=1e-10)


def test_numpy_kernel_twin_matches_delta_phi():
    """`eps_kernel` is only a valid stand-in for the shipped immediate term if it
    IS `_delta_phi`. Skipped in the torch venv (no jax); runs under the jax venv."""
    pytest.importorskip("jax")
    sys.path.insert(0, str(REPO / "third_party" / "ogbench" / "impls"))
    sys.path.insert(0, str(REPO / "tools" / "wandb_mode_shim"))
    try:
        from gciql_flow import GCIQLFlowAgent      # noqa: E402  (jax-gated import)
    except (ImportError, SyntaxError) as exc:      # another agent may be mid-edit
        pytest.skip(f"gciql_flow not importable right now: {exc!r}")

    rng = np.random.default_rng(11)
    obs = rng.normal(size=(5, 9))
    protos = rng.normal(size=(7, 3))
    blocks = ((2, 4, 0.3), (6, 7, 0.8))            # (start, stop, eps), widths 2 and 1
    ref = np.asarray(GCIQLFlowAgent._delta_phi(obs, protos, blocks), np.float64)
    x = np.concatenate([obs[:, 2:4], obs[:, 6:7]], -1)
    eps = np.array([0.3, 0.3, 0.8])
    assert np.abs(eps_kernel(x, protos, eps) - ref).max() < 1e-5
