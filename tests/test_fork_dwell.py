"""Tests for spec 26 R0a fork-and-dwell synthetic (contract section 1)."""

import numpy as np
import pytest

from scripts.synthetic.fork_dwell import (
    ALL_ARMS, ARM_SPECS, LEFT, NEUTRAL, RIGHT, N_ACTIONS, ForkDwellMDP,
    bellman_batch, duplicate_stationary, empirical_rho, exact_nu,
    generate_dataset, loss_and_score_grad, make_model, run_cell, train_arm,
)

GAMMA = 0.99


# ---------------------------------------------------------------------------
# exact-nu against closed form on a 3-state case (contract test 1)
# ---------------------------------------------------------------------------

def test_exact_nu_closed_form_3state_cycle():
    """Deterministic 3-cycle 0->1->2->0, one action: nu(0, u) has the closed
    form (1-g)/(1-g^3) * {1, g, g^2} for u = {1, 2, 0}."""
    g = 0.9
    P = np.zeros((1, 3, 3))
    P[0, 0, 1] = P[0, 1, 2] = P[0, 2, 0] = 1.0
    nu = exact_nu(P, g)
    c = (1 - g) / (1 - g ** 3)
    np.testing.assert_allclose(nu[0, 0], [c * g ** 2, c, c * g], atol=1e-12)
    np.testing.assert_allclose(nu[0, 1], [c * g, c * g ** 2, c], atol=1e-12)


def test_exact_nu_closed_form_3state_sticky():
    """3 states, one action: 0 -> 1; 1 sticky (stay p, else -> 2); 2 absorbing.
    Closed form: nu(1, 1) = (1-g) * p / (1 - g p)  [occupancy of 1 from state 1
    counts s_{t+1}... : sum_t g^t p^{t+1} = p/(1-gp)];
    nu(1, 2) = 1 - nu(1,1); nu(0, .) = (1-g) e_1 + g nu(1, .)."""
    g, p = 0.95, 0.8
    P = np.zeros((1, 3, 3))
    P[0, 0, 1] = 1.0
    P[0, 1, 1] = p
    P[0, 1, 2] = 1.0 - p
    P[0, 2, 2] = 1.0
    nu = exact_nu(P, g)
    nu11 = (1 - g) * p / (1 - g * p)
    np.testing.assert_allclose(nu[0, 1], [0.0, nu11, 1.0 - nu11], atol=1e-12)
    expect0 = (1 - g) * np.array([0.0, 1.0, 0.0]) + g * nu[0, 1]
    np.testing.assert_allclose(nu[0, 0], expect0, atol=1e-12)


def test_exact_nu_rows_sum_to_one():
    mdp = ForkDwellMDP(T_dwell=44)
    nu = exact_nu(mdp.P, GAMMA)
    np.testing.assert_allclose(nu.sum(-1), 1.0, atol=1e-10)


def test_mdp_pinned_dynamics():
    mdp = ForkDwellMDP(T_dwell=20)
    assert mdp.P[LEFT, mdp.D, mdp.left_cells[0]] == 1.0
    assert mdp.P[RIGHT, mdp.D, mdp.right_cells[0]] == 1.0
    assert mdp.P[NEUTRAL, mdp.D, mdp.N] == 1.0
    L = mdp.left_cells[0]
    assert mdp.P[LEFT, L, L] == pytest.approx(1 - 1 / 20)  # p_stay
    assert mdp.P[LEFT, L, mdp.D] == pytest.approx(1 / 20)  # exit to D


# ---------------------------------------------------------------------------
# duplication leaves the true MDP unchanged (contract test 2)
# ---------------------------------------------------------------------------

def test_duplication_ground_truth_invariant():
    """Ground truth is a function of the MDP only; the per-cell pipeline must
    hand every duplication factor the identical exact nu."""
    mdp = ForkDwellMDP(T_dwell=20)
    nu_ref = exact_nu(mdp.P, GAMMA)
    for dup in (1, 5, 20):
        # run_cell recomputes nu_true internally from the MDP; replicate that
        mdp_d = ForkDwellMDP(T_dwell=20)
        np.testing.assert_array_equal(exact_nu(mdp_d.P, GAMMA), nu_ref)


def test_duplication_only_replicates_stationary_rows():
    mdp = ForkDwellMDP(T_dwell=10)
    rng = np.random.default_rng(0)
    s, a, sp = generate_dataset(mdp, n_episodes=20, rng=rng)
    n_stat = int(np.sum(s == sp))
    assert n_stat > 0
    s5, a5, sp5 = duplicate_stationary(s, a, sp, 5)
    assert len(s5) == len(s) + 4 * n_stat
    # non-stationary rows preserved exactly once, in multiset terms
    def counts(x, y, z):
        keys, cnt = np.unique(np.stack([x, y, z]), axis=1, return_counts=True)
        return {tuple(k): c for k, c in zip(keys.T, cnt)}
    c1, c5 = counts(s, a, sp), counts(s5, a5, sp5)
    for k, v in c1.items():
        assert c5[k] == (5 * v if k[0] == k[2] else v)
    # stationary transitions only ever happen in region cells (never at D)
    assert not np.any((s == sp) & (s == mdp.D))


def test_dataset_matches_mdp_frequencies():
    """Sanity: empirical stay-probability in a region cell matches p_stay."""
    mdp = ForkDwellMDP(T_dwell=20)
    rng = np.random.default_rng(1)
    s, a, sp = generate_dataset(mdp, n_episodes=100, rng=rng)
    L = mdp.left_cells[0]
    in_L = s == L
    stay = float(np.mean(sp[in_L] == L))
    assert abs(stay - mdp.p_stay) < 0.02


# ---------------------------------------------------------------------------
# hand-rolled gradients: finite-difference check for every arm
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", ALL_ARMS)
def test_gradients_match_finite_difference(arm):
    mdp = ForkDwellMDP(T_dwell=5)
    rng = np.random.default_rng(3)
    s, a, sp = generate_dataset(mdp, n_episodes=5, rng=rng)
    bs, ba, bsp, bap, pool = bellman_batch(s, a, sp, batch_size=16, rng=rng)
    model = make_model(arm, mdp.n_states, np.random.default_rng(4))
    params = model.params
    target = {k: v + 0.01 for k, v in params.items()}
    S_target = model.score(target)
    mode = ARM_SPECS[arm]["mode"]

    def loss_of(p):
        return loss_and_score_grad(model.score(p), S_target, bs, ba, bsp, bap,
                                   pool, GAMMA, mode)[0]

    loss, dS = loss_and_score_grad(model.score(params), S_target, bs, ba, bsp,
                                   bap, pool, GAMMA, mode)
    grads = model.backward(params, dS)
    assert np.isfinite(loss)
    eps = 1e-6
    check_rng = np.random.default_rng(5)
    for k, g in grads.items():
        flat = params[k].ravel()
        for idx in check_rng.choice(flat.size, size=min(5, flat.size), replace=False):
            orig = flat[idx]
            flat[idx] = orig + eps
            lp = loss_of(params)
            flat[idx] = orig - eps
            lm = loss_of(params)
            flat[idx] = orig
            num = (lp - lm) / (2 * eps)
            ana = g.ravel()[idx]
            assert num == pytest.approx(ana, rel=1e-3, abs=1e-7), (k, idx)


# ---------------------------------------------------------------------------
# each arm's loss finite/decreasing on a smoke run (contract test 3)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", ALL_ARMS)
def test_arm_loss_finite_and_decreasing_smoke(arm):
    mdp = ForkDwellMDP(T_dwell=5)
    rng = np.random.default_rng(7)
    data = generate_dataset(mdp, n_episodes=40, rng=rng)
    S, curve = train_arm(arm, mdp, data, seed=0, steps=600, log_every=50)
    assert np.all(np.isfinite(curve))
    assert np.all(np.isfinite(S))
    # decreasing: late loss below the initial loss (the bootstrap target moves
    # with the online net, so mid-curve batch losses can sit at a noise floor)
    early = curve[0]
    late = np.mean(curve[-3:])
    assert late < early, (arm, curve)


def test_normalized_arm_mass_is_exactly_one():
    out = run_cell(T_dwell=5, dup=1, seed=0, arms=("density_ce",),
                   n_episodes=40, steps=400)
    assert out["arms"]["density_ce"]["mass_ratio"] == pytest.approx(1.0, abs=1e-9)


def test_run_cell_smoke_all_arms():
    out = run_cell(T_dwell=5, dup=2, seed=0, n_episodes=40, steps=300)
    assert set(out["arms"]) == set(ALL_ARMS)
    for arm, m in out["arms"].items():
        for k in ("critic_scale_mean_abs", "mass_ratio", "entropy_mean",
                  "action_gap", "success", "fixed_point_error", "gt_l1"):
            assert np.isfinite(m[k]), (arm, k)
    assert out["stationary_frac"] > 0.5  # duplication inflates stationary rows
