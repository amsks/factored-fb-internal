"""Tests for spec 26 R0a-v2 (contract: specs/26-bfm-roadmap.md, "R0a verdict
(2026-08-13) and the R0a-v2 redesign").

The centerpiece is the empirical-kernel invariance block: the whole redesign
exists because v1's row injection corrupted the kernel, so these tests must be
airtight -- byte-level equality of the weighted kernel and of its exact
successor measure across every weight factor, plus positive controls showing
the test DOES fire on the two corrupting variants (v1 row duplication and
transition-keyed weighting).
"""

import numpy as np
import pytest

from scripts.synthetic import fork_dwell as v1
from scripts.synthetic.fork_dwell_v2 import (
    ALL_ARMS, CHECKPOINTS, COLUMNS, CONSTRAINED_COLUMNS, CONTRACT_TRANSIENTS,
    CONTROL_COLUMN, EXPLORATORY_COLUMNS,
    DISTRACT, LEFT, NEUTRAL, N_ACTIONS, RIGHT, WEIGHTS, BilinearTrunkScore,
    ForkDwellV2MDP, JointTrunkScore, ARM_MODE, WeightedSampler,
    bellman_batch_weighted, compute_metrics, empirical_kernel,
    exact_empirical_nu, generate_dataset, make_model, predicted_nu, row_weights,
    run_cell, train_arm, weighted_rho,
)
from scripts.synthetic.fork_dwell import exact_nu, loss_and_score_grad

GAMMA = 0.99


def _dataset(T_dwell=20, n_episodes=60, seed=0, m=3):
    mdp = ForkDwellV2MDP(T_dwell, m=m)
    s, a, sp = generate_dataset(mdp, n_episodes, np.random.default_rng(seed))
    return mdp, s, a, sp


# ---------------------------------------------------------------------------
# THE redesign test: multiplicity as a weight leaves the empirical kernel alone
# ---------------------------------------------------------------------------

def test_empirical_kernel_byte_identical_across_weights():
    """Weighted empirical kernel is byte-for-byte the unweighted one, for every
    weight factor -- the property v1's row injection destroyed."""
    mdp, s, a, sp = _dataset()
    ref = empirical_kernel(s, a, sp, mdp.n_states)
    # every (s,a) actually visited, so the comparison is not vacuous
    counts = np.zeros((N_ACTIONS, mdp.n_states))
    np.add.at(counts, (a, s), 1.0)
    assert counts.min() > 0
    np.testing.assert_allclose(ref.sum(-1), 1.0, atol=1e-12)
    for wf in WEIGHTS:
        w = row_weights(s, mdp, wf)
        K = empirical_kernel(s, a, sp, mdp.n_states, w)
        assert K.tobytes() == ref.tobytes(), f"kernel moved at weight {wf}"


def test_exact_successor_measure_of_empirical_kernel_invariant():
    """The tabular fixed point itself (not just the kernel) is weight-invariant:
    the analytic statement behind the predicted-null control column."""
    mdp, s, a, sp = _dataset()
    ref = exact_empirical_nu(s, a, sp, mdp.n_states, GAMMA)
    for wf in WEIGHTS:
        w = row_weights(s, mdp, wf)
        nu = exact_empirical_nu(s, a, sp, mdp.n_states, GAMMA, w)
        assert nu.tobytes() == ref.tobytes(), f"tabular fixed point moved at {wf}"


def test_dataset_rows_are_never_touched_by_the_weighting():
    mdp, s, a, sp = _dataset()
    before = (s.copy(), a.copy(), sp.copy())
    for wf in WEIGHTS:
        w = row_weights(s, mdp, wf)
        WeightedSampler(w).draw(64, np.random.default_rng(0))
        weighted_rho(sp, w, mdp.n_states)
    for x, y in zip((s, a, sp), before):
        np.testing.assert_array_equal(x, y)
    assert len(s) == len(before[0])


def test_positive_control_v1_row_duplication_does_move_the_kernel():
    """Instrument check: the invariance test fires on v1's intervention."""
    mdp, s, a, sp = _dataset()
    ref = empirical_kernel(s, a, sp, mdp.n_states)
    for dup in (2, 5, 20):
        sd, ad, spd = v1.duplicate_stationary(s, a, sp, dup)
        K = empirical_kernel(sd, ad, spd, mdp.n_states)
        assert K.tobytes() != ref.tobytes()
        L = mdp.left_cells[0]
        assert K[LEFT, L, L] > ref[LEFT, L, L] + 1e-6   # P-hat(stay) inflated


def test_positive_control_transition_keyed_weights_move_the_kernel():
    """Why the weight is keyed on the STATE, not on ``s' == s``: a transition-
    keyed weight is v1 duplication in expectation and corrupts the kernel the
    same way (it also reproduces duplication's kernel exactly at integer w)."""
    mdp, s, a, sp = _dataset()
    ref = empirical_kernel(s, a, sp, mdp.n_states)
    for wf in (2, 5, 20):
        w_bad = np.where(s == sp, float(wf), 1.0)
        K_bad = empirical_kernel(s, a, sp, mdp.n_states, w_bad)
        assert K_bad.tobytes() != ref.tobytes()
        sd, ad, spd = v1.duplicate_stationary(s, a, sp, wf)
        np.testing.assert_allclose(
            K_bad, empirical_kernel(sd, ad, spd, mdp.n_states), atol=1e-12)


def test_weighting_does_shift_the_projection_distribution():
    """The intervention must not be a no-op: the sampling distribution and the
    query-pool marginal DO move with the weight factor."""
    mdp, s, a, sp = _dataset()
    rhos, decision_share = [], []
    for wf in WEIGHTS:
        w = row_weights(s, mdp, wf)
        rhos.append(weighted_rho(sp, w, mdp.n_states))
        drawn = WeightedSampler(w).draw(200_000, np.random.default_rng(1))
        frac_dwell = float(np.mean(mdp.dwell_states[s[drawn]]))
        expect = w[mdp.dwell_states[s]].sum() / w.sum()
        assert abs(frac_dwell - expect) < 0.01
        decision_share.append(float(np.mean(s[drawn] == mdp.D)))
    # the decision rows -- where the fork is learned -- are squeezed out
    assert all(decision_share[i] > decision_share[i + 1]
               for i in range(len(decision_share) - 1)), decision_share
    assert decision_share[0] > 5 * decision_share[-1]
    # and the query-pool marginal moves too
    assert all(not np.allclose(rhos[0], r) for r in rhos[1:])


def test_row_weights_are_state_keyed():
    mdp, s, a, sp = _dataset()
    w = row_weights(s, mdp, 5)
    assert np.all(w[mdp.dwell_states[s]] == 5.0)
    assert np.all(w[~mdp.dwell_states[s]] == 1.0)
    # dwell rows include exits (s' == D), not only self-loops
    assert np.any(mdp.dwell_states[s] & (sp == mdp.D))


def test_weighted_sampler_matches_weights():
    w = np.array([1.0, 3.0, 1.0, 5.0])
    draws = WeightedSampler(w).draw(200_000, np.random.default_rng(2))
    freq = np.bincount(draws, minlength=4) / len(draws)
    np.testing.assert_allclose(freq, w / w.sum(), atol=0.005)


# ---------------------------------------------------------------------------
# MDP / ground truth / features
# ---------------------------------------------------------------------------

def test_mdp_pinned_dynamics():
    mdp = ForkDwellV2MDP(T_dwell=20, m=3)
    assert mdp.n_states == 11
    for act, cells in ((LEFT, mdp.left_cells), (RIGHT, mdp.right_cells),
                       (DISTRACT, mdp.distract_cells)):
        np.testing.assert_allclose(mdp.P[act, mdp.D, cells], 1.0 / 3)
    assert mdp.P[NEUTRAL, mdp.D, mdp.N] == 1.0
    L = mdp.left_cells[0]
    assert mdp.P[LEFT, L, L] == pytest.approx(1 - 1 / 20)
    assert mdp.P[LEFT, L, mdp.D] == pytest.approx(1 / 20)
    assert not mdp.dwell_states[mdp.D] and mdp.dwell_states[mdp.N]


def test_exact_nu_rows_sum_to_one_and_left_is_optimal():
    mdp = ForkDwellV2MDP(T_dwell=44)
    nu = exact_nu(mdp.P, GAMMA)
    np.testing.assert_allclose(nu.sum(-1), 1.0, atol=1e-10)
    r = np.zeros(mdp.n_states)
    r[mdp.left_cells] = 1.0
    Q = np.einsum("asu,u->as", nu, r)
    assert np.argmax(Q[:, mdp.D]) == LEFT
    assert Q[LEFT, mdp.D] > Q[DISTRACT, mdp.D] > 0   # distractor is a real rival


def test_ground_truth_rank_exceeds_the_constrained_ranks():
    """d in {2,4} must genuinely under-parameterize the target (~10 needed)."""
    mdp = ForkDwellV2MDP(T_dwell=44)
    nu = exact_nu(mdp.P, GAMMA)
    rank = np.linalg.matrix_rank(nu.reshape(-1, mdp.n_states), tol=1e-8)
    assert rank >= 8, rank
    assert max(COLUMNS[c]["d"] for c in CONSTRAINED_COLUMNS) < rank


def test_feature_map_aliases_distractor_with_left():
    mdp = ForkDwellV2MDP(T_dwell=20, m=3, alias_eps=0.1)
    phi = mdp.features("aliased")
    coarse = 2 + 2 * mdp.m
    for k in range(mdp.m):
        L, X, R = mdp.left_cells[k], mdp.distract_cells[k], mdp.right_cells[k]
        np.testing.assert_array_equal(phi[L, :coarse], phi[X, :coarse])
        cos = phi[L] @ phi[X] / (np.linalg.norm(phi[L]) * np.linalg.norm(phi[X]))
        assert cos > 0.98                      # aliased, but not identical
        assert not np.allclose(phi[L], phi[X])
        assert phi[L, :coarse] @ phi[R, :coarse] == 0.0   # right is not aliased
    np.testing.assert_array_equal(mdp.features("onehot"), np.eye(mdp.n_states))


def test_dataset_matches_mdp_frequencies():
    mdp, s, a, sp = _dataset(T_dwell=20, n_episodes=100, seed=1)
    L = mdp.left_cells[0]
    stay = float(np.mean(sp[s == L] == L))
    assert abs(stay - mdp.p_stay) < 0.03


# ---------------------------------------------------------------------------
# Models: hand-rolled gradients, capacity wiring
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", ALL_ARMS)
@pytest.mark.parametrize("column", ("d2", "tab"))
def test_gradients_match_finite_difference(arm, column):
    mdp, s, a, sp = _dataset(T_dwell=5, n_episodes=5, seed=3)
    sampler = WeightedSampler(row_weights(s, mdp, 5))
    bs, ba, bsp, bap, pool = bellman_batch_weighted(
        s, a, sp, sampler, 16, np.random.default_rng(3))
    model = make_model(arm, mdp, column, np.random.default_rng(4))
    params = model.params
    S_target = model.score({k: v + 0.01 for k, v in params.items()})
    mode = ARM_MODE[arm]

    def loss_of(p):
        return loss_and_score_grad(model.score(p), S_target, bs, ba, bsp, bap,
                                   pool, GAMMA, mode)[0]

    loss, dS = loss_and_score_grad(model.score(params), S_target, bs, ba, bsp,
                                   bap, pool, GAMMA, mode)
    grads = model.backward(params, dS)
    assert np.isfinite(loss)
    eps = 1e-6
    rng = np.random.default_rng(5)
    for k, g in grads.items():
        flat = params[k].ravel()
        for idx in rng.choice(flat.size, size=min(5, flat.size), replace=False):
            orig = flat[idx]
            flat[idx] = orig + eps
            lp = loss_of(params)
            flat[idx] = orig - eps
            lm = loss_of(params)
            flat[idx] = orig
            num = (lp - lm) / (2 * eps)
            assert num == pytest.approx(g.ravel()[idx], rel=2e-3, abs=1e-7), (k, idx)


def test_capacity_knob_is_wired_to_the_column():
    mdp = ForkDwellV2MDP(T_dwell=20)
    for column, spec in COLUMNS.items():
        bil = make_model("density_ce", mdp, column, np.random.default_rng(0))
        mlp = make_model("joint_ce", mdp, column, np.random.default_rng(0))
        assert isinstance(bil, BilinearTrunkScore) and bil.d == spec["d"]
        assert isinstance(mlp, JointTrunkScore) and mlp.hidden == spec["hidden"]
        assert bil.phi.shape[0] == mdp.n_states
        assert bil.score(bil.params).shape == (N_ACTIONS, mdp.n_states, mdp.n_states)
        assert mlp.score(mlp.params).shape == (N_ACTIONS, mdp.n_states, mdp.n_states)
    # the trunk is shared: one W1 for all states, so regions compete for it
    bil = make_model("density_ce", mdp, "d2", np.random.default_rng(0))
    assert set(bil.params) == {"W1", "b1", "WF", "WB"}


def test_bilinear_rank_constraint_binds():
    """A d=2 bilinear score tensor has rank <= 2 per action slice."""
    mdp = ForkDwellV2MDP(T_dwell=20)
    bil = make_model("density_ce", mdp, "d2", np.random.default_rng(0))
    S = bil.score(bil.params)
    for a in range(N_ACTIONS):
        assert np.linalg.matrix_rank(S[a], tol=1e-8) <= 2


# ---------------------------------------------------------------------------
# Training / metrics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", ALL_ARMS)
def test_arm_loss_finite_and_decreasing_smoke(arm):
    mdp, s, a, sp = _dataset(T_dwell=5, n_episodes=40, seed=7)
    sampler = WeightedSampler(row_weights(s, mdp, 5))
    snaps, curve = train_arm(arm, mdp, (s, a, sp), sampler, "d4", seed=0,
                             steps=600, checkpoints=(200,), log_every=50)
    assert np.all(np.isfinite(curve))
    assert set(snaps) == {200, 600}
    assert all(np.all(np.isfinite(S)) for S in snaps.values())
    assert np.mean(curve[-3:]) < curve[0], (arm, curve)


def test_transient_checkpoints_are_logged():
    out = run_cell(T_dwell=5, weight=2, seed=0, column="d2",
                   arms=("density_ce",), n_episodes=30, steps=400,
                   checkpoints=(100, 200))
    by_step = out["arms"]["density_ce"]["by_step"]
    assert sorted(int(k) for k in by_step) == [100, 200, 400]
    assert out["arms"]["density_ce"]["final_step"] == 400
    # the contracted 1k/3k/10k transients are always logged
    assert set(CONTRACT_TRANSIENTS).issubset(CHECKPOINTS)
    assert CONTRACT_TRANSIENTS == (1000, 3000, 10000)
    for col, spec in COLUMNS.items():
        assert spec["steps"] >= max(CONTRACT_TRANSIENTS), col


def test_normalized_arm_mass_is_exactly_one():
    out = run_cell(T_dwell=5, weight=5, seed=0, column="d4",
                   arms=("density_ce",), n_episodes=30, steps=300,
                   checkpoints=(100,))
    assert out["arms"]["density_ce"]["mass_ratio"] == pytest.approx(1.0, abs=1e-9)


def test_run_cell_smoke_all_arms_and_metric_keys():
    out = run_cell(T_dwell=5, weight=10, seed=0, column="d2", n_episodes=30,
                   steps=300, checkpoints=(100,))
    assert set(out["arms"]) == set(ALL_ARMS)
    assert out["dwell_sample_frac"] > out["dwell_row_frac"]
    for arm, m in out["arms"].items():
        for k in ("critic_scale_mean_abs", "mass_ratio", "entropy_mean",
                  "action_gap", "margin", "margin_ratio", "success",
                  "alias_confusion", "fixed_point_error", "gt_l1"):
            assert np.isfinite(m[k]), (arm, k)


def test_run_cell_is_deterministic():
    kw = dict(T_dwell=5, weight=5, seed=1, column="d2", arms=("fb_unnorm",),
              n_episodes=30, steps=200, checkpoints=(100,))
    a1, a2 = run_cell(**kw), run_cell(**kw)
    assert a1["arms"]["fb_unnorm"]["margin"] == a2["arms"]["fb_unnorm"]["margin"]
    assert a1["rho"] == a2["rho"]


def test_dataset_depends_only_on_T_and_seed_not_on_weight():
    """The weight must not perturb data generation (else the columns are not
    comparable across weight factors)."""
    rhos = {}
    for wf in (1, 20):
        out = run_cell(T_dwell=5, weight=wf, seed=2, column="d2",
                       arms=("fb_unnorm",), n_episodes=20, steps=100,
                       checkpoints=(50,))
        rhos[wf] = out["rho"]
        assert out["n_transitions"] == 20 * 200
    assert rhos[1] != rhos[20]      # only the pool marginal moves


def test_margin_metric_is_exact_on_ground_truth():
    """Feeding the exact measure through the metric gives margin_ratio 1 and
    success 1 -- the control that the hardened score is not broken."""
    mdp = ForkDwellV2MDP(T_dwell=44)
    nu_true = exact_nu(mdp.P, GAMMA)
    rho = np.full(mdp.n_states, 1.0 / mdp.n_states)
    S = np.log(np.clip(nu_true / rho[None, None, :], 1e-30, None))
    m = compute_metrics(S, "density_ce", rho, mdp, nu_true)
    assert m["margin_ratio"] == pytest.approx(1.0, rel=1e-6)
    assert m["success"] == 1.0
    assert m["gt_l1"] < 1e-9
    assert 0.0 < m["alias_confusion"] < 1.0


def test_success_can_fail_under_aliasing():
    """The hardened task is not saturated by construction: a model that cannot
    tell L_k from X_k (coarse features only) loses the decision."""
    mdp = ForkDwellV2MDP(T_dwell=44)
    nu_true = exact_nu(mdp.P, GAMMA)
    coarse = 2 + 2 * mdp.m
    phi = mdp.features("aliased")[:, :coarse]        # drop the identity part
    # aliased read-out: mass that truly lands on X_k is credited to L_k too
    alias = phi @ phi.T
    alias = alias / np.clip(alias.sum(-1, keepdims=True), 1e-30, None)
    nu_alias = np.einsum("asu,uv->asv", nu_true, alias)
    rho = np.full(mdp.n_states, 1.0 / mdp.n_states)
    S = np.log(np.clip(nu_alias / rho[None, None, :], 1e-30, None))
    m = compute_metrics(S, "density_ce", rho, mdp, nu_true)
    assert abs(m["margin"]) < 1e-9     # margin destroyed by aliasing
    assert m["success"] == 0.0         # an exact tie with the distractor fails


def test_predicted_nu_normalized_sums_to_one():
    rng = np.random.default_rng(0)
    S = rng.standard_normal((N_ACTIONS, 11, 11))
    rho = rng.random(11)
    rho /= rho.sum()
    nu = predicted_nu(S, rho, GAMMA, normalized=True)
    np.testing.assert_allclose(nu.sum(-1), 1.0, atol=1e-12)


def test_control_column_is_the_converged_tabular_one():
    assert COLUMNS[CONTROL_COLUMN]["features"] == "onehot"
    assert COLUMNS[CONTROL_COLUMN]["steps"] >= 3 * COLUMNS["d2"]["steps"]
    assert COLUMNS[CONTROL_COLUMN]["d"] >= ForkDwellV2MDP(20).n_states


def test_gate_columns_are_the_contracted_ones():
    """The verdict may only read d in {2,4} (contract) and the tabular control;
    every other column is exploratory."""
    assert CONSTRAINED_COLUMNS == ("d2", "d4")
    assert all(COLUMNS[c]["d"] == int(c[1:]) for c in CONSTRAINED_COLUMNS)
    assert all(COLUMNS[c]["steps"] == COLUMNS["d2"]["steps"]
               for c in CONSTRAINED_COLUMNS)          # one FIXED budget
    assert CONTROL_COLUMN not in CONSTRAINED_COLUMNS + EXPLORATORY_COLUMNS
    assert set(COLUMNS) == set(CONSTRAINED_COLUMNS) | set(EXPLORATORY_COLUMNS) | {CONTROL_COLUMN}
