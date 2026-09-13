#!/usr/bin/env python3
"""Verify a deterministic four-state, rank-two successor counterexample.

Uses exact successor targets, NumPy linear algebra, and rational interval bounds.
There is no training, parameter search, sampled data, or learned actor. See
docs/notes/2026-09-11-deterministic-path-counterexample.md for the proof and scope.
"""

from fractions import Fraction
from itertools import product
import json

import numpy as np


ATOL = 1e-12
INTERVAL_TERMS = 32
INTERVAL_DENOMINATOR = 10**20


def log_interval(value):
    """Rational lower/upper bounds for log(value), using the atanh series.

    Reduce the argument to [1, 2). For y=(x-1)/(x+1), omitted positive terms
    in log(x)=2*sum(y**(2*k+1)/(2*k+1)) have a geometric upper bound.
    Every operation, including the tail bound, uses exact Fraction arithmetic.
    """
    value = Fraction(value)
    assert value > 0

    def reduced(argument):
        y = (argument - 1) / (argument + 1)
        y_squared = y * y
        power = y
        lower = Fraction(0)
        for k in range(INTERVAL_TERMS):
            lower += 2 * power / (2 * k + 1)
            power *= y_squared
        tail = 2 * power / ((2 * INTERVAL_TERMS + 1) * (1 - y_squared))
        return lower, lower + tail

    exponent = 0
    while value >= 2:
        value /= 2
        exponent += 1
    while value < 1:
        value *= 2
        exponent -= 1
    lower, upper = reduced(value)
    log_two_lower, log_two_upper = reduced(Fraction(2))
    if exponent >= 0:
        return lower + exponent * log_two_lower, upper + exponent * log_two_upper
    return lower + exponent * log_two_upper, upper + exponent * log_two_lower


def exp_interval(value):
    """Rational bounds for exp(value); all logits in this proof have |x|<1.

    For x>=0, bound the Taylor tail by its first term times a geometric sum.
    For x<0, invert the bounds for exp(-x).
    """
    value = Fraction(value)
    if value < 0:
        lower, upper = exp_interval(-value)
        return 1 / upper, 1 / lower
    assert value <= 2
    term = Fraction(1)
    lower = term
    for k in range(1, INTERVAL_TERMS + 1):
        term *= value / k
        lower += term
    first_omitted = term * value / (INTERVAL_TERMS + 1)
    tail = first_omitted / (1 - value / (INTERVAL_TERMS + 2))
    return lower, lower + tail


def rounded_interval(lower, upper):
    """Round outward to control rational denominator size before taking logs."""
    denominator = INTERVAL_DENOMINATOR
    lower_integer = (lower.numerator * denominator) // lower.denominator
    upper_integer = -((-upper.numerator * denominator) // upper.denominator)
    return Fraction(lower_integer, denominator), Fraction(upper_integer, denominator)


def bernoulli_kl_interval(p, q):
    p, q = Fraction(p), Fraction(q)
    first = log_interval(p / q)
    second = log_interval((1 - p) / (1 - q))
    return tuple(p * first[k] + (1 - p) * second[k] for k in (0, 1))


def categorical_kl_interval(numerators, forward, backward):
    """Bound mean KL to softmax(F B^T) without floating-point transcendents."""
    lower, upper = Fraction(0), Fraction(0)
    for i in range(4):
        logits = [sum(forward[i][k] * backward[j][k] for k in range(2)) for j in range(4)]
        exponentials = [exp_interval(logit) for logit in logits]
        normalizer = rounded_interval(
            sum(item[0] for item in exponentials), sum(item[1] for item in exponentials)
        )
        log_normalizer_lower = log_interval(normalizer[0])[0]
        log_normalizer_upper = log_interval(normalizer[1])[1]
        row_lower, row_upper = log_normalizer_lower, log_normalizer_upper
        for j in range(4):
            probability = Fraction(int(numerators[i, j]), 85)
            log_probability = log_interval(probability)
            row_lower += probability * (log_probability[0] - logits[j])
            row_upper += probability * (log_probability[1] - logits[j])
        lower += row_lower / 4
        upper += row_upper / 4
    return lower, upper


def check_probabilities(probabilities):
    assert np.all(probabilities > 0)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=ATOL, rtol=0)


def mean_kl(target, prediction):
    return float(np.mean(np.sum(target * np.log(target / prediction), axis=1)))


def bernoulli_kl(p, q):
    return float(p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q)))


def verify():
    gamma = 4 / 5
    # Columns are left/right actions. Both actions at every state are sampled.
    next_state = np.array([[0, 1], [0, 2], [1, 3], [2, 3]])
    transition = np.eye(4)[next_state.reshape(-1)]
    source_weights = np.full(8, 1 / 8)
    reference = source_weights @ transition
    continuation = transition.reshape(4, 2, 4).mean(axis=1)
    policy_average = np.kron(np.eye(4), np.full((1, 2), 1 / 2))
    np.testing.assert_array_equal(np.bincount(next_state.reshape(-1)), np.full(4, 2))
    np.testing.assert_allclose(reference, 1 / 4, atol=ATOL, rtol=0)
    np.testing.assert_allclose(continuation, continuation.T, atol=ATOL, rtol=0)
    np.testing.assert_allclose(reference @ continuation, reference, atol=ATOL, rtol=0)

    numerators = np.array([[43, 22, 12, 8], [22, 33, 18, 12], [12, 18, 33, 22], [8, 12, 22, 43]])
    target = numerators / 85
    solved_target = (1 - gamma) * np.linalg.inv(np.eye(4) - gamma * continuation)
    np.testing.assert_allclose(target, solved_target, atol=ATOL, rtol=0)
    check_probabilities(target)
    target_sa = transition @ target
    np.testing.assert_allclose(
        target_sa, (1 - gamma) * transition + gamma * transition @ policy_average @ target_sa,
        atol=ATOL, rtol=0,
    )

    # With these row/reference weights, the full eight-row weighted problem
    # has the same singular values and repeated-row optimum as the four-row N.
    left, singular, right = np.linalg.svd(target, full_matrices=False)
    least_squares = (left[:, :2] * singular[:2]) @ right[:2]
    expected_singular = np.array([1, 1 / (5 - 2 * np.sqrt(2)), 1 / 5, 1 / (5 + 2 * np.sqrt(2))])
    np.testing.assert_allclose(singular, expected_singular, atol=ATOL, rtol=0)
    assert singular[1] > singular[2] > singular[3] > 0
    weighted_target = np.sqrt(source_weights)[:, None] * target_sa / np.sqrt(reference)[None, :]
    weighted_left, weighted_singular, weighted_right = np.linalg.svd(weighted_target, full_matrices=False)
    weighted_fit = (weighted_left[:, :2] * weighted_singular[:2]) @ weighted_right[:2]
    fitted_sa = weighted_fit * np.sqrt(reference)[None, :] / np.sqrt(source_weights)[:, None]
    np.testing.assert_allclose(weighted_singular, singular, atol=ATOL, rtol=0)
    np.testing.assert_allclose(fitted_sa, transition @ least_squares, atol=ATOL, rtol=0)
    check_probabilities(least_squares)

    smooth = np.cos(np.pi * (np.arange(4) + 1 / 2) / 4) / np.sqrt(2)
    analytic_fit = np.full((4, 4), 1 / 4) + singular[1] * np.outer(smooth, smooth)
    np.testing.assert_allclose(least_squares, analytic_fit, atol=ATOL, rtol=0)
    backward_fb = np.column_stack((np.ones(4), 2 * smooth))
    forward_fb = np.column_stack((np.ones(4), 2 * singular[1] * smooth))
    np.testing.assert_allclose(forward_fb @ backward_fb.T / 4, least_squares, atol=ATOL, rtol=0)
    np.testing.assert_allclose(backward_fb.T @ backward_fb / 4, np.eye(2), atol=ATOL, rtol=0)
    assert np.ptp(np.sum(backward_fb**2, axis=1)) > 1  # Not spherical backward features.
    np.testing.assert_allclose(least_squares.min(), (20 - 9 * np.sqrt(2)) / 136, atol=ATOL, rtol=0)

    # This different, feasible FB model preserves every value for this goal.
    goal = 1
    goal_probabilities = target[:, goal]
    goal_feature = np.array([-1, 3, -1, -1]) / np.sqrt(3)
    backward_control = np.column_stack((np.ones(4), goal_feature))
    forward_control = np.column_stack((np.ones(4), (4 * goal_probabilities - 1) / np.sqrt(3)))
    control = forward_control @ backward_control.T / 4
    check_probabilities(control)
    assert np.linalg.matrix_rank(control, tol=ATOL) == 2
    np.testing.assert_allclose(backward_control.T @ backward_control / 4, np.eye(2), atol=ATOL, rtol=0)
    np.testing.assert_allclose(control[:, goal], target[:, goal], atol=ATOL, rtol=0)
    np.testing.assert_allclose(control[:, [0, 2, 3]], np.repeat((1 - goal_probabilities[:, None]) / 3, 3, axis=1), atol=ATOL, rtol=0)
    reward = np.eye(4)[goal]
    reward_vector = backward_control.T @ (reference * reward)
    np.testing.assert_allclose(forward_control @ reward_vector, target @ reward, atol=ATOL, rtol=0)
    squared_optimum = float(np.sum((least_squares - target) ** 2))
    squared_control = float(np.sum((control - target) ** 2))
    np.testing.assert_allclose(squared_optimum, np.sum(singular[2:] ** 2), atol=ATOL, rtol=0)
    assert squared_control > squared_optimum

    # Explicit rational candidate; its certificate does not require optimization.
    backward_exact = [[Fraction(-1), Fraction(1)], [Fraction(-1, 2), Fraction(-1)], [Fraction(1, 2), Fraction(-1)], [Fraction(1), Fraction(1)]]
    forward_exact = [[Fraction(-4, 5), Fraction(1, 10)], [Fraction(-2, 5), Fraction(-23, 100)], [Fraction(2, 5), Fraction(-23, 100)], [Fraction(4, 5), Fraction(1, 10)]]
    backward_density = np.array(backward_exact, dtype=float)
    forward_density = np.array(forward_exact, dtype=float)
    logits = forward_density @ backward_density.T
    categorical = np.exp(logits - logits.max(axis=1, keepdims=True))
    categorical /= categorical.sum(axis=1, keepdims=True)
    check_probabilities(categorical)
    assert np.linalg.matrix_rank(logits, tol=ATOL) == 2
    assert np.linalg.matrix_rank(categorical, tol=ATOL) == 4
    regret = mean_kl(target, categorical)
    regret_interval = categorical_kl_interval(numerators, forward_exact, backward_exact)
    assert regret_interval[1] < Fraction(3305, 1_000_000)
    assert float(regret_interval[0]) - ATOL <= regret <= float(regret_interval[1]) + ATOL

    # At every global optimum identical target rows share their prediction:
    # fix B and use strict convexity in each row's attainable centered logits.
    # Thus each pair of duplicate SA rows has total weight 1/4, not 1/8.
    bounds = {}
    certified_bounds = {}
    for state, good_next, bad_next in [(0, 1, 0), (2, 1, 3), (3, 2, 3)]:
        p, q = target[[good_next, bad_next], goal]
        assert p > q
        midpoint = (p + q) / 2
        bounds[state] = (bernoulli_kl(p, midpoint) + bernoulli_kl(q, midpoint)) / 4
        p_exact = Fraction(int(numerators[good_next, goal]), 85)
        q_exact = Fraction(int(numerators[bad_next, goal]), 85)
        midpoint_exact = (p_exact + q_exact) / 2
        first, second = bernoulli_kl_interval(p_exact, midpoint_exact), bernoulli_kl_interval(q_exact, midpoint_exact)
        certified_bounds[state] = tuple((first[k] + second[k]) / 4 for k in (0, 1))
    assert regret_interval[1] < certified_bounds[0][0]
    assert regret_interval[1] < certified_bounds[2][0]
    assert certified_bounds[0][0] > Fraction(4806, 1_000_000)
    assert certified_bounds[2][0] > Fraction(20199, 1_000_000)
    assert regret > bounds[3]  # The certificate does not cover the unused state 3.

    # Both full targets are misspecified. The centered log rank is exactly
    # three analytically: its odd block has positive determinant, and its
    # even, zero-mean input (1,-1,-1,1) has a nonzero, independent even image.
    centered_log = np.log(target) - np.log(target).mean(axis=1, keepdims=True)
    log_singular = np.linalg.svd(centered_log, compute_uv=False)
    assert np.linalg.matrix_rank(target, tol=ATOL) == 4
    assert np.linalg.matrix_rank(centered_log, tol=ATOL) == 3
    odd = np.array([[1, 0], [0, 1], [0, -1], [-1, 0]]) / np.sqrt(2)
    odd_block = odd.T @ centered_log @ odd
    expected_odd_block = np.array([[np.log(43 / 8), np.log(11 / 6)], [np.log(11 / 6), np.log(11 / 6)]])
    np.testing.assert_allclose(odd_block, expected_odd_block, atol=ATOL, rtol=0)
    np.testing.assert_allclose(np.linalg.det(odd_block), np.log(11 / 6) * np.log(129 / 44), atol=ATOL, rtol=0)
    assert np.linalg.det(odd_block) > 0
    even = np.array([1, -1, -1, 1]) / 2
    expected_even_image = np.array([np.log(43 / 33), np.log(4 / 9), np.log(4 / 9), np.log(43 / 33)]) / 2
    np.testing.assert_allclose(centered_log @ even, expected_even_image, atol=ATOL, rtol=0)
    assert np.linalg.norm(expected_even_image) > 0

    def action_values(prediction):
        return prediction[next_state, goal] / (1 - gamma)

    def policy_value(policy):
        successors = next_state[np.arange(4), policy]
        kernel = np.eye(4)[successors]
        return np.linalg.solve(np.eye(4) - gamma * kernel, reward[successors])

    true_gap = target[1, goal] - target[0, goal]
    fb_gap = least_squares[1, goal] - least_squares[0, goal]
    np.testing.assert_allclose(true_gap, 11 / 85, atol=ATOL, rtol=0)
    np.testing.assert_allclose(fb_gap, (1 - 3 * np.sqrt(2)) / 68, atol=ATOL, rtol=0)
    assert fb_gap < 0 < true_gap
    fb_policy = action_values(least_squares).argmax(axis=1)
    density_policy = action_values(categorical).argmax(axis=1)
    all_policies = np.array(list(product((0, 1), repeat=4)))
    all_returns = np.array([policy_value(policy)[0] for policy in all_policies])
    optimal_return = float(all_returns.max())
    np.testing.assert_allclose(optimal_return, 25 / 9, atol=ATOL, rtol=0)
    np.testing.assert_allclose(policy_value(fb_policy)[0], 0, atol=ATOL, rtol=0)
    np.testing.assert_allclose(policy_value(density_policy)[0], optimal_return, atol=ATOL, rtol=0)
    # The certificate fixes states 0 and 2; either action at states 1 and 3
    # gives the optimal start-0 return because state 3 is never reached.
    certified_policies = all_policies[(all_policies[:, 0] == 1) & (all_policies[:, 2] == 0)]
    assert len(certified_policies) == 4
    for policy in certified_policies:
        np.testing.assert_allclose(policy_value(policy)[0], optimal_return, atol=ATOL, rtol=0)

    def interval_report(interval):
        return {"lower_approximation": float(interval[0]), "upper_approximation": float(interval[1]), "width": float(interval[1] - interval[0])}

    return {
        "discount": gamma,
        "next_state_left_right": next_state.tolist(),
        "fixed_continuation": continuation.tolist(),
        "source_weights": source_weights.tolist(),
        "natural_reference": reference.tolist(),
        "normalized_successor_N": target.tolist(),
        "target_probability_rank": 4,
        "target_singular_values": singular.tolist(),
        "target_centered_log_rank": 3,
        "target_centered_log_singular_values": log_singular.tolist(),
        "least_squares_optimum": {"probabilities": least_squares.tolist(), "squared_loss": squared_optimum, "minimum_probability": float(least_squares.min()), "backward_squared_norms": np.sum(backward_fb**2, axis=1).tolist(), "normalized_start_goal_gap_right_minus_left": float(fb_gap), "unnormalized_start_goal_gap_right_minus_left": float(fb_gap / (1 - gamma)), "policy_left_0_right_1": fb_policy.tolist(), "return_from_0": float(policy_value(fb_policy)[0])},
        "goal_control_sufficient_fb": {"forward": forward_control.tolist(), "backward": backward_control.tolist(), "probabilities": control.tolist(), "squared_loss": squared_control, "exact_goal_values": True},
        "explicit_categorical_candidate": {"forward": forward_density.tolist(), "backward": backward_density.tolist(), "probabilities": categorical.tolist(), "mean_kl": regret, "kl_rational_interval": interval_report(regret_interval), "policy_left_0_right_1": density_policy.tolist(), "return_from_0": float(policy_value(density_policy)[0])},
        "wrong_or_tied_order_kl_bounds_at_global_optimum": bounds,
        "bound_rational_intervals": {state: interval_report(value) for state, value in certified_bounds.items()},
        "certified_decision_states": [0, 2],
        "certified_start_state": 0,
        "optimal_unnormalized_return_from_0": optimal_return,
        "true_normalized_start_goal_gap": float(true_gap),
        "certificate_scope": "Exact-successor approximation; unrestricted backward features; exact goal reward integration. Not a global TD or all-task-capacity theorem.",
        "all_assertions_passed": True,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
