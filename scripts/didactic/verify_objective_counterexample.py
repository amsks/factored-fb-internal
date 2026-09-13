#!/usr/bin/env python3
"""Verify a four-outcome, rank-two successor-model counterexample.

Run directly with Python and NumPy. This calculation uses exact outcome targets;
it performs no training, parameter search, or interaction with an environment.
Rows are (context, action) = (++), (+-), (-+), (--). The natural outcome
reference and the row distribution are both uniform. Reward on column 2 makes
action -1 optimal in each context.

The least-squares optimum reverses both decisions. A feasible categorical
predictor has KL regret below a universal bound for any wrong/tied decision.
Both backward-feature constructions have row norm sqrt(2) and second moment I.
A separate feasible FB predictor preserves action differences for every reward.

The two models have equal feature dimension but different probability classes:
the categorical comparator has probability rank four and logit rank two.
"""

import json

import numpy as np


ATOL = 1e-12


def mean_kl(target, prediction):
    return float(np.mean(np.sum(target * np.log(target / prediction), axis=1)))


def bernoulli_kl(p, q):
    return float(p * np.log(p / q) + (1 - p) * (np.log1p(-p) - np.log1p(-q)))


def check_probabilities(probabilities):
    assert np.all(probabilities > 0)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=ATOL, rtol=0)


def check_backward_features(backward):
    np.testing.assert_allclose(np.sum(backward**2, axis=1), 2, atol=ATOL, rtol=0)
    np.testing.assert_allclose(backward.T @ backward / 4, np.eye(2), atol=ATOL, rtol=0)


def action_difference(matrix):
    return matrix[[0, 2]] - matrix[[1, 3]]


def verify():
    context = np.array([1.0, 1.0, -1.0, -1.0])
    action = np.array([1.0, -1.0, 1.0, -1.0])
    e0 = np.ones(4) / 2
    e1 = np.array([1.0, -1.0, 1.0, -1.0]) / 2
    e3 = np.array([1.0, -1.0, -1.0, 1.0]) / 2
    c1 = 0.1 * context + 0.01 * action
    c3 = -0.005 * context + 0.05 * action
    target = 0.25 + c1[:, None] * e1 + c3[:, None] * e3
    check_probabilities(target)
    np.testing.assert_allclose(target.mean(axis=0), 0.25, atol=ATOL, rtol=0)

    # Uniform row/reference weights make the whitened density matrix equal P.
    left, singular, right = np.linalg.svd(target, full_matrices=False)
    least_squares = (left[:, :2] * singular[:2]) @ right[:2]
    assert singular[1] > singular[2] > ATOL
    assert singular[3] < ATOL
    np.testing.assert_allclose(least_squares, 0.25 + c1[:, None] * e1, atol=ATOL, rtol=0)
    forward_fb = np.column_stack((np.ones(4), 2 * c1))
    backward_fb = 2 * np.column_stack((e0, e1))
    check_backward_features(backward_fb)
    np.testing.assert_allclose(0.25 * forward_fb @ backward_fb.T, least_squares, atol=ATOL, rtol=0)
    check_probabilities(least_squares)

    # Independent signed outcome features give an explicit softmax comparator.
    backward_density = 2 * np.column_stack((e1, e3))
    forward_density = np.column_stack((np.arctanh(2 * c1), np.arctanh(2 * c3)))
    logits = forward_density @ backward_density.T
    categorical = np.exp(logits - logits.max(axis=1, keepdims=True))
    categorical /= categorical.sum(axis=1, keepdims=True)
    product = 0.25 * (1 + 2 * c1[:, None] * (2 * e1)) * (1 + 2 * c3[:, None] * (2 * e3))
    np.testing.assert_allclose(categorical, product, atol=ATOL, rtol=0)
    check_probabilities(categorical)
    check_backward_features(backward_density)
    assert np.linalg.matrix_rank(logits, tol=ATOL) == 2
    assert np.linalg.matrix_rank(categorical, tol=ATOL) == 4

    # Coarsen each row into goal/non-goal. Among wrong/tied orderings the
    # minimum pairwise Bernoulli KL occurs when both predictions equal the mean.
    bounds = []
    for i, j in [(0, 1), (2, 3)]:
        low, high = target[[i, j], 2]
        assert high > low
        mean = (low + high) / 2
        bounds.append((bernoulli_kl(low, mean) + bernoulli_kl(high, mean)) / 4)
    regret = mean_kl(target, categorical)
    assert regret < min(bounds)
    np.testing.assert_allclose(action_difference(target)[:, 2], -0.04, atol=ATOL, rtol=0)
    np.testing.assert_allclose(action_difference(least_squares)[:, 2], 0.01, atol=ATOL, rtol=0)
    assert np.all(action_difference(categorical)[:, 2] < 0)

    # A rank-two FB model can preserve every reward's action contrast, despite
    # the loss optimum failing to preserve even the column-2 goal decision.
    action_direction = 0.01 * e1 + 0.05 * e3
    sigma = np.sqrt(np.mean(action_direction**2))
    b1 = action_direction / sigma
    b2 = np.sqrt(2 - b1**2)
    backward_control = np.column_stack((b1, b2))
    forward_control = np.column_stack((4 * action * sigma, np.full(4, 1 / b2.mean())))
    control = 0.25 * forward_control @ backward_control.T
    check_probabilities(control)
    check_backward_features(backward_control)
    assert np.linalg.matrix_rank(control, tol=ATOL) == 2
    np.testing.assert_allclose(action_difference(control), action_difference(target), atol=ATOL, rtol=0)

    # Centering removes the softmax row-offset freedom. Use an explicit rank
    # tolerance because subtractive roundoff can inflate the default rank.
    centered_log = np.log(target) - np.log(target).mean(axis=1, keepdims=True)
    log_singular = np.linalg.svd(centered_log, compute_uv=False)
    assert log_singular[2] > 0.08
    assert log_singular[3] < ATOL

    def report_model(prediction, forward, backward):
        return {
            "probabilities": prediction.tolist(),
            "forward": forward.tolist(),
            "backward": backward.tolist(),
            "probability_rank": int(np.linalg.matrix_rank(prediction, tol=ATOL)),
            "weighted_squared_loss": float(np.sum((target - prediction) ** 2)),
            "mean_kl_regret": mean_kl(target, prediction),
            "goal_action_gaps": action_difference(prediction)[:, 2].tolist(),
        }

    return {
        "target_probabilities": target.tolist(),
        "natural_reference": target.mean(axis=0).tolist(),
        "target_singular_values": singular.tolist(),
        "target_centered_log_singular_values": log_singular.tolist(),
        "true_action_difference_vectors": action_difference(target).tolist(),
        "wrong_or_tied_decision_kl_bounds": bounds,
        "least_squares_optimum": report_model(least_squares, forward_fb, backward_fb),
        "categorical_comparator": report_model(categorical, forward_density, backward_density),
        "all_reward_control_comparator": report_model(control, forward_control, backward_control),
        "all_assertions_passed": True,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
