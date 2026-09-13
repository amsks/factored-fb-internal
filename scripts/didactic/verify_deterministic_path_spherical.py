#!/usr/bin/env python3
"""Exact rational certificate for spherical FB on the four-state path.

The objective, above the common regularizer minimum and divided by the
regression scale, is ||N-N Pi_B||_F^2 + 40 delta^2. This matches two identical
exact-target FB heads with gamma=.8 and ortho_coef=1000, or one head with
ortho_coef=500. See docs/notes/2026-09-11-deterministic-path-spherical-counterexample.md.
This is not a global TD theorem.
"""

from fractions import Fraction as Q
from itertools import combinations
import json

import numpy as np


def matmul(left, right):
    return [[sum(x * y for x, y in zip(row, column)) for column in zip(*right)] for row in left]


def transpose(matrix):
    return [list(row) for row in zip(*matrix)]


def trace(matrix):
    return sum(matrix[i][i] for i in range(len(matrix)))


def determinant(matrix):
    if len(matrix) == 1:
        return matrix[0][0]
    return sum((-1) ** j * matrix[0][j] * determinant([row[:j] + row[j + 1:] for row in matrix[1:]]) for j in range(len(matrix)))


def record(value):
    return {"exact": str(value), "decimal": float(value)}


def verify_reduced_td_potential(N, D, projector):
    """Certify the potential after exactly equilibrating the forward TD rows.

    This verifies the algebraic certificate for tr(N)-tr(N Pi)+200 delta^2.
    The source-equivalent gradient derivation is explained in the proof note;
    finite-time simultaneous updates and target lag are outside this result.
    """
    candidate = trace(N) - trace(matmul(N, projector))
    upper = Q(183, 500)
    assert candidate < upper
    assert Q(200) * Q(1, 20)**2 > upper
    mu = Q(5, 4)
    z = [-Q(9, 100), Q(0), Q(13, 50), Q(0)]
    beta = Q(3, 8)
    vector = [Q(239, 1000), Q(435, 1000), Q(559, 1000), Q(409, 1000)]
    residual = [[beta * int(i == j) + vector[i] * vector[j] - N[i][j] - mu * D[i][j] - z[i] * int(i == j) for j in range(4)] for i in range(4)]
    minors = {}
    for size in range(1, 5):
        for indices in combinations(range(4), size):
            minor = determinant([[residual[i][j] for j in indices] for i in indices])
            assert minor > 0
            minors[",".join(map(str, indices))] = record(minor)
    intercept = trace(N) - 2 * beta - sum(x * x for x in vector) + sum(z) / 2 - sum(abs(x) for x in z) / 38
    gap_upper = (upper - intercept) / mu
    assert intercept > upper
    assert gap_upper < -Q(17, 1000)

    # This B also satisfies forward TD stationarity after solving F exactly.
    # Its predicted goal values differ from the exact fixed-policy values,
    # but its greedy policy is optimal from every state.
    control_backward = [[Q(1), Q(-1)], [Q(1), Q(1)], [Q(1), Q(1)], [Q(1), Q(-1)]]
    assert matmul(transpose(control_backward), control_backward) == [[Q(4), Q(0)], [Q(0), Q(4)]]
    control_projector = [[x / 4 for x in row] for row in matmul(control_backward, transpose(control_backward))]
    control_prediction = matmul(N, control_projector)
    goal_values = [row[1] for row in control_prediction]
    assert goal_values == [Q(1, 5), Q(3, 10), Q(3, 10), Q(1, 5)]
    assert goal_values[1] > goal_values[0]
    assert goal_values[1] > goal_values[3]
    assert goal_values[2] > goal_values[3]
    control_value = trace(N) - trace(matmul(N, control_projector))
    assert control_value > candidate
    return {
        "objective": "tr(N)-tr(N Pi_B)+200 delta^2; expected FB with forward rows at TD equilibrium and instantaneous targets",
        "candidate_objective": record(candidate),
        "upper_used": record(upper),
        "dual_mu": record(mu),
        "dual_z": [record(x) for x in z],
        "dual_beta": record(beta),
        "dual_vector": [record(x) for x in vector],
        "positive_principal_minors": minors,
        "error_lower_intercept": record(intercept),
        "every_global_minimizer_normalized_goal_gap_upper": record(gap_upper),
        "control_sufficient_equilibrated_goal_predictions": [record(x) for x in goal_values],
        "control_sufficient_equilibrated_potential": record(control_value),
        "scope": "Global reduced-potential result. Requires exactly equilibrated forward TD rows and instantaneous targets; no convergence claim for coupled finite-time TD, Adam, target lag, or the Density TD objective.",
    }


def verify():
    numerators = [[43, 22, 12, 8], [22, 33, 18, 12], [12, 18, 33, 22], [8, 12, 22, 43]]
    N = [[Q(value, 85) for value in row] for row in numerators]
    A = matmul(N, N)
    # B=sqrt(2)*U. Each U row is unit length; U^T U=2I gives C=I.
    U = [[Q(1), Q(0)], [Q(255, 257), Q(32, 257)], [Q(0), Q(1)], [-Q(32, 257), Q(255, 257)]]
    assert all(sum(x * x for x in row) == 1 for row in U)
    assert matmul(transpose(U), U) == [[Q(2), Q(0)], [Q(0), Q(2)]]
    projector = [[value / 2 for value in row] for row in matmul(U, transpose(U))]
    assert matmul(projector, projector) == projector
    candidate = matmul(N, projector)
    candidate_error = trace(A) - trace(matmul(A, projector))
    assert candidate_error == sum((N[i][j] - candidate[i][j]) ** 2 for i in range(4) for j in range(4))
    upper = Q(81, 1000)
    assert 0 < candidate_error < upper
    assert all(value > 0 for row in candidate for value in row)
    candidate_gap = candidate[1][1] - candidate[0][1]
    assert candidate_gap < 0

    # D satisfies tr(D Pi)=(N Pi)[1,1]-(N Pi)[0,1].
    difference = [N[1][j] - N[0][j] for j in range(4)]
    D = [[(difference[i] * int(j == 1) + int(i == 1) * difference[j]) / 2 for j in range(4)] for i in range(4)]
    assert trace(matmul(D, projector)) == candidate_gap
    mu = Q(43, 50)
    z = [-Q(1, 10), -Q(1, 20), Q(17, 100), Q(0)]
    beta = Q(13, 100)
    vector = [Q(357, 1000), Q(469, 1000), Q(573, 1000), Q(478, 1000)]
    H = [[A[i][j] + mu * D[i][j] + z[i] * int(i == j) for j in range(4)] for i in range(4)]
    residual = [[beta * int(i == j) + vector[i] * vector[j] - H[i][j] for j in range(4)] for i in range(4)]
    assert residual == transpose(residual)
    minors = {}
    for size in range(1, 5):
        for indices in combinations(range(4), size):
            minor = determinant([[residual[i][j] for j in indices] for i in indices])
            assert minor > 0
            minors[",".join(map(str, indices))] = record(minor)

    # Any global minimizer has J<=candidate_error<.081. Since J=E+40 delta^2,
    # it has delta<.05; every leverage Pi_ii differs from .5 by less than 1/38.
    coefficient = Q(40)
    delta_limit = Q(1, 20)
    assert coefficient * delta_limit**2 > upper
    diagonal_deviation = delta_limit / (2 * (1 - delta_limit))
    assert diagonal_deviation == Q(1, 38)
    base = trace(A) - 2 * beta - sum(x * x for x in vector) + sum(z) / 2
    lower_intercept = base - sum(abs(x) for x in z) * diagonal_deviation
    # The PSD matrix gives E >= lower_intercept + mu * goal_gap.
    # Therefore every global minimizer has this strictly negative upper gap.
    gap_upper = (upper - lower_intercept) / mu
    assert lower_intercept > upper
    assert gap_upper < -Q(28, 1000)

    # A spherical, whitened rank-two model nevertheless predicts this goal's
    # values exactly and has positive, normalized probability rows.
    backward_control = [[Q(1), Q(1)], [Q(1), Q(1)], [Q(1), Q(-1)], [Q(1), Q(-1)]]
    forward_control = [[Q(1), 4 * row[1] - 1] for row in N]
    control = [[x / 4 for x in row] for row in matmul(forward_control, transpose(backward_control))]
    assert all(sum(x * x for x in row) == 2 for row in backward_control)
    assert matmul(transpose(backward_control), backward_control) == [[Q(4), Q(0)], [Q(0), Q(4)]]
    assert all(sum(row) == 1 and all(x > 0 for x in row) for row in control)
    assert [row[1] for row in control] == [row[1] for row in N]
    control_error = sum((N[i][j] - control[i][j]) ** 2 for i in range(4) for j in range(4))
    assert control_error > candidate_error

    return {
        "objective": "E + 40 delta^2; E=||N-N Pi_B||_F^2, C=B^T B/4 has eigenvalues 1+delta and 1-delta",
        "matching_fb_settings": {"discount": 0.8, "dimension": 2, "backward_norm": "sqrt(2)", "two_head_ortho_coefficient": 1000, "equivalent_one_head_ortho_coefficient": 500},
        "candidate_U_with_B_sqrt2_U": [[str(x) for x in row] for row in U],
        "candidate_error_and_objective": record(candidate_error),
        "candidate_goal_gap": record(candidate_gap),
        "candidate_normalized_successor_prediction": [[float(x) for x in row] for row in candidate],
        "candidate_row_masses": [float(sum(row)) for row in candidate],
        "objective_upper_used": record(upper),
        "dual_mu": record(mu),
        "dual_z": [record(x) for x in z],
        "dual_beta": record(beta),
        "dual_vector": [record(x) for x in vector],
        "positive_principal_minors_of_dual_residual": minors,
        "dual_residual_minimum_eigenvalue_numerical_check": float(np.linalg.eigvalsh(np.array(residual, dtype=float)).min()),
        "global_minimizer_delta_strict_upper": record(delta_limit),
        "leverage_deviation_strict_upper": record(diagonal_deviation),
        "error_lower_intercept": record(lower_intercept),
        "every_global_minimizer_normalized_goal_gap_upper": record(gap_upper),
        "spherical_control_witness_error": record(control_error),
        "spherical_control_witness_probabilities": [[float(x) for x in row] for row in control],
        "equilibrated_forward_td_potential": verify_reduced_td_potential(N, D, projector),
        "scope": "Global exact-target approximation theorem, actual soft orthogonality weight, spherical FB, fixed continuation and exact reward integration. Not a global TD theorem or a lambda=1 result.",
        "all_assertions_passed": True,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
