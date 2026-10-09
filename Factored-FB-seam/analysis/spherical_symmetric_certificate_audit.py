"""Independent exact-rational and finite-difference audit of the path certificate.

Run with the existing Factored-FB venv. This audits exact-target minimizers and
the forward-equilibrated backward TD potential, not joint-TD global convergence.
"""
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path
import json
import numpy as np

HERE = Path(__file__).resolve().parent
N = [[F(x, 85) for x in row] for row in
     [[43, 22, 12, 8], [22, 33, 18, 12], [12, 18, 33, 22], [8, 12, 22, 43]]]
A = [[sum(N[i][k] * N[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
diff = [N[1][i] - N[0][i] for i in range(4)]
D = [[(diff[i] * (j == 1) + diff[j] * (i == 1)) / 2
      for j in range(4)] for i in range(4)]
z = list(map(F, ['-.1', '-.05', '.17', '0']))
v = list(map(F, ['.357', '.469', '.573', '.478']))
beta, eta = F('.13'), F('.86')
H = [[A[i][j] + eta * D[i][j] + z[i] * (i == j)
      for j in range(4)] for i in range(4)]
M = [[beta * (i == j) + v[i] * v[j] - H[i][j]
      for j in range(4)] for i in range(4)]


def det(matrix):
    if len(matrix) == 1:
        return matrix[0][0]
    return sum((-1) ** j * matrix[0][j] * det(
        [row[:j] + row[j + 1:] for row in matrix[1:]]) for j in range(len(matrix)))


minors = {}
for size in range(1, 5):
    for subset in combinations(range(4), size):
        minor = det([[M[i][j] for j in subset] for i in subset])
        assert minor > 0
        minors[str(subset)] = str(minor)

U = [[F(1), F(0)], [F(255, 257), F(32, 257)],
     [F(0), F(1)], [-F(32, 257), F(255, 257)]]
Pi = [[sum(U[i][k] * U[j][k] for k in range(2)) / 2
       for j in range(4)] for i in range(4)]
for row in U:
    assert sum(x * x for x in row) == 1
for i in range(2):
    for j in range(2):
        assert sum(U[k][i] * U[k][j] for k in range(4)) == 2 * (i == j)
trace_a = sum(A[i][i] for i in range(4))
candidate = trace_a - sum(A[i][j] * Pi[j][i] for i in range(4) for j in range(4))
assert candidate == F(149734, 1856825) < F('.081')
baseline = trace_a - 2 * beta - sum(x * x for x in v) + sum(z) / 2 - sum(abs(x) for x in z) / 38
assert baseline > F('.081')
gap_bound = (F('.081') - baseline) / eta

# A positive, same-capacity, same-sphere, same-second-moment control witness.
nf = np.array(N, float)
b_control = np.array([[1, -1], [1, 1], [1, -1], [1, 1]], float)
f_control = np.column_stack([np.ones(4), 4 * nf[:, 1] - 1])
q_control = f_control @ b_control.T / 4
np.testing.assert_allclose(np.linalg.norm(b_control, axis=1), np.sqrt(2))
np.testing.assert_allclose(b_control.T @ b_control / 4, np.eye(2))
np.testing.assert_allclose(q_control.sum(axis=1), 1)
np.testing.assert_allclose(q_control[:, 1], nf[:, 1])
assert q_control.min() > 0
assert np.linalg.matrix_rank(q_control) == 2

# The actual frozen-target population FB gradient, with forward F solved to its
# TD fixed point, equals the gradient of -.5 tr(Pi G) for one head.
gamma = .8
dest = np.array([[0, 1], [0, 2], [1, 3], [2, 3]])
p = np.eye(4)[dest.ravel()]
t = p.reshape(4, 2, 4).mean(axis=1)
g = np.linalg.inv(np.eye(4) - gamma * t)
rng = np.random.default_rng(814)
b = rng.normal(size=(4, 2))
b *= np.sqrt(2) / np.linalg.norm(b, axis=1, keepdims=True)
c = b.T @ b / 4
ci = np.linalg.inv(c)
projector = b @ np.linalg.inv(b.T @ b) @ b.T
forward = p @ g @ b @ ci
next_forward = p @ t @ g @ b @ ci
frozen_bootstrap = next_forward @ b.T
analytic_gradient = -.25 * (np.eye(4) - projector) @ g @ b @ ci


def frozen_loss(candidate_b):
    residual = forward @ candidate_b.T - gamma * frozen_bootstrap
    return .5 * np.mean(residual ** 2) - np.mean(np.sum(p * residual, axis=1))


def reduced_potential(candidate_b):
    candidate_pi = candidate_b @ np.linalg.inv(candidate_b.T @ candidate_b) @ candidate_b.T
    return -.5 * np.trace(candidate_pi @ g)


def finite_gradient(function, x):
    answer = np.zeros_like(x)
    step = 1e-6
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            perturbation = np.zeros_like(x)
            perturbation[i, j] = step
            answer[i, j] = (function(x + perturbation) - function(x - perturbation)) / (2 * step)
    return answer


frozen_gradient = finite_gradient(frozen_loss, b)
reduced_gradient = finite_gradient(reduced_potential, b)
frozen_error = float(np.max(np.abs(frozen_gradient - analytic_gradient)))
potential_error = float(np.max(np.abs(reduced_gradient - analytic_gradient)))
assert frozen_error < 1e-7
assert potential_error < 1e-7

# With two heads, rho=1/4, mu=1/8 and gamma=.8, exact-target data loss is
# 25 ||N - Nhat||_F^2. Sphere gives tr(C)=2, so the orthogonality penalty
# lambda/2 ||C-I||_F^2 is lambda*delta^2. At lambda=1000 this is E+40delta^2.
exact_data_coefficient = F(2, 2) / (1 - F(4, 5)) ** 2
assert exact_data_coefficient == 25
exact_penalty_coefficient = F(1000) / exact_data_coefficient
assert exact_penalty_coefficient == 40
# The two-head reduced backward potential, after multiplying by (1-gamma),
# is tr(N)-tr(Pi N)+200delta^2; the added tr(N) is constant.
td_penalty_coefficient = (1 - F(4, 5)) * 1000
assert td_penalty_coefficient == 200

report = {
    'all_15_principal_minors_positive': True, 'principal_minors': minors,
    'candidate_squared_error_exact': str(candidate), 'candidate_squared_error': float(candidate),
    'baseline_exact': str(baseline), 'baseline': float(baseline),
    'global_gap_bound_exact': str(gap_bound), 'global_gap_bound': float(gap_bound),
    'stronger_gap_bound_from_exact_candidate': float((candidate - baseline) / eta),
    'capacity_witness_min_probability': float(q_control.min()),
    'exact_target_two_head_data_coefficient': int(exact_data_coefficient),
    'exact_target_normalized_delta_penalty': int(exact_penalty_coefficient),
    'forward_equilibrated_td_delta_penalty': int(td_penalty_coefficient),
    'one_head_frozen_target_gradient_max_error': frozen_error,
    'one_head_reduced_potential_gradient_max_error': potential_error,
    'scope': 'Global exact-target objective certificate. Backward TD potential verified only with forward at its TD fixed point; no joint-TD convergence theorem.'
}

# Independent exact-rational certificate for the reduced TD potential. This
# repeats the minor calculation with N, not N^2, and a separately supplied dual.
td_eta = F(5, 4)
td_z = [F(-9, 100), F(0), F(13, 50), F(0)]
td_beta = F(3, 8)
td_v = [F(x, 1000) for x in (239, 435, 559, 409)]
td_h = [[N[i][j] + td_eta * D[i][j] + td_z[i] * (i == j)
         for j in range(4)] for i in range(4)]
td_m = [[td_beta * (i == j) + td_v[i] * td_v[j] - td_h[i][j]
         for j in range(4)] for i in range(4)]
td_minors = {}
for size in range(1, 5):
    for subset in combinations(range(4), size):
        minor = det([[td_m[i][j] for j in subset] for i in subset])
        assert minor > 0
        td_minors[str(subset)] = str(minor)
trace_n = sum(N[i][i] for i in range(4))
td_candidate = trace_n - sum(N[i][j] * Pi[j][i] for i in range(4) for j in range(4))
assert td_candidate == F(7992, 21845) < F(183, 500)
# A global minimizer has 200*delta^2 <= candidate < .366, so delta < .05.
assert td_candidate / 200 < F(1, 20) ** 2
td_baseline = (trace_n - 2 * td_beta - sum(x * x for x in td_v)
               + sum(td_z) / 2 - sum(abs(x) for x in td_z) / 38)
td_gap_bound = (F(183, 500) - td_baseline) / td_eta
assert td_gap_bound < 0

td_sign = np.array([-1, 1, 1, -1])
td_control_b = np.column_stack([np.ones(4), td_sign])
td_control_pi = td_control_b @ td_control_b.T / 4
td_control_prediction = nf @ td_control_pi
np.testing.assert_allclose(np.linalg.norm(td_control_b, axis=1), np.sqrt(2))
np.testing.assert_allclose(td_control_b.T @ td_control_b / 4, np.eye(2))
np.testing.assert_allclose(td_control_prediction[:, 1], [.2, .3, .3, .2])
td_control_value = np.trace(nf) - np.trace(td_control_pi @ nf)
np.testing.assert_allclose(td_control_value, 10 / 17)
# Forward fixed point: F=P G B (C=I); its normalized predicted occupancy is
# (1-gamma) F B^T /4 = P N Pi. Thus the control witness uses equilibrated F.
td_control_forward = p @ g @ td_control_b
np.testing.assert_allclose((1 - gamma) * td_control_forward @ td_control_b.T / 4,
                           td_control_prediction[dest.ravel()])
td_control_actions = np.argmax(td_control_prediction[dest, 1], axis=1)
assert td_control_actions.tolist() == [1, 1, 0, 0]
report['reduced_td_certificate'] = {
    'all_15_principal_minors_positive': True, 'principal_minors': td_minors,
    'candidate_potential_exact': str(td_candidate), 'candidate_potential': float(td_candidate),
    'baseline_exact': str(td_baseline), 'baseline': float(td_baseline),
    'global_gap_bound_exact': str(td_gap_bound), 'global_gap_bound': float(td_gap_bound),
    'stronger_gap_bound_from_exact_candidate': float((td_candidate - td_baseline) / td_eta),
    'control_witness_goal_column': td_control_prediction[:, 1].tolist(),
    'control_witness_greedy_action_indices': td_control_actions.tolist(),
    'control_witness_potential': float(td_control_value),
    'minimum_numeric_certificate_eigenvalue': float(np.linalg.eigvalsh(np.array(td_m, float)).min()),
    'scope': 'Global minimizers of the actual expected backward semi-gradient potential with forward factors at their TD fixed point. Does not establish global convergence of simultaneous F/B updates or Polyak-target training.'
}
output = HERE / 'spherical_symmetric_certificate_audit.json'
output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
