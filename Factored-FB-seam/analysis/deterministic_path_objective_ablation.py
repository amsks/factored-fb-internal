"""Small exact-target ablation: square probability error within rank-two softmax.

This is an oracle projection, not temporal-difference training. All eight
state-action forward rows are independent; next-state candidate weights are
uniform. Retains every restart and compares existing feasible/reference fits.
"""

import json
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import check_grad, minimize
from scipy.special import logsumexp


ROOT = Path(__file__).resolve().parents[1]
GAMMA = 0.8
NEXT = np.array([[0, 1], [0, 2], [1, 3], [2, 3]])
N = np.array([[43, 22, 12, 8], [22, 33, 18, 12],
              [12, 18, 33, 22], [8, 12, 22, 43]], dtype=float) / 85
P = N[NEXT.ravel()]
M, D, C = 8, 2, 4


def unpack(x):
    return x[:M * D].reshape(M, D), x[M * D:].reshape(C, D)


def probabilities(f, b):
    logits = f @ b.T
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def objective(x):
    f, b = unpack(x)
    q = probabilities(f, b)
    residual = q - P
    probability_gradient = 2 * residual / M
    logit_gradient = q * (
        probability_gradient - (probability_gradient * q).sum(axis=1, keepdims=True)
    )
    gradient = np.r_[(logit_gradient @ b).ravel(), (logit_gradient.T @ f).ravel()]
    return float(np.mean(np.sum(residual ** 2, axis=1))), gradient


def describe(q):
    goal_values = q[:, 1].reshape(4, 2)
    actions = goal_values.argmax(axis=1)
    next_states = NEXT[np.arange(4), actions]
    policy_transition = np.eye(4)[next_states]
    reward = (next_states == 1).astype(float)
    values = np.linalg.solve(np.eye(4) - GAMMA * policy_transition, reward)
    return {
        "mean_row_squared_probability_error": float(np.mean(np.sum((P - q) ** 2, axis=1))),
        "uniform_reference_weighted_error_E": float(4 * np.mean(np.sum((P - q) ** 2, axis=1))),
        "mean_KL": float(np.mean(np.sum(P * np.log(P / q), axis=1))),
        "normalized_right_minus_left_goal_gaps": (goal_values[:, 1] - goal_values[:, 0]).tolist(),
        "greedy_actions_0_left_1_right": actions.tolist(),
        "greedy_start_return": float(values[0]),
        "normalized_prediction_rank": int(np.linalg.matrix_rank(q, tol=1e-8)),
        "prediction": q.tolist(),
    }


def gauge_and_rounded_candidate_checks(rational_f_sa, rational_b, first_run):
    # Affine changes of the backward feature coordinates preserve softmax,
    # including translations, whose only effect is a row-constant logit shift.
    a = np.sqrt(2 / 5)
    rational_f = rational_f_sa[[0, 1, 3, 5]]
    b_new = a * np.array([[-2, 1], [-1, -2], [1, -2], [2, 1]])
    f_new = rational_f @ np.diag([1 / (2 * a), 2 / (3 * a)])
    old_q = probabilities(rational_f, rational_b)
    new_q = probabilities(f_new, b_new)
    np.testing.assert_allclose(old_q, new_q, atol=2e-16, rtol=0)
    np.testing.assert_allclose(np.sum(b_new ** 2, axis=1), 2, atol=5e-16, rtol=0)
    np.testing.assert_allclose(b_new.T @ b_new / 4, np.eye(2), atol=5e-16, rtol=0)
    gauge = {
        "B_formula": "sqrt(2/5)*[[-2,1],[-1,-2],[1,-2],[2,1]]",
        "F_formula": "F_original @ diag(1/(2*a),2/(3*a)), a=sqrt(2/5)",
        "logit_change": "Subtracts F_original[:,1]/3 from each complete row; softmax unchanged.",
        "F": f_new.tolist(), "B": b_new.tolist(),
        "row_squared_norms": np.sum(b_new ** 2, axis=1).tolist(),
        "second_moment": (b_new.T @ b_new / 4).tolist(),
        "maximum_probability_difference": float(np.max(np.abs(old_q - new_q))),
        "probability_rank": int(np.linalg.matrix_rank(old_q)),
        "probability_singular_values": np.linalg.svd(old_q, compute_uv=False).tolist(),
    }
    rounded_f = np.round(np.asarray(first_run["F"]), 4)
    rounded_b = np.round(np.asarray(first_run["B"]), 4)
    rounded_q = probabilities(rounded_f, rounded_b)
    rounded = {
        "decimal_places": 4, "F": rounded_f.tolist(), "B": rounded_b.tolist(),
        "uniform_reference_weighted_error_E": float(4 * np.mean(np.sum((rounded_q - P) ** 2, axis=1))),
        "eight_independent_row_wrong_start_simplex_E_lower_bound": (11 / 85) ** 2 / 3,
        "normalized_start_gap": float(rounded_q[1, 1] - rounded_q[0, 1]),
        "scope": "Floating evaluation of explicit rounded feasible factors; an interval proof is a separate step.",
    }
    return gauge, rounded


def main():
    existing = next(item for item in json.loads(
        (ROOT / "analysis/deterministic_didactic/path_exact_fit.json").read_text()
    ) if item["gamma"] == GAMMA)
    rational_f = np.array([[-.8, .1], [-.4, -.23], [.4, -.23], [.8, .1]])[NEXT.ravel()]
    rational_b = np.array([[-1, 1], [-.5, -1], [.5, -1], [1, 1]])
    centered_log = np.log(P) - np.log(P).mean(axis=1, keepdims=True)
    u, s, vt = np.linalg.svd(centered_log, full_matrices=False)
    starts = [
        ("centered_log_svd", u[:, :D] * np.sqrt(s[:D]), vt[:D].T * np.sqrt(s[:D])),
        ("existing_ce_fit", np.asarray(existing["density_F"]), np.asarray(existing["density_B"])),
        ("rational_ce_candidate", rational_f, rational_b),
    ]
    rng = np.random.default_rng(20260911)
    for seed in range(5):
        starts.append((f"random_{seed}", rng.normal(size=(M, D)), rng.normal(size=(C, D))))

    test_x = rng.normal(size=M * D + C * D)
    gradient_error = check_grad(lambda x: objective(x)[0], lambda x: objective(x)[1], test_x)
    assert gradient_error < 1e-7, gradient_error
    runs = []
    for name, f, b in starts:
        opt = minimize(objective, np.r_[f.ravel(), b.ravel()], jac=True, method="L-BFGS-B",
                       options={"maxiter": 2000, "ftol": 1e-15, "gtol": 1e-10, "maxls": 40})
        f, b = unpack(opt.x)
        q = probabilities(f, b)
        record = {"initialization": name, "success": bool(opt.success), "message": str(opt.message),
                  "iterations": int(opt.nit), "function_evaluations": int(opt.nfev),
                  "final_gradient_norm": float(np.linalg.norm(opt.jac)),
                  "F": f.tolist(), "B": b.tolist(), **describe(q)}
        runs.append(record)
        print(name, "MSE", record["mean_row_squared_probability_error"],
              "gap", record["normalized_right_minus_left_goal_gaps"][0],
              "return", record["greedy_start_return"], "grad", record["final_gradient_norm"])

    spherical = json.loads((ROOT / "PAPER/ICLR/figures/didactic/deterministic_path_spherical/verification_data.json").read_text())
    comparisons = {
        "exact_target": describe(P),
        "unrestricted_linear_rank2_global_SVD": describe(np.asarray(existing["H"])),
        "existing_softmax_rank2_CE_numerical_fit": describe(np.asarray(existing["density_Q"])),
        "softmax_rank2_rational_CE_candidate": describe(probabilities(rational_f, rational_b)),
        "spherical_FB_feasible_candidate_not_global_value": describe(
            np.asarray(spherical["candidate_normalized_successor_prediction"])[NEXT.ravel()]),
    }
    report = {
        "scope": "Exact-target numerical fits only; no TD, actor, sphere or orthogonality penalty in softmax ablation.",
        "gamma": GAMMA, "rank_of_logits": D, "row_weights": [1 / M] * M,
        "reference": [1 / C] * C, "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "optimizer": "L-BFGS-B; maxiter 2000; ftol 1e-15; gtol 1e-10; maxls 40",
        "rng_seed": 20260911, "analytic_gradient_finite_difference_error": float(gradient_error),
        "comparisons": comparisons, "all_runs": runs,
        "best_run_initialization": min(runs, key=lambda r: r["mean_row_squared_probability_error"])["initialization"],
    }
    gauge, rounded = gauge_and_rounded_candidate_checks(rational_f, rational_b, runs[0])
    report["density_CE_candidate_spherical_identity_second_moment_gauge"] = gauge
    report["rounded_squared_softmax_candidate"] = rounded
    (ROOT / "analysis/deterministic_path_objective_ablation.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
