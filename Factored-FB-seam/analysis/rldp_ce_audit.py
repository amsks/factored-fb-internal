"""Small CPU diagnostics for the 2026-09-11 RLDP categorical retrofit audit.

Read-only with respect to production code; no optimization or training. Run:
  JAX_PLATFORMS=cpu ../Factored-FB/.venv/bin/python \
    /tmp/carry-drop-python.py analysis/rldp_ce_audit.py
"""
from pathlib import Path
import dataclasses
import hashlib
import json
import runpy
import sys

EXPECT_FIXED = "--fixed" in sys.argv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
fx = runpy.run_path(str(ROOT / "tests/impls/test_rldp_density.py"))
RLDPCritic = fx["RLDPCritic"]
critic, params, target = fx["_build"](temp=1.0, ortho_coef=1.0)
batch = fx["_batch"]()
cond, boot = fx["_cond_boot"]()
target = {**target, "backward": jax.tree.map(lambda x: -x, target["backward"])}


def norm(tree):
    return float(np.sqrt(sum(float(jnp.vdot(x, x)) for x in jax.tree.leaves(tree))))


def plain_loss(c, p=params, t=target):
    return RLDPCritic.stage_loss(c, "fb", p, t, batch, cond, boot)[0]


def density_loss(c, p=params, t=target):
    return c.stage_loss("fb", p, t, batch, cond, boot)[0]


out = {"scope": "CPU value/gradient checks at fixed initialized arrays; no training"}
out["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (ROOT / "impls/critics/rldp.py", ROOT / "impls/critics/rldp_density.py")}
for flag in ("fb_target_online_b", "ortho_inert"):
    a, b = (dataclasses.replace(critic, **{flag: v}) for v in (False, True))
    out[flag] = {
        "plain_loss_false": float(plain_loss(a)),
        "plain_loss_true": float(plain_loss(b)),
        "density_loss_false": float(density_loss(a)),
        "density_loss_true": float(density_loss(b)),
    }

ce_grad = jax.grad(lambda p: critic.stage_loss("fb", p, target, batch, cond, boot)[1]["density_ce"])(params)
out["ce_gradient_norms"] = {k: norm(v) for k, v in ce_grad.items()}
joint_plain_grad = jax.grad(lambda p: plain_loss(critic, p))(params)
joint_density_grad = jax.grad(lambda p: density_loss(critic, p))(params)
out["joint_representation_gradient_parity"] = {
    k: norm(jax.tree.map(lambda a, b: a - b, joint_plain_grad[k], joint_density_grad[k]))
    for k in ("backward", "predictor")
}
terminal_batch = {**batch, "terminals": jnp.array([0., 1., 0., 1., 0., 0.])}
manual_t, manual_ce = fx["_np_target_and_ce"](critic, params, target, terminal_batch, cond, boot)
_, terminal_info = critic.stage_loss("fb", params, target, terminal_batch, cond, boot)
target_gradient = jax.grad(lambda t: density_loss(critic, t=t))(target)
out["categorical_target"] = {
    "manual_ce": float(manual_ce), "actual_ce": float(terminal_info["density_ce"]),
    "max_row_sum_error": float(np.max(np.abs(manual_t.sum(-1) - 1))),
    "minimum_target_entry": float(manual_t.min()),
    "terminal_rows_onehot_error": float(np.max(np.abs(manual_t[:, [1, 3], :] - np.eye(len(cond["z"]))[[1, 3]][None]))),
    "all_target_parameter_gradient_norm": norm(target_gradient),
}

mr = dataclasses.replace(critic, phase="mr", ortho_inert=True)
gp = jax.grad(lambda p: plain_loss(mr, p))(params)
gd = jax.grad(lambda p: density_loss(mr, p))(params)
out["phase_mr"] = {
    "active_groups": list(mr.stage_groups("fb")),
    "train_actor": mr.TRAIN_ACTOR,
    "backward_gradient_difference": norm(jax.tree.map(lambda a, b: a - b, gp["backward"], gd["backward"])),
    "predictor_gradient_difference": norm(jax.tree.map(lambda a, b: a - b, gp["predictor"], gd["predictor"])),
}
out["phase_fb"] = {"active_groups": list(dataclasses.replace(critic, phase="fb").stage_groups("fb"))}

pool = critic._B(params["backward"], batch["next_observations"])
qcond = {**cond, "_pool_B": pool}
qs = []
for k in (0.0, 0.5):
    c = dataclasses.replace(critic, actor_pessimism=k, actor_q_center=False)
    qp, hp = RLDPCritic.q_fn(c, params, batch["observations"], qcond, batch["actions"])
    qd, hd = c.q_fn(params, batch["observations"], qcond, batch["actions"])
    qs.append((qp, qd, hp, hd))
out["actor_pessimism"] = {
    "plain_max_q_change": float(jnp.max(jnp.abs(qs[1][0] - qs[0][0]))),
    "density_max_q_change": float(jnp.max(jnp.abs(qs[1][1] - qs[0][1]))),
    "density_max_head_disagreement": float(jnp.max(jnp.abs(qs[0][3][0] - qs[0][3][1]))),
}

q, h = critic.q_fn(params, batch["observations"], qcond, batch["actions"])
_, raw_heads = dataclasses.replace(critic, actor_q_center=False).q_fn(params, batch["observations"], qcond, batch["actions"])
out["flowbc_denominator"] = {
    "centered_head_mean_abs": float(jnp.abs(h).mean()),
    "raw_heads_mean_abs": float(jnp.abs(raw_heads).mean()),
}
# Identical rows/tasks/actions are allowed; the categorical actor's centered scale is zero.
same_obs = jnp.repeat(batch["observations"][:1], len(q), axis=0)
same_actions = jnp.repeat(batch["actions"][:1], len(q), axis=0)
same_cond = {**qcond, "z": jnp.repeat(cond["z"][:1], len(q), axis=0)}
q_same, h_same = critic.q_fn(params, same_obs, same_cond, same_actions)
out["flowbc_denominator"]["identical_rows_centered_abs"] = float(jnp.abs(h_same).mean())
out["flowbc_denominator"]["identical_rows_q_abs"] = float(jnp.abs(q_same).mean())

# Default inference z=E[B r] is not the linear reward coefficient unless E[BB^T]=I.
# The exact successor-measure least-squares head supplies C^{-1} implicitly; the
# categorical reward readout B z does not. This is a model-mismatch example, not a
# claim about learned B's empirical covariance or the cause of observed run results.
B = np.sqrt(2) * np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
weights = np.array([0.7, 0.1, 0.1, 0.1])
C = (B.T * weights) @ B
coefficient = np.array([1.0, 2.0])
reward = B @ coefficient
z = (weights * reward) @ B
proxy = B @ z
out["nonwhite_reward_inference_counterexample"] = {
    "B_row_norms": np.linalg.norm(B, axis=1).tolist(),
    "rho": weights.tolist(), "second_moment": C.tolist(),
    "linear_reward_coefficient": coefficient.tolist(), "inferred_z_before_unit_projection": z.tolist(),
    "true_reward_gap_outcome2_minus0": float(reward[2] - reward[0]),
    "density_proxy_gap_outcome2_minus0": float(proxy[2] - proxy[0]),
    "covariance_corrected_gap": float((B @ np.linalg.solve(C, z))[2] - (B @ np.linalg.solve(C, z))[0]),
}

paths = [ROOT / f"exp/{env}-play-v0_{name}_flowbc_seed_0/config.json"
         for env in ("cube-single", "cube-double") for name in ("rldp", "rldp_density")]
out["resolved_current_configs"] = {}
keys = {"batch_size", "discount", "z_dim", "num_parallel", "ortho_coef", "train_goal_ratio", "norm_z", "fb_pessimism_penalty", "backward", "backward_input", "dataset_class", "temp", "actor_temp", "actor_q_center", "actor"}
for path in paths:
    d = json.loads(path.read_text()); c = d.get("config", d)
    out["resolved_current_configs"][str(path.relative_to(ROOT))] = {k: v for k, v in c.items() if k in keys or k.startswith("rldp_")}

assert out["fb_target_online_b"]["plain_loss_false"] != out["fb_target_online_b"]["plain_loss_true"]
assert (out["fb_target_online_b"]["density_loss_false"] != out["fb_target_online_b"]["density_loss_true"]) == EXPECT_FIXED
assert out["ortho_inert"]["plain_loss_false"] != out["ortho_inert"]["plain_loss_true"]
assert (out["ortho_inert"]["density_loss_false"] != out["ortho_inert"]["density_loss_true"]) == EXPECT_FIXED
assert out["ce_gradient_norms"]["backward"] == 0.0
assert out["ce_gradient_norms"]["predictor"] == 0.0
assert out["ce_gradient_norms"]["forward"] > 0.0
assert max(out["joint_representation_gradient_parity"].values()) == 0.0
assert abs(out["categorical_target"]["manual_ce"] - out["categorical_target"]["actual_ce"]) < 1e-12
assert out["categorical_target"]["max_row_sum_error"] < 1e-12
assert out["categorical_target"]["terminal_rows_onehot_error"] == 0.0
assert out["categorical_target"]["all_target_parameter_gradient_norm"] == 0.0
assert (out["phase_mr"]["backward_gradient_difference"] < 1e-10) == EXPECT_FIXED
assert out["actor_pessimism"]["plain_max_q_change"] > 0.0
assert (out["actor_pessimism"]["density_max_q_change"] > 0.0) == EXPECT_FIXED
assert out["actor_pessimism"]["density_max_head_disagreement"] > 0.0
out["status"] = "PASS: reproduced ignored flags, gradient routing, and finite-pool readout checks"
dest = Path(__file__).with_suffix(".json")
if EXPECT_FIXED:
    before = json.loads(dest.read_text())
    unchanged = ("ce_gradient_norms", "categorical_target", "joint_representation_gradient_parity", "flowbc_denominator")
    assert all(out[k] == before[k] for k in unchanged)
    assert all(out[k]["density_loss_false"] == before[k]["density_loss_false"]
               for k in ("fb_target_online_b", "ortho_inert"))
    out["default_behavior_exactly_unchanged_from_prepatch"] = True
    out["status"] = "PASS: repaired flags; default values and gradient checks exactly unchanged"
    dest = dest.with_name(dest.stem + "_after_fix.json")
dest.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
