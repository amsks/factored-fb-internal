"""evals/representation_profile.py — why does FB fail at transport?

Four probes (value landscape, z-decoding+sparsity, B-resolution,
coverage) over FB checkpoints. Pure logic here is unit-tested with
stubs; MuJoCo rollouts reuse evals.phase_probe. No training/eval code
is modified. See
docs/superpowers/specs/2026-05-18-fb-representation-failure-profile-design.md
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch


@torch.no_grad()
def _ensemble_q(F: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """F [P, B, z_dim], z [z_dim] or [B, z_dim] -> Q [B] (mean over heads)."""
    if z.dim() == 1:
        z = z.view(1, 1, -1)
    elif z.dim() == 2:
        z = z.unsqueeze(0)
    q = (F * z).sum(-1)            # [P, B]
    return q.mean(0)              # [B]


@torch.no_grad()
def q_values(model, obs: torch.Tensor, action: torch.Tensor,
             z: torch.Tensor) -> np.ndarray:
    F = model.forward_map(obs, z, action)
    return _ensemble_q(F, z).cpu().numpy()


@torch.no_grad()
def v_values(model, agent, obs: torch.Tensor, z: torch.Tensor) -> np.ndarray:
    action = agent.act(obs, z)
    if isinstance(action, np.ndarray):
        action = torch.as_tensor(action, dtype=torch.float32)
    return q_values(model, obs, action.to(obs.device).float(), z)


def transport_mask(ep: Dict[str, Any], thr) -> np.ndarray:
    """Per-step boolean: cube lifted above table and gripper closed."""
    cube = np.asarray(ep["cube"], dtype=np.float64).reshape(-1, 3)
    grip = np.asarray(ep["grip"], dtype=np.float64).reshape(-1)
    lift = cube[:, 2] - float(ep["table_z"])
    return (lift > thr.delta_lift) & (grip > thr.tau_grip)


def rollout_for_profile(env, agent, z, n_episodes: int, thr,
                        seed_offset: int = 0) -> List[Dict[str, Any]]:
    """S0 rollouts with obs/action recorded, each tagged with outcome,
    per-step cube->goal distance, and a transport-region mask."""
    from evals.phase_probe import classify_phases, rollout_with_phase_signals

    eps = rollout_with_phase_signals(
        env, agent, z, n_episodes, thr, scenario="S0", record_obs=True)
    out = []
    for ep in eps:
        cls = classify_phases(ep, thr)
        if cls["success"]:
            outcome = "success"
        elif cls["fail_phase"] == "transport":
            outcome = "transport_fail"
        else:
            outcome = "other"
        cube = np.asarray(ep["cube"], dtype=np.float64).reshape(-1, 3)
        goal = np.asarray(ep["goal"], dtype=np.float64).reshape(3)
        d = np.linalg.norm(cube - goal, axis=1)
        out.append({
            "obs": ep["obs"], "action": ep["action"],
            "cube": cube, "goal": goal, "d": d,
            "grip": np.asarray(ep["grip"], np.float64).reshape(-1),
            "table_z": float(ep["table_z"]),
            "transport_mask": transport_mask(ep, thr),
            "outcome": outcome, "length": ep["length"],
        })
    return out


def _slope(y: np.ndarray, x: np.ndarray) -> float:
    """OLS slope of y on x; nan if <2 distinct x."""
    if len(x) < 2 or np.ptp(x) < 1e-12:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def probe_value_landscape(model, agent, episodes, z):
    """Per-episode slope of V w.r.t. (-d) over the transport region,
    plus a long-form (outcome, d, V) frame for binned plots."""
    import pandas as pd

    per_ep_rows, per_step_rows = [], []
    for i, ep in enumerate(episodes):
        obs = torch.as_tensor(np.asarray(ep["obs"]), dtype=torch.float32)
        if obs.shape[0] == 0:
            continue
        V = v_values(model, agent, obs, z)             # [T]
        d = np.asarray(ep["d"], np.float64)
        m = np.asarray(ep["transport_mask"], bool)
        for dd, vv, mm in zip(d, V, m):
            per_step_rows.append({"episode": i, "outcome": ep["outcome"],
                                  "d": float(dd), "V": float(vv),
                                  "transport": bool(mm)})
        if m.sum() >= 2:
            slope = _slope(V[m], -d[m])
            v_secure = float(V[m][0])
            v_end = float(V[-1])
        else:
            slope, v_secure, v_end = float("nan"), float("nan"), float("nan")
        per_ep_rows.append({"episode": i, "outcome": ep["outcome"],
                            "slope_V_negd": slope, "V_at_secure": v_secure,
                            "V_at_end": v_end,
                            "n_transport_steps": int(m.sum())})
    return pd.DataFrame(per_ep_rows), pd.DataFrame(per_step_rows)


@torch.no_grad()
def probe_z_decoding(model, next_obs: torch.Tensor, physics: np.ndarray,
                     goal: np.ndarray, z: torch.Tensor,
                     relabel_metrics: Dict[str, float],
                     topk: int = 256,
                     goal_thresh: float = 0.04) -> Dict[str, float]:
    """How sparse are goal states, and does z point at 'placed' states?"""
    B = model.backward_map(next_obs.float())              # [N, z_dim]
    scores = (B * z.view(1, -1)).sum(-1).cpu().numpy()    # B·z
    cube = np.asarray(physics, np.float64)[:, 14:17]
    d = np.linalg.norm(cube - np.asarray(goal, np.float64).reshape(1, 3),
                        axis=1)
    k = int(min(topk, len(scores)))
    idx = np.argsort(-scores)[:k]
    ns = float(relabel_metrics.get("relabel_reward#num_samples", 0.0)) or 1.0
    return {
        "relabel_pos_frac": float(
            relabel_metrics.get("relabel_reward#nonzero", 0.0)) / ns,
        "topk_mean_d": float(d[idx].mean()),
        "topk_pct_at_goal": float(np.mean(d[idx] < goal_thresh)),
        "topk": k,
    }
