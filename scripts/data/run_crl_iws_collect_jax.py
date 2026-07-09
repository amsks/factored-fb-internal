"""Online CRL + IWS data collection script (JAX implementation).

Matches Unsupervised_Manipulation SGCRL / IWS pipeline performance while
running the agent entirely in JAX/Flax.  The IWR (Interaction-Weighted
Regression) sampling strategy is preserved faithfully through a NumPy port of
UM's VectorizedReplayBuffer with IWS event caching.

Usage
-----
    python scripts/data/run_crl_iws_collect.py \\
        --env AirHockeyPuckGoalPosition-v0 \\
        --num-envs 64 \\
        --total-steps 500000 \\
        --output datasets/crl-iws-air-hockey/buffer

    # then train offline FB / GCIQL on the exported episodes
    python train.py domain=crl_iws_air_hockey \\
        data_path=datasets device=cpu
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from agents.crl_iws.agent import CRLIWSAgent
from agents.crl_iws.replay import IWSReplayBuffer


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def _max_episode_steps(vec_env, default: int) -> int:
    spec = getattr(vec_env, "spec", None)
    if spec is None:
        return default
    first = spec[0] if isinstance(spec, (list, tuple)) else spec
    return getattr(first, "max_episode_steps", None) or default


def _make_single_env(env_name: str, seed: int):
    try:
        import rlmini_envs
        env = rlmini_envs.make(
            env_name,
            seed=seed,
            gym_to_gymnasium=True,
            select_obs=True,
            obs_flat=True,
            obs_include_metadata=False,
        )
        return env
    except ImportError:
        pass
    # Fallback: plain Gymnasium registration
    import gymnasium as gym
    return gym.make(env_name)


def _make_vec_env(env_name: str, num_envs: int, seed: int):
    """Create a SyncVectorEnv wrapping `num_envs` copies of `env_name`."""
    import gymnasium as gym

    def _make(i):
        def _init():
            return _make_single_env(env_name, seed + i)
        return _init

    return gym.vector.SyncVectorEnv([_make(i) for i in range(num_envs)])


# ---------------------------------------------------------------------------
# Phase label computation (mirrors UM InteractionReplayManager)
# ---------------------------------------------------------------------------


def compute_phase_labels(
    states: np.ndarray,          # (N, state_dim)
    next_states: np.ndarray,     # (N, state_dim)
    active_post_contact: np.ndarray,  # (N,) countdown int
    *,
    contact_distance: float = 0.09,
    near_contact_distance: float = 0.18,
    post_contact_steps: int = 20,
    dones: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (phase_labels, updated_post_contact_countdown).

    Phase 0: passive  Phase 1: near_contact
    Phase 2: contact  Phase 3: post_contact
    """
    if states.shape[-1] < 6:
        n = states.shape[0]
        return np.zeros(n, dtype=np.int64), active_post_contact.copy()

    def _dist(s: np.ndarray) -> np.ndarray:
        return np.linalg.norm(s[:, 0:2] - s[:, 4:6], axis=-1)

    dist = np.minimum(_dist(states), _dist(next_states))

    near_contact = dist <= near_contact_distance
    contact = dist <= contact_distance
    post_contact = active_post_contact > 0

    phase = np.zeros(states.shape[0], dtype=np.int64)
    phase = np.where(near_contact, 1, phase)
    phase = np.where(post_contact, 3, phase)
    phase = np.where(contact, 2, phase)

    countdown = np.clip(active_post_contact - 1, 0, None)
    countdown = np.where(contact, post_contact_steps, countdown)
    if dones is not None:
        countdown = np.where(dones.astype(bool), 0, countdown)

    return phase, countdown


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def run_collection(args: argparse.Namespace) -> None:
    rng = jax.random.PRNGKey(args.seed)
    np.random.seed(args.seed)

    print(f"[crl-iws] Creating {args.num_envs} envs: {args.env}")
    vec_env = _make_vec_env(args.env, args.num_envs, args.seed)

    obs_space = vec_env.single_observation_space
    act_space = vec_env.single_action_space

    state_dim = int(np.prod(obs_space.shape))
    action_dim = int(np.prod(act_space.shape))
    goal_dim = state_dim  # use full state as achieved goal (HER)

    action_low = np.asarray(act_space.low, dtype=np.float32)
    action_high = np.asarray(act_space.high, dtype=np.float32)

    print(f"[crl-iws] state_dim={state_dim}  action_dim={action_dim}")

    agent = CRLIWSAgent.create(
        seed=args.seed,
        state_dim=state_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        hidden_dims=tuple(args.hidden_dims),
        repr_dim=args.repr_dim,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        entropy_coeff=args.entropy_coeff,
        logsumexp_coeff=args.logsumexp_coeff,
        random_goals=args.random_goals,
        use_action_entropy=args.use_action_entropy,
    )

    replay = IWSReplayBuffer(
        capacity_transitions=args.replay_capacity,
        max_episode_steps=_max_episode_steps(vec_env, args.max_episode_steps),
        num_envs=args.num_envs,
        state_dim=state_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        contact_distance=args.contact_distance,
    )

    # Reset
    raw_obs, _ = vec_env.reset(seed=list(range(args.seed, args.seed + args.num_envs)))
    obs = raw_obs.reshape(args.num_envs, state_dim).astype(np.float32)
    replay.reset(obs)
    active_post_contact = np.zeros(args.num_envs, dtype=np.int64)

    total_steps = 0
    total_updates = 0
    last_log = 0
    last_metrics: dict[str, float] = {}
    t0 = time.perf_counter()

    print(f"[crl-iws] Training for {args.total_steps} env steps "
          f"({args.random_exploration_steps} random)")

    while total_steps < args.total_steps:
        # ---- Action selection ----
        if total_steps < args.random_exploration_steps:
            action = vec_env.action_space.sample()
        else:
            rng, act_rng = jax.random.split(rng)
            # Use current state as goal placeholder (agent picks action toward future)
            # In practice the env doesn't need a goal at collection time — the actor
            # is conditioned on a randomly-rolled goal for exploration diversity.
            rng_goals = jax.random.split(act_rng, 1)[0]
            shuffled_obs = obs[np.random.permutation(args.num_envs)]
            action_jax = agent.sample_actions(
                jax.device_put(obs),
                jax.device_put(shuffled_obs),
                seed=act_rng,
                temperature=1.0,
            )
            action = np.asarray(action_jax)

        # ---- Environment step ----
        raw_next, _, terminated, truncated, info = vec_env.step(action)
        next_obs = raw_next.reshape(args.num_envs, state_dim).astype(np.float32)
        done = (np.asarray(terminated) | np.asarray(truncated)).astype(bool)

        phase_labels, active_post_contact = compute_phase_labels(
            obs, next_obs, active_post_contact,
            contact_distance=args.contact_distance,
            near_contact_distance=args.near_contact_distance,
            post_contact_steps=args.post_contact_steps,
            dones=done,
        )

        replay.append(
            states=obs,
            achieved_goals=obs,
            actions=action.astype(np.float32),
            next_states=next_obs,
            next_achieved_goals=next_obs,
            dones=done,
        )

        obs = next_obs
        total_steps += args.num_envs

        # ---- Agent update ----
        if (
            total_steps >= args.random_exploration_steps
            and replay.can_sample(args.batch_size)
        ):
            for _ in range(args.updates_per_iter):
                batch = replay.sample(
                    args.batch_size,
                    future_discount=args.future_discount,
                    context_repetition_factor=args.context_repetition_factor,
                    contact_sigma=args.contact_sigma,
                    random_fraction=args.iws_random_fraction,
                    full_episode_future=args.iws_full_episode_future,
                    event_selection=args.iws_event_selection,
                    use_iws=True,
                )
                jax_batch = {
                    "states": jax.device_put(batch.states),
                    "next_states": jax.device_put(batch.next_states),
                    "goals": jax.device_put(batch.goals),
                    "actions": jax.device_put(batch.actions),
                }
                agent, last_metrics = agent.update(jax_batch)
                total_updates += 1

        # ---- Logging ----
        if total_steps - last_log >= args.log_every:
            elapsed = max(time.perf_counter() - t0, 1e-6)
            sps = float(total_steps - last_log) / max(time.perf_counter() - t0, 1e-6)
            rpt = replay.stats()
            print(
                f"[crl-iws] steps={total_steps:>8d}  updates={total_updates:>6d}  "
                f"sps={total_steps/elapsed:.0f}  "
                f"episodes={rpt['replay/episodes']}  "
                f"iws_events={rpt['replay/iws_events']}  "
                + ("  ".join(f"{k}={v:.3f}" for k, v in last_metrics.items()
                             if k in ("critic_loss", "actor_loss", "policy_entropy")))
            )
            last_log = total_steps

    vec_env.close()

    # ---- Export ----
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[crl-iws] Exporting episodes to {out_dir}")
    paths = replay.export_npz(out_dir, max_episodes=args.max_export_episodes)

    manifest = {
        "source": "crl_iws_jax",
        "env": args.env,
        "total_env_steps": total_steps,
        "total_updates": total_updates,
        "num_episodes": len(paths),
        "episodes": [str(p.resolve()) for p in paths],
    }
    (out_dir.parent / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"[crl-iws] Wrote {len(paths)} episodes → {out_dir}")
    if paths:
        d = np.load(paths[0])
        print(f"[crl-iws] obs_dim={d['observation'].shape[-1]}  "
              f"act_dim={d['action'].shape[-1]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)

    # Environment
    ap.add_argument("--env", default="AirHockeyPuckGoalPosition-v0")
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--max-episode-steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)

    # Collection
    ap.add_argument("--total-steps", type=int, default=500_000)
    ap.add_argument("--random-exploration-steps", type=int, default=5_000)
    ap.add_argument("--output", default="datasets/crl-iws/buffer")
    ap.add_argument("--max-export-episodes", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=10_000)

    # Agent hyperparameters (match UM defaults)
    ap.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256])
    ap.add_argument("--repr-dim", type=int, default=64)
    ap.add_argument("--actor-lr", type=float, default=6e-4)
    ap.add_argument("--critic-lr", type=float, default=3e-4)
    ap.add_argument("--entropy-coeff", type=float, default=0.0)
    ap.add_argument("--logsumexp-coeff", type=float, default=0.1)
    ap.add_argument("--random-goals", type=float, default=0.5)
    ap.add_argument("--use-action-entropy", action="store_true", default=True)

    # Replay / IWS hyperparameters
    ap.add_argument("--replay-capacity", type=int, default=200_000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--updates-per-iter", type=int, default=32)
    ap.add_argument("--future-discount", type=float, default=0.99)
    ap.add_argument("--context-repetition-factor", type=int, default=4,
                    help="IWS context repetition (IWR batch grouping around events)")
    ap.add_argument("--contact-distance", type=float, default=0.09)
    ap.add_argument("--near-contact-distance", type=float, default=0.18)
    ap.add_argument("--post-contact-steps", type=int, default=20)
    ap.add_argument("--contact-sigma", type=float, default=20.0)
    ap.add_argument("--iws-random-fraction", type=float, default=0.0)
    ap.add_argument("--iws-full-episode-future", action="store_true", default=False)
    ap.add_argument("--iws-event-selection", choices=["uniform", "first"], default="uniform")

    return ap.parse_args()


if __name__ == "__main__":
    run_collection(_parse_args())
