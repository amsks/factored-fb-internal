"""data_collection/collect_mw_play_resample.py — OGBench-style PLAY data for Metaworld:
long trajectories that chain target relocations, no terminal dwell.

The cube-play property the other regimes lack (phenomenon.tex E7/Ph.III): 1000-step
task-agnostic trajectories whose goal marginal is spread over the workspace instead of
collapsing onto one per-episode target. Here each trajectory runs `traj_len` wrapper steps
and the target is RE-DRAWN from the env's own goal space
  * immediately when `success` fires (no dwell — the oracle moves on like cube play), and
  * every `seg_max` steps regardless (no stuck segments),
by writing `env.unwrapped._target_pos`, which is exactly what the observation's goal dims
(obs[36:39]) and the scripted oracle read — so the recorded behaviour is coherent
chase-current-target play, and within-trajectory relabeling now crosses target contexts
through transitions the data actually contains (the train-side repair of Ph.II; the
eval-side fixed-frame conditioning of env/unified.py:167 is NOT touched by this regime).

Truncation: Metaworld's raw envs truncate at max_path_length=500 simulator steps; we zero
`curr_path_length` before every wrapper step (action-repeat 2), so a trajectory of
`traj_len` wrapper steps never hits it.

Output: one packed [N, traj_len, ·] npz, byte-compatible with env/unified.py::load_packed:

    MUJOCO_GL=egl python data_collection/collect_mw_play_resample.py \
        --task mw-push --episodes 200 --out $SCRATCH/fb-data/play-resample
"""
import argparse
import json
import os

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

# Mild exploration mixture per SEGMENT (between target re-draws), gentler than play-noise:
# the point of this regime is coverage via relocation chaining, not via action noise.
PLAY_DEFAULTS = dict(eps_max=0.1, noise_max=0.1, expert_frac=0.0)
TRAJ_LEN = 500          # wrapper steps (= 1000 simulator steps), cube-play scale
SEG_MAX = 60            # re-draw the target after this many steps without success


def _policy(task):
    import metaworld.policies as P
    base = task.split("-", 1)[-1]
    overrides = {"peg-insert-side": "SawyerPegInsertionSideV2Policy"}
    name = overrides.get(base, "Sawyer" + "".join(p.capitalize() for p in base.split("-")) + "V2Policy")
    if not hasattr(P, name):
        raise KeyError(f"no scripted policy for {base!r} (tried {name})")
    return getattr(P, name)()


def collect(task, episodes, traj_len, seg_max, eps_max, noise_max, expert_frac, seed0):
    from env.metaworld_env import make_metaworld_env
    policy = _policy(task)
    env, _horizon = make_metaworld_env(task)
    raw = env.unwrapped
    raw.goal_space.seed(seed0 + 7)
    rng = np.random.RandomState(seed0)

    O, A, NO, R, SU, AG = [], [], [], [], [], []
    reloc_counts, succ_steps = [], 0
    for e in range(episodes):
        np.random.seed(seed0 + e)                   # MW draws rand_vec from the global RNG
        obs, info = env.reset()
        st = np.asarray(obs, np.float32).reshape(-1)

        def draw_mixture():
            if rng.rand() < expert_frac:
                return 0.0, 0.0
            return rng.rand() * eps_max, rng.rand() * noise_max

        eps, noise = draw_mixture()
        seg, relocs = 0, 0
        o, a, no, r, su = [], [], [], [], []
        for _t in range(traj_len):
            raw.curr_path_length = 0                # defeat max_path_length truncation
            act = np.asarray(policy.get_action(st.astype(np.float64)), np.float32).reshape(-1)
            act = act + noise * rng.randn(*act.shape).astype(np.float32)
            if rng.rand() < eps:
                act = rng.uniform(-1.0, 1.0, size=act.shape).astype(np.float32)
            act = np.clip(act, -1.0, 1.0)
            nobs, rew, _term, _trunc, info = env.step(act)
            nst = np.asarray(nobs, np.float32).reshape(-1)
            hit = float(info.get("success", 0.0))
            o.append(st); a.append(act); no.append(nst)
            r.append(float(rew)); su.append(hit)
            st = nst
            seg += 1
            if hit > 0 or seg >= seg_max:           # relocate: new target, no dwell
                raw._target_pos = raw.goal_space.sample().astype(np.float64)
                seg = 0
                relocs += int(hit > 0)
                eps, noise = draw_mixture()
        O.append(np.stack(o)); A.append(np.stack(a)); NO.append(np.stack(no))
        R.append(np.asarray(r, np.float32)); SU.append(np.asarray(su, np.float32))
        AG.append(np.zeros((traj_len, 3), np.float32))
        reloc_counts.append(relocs)
        succ_steps += int(np.sum(su))
    env.close()
    data = dict(observation=np.stack(O), action=np.stack(A), next_observation=np.stack(NO),
                achieved_goal=np.stack(AG), reward=np.stack(R),
                terminal=np.zeros((episodes, traj_len), bool),
                timeout=np.stack([np.eye(traj_len, dtype=bool)[-1]] * episodes),
                success=np.stack(SU).astype(bool))
    stats = dict(relocations_per_traj=float(np.mean(reloc_counts)),
                 dwell_frac=float(succ_steps / (episodes * traj_len)))
    return data, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="mw-<name>, e.g. mw-push")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--traj_len", type=int, default=TRAJ_LEN)
    ap.add_argument("--seg_max", type=int, default=SEG_MAX)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="root; writes <out>/data/metaworld/<task>.npz")
    ap.add_argument("--eps_max", type=float, default=PLAY_DEFAULTS["eps_max"])
    ap.add_argument("--noise_max", type=float, default=PLAY_DEFAULTS["noise_max"])
    ap.add_argument("--expert_frac", type=float, default=PLAY_DEFAULTS["expert_frac"])
    args = ap.parse_args()

    data, stats = collect(args.task, args.episodes, args.traj_len, args.seg_max,
                          args.eps_max, args.noise_max, args.expert_frac, args.seed * 100000)
    out_dir = os.path.join(args.out, "data", "metaworld")
    os.makedirs(out_dir, exist_ok=True)
    npz = os.path.join(out_dir, f"{args.task}.npz")
    np.savez_compressed(npz, **data)
    meta = dict(task=args.task, regime="play-resample", num_episodes=args.episodes,
                traj_len=args.traj_len, seg_max=args.seg_max, seed=args.seed,
                eps_max=args.eps_max, noise_max=args.noise_max,
                expert_frac=args.expert_frac, **stats)
    with open(os.path.join(out_dir, f"{args.task}.meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[mw-resample] {args.task}: {args.episodes} x {args.traj_len} steps, "
          f"relocs/traj {stats['relocations_per_traj']:.1f}, "
          f"dwell {stats['dwell_frac']:.3f} -> {npz}")


if __name__ == "__main__":
    main()
