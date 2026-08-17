"""data_collection/collect_ms_play_resample.py — OGBench-style PLAY data for ManiSkill:
long trajectories that chain target relocations on a fixed timer.

This is `collect_ms_play.py`'s rollout loop (newt TD-MPC2 oracle, obs padding, Vulkan ICD)
carrying every lesson v3 of `collect_mw_play_resample.py` learned, and it exists to remove the
size confound from Arm E's ManiSkill null (spec E2,
docs/specs/2026-08-10-deconfound-size-coverage.md): 1,000,000 transitions per task instead of
the upstream set's 48,000.

What it inherits from MW v3:
  * the target is re-drawn on a TIMER ALONE, never on success (OGBench's plan-completion
    trigger; success is computed and deliberately not consulted);
  * the target moves through the env's OWN goal mechanism — the `goal_site` / `goal_region`
    actor pose, which is exactly what `_get_obs_extra` reads — never by writing observation
    dims directly;
  * no epsilon-random actions (absorbing failure with a stateless oracle), Gaussian action
    noise only;
  * reject-and-retry on trajectories whose cube left the workspace, with `rejected_trajectories`
    recorded so a regression is visible instead of silent;
  * no zero-velocity synthetic frames at relocations;
  * `goal_switch` recorded, which OGBench does not, so segment boundaries stay recoverable.

Where it deliberately DIFFERS from MW v3: the goal channel is refreshed by recomputing the
env's own observation (`BaseEnv.get_obs()`), not by splicing dims. MW had to splice because
its observation is [curr 18 | prev 18 | goal 3] and `_get_obs()` rebuilds both halves from the
current frame, emitting a zero-velocity frame. ManiSkill's state observation carries no
prev-frame stack — velocities come straight out of the sim — so recomputing is exact, and it
picks up the DERIVED channels a splice would miss (PickCube's obs carries `obj_to_goal_pos` at
39:42 as well as `goal_pos` at 26:29). `--verify_redraw` asserts, at the first relocation of
every run, that recomputing changed the goal dims and nothing else.

STRUCTURE IS NOT THE SAME FOR THE TWO TASKS, and that is a measured constraint, not an
oversight — see TASKS below.

    VK_ICD_FILENAMES=$SCRATCH/vulkan/nvidia_icd.json .venv-gen/bin/python \
        data_collection/collect_ms_play_resample.py --task ms-pick-cube \
        --episodes 8000 --out $SCRATCH/fb-data/ms-play-resample
"""
import argparse
import json
import os
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
# Invoked by file path (not -m), so the repo root is not on sys.path; env.* needs it.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Gaussian action noise only. The spec's ceiling is sigma 0.1; 0.05 is the upstream collector's
# value and the milder end is the right default for the same reason MW v3 settled on 0.02 --
# ManiSkill's 25-step manipulation is fragile, and a knocked-off cube is an ABSORBING failure
# for a stateless oracle. `escape_frac` in the meta makes any regression here visible.
NOISE_MAX = 0.05

# Per-task structure. `traj_len` and `seg_max` are quoted against the 25-step eval horizon H;
# the cube-play targets they are matched to are 5.0 x H per trajectory and 0.435 x H per
# segment (measured from cube-single-play-v0: 1001 steps, 11.56 relocations).
#
# ms-pick-cube MATCHES cube: its goal is a free point in 3-space that the pick policy can
# carry the cube to over and over, so relocations chain indefinitely. Confirmed on the
# upstream 400-step data: dwell 0.295 and FLAT across the trajectory (0.300 over the first 100
# steps, 0.290 over the last), median object-goal distance 0.021 against a 0.025 success
# radius, and the cube never leaves the table.
#
# ms-push-cube CANNOT match it, and this is geometric rather than fixable. PushCube's own reset
# always places the goal at cube_xy + [0.1 + goal_radius, 0], so the TD-MPC2 oracle only ever
# learned to push in +x and cannot pull; each relocation therefore has to advance the cube
# ~0.2 m down a reachable band that is only ~0.4 m long. Chaining cube-scale relocation counts
# marches the cube straight out of the workspace, which is exactly what the upstream 400-step
# set did: dwell decays 0.070 (first 100 steps) -> 0.006 (last 100), median object-goal
# distance 0.306 against a 0.10 radius, 10% of frames with |obj_x| > 0.4, and 63% of segments
# had the goal BEHIND the cube (reached on 17.6% of them, against 100% for the segments whose
# goal landed in the pushable band). So push-cube is collected at ONE relocation per
# trajectory, the count its geometry admits, and is reported as a structure-mismatched control
# rather than a cube-matched replication. `relocations_per_traj` in the meta states it.
TASKS = {
    "ms-pick-cube": dict(
        traj_len=125, seg_max=11,          # 5.00 x H, 0.44 x H  -> ~11 relocations (cube: 11.56)
        goal=slice(26, 29), obj=slice(29, 32), planar=False, radius=0.025,
        escape_z=0.0, escape_xy=0.35, episodes_for_1m=8000,
    ),
    "ms-push-cube": dict(
        traj_len=50, seg_max=25,           # 2.00 x H, 1.00 x H  -> 1 relocation (see above)
        goal=slice(25, 28), obj=slice(28, 31), planar=True, radius=0.10,
        escape_z=0.0, escape_xy=0.45, episodes_for_1m=20000,
    ),
}

# PushCube's measured pushable envelope (from the upstream set, per-segment reach rates above):
# a target ahead of the cube by 0.0-0.3 m in +x is reachable (0.47 / 0.28 / 0.68 by 0.1 m
# band), beyond +0.3 m nothing was ever reached, and behind the cube almost nothing was. The
# band is bracketed around the env's own +0.2 offset. A target must also stay inside the arm's
# reach: every successful segment in the upstream set had goal_x <= 0.268 and |goal_y| <= 0.163.
PUSH_DX = (0.10, 0.30)
PUSH_DY = 0.10
PUSH_X_MAX, PUSH_Y_MAX = 0.30, 0.20

MAX_ATTEMPTS = 8        # re-rolls per episode before giving up on a healthy one


def _flag(x):
    return bool(x.any()) if hasattr(x, "any") else bool(x)


def _goal_drawer(task, base, rng):
    """-> (draw(obj_xyz) -> target_xyz | None, set_goal(target_xyz)).

    `draw` returns None when no target inside the task's reachable envelope exists for the
    cube's current position; the caller treats that as an unhealthy trajectory. Constants come
    off the env instance, never hardcoded, so a ManiSkill version bump that moves the spawn box
    moves the re-draw distribution with it.
    """
    import torch
    from mani_skill.utils.structs import Pose

    if task == "ms-pick-cube":
        # The env's own reset distribution for goal_site (pick_cube.py::_initialize_episode):
        # xy uniform over the cube spawn box, z uniform in [cube_half_size, +max_goal_height].
        cx, cy = base.cube_spawn_center
        hs, zmax, zc = base.cube_spawn_half_size, base.max_goal_height, base.cube_half_size

        def draw(_obj):
            return np.array([cx + rng.uniform(-hs, hs), cy + rng.uniform(-hs, hs),
                             zc + rng.uniform(0.0, zmax)], np.float64)

        def set_goal(t):
            base.goal_site.set_pose(Pose.create_from_pq(
                p=torch.tensor(t, dtype=torch.float32, device=base.device).reshape(1, 3)))
        return draw, set_goal

    if task == "ms-push-cube":
        from transforms3d.euler import euler2quat
        q = np.asarray(euler2quat(0, np.pi / 2, 0), np.float32)   # target faces up, as at reset

        def draw(obj):
            # Ahead of the cube in +x (the only direction this oracle can push), inside the
            # measured envelope. Rejection-sample rather than clip: clipping would pile targets
            # onto the envelope boundary and distort the goal marginal E2 is measuring.
            for _ in range(32):
                t = np.array([obj[0] + rng.uniform(*PUSH_DX),
                              obj[1] + rng.uniform(-PUSH_DY, PUSH_DY), 1e-3], np.float64)
                if t[0] <= PUSH_X_MAX and abs(t[1]) <= PUSH_Y_MAX:
                    return t
            return None                     # cube is past the band: unhealthy trajectory

        def set_goal(t):
            base.goal_region.set_pose(Pose.create_from_pq(
                p=torch.tensor(t, dtype=torch.float32, device=base.device).reshape(1, 3),
                q=torch.tensor(q, dtype=torch.float32, device=base.device).reshape(1, 4)))
        return draw, set_goal

    raise SystemExit(
        f"{task!r} is not supported by this collector (known: {sorted(TASKS)}). Re-drawing a "
        f"target only makes sense where the goal is a pose the object can be moved to and the "
        f"env exposes it as an actor; StackCube's goal is another cube and LiftPegUpright's is "
        f"an orientation.")


def _obs_now(base):
    """The env's own observation, recomputed from the current sim state."""
    from mani_skill.utils import common
    return np.asarray(common.to_numpy(common.unbatch(base.get_obs())), np.float32).reshape(-1)


def _check_redraw(task, before, after, spec):
    """Assert that moving the goal actor changed the goal channel and NOTHING else.

    This is the gate on the two things that could silently invalidate the whole collection:
    that `set_pose` took effect at all (it is a no-op if the sim needs an explicit apply), and
    that no observation dim outside the goal's own algebra moved. PickCube carries
    `obj_to_goal_pos` at 39:42, so the expected change set is wider than the `goal_pos` slice
    -- and it is precisely that derived channel a naive splice would leave stale.
    """
    moved = set(np.nonzero(np.abs(after - before) > 1e-6)[0].tolist())
    goal_dims = set(range(spec["goal"].start, spec["goal"].stop))
    derived = set(range(39, 42)) if task == "ms-pick-cube" else set()
    if task == "ms-push-cube":
        # The push draw pins goal z to the table plane (1e-3, same as the env's own reset),
        # so the goal's last dim legitimately never moves on a re-draw; requiring it would
        # fail every healthy collection. x,y still must move.
        goal_dims -= {spec["goal"].stop - 1}
    if not goal_dims <= moved:
        raise RuntimeError(
            f"{task}: re-drawing the target did not change obs dims {sorted(goal_dims - moved)}. "
            f"The goal actor's set_pose did not reach the observation -- every relocation in "
            f"this collection would be a no-op.")
    extra = moved - goal_dims - derived
    if extra:
        raise RuntimeError(
            f"{task}: re-drawing the target also changed obs dims {sorted(extra)}, which are "
            f"not the goal channel or its derived differences. The observation layout in "
            f"TASKS/_check_redraw is stale for this ManiSkill version.")


def collect(task, episodes, traj_len, seg_max, noise_max, seed0, verify_redraw=True):
    import torch
    from data_collection.collect_ms_play import _load_agent

    spec = TASKS[task]
    agent, pad_dim, real_act, task_idx = _load_agent(task)
    from env.maniskill_env import make_maniskill_env
    # The registered envs truncate at max_episode_steps=50; a play trajectory is longer than
    # that, so the limit is raised to the trajectory length (the env is otherwise identical to
    # the one eval and the parity probe use).
    env, H = make_maniskill_env(task, max_episode_steps=traj_len)
    base = env.unwrapped
    rng = np.random.RandomState(seed0)
    draw_target, set_goal = _goal_drawer(task, base, rng)
    tt = torch.tensor([task_idx], device="cuda:0")

    O, A, NO, R, SU, AG, SW = [], [], [], [], [], [], []
    esc_steps, succ_steps = 0, 0
    reloc_counts, reloc_hit_counts = [], []
    attempt, rejected, unreachable = 0, 0, 0
    checked = not verify_redraw
    e = 0
    while len(O) < episodes:
        obs, info = env.reset(seed=seed0 + e + 1000000 * attempt)
        st = np.asarray(obs, np.float32).reshape(-1)
        noise = rng.rand() * noise_max
        seg, relocs, relocs_hit = 0, 0, 0
        o, a, no, r, su, sw = [], [], [], [], [], []
        dead = False
        for _t in range(traj_len):
            po = np.concatenate([st, np.zeros(pad_dim - st.shape[0], np.float32)])
            with torch.no_grad():
                act = agent(torch.tensor(po, device="cuda:0").unsqueeze(0), t0=(_t == 0),
                            eval_mode=False, task=tt, mpc=False)
            act = np.asarray(act.squeeze(0).cpu(), np.float32)[:real_act]
            act = np.clip(act + noise * rng.randn(*act.shape).astype(np.float32), -1.0, 1.0)
            nobs, rew, _term, _trunc, info = env.step(act)
            nst = np.asarray(nobs, np.float32).reshape(-1)
            hit = float(_flag(info.get("success", 0)))
            o.append(st); a.append(act); no.append(nst)
            r.append(float(rew)); su.append(hit)
            st = nst
            seg += 1
            sw.append(0.0)
            if seg >= seg_max and _t < traj_len - 1:
                # Timer alone. `hit` is recorded but never consulted here -- OGBench computes
                # success during generation and discards it, and v1 of the MW collector re-drew
                # on success, which collapsed segments to 10-13 steps.
                t = draw_target(nst[spec["obj"]])
                if t is None:                     # no reachable target left (push-cube)
                    dead = True
                    break
                set_goal(t)
                # Recompute rather than splice: exact for a ManiSkill state obs, and it picks
                # up derived goal channels. The stored frame is the NEXT transition's
                # observation, so the segment's first row already carries the new target -- the
                # off-by-one that left MW v1 chasing a target the reward was not computed
                # against on 7-10% of transitions.
                after = _obs_now(base)
                if not checked:
                    _check_redraw(task, st, after, spec)
                    checked = True
                st = after
                seg = 0
                relocs += 1
                relocs_hit += int(hit > 0)        # did the segment end having reached it
                sw[-1] = 1.0
                noise = rng.rand() * noise_max

        no_arr = np.stack(no) if no else np.zeros((0, st.shape[0]), np.float32)
        # Unhealthy = the cube left the workspace. ManiSkill's cube can be knocked off the
        # table and the oracle has no recovery behaviour, so every remaining step of such a
        # trajectory carries a goal that can never be reached again -- the MS analogue of the
        # MW absorbing-escape defect. Retry the episode with a fresh seed rather than truncate,
        # which would break the fixed-horizon [N, T, .] pack.
        if dead:
            n_esc = 0
        else:
            oz = no_arr[:, spec["obj"]]
            n_esc = int(((oz[:, 2] < spec["escape_z"]) |
                         (np.abs(oz[:, :2]).max(1) > spec["escape_xy"])).sum())
        if (dead or n_esc) and attempt < MAX_ATTEMPTS:
            rejected += 1
            unreachable += int(dead)
            attempt += 1
            continue
        attempt = 0
        e += 1
        O.append(np.stack(o)); A.append(np.stack(a)); NO.append(no_arr)
        R.append(np.asarray(r, np.float32)); SU.append(np.asarray(su, np.float32))
        SW.append(np.asarray(sw, np.float32))
        # achieved_goal is the object position -- the quantity success is measured on. The
        # play-noise collector wrote zeros here.
        AG.append(no_arr[:, spec["obj"]].astype(np.float32))
        esc_steps += n_esc
        succ_steps += int(np.sum(su))
        reloc_counts.append(relocs); reloc_hit_counts.append(relocs_hit)
        if e % 200 == 0:
            print(f"[ms-resample] {task}: {e}/{episodes} trajs, "
                  f"dwell {succ_steps / (e * traj_len):.3f}, rejected {rejected}", flush=True)
    env.close()

    n = episodes
    data = dict(observation=np.stack(O), action=np.stack(A), next_observation=np.stack(NO),
                achieved_goal=np.stack(AG), reward=np.stack(R),
                terminal=np.zeros((n, traj_len), bool),
                timeout=np.stack([np.eye(traj_len, dtype=bool)[-1]] * n),
                success=np.stack(SU).astype(bool),
                goal_switch=np.stack(SW).astype(bool))
    stats = dict(relocations_per_traj=float(np.mean(reloc_counts)),
                 success_relocations_per_traj=float(np.mean(reloc_hit_counts)),
                 dwell_frac=float(succ_steps / (n * traj_len)),
                 escape_frac=float(esc_steps / (n * traj_len)),
                 rejected_trajectories=int(rejected),
                 rejected_unreachable=int(unreachable),
                 segment_over_horizon=float(seg_max / H),
                 traj_over_horizon=float(traj_len / H))
    return data, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help=" | ".join(sorted(TASKS)))
    ap.add_argument("--episodes", type=int, default=None,
                    help="default: the count that reaches 1,000,000 transitions for the task")
    ap.add_argument("--traj_len", type=int, default=None, help="default: per-task (see TASKS)")
    ap.add_argument("--seg_max", type=int, default=None, help="default: per-task (see TASKS)")
    ap.add_argument("--noise_max", type=float, default=NOISE_MAX)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="root; writes <out>/data/maniskill/<task>.npz")
    ap.add_argument("--no_verify_redraw", action="store_true",
                    help="skip the first-relocation observation-layout assertion (don't)")
    args = ap.parse_args()

    if args.task not in TASKS:
        raise SystemExit(f"unknown task {args.task!r} (known: {sorted(TASKS)})")
    spec = TASKS[args.task]
    traj_len = args.traj_len or spec["traj_len"]
    seg_max = args.seg_max or spec["seg_max"]
    episodes = args.episodes or spec["episodes_for_1m"]

    data, stats = collect(args.task, episodes, traj_len, seg_max, args.noise_max,
                          args.seed * 100000, verify_redraw=not args.no_verify_redraw)
    out_dir = os.path.join(args.out, "data", "maniskill")
    os.makedirs(out_dir, exist_ok=True)
    npz = os.path.join(out_dir, f"{args.task}.npz")
    np.savez_compressed(npz, **data)
    meta = dict(task=args.task, family="maniskill", regime="play-resample", version=1,
                policy="tdmpc2-goal-resample", num_episodes=episodes, traj_len=traj_len,
                seg_max=seg_max, seed=args.seed, noise_max=args.noise_max, eps_max=0.0,
                transitions=int(episodes * traj_len),
                obs_dim=int(data["observation"].shape[-1]),
                act_dim=int(data["action"].shape[-1]), **stats)
    with open(os.path.join(out_dir, f"{args.task}.meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[ms-resample] {args.task}: {episodes} x {traj_len} = {episodes * traj_len} "
          f"transitions, relocs/traj {stats['relocations_per_traj']:.2f}, "
          f"dwell {stats['dwell_frac']:.3f}, escape {stats['escape_frac']:.4f}, "
          f"rejected {stats['rejected_trajectories']} -> {npz}", flush=True)


if __name__ == "__main__":
    main()
