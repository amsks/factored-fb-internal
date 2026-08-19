"""data_collection/collect_mw_play_resample.py — OGBench-style PLAY data for Metaworld:
long trajectories that chain target relocations on a fixed timer.

The cube-play property the other regimes lack (phenomenon.tex E7/Ph.III): long
task-agnostic trajectories whose goal marginal is spread over the workspace instead of
collapsing onto one per-episode target. Each trajectory runs `traj_len` wrapper steps and
the target is RE-DRAWN from the env's own goal space every `seg_max` steps — on the timer
alone, never on success, matching OGBench's plan-completion trigger — by writing
`env.unwrapped._target_pos`, which is exactly what the observation's goal dims (obs[36:39])
and the scripted oracle read. So the recorded behaviour is coherent chase-current-target
play, and within-trajectory relabeling crosses target contexts through transitions the data
actually contains (the train-side repair of Ph.II; the eval-side fixed-frame conditioning of
env/unified.py:167 is NOT touched by this regime).

v3 (2026-08-05) is the version audited line-by-line against OGBench's own generation script
and against the Metaworld env internals. It fixes, relative to v2: epsilon-random actions
(absorbing puck loss — see PLAY_DEFAULTS), the success-triggered dwell (OGBench uses a pure
timer — see SEG_MAX), an out-of-band `_get_obs()` that emitted zero-velocity frames at every
relocation, a goal marker that never moved in the rendered scene, and an unclipped goal
channel. It also records `goal_switch`, which OGBench does not: cube play leaves segment
boundaries unrecoverable, and having them makes the relabeling analysis possible.

TWO structural differences remain and CANNOT be fixed here:
  1. Metaworld's observation CONTAINS the current target at dims 36:39, whereas OGBench cube
     observations carry no goal at all. An unconditional BC actor therefore solves the task
     by reading its own input. Masking the emitted goals is NOT enough to stop this — use
     `obs_mask_dims=[36,37,38]` (impls/utils/datasets.py), which blanks the observation
     itself; `goal_mask_dims` only blanks the goal vector.
  2. The Metaworld oracle is closed-loop reactive with no plan and no completion signal,
     while OGBench's is plan-based with an explicit `done`. Segment ends here are a timer
     rather than task completion, and there is no waypoint randomisation to draw
     suboptimality from, so exploration has to come from action-space noise instead.

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
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
# Invoked by file path (not -m), so the repo root is not on sys.path; env.* needs it.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Exploration mixture per SEGMENT. eps_max defaults to ZERO, matching OGBench, which sets
# p_random_action=0 for every `play` dataset (data_gen_scripts/commands.sh) -- its
# suboptimality comes entirely from randomised plan waypoints and smoothed setpoint noise,
# never from replacing the action. Here epsilon-uniform actions are not merely off-contract,
# they are destructive: a uniform draw on the gripper dims knocks the puck out of the
# reachable workspace, the stateless Metaworld oracle has no recovery behaviour, and the
# collector never resets mid-trajectory -- so the failure is ABSORBING. Measured on mw-push,
# eps_max=0.1 sent the puck outside the workspace on 36% of steps and left 4/8 trajectories
# scoring exactly zero successes for every remaining step. Gaussian noise perturbs the oracle
# without discarding it, so exploration lives there.
#
# noise_max is 0.02 for the same reason, one step milder. Measured on mw-push over 8 x 500
# steps: sigma 0.10 -> 9.4% of steps with the puck outside the workspace and 2/8 trajectories
# permanently dead; 0.05 -> 8.5% and 1/8; 0.02 -> ZERO. Sigma 0.05 matches cube's at-rest
# fraction more closely (0.503 vs 0.500, against 0.562 at 0.02), but that is a secondary
# distributional statistic and this is a correctness property: a dead trajectory contributes
# transitions whose goal channel is unreachable for every remaining step. OGBench has no
# analogue -- its block cannot leave the bin -- so zero dead data is the faithful choice.
# `escape_frac` in the meta makes any regression here visible instead of silent.
PLAY_DEFAULTS = dict(eps_max=0.0, noise_max=0.02, expert_frac=0.0)
# Structure matched to OGBench cube-single-play-v0, measured from the stock npz (1000 x 1001
# steps: 11.56 relocations per trajectory, block at rest 49.98% of steps) rather than assumed.
# Cube runs 1001-step trajectories against a 200-step eval horizon (ratio 5.0) and re-draws on
# PLAN COMPLETION -- a ~87-step timer, i.e. 0.435 x eval horizon. Metaworld's eval horizon is
# 100 wrapper steps, so the matched trajectory is 500 steps with a 44-step timer -> 11.4
# relocations.
#
# The trigger is a pure timer, as in OGBench: `success` is deliberately NOT consulted. OGBench
# computes success every step during generation and discards it (manipspace_env.py) -- the
# plan runs to completion whether or not the block arrived. Two earlier versions of this file
# got that wrong in opposite directions: v1 re-drew the instant success fired (10-13 step
# segments), and v2 added an explicit post-success dwell on the mistaken reading that cube
# play "holds" at the goal. It does not. Cube's 50%-at-rest is the block sitting still while
# the arm approaches it and then retreats to a random idle pose -- an artefact of a
# pick-and-place plan, not a hold. Under a timer the Metaworld oracle reaches the target
# early in the segment and then simply keeps it there, which reproduces that at-rest structure
# as a consequence instead of legislating it.
TRAJ_LEN = 500          # wrapper steps (= 1000 simulator steps), cube-play scale
SEG_MAX = 44            # re-draw the target every this many steps, regardless of success
OBJ_SLICE = slice(4, 7)   # obs dims holding object xyz -- what success is measured on
GOAL_SLICE = slice(36, 39)  # obs dims holding the current target (_target_pos)
# Only these two Metaworld tasks are semantically sound under target re-drawing: their goal is
# a free point the object can be carried to, and goal_space covers the reachable set. Measured
# elsewhere in the suite, mw-bin-picking's goal_space is measure-zero (every re-draw is the
# same point) and mw-button-press's is 64x larger than the reachable set (~98% of re-draws
# unachievable); the fixture tasks (assembly/basketball/box-close/door-open) have targets that
# are poses of a fixture rather than free points.
SUPPORTED = ("mw-push", "mw-pick-place")
# |object x| beyond this is outside the reachable workspace (the goal box spans x in
# [-0.1, 0.1]); the puck slides freely and the stateless oracle never recovers it.
ESCAPE_X = 0.15
MAX_ATTEMPTS = 8        # re-rolls per episode before accepting an unhealthy one


def _policy(task):
    import metaworld.policies as P
    base = task.split("-", 1)[-1]
    overrides = {"peg-insert-side": "SawyerPegInsertionSideV2Policy"}
    name = overrides.get(base, "Sawyer" + "".join(p.capitalize() for p in base.split("-")) + "V2Policy")
    if not hasattr(P, name):
        raise KeyError(f"no scripted policy for {base!r} (tried {name})")
    return getattr(P, name)()


def _reset_target_z_pool(env, raw, n_probe, seed0):
    """The z values the env's OWN reset produces for _target_pos.

    goal_space is not the reset distribution. For mw-push goal_space draws z ~ U[0.01, 0.02]
    while every reset puts the target at exactly z = 0.02, so sampling goal_space wholesale
    put 100% of re-drawn push targets below the table -- a 5 mm bias against a 50 mm success
    radius. Re-drawn targets therefore take xy from goal_space (that is where the coverage
    comes from) and z from this empirical pool, which is on-manifold by construction and
    degenerates to the single correct value for push while keeping real z diversity for
    pick-place."""
    zs = []
    for i in range(n_probe):
        np.random.seed(seed0 + 900000 + i)
        env.reset()
        zs.append(float(raw._target_pos[2]))
    return np.asarray(zs, np.float64)


def collect(task, episodes, traj_len, seg_max, eps_max, noise_max, expert_frac, seed0):
    if task not in SUPPORTED:
        raise SystemExit(
            f"{task!r} is not supported by this collector (see SUPPORTED). Re-drawing "
            f"_target_pos only makes sense where the goal is a free point the object can be "
            f"carried to and goal_space covers the reachable set; elsewhere it yields a "
            f"degenerate or unachievable goal marginal.")
    from env.metaworld_env import make_metaworld_env
    policy = _policy(task)
    env, _horizon = make_metaworld_env(task)
    raw = env.unwrapped
    raw.goal_space.seed(seed0 + 7)
    rng = np.random.RandomState(seed0)
    z_pool = _reset_target_z_pool(env, raw, 200, seed0)

    # step() clips the emitted observation to sawyer_observation_space, whose goal sub-box IS
    # goal_space, while reward/success read the unclipped _target_pos. Keep the bounds so the
    # spliced goal channel matches exactly what step() would have produced.
    glo = np.asarray(raw.sawyer_observation_space.low[GOAL_SLICE], np.float64)
    ghi = np.asarray(raw.sawyer_observation_space.high[GOAL_SLICE], np.float64)

    def draw_target():
        t = raw.goal_space.sample().astype(np.float64)
        t[2] = z_pool[rng.randint(len(z_pool))]
        return t

    O, A, NO, R, SU, AG, SW = [], [], [], [], [], [], []
    esc_steps = 0
    reloc_counts, reloc_hit_counts, succ_steps = [], [], 0
    attempt, rejected = 0, 0
    e = 0
    while len(O) < episodes:
        # Reject-and-retry on an unhealthy trajectory, the same shape as OGBench's own
        # `while True: ... if not healthy: continue` loop for scene (generate_manipspace.py).
        # Here "unhealthy" means the object left the reachable workspace: Metaworld's puck
        # slides freely, the scripted oracle is stateless and cannot recover it, and the
        # collector never resets mid-trajectory -- so every remaining step carries a goal that
        # can never be reached again. Measured at 1M scale before this guard: 10.2% of mw-push
        # trajectories escaped and 162 of 203 were dead for the rest of their length, giving a
        # 0.082 success rate after the escape against 0.809 overall. Retrying costs ~10% more
        # collection and removes the dead states entirely; truncating instead would break the
        # fixed-horizon [N, T, .] pack.
        seed_e = seed0 + e + 1000000 * attempt
        np.random.seed(seed_e)                      # MW draws rand_vec from the global RNG
        obs, info = env.reset()
        st = np.asarray(obs, np.float32).reshape(-1)

        def draw_mixture():
            if rng.rand() < expert_frac:
                return 0.0, 0.0
            return rng.rand() * eps_max, rng.rand() * noise_max

        eps, noise = draw_mixture()
        seg, relocs, relocs_hit = 0, 0, 0
        o, a, no, r, su, sw = [], [], [], [], [], []
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
            sw.append(0.0)
            if seg >= seg_max:
                # Re-draw on the timer alone, then splice the new target into the stored
                # observation. v1 re-drew after `st = nst` and never refreshed, so the first
                # observation of every segment still carried the OLD target while the env had
                # already switched -- the oracle chased a target the reward was not computed
                # against, on one step per relocation (7-10% of all transitions).
                #
                # The refresh must be a splice, not a `_get_obs()` call. Metaworld's obs is
                # [curr 18 | prev 18 | goal 3] and `_get_obs` rebuilds it with BOTH halves set
                # to the current frame, i.e. zero apparent velocity -- an observation the eval
                # env never produces (its reset frame is burned by MetaWorldWrapper.reset).
                # v2 called it and put such a frame at 2.35% (push) / 2.70% (pick-place) of
                # all rows, landing exactly on the goal-channel jumps this regime exists to
                # create. Splicing keeps the true frame stack and touches only the goal dims.
                t = draw_target()
                raw._target_pos = t
                # mj_forward recomputes every site xpos from model.site_pos, so writing
                # data.site(...).xpos inside step() is clobbered; the goal marker only moves
                # if model.site_pos moves. Without this the rendered target stays at the
                # episode's reset position forever and any video of this data is wrong.
                try:
                    raw.model.site("goal").pos = t
                except Exception:                   # noqa: BLE001  (task without a goal site)
                    pass
                st = nst.copy()
                # step() clips the observation to sawyer_observation_space, whose goal box is
                # goal_space, while reward/success use the unclipped _target_pos. Clip here too
                # so the stored goal channel can never disagree with what step() would emit.
                st[GOAL_SLICE] = np.clip(t, glo, ghi).astype(np.float32)
                seg = 0
                relocs += 1
                relocs_hit += int(bool(hit > 0))    # did the segment end having reached it
                sw[-1] = 1.0
                eps, noise = draw_mixture()
        no_arr = np.stack(no)
        # Object driven outside the goal box by more than its own width => unrecoverable.
        n_esc = int((np.abs(no_arr[:, OBJ_SLICE][:, 0]) > ESCAPE_X).sum())
        if n_esc and attempt < MAX_ATTEMPTS:
            rejected += 1
            attempt += 1                            # re-roll THIS episode with a fresh seed
            continue
        attempt = 0
        e += 1
        O.append(np.stack(o)); A.append(np.stack(a)); NO.append(no_arr)
        R.append(np.asarray(r, np.float32)); SU.append(np.asarray(su, np.float32))
        SW.append(np.asarray(sw, np.float32))
        esc_steps += n_esc
        # achieved_goal is the object position -- the quantity success is actually measured
        # on (||next_obs[4:7] - next_obs[36:39]|| < radius). It used to be written as zeros.
        AG.append(no_arr[:, OBJ_SLICE].astype(np.float32))
        reloc_counts.append(relocs); reloc_hit_counts.append(relocs_hit)
        succ_steps += int(np.sum(su))
    env.close()
    data = dict(observation=np.stack(O), action=np.stack(A), next_observation=np.stack(NO),
                achieved_goal=np.stack(AG), reward=np.stack(R),
                terminal=np.zeros((episodes, traj_len), bool),
                timeout=np.stack([np.eye(traj_len, dtype=bool)[-1]] * episodes),
                success=np.stack(SU).astype(bool),
                goal_switch=np.stack(SW).astype(bool))
    stats = dict(relocations_per_traj=float(np.mean(reloc_counts)),
                 success_relocations_per_traj=float(np.mean(reloc_hit_counts)),
                 dwell_frac=float(succ_steps / (episodes * traj_len)),
                 escape_frac=float(esc_steps / (episodes * traj_len)),
                 rejected_trajectories=int(rejected))
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
    meta = dict(task=args.task, regime="play-resample", version=3, num_episodes=args.episodes,
                traj_len=args.traj_len, seg_max=args.seg_max,
                seed=args.seed, eps_max=args.eps_max, noise_max=args.noise_max,
                expert_frac=args.expert_frac, **stats)
    with open(os.path.join(out_dir, f"{args.task}.meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[mw-resample] {args.task}: {args.episodes} x {args.traj_len} steps, "
          f"relocs/traj {stats['relocations_per_traj']:.1f}, "
          f"dwell {stats['dwell_frac']:.3f} -> {npz}")


if __name__ == "__main__":
    main()
