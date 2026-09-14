"""Reward families on the frozen 1M OGBench cube checkpoints (evaluation only).

Question: do the zero-shot models handle state rewards that are NOT goal-reaching? Every
family below is a function of the state only (cube positions read from the simulator qpos,
table-plane coordinates). For each (family, method, seed) the task index z is inferred from
10k dataset transitions labelled with that reward -- the same reward-inference route
impls/main.py's eval_relabel block uses for the paper's OGBench table (10k random train
rows, reward on the NEXT state, shift 1.0, `agent.infer_eval_cond`, goal frames dropped) --
and the frozen policy is rolled for 20 full-length episodes from the env's own resets. The
recorded quantity is the requested reward's own return (mean per-step, and discounted at
0.99), never a success rate, except where the family asks for one (R5).

Floors: a uniform-random-action policy and the dataset's behaviour-cloning policy, which is
the flow vector field `actor_vf` every FlowBC checkpoint carries (it is trained with the
flow-matching BC loss only; the value term touches the one-step head `actor`, never
`actor_vf` -- impls/actors/flowbc.py::loss). The BC floor is read off the fb_tdj
checkpoints' actor_vf, one per seed.

Machinery reused verbatim: scripts/probes/cube_bottleneck/agents.py (load_arm_eval,
discover_roots), restore.make_env, the `BatchedInferredCondAgent` lockstep wrapper.

Usage (one job = one env x one method x one seed, every family of that env):
  python scripts/reward_families/eval_families.py --env cube-single-play-v0 \
      --method density_fb_z --seed 0 [--episodes 20] [--families R1,R2] [--smoke]
  python scripts/reward_families/eval_families.py --env cube-single-play-v0 --method random
  python scripts/reward_families/eval_families.py --env cube-single-play-v0 --method bc --seed 0
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")
_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", _cvd.split(",")[0].strip() or "0")

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.probes.cube_bottleneck import agents as A          # noqa: E402
from scripts.probes.cube_bottleneck import restore as R         # noqa: E402

# ── constants (every one is recorded in the output JSON) ───────────────────────────────
TABLE_Z = 0.02          # a resting cube's centre height (half-size); "height above table" = z - 0.02
RADIUS = 0.06           # region disc radius, metres, in the table plane (x, y)
SUCCESS_RADIUS = 0.04   # the env's own success tolerance (cube_env._compute_successes)
LIFT_SCALE = 0.10       # R4: r = clip((z - TABLE_Z) / 0.10, 0, 1)
KEEP_PENALTY = 0.5      # R6: -0.5 per step cube2 is outside its start disc
PREF_MINOR = 0.3        # R3: weight of the less-preferred region
GAMMA = 0.99
SHIFT = 1.0             # eval_relabel_shift of the table protocol, added to every label
N_INFER = 10000         # eval_relabel_size of the table protocol
CUBE_QPOS_OFFSETS = (14, 21)   # env/ogbench.py::CUBE_QPOS_OFFSETS (free joint = 3 pos + 4 quat)
INFER_SEED = 20260914   # the 10k relabel rows: same rows for every method (np.random re-seeded)
RESET_SEED = 20260914   # env reset seeds: same resets for every method

# Regions, chosen inside the dataset's cube-position support (x in [0.30, 0.55], y in
# [-0.30, 0.30]; each disc covers ~7% of dataset rows) and at least 0.06 m from every eval
# reset position of the env (so no episode starts inside a region). None coincides with an
# eval-task goal.
REGIONS = {
    "cube-single-play-v0": dict(A=(0.48, 0.23), B=(0.48, -0.23)),
    # cube-double resets put a cube at (0.425, 0.20) +- 0.01, so A sits further out.
    "cube-double-play-v0": dict(A=(0.50, 0.23), B0=(0.50, -0.10), r6_task=2),
}

# method label -> (critic dir name, exp roots in preference order)
METHODS = {
    "fb_tdj":           ("fb_tdj", ("exp", "exp/tdjtab")),
    "density_fb_z":     ("density_fb_z", ("exp/tdjtab", "exp")),
    "hilp_tdjc":        ("hilp_tdjc", ("exp", "exp/hlevers")),
    "hilp_density_reg": ("hilp_density_reg", ("exp/hlevers",)),
    "rldp":             ("rldp", ("exp/ortho1",)),          # ortho-1 checkpoints, by request
    "rldp_density_reg": ("rldp_density_reg", ("exp/levers",)),
}
PRETTY = {"fb_tdj": "FB", "density_fb_z": "Density-FB", "hilp_tdjc": "HILP",
          "hilp_density_reg": "Density-HILP", "rldp": "RLDP", "rldp_density_reg": "Density-RLDP",
          "random": "Random", "bc": "BC"}
BC_SOURCE = "fb_tdj"    # whose actor_vf the BC floor is read from


# ── rewards: cube [N, C, 3] -> [N] ──────────────────────────────────────────────────────
def in_disc(cube_xyz, centre, radius=RADIUS):
    c = np.asarray(centre, np.float64)
    return (np.linalg.norm(cube_xyz[..., :2] - c, axis=-1) < radius).astype(np.float64)


def make_families(env_name: str, task_infos) -> Dict[str, Dict[str, Any]]:
    """Every family of `env_name`: name -> dict(reward=fn(cube[N,C,3])->[N], reset=..., aux=...).

    `reset` is None for the env's own reset distribution (random task, permuted cubes, the
    task_id is irrelevant to the reward), or a dict(task_id=..., permute=...) where the
    family needs a fixed start (R5's target, R6's start disc).
    `aux` lists extra per-step indicator functions to record (time in A / B, success...).
    """
    reg = REGIONS[env_name]
    fam: Dict[str, Dict[str, Any]] = {}
    if env_name == "cube-single-play-v0":
        Ar, Br = reg["A"], reg["B"]
        inA = lambda c: in_disc(c[:, 0], Ar)
        inB = lambda c: in_disc(c[:, 0], Br)
        aux_ab = {"in_A": inA, "in_B": inB}
        fam["R1_region"] = dict(reward=inA, reset=None, aux=aux_ab,
                                definition=f"r = 1[cube in A], A = disc({Ar}, r={RADIUS})")
        fam["R2_either"] = dict(reward=lambda c: np.maximum(inA(c), inB(c)), reset=None, aux=aux_ab,
                                definition=f"r = max(1[A], 1[B]), A={Ar}, B={Br}")
        fam["R3_pref_a"] = dict(reward=lambda c: inA(c) + PREF_MINOR * inB(c), reset=None, aux=aux_ab,
                                definition=f"r_a = 1[A] + {PREF_MINOR}*1[B]")
        fam["R3_pref_b"] = dict(reward=lambda c: PREF_MINOR * inA(c) + inB(c), reset=None, aux=aux_ab,
                                definition=f"r_b = {PREF_MINOR}*1[A] + 1[B]")
        fam["R4_lift"] = dict(reward=lambda c: np.clip((c[:, 0, 2] - TABLE_Z) / LIFT_SCALE, 0, 1),
                              reset=None, aux={"lifted_3cm": lambda c: (c[:, 0, 2] - TABLE_Z > 0.03).astype(float)},
                              definition=f"r = clip((z - {TABLE_Z}) / {LIFT_SCALE}, 0, 1)")
        # R5: one sub-family per eval task (target = that task's goal cube position); the
        # dense reward and the sparse success reward are two labellings of the same target.
        for t, info in enumerate(task_infos, start=1):
            g = np.asarray(info["goal_xyzs"], np.float64)[0]
            dist = lambda c, g=g: np.linalg.norm(c[:, 0] - g, axis=-1)
            hit = lambda c, g=g: (np.linalg.norm(c[:, 0] - g, axis=-1) <= SUCCESS_RADIUS).astype(float)
            aux5 = {"dense": lambda c, d=dist: -d(c), "hit": hit}
            fam[f"R5_dense_t{t}"] = dict(reward=lambda c, d=dist: -d(c), reset=dict(task_id=t, permute=True),
                                         aux=aux5, definition=f"r = -||cube - g||, g = {g.tolist()} (task {t} goal)")
            fam[f"R5_sparse_t{t}"] = dict(reward=hit, reset=dict(task_id=t, permute=True), aux=aux5,
                                          definition=f"r = 1[||cube - g|| <= {SUCCESS_RADIUS}], g = {g.tolist()} (task {t} goal)")
    elif env_name == "cube-double-play-v0":
        Ar, B0 = reg["A"], reg["B0"]
        inA0 = lambda c: in_disc(c[:, 0], Ar)
        inA1 = lambda c: in_disc(c[:, 1], Ar)
        inB0_1 = lambda c: in_disc(c[:, 1], B0)
        fam["R6_move_keep"] = dict(
            reward=lambda c: inA0(c) - KEEP_PENALTY * (1.0 - inB0_1(c)),
            reset=dict(task_id=reg["r6_task"], permute=False),
            aux={"cube1_in_A": inA0, "cube2_in_B0": inB0_1},
            definition=f"r = 1[cube1 in A] - {KEEP_PENALTY}*1[cube2 outside B0], A={Ar}, B0={B0}; "
                       f"reset = task {reg['r6_task']} init, cube order fixed (cube2 starts in B0)")
        fam["R7_either_cube"] = dict(reward=lambda c: np.maximum(inA0(c), inA1(c)), reset=None,
                                     aux={"cube1_in_A": inA0, "cube2_in_A": inA1},
                                     definition=f"r = max(1[cube1 in A], 1[cube2 in A]), A={Ar}")
    else:
        raise ValueError(env_name)
    return fam


def cube_from_qpos(qpos: np.ndarray, n_cubes: int) -> np.ndarray:
    """[N, D] qpos -> [N, C, 3] cube positions (metres)."""
    qpos = np.asarray(qpos, np.float64)
    return np.stack([qpos[:, o:o + 3] for o in CUBE_QPOS_OFFSETS[:n_cubes]], axis=1)


# ── policies ────────────────────────────────────────────────────────────────────────────
class BatchedCondAgent:
    """The `BatchedInferredCondAgent` of agents.relabel_cond_agents: one inferred z, tiled
    over a batch of lockstep observations, handed through the `goals` dict channel (no
    'goal' key, so every critic's map_eval_cond passes it through); env goals are never
    forwarded."""

    def __init__(self, agent):
        import jax.numpy as jnp
        self.agent = agent
        self.cond_eval = agent.cond_eval
        self.z = jnp.asarray(agent.cond_eval["z"])

    def sample_actions(self, observations, seed=None, temperature=0.0):
        import jax.numpy as jnp
        obs = jnp.asarray(observations)
        z = jnp.broadcast_to(self.z[None], (obs.shape[0], self.z.shape[-1]))
        return self.agent.sample_actions(observations, goals={"z": z}, seed=seed,
                                         temperature=temperature)


class BCFlowAgent:
    """Pure behaviour cloning: the checkpoint's flow vector field `actor_vf` integrated from
    Gaussian noise (FlowBCActor._rollout), with no critic anywhere in the loop."""

    def __init__(self, agent):
        import jax
        actor, params = agent.actor, agent.net["actor_vf"].params
        self._fn = jax.jit(lambda obs, noise: actor._rollout(params, obs, noise))
        self.adim = int(actor.actor.action_dim)

    def sample_actions(self, observations, seed=None, temperature=0.0):
        import jax
        import jax.numpy as jnp
        obs = jnp.asarray(observations)
        noise = jax.random.normal(seed, (obs.shape[0], self.adim))
        return self._fn(obs, noise)


class RandomAgent:
    def __init__(self, adim: int, rng: np.random.RandomState):
        self.adim, self.rng = adim, rng

    def sample_actions(self, observations, seed=None, temperature=0.0):
        return self.rng.uniform(-1, 1, size=(len(observations), self.adim))


# ── reward inference (the eval_relabel protocol with an arbitrary state reward) ─────────
def infer_index(arm: A.Arm, reward_fn: Callable, n_cubes: int) -> Dict[str, Any]:
    """z from N_INFER random TRAIN transitions labelled `reward_fn(next-state cube
    positions) + SHIFT`, through `agent.infer_eval_cond`, with the Route B validity guard
    of impls/main.py (conditioning must change, be finite and have a non-zero norm)."""
    import jax.numpy as jnp
    from env.ogbench import OGBenchAdapter

    ds = OGBenchAdapter._column_backed(arm.dataset)
    np.random.seed(INFER_SEED)
    idxs = ds.get_random_idxs(N_INFER)
    nxt = np.minimum(idxs + 1, ds.size - 1)
    obs = np.asarray(ds["observations"][nxt], np.float32)
    cube = cube_from_qpos(ds["qpos"][nxt], n_cubes)
    r_raw = np.asarray(reward_fn(cube), np.float64)
    rew = (r_raw + SHIFT).astype(np.float32)
    pre = arm.agent.cond_eval
    ag = arm.agent.infer_eval_cond(jnp.asarray(obs), jnp.asarray(rew))
    z = np.asarray(ag.cond_eval["z"], np.float64).ravel()
    norm = float(np.linalg.norm(z))
    changed = not np.allclose(z, np.asarray(pre.get("z", 0.0), np.float64).ravel())
    finite = bool(np.all(np.isfinite(z)))
    reason = None
    if not changed:
        reason = "reward inference was a no-op (cond unchanged)"
    elif not finite:
        reason = "inferred z is not finite"
    elif not norm > 0:
        reason = "inferred z has zero norm"
    cos1 = float(z.sum() / (norm * np.sqrt(len(z)))) if norm > 0 else float("nan")
    return dict(agent=ag, z=z, route_b_valid=reason is None, invalid_reason=reason or "",
                inferred_cond_norm=norm, inferred_cond_cos_neutral=cos1,
                label_mean=float(rew.mean()), label_std=float(rew.std()),
                label_pos_frac=float(np.mean(r_raw > 0)), label_raw_mean=float(r_raw.mean()))


# ── rollouts ────────────────────────────────────────────────────────────────────────────
def make_pool(env_name: str, n: int) -> List[Any]:
    envs = [R.make_env(env_name, seed=RESET_SEED + i) for i in range(n)]
    for e in envs:
        # Full-length episodes: the env would otherwise end the episode the instant the
        # DRAWN task's goal is met, which has nothing to do with the requested reward.
        e.unwrapped._terminate_at_goal = False
    return envs


def rollout(policy, envs, reset, reward_fn, aux: Dict[str, Callable], n_cubes: int,
            fam_idx: int, rng_key) -> Dict[str, Any]:
    """`len(envs)` episodes in lockstep. Rewards are computed on the state AFTER each step
    (as the labels are on next states). Returns per-episode metrics + per-step traces."""
    from impls.utils.evaluation import supply_rng

    actor_fn = supply_rng(policy.sample_actions, rng=rng_key)
    n = len(envs)
    obs = []
    for m, e in enumerate(envs):
        u = e.unwrapped
        u._permute_blocks = True if reset is None else bool(reset.get("permute", True))
        opts = dict(render_goal=False)
        if reset is not None:
            opts["task_id"] = int(reset["task_id"])
        else:
            np.random.seed(RESET_SEED + 1000 * fam_idx + m)     # the random task draw
        ob, _ = e.reset(seed=RESET_SEED + 1000 * fam_idx + m, options=opts)
        obs.append(np.asarray(ob, np.float32))
    obs = np.stack(obs)
    init_cube = np.stack([cube_from_qpos(e.unwrapped._data.qpos[None], n_cubes)[0] for e in envs])
    alive = np.ones(n, bool)
    rews: List[np.ndarray] = []
    auxs: Dict[str, List[np.ndarray]] = {k: [] for k in aux}
    t = 0
    while alive.any():
        acts = np.clip(np.asarray(actor_fn(observations=obs, temperature=0.0)), -1, 1)
        cube = np.zeros((n, n_cubes, 3))
        for m, e in enumerate(envs):
            if not alive[m]:
                continue
            ob, _r, term, trunc, _info = e.step(acts[m].astype(np.float32))
            obs[m] = np.asarray(ob, np.float32)
            cube[m] = cube_from_qpos(e.unwrapped._data.qpos[None], n_cubes)[0]
            if term or trunc:
                alive[m] = False
        rews.append(np.asarray(reward_fn(cube), np.float64))
        for k, f in aux.items():
            auxs[k].append(np.asarray(f(cube), np.float64))
        t += 1
    assert not alive.any() and len(set([t])) == 1
    Rm = np.stack(rews, axis=1)                        # [n, T]
    T = Rm.shape[1]
    disc = (GAMMA ** np.arange(T))
    out = dict(
        T=int(T),
        mean_reward=Rm.mean(axis=1).tolist(),
        discounted_return=(Rm * disc).sum(axis=1).tolist(),
        undiscounted_return=Rm.sum(axis=1).tolist(),
        final_reward=Rm[:, -1].tolist(),
        init_cube=init_cube.tolist(),
        reward_trace_mean=Rm.mean(axis=0).tolist(),  # [T] over episodes
    )
    for k, v in auxs.items():
        M = np.stack(v, axis=1)
        out[f"aux_{k}_mean"] = M.mean(axis=1).tolist()       # time fraction per episode
        out[f"aux_{k}_any"] = M.max(axis=1).tolist()         # any-step per episode
    return out


# ── main ────────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--method", required=True, help="a METHODS key, 'random' or 'bc'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epoch", type=int, default=1000000)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--families", default="", help="comma list; default = all of the env")
    ap.add_argument("--out", default=str(ROOT / "analysis/reward_families"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    import jax

    t0 = time.time()
    n_cubes = 2 if "double" in args.env else 1
    out_dir = Path(args.out) / args.env
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "random" if args.method == "random" else f"{args.method}_seed{args.seed}"
    out_path = out_dir / (f"{tag}{'_smoke' if args.smoke else ''}.json")

    arm: Optional[A.Arm] = None
    if args.method != "random":
        critic, roots = METHODS[BC_SOURCE if args.method == "bc" else args.method]
        runs = A.discover_roots(args.env, [critic], [args.seed], args.epoch, roots=roots)
        assert runs, f"no {critic} seed {args.seed} checkpoint at {args.epoch} under {roots}"
        run = runs[0]
        print(f"[load] {run['run_dir']} epoch {args.epoch}", flush=True)
        arm = A.load_arm_eval(run["run_dir"], args.epoch)
        task_infos = arm.env.unwrapped.task_infos
        run_dir = str(run["run_dir"])
    else:
        task_infos = R.make_env(args.env).unwrapped.task_infos
        run_dir = ""

    fams = make_families(args.env, task_infos)
    if args.families:
        keep = args.families.split(",")
        fams = {k: v for k, v in fams.items() if k in keep}
        assert fams, f"no family matched {keep}"
    envs = make_pool(args.env, args.episodes)
    adim = int(envs[0].action_space.shape[0])

    results = dict(env=args.env, method=args.method, seed=args.seed, epoch=args.epoch,
                   run_dir=run_dir, episodes=args.episodes, gamma=GAMMA, shift=SHIFT,
                   n_infer=N_INFER, radius=RADIUS, table_z=TABLE_Z, lift_scale=LIFT_SCALE,
                   success_radius=SUCCESS_RADIUS, keep_penalty=KEEP_PENALTY, pref_minor=PREF_MINOR,
                   regions={k: v for k, v in REGIONS[args.env].items()},
                   infer_seed=INFER_SEED, reset_seed=RESET_SEED, families={})
    for fi, (name, fam) in enumerate(fams.items()):
        tf = time.time()
        rec: Dict[str, Any] = dict(definition=fam["definition"], reset=fam["reset"])
        if args.method == "random":
            policy = RandomAgent(adim, np.random.RandomState(RESET_SEED + fi))
        elif args.method == "bc":
            policy = BCFlowAgent(arm.agent)
        else:
            inf = infer_index(arm, fam["reward"], n_cubes)
            rec.update({k: v for k, v in inf.items() if k not in ("agent", "z")})
            rec["z_head"] = inf["z"][:8].tolist()
            print(f"[{name}] z norm {inf['inferred_cond_norm']:.3f} cos1 {inf['inferred_cond_cos_neutral']:.3f} "
                  f"valid {inf['route_b_valid']} label mean {inf['label_mean']:.3f} pos {inf['label_pos_frac']:.4f}",
                  flush=True)
            if not inf["route_b_valid"]:
                print(f"[{name}] SKIPPED rollout: {inf['invalid_reason']}", flush=True)
                results["families"][name] = rec
                continue
            policy = BatchedCondAgent(inf["agent"])
        roll = rollout(policy, envs, fam["reset"], fam["reward"], fam["aux"], n_cubes, fi,
                       jax.random.PRNGKey(RESET_SEED + 7919 * args.seed + fi))
        rec.update(roll)
        mr = float(np.mean(roll["mean_reward"]))
        print(f"[{name}] T={roll['T']} mean/step {mr:.4f} +- {np.std(roll['mean_reward']):.4f} "
              f"disc {np.mean(roll['discounted_return']):.3f} "
              + " ".join(f"{k}={np.mean(v):.3f}" for k, v in roll.items() if k.startswith("aux_") and k.endswith("_mean"))
              + f"  ({time.time() - tf:.0f}s)", flush=True)
        assert np.isfinite(mr)
        results["families"][name] = rec
        with open(out_path, "w") as f:      # incremental
            json.dump(results, f)
    results["wall_s"] = time.time() - t0
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"[done] {out_path} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
