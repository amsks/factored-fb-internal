"""data_collection/collect_mw_standalone.py — Metaworld data via the built-in SCRIPTED ORACLE,
torch-free and self-contained on Vista.

Why this exists: the published Factor-FB-Offline dataset has no mw-pick-place / mw-push, and
its play counterpart (TommyShen/factored-fb-play) is unreachable. But Metaworld ships a
scripted oracle for every MT50 task, so both regimes can be regenerated here with no
TD-MPC2 checkpoint and no torch:

    expert      = oracle, zero exploration            (expert_frac=1.0)
    play-noise  = oracle + per-episode noise mixture  (eps_max/noise_max/expert_frac)

The exploration mixture is transcribed from tl-play-data's collect_play.py::rollout_mw so the
data matches what that pipeline would have produced. The env is env/metaworld_env.py — the
same wrapper training/eval use (action-repeat 2, horizon 100) — so obs/action layout is
byte-identical to the eval env and to the expert Metaworld npz (obs 39, act 4).

Output is one packed [N,T,·] npz per (task, regime), the layout env/unified.py::load_packed
reads:

    MUJOCO_GL=egl python data_collection/collect_mw_standalone.py \
        --task mw-pick-place --regime play-noise --episodes 1000 --out $SCRATCH/fb-data/mw_play
"""
import argparse
import os

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

# Per-family play-noise defaults from tl-play-data (Metaworld row).
PLAY_DEFAULTS = dict(eps_max=0.3, noise_max=0.2, expert_frac=0.3)
HORIZON = 100


def _policy(task):
    """Metaworld scripted oracle for a `mw-<name>` slug (SawyerCamelCaseV2Policy)."""
    import metaworld.policies as P
    base = task.split("-", 1)[-1]
    overrides = {"peg-insert-side": "SawyerPegInsertionSideV2Policy"}
    name = overrides.get(base, "Sawyer" + "".join(p.capitalize() for p in base.split("-")) + "V2Policy")
    if not hasattr(P, name):
        raise KeyError(f"no scripted policy for {base!r} (tried {name})")
    return getattr(P, name)()


def collect(task, episodes, regime, eps_max, noise_max, expert_frac, seed0):
    """-> dict of packed [N,T,·] arrays. `regime` only sets whether exploration is applied;
    expert forces every episode to the no-noise branch."""
    from env.metaworld_env import make_metaworld_env
    policy = _policy(task)
    env, horizon = make_metaworld_env(task)
    rng = np.random.RandomState(seed0)
    O, A, NO, R, SU, AG = [], [], [], [], [], []
    ep_kept = 0
    for e in range(episodes):
        np.random.seed(seed0 + e)                       # MW draws rand_vec from the global RNG
        obs, info = env.reset()
        if regime == "expert":
            eps = noise = 0.0
        else:
            expert = rng.rand() < expert_frac           # a fraction of episodes stay clean
            eps = 0.0 if expert else rng.rand() * eps_max
            noise = 0.0 if expert else rng.rand() * noise_max
        o, a, no, r, su = [], [], [], [], []
        st = np.asarray(obs, np.float32).reshape(-1)
        for _ in range(horizon):
            act = np.asarray(policy.get_action(st.astype(np.float64)), np.float32).reshape(-1)
            act = act + noise * rng.randn(*act.shape).astype(np.float32)
            if rng.rand() < eps:                        # epsilon-random action mixing
                act = rng.uniform(-1.0, 1.0, size=act.shape).astype(np.float32)
            act = np.clip(act, -1.0, 1.0)
            nobs, rew, term, trunc, info = env.step(act)
            nst = np.asarray(nobs, np.float32).reshape(-1)
            o.append(st); a.append(act); no.append(nst)
            r.append(float(rew)); su.append(float(info.get("success", 0.0)))
            st = nst
            if term or trunc:
                break
        if len(a) != horizon:                           # keep the pack dense/fixed-horizon
            continue
        O.append(np.stack(o)); A.append(np.stack(a)); NO.append(np.stack(no))
        R.append(np.asarray(r, np.float32)); SU.append(np.asarray(su, np.float32))
        AG.append(np.zeros((horizon, 3), np.float32))   # placeholder, matching Factor-FB schema
        ep_kept += 1
    env.close()
    return dict(observation=np.stack(O), action=np.stack(A), next_observation=np.stack(NO),
                achieved_goal=np.stack(AG), reward=np.stack(R),
                terminal=np.zeros((ep_kept, horizon), bool),
                timeout=np.stack([np.eye(horizon, dtype=bool)[-1]] * ep_kept),
                success=np.stack(SU).astype(bool))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="mw-<name>, e.g. mw-pick-place")
    ap.add_argument("--regime", choices=["expert", "play-noise"], required=True)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="root; writes <out>/data/metaworld/<task>.npz")
    ap.add_argument("--eps_max", type=float, default=PLAY_DEFAULTS["eps_max"])
    ap.add_argument("--noise_max", type=float, default=PLAY_DEFAULTS["noise_max"])
    ap.add_argument("--expert_frac", type=float, default=PLAY_DEFAULTS["expert_frac"])
    args = ap.parse_args()

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    d = collect(args.task, args.episodes, args.regime,
                args.eps_max, args.noise_max, args.expert_frac, args.seed * 100000)
    outdir = os.path.join(args.out, "data", "metaworld")
    os.makedirs(outdir, exist_ok=True)
    fp = os.path.join(outdir, f"{args.task}.npz")
    np.savez_compressed(fp, **d)
    import json
    meta = dict(task=args.task, family="metaworld", regime=args.regime,
                policy="metaworld-scripted-oracle",
                num_episodes=int(d["action"].shape[0]), episode_length=HORIZON,
                obs_dim=int(d["observation"].shape[-1]), act_dim=int(d["action"].shape[-1]),
                ep_success=float(d["success"].any(1).mean()),
                at_goal_frac=float(d["success"].mean()),
                eps_max=args.eps_max, noise_max=args.noise_max, expert_frac=args.expert_frac)
    json.dump(meta, open(os.path.join(outdir, f"{args.task}.meta.json"), "w"), indent=1)
    print(f"[mw-collect] {args.task} [{args.regime}]: {meta['num_episodes']} eps, "
          f"ep-success={meta['ep_success']:.3f}, at-goal-frac={meta['at_goal_frac']:.3f} -> {fp}",
          flush=True)


if __name__ == "__main__":
    main()
