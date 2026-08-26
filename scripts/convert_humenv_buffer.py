"""Convert Meta Motivo's released HumEnv replay buffer into this repo's per-episode layout.

    $SCRATCH/.venv-humenv/bin/python scripts/convert_humenv_buffer.py \
        --src $SCRATCH/humenv-assets/metamotivo-S-1/data/buffer.hdf5 \
        --dst $SCRATCH/fb-data/humenv-S1

Writes <dst>/buffer/episode_<i>_<T>.npz, the layout `env/unified.py::load_unified` reads, so
NO new loader is needed -- the domain yaml just points `data_dir` here.

WHY per-episode and not the packed [N,T,...] npz: `load_packed` asserts a fixed horizon, and
this buffer is ragged (16,600 episodes of 300 steps + 50 of 400). The per-episode path keeps
all 16,650 episodes and all 5,000,000 transitions; packing would force dropping or truncating
the long ones.

RUNS IN THE humenv VENV, not the jax one: h5py is only installed there. The training venv
never sees hdf5 -- it reads the npz this writes.

TWO THINGS THE SOURCE SCHEMA DOES THAT THIS FILE NORMALISES:
  * `observation` is (T+1, 358) -- one row longer than `action`. So observations = obs[:-1]
    and next_observations = obs[1:]; there is no separate next_observation field to trust.
  * `reward` is FB-CPR's OWN z-conditioned training reward, NOT a task reward. It is carried
    through so the column exists, but any run that reads it is reading the behaviour agent's
    objective. Leave `keep_dataset_rewards` UNSET in the domain yaml so GCDataset overwrites
    it with the goal-reaching reward -- that is the whole point of the goal-reaching arm.
    A reward-family arm would need a relabelling pass instead, like env/exorl_env.py does.
"""
import argparse
import os

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="path to buffer.hdf5")
    ap.add_argument("--dst", required=True, help="output root; episodes land in <dst>/buffer")
    ap.add_argument("--max_episodes", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    out = os.path.join(args.dst, "buffer")
    os.makedirs(out, exist_ok=True)

    n_tr = 0
    with h5py.File(args.src, "r") as hf:
        keys = sorted(hf.keys(), key=lambda k: int(k.split("_")[1]))
        if args.max_episodes:
            keys = keys[: args.max_episodes]
        for i, k in enumerate(keys):
            g = hf[k]
            obs = g["observation"][:].astype(np.float32)      # (T+1, 358)
            act = g["action"][:].astype(np.float32)           # (T, 69)
            rew = g["reward"][:].astype(np.float32).reshape(-1)
            T = act.shape[0]
            assert obs.shape[0] == T + 1, f"{k}: obs {obs.shape} vs action {act.shape}"
            np.savez(
                os.path.join(out, f"episode_{i}_{T}.npz"),
                observation=obs[:-1],
                next_observation=obs[1:],
                action=act,
                reward=rew,
            )
            n_tr += T
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(keys)} episodes, {n_tr:,} transitions", flush=True)

    print(f"done: {len(keys):,} episodes, {n_tr:,} transitions -> {out}")
    print(f"1M-rule: {'PASSES' if n_tr >= 1_000_000 else 'FAILS'}")


if __name__ == "__main__":
    main()
