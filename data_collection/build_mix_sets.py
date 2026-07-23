"""data_collection/build_mix_sets.py — assemble the FIXED 128/64 teacher-pretraining sets
from the original expert demos + the random (eps=1.0) and eps-exploration (eps=0.1) pools.

Composition (nested: the 64-set is a prefix of the 128-set's selections):
    128 = 102 original + 13 random + 13 eps      (~80% / 10% / 10%)
     64 =  51 original +  6 random +  7 eps
Selection is deterministic (np.random.default_rng(128)) so the sets are reproducible.
Original data is READ-only; output goes to a separate collected_mix/set{N} tree.

    <newt>/.venv/bin/python data_collection/build_mix_sets.py
"""
import glob
import json
import os
import re
import shutil

import numpy as np

COLL = "/raid/tongle/rl/fb-data/collected"
MIX = "/raid/tongle/rl/fb-data/collected_mix"
TASKS = {"b2d-center-ctrl": "Box2D", "b2d-center-ctrl-target": "Box2D", "b2d-center-wall": "Box2D",
         "b2d-goal-ctrl-maze": "Box2D", "b2d-goal-ctrl-target": "Box2D",
         "ms-pick-cube": "ManiSkill", "ms-push-cube": "ManiSkill",
         "mw-push": "Metaworld", "mw-pick-place": "Metaworld"}
COMP = {128: (102, 13, 13), 64: (51, 6, 7)}
SELECT_SEED = 128


def _eps_files(d):
    return sorted(glob.glob(f"{d}/buffer/episode_*.npz"),
                  key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[0]))


def main():
    rng = np.random.default_rng(SELECT_SEED)
    for task, fam in TASKS.items():
        orig = _eps_files(f"{COLL}/{fam}__{task}")
        rand = _eps_files(f"{MIX}/random/{fam}__{task}")
        epsl = _eps_files(f"{MIX}/eps/{fam}__{task}")
        picks = {"orig": [orig[i] for i in rng.choice(len(orig), 102, replace=False)],
                 "rand": [rand[i] for i in rng.choice(len(rand), 13, replace=False)],
                 "eps": [epsl[i] for i in rng.choice(len(epsl), 13, replace=False)]}
        for N, (no, nr, ne) in COMP.items():
            out = f"{MIX}/set{N}/{fam}__{task}/buffer"
            os.makedirs(out, exist_ok=True)
            sel = picks["orig"][:no] + picks["rand"][:nr] + picks["eps"][:ne]
            for j, src in enumerate(sel):
                shutil.copy(src, f"{out}/episode_{j:06d}.npz")
            json.dump(dict(env_family=fam.lower(), task=task, num_episodes=N,
                           composition=dict(original=no, random=nr, eps_explore=ne),
                           eps=0.1, select_seed=SELECT_SEED,
                           note="80% original + 10% random + 10% eps(0.1); 64 nested in 128"),
                      open(f"{MIX}/set{N}/{fam}__{task}/meta.json", "w"), indent=1)
        print(f"  {task}: set128 + set64 built", flush=True)
    print("done")


if __name__ == "__main__":
    main()
