"""data_collection/build_clean_sets.py — the CLEAN teacher-pretraining sets: 128/64 demos,
100% ORIGINAL expert (no random, no eps-exploration), for all 10 dev tasks INCLUDING
cube-single (OGBench). Fixed selection (rng seed 128), 64 nested in 128. The no-noise
counterpart of build_mix_sets.py -> collected_mix/set{N}_clean/.
"""
import glob, json, os, re, shutil
import numpy as np

COLL = "/raid/tongle/rl/fb-data/collected"
CUBE = "/raid/tongle/rl/fb-data/unified/cube-single-play-v0"
MIX = "/raid/tongle/rl/fb-data/collected_mix"
TASKS = {"b2d-center-ctrl": "Box2D", "b2d-center-ctrl-target": "Box2D", "b2d-center-wall": "Box2D",
         "b2d-goal-ctrl-maze": "Box2D", "b2d-goal-ctrl-target": "Box2D",
         "ms-pick-cube": "ManiSkill", "ms-push-cube": "ManiSkill",
         "mw-push": "Metaworld", "mw-pick-place": "Metaworld", "cube-single": "OGBench"}


def _eps(d):
    return sorted(glob.glob(f"{d}/buffer/episode_*.npz"),
                  key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[0]))


def main():
    rng = np.random.default_rng(128)
    for task, fam in TASKS.items():
        src = CUBE if task == "cube-single" else f"{COLL}/{fam}__{task}"
        files = _eps(src)
        pick = [files[i] for i in rng.choice(len(files), 128, replace=False)]
        for N in (128, 64):
            out = f"{MIX}/set{N}_clean/{fam}__{task}/buffer"
            os.makedirs(out, exist_ok=True)
            for j, s in enumerate(pick[:N]):
                shutil.copy(s, f"{out}/episode_{j:06d}.npz")
            json.dump(dict(env_family=fam.lower(), task=task, num_episodes=N,
                           composition=dict(original=N, random=0, eps_explore=0), select_seed=128,
                           note="CLEAN: 100% original demos, no noise/disturbance"),
                      open(f"{MIX}/set{N}_clean/{fam}__{task}/meta.json", "w"), indent=1)
        print(f"  {task}: clean set128+set64")


if __name__ == "__main__":
    main()
