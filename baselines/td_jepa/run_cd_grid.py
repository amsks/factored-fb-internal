"""Run td_jepa's own HILP OGBench sweep on cube-double only, one trial per process.

Reuses scripts/train/proprio/launch_sf_ogbench.py unchanged (BASE_CFG, BASE_AGENT_CFG["hilp"],
the reward-inference evaluation block); only the sweep is narrowed to cube-double and the first
3 of td_jepa's 10 seeds, and launch_trials is wrapped so `--index i` runs trial i alone.

    .venv/bin/python run_cd_grid.py --list            # print the 12 trials
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python run_cd_grid.py --index 0
"""

import argparse

import scripts.train.proprio.launch_sf_ogbench as L

SEEDS = [3917, 3502, 8948]  # first three of td_jepa's sweep seeds


def sweep_hilp_cd():
    return {
        "seed": SEEDS,
        "env.domain": ["cube-double-play-v0"],
        "agent.train.bc_coeff": [3.0],
        "agent.train.lr_features": [1.0e-4, 1.0e-5],
        "agent.train.prob_random_goal": [0.375, 0.5],
    }


p = argparse.ArgumentParser()
p.add_argument("--index", type=int, default=None)
p.add_argument("--list", action="store_true")
p.add_argument("--data_path", default="/home/adityamohan/tdj_ogbench/datasets")
p.add_argument("--workdir_root", default="/home/adityamohan/tdj_ogbench/results/hilp_cd")
a = p.parse_args()

L.sweep_hilp_cd = sweep_hilp_cd
_launch = L.launch_trials


def _one(base_cfg, trials, first_only, dry, slurm, exca):
    if a.list:
        for i, t in enumerate(trials):
            print(i, t["seed"], t["agent.train.lr_features"], t["agent.train.prob_random_goal"], t["work_dir"])
        return
    return _launch(base_cfg, [trials[a.index]], False, False, False, False)


L.launch_trials = _one
L.main(L.LaunchArgs(data_path=a.data_path, workdir_root=a.workdir_root, sweep_config="sweep_hilp_cd", sf_agent="hilp"))
