"""Run ONE td_jepa trial with a pinned config -- our replication driver.

Their launchers fan a sweep out over lr_b x ortho_coef x 5 seeds and run it
sequentially in-process, which gives no control over which GPU a cell lands on and
no way to pin a single configuration. This imports their BASE_CFG untouched and runs
exactly one trial, so each cell is its own process with its own CUDA_VISIBLE_DEVICES.

Nothing in metamotivo/ or scripts/ is modified: the config that reaches
TrainConfig(...).build().train() is theirs plus the overrides named on the CLI.

  python run_one.py --agent rldp --domain walker --seed 0 --steps 500000
  python run_one.py --agent hilp --domain walker --seed 0 --steps 500000
"""
import argparse, copy, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

p = argparse.ArgumentParser()
p.add_argument("--agent", required=True, choices=["rldp", "hilp"])
p.add_argument("--domain", default="walker")
p.add_argument("--seed", type=int, required=True)
p.add_argument("--steps", type=int, default=500_000)
p.add_argument("--ortho_coef", type=float, default=0.01)  # middle of their sweep {.001,.01,.1}
p.add_argument("--lr_b", type=float, default=1e-4)        # their sweep's first value
p.add_argument("--lr_features", type=float, default=1e-4) # hilp's counterpart
p.add_argument("--data_path", default=os.path.expanduser("~/exorl_data/exorl/datasets"))
p.add_argument("--work_dir", required=True)
p.add_argument("--eval_episodes", type=int, default=10)       # their launcher's value
p.add_argument("--inference_samples", type=int, default=10_000)
args = p.parse_args()

from metamotivo.envs.dmc_tasks import ALL_TASKS
from metamotivo.misc.launcher_utils import flatten, unflatten

if args.agent == "rldp":
    from scripts.train.proprio.launch_rldp_dmc import BASE_CFG
    cfg = copy.deepcopy(BASE_CFG)
    extra = {"agent.train.ortho_coef": args.ortho_coef, "agent.train.lr_b": args.lr_b}
else:
    from scripts.train.proprio.launch_sf_dmc import BASE_AGENT_CFG, BASE_CFG
    cfg = copy.deepcopy(BASE_CFG)
    flat = flatten(cfg)
    flat.update(flatten(BASE_AGENT_CFG["hilp"]))   # exactly what their main() does
    cfg = unflatten(flat)
    extra = {"agent.train.lr_features": args.lr_features}

tasks = ALL_TASKS[args.domain]
flat = flatten(cfg)
flat.update({
    "work_dir": args.work_dir,
    "data.dataset_root": args.data_path,
    "data.domain": args.domain,
    "env.domain": args.domain,
    "env.task": tasks[0],
    "seed": args.seed,
    "num_train_steps": args.steps,
    "use_wandb": False,
    **extra,
})
cfg = unflatten(flat)
cfg["evaluations"] = [{
    "name": "dmc_reward_eval",
    "env": {"name": "dmc", "domain": args.domain, "task": tasks[0]},
    "tasks": tasks,
    "num_episodes": args.eval_episodes,
    "num_inference_samples": args.inference_samples,
}]

from train import TrainConfig

print(f"[run_one] agent={args.agent} domain={args.domain} seed={args.seed} "
      f"steps={args.steps} tasks={tasks}", flush=True)
TrainConfig(**cfg).build().train()
