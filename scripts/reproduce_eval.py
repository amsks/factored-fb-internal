"""Reproduce the training-time eval on the final checkpoints (plain vs encmatch).

If encmatch scores ~0 here too => policy genuinely bad. If it scores ~0.5 =>
the training-time 0% was an eval-harness artifact, not the learned policy.
"""
import sys
import numpy as np

from evals.analysis import load_cfg, build_env_and_agent, load_checkpoint
from data.ogbench import load_ogbench_dataset
from evals.ogbench import OGBenchEvaluator

RUNS = "/dev/shm/factored-fb/runs"
DATA = "/dev/shm/factored-fb/datasets"
CASES = {
    "plain":    f"{RUNS}/20260607_070225__fb_state_plain__s0",
    "encmatch": f"{RUNS}/20260607_062100__fb_state_encmatch__s0",
}
N_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def run(name, rundir):
    cfg = load_cfg(f"{rundir}/.hydra/config.yaml", device="cpu")
    env, agent, obs_space, action_dim = build_env_and_agent(cfg)
    load_checkpoint(agent, f"{rundir}/checkpoints/final.pt")
    agent.model.train(False)
    buffer = load_ogbench_dataset(
        domain=cfg.domain, data_path=DATA, obs_type=cfg.obs_type,
        load_n_episodes=cfg.load_n_episodes, frame_stack=int(cfg.get("frame_stack", 1)))
    ev = OGBenchEvaluator(
        domain=cfg.domain, agent=agent, offline_buffer=buffer,
        relabel_size=cfg.eval_relabel_size, n_episodes=N_EP,
        shift_reward=cfg.eval_shift_reward, obs_type=cfg.obs_type,
        frame_stack=int(cfg.get("frame_stack", 1)), seed=cfg.seed, device="cpu",
        save_videos=False, use_wandb=False)
    m = ev.run(step=0)
    print(f"\n===== {name}: eval/success = {m.get('eval/success'):.3f} (n_ep={N_EP}/task) =====")
    for k, v in sorted(m.items()):
        if k.endswith("/success"):
            print(f"   {k:45s} {v:.3f}")


if __name__ == "__main__":
    for name, rundir in CASES.items():
        run(name, rundir)
