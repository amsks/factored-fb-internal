"""OGBench adapter — the behaviour impls/main.py had before adapters existed, moved behind
the DomainAdapter seam with no change in semantics.
"""
from impls.utils.env_utils import make_env_and_datasets
from impls.utils.evaluation import evaluate as ogbench_evaluate


class OGBenchAdapter:
    name = "ogbench"

    def make_env_and_datasets(self, cfg):
        return make_env_and_datasets(cfg["env_name"], cfg.get("frame_stack"))

    def eval_tasks(self, env, cfg):
        # One logical eval per domain (like the unified adapter) so main.py writes ONE agg
        # jsonl line per step; the OGBench multi-task loop is folded into evaluate() below.
        return [None]

    def evaluate(self, agent, env, task, cfg, num_episodes, num_video):
        # Mean success over the env's fixed (init, goal) task_infos -> the standard
        # OGBench cube-single number, as a single 'success' scalar. No at-goal ticks
        # (that metric is Box2D-only).
        import numpy as np
        n_tasks = len(env.unwrapped.task_infos)
        succ, renders = [], []
        for tid in range(1, n_tasks + 1):
            stats, _, r = ogbench_evaluate(
                agent, env, task_id=tid, config=cfg,
                num_eval_episodes=num_episodes,
                num_video_episodes=num_video if tid == 1 else 0)
            stats = dict(stats)
            succ.append(float(stats.get("success", next(iter(stats.values())))))
            renders += list(r)
        return {"success": float(np.mean(succ))}, renders
