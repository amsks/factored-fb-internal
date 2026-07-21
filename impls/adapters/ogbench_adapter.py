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
        # OGBench envs carry a fixed set of (init, goal) task_infos, 1-indexed.
        return list(range(1, len(env.unwrapped.task_infos) + 1))

    def evaluate(self, agent, env, task, cfg, num_episodes, num_video):
        stats, _, renders = ogbench_evaluate(
            agent, env, task_id=task, config=cfg,
            num_eval_episodes=num_episodes, num_video_episodes=num_video)
        return {k: float(v) for k, v in dict(stats).items()}, renders
