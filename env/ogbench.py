"""OGBench adapter — the behaviour impls/main.py had before adapters existed, moved behind
the DomainAdapter seam.

By DEFAULT it is byte-for-byte the original OGBench behaviour: one eval row per task_id
(the collaborator's runs are unaffected). Set `eval_aggregate: true` in the domain config to
instead fold the task_infos into a single mean-success row (what our unified campaign +
agg-jsonl reporting expect, so cube-single lines up with the Box2D/MW/MS tasks).
"""
from impls.utils.env_utils import make_env_and_datasets
from impls.utils.evaluation import evaluate as ogbench_evaluate


class OGBenchAdapter:
    name = "ogbench"

    def make_env_and_datasets(self, cfg):
        env, tr, va = make_env_and_datasets(cfg["env_name"], cfg.get("frame_stack"))
        tck = cfg.get("teacher_ckpt")
        if tck:
            import numpy as np
            from env.teacher import load_teacher
            from impls.utils.datasets import FBDataset
            t = load_teacher(tck)
            d = dict(tr)
            d["teacher_actions"] = t.relabel(np.asarray(d["observations"], np.float32))
            tr = FBDataset.create(**d)
            print(f"[env] relabeled with frozen teacher {tck}", flush=True)
        return env, tr, va

    def eval_tasks(self, env, cfg):
        if cfg.get("eval_aggregate"):
            return [None]                                    # one folded mean-success row
        return list(range(1, len(env.unwrapped.task_infos) + 1))  # DEFAULT: per-task (original)

    def evaluate(self, agent, env, task, cfg, num_episodes, num_video):
        if task is not None:
            # Original per-task_id behaviour, unchanged.
            stats, _, renders = ogbench_evaluate(
                agent, env, task_id=task, config=cfg,
                num_eval_episodes=num_episodes, num_video_episodes=num_video)
            return {k: float(v) for k, v in dict(stats).items()}, renders
        # Aggregate: mean success over all task_infos, one scalar (no at-goal ticks: Box2D-only).
        import numpy as np
        n_tasks = len(env.unwrapped.task_infos)
        succ, renders = [], []
        for tid in range(1, n_tasks + 1):
            stats, _, r = ogbench_evaluate(
                agent, env, task_id=tid, config=cfg,
                num_eval_episodes=num_episodes, num_video_episodes=num_video if tid == 1 else 0)
            succ.append(float(dict(stats).get("success", next(iter(dict(stats).values())))))
            renders += list(r)
        return {"success": float(np.mean(succ))}, renders
