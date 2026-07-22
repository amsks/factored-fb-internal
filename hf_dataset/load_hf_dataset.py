#!/usr/bin/env python3
"""hf_dataset/load_hf_dataset.py — load a packed task from the HuggingFace hub (or a
local export) back into per-episode trajectories.

    from hf_dataset.load_hf_dataset import load_task, list_tasks

    # from the hub (downloads + caches the single task file):
    eps = load_task("ms-pick-cube", repo_id="<username>/factoredfb-testbed-demos")
    # or from a local export:
    eps = load_task("b2d-goal-ctrl", export="/raid/tongle/rl/fb-data/hf_export")

    obs, act = eps["observation"], eps["action"]   # [N, T, obs_dim], [N, T, act_dim]

Returns a dict of stacked arrays with the 8 unified fields. To feed Factored-FB
training, point train.py at the *original* per-episode collected root instead
(data_format=unified, build_env=false) — this helper is for inspection / external use.
"""
import json
from pathlib import Path

import numpy as np

FAMILY_OF = {"ms-": "maniskill", "mw-": "metaworld", "b2d-": "box2d", "p2d-": "pusher2d"}


def _family(task: str) -> str:
    for pre, fam in FAMILY_OF.items():
        if task.startswith(pre):
            return fam
    raise ValueError(f"unknown task prefix: {task!r}")


def load_task(task: str, *, repo_id: str | None = None, export: str | None = None) -> dict:
    fam = _family(task)
    rel = f"data/{fam}/{task}.npz"
    if export is not None:
        fp = Path(export) / rel
    else:
        assert repo_id, "pass repo_id (hub) or export (local dir)"
        from huggingface_hub import hf_hub_download
        fp = hf_hub_download(repo_id, rel, repo_type="dataset")
    z = np.load(fp, allow_pickle=True)
    return {k: z[k] for k in z.files}


def list_tasks(*, repo_id: str | None = None, export: str | None = None) -> list:
    if export is not None:
        man = json.loads((Path(export) / "manifest.json").read_text())
    else:
        assert repo_id, "pass repo_id (hub) or export (local dir)"
        from huggingface_hub import hf_hub_download
        man = json.loads(Path(hf_hub_download(repo_id, "manifest.json",
                                              repo_type="dataset")).read_text())
    return [(t["family"], t["task"], t["num_episodes"]) for t in man["tasks"]]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default="/raid/tongle/rl/fb-data/hf_export")
    ap.add_argument("--task", default=None)
    args = ap.parse_args()
    if args.task:
        eps = load_task(args.task, export=args.export)
        print({k: v.shape for k, v in eps.items()})
    else:
        for fam, task, n in list_tasks(export=args.export):
            print(f"{fam:9s} {task:26s} {n} episodes")
