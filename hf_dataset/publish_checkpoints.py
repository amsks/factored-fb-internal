#!/usr/bin/env python3
"""hf_dataset/publish_checkpoints.py — publish the TD-MPC2 policies used to COLLECT the
Box2D demos to a HuggingFace *model* repo, with a card + manifest.

Metaworld / ManiSkill demos were collected with the **public** multitask checkpoint
`nicklashansen/newt` (https://huggingface.co/nicklashansen/newt) — that is linked in
the card, not re-uploaded. Only the from-scratch Box2D policies are published here.

    <newt>/.venv/bin/python hf_dataset/publish_checkpoints.py \
        --collected /raid/tongle/rl/fb-data/collected \
        --logs      /raid/tongle/rl/fb-data/collect_logs \
        --repo-id   TommyShen/Factor-FB-Offline-tdmpc2

Each Box2D policy is uploaded to  checkpoints/box2d/<task>.pt  (the exact file recorded
in that task's dataset meta.json — the one actually used for inference).
"""
import argparse
import glob
import json
import os
import re
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo

CARD = """---
license: mit
tags:
  - reinforcement-learning
  - tdmpc2
  - model-based-rl
  - robotics
  - box2d
pipeline_tag: reinforcement-learning
---

# Factor-FB-Offline — TD-MPC2 policies (Box2D)

The **expert TD-MPC2 policies used to collect the Box2D split** of
[`TommyShen/Factor-FB-Offline`](https://huggingface.co/datasets/TommyShen/Factor-FB-Offline).
One single-task policy per Box2D task, trained from scratch on the `rlmini_envs` Box2D
tasks (dense potential-shaped reward; `success` measured from the raw sparse condition).

> Metaworld / ManiSkill demos in the dataset were collected with the **public** multitask
> checkpoint [`nicklashansen/newt`](https://huggingface.co/nicklashansen/newt) — use that
> repo for those; it is not mirrored here.

## Checkpoints

{table}

`collection_success` = mean episode success over the collection rollouts (the honest
empirical success rate; the training-time eval used only 4 episodes so was noisier).

## Architecture

TD-MPC2 (`model_size=B`): `latent_dim=512`, `mlp_dim=512`, 5-Q ensemble, single-task
(`task_dim=0`). Observation/action are **zero-padded to newt's multitask size**
(`obs→128`, `action→16`) at inference, then stripped back to the raw dims when writing
the dataset. Files are full agent checkpoints (weights + optimizer), so they also resume
training.

## Load & run (inference)

These load with newt's TD-MPC2 stack. The exact loader used to collect the data is
`data_collection/collect_expert.py` in
[Factored-FB](https://github.com/LUH-AI/Factored-FB) (branch `tl-alignment`):

```bash
hf download TommyShen/Factor-FB-Offline-tdmpc2 checkpoints/box2d/b2d-goal-ctrl.pt

# roll it out / collect more:
<newt>/.venv/bin/python data_collection/collect_expert.py \
    task=b2d-goal-ctrl num_envs=50 +num_episodes=100 +success_only=true \
    +ckpt=<downloaded>.pt +out=<dir>
```

Loading detail: build a `model_size=B` single-task `WorldModel`, drop the checkpoint's
`_action_masks`/`_task_emb` buffers, and `load_state_dict(..., strict=False)` (see
`_robust_load` in `collect_expert.py`). `checkpoint_manifest.json` lists per-task
`obs_dim`/`act_dim`.
"""


def collection_success(logdir: str, task: str):
    fp = Path(logdir) / f"{task}.log"
    if not fp.exists():
        return None
    vals = [float(x) for x in re.findall(r"episode_success=([0-9.]+)", fp.read_text())]
    return round(sum(vals) / len(vals), 3) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collected", default="/raid/tongle/rl/fb-data/collected")
    ap.add_argument("--logs", default="/raid/tongle/rl/fb-data/collect_logs")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    api = HfApi()
    create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    entries = []
    for d in sorted(glob.glob(f"{args.collected}/Box2D__b2d-*")):
        m = json.load(open(f"{d}/meta.json"))
        task, ck = m["task"], m["checkpoint"]
        assert os.path.exists(ck), f"missing checkpoint {ck}"
        repo_path = f"checkpoints/box2d/{task}.pt"
        print(f"[publish] uploading {task} ({os.path.getsize(ck)/1e6:.0f} MB) -> {repo_path}")
        api.upload_file(path_or_fileobj=ck, path_in_repo=repo_path,
                        repo_id=args.repo_id, repo_type="model",
                        commit_message=f"add {task} TD-MPC2 policy")
        entries.append({
            "task": task, "file": repo_path,
            "algo": "tdmpc2", "model_size": "B",
            "obs_dim": m["obs_dim"], "act_dim": m["act_dim"],
            "obs_pad": 128, "act_pad": 16, "task_dim": 0,
            "collection_success": collection_success(args.logs, task),
            "source_checkpoint": ck,
        })

    manifest = {
        "algo": "tdmpc2", "model_size": "B", "task_dim": 0,
        "dataset": "TommyShen/Factor-FB-Offline",
        "note": "single-task Box2D policies; obs/action zero-padded to newt's multitask "
                "size (128/16) at inference. Metaworld/ManiSkill used nicklashansen/newt.",
        "checkpoints": entries,
    }
    rows = "\n".join(
        f"| `{e['task']}` | {e['obs_dim']} | {e['act_dim']} | "
        f"{e['collection_success'] if e['collection_success'] is not None else '—'} | "
        f"`{e['file']}` |"
        for e in entries)
    table = ("| task | obs_dim | act_dim | collection_success | file |\n"
             "|---|---|---|---|---|\n" + rows)

    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2))
        (Path(tmp) / "README.md").write_text(CARD.format(table=table))
        api.upload_folder(folder_path=tmp, repo_id=args.repo_id, repo_type="model",
                          commit_message="add model card + checkpoint manifest")

    print(f"[publish] done -> https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
