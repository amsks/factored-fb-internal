---
license: mit
task_categories:
  - reinforcement-learning
tags:
  - offline-rl
  - imitation-learning
  - robotics
  - manipulation
  - forward-backward
  - tdmpc2
  - metaworld
  - maniskill
  - box2d
pretty_name: FactoredFB Testbed Expert Demos
size_categories:
  - 1M<n<10M
---

# FactoredFB Testbed — Expert Demonstrations

Offline expert-demonstration datasets for the **Factored-FB** testbed, spanning three
environment families (Metaworld, ManiSkill, Box2D). Each task ships ~1000 episodes
collected with an **expert TD-MPC2 policy**, stored in a single **unified format** so
one offline loader consumes all of them.

Intended for offline goal-conditioned / factored RL (Forward–Backward representations,
FB+FlowBC, GCIQL, CRL, …) and imitation learning.

## Families & tasks

| Family | Tasks | Steps/ep | Freq | Action | Policy | Filter |
|---|---|---|---|---|---|---|
| **Metaworld** | mw-assembly, mw-basketball, mw-bin-picking, mw-box-close, mw-button-press | 100 | 40 Hz | 4 | newt TD-MPC2 (pretrained) | as-collected (~0.9–1.0 success) |
| **ManiSkill** | ms-pick-cube, ms-push-cube, ms-stack-cube, ms-poke-cube, ms-lift-peg | 25 | 10 Hz | — | newt TD-MPC2 (pretrained) | as-collected (~0.9–1.0 success) |
| **Box2D** (loco) | center-ctrl, goal-ctrl, goal-ctrl-maze, center-wall-navigation | 200 | — | 2 | TD-MPC2 (trained from scratch) | **success-only** |
| **Box2D** (manip) | goal-ctrl-target, center-ctrl-target-hard, center-ctrl-target, center-wall | 200 | — | 2 | TD-MPC2 (trained from scratch) | **success-only** |

Exact per-task counts, dims, and success rates live in [`manifest.json`](manifest.json).

- **Metaworld / ManiSkill** policies are the pretrained multitask experts from
  [`nicklashansen/newt`](https://huggingface.co/nicklashansen/newt).
- **Box2D** policies were trained from scratch (single-task TD-MPC2) on the
  `rlmini_envs` Box2D tasks; only successful episodes are kept.

## Format

Repository layout:

```
data/<family>/<task>.npz     # one file per task, arrays stacked over episodes
data/<family>/<task>.meta.json
manifest.json                # global + per-task index/stats
```

Each `<task>.npz` holds the 8 unified fields (fixed horizon `T` per task):

| field | shape | dtype |
|---|---|---|
| `observation`, `next_observation` | `[N, T, obs_dim]` | float32 |
| `action` | `[N, T, act_dim]` | float32 |
| `achieved_goal` | `[N, T, 3]` | float32 |
| `reward`, `terminal`, `timeout`, `success` | `[N, T]` | float32 / bool |

## Load

```python
import numpy as np
from huggingface_hub import hf_hub_download

fp = hf_hub_download("<username>/factoredfb-testbed-demos",
                     "data/maniskill/ms-pick-cube.npz", repo_type="dataset")
z = np.load(fp)
obs, act, rew = z["observation"], z["action"], z["reward"]   # [N,T,·], [N,T,·], [N,T]
```

Helper (`list_tasks`, `load_task`) in `hf_dataset/load_hf_dataset.py` of the
[Factored-FB](https://github.com/LUH-AI/Factored-FB) repo (branch `tl-alignment`).

## Caveats

- **`achieved_goal` is a placeholder (zeros).** FB *training* does not use it; a goal
  relabeler fills it downstream. Do not treat it as a real achieved-goal signal.
- **The goal is baked into `observation`** for Metaworld/ManiSkill (newt obs), so these
  are not the classic zero-shot-goal FB observations — factor them out if you need that.
- **Box2D `reward` is potential-shaped** (dense, training signal only); the `success`
  field is computed from the raw sparse condition and is the authoritative metric.

## Provenance / reproduce

Collected with `data_collection/collect_expert.py` + `run_collection*.sh` and packed with
`hf_dataset/pack_and_manifest.py` in the Factored-FB repo. Re-run those + `push_to_hub.py`
to extend or refresh the dataset.
