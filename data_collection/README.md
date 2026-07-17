# data_collection/ — offline dataset collection (isolated)

Self-contained collection infra for the Factored-FB testbed. Kept in its own folder
so it does not touch the training code. Output is the **unified format**
(`data/unified.py`), which FB trains on offline (`train.py build_env=false`).

## What produces data

| Backend | Policy | Where it runs | Script |
|---|---|---|---|
| **Metaworld / ManiSkill** | pretrained newt TD-MPC2 experts (HuggingFace `nicklashansen/newt`) | **newt's venv** (`/raid/tongle/rl/newt/.venv`) | `collect_expert.py` |
| **Box2D / Pusher2D** | TD-MPC2 trained from scratch (no pretrained ckpt) | (trained separately, then collected the same way) | — |

`collect_expert.py` reuses newt's TD-MPC2 agent + env-making via an `envs`-package
stub that (a) avoids newt's eager all-backend imports and (b) avoids clashing with
Factored-FB's own `envs/` package. It rolls the planner over `num_envs` parallel
episodes and writes one unified `.npz` per episode. newt pads obs/action to a common
multitask size; we **strip the trailing zero-padding** back to the raw env dims.

## Usage

```bash
# one task (run in newt's venv, from the Factored-FB repo root):
/raid/tongle/rl/newt/.venv/bin/python data_collection/collect_expert.py \
    task=ms-pick-cube num_envs=50 +num_episodes=1000 +success_only=false \
    +checkpoint_dir=<newt-ckpt-snapshot-dir> +out=/raid/tongle/rl/fb-data/collected

# all validated tasks across GPUs:
bash data_collection/run_collection.sh
```

Output: `<out>/<ManiSkill|Metaworld>__<task>/buffer/episode_*.npz` + `meta.json`.

## Then train FB on it (offline)

```bash
# build_env=false derives obs/action dims from the data; no in-process env, no eval.
.venv/bin/python train.py domain=collected_ms_pick_cube \
    data_path=/raid/tongle/rl/fb-data/collected num_train_steps=... eval_every=0
```

(See `configs/domain/collected_ms_pick_cube.yaml` for the per-task training config.)

## Notes

- Recorded per step: `observation, action, next_observation, reward, terminal,
  timeout, success`. `achieved_goal` is a **placeholder (zeros)** — FB training does
  not use it; the goal evaluator/relabeler fills it properly later.
- ManiSkill episodes are 25 steps (10 Hz), Metaworld 100 steps (40 Hz) — so 1000
  episodes ≈ 25k / 100k transitions respectively. Scale `num_episodes` up for a
  full ~1M-transition dataset.
- venvs: collection needs newt's venv (mani_skill / metaworld fork); FB training uses
  Factored-FB's `.venv`. Data crosses via the npz files.
