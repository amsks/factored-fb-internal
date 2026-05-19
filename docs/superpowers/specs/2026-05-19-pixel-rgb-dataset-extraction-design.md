# Pixel (RGB) Dataset Extraction + FB-Pixel Smoke Run — Design Spec

**Date:** 2026-05-19
**Status:** Approved (pending written-spec review)
**Branch:** `gciql-pixel-integration`

## Goal

Enable an FB-pixel training run on `visual-cube-single-play-v0` by producing
the offline RGB dataset the (already-committed) pixel pipeline needs. The pixel
code, loader, buffer, config, and runner were landed in the
`2026-05-19-pixel-fb-gciql-integration` plan, but **no pixel dataset exists and
no script produces one** — both `scripts/extract_ogbench.py` and
`scripts/download_data.py` are state-only. This spec adds a faithful replica of
td_jepa's visual extraction and runs a smoke training job.

## Source of Truth

`/home/adityamohan/git/baselines/td_jepa/scripts/data_processing/ogbench/extract_all.py`
— specifically its visual branch:

```python
if "visual" in env_name:
    episode["observation"] = np.zeros((length + 1, 1), dtype=np.float32)
    episode["pixels"]      = np.moveaxis(data["observations"][start:end+1], -1, 1)  # CHW
else:
    episode["observation"] = data["observations"][start:end+1]
    episode["pixels"]      = np.zeros((length + 1, 1), dtype=np.float32)
```

`scripts/extract_ogbench.py` is already a faithful adaptation of this script
minus the visual branch. This spec adds the visual branch.

## Hard Environment Constraints (discovered during design)

- **Root disk `/` is 99% full (~5.2 GB free).** Writing the dataset into the
  repo (`datasets/`) OR letting OGBench download into its default cache
  (`~/.ogbench/data`, on root) would fill the disk and fail.
- **`/dev/shm` (tmpfs) has 276 GB free; box has 728 GB RAM.** All large data —
  the OGBench monolithic download AND the extracted per-episode files — MUST
  live under `/dev/shm`.
- 8× idle NVIDIA L4 (23 GB each) available for the run.

## Decisions (locked with user)

1. **HWC storage, loader unchanged.** Deliberate deviation from td_jepa: store
   `pixels` as **HWC `[T, H, W, C]`** (i.e. `data["observations"]` *without*
   the `np.moveaxis(-1, 1)`). Rationale: the committed loader
   `data/ogbench.py::load_transitions` already does `np.moveaxis(ep["pixels"],
   -1, 1)` (HWC→CHW). Storing HWC keeps the committed loader / buffer / config /
   env wiring **completely untouched**. (td_jepa stores CHW because its own
   loader does *not* re-transpose; we choose HWC to fit this repo's loader.)
2. **Explicit `--out_env` override** (no implicit prefix-stripping). The script
   downloads OGBench `visual-cube-single-play-v0` but writes per-episode files
   to a configurable output env-folder. Default `--out_env` == `env_name`
   (exact td_jepa behavior). We invoke with `--out_env cube-single-play-v0` so
   the committed loader path `Path(data_path)/cfg.domain/"buffer"` (with
   `cfg.domain == cube-single-play-v0` from `visual_cube_single.yaml`) resolves.
3. **Scratch on tmpfs.** `--output_folder` and a new `--cache_dir` both default
   to repo-relative behavior but are invoked pointing at `/dev/shm/factored-fb/`.
4. **Smoke first, decide after.** Run ~50k-step FB-pixel smoke; decide on the
   full 1M run based on its result.
5. **wandb online.** `use_wandb=true`, project `factored-fb` (train.py default).

## Components

### Component 1 — `scripts/extract_ogbench.py` visual support

Modify `extract(env_name, output_folder, dataset_path, cache_dir, out_env)`:

- **New `--cache_dir` arg** (default `None`). When set and `dataset_path` is
  `None`, pass it through as
  `make_env_and_datasets(env_name, dataset_dir=cache_dir)` and read the
  monolithic npz from `Path(cache_dir).expanduser()/(env_name + ".npz")`
  instead of `DEFAULT_DATASET_DIR`. This redirects the OGBench download off the
  full root disk. State path behavior with `cache_dir=None` is unchanged.
- **New `--out_env` arg** (default `None` → use `env_name`). Output dir becomes
  `Path(output_folder)/out_env/"buffer"`.
- **Visual branch** in the per-episode loop, faithful to td_jepa except HWC:

  ```python
  if "visual" in env_name:
      ep["observation"] = np.zeros((length + 1, 1), dtype=np.float32)
      ep["pixels"]      = data["observations"][start:end + 1]   # HWC, no moveaxis
  else:
      ep["observation"] = data["observations"][start:end + 1]
      # state path: no "pixels" key (unchanged; matches existing state episodes
      # and the loader's obs_type=="state" path which never reads "pixels")
  ```

  All other per-episode fields unchanged from current code: action shift
  (`ep["action"][1:] = ep["action"][:-1]; ep["action"][0] = 0.0`), `physics`
  from `qpos` (+ `button_states` if present), `discount` all-ones,
  `np.savez_compressed`, filename `episode_{ep_idx:06d}_{length}.npz`.

  > Note: td_jepa also writes a dummy all-zero `pixels` for state envs. We do
  > NOT, to keep existing state episodes byte-compatible and avoid bloating
  > state datasets. The state loader path never reads `pixels`, so this is safe
  > and is a documented deviation.

- **Add `ALL_ENVS_VISUAL`** list (mirrors td_jepa's visual env names) and allow
  `--env visual-...`. `main()` extracts `ALL_ENVS_STATE` by default (unchanged);
  `--env` selects a single env (state or visual).
- **Docstring**: replace "State-based only (no pixel branch)" with accurate
  text describing the visual branch and the HWC deviation.

### Component 2 — Faithfulness note correction

`docs/superpowers/notes/2026-05-19-pixel-fb-faithfulness.md` currently claims
*"td_jepa stores HWC and relies on its own loader; we normalize at load time"*.
This is **factually wrong** — td_jepa stores CHW. Correct it to: td_jepa stores
CHW; this repo stores **HWC** at extraction and converts HWC→CHW in
`load_transitions`; net pixel orientation reaching `DrQEncoder` is identical.

### Component 3 — Generate the dataset (one-time, manual command)

```
mkdir -p /dev/shm/factored-fb/ogbench_cache /dev/shm/factored-fb/datasets
.venv/bin/python scripts/extract_ogbench.py \
    --env visual-cube-single-play-v0 \
    --out_env cube-single-play-v0 \
    --output_folder /dev/shm/factored-fb/datasets \
    --cache_dir   /dev/shm/factored-fb/ogbench_cache
```

Produces `/dev/shm/factored-fb/datasets/cube-single-play-v0/buffer/episode_*.npz`.

### Component 4 — FB-pixel smoke run

```
.venv/bin/python train.py domain=visual_cube_single \
    data_path=/dev/shm/factored-fb/datasets \
    seed=0 use_wandb=true num_train_steps=50000
```

`cfg.domain == cube-single-play-v0` (from `visual_cube_single.yaml`) →
loader resolves `/dev/shm/factored-fb/datasets/cube-single-play-v0/buffer/`.
`create_ogbench_env` prepends `visual-` for the env. After the smoke completes,
report metrics/stability and let the user decide on the full 1M run.

## Data Flow

```
OGBench (visual-cube-single-play-v0.npz, HWC uint8 images in "observations")
  │  make_env_and_datasets(dataset_dir=/dev/shm/.../ogbench_cache)
  ▼
extract_ogbench.py  → split by terminals → per-episode npz
  observation = zeros[(L+1,1)]   (dummy, td_jepa-faithful)
  pixels      = observations[start:end+1]  (HWC, NO moveaxis)
  action      = shifted; physics = qpos(+buttons); discount = ones
  ▼  /dev/shm/factored-fb/datasets/cube-single-play-v0/buffer/episode_*.npz
load_transitions(obs_type="pixels")  → np.moveaxis(pixels,-1,1) → CHW [T,C,H,W]
  ▼
DictBuffer(frame_stack=3) → sample-time channel-stack → 9ch CHW
  ▼
DrQEncoder (Conv2d in_channels=9)  → FB-pixel training
```

## Testing

- **Unit test** `tests/test_extract_ogbench_visual.py`: build a tiny synthetic
  monolithic `.npz` in `tmp_path` (`observations` HWC `[N,4,4,3]` uint8,
  `actions`, `qpos`, `terminals` with 2 episodes), call `extract(...)` with a
  `visual-` env name and `dataset_path`/`out_env`/`output_folder` pointed at
  `tmp_path`. Assert:
  - output dir is `tmp_path/<out_env>/buffer` (override honored),
  - episode `observation` is zeros shape `(L+1, 1)`,
  - episode `pixels` is HWC (last axis == 3, equals source slice — no moveaxis),
  - action shift applied (`action[0] == 0`),
  - filename pattern `episode_000000_<len>.npz`.
  - A second assertion for the **state** path (non-visual env name): episode has
    real `observation`, no `pixels` key.
- **End-to-end sanity** (manual, post-extraction, not a pytest): load 2–3 real
  extracted episodes via `load_transitions(files, obs_type="pixels")`; assert
  `observation.shape == (n, 3, 64, 64)`, dtype `uint8`, `timestep` present.
- **Regression**: existing pixel-FB suite stays green
  (`tests/test_pixel_data_loading.py` etc. — state path untouched).
- No training assertions; the smoke run is validated by "trains N steps without
  crashing + produces finite losses", reported to the user.

## Out of Scope

- Full 1M FB-pixel run (gated on smoke result + user decision).
- GCIQL-pixel run / FB-vs-GCIQL sweep (separate deferred follow-ups).
- Visual extraction for envs other than `visual-cube-single-play-v0` (the
  script will *support* them via `ALL_ENVS_VISUAL`/`--env`, but only
  cube-single is generated/validated here).
- Persisting the dataset beyond tmpfs (it is RAM-backed and ephemeral; a
  re-extract is the recovery path — acceptable for a smoke).

## Risks

- **tmpfs is RAM-backed & ephemeral**: dataset lost on reboot; consumes RAM
  (visual-cube-single ≈ low tens of GB; 728 GB RAM, 276 GB shm free — ample).
- **Download size/time**: the monolithic visual npz is large; one-time, slow.
- **OGBench API**: relies on `make_env_and_datasets(dataset_dir=...)` honoring
  the override — verified present in the installed `ogbench.utils` signature.
