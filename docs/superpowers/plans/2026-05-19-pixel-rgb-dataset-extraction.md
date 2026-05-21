# Pixel (RGB) Dataset Extraction + FB-Pixel Smoke Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a td_jepa-faithful visual branch to `scripts/extract_ogbench.py`, generate the `visual-cube-single-play-v0` RGB offline dataset onto tmpfs, and launch a ~50k-step FB-pixel smoke run.

**Architecture:** Extend the existing state-only `extract(...)` with a `"visual" in env_name` branch that writes a dummy zero `observation` plus an **HWC** `pixels` array (deliberate deviation from td_jepa's CHW so the already-committed `data/ogbench.py::load_transitions` HWC→CHW `moveaxis` is correct without touching the loader). Add `--out_env` (output folder override → committed loader path) and `--cache_dir` (OGBench download redirect off the 99%-full root disk). All large data lives under `/dev/shm/factored-fb/`.

**Tech Stack:** Python, numpy, OGBench (`ogbench.utils.make_env_and_datasets`), pytest, Hydra, PyTorch.

**Spec:** `docs/superpowers/specs/2026-05-19-pixel-rgb-dataset-extraction-design.md`

**Source of truth for the port:** `/home/adityamohan/git/baselines/td_jepa/scripts/data_processing/ogbench/extract_all.py` (visual branch).

**Conventions (locked):**
- Commit messages: **do not** add a Claude `Co-Authored-By` trailer (repo convention).
- Tests run with `.venv/bin/python -m pytest` (`uv run` is broken in this repo).
- State path of `extract_ogbench.py` must remain byte-for-byte behaviorally unchanged when `--cache_dir`/`--out_env` are not passed and the env is non-visual.

**Known pre-existing failures (NOT regressions, out of scope):** `tests/test_analysis.py`, `tests/test_phase_probe.py`, `tests/test_representation_profile.py` (FAIL) and `tests/test_paper_fragment.py`, `tests/test_plot_sweep.py`, `tests/test_wandb_pull.py` (collection ERROR) — all missing third-party deps (e.g. `pandas`).

---

## File Structure

- **Modify** `scripts/extract_ogbench.py` — add `cache_dir`/`out_env` params to `extract()`, the visual branch, `ALL_ENVS_VISUAL`, the two new CLI args, and an accurate docstring. Single responsibility: monolithic OGBench npz → per-episode npz.
- **Create** `tests/test_extract_ogbench_visual.py` — unit test for the visual branch, the `--out_env` override, and the unchanged state path, using a synthetic in-`tmp_path` monolithic npz (no network).
- **Modify** `docs/superpowers/notes/2026-05-19-pixel-fb-faithfulness.md` — correct the factually-wrong "td_jepa stores HWC" claim.
- **Operational (no source change)** — dataset generation, loader sanity, smoke run are explicit tasks with exact commands and pass/fail criteria.

---

### Task 1: Visual branch + `--out_env` + `--cache_dir` in `extract_ogbench.py`

**Files:**
- Modify: `scripts/extract_ogbench.py`
- Test: `tests/test_extract_ogbench_visual.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_extract_ogbench_visual.py`:

```python
import importlib.util
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "extract_ogbench", REPO / "scripts" / "extract_ogbench.py"
)
extract_ogbench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(extract_ogbench)


def _write_monolithic(path: Path, env_name: str, n_eps=2, ep_len=3, h=4, w=4, c=3, act=2):
    """Two episodes; terminals mark the last step of each episode."""
    T = n_eps * ep_len
    if "visual" in env_name:
        observations = (np.random.rand(T, h, w, c) * 255).astype(np.uint8)  # HWC
    else:
        observations = np.random.rand(T, 7).astype(np.float32)
    actions = np.random.rand(T, act).astype(np.float32)
    qpos = np.random.rand(T, 5).astype(np.float32)
    terminals = np.zeros(T, dtype=np.float32)
    for e in range(n_eps):
        terminals[(e + 1) * ep_len - 1] = 1.0
    np.savez(
        path / (env_name + ".npz"),
        observations=observations,
        actions=actions,
        qpos=qpos,
        terminals=terminals,
    )
    return observations, actions


def test_visual_branch_writes_hwc_pixels_and_dummy_obs_to_out_env(tmp_path):
    src = tmp_path / "cache"
    src.mkdir()
    out = tmp_path / "out"
    env = "visual-cube-single-play-v0"
    observations, actions = _write_monolithic(src, env, n_eps=2, ep_len=3, h=4, w=4, c=3)

    extract_ogbench.extract(
        env_name=env,
        output_folder=str(out),
        dataset_path=str(src),
        out_env="cube-single-play-v0",
    )

    buf = out / "cube-single-play-v0" / "buffer"   # --out_env override honored
    files = sorted(buf.glob("episode_*.npz"))
    assert len(files) == 2, f"expected 2 episodes, got {files}"
    assert files[0].name == "episode_000000_2.npz"  # length = end - start = 2

    ep0 = np.load(files[0])
    # episode 0 spans source rows [0, 1, 2] (3 frames)
    assert ep0["observation"].shape == (3, 1)            # dummy zeros, (L+1, 1)
    assert not ep0["observation"].any()
    assert ep0["pixels"].shape == (3, 4, 4, 3)           # HWC, NO moveaxis
    np.testing.assert_array_equal(ep0["pixels"], observations[0:3])
    assert ep0["action"][0].tolist() == [0.0, 0.0]       # action shift applied
    np.testing.assert_array_equal(ep0["action"][1:], actions[0:2])
    assert ep0["physics"].shape == (3, 5)


def test_state_path_unchanged_no_pixels_key(tmp_path):
    src = tmp_path / "cache"
    src.mkdir()
    out = tmp_path / "out"
    env = "cube-single-play-v0"
    observations, _ = _write_monolithic(src, env, n_eps=1, ep_len=4)

    extract_ogbench.extract(env_name=env, output_folder=str(out), dataset_path=str(src))

    buf = out / "cube-single-play-v0" / "buffer"   # default out_env == env_name
    files = sorted(buf.glob("episode_*.npz"))
    assert len(files) == 1
    ep = np.load(files[0])
    assert "pixels" not in ep.files
    assert ep["observation"].shape == (4, 7)
    np.testing.assert_array_equal(ep["observation"], observations[0:4])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_extract_ogbench_visual.py -q`
Expected: FAIL — `TypeError: extract() got an unexpected keyword argument 'out_env'`

- [ ] **Step 3: Replace the module docstring**

In `scripts/extract_ogbench.py`, replace lines 1-14 (the opening docstring) with:

```python
"""
scripts/extract_ogbench.py — Split OGBench's monolithic .npz datasets into
per-episode .npz files under <output>/<out_env>/buffer/episode_XXXXXX_<len>.npz.

Faithful adaptation of td_jepa's scripts/data_processing/ogbench/extract_all.py.
State envs write a real "observation" (no "pixels" key). Visual envs write a
dummy zero "observation" and a "pixels" array. DEVIATION FROM td_jepa: td_jepa
stores pixels CHW (np.moveaxis(...,-1,1)); we store **HWC** so this repo's
data/ogbench.py::load_transitions (which does its own HWC->CHW moveaxis) is
correct with the loader left untouched.

Usage:
    python scripts/extract_ogbench.py --output_folder datasets
    python scripts/extract_ogbench.py --env cube-single-play-v0 --output_folder datasets
    python scripts/extract_ogbench.py --env visual-cube-single-play-v0 \\
        --out_env cube-single-play-v0 \\
        --output_folder /dev/shm/factored-fb/datasets \\
        --cache_dir /dev/shm/factored-fb/ogbench_cache

--out_env overrides the output env-folder name (default: the env name; matches
td_jepa). --cache_dir redirects the OGBench monolithic download off the default
~/.ogbench cache (needed when the root disk is full). By default downloads via
ogbench.utils.make_env_and_datasets then splits into per-episode files.
"""
```

- [ ] **Step 4: Add `ALL_ENVS_VISUAL` after `ALL_ENVS_STATE`**

In `scripts/extract_ogbench.py`, immediately after the `ALL_ENVS_STATE` list (currently ending at line 40 with `]`), add:

```python


ALL_ENVS_VISUAL = [
    "visual-antmaze-medium-navigate-v0",
    "visual-antmaze-large-navigate-v0",
    "visual-antmaze-medium-stitch-v0",
    "visual-antmaze-large-stitch-v0",
    "visual-antmaze-medium-explore-v0",
    "visual-antmaze-large-explore-v0",
    "visual-cube-single-play-v0",
    "visual-cube-double-play-v0",
    "visual-scene-play-v0",
    "visual-puzzle-3x3-play-v0",
]
```

- [ ] **Step 5: Rewrite `extract()` with `cache_dir`/`out_env` + visual branch**

In `scripts/extract_ogbench.py`, replace the entire `extract` function (currently lines 43-87) with:

```python
def extract(
    env_name: str,
    output_folder: str,
    dataset_path: str | None = None,
    cache_dir: str | None = None,
    out_env: str | None = None,
) -> None:
    from ogbench.utils import DEFAULT_DATASET_DIR, make_env_and_datasets

    if dataset_path is not None:
        dataset_file = Path(dataset_path) / (env_name + ".npz")
    else:
        # Triggers download into the chosen cache (default ~/.ogbench) if needed.
        if cache_dir is not None:
            make_env_and_datasets(env_name, dataset_dir=cache_dir)
            base = Path(cache_dir).expanduser()
        else:
            make_env_and_datasets(env_name)
            base = Path(DEFAULT_DATASET_DIR).expanduser()
        dataset_file = base / (env_name + ".npz")

    print(f"[{env_name}] dataset: {dataset_file}")
    out_env = out_env if out_env is not None else env_name
    out_dir = Path(output_folder) / out_env / "buffer"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{env_name}] writing episodes to: {out_dir}")

    data = np.load(dataset_file)
    data = {k: data[k] for k in data.keys()}

    nz = data["terminals"].ravel().nonzero()[0]
    ends = np.arange(data["terminals"].shape[0])[nz]
    starts = np.concatenate(([0], ends[:-1] + 1))
    lengths = ends - starts

    is_visual = "visual" in env_name
    for ep_idx, (start, length, end) in enumerate(tqdm(list(zip(starts, lengths, ends)))):
        ep = {
            "action": data["actions"][start : end + 1].copy(),
            "physics": data["qpos"][start : end + 1],
            "reward": np.zeros((length + 1, 1), dtype=np.float32),
            "discount": np.ones((length + 1, 1), dtype=np.float32),
        }
        if is_visual:
            # td_jepa-faithful: dummy state obs; pixels carry the images.
            # DEVIATION: store HWC (no np.moveaxis) so load_transitions'
            # HWC->CHW moveaxis is correct and the loader stays untouched.
            ep["observation"] = np.zeros((length + 1, 1), dtype=np.float32)
            ep["pixels"] = data["observations"][start : end + 1]
        else:
            ep["observation"] = data["observations"][start : end + 1]
        # Shift actions so action[t] is the action that led to observation[t].
        # First-step action is set to zero (matches td_jepa convention).
        ep["action"][1:] = ep["action"][:-1]
        ep["action"][0] = 0.0
        # NOTE: td_jepa's extract_all.py leaves discount all-ones and never
        # marks a terminal step; the loader treats every transition as
        # non-terminal (TD bootstrap always uses discount = gamma).
        # Some envs have extra button states concatenated into physics.
        if "button_states" in data:
            ep["physics"] = np.concatenate(
                [ep["physics"], data["button_states"][start : end + 1]], axis=-1
            )

        filename = f"episode_{ep_idx:06d}_{length}.npz"
        np.savez_compressed(out_dir / filename, **ep)
```

- [ ] **Step 6: Add `--out_env` / `--cache_dir` CLI args and thread them**

In `scripts/extract_ogbench.py` `main()` (currently lines 90-103), replace the body with:

```python
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output_folder", default="datasets",
                    help="Where to write <out_env>/buffer/episode_*.npz (default: datasets)")
    ap.add_argument("--dataset_path", default=None,
                    help="Optional: dir containing a pre-downloaded <env>.npz. "
                         "If omitted, ogbench downloads to --cache_dir (or its default).")
    ap.add_argument("--cache_dir", default=None,
                    help="Redirect the OGBench download cache (passed as "
                         "make_env_and_datasets(dataset_dir=...)). Use this when "
                         "the default ~/.ogbench location is on a full disk.")
    ap.add_argument("--out_env", default=None,
                    help="Override the output env-folder name (default: --env). "
                         "E.g. --env visual-cube-single-play-v0 "
                         "--out_env cube-single-play-v0 so the committed loader "
                         "path datasets/<domain>/buffer resolves.")
    ap.add_argument("--env", default=None,
                    help="A single OGBench env (state or visual). "
                         "If omitted, extracts all state envs.")
    args = ap.parse_args()

    envs = [args.env] if args.env else ALL_ENVS_STATE
    for env_name in envs:
        extract(env_name, args.output_folder, args.dataset_path,
                cache_dir=args.cache_dir, out_env=args.out_env)
```

- [ ] **Step 7: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_extract_ogbench_visual.py -q`
Expected: PASS (2 passed)

- [ ] **Step 8: Regression — pixel-FB loader suite still green**

Run: `.venv/bin/python -m pytest tests/test_pixel_data_loading.py tests/test_frame_stack_buffer.py -q`
Expected: PASS (all)

- [ ] **Step 9: Commit**

```bash
git add scripts/extract_ogbench.py tests/test_extract_ogbench_visual.py
git commit -m "feat(pixel-fb): td_jepa-faithful visual branch + --out_env/--cache_dir in extract_ogbench"
```

---

### Task 2: Correct the faithfulness note

**Files:**
- Modify: `docs/superpowers/notes/2026-05-19-pixel-fb-faithfulness.md`

- [ ] **Step 1: Read the current note**

Run: `cat docs/superpowers/notes/2026-05-19-pixel-fb-faithfulness.md`
Locate the `**CHW storage:**` bullet, which currently claims *"td_jepa stores
HWC and relies on its own loader; we normalize at load time"* (factually wrong —
td_jepa's `extract_all.py` does `np.moveaxis(...,-1,1)`, i.e. stores **CHW**).

- [ ] **Step 2: Replace the CHW-storage bullet**

Replace the existing `- **CHW storage:** ...` bullet (the full bullet, through
its closing paren) with:

```markdown
- **Pixel orientation (CHW vs HWC):** td_jepa's `extract_all.py` stores pixels
  **CHW** (`np.moveaxis(observations, -1, 1)`) and its loader does NOT
  re-transpose. This repo instead stores pixels **HWC** at extraction
  (`scripts/extract_ogbench.py` visual branch, no moveaxis) and converts
  HWC->CHW inside `data/ogbench.py::load_transitions`
  (`np.moveaxis(ep["pixels"], -1, 1)`). The orientation reaching `DrQEncoder`
  (`(N, C, H, W)` after frame-stack -> 9 channels) is therefore identical to
  td_jepa; only the on-disk layout and the location of the single transpose
  differ. This is an intentional, documented deviation chosen so the committed
  loader/buffer/config/env wiring is left untouched.
- **State episodes:** td_jepa also writes an all-zero dummy `pixels` key for
  state envs; this repo does NOT (state loader path never reads `pixels`),
  keeping state episodes compact and byte-compatible with prior runs.
```

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/notes/2026-05-19-pixel-fb-faithfulness.md
git commit -m "docs(pixel-fb): correct faithfulness note — HWC storage is our deviation, td_jepa is CHW"
```

---

### Task 3: Generate the `visual-cube-single-play-v0` dataset onto tmpfs

No source changes. This is a one-time data-generation step. The root disk is
99% full, so **both** the OGBench download and the per-episode output go to
`/dev/shm`.

**Files:** none (writes to `/dev/shm/factored-fb/`).

- [ ] **Step 1: Pre-flight — confirm tmpfs headroom**

Run: `df -h /dev/shm && free -g | head -2`
Expected: `/dev/shm` shows ≥ ~50 GB Avail (design measured 276 GB free, 728 GB
RAM). If Avail is unexpectedly small, STOP and report — do not fill tmpfs.

- [ ] **Step 2: Make the scratch dirs**

Run:
```bash
mkdir -p /dev/shm/factored-fb/ogbench_cache /dev/shm/factored-fb/datasets
```
Expected: no output, exit 0.

- [ ] **Step 3: Extract (download + split). Long-running.**

Run (from repo root):
```bash
.venv/bin/python scripts/extract_ogbench.py \
    --env visual-cube-single-play-v0 \
    --out_env cube-single-play-v0 \
    --output_folder /dev/shm/factored-fb/datasets \
    --cache_dir /dev/shm/factored-fb/ogbench_cache
```
Expected: prints `[visual-cube-single-play-v0] dataset: /dev/shm/factored-fb/ogbench_cache/visual-cube-single-play-v0.npz`, then a tqdm bar over episodes, exit 0. (Large download + per-episode write; may take many minutes.)

- [ ] **Step 4: Verify the dataset landed where the loader expects**

Run:
```bash
ls /dev/shm/factored-fb/datasets/cube-single-play-v0/buffer/ | head
ls /dev/shm/factored-fb/datasets/cube-single-play-v0/buffer/ | wc -l
du -sh /dev/shm/factored-fb/datasets/cube-single-play-v0/buffer/
```
Expected: many `episode_000000_<len>.npz` files (count > 0), nonzero size. If the
buffer dir is empty, STOP — re-check Task 3 Steps 2-3.

---

### Task 4: End-to-end loader sanity on real extracted episodes

No source changes. Confirms the committed pixel loader consumes the real
extracted data with the expected CHW orientation/dtype before committing GPU.

**Files:** none.

- [ ] **Step 1: Load a few real episodes through the committed loader**

Run (from repo root):
```bash
.venv/bin/python - <<'EOF'
import glob, numpy as np
from data.ogbench import load_transitions
files = sorted(glob.glob(
    "/dev/shm/factored-fb/datasets/cube-single-play-v0/buffer/episode_*.npz"))[:3]
assert files, "no extracted episodes found"
s = load_transitions(files, obs_type="pixels")
o = s["observation"]
print("observation:", o.shape, o.dtype)
print("next obs   :", s["next"]["observation"].shape)
print("timestep   :", s["timestep"][:8], s["timestep"].dtype)
# CHW after the loader's HWC->CHW moveaxis: (N, C, H, W), uint8.
assert o.dtype == np.uint8, o.dtype
assert o.ndim == 4 and o.shape[1] in (1, 3), o.shape  # per-frame channels
assert "timestep" in s
print("OK: loader consumes extracted pixels as CHW uint8")
EOF
```
Expected: prints shapes with a 4-D `observation`, `dtype uint8`, channel dim 3
(single frame; frame-stacking to 9ch happens later in `DictBuffer.sample`), and
`OK: loader consumes extracted pixels as CHW uint8`. If `observation` is not
4-D or not `uint8`, STOP — the extraction/loader orientation is wrong;
re-examine Task 1 Step 5 (must store HWC, no moveaxis).

---

### Task 5: Launch the FB-pixel smoke run (~50k steps, wandb online)

No source changes. Validates the full pixel pipeline trains end-to-end.

**Files:** none.

- [ ] **Step 1: Sanity-check the Hydra composition without training**

Run (from repo root):
```bash
.venv/bin/python - <<'EOF'
from hydra import compose, initialize
with initialize(version_base="1.3", config_path="configs"):
    cfg = compose(config_name="train",
                  overrides=["domain=visual_cube_single",
                             "data_path=/dev/shm/factored-fb/datasets",
                             "num_train_steps=50000", "use_wandb=true", "seed=0"])
assert cfg.obs_type == "pixels", cfg.obs_type
assert cfg.domain == "cube-single-play-v0", cfg.domain
assert str(cfg.data_path) == "/dev/shm/factored-fb/datasets", cfg.data_path
assert int(cfg.num_train_steps) == 50000, cfg.num_train_steps
assert cfg.use_wandb is True
print("OK: cfg composes; loader will read",
      f"{cfg.data_path}/{cfg.domain}/buffer")
EOF
```
Expected: `OK: cfg composes; loader will read /dev/shm/factored-fb/datasets/cube-single-play-v0/buffer`. If any assert fails, STOP and reconcile the override/config before launching.

- [ ] **Step 2: Launch the smoke run (background, logged)**

Run (from repo root):
```bash
mkdir -p logs
nohup .venv/bin/python train.py \
    domain=visual_cube_single \
    data_path=/dev/shm/factored-fb/datasets \
    seed=0 use_wandb=true num_train_steps=50000 \
    > logs/fb_pixel_smoke_s0.log 2>&1 &
echo "launched pid=$! -> logs/fb_pixel_smoke_s0.log"
```
Expected: prints a pid. The process runs in the background.

- [ ] **Step 2a (executor note):** This is the deferred long-running training
  job. Do NOT busy-wait. Poll with a long-interval check (e.g. `tail -n 40
  logs/fb_pixel_smoke_s0.log` and `pgrep -fa "train.py"`); the run produces
  finite losses and a wandb URL within the first minutes if healthy.

- [ ] **Step 3: Confirm healthy startup (data load + first train steps)**

After the process has had time to load the buffer and start training, run:
```bash
tail -n 50 logs/fb_pixel_smoke_s0.log
grep -m1 -iE "wandb.*run|http.*wandb" logs/fb_pixel_smoke_s0.log || true
grep -iE "error|traceback|cuda out of memory" logs/fb_pixel_smoke_s0.log | tail || true
```
Expected: no traceback / OOM; a wandb run URL printed; loss values that are
finite (not NaN/Inf). If a traceback or OOM appears, STOP, capture the log tail,
and report — do not relaunch blindly.

- [ ] **Step 4: Report and hand the full-run decision to the user**

Summarize for the user: did the smoke reach 50k steps (or is it on track),
are losses finite/decreasing, the wandb URL, and tmpfs/RAM headroom. Then ask
whether to launch the full **1M-step** run (same command, drop
`num_train_steps=50000`). Do NOT start the full run without explicit approval —
the spec gates it on the smoke result.

---

## Self-Review

**Spec coverage:**
- Component 1 (visual branch, HWC, `--out_env`, `--cache_dir`, `ALL_ENVS_VISUAL`, docstring) → Task 1 ✓
- Component 2 (faithfulness-note correction) → Task 2 ✓
- Component 3 (generate dataset to tmpfs, cache redirect off full root disk) → Task 3 ✓
- Component 4 (FB-pixel smoke run, wandb online, ~50k, decide-after) → Task 5 ✓
- Testing: unit test (visual branch + `--out_env` + unchanged state path, no `pixels` key) → Task 1 Step 1; manual end-to-end loader sanity → Task 4; regression of committed pixel suite → Task 1 Step 8 ✓
- Decisions 1-5 (HWC/loader-unchanged, explicit `--out_env`, tmpfs, smoke-first, wandb online) → Tasks 1,3,5 ✓
- Out-of-scope respected: no full run without approval (Task 5 Step 4), no other envs generated, no loader change ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; every command has an expected result and an explicit STOP condition on failure.

**Type/interface consistency:** `extract(env_name, output_folder, dataset_path=None, cache_dir=None, out_env=None)` defined in Task 1 Step 5, called with exactly these kwargs in Task 1 Step 6 (`main`) and in the Task 1 Step 1 test. CLI flags `--env/--out_env/--cache_dir/--output_folder/--dataset_path` defined in Task 1 Step 6 and used verbatim in Tasks 3 & 5. Output path `Path(output_folder)/out_env/"buffer"` (Task 1) matches the loader path `Path(data_path)/cfg.domain/"buffer"` exercised in Tasks 4 & 5 because `--out_env cube-single-play-v0` == `cfg.domain` from `visual_cube_single.yaml`. Episode keys written (`observation`, `pixels`, `action`, `physics`, `reward`, `discount`) match what `load_transitions` reads (`pixels`, `action`, `physics`).
