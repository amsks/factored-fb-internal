# Pixel Diagnostics Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the deep diagnostic profilers (value-gradient, B-resolution, coverage, phase funnel) run on FB pixel checkpoints and fully generalize the GCIQL profiler for pixels, with a pixel runner + a local smoke test.

**Architecture:** Add a `feature` seam to the shared `probe_coverage` (`"obs"` = current state path, unchanged; `"cube"` = physics-space NN over cube-xyz). Profilers derive the feature from `obs_type`. The FB probes are otherwise already obs-type-driven (the model encodes pixels internally; the pixel buffer carries `physics`); `phase_probe.py` needs no change. The GCIQL profiler gains obs_type/feature plumbing. A runner loops the 5 pixel seeds; a smoke runs a reduced FB pixel profile against the real s0 checkpoint using a synthetic pixel buffer.

**Tech Stack:** Python 3 (`.venv`), numpy, pandas, torch, MuJoCo/ogbench, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-05-20-pixel-diagnostics-pipeline-design.md`

---

## Conventions & ground rules (read first)

- **Run Python/tests via `.venv/bin/python`** (torch + rliable arch8 shim live there). Tests: `.venv/bin/python -m pytest`.
- **Commit only `scripts/`, `evals/`, `tests/`, `docs/`.** `analysis/`, `RESULTS/`, `datasets/` are untracked and must NOT be `git add`-ed.
- `scripts/` and `evals/` are packages; tests import `from evals._profile_core import ...`, `from scripts import gciql_profile as gp`.
- The state path must stay byte-for-byte equivalent: every new parameter defaults to the current behavior (`feature="obs"`, `coverage_feature="obs"`).
- This round is **build + smoke only**: no full 5-seed run, no figures, no paper edits.

## File structure

- `evals/_profile_core.py` (modify) — `probe_coverage` gains `feature` + `_episode_feature` helper.
- `evals/representation_profile.py` (modify) — `run_representation_profile` gains `coverage_feature`, selects ref.
- `scripts/representation_profile.py` (modify) — pass `coverage_feature` derived from `cfg.obs_type`.
- `scripts/gciql_profile.py` (modify) — obs_type in `parse_flags`, `feature` in `episodes_to_frames` + `_load_ref_obs`, `--obs-type` CLI, main wiring.
- `scripts/run_pixel_profile.sh` (create) — per-seed runner + aggregate + guarded GCIQL branch.
- `scripts/smoke_pixel_profile.sh` (create) — local synthetic-buffer smoke on the real s0 checkpoint.
- `tests/test_profile_coverage_feature.py` (create).
- `tests/test_representation_profile.py` (modify) — add `coverage_feature="cube"` stub test.
- `tests/test_gciql_profile.py` (modify) — add `_load_ref_obs`/`episodes_to_frames` cube tests + `--obs-type` arg test.

---

## Task 1: `feature` seam in `probe_coverage`

**Files:**
- Modify: `evals/_profile_core.py` (`probe_coverage`)
- Test: `tests/test_profile_coverage_feature.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_profile_coverage_feature.py`:

```python
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals._profile_core import probe_coverage
from evals.phase_probe import Thresholds


def _cube_ep(cube_xyz, grip, tmask, outcome="success"):
    return {"cube": np.asarray(cube_xyz, float).reshape(-1, 3),
            "grip": np.asarray(grip, float),
            "transport_mask": np.asarray(tmask, bool),
            "table_z": 0.02, "outcome": outcome}


def test_coverage_feature_cube_nn_distance():
    ref = np.array([[0, 0, 0], [10, 10, 10]], float)
    ep = _cube_ep([[0, 0, 0]], grip=[0.0], tmask=[False])
    df = probe_coverage([ep], ref, Thresholds(), feature="cube")
    assert set(df.columns) >= {"outcome", "region", "nn_dist"}
    assert df["region"].iloc[0] == "reach"
    assert abs(float(df["nn_dist"].iloc[0])) < 1e-6   # sits on a ref point


def test_coverage_feature_cube_needs_no_obs_key():
    ref = np.array([[0, 0, 0.02], [0.4, 0.1, 0.2]], float)
    ep = _cube_ep([[0.4, 0.1, 0.2], [0.0, 0.0, 0.02]],
                  grip=[1.0, 1.0], tmask=[True, False])
    df = probe_coverage([ep], ref, Thresholds(), feature="cube")
    assert len(df) == 2
    assert "obs" not in ep


def test_coverage_feature_obs_unchanged():
    ref = np.zeros((10, 4), float)
    ep = {"obs": np.array([[0, 0, 0, 0], [9, 9, 9, 9]], float),
          "grip": [0.0, 1.0], "transport_mask": [False, True],
          "cube": [[0, 0, 0.02], [0, 0, 0.2]],
          "table_z": 0.02, "outcome": "success"}
    df = probe_coverage([ep], ref, Thresholds())  # default feature="obs"
    assert df["nn_dist"].iloc[1] > df["nn_dist"].iloc[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_profile_coverage_feature.py -v`
Expected: FAIL — `probe_coverage() got an unexpected keyword argument 'feature'`.

- [ ] **Step 3: Add the `feature` seam**

In `evals/_profile_core.py`, add a helper above `probe_coverage`:

```python
def _episode_feature(ep: Dict[str, Any], feature: str) -> np.ndarray:
    """Per-step feature matrix for coverage. 'obs' = raw observation
    (state path); 'cube' = cube xyz (physics-space, for pixels)."""
    if feature == "cube":
        return np.asarray(ep["cube"], np.float64).reshape(-1, 3)
    return np.asarray(ep["obs"], np.float64)
```

Then change `probe_coverage` to accept `feature` and use the helper:

```python
def probe_coverage(episodes, ref_obs: np.ndarray, thr,
                   max_ref: int = 5000, seed: int = 0,
                   feature: str = "obs") -> "Any":
    """Per-step nearest-neighbour L2 distance to the standardised offline
    reference, tagged by outcome and region. `feature` selects the space:
    'obs' (state observation, default) or 'cube' (physics-space cube xyz)."""
    import pandas as pd

    ref = np.asarray(ref_obs, np.float64)
    if len(ref) > max_ref:
        rng = np.random.default_rng(seed)
        ref = ref[rng.choice(len(ref), max_ref, replace=False)]
    mu = ref.mean(0)
    sd = ref.std(0) + 1e-6
    refn = (ref - mu) / sd
    rows = []
    for ep in episodes:
        feat = _episode_feature(ep, feature)
        if feat.shape[0] == 0:
            continue
        on = (feat - mu) / sd
        d2 = (on ** 2).sum(1, keepdims=True) + (refn ** 2).sum(1) \
            - 2.0 * on @ refn.T
        nn = np.sqrt(np.maximum(d2.min(1), 0.0))
        reg = _region(ep, thr)
        for r, n in zip(reg, nn):
            rows.append({"outcome": ep["outcome"], "region": str(r),
                         "nn_dist": float(n)})
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_profile_coverage_feature.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add evals/_profile_core.py tests/test_profile_coverage_feature.py
git commit -m "feat(profile_core): coverage feature seam (obs|cube) for pixel support"
```

---

## Task 2: FB profiler coverage-feature wiring

**Files:**
- Modify: `evals/representation_profile.py` (`run_representation_profile`)
- Modify: `scripts/representation_profile.py` (the `run_representation_profile(...)` call)
- Test: `tests/test_representation_profile.py` (add one test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_representation_profile.py`:

```python
def test_run_representation_profile_cube_coverage():
    m, ag = _StubModel(), _StubAgent()

    def infer_z(task):
        return torch.tensor([1.0, 0.0, 0.0]), {
            "relabel_reward#nonzero": 2.0,
            "relabel_reward#num_samples": 10.0}

    def make_env(task):
        return _PhaseEnv()

    def sample_buffer(n):
        phys = np.zeros((n, 20), np.float32)   # cube ref at origin
        return {"next_obs": torch.zeros((n, 5)), "physics": phys,
                "action": np.zeros((n, 2), np.float32)}

    out = rp.run_representation_profile(
        model=m, agent=ag, infer_z=infer_z, make_env=make_env,
        sample_buffer=sample_buffer,
        relabel_fn_for=lambda t: (lambda p, a: np.zeros((len(p), 1),
                                                        np.float32)),
        goal_for=lambda t: np.array([0.0, 0.0, 0.20]),
        tasks=["t1"], n_episodes=1, thr=pp.Thresholds(),
        buffer_sample=8, topk=4, seed=0, coverage_feature="cube")
    assert set(out["coverage"].columns) >= {"task", "outcome",
                                            "region", "nn_dist"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_representation_profile.py::test_run_representation_profile_cube_coverage -v`
Expected: FAIL — `run_representation_profile() got an unexpected keyword argument 'coverage_feature'`.

- [ ] **Step 3: Add `coverage_feature` to `run_representation_profile`**

In `evals/representation_profile.py`, change the signature:

```python
def run_representation_profile(*, model, agent, infer_z, make_env,
                               sample_buffer, relabel_fn_for, goal_for,
                               tasks, n_episodes, thr, buffer_sample,
                               topk, seed, coverage_feature="obs"):
```

Then replace the coverage block at the end of the per-task loop:

```python
        ref_obs = (smp["next_obs"].detach().cpu().numpy()
                   if hasattr(smp["next_obs"], "detach")
                   else np.asarray(smp["next_obs"]))
        cov = probe_coverage(eps, ref_obs, thr, seed=seed)
        cov.insert(0, "task", task)
        cv.append(cov)
```

with:

```python
        if coverage_feature == "cube":
            cov_ref = physics[:, 14:17]
        else:
            cov_ref = (smp["next_obs"].detach().cpu().numpy()
                       if hasattr(smp["next_obs"], "detach")
                       else np.asarray(smp["next_obs"]))
        cov = probe_coverage(eps, cov_ref, thr, seed=seed,
                             feature=coverage_feature)
        cov.insert(0, "task", task)
        cv.append(cov)
```

(`physics` is already defined earlier in the loop as
`physics = np.asarray(smp["physics"], np.float32)`.)

Also import `probe_coverage` — it is already imported at the top of
`evals/representation_profile.py` via `from evals._profile_core import (...)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_representation_profile.py -v`
Expected: PASS (all prior tests + the new one).

- [ ] **Step 5: Wire the script to derive the feature from obs_type**

In `scripts/representation_profile.py`, change the `run_representation_profile(...)` call (currently ending `seed=cfg.seed)`):

```python
    res = run_representation_profile(
        model=agent.model, agent=agent, infer_z=infer_z,
        make_env=make_env, sample_buffer=sample_buffer,
        relabel_fn_for=relabel_fn_for, goal_for=goal_for, tasks=tasks,
        n_episodes=args.n_episodes, thr=thr,
        buffer_sample=args.buffer_sample, topk=args.topk,
        seed=cfg.seed,
        coverage_feature=("cube" if cfg.obs_type == "pixels" else "obs"))
```

- [ ] **Step 6: Commit**

```bash
git add evals/representation_profile.py scripts/representation_profile.py tests/test_representation_profile.py
git commit -m "feat(representation_profile): physics-space coverage for pixel obs_type"
```

---

## Task 3: Generalize the GCIQL profiler for pixels

**Files:**
- Modify: `scripts/gciql_profile.py` (`parse_flags`, `episodes_to_frames`, `_load_ref_obs`, `build_argparser`, `main`)
- Test: `tests/test_gciql_profile.py` (add three tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gciql_profile.py`:

```python
def test_build_argparser_has_obs_type():
    ap = gp.build_argparser()
    ns = ap.parse_args(["--run-dir", "/x", "--out", "/o",
                        "--obs-type", "pixels"])
    assert ns.obs_type == "pixels"


def test_load_ref_obs_cube_reads_physics(tmp_path):
    buf = tmp_path / "buffer"
    buf.mkdir(parents=True)
    T = 5
    phys = np.zeros((T, 21), np.float32)
    phys[:, 14:17] = np.arange(T * 3).reshape(T, 3)
    np.savez(buf / "episode_000000_5.npz",
             observation=np.zeros((T, 28), np.float32),
             physics=phys, action=np.zeros((T, 5), np.float32))
    ref = gp._load_ref_obs(str(tmp_path), feature="cube")
    assert ref.shape[1] == 3
    assert np.allclose(np.sort(ref.ravel()),
                       np.sort(phys[:, 14:17].ravel()))


def test_episodes_to_frames_cube_coverage():
    d = [0.30, 0.25, 0.20, 0.15, 0.10, 0.02]
    v = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    grip = [1.0] * 6
    cube_z = [0.20] * 6
    eps = [_ep("success", d, v, grip, cube_z)]
    cube_ref = np.zeros((10, 3), np.float32)
    vl, cov, fun, vs = gp.episodes_to_frames(
        eps, Thresholds(), cube_ref, feature="cube")
    assert set(cov.columns) >= {"task", "outcome", "region", "nn_dist"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_gciql_profile.py -k "obs_type or cube" -v`
Expected: FAIL — no `--obs-type` arg; `_load_ref_obs`/`episodes_to_frames` lack a `feature` kwarg.

- [ ] **Step 3: Add `obs_type` to `parse_flags`**

In `scripts/gciql_profile.py`, change `parse_flags`:

```python
def parse_flags(run_dir) -> Dict[str, Any]:
    """Read OGBench flags.json from a GCIQL seed dir."""
    data = json.loads((Path(run_dir) / "flags.json").read_text())
    return {"env_name": data.get("env_name", "cube-single-play-v0"),
            "seed": int(data.get("seed", 0)),
            "agent": data.get("agent", "agents/gciql.py"),
            "obs_type": data.get("obs_type", "state")}
```

- [ ] **Step 4: Add `feature` to `episodes_to_frames`**

Change the signature and the coverage call:

```python
def episodes_to_frames(episodes: List[Dict[str, Any]], thr: Thresholds,
                       ref_obs: np.ndarray, feature: str = "obs"):
```

and replace `cov = probe_coverage(cov_eps, ref_obs, thr)` with:

```python
    cov = probe_coverage(cov_eps, ref_obs, thr, feature=feature)
```

(`cov_eps` already carries `"cube"`, so `feature="cube"` works.)

- [ ] **Step 5: Add `feature` to `_load_ref_obs`**

```python
def _load_ref_obs(dataset_path: str, max_n: int = 5000,
                  feature: str = "obs") -> np.ndarray:
    """Reference array for the coverage probe. feature='obs' stacks
    next/observation (state path); feature='cube' stacks cube xyz
    (physics[:,14:17]) for the pixel path."""
    buf = Path(dataset_path) / "buffer"
    files = sorted(buf.glob("episode_*.npz"))[:200]
    obs = []
    for f in files:
        with np.load(f) as z:
            if feature == "cube":
                arr_i = np.asarray(z["physics"], np.float32)[:, 14:17]
            else:
                key = ("observation" if "observation" in z
                       else "observations" if "observations" in z
                       else "obs")
                arr_i = np.asarray(z[key], np.float32)
        obs.append(arr_i)
    arr = np.concatenate(obs, 0) if obs else np.zeros((1, 1), np.float32)
    if len(arr) > max_n:
        rng = np.random.default_rng(0)
        arr = arr[rng.choice(len(arr), max_n, replace=False)]
    return arr
```

- [ ] **Step 6: Add `--obs-type` to `build_argparser` and wire `main`**

In `build_argparser`, add before `return ap`:

```python
    ap.add_argument("--obs-type", default=None,
                    help="state|pixels; default reads flags.json")
```

In `main`, replace the body after `_jax_rollout` with:

```python
    episodes, flags = _jax_rollout(
        args.run_dir, args.step, task_ids, args.n_episodes,
        args.max_steps)
    obs_type = args.obs_type or flags.get("obs_type", "state")
    feature = "cube" if obs_type == "pixels" else "obs"
    ref = _load_ref_obs(args.dataset_path, feature=feature)
    vl, cov, fun, vs = episodes_to_frames(episodes, Thresholds(), ref,
                                          feature=feature)
```

(The rest of `main` — the `.to_parquet` writes and `metrics.json` — is
unchanged. The jax rollout already records cube/eff/grip from privileged
info, so it is obs_type-agnostic; the pixel jax path is structured but
remains untested until GCIQL pixel checkpoints exist.)

- [ ] **Step 7: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_gciql_profile.py -v`
Expected: PASS (all prior tests + the three new ones).

- [ ] **Step 8: Commit**

```bash
git add scripts/gciql_profile.py tests/test_gciql_profile.py
git commit -m "feat(gciql_profile): obs_type-aware pixel path + physics-space coverage"
```

---

## Task 4: Pixel profiling runner

**Files:**
- Create: `scripts/run_pixel_profile.sh`

- [ ] **Step 1: Create the runner**

```bash
#!/usr/bin/env bash
# Run the FB pixel diagnostic profiling over all pixel seeds, then aggregate.
# The GCIQL-pixel branch activates only when GCIQL_ROOT is set (checkpoints
# arrive later). Set DRY_RUN=1 to print the commands without executing.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PY="${PY:-.venv/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-RESULTS/fb-pixel-results}"
DATA_PATH="${DATA_PATH:-/dev/shm/factored-fb/datasets}"
MUJOCO_GL="${MUJOCO_GL:-egl}"
N_EPISODES="${N_EPISODES:-10}"
BUFFER_SAMPLE="${BUFFER_SAMPLE:-20000}"
REPR_OUT="${REPR_OUT:-analysis/representation_profile_pixel}"
PHASE_OUT="${PHASE_OUT:-analysis/phase_probe_pixel}"
GCIQL_ROOT="${GCIQL_ROOT:-}"
DRY_RUN="${DRY_RUN:-0}"

run() { echo "+ $*"; [ "$DRY_RUN" = "1" ] || "$@"; }

shopt -s nullglob
for d in "$RESULTS_ROOT"/*__s*; do
  [ -d "$d" ] || continue
  s=$(basename "$d" | sed -E 's/.*__s([0-9]+)$/\1/')
  cfg="$d/.hydra/config.yaml"
  ckpt="$d/checkpoints/final.pt"
  run "$PY" scripts/representation_profile.py \
      --config "$cfg" --checkpoint "$ckpt" \
      --out "$REPR_OUT/s${s}_final" \
      --data-path "$DATA_PATH" --mujoco-gl "$MUJOCO_GL" \
      --n-episodes "$N_EPISODES" --buffer-sample "$BUFFER_SAMPLE"
  run "$PY" scripts/phase_probe.py \
      --config "$cfg" --checkpoint "$ckpt" \
      --out "$PHASE_OUT/s${s}_final" \
      --data-path "$DATA_PATH" --mujoco-gl "$MUJOCO_GL" \
      --n-episodes "$N_EPISODES"
done

run "$PY" scripts/representation_profile_aggregate.py --root "$REPR_OUT"

if [ -n "$GCIQL_ROOT" ]; then
  for d in "$GCIQL_ROOT"/sd*; do
    [ -d "$d" ] || continue
    s=$(basename "$d")
    run "$PY" scripts/gciql_profile.py \
        --run-dir "$d" --out "analysis/gciql_profile_pixel/${s}_final" \
        --obs-type pixels --dataset-path "$DATA_PATH/cube-single-play-v0"
  done
fi
echo "[run_pixel_profile] done"
```

- [ ] **Step 2: Syntax-check the runner**

Run: `bash -n scripts/run_pixel_profile.sh && echo SYNTAX_OK`
Expected: `SYNTAX_OK`.

- [ ] **Step 3: Dry-run to verify it enumerates the 5 seeds**

Run: `DRY_RUN=1 bash scripts/run_pixel_profile.sh | grep -c representation_profile.py`
Expected: `5` (one representation_profile invocation per pixel seed; the aggregate line also matches, so expect `6` if it counts the aggregate — verify the seed loop prints 5 distinct `s0..s4` paths).

Run: `DRY_RUN=1 bash scripts/run_pixel_profile.sh | grep -oE 's[0-9]+_final' | sort -u`
Expected: `s0_final s1_final s2_final s3_final s4_final` (5 lines).

- [ ] **Step 4: Commit**

```bash
git add scripts/run_pixel_profile.sh
git commit -m "feat(run_pixel_profile): per-seed pixel profiling runner + GCIQL branch"
```

---

## Task 5: Local smoke (synthetic buffer, real s0 checkpoint)

**Files:**
- Create: `scripts/smoke_pixel_profile.sh`

This smoke verifies the FB pixel profiling plumbing locally without the multi-GB
visual dataset: it builds a tiny synthetic pixel buffer (real physics from the
local state dataset + zero pixel frames) and runs a reduced
`representation_profile.py` against the real s0 pixel checkpoint.

- [ ] **Step 1: Create the smoke script**

```bash
#!/usr/bin/env bash
# Local smoke for FB pixel representation_profile. Builds a tiny SYNTHETIC
# pixel buffer (real physics from the local state dataset + placeholder pixel
# frames) and runs a reduced profile against the real s0 pixel checkpoint.
# Verifies plumbing + cube-space coverage; not scientific validity.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PY="${PY:-.venv/bin/python}"
STATE_DS="${STATE_DS:-datasets/cube-single-play-v0/buffer}"
MUJOCO_GL="${MUJOCO_GL:-glfw}"

S0_DIR=$(ls -d RESULTS/fb-pixel-results/*__s0 2>/dev/null | head -1)
[ -n "$S0_DIR" ] || { echo "FAIL: no s0 pixel run dir under RESULTS/fb-pixel-results"; exit 1; }
[ -d "$STATE_DS" ] || { echo "FAIL: no local state dataset at $STATE_DS"; exit 1; }

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
SYN="$TMP/cube-single-play-v0/buffer"
mkdir -p "$SYN"

"$PY" - "$STATE_DS" "$SYN" <<'PY'
import sys
import numpy as np
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
files = sorted(src.glob("episode_*.npz"))[:8]
assert files, f"no state episodes under {src}"
for f in files:
    z = dict(np.load(f))
    T = z["observation"].shape[0]
    z["pixels"] = np.zeros((T, 64, 64, 3), np.uint8)
    np.savez(dst / f.name, **z)
print(f"[smoke] wrote {len(files)} synthetic pixel episodes -> {dst}")
PY

OUT="$TMP/repr/s0_final"
"$PY" scripts/representation_profile.py \
    --config "$S0_DIR/.hydra/config.yaml" \
    --checkpoint "$S0_DIR/checkpoints/final.pt" \
    --out "$OUT" --data-path "$TMP" --mujoco-gl "$MUJOCO_GL" \
    --tasks cube-single-play-singletask-task1-v0 \
    --n-episodes 2 --buffer-sample 256

"$PY" - "$OUT" <<'PY'
import sys
import pandas as pd
from pathlib import Path
out = Path(sys.argv[1])
for name in ("value_landscape", "value_steps", "z_decoding",
             "b_resolution", "coverage"):
    p = out / f"{name}.parquet"
    assert p.exists(), f"missing {p}"
cov = pd.read_parquet(out / "coverage.parquet")
assert {"outcome", "region", "nn_dist"} <= set(cov.columns), list(cov.columns)
assert len(cov) > 0, "empty coverage"
print("[smoke] PASS — parquets written; coverage rows:", len(cov))
PY
echo "[smoke_pixel_profile] PASS"
```

- [ ] **Step 2: Syntax-check**

Run: `bash -n scripts/smoke_pixel_profile.sh && echo SYNTAX_OK`
Expected: `SYNTAX_OK`.

- [ ] **Step 3: Run the smoke (this is the key verification)**

Run: `bash scripts/smoke_pixel_profile.sh`
Expected: ends with `[smoke_pixel_profile] PASS`. If it fails on a real shape/API mismatch in the pixel FB path, FIX the underlying profiler code (that is the point of the smoke), re-run, and only continue once it passes. Do not weaken the assertions to force a pass.

- [ ] **Step 4: Commit**

```bash
git add scripts/smoke_pixel_profile.sh
git commit -m "test(pixel-diag): local synthetic-buffer smoke for FB pixel profiling"
```

---

## Task 6: Full verification

**Files:** none modified (verification + plan-doc commit).

- [ ] **Step 1: Run the full test suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all tests pass (new coverage-feature, FB cube-coverage, GCIQL obs_type/cube tests + all existing).

- [ ] **Step 2: Confirm the commit boundary**

Run: `git status --short`
Expected: no tracked changes outstanding; `analysis/`, `RESULTS/`, `datasets/` remain untracked (`??`) and are NOT staged.

- [ ] **Step 3: Commit the plan document**

```bash
git add docs/superpowers/plans/2026-05-20-pixel-diagnostics-pipeline.md
git commit -m "docs(pixel-diag): implementation plan — pixel diagnostics pipeline"
```

- [ ] **Step 4: Report**

Summarize: coverage `feature` seam added; FB profiler emits physics-space coverage for pixels; GCIQL profiler fully obs_type-aware (pixel jax path untested until checkpoints exist); runner + smoke added and smoke verified on the real s0 checkpoint. Remind the user of the full-run command:
`MUJOCO_GL=egl DATA_PATH=/dev/shm/factored-fb/datasets bash scripts/run_pixel_profile.sh`
and that figures + paper are the next round once the heavy run produces parquets.

---

## Self-review notes

- **Spec coverage:** coverage feature seam (Task 1); FB physics-space coverage by obs_type (Task 2); GCIQL full generalization — parse_flags obs_type, episodes_to_frames feature, _load_ref_obs cube, --obs-type, main wiring (Task 3); runner with guarded GCIQL branch (Task 4); smoke on real s0 checkpoint (Task 5); aggregates unchanged + schema-compatible (no task needed — verified via the runner's aggregate call and existing aggregate tests); commit boundary + full suite (Task 6). `phase_probe.py` needs no change (already obs_type-driven) — exercised by the runner. All covered.
- **State path unchanged:** every new param defaults to the current behavior (`feature="obs"`, `coverage_feature="obs"`, `--obs-type` default None → resolves to flags/`state`).
- **Type/name consistency:** `probe_coverage(..., feature="obs")` used identically in Task 1 (def), Task 2 (`feature=coverage_feature`), Task 3 (`feature=feature`); `_load_ref_obs(..., feature=...)` and `episodes_to_frames(..., feature=...)` signatures match their call sites in `main`; `run_representation_profile(..., coverage_feature=...)` matches the script call.
- **Untested-path honesty:** the GCIQL pixel jax rollout cannot be end-to-end tested now (no checkpoints); only its pure seams are unit-tested, and the runner gates it behind `GCIQL_ROOT`.
```
