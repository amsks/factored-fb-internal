# Handoff: run the FB↔GCIQL bridge triplet on Metaworld (mw-bin-picking / mw-button-press)

**Date:** 2026-07-22
**Author of handoff:** prior agent session (exploration only — no code written, no jobs launched)
**Reader:** the agent that will build the Metaworld integration
**Working tree:** `/work/11708/amsks/vista/git/Factored-FB`, branch `feat/jaxify` (TACC Vista, GH200/aarch64)

> **TL;DR.** The checked-out branch `feat/jaxify` is **OGBench cube-single only** — it has
> **no Metaworld env, no HF-dataset loader, and its eval is hardwired to OGBench's
> goal-conditioned API**. The user says branch **`tl-jax-unify` already contains the Metaworld
> integration**. That branch was **NOT yet inspected** this session (TACC login node was
> fork-saturated; `git` calls could not complete). **Step 0 for the next agent is to inspect
> `tl-jax-unify`** (commands in §7) and decide: run there directly, or port its Metaworld path
> onto `feat/jaxify`. Do **not** rebuild Metaworld from scratch until you have confirmed what
> `tl-jax-unify` already provides.

---

## 1. The experiment (from `promtp.md`, verbatim intent)

Run a triplet of critics × {FlowBC, DDPGBC} actors × **2 Metaworld tasks** × **3 seeds** ×
**~300k steps**, on **expert** offline data.

- **Hard task = `mw-bin-picking`** — chosen as the exact structural analog of cube-single
  (grasp → transport, interaction variable, object-position hit coordinate); deliberately
  avoids `mw-assembly`'s insertion-precision confound.
- **Easy task = `mw-button-press`** — deliberately has **no** grasp-transport, making it the
  antmaze-style **domain-flip control**.
- **Pre-registered prediction (expert-data caveat):** occupancy grounding should be *less*
  toxic on unimodal expert trajectories; if it still floors on bin-picking, the mixing-mass
  mechanism needs sharpening.

### 1a. ⚠️ Unresolved inconsistency in the request — reconcile with the user before launching

`promtp.md` names **two different, non-matching agent sets**:

- **"Agents:" line** → `FB + {FlowBC, DDPGBC}`, `Bilinear GCIQL + {FlowBC, DDPGBC}`,
  `normal GCIQL + {FlowBC, DDPGBC}` = **3 critics × 2 actors = 6 configs**. All three critics
  (`fb`, `iql_bilinear`, `gciql`) and both actors (`flowbc`, `ddpgbc`) **exist on `feat/jaxify`**.
- **"triplet" line** → `bilinear GCIQL / FP-NoOrtho / occupancy control` = **3 critics**, of
  which only `bilinear GCIQL` (= `iql_bilinear`) exists on `feat/jaxify`. **`FP-NoOrtho` and
  `occupancy control` are NOT critic names on `feat/jaxify`** (see §3c). They are presumably on
  `tl-jax-unify`.

**These two lists are not the same experiment.** Ask the user which they want, or confirm the
mapping (my best guess: the "triplet" is the real design and `FP-NoOrtho`/`occupancy` are
`tl-jax-unify` critics; the "Agents:" line may be an earlier/looser draft). Do not silently
pick one.

### 1b. Name → critic mapping (what is known)

| User's term | `feat/jaxify` critic | Notes |
|---|---|---|
| Bilinear GCIQL | `iql_bilinear` | Exists. `Q(s,a,g)=⟨φ(s,a),ψ(g)⟩/√d`, expectile-V + twin-Q. Config: `impls/configs/critic/iql_bilinear.yaml`. Has an `iql_ortho_coef` knob (FB orthonormality on the goal leg) currently `0.0`. |
| normal GCIQL | `gciql` | Exists. |
| FB | `fb` | Exists. |
| **FP-NoOrtho** | **unknown** | Not on `feat/jaxify`. Likely a flow-policy / forward-map variant with orthonormality **off**. *Could* be `iql_bilinear` with `iql_ortho_coef=0` (already the default) — **do not assume**; check `tl-jax-unify`. |
| **occupancy control** | **unknown** | Not on `feat/jaxify`. The "occupancy grounding / mixing-mass" language in the prediction points to a dedicated occupancy critic. Check `tl-jax-unify`. |

---

## 2. Entry point & how a run is specified (`feat/jaxify`)

`impls/main.py` — `python -m impls.main` with absl flags:

- `--critic` (default `fb`) → `impls/configs/critic/<name>.yaml`
- `--actor` (default `flowbc`) → `impls/configs/actor/<name>.yaml`
- `--domain` (default `cube_single`) → `impls/configs/domain/<name>.yaml` (supplies `env_name`)
- `--override key=value` (repeatable; dotlist; `~key` deletes a key)
- `--seed`, `--train_steps` (default 1e6), `--log_interval` (5000), `--eval_interval` (100000),
  `--save_interval` (1e6), `--eval_episodes` (20), `--save_dir` (`exp/`),
  `--restore_path`, `--restore_epoch`, `--infer_cond_episodes`.

Config is composed by `impls/utils/config_utils.py:load_config(critic, actor, domain, overrides)`
merging **domain + critic + actor** YAML (OmegaConf) into a plain dict. `REQUIRED_KEYS`
(config_utils.py:17-130) enforces per-critic keys up front — a missing key is a named
`ConfigError`, not a bare KeyError. **A new critic must add its required-key tuple there.**

Training loop (main.py:99-134): sample batch → `agent.update` → periodic eval over
`env.unwrapped.task_infos` tasks via `evaluate(...)`.

---

## 3. What `feat/jaxify` has (and does not)

### 3a. Critics (`impls/configs/critic/`, `impls/critics/`)
`fb`, `gciql`, `isfb`, `iql_bilinear`, `simple_sf`, `sf_feature`, `centered_bilinear`,
`fb_forward_bilinear`. **No `occupancy`, no `fp_noortho`.**

### 3b. Actors (`impls/configs/actor/`)
`flowbc` (flow-matching + distilled one-step), `ddpgbc` (DDPG+BC), `awr`, `fql`.
All 8×4 critic×actor pairs are valid (`impls/agents/__init__.py` + `check_combo` in
`impls/critics/base.py`).

### 3c. Domains (`impls/configs/domain/`)
**Only** `cube_single.yaml` → `env_name: cube-single-play-v0`. No Metaworld domain files.

### 3d. Env + dataset construction — **this is the blocker**
`impls/utils/env_utils.py:74-82` — `make_env_and_datasets` calls **only**
`ogbench.make_env_and_datasets(dataset_name, compact_dataset=True)` and wraps in `FBDataset`.
**There is no Metaworld / box2d / HF path anywhere.**

`impls/utils/datasets.py`:
- `Dataset` (compact FrozenDict; infers `next_observations` by index+1 unless present; `valids`
  masks trajectory boundaries).
- **`FBDataset.sample`** returns exactly: `observations`, `actions`, `next_observations`
  (all float32), `rewards` `[B,1]` (zeros if absent — play data is unlabeled), `terminals`
  `[B]` zeros (always-γ bootstrap). **This is the batch shape a Metaworld loader must produce.**
- `GCDataset` (goal-conditioned critics `gciql`/`iql_bilinear`/etc.) needs `terminals` marking
  episode ends and does goal sampling (`value_goals`, `actor_goals`, `masks`, `rewards`) — it
  asserts `terminal_locs[-1] == size-1` (datasets.py:94), i.e. **the flattened dataset must end
  exactly on a terminal**. Any Metaworld loader feeding a GC critic must set per-step
  `terminals` correctly from the npz `terminal`/`timeout`.
- `make_dataset(base, config)` picks `GCDataset` iff `config['dataset_class']=='GCDataset'`.

### 3e. Eval — **OGBench-coupled, needs a Metaworld path**
`impls/utils/evaluation.py:37-80`: `env.reset(options=dict(task_id=..., render_goal=...))`,
reads `info['goal']`, iterates `env.unwrapped.task_infos`. **Metaworld's Gymnasium API has no
`task_id` reset option, no `info['goal']`, no `task_infos`.** A Metaworld eval must either
(a) provide a shim env exposing that surface, or (b) add a separate eval branch using
Metaworld's native `info['success']` as the metric. Note the dataset already carries a
per-step `success` field for offline reference.

---

## 4. The data (HuggingFace) — confirmed schema

**`https://huggingface.co/datasets/TommyShen/Factor-FB-Offline`** (~1.57 GB, **expert** demos):
- Layout: `data/<family>/<task>.npz` + `data/<family>/<task>.meta.json` + top-level `manifest.json`.
- Families include **`metaworld`** (`mw-assembly, mw-basketball, mw-bin-picking, mw-box-close,
  mw-button-press`), **`box2d`**, and **`maniskill`**.
- Per-npz **episodic** arrays (N episodes × T fixed horizon):
  - `observation`, `next_observation` — `[N, T, obs_dim]` float32
  - `action` — `[N, T, act_dim]` float32
  - `achieved_goal` — `[N, T, 3]` float32 (likely object/end-effector xyz)
  - `reward`, `terminal`, `timeout`, `success` — `[N, T]` (float32/bool)
- **Loader must flatten `[N,T,·] → [N·T,·]`** and build `valids`/`terminals` from
  `terminal`|`timeout` so trajectory boundaries (and the GCDataset end-on-terminal assert) hold.

**`https://huggingface.co/TommyShen/Factor-FB-Offline-tdmpc2`** (checkpoints):
- **Box2D only.** Expert TD-MPC2 single-task `.pt` checkpoints under `checkpoints/box2d/`
  (weights+optimizer), `model_size=B`, `load_state_dict(..., strict=False)`,
  `checkpoint_manifest.json` gives per-task obs/act dims.
- **Metaworld & ManiSkill demos in the dataset were generated from `nicklashansen/newt`
  (a public multitask checkpoint), NOT from this tdmpc2 repo.** So for the **Metaworld** runs
  you only need the **dataset** npz files — the tdmpc2 checkpoints are irrelevant unless you
  later do box2d.

**Download note:** compute nodes have **no internet**. Pre-download the npz on a login node,
e.g. `hf download TommyShen/Factor-FB-Offline data/metaworld/mw-bin-picking.npz --repo-type dataset`
(and `mw-button-press.npz`), into a `$SCRATCH` cache the job reads.

---

## 5. The sbatch harness (`scripts/`, already written, cube_single-shaped)

- `scripts/vista_setup.sh` — one-time, **run on a login node** (needs internet). Builds a
  **persistent** venv at `$SCRATCH/.venv-jax-fb` from `requirements.txt`, verifies JAX, and
  pre-caches the OGBench dataset. **Must be extended** to `hf download` the Metaworld npz.
- `scripts/vista_train.slurm` — GH200 queue (`-p gh`, 1 node, 1 task, `-t 24:00:00`; use
  `gh-dev`, max 2h, for tests). Knobs via env: `CRITIC`, `ACTOR`, `DOMAIN`, `SEED`,
  `TRAIN_STEPS`, `SAVE_DIR` (default `$SCRATCH/factored-fb-exp`), `VENV`. Sets
  `CUDA_VISIBLE_DEVICES=0`, `MUJOCO_GL=egl`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`,
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`, `WANDB_MODE=offline`. Runs
  `srun $PY -m impls.main --critic --actor --domain --seed --train_steps --save_dir`.
- Submit: `sbatch -A <alloc> scripts/vista_train.slurm`; find alloc via
  `sacctmgr -pns show assoc user=$USER format=Account`.
- **`scripts/` is currently untracked** (git status shows `?? scripts/`).

For the run matrix, submit one job per (critic, actor, task, seed) with those env knobs, e.g.
`CRITIC=iql_bilinear ACTOR=flowbc DOMAIN=mw_bin_picking SEED=0 TRAIN_STEPS=300000 sbatch -A <alloc> scripts/vista_train.slurm`.
6 agent-configs × 2 tasks × 3 seeds = **36 jobs** (or 3×2×3 = 18 if the "triplet" is 3 total
configs — resolve §1a first).

---

## 6. If porting Metaworld onto `feat/jaxify` (only if `tl-jax-unify` can't be run directly)

Checklist (each item is why it's needed, per §3):
1. **`impls/configs/domain/mw_bin_picking.yaml`, `mw_button_press.yaml`** — supply an
   `env_name` (or a `dataset_path` + Metaworld env id) the loader understands.
2. **Metaworld branch in `env_utils.make_env_and_datasets`** — detect the domain, build the
   Metaworld Gymnasium env, and load the npz → flatten `[N,T,·]` → `Dataset.create(...)` with
   `observations/actions/next_observations/terminals/valids`.
3. **Metaworld eval path** — either a wrapper exposing `task_infos`/`info['goal']`/`task_id`
   reset, or a separate eval branch in `main.py`/`evaluation.py` using native `info['success']`.
4. **Critic configs for `FP-NoOrtho` and `occupancy control`** (§1b) — plus their
   `REQUIRED_KEYS` tuples in `config_utils.py` and, if new critic classes, `impls/critics/*.py`
   + registration in `impls/agents/__init__.py`. **Prefer cherry-picking these from
   `tl-jax-unify` over re-deriving them.**
5. **`requirements.txt`** — add `metaworld` (+ `huggingface_hub`/`datasets` for download).
   ⚠️ Verify Metaworld + its MuJoCo build install on **aarch64/GH200**; this is an unvalidated risk.
6. **`vista_setup.sh`** — add the `hf download` of the Metaworld npz to `$SCRATCH`.
7. **Parity/smoke** — the repo's discipline is 1e-10 parity fixtures per critic
   (`tests/impls/`). At minimum add a smoke test that one Metaworld (critic,actor) pair runs a
   train step + one eval episode. `make smoke` / `make test` exist.

---

## 7. STEP 0 for the next agent — inspect `tl-jax-unify` (commands)

The login node was fork-saturated (even `ls` took minutes); a batched git inspection was
launched but had not returned. Re-run it (batch into **one** shell call to amortize the
`/etc/tacc/profile` fork tax; expect a slow start):

```bash
cd /work/11708/amsks/vista/git/Factored-FB
REF=tl-jax-unify; git rev-parse --verify --quiet "$REF" >/dev/null || REF=origin/tl-jax-unify
git diff --stat feat/jaxify.."$REF"                        # what the branch changes
git ls-tree -r --name-only "$REF" -- impls/configs/domain  # mw_* domain files?
git ls-tree -r --name-only "$REF" -- impls/configs/critic   # occupancy / fp_noortho?
git grep -lI -i -e metaworld -e occupancy -e noortho "$REF" # where the integration lives
git show "$REF:impls/utils/env_utils.py"                    # the Metaworld env/data loader
git ls-tree -r --name-only "$REF" -- scripts                # its sbatch harness
```

Then decide:
- **(A) Run directly on `tl-jax-unify`** — if it has the domains, critics, loader, and a
  working harness, `git switch tl-jax-unify` (stash/commit `scripts/` first) or use a
  `git worktree` and launch there. **Fastest path; try this first.**
- **(B) Cherry-pick** its Metaworld pieces (env_utils branch, domain/critic configs, eval
  path) onto `feat/jaxify` per §6.

---

## 8. Environment gotchas (TACC Vista)

- **Login node is currently fork-saturated** ("`fork: Resource temporarily unavailable`") —
  interactive commands take **minutes**; batch shell work aggressively. Does not affect
  submitted jobs (they run on compute nodes).
- **GH200 / aarch64.** JAX CUDA12 aarch64 wheels come via `ogbench[train]`; no CUDA module
  needed. Verify Metaworld/MuJoCo builds here — **unvalidated**.
- **Compute nodes: no outbound internet.** Materialize venv **and** all datasets on a login
  node up front (into `$SCRATCH`). `WANDB_MODE=offline`; `wandb sync` later from a login node.
- `MUJOCO_GL=egl` for headless eval render; `main.py` also pins `MUJOCO_EGL_DEVICE_ID`.
- One visible GPU per node (`CUDA_VISIBLE_DEVICES=0`).

---

## 9. Open decisions for the user / building agent
1. **§1a** — is the experiment the **6-config** "Agents:" list or the **3-critic** "triplet"?
   And what exactly are `FP-NoOrtho` and `occupancy control`?
2. Run directly on `tl-jax-unify` (A) or port onto `feat/jaxify` (B)?
3. Eval metric on Metaworld: native `success` vs a goal-conditioned shim?
4. `TRAIN_STEPS=300000`, 3 seeds (e.g. 0/1/2), `eval_interval`/`save_interval` for 300k
   (defaults are tuned for 1e6 — likely want `--eval_interval 25000 --save_interval 100000`).
5. Allocation account for `sbatch -A`.

## 10. This session did NOT
- Write or modify any code. - Launch any sbatch job. - Inspect `tl-jax-unify` (blocked).
- Download any dataset. Nothing is in flight except a possibly-still-running read-only git
  inspection of `tl-jax-unify` (background task).
