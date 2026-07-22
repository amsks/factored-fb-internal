# feat/jaxify Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `feat/jaxify` into a clean, standalone, OGBench-shaped **JAX-only** repo: `impls/` is the sole implementation, all torch code + the old JAX shim/campaign path + research artifacts are removed, and the README describes the new world.

**Architecture:** `impls/` is a self-contained JAX package (critic × actor seam) that imports only from `impls/` + the `ogbench` package (installed editable from `third_party/ogbench`). It has **zero** dependency on the torch top-level, so deletion is safe and verified by keeping the `tests/impls/` suite green throughout. Work happens in the isolated worktree `/home/adityamohan/git/Factored-FB-jaxify` on branch `feat/jaxify`.

**Tech Stack:** JAX 0.10, Flax 0.12.7, Optax, `ogbench` (editable from `third_party/ogbench[train]`), pytest. Venv: `/dev/shm/.venv-jax`.

## Global Constraints

- **Test command (the gate):** `XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python -m pytest tests/impls/ -q`
- **Baseline:** the `tests/impls/` pass set recorded in Task 1 must stay identical after every deletion. A deletion that changes any pass/fail is a mistake — investigate before continuing.
- **Never** touch the main worktree (`/home/adityamohan/git/Factored-FB`) or other branches. All work is branch-scoped commits on `feat/jaxify`.
- **Commits:** no Claude co-author line (project convention).
- **`impls/` behaviour is frozen** — no changes beyond deletions. The coverage audit (Task 1) confirmed **no live gaps** requiring a port.
- **Keep:** `impls/`, `tests/impls/` (minus the 6 `_gen_*.py` generators), `third_party/ogbench` (runtime dep), `docs/`, `.githooks/`, `LICENSE`, and the pruned packaging files.
- **Deferred (do NOT port):** pixel/DrQ path, scene domain + `value_split`. Dead-path knobs (measure-family, anneal-family, conservatism) are dropped, not ported.

---

### Task 1: Coverage audit doc + recorded baseline

Proves `impls/` covers every *live* research knob before anything is deleted, and pins the test baseline.

**Files:**
- Create: `docs/jaxify-coverage-audit.md`

**Interfaces:**
- Produces: `docs/jaxify-coverage-audit.md` (referenced by the README in Task 8) and a recorded baseline pass count (used as the gate in Tasks 2–7, 9).

- [ ] **Step 1: Record the test baseline**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python -m pytest tests/impls/ -q 2>&1 | tail -3`
Expected: `N passed` (record the exact N — this is the number every later task must reproduce). If any test needs a GPU and fails on CPU, note it as an allowed exception and re-run that file with the default device.

- [ ] **Step 2: Write the audit doc**

Create `docs/jaxify-coverage-audit.md` with this content (the classification was completed during planning — every zero-coverage knob is dead-path, out-of-scope, renamed, or deferred; there are **no live gaps**):

```markdown
# Jaxify coverage audit — gciql_flow.py / configs/gciql knobs vs impls/

Verdict: **no live gap.** Every research knob used by the deleted JAX shim
(`tools/wandb_mode_shim/gciql_flow.py`) and `configs/gciql/*.yaml` is either
covered by an `impls/` critic/actor, renamed, a dead-path drop, or an explicitly
deferred feature. `impls/` therefore fully replaces the shim for in-scope work.

| Knob | Role | impls/ status | Note |
|---|---|---|---|
| value_arch | bilinear vs mlp value head | covered | encoded by critic choice (iql_bilinear / centered_bilinear / simple_sf) + `value_arch` on centered/simple_sf |
| value_latent_dim | goal-bottleneck rank d | covered | `value_latent_dim` in bilinear critic configs |
| bc_coeff | flow-BC distillation weight | covered | `bc_coeff` in configs/actor/flowbc.yaml |
| project_z | sphere-norm z/psi | covered | `project_z` across critics + utils/fb_math |
| fb_forward_z | z-indexed forward | covered | `fb_forward_z` in fb_forward_bilinear / sf_feature |
| fwd_two_tower | two-tower phi | covered | `phi_module='two_tower'` in fb_forward_bilinear |
| sf_feature_td, sf_coef, sf_base, sf_ortho_coef, sf_reward_regress, sf_joint_psi | SF-feature family | covered | critics/sf_feature.py |
| actor_cond, actor_z_from_critic, actor_z_infer | actor conditioning | covered | iql_bilinear / fb_forward_bilinear |
| expectile, double_q, iql_ortho_coef | IQL head scalars | covered | bilinear critics |
| centered_measure, centered_measure_coef, center_b_samples | centering/dueling | covered | `mode` enum + `centered_measure_coef` in centered_bilinear |
| measure_coef, measure_legs, measure_ortho_coef, measure_stopgrad_phi, measure_scale_norm, measure_test_states, measure_insample | FB successor-measure family | covered (structural) | fb_forward_bilinear `objective=measure/mix/isfb` — most are UNVERIFIED/dead paths per T7/T8 + measure-cell campaigns |
| measure_only | measure objective select | renamed | `objective=measure` |
| psi_base, simplesf_ground_psi | psi grounding | covered | iql_bilinear / simple_sf |
| isfb, alpha, awr_beta | isfb objective | covered | critics/isfb.py + fb_forward_bilinear isfb terms |
| flow_levels | flow-actor ODE steps | renamed | `flow_steps` in flowbc + fql actors |
| actor_kind | actor type select | covered | the actor NAME (flowbc/ddpgbc/awr/fql) via the combo seam |
| anneal_ramp, anneal_measure, anneal_lr_warmup, anneal_iql_fade, anneal_simplesf_ground | curriculum anneal | **dead-path drop** | anneal campaign concluded the measure fixed point is bad on cube (L_IQL was propping the basin) |
| measure_z_random_frac | random-z frac in measure | **dead-path drop** | measure family negative |
| freeze_critic_ckpt | frozen-critic ablation | **drop** (experimental) | `impls/main.py` has restore_path/restore_epoch; the freeze ablation is reconstructible, not a core knob |
| subgoal_steps, low_actor_rep_grad | HIQL knobs | **out of scope** | HIQL is not a bridge agent |
| value_split | scene factored value | **deferred** | scene domain not ported (cube-single only) |
| encoder, p_aug, frame_stack | pixel/DrQ | **deferred** | impls/ is state-only |
```

- [ ] **Step 3: Commit**

```bash
git add docs/jaxify-coverage-audit.md
git commit -m "docs(jaxify): coverage audit — impls/ covers every live gciql_flow knob (no gap)"
```

---

### Task 2: Delete torch top-level modules + torch package dirs

**Files:**
- Delete: `agents/`, `buffers/`, `envs/`, `evals/`, `base_config.py`, `base_model.py`, `nn_models.py`, `normalizers.py`, `train.py`, `resume.py`, `run_gciql.py`, `utils.py`

**Interfaces:**
- Consumes: nothing (these are the torch stack; `impls/` imports none of them — verified).
- Produces: a tree with no torch entrypoints.

- [ ] **Step 1: Confirm nothing kept imports these**

Run: `grep -rnE '^(from|import) (agents|buffers|envs|evals|base_config|base_model|nn_models|normalizers|utils|train|resume|run_gciql)\b' impls/ tests/impls/ third_party/ | grep -vE 'impls\.|third_party/ogbench'`
Expected: no output (empty). If any line appears, STOP and investigate — it means a kept file depends on a torch module.

- [ ] **Step 2: Delete**

```bash
git rm -r agents buffers envs evals
git rm base_config.py base_model.py nn_models.py normalizers.py train.py resume.py run_gciql.py utils.py
```

- [ ] **Step 3: Verify the gate is still green**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python -m pytest tests/impls/ -q 2>&1 | tail -3`
Expected: `N passed` — identical N to Task 1.

- [ ] **Step 4: Commit**

```bash
git commit -m "chore(jaxify): remove torch stack (agents/, buffers/, envs/, evals/, top-level torch modules) — impls/ is self-contained"
```

---

### Task 3: Delete the old JAX shim + campaign configs

**Files:**
- Delete: `tools/` (only `tools/wandb_mode_shim/*` — the gciql_flow/isfb/hiql/gcivl/crl/drq/flow shims), `configs/` (top-level torch/shim Hydra tree: `configs/agent`, `configs/domain`, `configs/hydra`, `configs/gciql`, `configs/train.yaml`, `configs/.gitkeep`)

**Interfaces:**
- Consumes: nothing kept. `impls/` reads its OWN configs at `impls/configs` (`CONFIG_ROOT` in `impls/utils/config_utils.py` points to `impls/configs`, verified), never the top-level `configs/`.

- [ ] **Step 1: Confirm impls uses impls/configs, not top-level configs/**

Run: `grep -n CONFIG_ROOT impls/utils/config_utils.py`
Expected: a line joining `dirname(dirname(__file__))` with `"configs"` (i.e. `impls/configs`), NOT the repo-root `configs/`.

- [ ] **Step 2: Delete**

```bash
git rm -r tools configs
```

- [ ] **Step 3: Verify the gate**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python -m pytest tests/impls/ -q 2>&1 | tail -3`
Expected: `N passed` — identical N.

- [ ] **Step 4: Commit**

```bash
git commit -m "chore(jaxify): remove old JAX shim (tools/wandb_mode_shim) + torch/shim configs/ — impls/ uses impls/configs"
```

---

### Task 4: Delete orphaned tests + inert fixture generators

Keep `tests/impls/` (the JAX-survivor suite) and its committed `.npz` golden fixtures. Delete every torch/shim/psm/rldp/atlas test, the non-impls test dirs, and the 6 `_gen_*.py` fixture generators (inert once the torch/shim reference is gone).

**Files:**
- Delete: all top-level `tests/test_*.py`, `tests/agents/`, `tests/atlas/`, `tests/parity/`, `tests/scripts/`, `tests/jax_subprocess.py`, and `tests/impls/fixtures/_gen_*.py`
- Keep: `tests/__init__.py`, `tests/impls/**` (minus generators), all `tests/impls/fixtures/*.npz`

**Interfaces:**
- Consumes: `tests/impls/` imports only from `tests.impls.*` (verified) — no dependency on the deleted test files/dirs.

- [ ] **Step 1: Confirm tests/impls is self-contained**

Run: `grep -rnE '^(from|import) tests' tests/impls/ | grep -vE 'tests\.impls'`
Expected: no output (empty).

- [ ] **Step 2: Delete the 6 inert fixture generators (keep the .npz)**

```bash
git rm tests/impls/fixtures/_gen_*.py
```

- [ ] **Step 3: Delete the non-impls tests**

```bash
git rm -r tests/agents tests/atlas tests/parity tests/scripts
git rm tests/jax_subprocess.py
git rm tests/test_*.py
```

- [ ] **Step 4: Verify the gate (and that fixtures survive)**

Run: `ls tests/impls/fixtures/*.npz | wc -l && XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python -m pytest tests/impls/ -q 2>&1 | tail -3`
Expected: `8` npz files present, then `N passed` — identical N. (The parity tests read the committed `.npz`, not the generators.)

- [ ] **Step 5: Commit**

```bash
git commit -m "chore(jaxify): drop torch/shim/psm/rldp/atlas tests + inert fixture generators; keep tests/impls + golden .npz"
```

---

### Task 5: Delete the scripts/ tree

Every `scripts/*` entry is a torch/shim launch, campaign, or analysis script; none reference `impls/` (verified — `grep -rlE 'impls\.main|from impls|python -m impls' scripts/` is empty).

**Files:**
- Delete: `scripts/`

- [ ] **Step 1: Re-confirm no script references impls**

Run: `grep -rlE 'impls\.main|from impls|python -m impls|impls/main' scripts/ 2>/dev/null | head`
Expected: no output (empty).

- [ ] **Step 2: Delete**

```bash
git rm -r scripts
```

- [ ] **Step 3: Verify the gate**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python -m pytest tests/impls/ -q 2>&1 | tail -3`
Expected: `N passed` — identical N.

- [ ] **Step 4: Commit**

```bash
git commit -m "chore(jaxify): remove torch/shim/analysis scripts/ tree (none drive impls/)"
```

---

### Task 6: Remove research artifacts from the branch

`feat/jaxify` becomes code-only. This is branch-scoped: other branches keep these tracked, and the **main worktree's** on-disk copies (a separate directory) are untouched.

**Files:**
- Delete (from branch tree + this worktree's disk): `PAPER/`, `analysis/`, `results/`, `data/`
- Modify: `.gitignore` (add the four paths so the branch stays clean)

- [ ] **Step 1: Delete the artifact trees**

```bash
git rm -r PAPER analysis results data
```

- [ ] **Step 2: Add them to .gitignore**

Append to `.gitignore`:

```
# Research artifacts — kept on other branches + the main worktree, not on feat/jaxify
PAPER/
analysis/
results/
data/
```

- [ ] **Step 3: Verify main worktree is untouched**

Run: `ls -d /home/adityamohan/git/Factored-FB/PAPER >/dev/null 2>&1 && echo "main worktree PAPER intact"`
Expected: `main worktree PAPER intact` (the main checkout's separate on-disk copy is not affected by a `git rm` in this worktree).

- [ ] **Step 4: Commit**

```bash
git add .gitignore
git commit -m "chore(jaxify): drop research artifacts (PAPER/analysis/results/data) — clean code-only branch"
```

---

### Task 7: Prune packaging to JAX-only

**Files:**
- Delete: `requirements.txt` (torch-based), `uv.lock` (torch-era lock)
- Modify: `requirements-jax.txt` → rename to `requirements.txt` (the canonical JAX deps), `pyproject.toml`, `Makefile`

**Interfaces:**
- Produces: a `requirements.txt` that installs `-e ./third_party/ogbench[train]` (jax/flax/ogbench) + `tqdm` + test/logging deps; a `Makefile` with only JAX targets; a `pyproject.toml` with the pytest config kept and torch/exca deps removed.

- [ ] **Step 1: Make the JAX requirements canonical**

```bash
git rm requirements.txt uv.lock
git mv requirements-jax.txt requirements.txt
```

- [ ] **Step 2: Ensure requirements.txt carries the test/logging deps impls needs**

Edit `requirements.txt` so it contains (append any missing):

```
-e ./third_party/ogbench[train]
tqdm
wandb
ml_collections
pytest
```

- [ ] **Step 3: Prune pyproject.toml**

Open `pyproject.toml`; remove any `[project.dependencies]`/torch/exca/DMC entries and any torch-only tool config. **Keep** the `[tool.pytest.ini_options]` block (the suite's `configfile` is `pyproject.toml`, and `tests/impls/conftest.py` relies on x64 config). Verify the pytest section still resolves:

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python -m pytest tests/impls/ -q --collect-only 2>&1 | tail -3`
Expected: collection succeeds (`N tests collected`), no config error.

- [ ] **Step 4: Prune the Makefile**

Open `Makefile`; delete torch targets (`make smoke` over Box obs, torch install). Keep/replace with JAX targets:

```makefile
install:
	/dev/shm/.venv-jax/bin/pip install -r requirements.txt

test:
	XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python -m pytest tests/impls/ -q
```

- [ ] **Step 5: Verify the gate**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python -m pytest tests/impls/ -q 2>&1 | tail -3`
Expected: `N passed` — identical N.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "build(jaxify): prune packaging to JAX-only (requirements/pyproject/Makefile); drop torch reqs + uv.lock"
```

---

### Task 8: Rewrite the README

**Files:**
- Modify: `README.md` (full rewrite)

- [ ] **Step 1: Replace README.md**

Write a JAX-first README covering, in order:
1. **What it is** — a clean standalone JAX implementation of the FB↔GCIQL bridge on state-based OGBench cube-single. One implementation: `impls/`.
2. **Method** — `Q(s,a,z)=F(s,z,a)·B(s')`; z inferred at test time as `z=project(Σ rᵢ·B(next_obsᵢ))`; no fine-tuning.
3. **The critic × actor seam** — from `impls/agents/__init__.py`: critics `{fb, gciql, isfb, iql_bilinear, simple_sf, sf_feature, centered_bilinear, fb_forward_bilinear}` × actors `{flowbc, ddpgbc, awr, fql}`, composed and validated by `check_combo`. State the valid pairs (read them from `check_combo` / the existing combo-validation test) as a matrix.
4. **Install** — venv `/dev/shm/.venv-jax`; `pip install -r requirements.txt` (installs `ogbench` editable from `third_party/ogbench[train]` + jax/flax).
5. **Run** — `impls/main.py` with Hydra/OmegaConf composition over `impls/configs/{critic,actor,domain}` groups; show one concrete command (a valid critic×actor pair on cube-single) with `save_dir`/`restore_path` flags.
6. **Data** — how to obtain the OGBench cube-single dataset (`ogbench.make_env_and_datasets`, `compact_dataset=True`).
7. **Tests / parity** — the gate command; note that `tests/impls/fixtures/*.npz` are **frozen golden values** captured at 1e-10 torch parity (generators were removed with the torch reference; regeneration requires checking out a pre-cleanup ref). Link `docs/jaxify-coverage-audit.md`.
8. **Scope** — state-only, cube-single; pixel/DrQ + scene domain are deferred; PSM/RLDP/TD-MPC2 live on their own branches.

Remove all torch-era content (`FBAgent`/`FBFlowBCAgent`, `device=cpu`, torch `.venv`, `make smoke` over Box obs, exca/DMC mentions).

- [ ] **Step 2: Sanity-check every command in the README runs**

Run the install/test/run commands from the README (at least the test gate and one `impls/main.py --help` or a 1-step train). Fix any that don't match reality.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(jaxify): rewrite README for the JAX-only impls/ world (seam, install, run, parity)"
```

---

### Task 9: Final verification

**Files:** none (verification only).

- [ ] **Step 1: Full suite green, identical pass set**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python -m pytest tests/impls/ -q 2>&1 | tail -3`
Expected: `N passed` — identical N to Task 1.

- [ ] **Step 2: No surviving reference to any deleted torch/shim module**

Run: `grep -rnE '(from|import) (agents|buffers|envs|evals|base_model|base_config|nn_models|normalizers|train|resume|run_gciql|gciql_flow)\b' . --include=*.py | grep -vE 'third_party/|impls\.|impls/|docs/'`
Expected: no output (empty). (`third_party/`, `impls.*`, and `docs/` are excluded — `impls.utils.torch_to_flax` is a kept pure-JAX helper; docs mention old names historically.)

- [ ] **Step 3: End-to-end smoke train step**

Run one train step of a valid pair through `impls/main.py` on cube-single (use the command shown in the README, capped to a few steps). Expected: it builds the agent, runs update steps, and exits cleanly.

- [ ] **Step 4: Confirm the tree shape is OGBench-like**

Run: `git ls-tree feat/jaxify --name-only | sort`
Expected: top level is `impls/`, `tests/`, `third_party/`, `docs/`, `.githooks/`, `README.md`, `LICENSE`, `Makefile`, `pyproject.toml`, `requirements.txt` — and nothing torch/shim/artifact.

- [ ] **Step 5: Final summary commit (if any stray tidy-ups)**

Only if Steps 1–4 surfaced a small fix; otherwise the branch is done.

---

## Self-Review

- **Spec coverage:** Phase 1 audit → Task 1. Phase 2 deletions → Tasks 2–7 (torch, shim, tests, scripts, artifacts, packaging). Phase 3 README → Task 8. Verification (before/after green, grep clean, smoke) → Tasks 2–7 gates + Task 9. Deferred pixel/scene and dead-path drops recorded in the audit (Task 1). All spec sections map to a task.
- **Placeholder scan:** the one runtime value not knowable at authoring time is the baseline pass count `N`; it is *recorded* in Task 1 Step 1 and referenced symbolically thereafter — this is a recorded measurement, not a placeholder.
- **Type/name consistency:** the test-gate command, the `N` baseline, and the keep/delete file lists are used identically across tasks. `torch_to_flax` (kept, pure-JAX) is explicitly excluded from the grep gates so it never reads as a torch dependency.
