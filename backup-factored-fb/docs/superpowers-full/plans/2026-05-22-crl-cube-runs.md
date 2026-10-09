# CRL on cube-single (state + pixel/DrQ) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add OGBench's CRL baseline to the cube-single matrix on two modalities — state (`cube-single-play-v0`) and pixel with FB's DrQ front-end (`visual-cube-single-play-v0`) — then smoke-test and launch a 10-state + 5-pixel-seed sweep.

**Architecture:** CRL needs **no agent/Python code changes** — it already exposes `encoder`/`frame_stack`/`p_aug` and reads the global `encoder_modules`, so the existing DrQ injection (`tools/wandb_mode_shim/sitecustomize.py`) and `run_gciql.py`'s `OGBENCH_DRQ_FRONTEND` gating apply unchanged. Work = two new Hydra configs (clones of the GCIQL ones with algo/alpha/run_group deltas), a small generalization of the sweep launcher (config-name log naming + explicit `JOBS` override + a `DRYRUN` guard), config argv tests, then a smoke gate and a two-batch launch.

**Tech Stack:** Hydra configs, JAX/Flax OGBench (`/dev/shm/.venv-jax`), pytest under `.venv`, bash launcher scripts.

---

## Spec reference

`docs/superpowers/specs/2026-05-22-crl-cube-runs-design.md` (committed).

## Critical context (verified against the codebase)

- **`run_gciql.py` is agent-agnostic.** `build_ogbench_argv` (`run_gciql.py:25-43`) emits `--agent=<agent_file>` and one `--agent.<k>=<v>` per `agent_overrides` entry; `build_child_env` sets `OGBENCH_DRQ_FRONTEND=1` iff `agent_overrides.encoder == 'drq'`. So CRL works by config alone.
- **CRL config knobs** (`third_party/ogbench/impls/agents/crl.py` `get_config`): `encoder` (placeholder str), `frame_stack` (placeholder int), `p_aug` (0.0), `alpha` (0.1 default), `batch_size` (1024 default), `actor_loss` ('ddpgbc'). It builds `encoder_modules[config['encoder']]` five times (critic_state/goal, value_state/goal, actor GCEncoder); the smoke run validates the pixel path.
- **OGBench CRL cube recipe** (`impls/hyperparameters.sh`): state `--agent.alpha=3.0`; pixel `--agent.alpha=3.0 --agent.batch_size=256 --agent.encoder=impala_small --agent.p_aug=0.5`. DrQ variant swaps `encoder=drq, frame_stack=3, p_aug=1.0`.
- **Templates to clone:** `configs/gciql/cube_single_state.yaml` and `configs/gciql/cube_single_visual_drq.yaml`. Storage default (`configs/gciql/storage/shm.yaml`): `output_root=/dev/shm/gciql_outputs`, tmpfs (reboot-wiped).
- **Test harness:** run with `.venv/bin/python -m pytest` (repo memory: `uv run` is broken). Existing config tests live in `tests/test_drq_configs.py` and use a `_argv(name)` helper.
- **Hard constraint:** never edit `third_party/ogbench`.

## File structure

| File | Create/Modify | Responsibility |
|------|---------------|----------------|
| `configs/gciql/cube_single_state_crl.yaml` | Create | CRL on state cube-single (alpha=3.0, 1M steps, no encoder). |
| `configs/gciql/cube_single_visual_crl_drq.yaml` | Create | CRL on pixel cube-single with the DrQ front-end (alpha=3.0, encoder=drq). |
| `tests/test_drq_configs.py` | Modify | Add two argv tests for the new configs. |
| `scripts/_drq_worker.sh` | Modify | Name per-run logs from the config name (handles crl/state/pixel). |
| `scripts/run_drq_pixel_sweep.sh` | Modify | Accept explicit `JOBS` override + `DRYRUN` guard; generic tip line. |

---

## Task 1: CRL state config + argv test

**Files:**
- Create: `configs/gciql/cube_single_state_crl.yaml`
- Test: `tests/test_drq_configs.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_drq_configs.py`:

```python
def test_crl_state_config_argv():
    argv = _argv("cube_single_state_crl")
    assert "--agent=agents/crl.py" in argv
    assert "--agent.alpha=3.0" in argv
    assert "--run_group=factored-fb-crl" in argv
    assert "--env_name=cube-single-play-v0" in argv
    assert "--eval_on_cpu=1" in argv
    # state observations => no visual encoder override
    assert not any(a.startswith("--agent.encoder=") for a in argv)
```

- [ ] **Step 2: Run it — expect FAIL (config missing)**

Run: `.venv/bin/python -m pytest tests/test_drq_configs.py::test_crl_state_config_argv -v`
Expected: FAIL with Hydra `MissingConfigException` (cube_single_state_crl not found).

- [ ] **Step 3: Create `configs/gciql/cube_single_state_crl.yaml`**

```yaml
# CRL on state cube-single. Clone of cube_single_state.yaml; the only deltas are
# the agent (agents/crl.py), alpha (3.0, OGBench's CRL cube recipe per
# impls/hyperparameters.sh), and run_group — so the only difference vs the GCIQL
# state baseline is the RL algorithm. State observations => no visual encoder.
defaults:
  - storage: shm
  - _self_
variant: state
env_name: cube-single-play-v0
agent_file: agents/crl.py
seed: 0
train_steps: 1000000
log_interval: 5000
eval_interval: 100000
save_interval: 1000000
eval_episodes: 50
eval_on_cpu: 1
run_group: factored-fb-crl
agent_overrides:
  alpha: 3.0
wandb_mode: online
```

- [ ] **Step 4: Run the test — expect PASS**

Run: `.venv/bin/python -m pytest tests/test_drq_configs.py::test_crl_state_config_argv -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add configs/gciql/cube_single_state_crl.yaml tests/test_drq_configs.py
git commit -m "feat(crl-cube): CRL state config + argv test"
```

---

## Task 2: CRL pixel/DrQ config + argv test

**Files:**
- Create: `configs/gciql/cube_single_visual_crl_drq.yaml`
- Test: `tests/test_drq_configs.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_drq_configs.py`:

```python
def test_crl_pixel_drq_config_argv():
    argv = _argv("cube_single_visual_crl_drq")
    assert "--agent=agents/crl.py" in argv
    assert "--agent.encoder=drq" in argv
    assert "--agent.frame_stack=3" in argv
    assert "--agent.p_aug=1.0" in argv
    assert "--agent.alpha=3.0" in argv
    assert "--agent.batch_size=256" in argv
    assert "--run_group=factored-fb-crl-pixel-drq" in argv
    assert "--env_name=visual-cube-single-play-v0" in argv
```

- [ ] **Step 2: Run it — expect FAIL (config missing)**

Run: `.venv/bin/python -m pytest tests/test_drq_configs.py::test_crl_pixel_drq_config_argv -v`
Expected: FAIL with Hydra `MissingConfigException`.

- [ ] **Step 3: Create `configs/gciql/cube_single_visual_crl_drq.yaml`**

```yaml
# CRL on visual cube-single with FB's DrQ visual front-end (encoder + random_shifts
# augmentation + frame_stack=3 + RGBNorm), so the only FB-vs-baseline difference is
# the RL algorithm. encoder=drq is registered, and p_aug=1.0 + the augment swap are
# wired, by tools/wandb_mode_shim (gated on OGBENCH_DRQ_FRONTEND, which run_gciql.py
# sets because encoder == 'drq'). Deltas vs cube_single_visual_drq.yaml: agent
# (agents/crl.py), alpha (3.0, OGBench's CRL visual recipe), run_group.
defaults:
  - storage: shm
  - _self_
variant: visual
env_name: visual-cube-single-play-v0
agent_file: agents/crl.py
seed: 0
train_steps: 500000
log_interval: 5000
eval_interval: 100000
save_interval: 50000
eval_episodes: 50
eval_on_cpu: 0
run_group: factored-fb-crl-pixel-drq
agent_overrides:
  alpha: 3.0
  batch_size: 256
  encoder: drq
  frame_stack: 3
  p_aug: 1.0
wandb_mode: online
```

- [ ] **Step 4: Run the test — expect PASS**

Run: `.venv/bin/python -m pytest tests/test_drq_configs.py::test_crl_pixel_drq_config_argv -v`
Expected: PASS.

- [ ] **Step 5: Run the whole config test file (regression)**

Run: `.venv/bin/python -m pytest tests/test_drq_configs.py -v`
Expected: all 4 tests PASS (2 existing DrQ + 2 new CRL).

- [ ] **Step 6: Commit**

```bash
git add configs/gciql/cube_single_visual_crl_drq.yaml tests/test_drq_configs.py
git commit -m "feat(crl-cube): CRL pixel/DrQ config + argv test"
```

---

## Task 3: Generalize the sweep launcher (log naming + JOBS + DRYRUN)

The launcher hardcodes `{gciql,gcivl}_pixel_drq_...` log names and builds jobs only as CONFIGS×SEEDS (one SEEDS list). Make logs config-name-driven, add an explicit `JOBS` override for the heterogeneous 10+5 matrix, and add a `DRYRUN` guard so the assignment is verifiable without spawning training.

**Files:**
- Modify: `scripts/_drq_worker.sh:16-17`
- Modify: `scripts/run_drq_pixel_sweep.sh:28-32`, `:49-53`, `:56`

- [ ] **Step 1: Generalize per-run log naming in `scripts/_drq_worker.sh`**

Replace these two lines (`scripts/_drq_worker.sh:16-17`):

```bash
    case "$cfg" in *gcivl*) sh=gcivl ;; *) sh=gciql ;; esac
    log="logs/${sh}_pixel_drq_s${seed}_${TS}.log"
```

with:

```bash
    log="logs/${cfg}_s${seed}_${TS}.log"
```

(So a job `cube_single_visual_crl_drq:3` logs to `logs/cube_single_visual_crl_drq_s3_<ts>.log`; state logs are no longer mislabeled `pixel_drq`. Affects future runs only.)

- [ ] **Step 2: Add the `JOBS` override in `scripts/run_drq_pixel_sweep.sh`**

Replace the job-list block (`scripts/run_drq_pixel_sweep.sh:28-32`):

```bash
# Build job list "config:seed" (configs outer, seeds inner).
JOBS=()
for cfg in "${CONFIGS[@]}"; do
    for s in "${SEEDS[@]}"; do JOBS+=("$cfg:$s"); done
done
```

with:

```bash
# Build job list "config:seed". An explicit JOBS env override (space-separated
# "config:seed") wins — needed for heterogeneous matrices (e.g. 10 state + 5
# pixel seeds). Otherwise build the CONFIGS x SEEDS grid (back-compat).
read -r -a _jobs_in <<< "${JOBS:-}"
if [ ${#_jobs_in[@]} -gt 0 ]; then
    JOBS=("${_jobs_in[@]}")
else
    JOBS=()
    for cfg in "${CONFIGS[@]}"; do
        for s in "${SEEDS[@]}"; do JOBS+=("$cfg:$s"); done
    done
fi
```

- [ ] **Step 3: Add the `DRYRUN` guard in `scripts/run_drq_pixel_sweep.sh`**

Replace the worker-spawn block (`scripts/run_drq_pixel_sweep.sh:49-53`):

```bash
    GPU="$g" GJOBS="${gjobs[*]}" TS="$TS" WANDB_MODE="$WANDB_MODE" STORAGE="$STORAGE" \
        nohup bash "$REPO/scripts/_drq_worker.sh" \
        > "logs/drq_sweep_worker_g${g}_${TS}.log" 2>&1 &
    echo "  worker pid=$! gpu=$g" | tee -a "$manifest"
    sleep 8  # stagger JIT/EGL startup to avoid a host-RAM OOM spike
```

with:

```bash
    if [ "${DRYRUN:-0}" = "1" ]; then
        echo "  [dryrun] would launch worker gpu=$g jobs=${gjobs[*]}" | tee -a "$manifest"
        continue
    fi
    GPU="$g" GJOBS="${gjobs[*]}" TS="$TS" WANDB_MODE="$WANDB_MODE" STORAGE="$STORAGE" \
        nohup bash "$REPO/scripts/_drq_worker.sh" \
        > "logs/drq_sweep_worker_g${g}_${TS}.log" 2>&1 &
    echo "  worker pid=$! gpu=$g" | tee -a "$manifest"
    sleep 8  # stagger JIT/EGL startup to avoid a host-RAM OOM spike
```

- [ ] **Step 4: Make the tip line generic** — replace `scripts/run_drq_pixel_sweep.sh:56`:

```bash
echo "[drq-sweep] tail a run: tail -f logs/gciql_pixel_drq_s0_${TS}.log"
```

with:

```bash
echo "[drq-sweep] tail a run: tail -f logs/<config>_s<seed>_${TS}.log  (see $manifest)"
```

- [ ] **Step 5: Syntax-check both scripts**

Run: `bash -n scripts/_drq_worker.sh && bash -n scripts/run_drq_pixel_sweep.sh && echo OK`
Expected: `OK` (no syntax errors).

- [ ] **Step 6: Verify the JOBS override + assignment with a DRYRUN**

Run:
```bash
DRYRUN=1 GPUS="6 7" \
JOBS="cube_single_visual_crl_drq:0 cube_single_visual_crl_drq:1 cube_single_state_crl:0 cube_single_state_crl:1" \
bash scripts/run_drq_pixel_sweep.sh
```
Expected: prints a manifest with `jobs=4`, and a round-robin split — GPU 6 gets `cube_single_visual_crl_drq:0 cube_single_state_crl:0`, GPU 7 gets `cube_single_visual_crl_drq:1 cube_single_state_crl:1`, each line marked `[dryrun] would launch worker`. No python process starts (`pgrep -af run_gciql.py` shows nothing new).

- [ ] **Step 7: Verify back-compat (no JOBS) still builds the grid**

Run:
```bash
DRYRUN=1 GPUS="6 7" SEEDS="0 1" CONFIGS="cube_single_state_crl" \
bash scripts/run_drq_pixel_sweep.sh
```
Expected: `jobs=2`; GPU 6 ← `cube_single_state_crl:0`, GPU 7 ← `cube_single_state_crl:1`.

- [ ] **Step 8: Commit**

```bash
git add scripts/_drq_worker.sh scripts/run_drq_pixel_sweep.sh
git commit -m "feat(crl-cube): config-name log naming + JOBS/DRYRUN in sweep launcher"
```

---

## Task 4: Smoke test both configs (manual gate)

No new code. Confirms each config trains end-to-end before the full sweep. Needs `/dev/shm/.venv-jax` (else `make install-jax`) and free, healthy GPUs (6, 7). Each smoke is ~2k steps with wandb disabled.

- [ ] **Step 1: GCIVL/state — CRL state smoke on GPU 6**

```bash
CUDA_VISIBLE_DEVICES=6 .venv/bin/python run_gciql.py \
    --config-name cube_single_state_crl \
    storage=shm seed=0 wandb_mode=disabled \
    train_steps=2000 eval_interval=1000 eval_episodes=2 save_interval=2000 \
    run_group=crl-smoke
```
Check: starts without error (`agents/crl.py` loads, no encoder needed for state), training loss decreases over 2k steps, eval at 1000/2000 runs, a checkpoint `params_2000.pkl` appears under `/dev/shm/gciql_outputs/factored-fb/crl-smoke/sd000_*/`.

- [ ] **Step 2: CRL pixel/DrQ smoke on GPU 7**

```bash
CUDA_VISIBLE_DEVICES=7 .venv/bin/python run_gciql.py \
    --config-name cube_single_visual_crl_drq \
    storage=shm seed=0 wandb_mode=disabled \
    train_steps=2000 eval_interval=1000 eval_episodes=2 save_interval=2000 \
    run_group=crl-smoke
```
Check (the key validation for this plan):
- no `KeyError: 'drq'` — the encoder resolves for CRL,
- CRL's 5 encoders + the 18-channel actor GCEncoder concat build with **no shape error**,
- fits on one 23 GB GPU (no CUDA OOM),
- loss decreases; eval at 1000/2000 runs (no crash from the concat),
- throughput healthy (~20+ it/s, **not** the ~20 s/it degraded-GPU pathology),
- `params_2000.pkl` written under `/dev/shm/gciql_outputs/factored-fb/crl-smoke/sd000_*/`.

- [ ] **Step 3: Record the outcome**

Note both smoke results (it/s, loss trend, checkpoint path) in the run notes. If the pixel smoke OOMs or errors on the concat, STOP and report before launching the sweep. (No commit — these write only to `/dev/shm`.)

---

## Task 5: Launch the sweep — batch 1 now on GPUs 6,7 (manual gate)

Launch only after Task 4 passes. Batch 1 takes all 5 pixel seeds + state seeds 0–4 (10 jobs) on the free, healthy GPUs 6,7; batch 2 (Task 6) takes state seeds 5–9 on 3,5 once GCIVL frees them. Pixel jobs are listed first so they start immediately on both GPUs. Run groups come from the configs (`factored-fb-crl`, `factored-fb-crl-pixel-drq`); per-run logs land in `logs/`.

- [ ] **Step 1: Confirm GPUs 6,7 are free**

Run: `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader`
Expected: indices 6 and 7 show ~0 MiB used / 0 % util.

- [ ] **Step 2: Launch batch 1**

```bash
GPUS="6 7" WANDB_MODE=online STORAGE=shm \
JOBS="cube_single_visual_crl_drq:0 cube_single_visual_crl_drq:1 cube_single_visual_crl_drq:2 cube_single_visual_crl_drq:3 cube_single_visual_crl_drq:4 cube_single_state_crl:0 cube_single_state_crl:1 cube_single_state_crl:2 cube_single_state_crl:3 cube_single_state_crl:4" \
bash scripts/run_drq_pixel_sweep.sh
```
Expected: a manifest `logs/drq_sweep_<ts>.manifest` printed with `jobs=10 gpus=6 7`, two worker PIDs (one per GPU), and the per-GPU job assignment.

- [ ] **Step 3: Confirm the runs are alive**

Run: `pgrep -af 'main.py' | grep -E 'crl' ; sleep 30; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader`
Expected: two `main.py ... agents/crl.py` processes; GPUs 6,7 climbing in memory/util. Tail a run: `tail -f logs/cube_single_visual_crl_drq_s0_<ts>.log`.

---

## Task 6: Launch the sweep — batch 2 on GPUs 3,5 when free (manual gate)

Do this once the in-flight GCIVL DrQ runs finish and GPUs 3,5 are idle (~4–5h after the GCIVL launch). Batch 2 covers the remaining state seeds 5–9. Seed dirs are independent, so there is no collision with batch 1.

- [ ] **Step 1: Confirm GPUs 3,5 are free**

Run: `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader`
Expected: indices 3 and 5 show ~0 MiB / 0 %. If the GCIVL runs are still going, wait.

- [ ] **Step 2: Reassess balance (judgment, not a command)**

Check batch-1 progress (`tail logs/drq_sweep_worker_g6_*.log logs/drq_sweep_worker_g7_*.log`). The default batch 2 is state seeds 5–9. If batch 1 has barely touched its state jobs and 3,5 will otherwise idle, it's fine — state runs are cheap; the goal is just to not waste 3,5. Do **not** re-queue any seed already started in batch 1.

- [ ] **Step 3: Launch batch 2**

```bash
GPUS="3 5" WANDB_MODE=online STORAGE=shm \
JOBS="cube_single_state_crl:5 cube_single_state_crl:6 cube_single_state_crl:7 cube_single_state_crl:8 cube_single_state_crl:9" \
bash scripts/run_drq_pixel_sweep.sh
```
Expected: manifest with `jobs=5 gpus=3 5`; GPU 3 ← seeds 5,7,9 and GPU 5 ← seeds 6,8 (round-robin).

- [ ] **Step 4: Confirm alive**

Run: `pgrep -af 'main.py' | grep -E 'cube_single_state_crl|agents/crl.py'`
Expected: new `agents/crl.py` state processes on 3,5. Tail: `tail -f logs/cube_single_state_crl_s5_<ts>.log`.

- [ ] **Step 5: Record the full sweep manifest paths**

Note both batch manifests (`logs/drq_sweep_<ts>.manifest`) and the run-group output dirs (`/dev/shm/gciql_outputs/factored-fb/factored-fb-crl{,-pixel-drq}/`) in the run notes. Reminder: `/dev/shm` is tmpfs — copy completed runs to durable storage before any reboot (out of scope for this plan).

---

## Self-review (completed during planning)

- **Spec coverage:** Component 1 configs → Tasks 1,2; Component 2 launcher generalization → Task 3; Component 3 argv tests → Tasks 1,2 (+ regression in Task 2 Step 5); smoke gate → Task 4; two-batch launch (10 state + 5 pixel, 6,7 then 3,5) → Tasks 5,6. CRL 5-encoder/18-ch risk → exercised by Task 4 Step 2. `/dev/shm` tmpfs caveat → Task 6 Step 5.
- **Placeholder scan:** none — every config/test/script step has complete content; the only manual steps (smoke, launch) have concrete commands and explicit pass criteria.
- **Type/name consistency:** config names `cube_single_state_crl` / `cube_single_visual_crl_drq`, run_groups `factored-fb-crl` / `factored-fb-crl-pixel-drq`, log scheme `logs/<cfg>_s<seed>_<ts>.log`, env vars `JOBS` / `DRYRUN` / `GPUS` — used identically across Tasks 1–6. No agent code changed (verified: `build_ogbench_argv` and `build_child_env` are agent-agnostic).
