# SGCRL Air Hockey: Unsupervised_Manipulation vs Factored-FB

**Date:** 2026-06-26  
**Task:** `AirHockeyPuckGoalPosition-v0` (plain SGCRL, no IWS)  
**Config:** `puck_goal_real_transfer_s4.yaml` / UM equivalent  
**W&B project:** [calcharles/air_hockey](https://wandb.ai/calcharles/air_hockey)

---

## Summary

At ~3M environment steps (seed 7, 256 train envs, 512×512 MLP, repr_dim=256), **Unsupervised_Manipulation (UM) reaches ~28–31% `eval/success_once` while the Factored-FB (FB) port reaches ~22%**. Training losses (actor/critic) track closely, but **training-time success is also lower on FB** (~8% per-episode vs ~15% on UM at the same step count), so the gap is not explained by eval alone.

The two code paths are intentionally aligned (PyTorch SGCRL + episode replay + geometric future goals), but several **protocol, infrastructure, and RNG** differences remain. None of the core algorithm hyperparameters differ for SGCRL mode; the most actionable gaps are **eval seed mismatch**, **run fragmentation / resume without replay**, and **unverified env/action-space parity**.

---

## Observed numbers (@ ~3M env steps)

| Source | Local run | W&B | Steps | Best `eval/success_once` @ 3M | Notes |
|--------|-----------|-----|-------|-------------------------------|-------|
| UM | `20260624_150602_*` | [fk7khukf](https://wandb.ai/calcharles/air_hockey/runs/fk7khukf) | 3.5M | **31.3%** | `cpytorch` backend (falls back to `pytorch` on aarch64) |
| UM | `20260625_193218_*` | [xyn93rt3](https://wandb.ai/calcharles/air_hockey/runs/xyn93rt3) | 3.7M | **28.1%** @ 2.5M | plain `pytorch` |
| FB | `20260625_210426_*` | [76f1tn2t](https://wandb.ai/calcharles/air_hockey/runs/76f1tn2t) | 3.1M | **21.9%** | only FB run with a full eval curve |

Eval curve (same checkpoints, approximate step alignment):

| Env steps | UM `150602` | UM `193218` | FB `210426` |
|-----------|-------------|-------------|-------------|
| 0.5M | 0% | 0% | 3.1% |
| 1.0M | 9.4% | 12.5% | 6.3% |
| 1.5M | 12.5% | 12.5% | 15.6% |
| 2.0M | 21.9% | 25.0% | 9.4% |
| 2.5M | 18.8% | **28.1%** | 18.8% |
| 3.0M | **31.3%** | 28.1% | **21.9%** |

Training losses at ~3M are similar (actor ≈ −0.28 to −0.29, critic ≈ 3.9–4.0), but **training success differs**:

- UM `193218`: `train/success_once` ≈ 0.15 at ~3M steps  
- FB `210426`: `train/success_once` ≈ 0.08 at the same checkpoint (cumulative ≈ 0.02)

---

## Implementation map

| Component | Unsupervised_Manipulation | Factored-FB |
|-----------|---------------------------|-------------|
| Launcher | `scripts/train_air_hockey_task.sh` | `scripts/agents/crl_iws/air_hockey_sgcrl.sh` |
| **Multiseed launcher** | [`scripts/train_air_hockey_sgcrl_multiseed.sh`](../../Unsupervised_Manipulation/scripts/train_air_hockey_sgcrl_multiseed.sh) | [`scripts/agents/crl_iws/air_hockey_sgcrl_multiseed.sh`](../scripts/agents/crl_iws/air_hockey_sgcrl_multiseed.sh) |
| Training loop | `src/.../training/loops.py` | `scripts/data/run_crl_iws_train.py` |
| Agent | `algorithms/pytorch/sgcrl.py` | `agents/crl_iws/agent_torch.py` |
| Replay | `data/episodes.py` (`VectorizedReplayBuffer`, GPU tensors) | `agents/crl_iws/replay.py` (`IWSReplayBuffer`, NumPy) |
| Env wrapper | `envs/air_hockey.py` + `air_hockey_tasks.py` | `agents/crl_iws/air_hockey_env.py` |
| Eval | `training/loops.py` → `evaluate_policy_metrics` | `agents/crl_iws/eval.py` |

Migration notes: `.cursor/CRL_IWS_MIGRATION_PLAN.md`

---

## What already matches (ruled out as primary causes)

These were checked against source and configs; differences here are **not** expected to explain a ~10-point eval gap on their own.

1. **SGCRL hyperparameters** — 256 envs, batch 64, 16 updates/iter, 200k random explore, 512×512 MLP, repr_dim 256, lr 3e-4, cosine CPC, `random_goals=0.5`, `future_discount=0.99`, `logsumexp_coeff=0.01`, contact_distance 0.09 for SGCRL.
2. **Actor / critic architecture** — tanh-squashed Normal actor, dual MLP encoders, dot/cosine CPC loss, actor objective `-diag(logits)`.
3. **Goal relabeling (SGCRL mode)** — uniform episode + step sampling, geometric future index, goal = `achieved_goals[future_index]` (not desired goal).
4. **Observation / goal construction** — strip last 2 dims from raw obs; achieved goal = puck `(x,y)`; desired goal from `env.get_desired_goal()`.
5. **Air-hockey YAML** — same sysid config (`randomize_paddle_spawn: true`, hist_len=4, etc.).
6. **Terminate on success** — both cut episodes early on success during training.
7. **Policy conditioning at rollout** — both feed `concat(state, desired_goal)` to the actor during data collection.

---

## Likely causes (ranked)

### 1. Eval protocol mismatch (inflates *reported* gap)

| Setting | UM | FB |
|---------|----|----|
| Eval base seed | `evaluation.base_seed = 0` | training `--seed 7` passed into eval |
| Eval schedule | aligned to `train_num_envs` (256) → 500224, 1000192, … | exact multiples → 500000, 1000000, … |
| Eval env | persistent eval env pool, `auto_reset=False` | fresh `SyncVectorEnv` per batch, `auto_reset=False` |

**Impact:** Different eval seeds alone can shift `success_once` by several points on a 32-episode eval. This does **not** explain lower **training** success on FB, but it makes W&B curves not directly comparable without aligning seeds.

**Fix:** Run FB with eval seeds derived from `base_seed=0` (or run UM with `base_seed=7`) and the same schedule alignment.

References:

- UM: `training/loops.py` → `evaluate_policy_metrics`, `_build_eval_schedule`
- FB: `agents/crl_iws/eval.py`, `run_crl_iws_train.py` → `_next_eval_step`

---

### 2. Run fragmentation and resume without replay (FB training quality)

FB SGCRL work split across several runs:

| Run | Steps | Eval logged? |
|-----|-------|--------------|
| `20260625_163329` | 1.4M | no |
| `20260625_170713` | **5.6M** | no |
| `20260625_210426` | 3.1M (with eval) | yes |

Resume loads **agent weights only** — the replay buffer is always re-created empty:

```python
# FB: scripts/data/run_crl_iws_train.py
replay = IWSReplayBuffer(...)
if args.resume_from:
    payload = _load_training_checkpoint(...)  # agent only
```

UM behaves the same on resume (`loops.py` re-initializes `replay_manager`). Any resumed FB run trains with a **cold replay buffer** while the policy is mid-training, which hurts sample diversity and can depress success until the buffer refills (~200k transitions minimum, but meaningful diversity needs many episodes).

**Impact:** High for resumed runs; the headline comparison uses FB `210426` which may or may not have been a clean continuous run (W&B metadata shows `resume_from=None`, but other attempts used checkpoint resume).

**Fix:** Compare only **continuous, non-resumed** runs; or add replay serialization to checkpoints.

---

### 3. Replay implementation: NumPy (FB) vs Torch (UM)

| | UM | FB |
|--|----|----|
| Storage | `VectorizedReplayBuffer` on `device=cuda` | `IWSReplayBuffer` in NumPy |
| RNG | `torch.rand` / `torch.multinomial` on GPU | `np.random` on CPU |
| Sampling logic | `context_mode="episode"`, rep=1 for SGCRL | `use_iws=False`, `context_repetition_factor=1` |

The **algorithms are the same on paper**, but RNG streams diverge immediately. Over 3M steps this produces different trajectories even with the same seed.

**Impact:** Medium — explains non-identical curves; unlikely to alone cause a stable 10-point deficit unless coupled with other bugs.

**Fix:** Log-side-by-side replay batch hashes for the first N updates with frozen env RNG.

---

### 4. Vectorized env stepping: `SyncVectorEnv` (FB) vs custom loop (UM)

- **FB:** `gymnasium.vector.SyncVectorEnv` over `AirHockeyCRLIWSEnv` factories (`air_hockey_env.py`).
- **UM:** `AirHockeyVectorGoalEnv` steps a Python list of envs sequentially (`envs/air_hockey.py`).

Both use independent Box2D sims per env, but ordering, reset timing, and info dict layout differ. FB hardcodes the **action space** as `Box(-1, 1)` in the wrapper instead of reading from the underlying env:

```python
# agents/crl_iws/air_hockey_env.py
self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
```

UM uses `env.single_action_space` from air-hockey-rl. If upstream bounds ever differ from `[-1, 1]`, FB would silently mis-scale actions for the critic (`_scale_action` assumes wrapper bounds match tanh output).

**Impact:** Medium if bounds mismatch; low if bounds are truly `[-1,1]` (likely for this task).

**Fix:** Assert `env.action_space` matches wrapper; log first-step action range from both repos.

---

### 5. Air-hockey package source path

| Run | `air_hockey_repo` |
|-----|-------------------|
| UM `150602` | `../rlmini-envs` |
| UM `193218` | `../air-hockey-rl` |
| FB | via `rlmini-envs` symlink → air-hockey-rl submodule |

Different checkouts or submodule SHAs can change physics, success detection, or observation layout without any algorithm change.

**Impact:** Medium until commit SHAs are pinned and matched.

**Fix:** Log `airhockey` / submodule git SHA in both trackers; pin to the same revision.

---

### 6. Backend: `cpytorch` vs plain PyTorch (UM only)

UM defaults to `ALGORITHM_BACKEND=cpytorch` (`train_air_hockey_task.sh`); on TACC aarch64 this **falls back to `pytorch`** because Triton/`torch.compile` fails. FB always uses plain PyTorch (`agent_torch.py`).

**Impact:** Low on GH200 aarch64 (both effectively PyTorch). Could matter on x86 if UM actually compiles.

---

### 7. Missing eval on early FB long run

`20260625_170713` reached **5.6M steps** but logged **no eval metrics** locally (eval wiring was added later in `210426`). The best FB number we quote comes from a **shorter, different** run, not from the longest training effort.

**Impact:** We may be **under-estimating** FB if the 5.6M policy is better but unmeasured — or over-estimating UM if its curve is noisier. Needs eval on the 5.6M checkpoint.

---

## What does *not* explain the gap

- **IWS sampling** — disabled for SGCRL in both (`use_iws=False` / `context_repetition_factor=1`).
- **Different task config** — same real_transfer_s4 YAML content.
- **Critic score type** — both use `cosine` for the compared runs.
- **Desired vs achieved goal at train time** — both condition the actor on desired goal; both relabel with achieved future goals in the critic batch.

---

## Recommended validation plan

1. **Matched eval protocol** — same `base_seed`, same schedule alignment (`align_to=256`), same 32 episodes / 128 eval envs. Re-log one checkpoint from each repo.
2. **Continuous runs only** — fresh 3M-step SGCRL with no resume; no checkpoint comparison across fragmented runs.
3. **Pin air-hockey-rl SHA** — record in `config.json` / W&B config on both sides.
4. **Action-space audit** — print underlying env `action_space` vs FB wrapper bounds; verify stored replay actions match env-step actions.
5. **Replay parity test** — unit test: fixed episode tensor → assert identical `(states, actions, goals)` batches between UM and FB samplers (SGCRL mode).
6. **Eval the 5.6M FB checkpoint** — load `runs/crl-iws-air-hockey/checkpoints/checkpoint_05605376.pt` and run `agents/crl_iws/eval.py` with UM-matched seeds.
7. **Optional:** save/load replay buffer in checkpoints before trusting resume for long runs.

---

## Quick commands

### Multiseed SGCRL campaign (recommended)

4 seeds (`7 11 42 43`), auto-resume on disconnect, W&B group `air_hockey_sgcrl_multiseed`.

| Repo | Launch script |
|------|---------------|
| **Factored-FB** | [`scripts/agents/crl_iws/air_hockey_sgcrl_multiseed.sh`](../scripts/agents/crl_iws/air_hockey_sgcrl_multiseed.sh) |
| **Unsupervised_Manipulation** | [`../Unsupervised_Manipulation/scripts/train_air_hockey_sgcrl_multiseed.sh`](../../Unsupervised_Manipulation/scripts/train_air_hockey_sgcrl_multiseed.sh) |

**Factored-FB:**

```bash
cd Factored-FB
mkdir -p logs
screen -S sgcrl_fb
bash scripts/agents/crl_iws/air_hockey_sgcrl_multiseed.sh 2>&1 | tee logs/sgcrl_multiseed.log
# Ctrl-a d to detach; screen -r sgcrl_fb to reattach
```

**Unsupervised_Manipulation:**

```bash
cd Unsupervised_Manipulation
mkdir -p logs
screen -S sgcrl_um
bash scripts/train_air_hockey_sgcrl_multiseed.sh 2>&1 | tee logs/sgcrl_multiseed.log
# Ctrl-a d to detach; screen -r sgcrl_um to reattach
```

Preview without launching:

```bash
DRY_RUN=1 bash scripts/agents/crl_iws/air_hockey_sgcrl_multiseed.sh          # FB
DRY_RUN=1 bash scripts/train_air_hockey_sgcrl_multiseed.sh                 # UM
```

By default all seeds run **in parallel** (`MAX_PARALLEL` = number of seeds). With one
GPU, set `GPUS=0` and each job shares that device. Use `MAX_PARALLEL=1` for sequential.

```bash
# 4 seeds in parallel on GPU 0 (default)
SEEDS="7 11 42 43" GPUS="0" bash scripts/agents/crl_iws/air_hockey_sgcrl_multiseed.sh

# 4 seeds on 4 GPUs
SEEDS="7 11 42 43" GPUS="0 1 2 3" bash scripts/agents/crl_iws/air_hockey_sgcrl_multiseed.sh

# sequential on one GPU
MAX_PARALLEL=1 SEEDS="7 11 42 43" GPUS="0" bash scripts/agents/crl_iws/air_hockey_sgcrl_multiseed.sh
```

**Where outputs land (both repos, per seed):**

| Output | Path |
|--------|------|
| Metrics, checkpoints, eval videos | `runs/air_hockey/sgcrl_seed<SEED>/` |
| Per-seed launcher log | `logs/air_hockey_sgcrl_multiseed_<timestamp>/seed_<SEED>.log` |
| W&B dashboard | [wandb.ai/calcharles/air_hockey?group=air_hockey_sgcrl_multiseed](https://wandb.ai/calcharles/air_hockey?group=air_hockey_sgcrl_multiseed) |

After a disconnect, re-run the same launch script; incomplete seeds resume from the latest checkpoint in their seed directory.

---

## Multiseed SGCRL campaign (2026-06-26)

Four seeds (`7 11 42 43`), parallel on one GPU, W&B group `air_hockey_sgcrl_multiseed`. The run stopped early (~50 min, external kill) at ~10% of 20M steps.

### Timestep-synchronized comparisons

Eval and checkpoint milestones are aligned to **`ceil(step / num_envs) * num_envs`** with `num_envs=256` (matches UM `loops.py`). Examples @ 500k nominal:

| Nominal | Aligned step |
|---------|--------------|
| 0.5M | **500,224** |
| 1.0M | **1,000,192** |
| 1.5M | **1,500,160** |
| 2.0M | **2,000,128** |

Factored-FB `run_crl_iws_train.py` now uses the same schedule as UM. **Compare only at these exact steps** — do not match FB `500,000` against UM `500,224`.

**Compare SGCRL or IWS multiseed runs:**

```bash
cd Factored-FB
python3 scripts/agents/crl_iws/compare_air_hockey_multiseed.py --campaign sgcrl
python3 scripts/agents/crl_iws/compare_air_hockey_multiseed.py --campaign iws
python3 scripts/agents/crl_iws/verify_iws_parity.py   # config audit + synced IWS table
```

Optional: `--max-milestone 3000000` to limit the table; `--align-to 256` (default).

### FB vs UM: are they matched?

**Yes — close enough at matched eval steps**, within seed noise. The earlier ~10pt gap at 3M (single-seed) is **not reproduced** in this multiseed snapshot at 1.0–1.5M:

| Eval @ | FB mean ± std | UM mean ± std | Δ (UM−FB) |
|--------|---------------|---------------|-----------|
| 0.5M | 3.1% ± 2.2% | 3.9% ± 2.6% | +0.8pp |
| 1.0M | 10.2% ± 9.7% | 7.8% ± 7.2% | −2.3pp |
| 1.5M | 19.5% ± 4.6% | 14.1% ± 5.2% | −5.5pp |

Per-seed @ 1.5M: FB leads on seeds 7/42, UM leads on 43, ~tie on 11. High variance across 4 seeds — **treat as parity pending longer runs**, not proof of identical performance.

**Infrastructure note:** FB checkpoints were written to a shared default dir (`runs/crl-iws-air-hockey/checkpoints/`) instead of per-seed run dirs — **fixed** (`checkpoint_dir` default is now `None` → `<run_dir>/checkpoints/`).

---

## IWS multiseed comparison (Interaction-Weighted Sampling)

IWS = interaction-weighted replay (UM: `interaction_weighted_sampling`; FB: `--algorithm iws`). Same multiseed layout as SGCRL.

### Fixes applied before IWS campaign

| Issue | Fix |
|-------|-----|
| FB IWS used MetaWorld contact radii (`σ=20`, `d=0.09`, near=0.18) | `_resolve_algorithm_config` now uses UM `iws.yaml` air_hockey defaults: `σ=30`, `d=0.12`, near=**0.24**, rep=8, full-episode future |
| FB checkpoints saved to global dir, breaking multiseed resume | `checkpoint_dir` default `None`; checkpoints go to `runs/air_hockey/iws_seed<SEED>/checkpoints/` |
| No multiseed IWS launchers | Added scripts below |

### Launch IWS multiseed (4 seeds, parallel on GPU 0)

**Factored-FB:**

```bash
cd Factored-FB
mkdir -p logs
screen -S iws_fb
bash scripts/agents/crl_iws/air_hockey_iws_multiseed.sh 2>&1 | tee logs/iws_multiseed.log
```

**Unsupervised_Manipulation:**

```bash
cd Unsupervised_Manipulation
mkdir -p logs
screen -S iws_um
bash scripts/train_air_hockey_iws_multiseed.sh 2>&1 | tee logs/iws_multiseed.log
```

| Repo | Multiseed script | Run dir per seed | W&B group |
|------|------------------|------------------|-----------|
| FB | [`scripts/agents/crl_iws/air_hockey_iws_multiseed.sh`](../scripts/agents/crl_iws/air_hockey_iws_multiseed.sh) | `runs/air_hockey/iws_seed<SEED>/` | `air_hockey_iws_multiseed` |
| UM | [`scripts/train_air_hockey_iws_multiseed.sh`](../../Unsupervised_Manipulation/scripts/train_air_hockey_iws_multiseed.sh) | `runs/air_hockey/iws_seed<SEED>/` | `air_hockey_iws_multiseed` |

Preview: `DRY_RUN=1 bash …/air_hockey_iws_multiseed.sh`

### IWS representation parity checklist

Both repos should train the **same contrastive representations**:

| Component | FB | UM |
|-----------|----|----|
| Critic encoders | MLP 512→512→**256**, ReLU | `FeedForwardHead` 512→512→**256**, ReLU |
| Score | cosine dot product | cosine (`critic_score_type=cosine`) |
| Loss | CPC CE + 0.01·logsumexp² | same (`critic_type=cpc`) |
| Actor goals | 50% random roll (`random_goals=0.5`) | same |
| Batch goals | future **achieved_goal** | future **achieved_goal** |
| IWS sampling | rep=8, σ=30, d=0.12, near=0.24, full-episode future | same (`iws.yaml` air_hockey) |
| Phase labels | paddle–puck distance from env info | `hand_position` / `object_position` metrics |

**Audit your runs:**

```bash
cd Factored-FB
python3 scripts/agents/crl_iws/verify_iws_parity.py
```

**Fixed for parity (2026-06-30):** UM replay was using `state[..., 0:2]` / `[..., 4:6]` for IWS spatial weights on 30-dim air-hockey history obs (wrong geometry). It now stores per-step paddle–puck distances like FB (`hand_object_distances` in `episodes.py` + `interaction_replay.py`).

**Re-run UM IWS seeds after that fix** so new episodes use correct IWS spatial weighting. Existing UM checkpoints trained before the fix used phase-only event detection with incorrect spatial weights.

Check `replay/iws_events` in FB logs — should stay **≫100** per seed; seed 43 dropping to ~48 indicates broken IWS event cache (restart recommended).

---

### Single-seed baselines

**UM (reference):**

```bash
cd ../Unsupervised_Manipulation
bash scripts/train_air_hockey_task.sh   # ALGORITHM=sgcrl SEED=7
```

**FB (port):**

```bash
cd Factored-FB
bash scripts/agents/crl_iws/air_hockey_sgcrl.sh   # SEED=7
```

**Short smoke (FB, no W&B):**

```bash
uv run python scripts/data/run_crl_iws_train.py \
  --algorithm sgcrl \
  --total-steps 500000 \
  --num-envs 64 \
  --random-exploration-steps 5000 \
  --wandb-mode disabled
```

---

## Related docs

- [install.md](./install.md) — env / rlmini-envs setup  
- [../.cursor/CRL_IWS_MIGRATION_PLAN.md](../.cursor/CRL_IWS_MIGRATION_PLAN.md) — port plan and source file map  
- [rng_equivalence_findings.md](./rng_equivalence_findings.md) — methodology for proving RNG parity (FB vs td_jepa; same idea applies here)
