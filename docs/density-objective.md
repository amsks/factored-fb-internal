# The density successor measure — code guide

`measure_density` (spec 18) and its behavior-policy variant `measure_density_behavior`
(spec 19a). This is the reading order for someone who wants to understand the objective
from the code, plus how to train and evaluate it.

Specs (untracked, local only): `PAPER/Multistep-FB/specs/18-density-measure.md`,
`19a-behavior-density-impl.md`. Derivations: `PAPER/Multistep-FB/delta-measure.tex`.

---

## 1. What the objective is

The successor measure `M(s,a,g)` is the expected discounted time spent in a goal region.
Its total mass over the whole configuration space is always `1/(1-γ)`, so scaling by
`(1-γ)` turns it into a probability distribution over object configurations:

```
nu(.|s,a,g) = (1-gamma) * E[ sum_t gamma^t * delta(x_{t+1}) ]
```

"Where does my discounted future time go, as a distribution." Rewards are counted at the
NEXT state (`x_{t+1}`) — that convention is load-bearing and is asserted in the tests.

The predecessor (`measure_delta_partition`, spec 17b) represented this over 256 FIXED
prototype configurations: an MLP with K output logits. Density removes the fixed support
using two towers,

```
F(s,a,g) -> R^d ,   B(u) -> R^d ,   f(s,a,g,u) = F(s,a,g)^T B(u) / tau
```

so any configuration `u` can be scored. A distribution is obtained by softmaxing over a
query set `U` that is **resampled every update** (round 1: the batch's own next-state
object coordinates, so `M == B`). The query count is therefore a Monte-Carlo sample
size, not a spatial resolution.

TD target and loss (per head; `k` is a soft kernel around the observed next state,
`w_bar+` the target towers' distribution at the bootstrap action):

```
t = sg[ (1-gamma) * k + gamma * w_bar+ ]        # rows on the simplex
L = -E[ sum_j t_j * log w_j ]                    # cross-entropy between distributions
```

Non-negativity and unit mass are structural (softmax). Consequently this path carries
**no orthonormalization, no mass anchor, no scale normalization** — each of which was
separately measured to be harmful here (ortho took cube-single 0.967 -> 0.225).

---

## 2. Files to read, in order

Everything lives in one agent file plus its tests.

### 2.1 The agent — `tools/wandb_mode_shim/gciql_flow.py`

Read these in this order (line numbers drift; `grep -n` the symbol):

| what | symbol | ~line |
|---|---|---|
| the knobs + their docstrings | `config.measure_density` | 4013 |
| behavior-variant knob | `config.measure_density_behavior` | 4024 |
| the query set (eps-scaled) | `_density_queries` | 1430 |
| raw block columns (kernel input) | `_raw_block_cols` | 1442 |
| the soft kernel, shared with 17b | `_delta_phi` | 1403 |
| **the objective itself** | `density_loss` | 1975 |
| loss wiring (measure_only branch) | `total_loss` | 2415, density at 2481 |
| actor q-term (frozen params) | `actor_loss` | 950, density at 1089 |
| module registration + lattice | `create` | density block at 3147 |
| target soft-update for both towers | `update` | 2587 |

`density_loss` is the whole objective in ~40 lines and is the one function to read if you
read only one.

**The subtlety most likely to trip you up**: `_delta_phi` applies the per-block `eps`
scaling *internally*. So the kernel term is fed **raw** block rows (`_raw_block_cols`)
while the `B` tower is fed **pre-scaled** rows (`_density_queries`). Double-scaling
silently learns a different object with no visible symptom.

Second subtlety: the softmax is over the **query axis**, not the batch axis. The
batch-axis version also runs and also produces a plausible loss curve.

### 2.2 The behavior variant (spec 19a)

Two deltas on the above, for the occupancy model ORS consumes:
- the `F` tower sees `[s, a]` — no goal conditioning;
- the bootstrap action is the dataset's next action (SARSA), not a sample from the
  learned actor, which is goal-conditioned BC and would estimate a different policy.

Supporting change: `third_party/ogbench/impls/utils/datasets.py` gains a gated
`batch['next_actions']` (~line 225). That flag is **forced on** by the behavior knob
inside `create()`; note `create()` shadows its argument with `dict(config)`, so both the
local copy and the caller's object are written — GCDataset holds a live reference to the
caller's config and re-reads the flag per sample.

### 2.3 Tests (the executable contract)

| file | covers |
|---|---|
| `tests/test_gciql_measure_density.py` | the objective: CE vs numpy reference, simplex rows, eps-scaling, grad hygiene, lattice, default-off bit-identity |
| `tests/test_gciql_measure_density_behavior.py` | goal-free F input, SARSA bootstrap, pure-BC actor, `next_actions` boundary handling |
| `tests/test_eval_density_zeroshot.py` | both zero-shot readouts + pool sweep |
| `tests/test_eval_measure_g4_spearman.py` | the calibration probe, incl. its two controls |

Run: `.venv/bin/python -m pytest tests/test_gciql_measure_density.py -q` (NOT `uv run`;
the jax subprocess harness resolves the jax venv itself).

### 2.4 Configs

- `configs/gciql/cube_single_state_measure_density_d50.yaml` — the main arm
- `configs/gciql/cube_single_state_measure_density_hard_d50.yaml` — `measure_density_hard`
  ablation (immediate term becomes a one-hot at the row's own successor instead of the
  eps-kernel), which separates "smoothing matters" from "simplex TD matters"
- `configs/gciql/{cube_single,cube_double,scene}_state_measure_density_behavior_d50.yaml`
- `configs/gciql/cube_single_state_bconly_d50.yaml` — the BC-only control; same recipe
  with the actor's q-term zeroed, so a gap isolates the measure reaching the actor from
  the `bc_coeff=3.0` cloning term

---

## 3. How to run it

### 3.1 Train

```bash
XLA_FLAGS=--xla_gpu_enable_command_buffer= \
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python run_gciql.py --config-name cube_single_state_measure_density_d50 \
    seed=0 output_root=/home/adityamohan/exp_density exp_name=dens_s0
```

`XLA_FLAGS=--xla_gpu_enable_command_buffer=` disables XLA's CUDA-graph capture. Without
it we lost two runs to `CUDA_ERROR_STREAM_CAPTURE_INVALIDATED` raised from a device_get
at logging time, on two different GPUs. It costs a little throughput.

A 500k run is roughly 1-2.5h on one GPU. For a smoke test add
`train_steps=300 log_interval=100 eval_interval=1000000 wandb_mode=offline`.

### 3.2 What to watch in `train.csv`

| key | healthy on cube-single | meaning |
|---|---|---|
| `density_ce` | falls, plateaus below `log(M)` | the cross-entropy |
| `density_ess_median` | 87-91 (gate: >= 4) | effective support; **median**, not the mean — a one-head or bimodal collapse hides in a mean |
| `density_ess_h0/h1` | within ~1.5 of each other | per-head split |
| `density_k_self` | ~0.057 | mass the row's own successor gets. This is the HEALTHY baseline at eps=0.4 — do not read 0.06 as coverage failure. It is batch-size dependent, and ~0.5 on cube-double because a 6-D conjunctive space separates points more at the same eps |
| `density_sarsa_subst` | ~1/traj_len (~0.001) | behavior mode only; the fraction of rows whose next action was substituted at a trajectory boundary. If this key is ABSENT, `next_actions` was not forced on and the bootstrap is wrong |

### 3.3 Zero-shot multi-reward evaluation

```bash
JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
  /dev/shm/.venv-jax/bin/python scripts/rescuing_fb/eval_density_zeroshot.py \
    --run-dir <run dir> --checkpoint-step 500000 --retrieval-k 1 \
    --episodes 25 --horizon 0 --seed 0 --pool_size 16384 \
    --out results/dzs.json
```

Use `--pool_size 16384`, not the 4096 default: the readout is under-converged at 4096
(per-task success drift 0.120 vs 16384) and converged by 16384 (drift 0.000 to 32768).
It OOM-kills at 65536 — the ridge fit for the SECONDARY readout scales badly. The
*ordering* of `r_W`, by contrast, is pool-size-insensitive.

The script reports two readouts and they are not interchangeable — PRIMARY is exact and
linear in the reward; SECONDARY is a ridge-fit compact reward code. Quote them
separately. Every output JSON records gamma, hit-spec, pool size and seed, the readout
formula and the decode mode, so no number can be read without its convention.

This eval is deterministic: an identical repeat is bit-identical. Use a repeat run as
the noise floor before attributing any difference to a model.

### 3.4 Calibration

```bash
# G4: model region masses vs Monte-Carlo ground truth (gate: Spearman >= 0.65)
JAX_PLATFORMS=cpu .../python scripts/rescuing_fb/eval_measure_g4_spearman.py --help

# ORS gate: r_W vs the dataset's own empirical discounted future distance
JAX_PLATFORMS=cpu .../python scripts/rescuing_fb/ors_calibration_probe.py \
  --run-dir <behavior-density run dir> --checkpoint-step 500000 --pool-size 4096
```

Gate at the pool size you SHIP. Report the per-multiplier breakdown next to any pooled
Spearman: box size drives both variables, so pooling reads higher than every column but
the largest (cube-single: 0.41-0.48 at the native eps, 0.83-0.85 at 4x).

---

## 4. Results on the record (cube-single, 500k, 3 seeds)

| arm | success |
|---|---|
| BC-only control (cloning alone) | 0.111 |
| delta (spec 17a) | 0.207 |
| ORS shaping (spec 19) | 0.739 |
| partition K256 (spec 17b) | 0.884 |
| **density (spec 18)** | **0.929** |
| first-passage no-ortho (anchor) | 0.967 |

Density is the best one-stage occupancy measure we have, and unlike the two-phase
curriculum it trains from scratch with no warm start and no phase switch. Zero-shot
multi-reward on the same checkpoints: dirac 0.67-0.89, dense 0.64-0.86, region 0.26-0.28
(the partition predecessor gets only 0.04-0.18 on the region family — removing the fixed
support pays off exactly where a finite prototype set has to interpolate).

**Open, and important**: density has only ever been trained on cube-single. The
environments that discriminate between methods are cube-double and scene, where the
two-phase numbers exist and density's do not.

## 5. Gotchas worth knowing before you change anything

- Any new knob must be provably load-bearing. Spec 17b once shipped a loss silently
  multiplied by a coefficient defaulting to `0.0`: it trained nothing while logging a
  plausible loss and a plausible entropy, which would have read as a scientific null.
- Never add a pytree data field to the agent. `restore_agent` runs
  `flax.serialization.from_state_dict` over the WHOLE agent, so a new leaf breaks restore
  for every pre-existing checkpoint — and it only surfaces at collection time. Use
  `nonpytree_field` (see `delta_protos`).
- Instrument the exact statistic a gate names. `density_ess` was logged as a mean while
  the gate specified a median.
- Hydra: `+agent_overrides.foo=` FAILS when the key already exists in the config; use
  `++`. Confirm a launch by success markers and exit status, never by the absence of
  error strings in a log.
