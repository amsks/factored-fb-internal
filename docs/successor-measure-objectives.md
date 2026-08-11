# Successor-measure objectives — code guide

Two successor measures live in `tools/wandb_mode_shim/gciql_flow.py`, and this is the
reading order for both — what the objective is, which symbols to read, how to train it,
and how to evaluate it.

| | method | stages | best numbers |
|---|---|---|---|
| §1-§5 | **density** (`measure_density`, spec 18) | one, from scratch | cube-single 0.929 |
| §6-§10 | **two-phase ball-visitation + hindsight** | two, warm-started | scene 0.927, cube-double 0.53, cube-single ~0.93 |

Density is the better single-stage method and the one with a normalized, structurally
valid parameterization. The two-phase variant is the better method *overall* today,
because it is the only one measured on all three environments. Both learn the same
underlying object — expected discounted time spent in a goal region — and differ in how
they parameterize and supervise it.

Specs (untracked, local only): `PAPER/Multistep-FB/specs/18-density-measure.md`,
`19a-behavior-density-impl.md`, `PAPER/rescuing-fb/specs/08-two-phase-next-experiments.md`.
Derivations: `PAPER/Multistep-FB/delta-measure.tex`. Ledger:
`PAPER/rescuing-fb/PHASE2-LEDGER.md`.

---

# The density successor measure

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

### 2.4 The two immediate-target variants: SOFT and HARD

The CE target is `t = sg[(1-gamma)*k + gamma*w_bar+]`. The **immediate term `k`** has two
forms, and which one you want is now settled:

| | `k` | knob |
|---|---|---|
| **soft** | the eps-kernel `_delta_phi(nobs, u_raw)` — the row's `(1-gamma)` mass is smeared over kernel-near queries | default (nothing set) |
| **hard** | a one-hot at the row's OWN successor — all the `(1-gamma)` mass on the observed next state | `measure_density_hard: true` |

**Use HARD.** Spec 21 seeded both to three seeds on cube-single and the gate read out:

| cube-single, 3 seeds | goal success | zero-shot dirac | dense | region |
|---|---|---|---|---|
| soft | 0.929 | 0.816 | 0.725 | 0.293 |
| **hard** | 0.915 (last-4 0.951) | **0.888** | **0.912** | **0.647** |

Hard more than doubles region-family transfer and wins both point-goal families, at no
goal-reaching cost — the 0.929/0.915 difference is inside seed noise, and on last-4-eval
means hard is actually ahead. The soft kernel was diluting the readout.

A soft/hard **mixture** knob (`measure_density_hard_mix`, alpha in [0,1]) was fully
specified in spec 21 §3 and then deliberately NOT implemented: the goal-success cost that
motivated it turned out to be a single noisy seed (one hard run's mid-training evals swung
1.00 -> .876). If you are tempted to build it, re-read that spec's §1 decision table first.

⚠ Every density number in the paper as of 2026-08-11 is the **soft** arm (cube-single
.929, cube-double .736, scene .628). Hard reruns of cube-double/scene are in flight.

### 2.5 Actor kinds

The density q-readout plugs into three extraction paths (spec 22), selected by
`actor_kind`, all sharing `_density_actor_q`:

- `flow` (default) — the FlowBC actor
- `ddpgbc` — stock gaussian DDPG+BC; needs `alpha` (OGBench's own value: 1.0 for
  cube-single/double/scene, 0.3 for antmaze) and `actor_q_center: true`
- `awr_flow` — advantage-weighted, baseline = min-head density q at a stop-gradded flow
  rollout

Use this when you need to show a result is not extraction-carried. On cube-single,
swapping flow -> ddpgbc costs density .05 (.929 -> .879) while stock GCIQL loses .31
(.870 -> .563).

⚠ `ddpgbc`/`awr_flow` must never delegate to `GCIQLAgent.actor_loss` under `measure_only`:
the bilinear critic is dead weight there, so delegation silently maximizes a never-trained
critic while the density CE trains beside it. This is tested
(`tests/test_gciql_density_actors.py`) — do not "simplify" those branches back into a
delegation.

### 2.6 Configs

| config | what |
|---|---|
| `{cube_single,cube_double,scene}_state_measure_density_d50.yaml` | soft, flow actor |
| `{cube_single,cube_double,scene}_state_measure_density_hard_d50.yaml` | **hard**, flow actor |
| `{cube_single,cube_double,scene}_state_measure_density_d50_ddpgbc.yaml` | soft, DDPG+BC |
| `{cube_double,scene}_state_measure_density_hard_d50_ddpgbc.yaml` | hard, DDPG+BC |
| `cube_single_state_measure_density_d50_awrflow.yaml` | soft, AWR-flow |
| `antmaze_medium_state_measure_density_d50{,_ddpgbc,_awrflow}.yaml` | locomotion, hit-spec `0:2:0.5` |
| `point_mass_maze_rnd_seed0_measure_density_d50.yaml` | ExoRL/DMC (spec 23 P0) |
| `{cube_single,cube_double,scene}_state_measure_density_behavior_d50.yaml` | behavior mode (19a), for ORS |
| `{cube_single,cube_double}_state_bconly_d50.yaml` | BC-only floor: same recipe, actor q-term zeroed |

The BC-only floors are what make any of these numbers attributable: cube-single 0.111,
cube-double 0.019.

---

## 3. How to run it

### 3.1 Train

```bash
# SOFT (the historical default)
XLA_FLAGS=--xla_gpu_enable_command_buffer= \
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python run_gciql.py --config-name cube_single_state_measure_density_d50 \
    seed=0 output_root=/home/adityamohan/exp_density exp_name=dens_s0

# HARD (what you should actually run -- see 2.4)
XLA_FLAGS=--xla_gpu_enable_command_buffer= \
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python run_gciql.py --config-name cube_double_state_measure_density_hard_d50 \
    seed=0 output_root=/home/adityamohan/exp_density exp_name=hard_s0

# HARD + a stock actor (actor-independence check)
XLA_FLAGS=--xla_gpu_enable_command_buffer= \
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python run_gciql.py --config-name scene_state_measure_density_hard_d50_ddpgbc \
    seed=0 output_root=/home/adityamohan/exp_density exp_name=hard_ddpgbc_s0
```

Multi-cell waves: `scripts/rescuing_fb/launch_hard_xenv.sh` (hard on cube-double/scene),
`launch_density_ddpgbc_xenv.sh` (DDPG+BC on both, soft and hard), `launch_spec22.sh`
(antmaze + actor ablation). Two hazards baked into their headers, both learned the hard
way: every launcher's `active()` counts EVERY `main.py` job so waves share the box without
oversubscribing, and **do not kill a dispatcher to change `MAX`** — it is the pipeline
parent of its children and the signal propagates (that killed two cells). One launcher
also short-circuits `DRY_RUN` before the concurrency wait; the older ones block forever on
a full box instead of printing the plan.

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

---

# The two-phase ball-visitation measure — the other successor measure

The density objective above is one stage. The older and, across environments, still the
best-performing successor measure is the **two-phase curriculum**: pretrain a
first-passage measure, then warm-start a ball-visitation occupancy measure from it. This
section is the same kind of guide for that method.

Specs: `PAPER/rescuing-fb/specs/08-two-phase-next-experiments.md`; ledger
`PAPER/rescuing-fb/PHASE2-LEDGER.md` (the §9.16 sections are the mechanism work).

## 6. What the objective is

Two different objects, learned in sequence over the same bilinear critic
`M(s,a,g) = phi(s,a)^T psi(g) / sqrt(d)`.

**Phase 1 — first-passage (absorbing).** How soon do you FIRST enter the goal region:

```
diff = M - hit - gamma * (1 - hit) * sg[tM]
```

The `(1 - hit)` factor is the absorption: once you land in the region the recursion
stops, so the fixed point is bounded in `[0,1]` and behaves like `E[gamma^T]` for the
first arrival time `T`. This is NOT occupancy — it cannot answer "how much time do I
spend there", so it does not give you a zero-shot reward readout on its own. It is
trained to provide the representation (in particular the backward map `psi`) that phase 2
starts from.

**Phase 2 — ball-visitation (non-absorbing).** Drop the absorption and you get genuine
occupancy, revisits counted:

```
diff = M - hit - gamma * sg[tM]          # M = E[ sum_t gamma^t * 1[x_{t+1} in region] ]
```

That IS the successor measure — the same object the density head represents, but as an
unbounded scalar per `(s,a,g)` rather than a normalized distribution. Fixed point up to
`1/(1-gamma)`.

Phase 2 does not train from scratch, which is the entire reason the curriculum exists.

**The columns.** The measure is fit as a `[B,B]` matrix: rows are batch transitions,
columns are test points `g_j`, and `hit[i,j] = 1[obj(s'_i) is within eps of g_j]`. Which
states serve as columns is the `measure_testpoint_dist` knob and it turns out to be the
load-bearing choice:

- `batch` — columns are the batch's own next states. On conjunctive environments the
  off-diagonal hit rate is ~0.046%, the targets collapse to the action-independent dwell
  constant `E[h]/(1-gamma)`, and the actor decays.
- `hindsight` — columns are `value_goals` (trajectory futures). Grounded hit events
  appear at a rate independent of the region's sparsity. This is the only arm that works
  on multi-object environments.

## 7. Files to read, in order

Same file, `tools/wandb_mode_shim/gciql_flow.py`.

| what | symbol | ~line |
|---|---|---|
| hit-spec grammar (blocks + per-block eps) | `_parse_hit_spec` | 35 |
| the `[B,B]` arrival indicator | `_hit_matrix` | 1384 |
| **the TD residual reduction** (absorbing vs not, scale_norm) | `_measure_terms` | 1488 |
| **the objective itself** (columns, bootstrap, ortho, hindsight branch) | `measure_loss` | 1574 |
| the hindsight/batch column switch | `tpd = ...` inside `measure_loss` | 1674 |
| knobs | `measure_first_passage` / `measure_fp_absorbing` / `measure_stopgrad_psi` / `measure_testpoint_dist` | 3994 / 3996 / 3991 / 4078 |
| warm-start restore | `warm_start_ckpt` | 4128 |

`_measure_terms`'s docstring is the best single explanation of the residual, including
why `measure_scale_norm` exists (it makes the gradient dimensionless so the unbounded
`<F,z>` cannot blow up the shared critic).

Reading order that works: `_parse_hit_spec` -> `_hit_matrix` -> `_measure_terms` ->
`measure_loss`.

## 8. How to run it

Two stages. Phase 1 first:

```bash
XLA_FLAGS=--xla_gpu_enable_command_buffer= XLA_PYTHON_CLIENT_PREALLOCATE=false \
CUDA_VISIBLE_DEVICES=0 .venv/bin/python run_gciql.py \
  --config-name cube_double_state_measure_firstpassage_noortho_d50 \
  seed=0 output_root=$OUT exp_name=cd_fp_s0
```

Then phase 2, warm-started from that run directory:

```bash
XLA_FLAGS=--xla_gpu_enable_command_buffer= XLA_PYTHON_CLIENT_PREALLOCATE=false \
CUDA_VISIBLE_DEVICES=1 .venv/bin/python run_gciql.py \
  --config-name cube_double_state_measure_ballvisit_hs_d50 seed=0 \
  ++agent_overrides.warm_start_ckpt=$OUT/factored-fb/<phase1 group>/cd_fp_s0 \
  ++agent_overrides.warm_start_epoch=500000 \
  output_root=$OUT exp_name=cd_p2h_s0
```

`scripts/rescuing_fb/launch_p2fix_arms.sh` is the worked example of the whole pattern
(it also demonstrates the pre-flight check that the phase-1 checkpoint exists before
launching, which is worth keeping).

Configs:

| stage | config |
|---|---|
| phase 1 | `configs/gciql/{cube_single,cube_double,scene}_state_measure_firstpassage_noortho_d50.yaml` |
| phase 2, hindsight (the good arm) | `configs/gciql/{cube_double,scene}_state_measure_ballvisit_hs_d50.yaml` |
| phase 2, batch columns (the arm that collapses) | `configs/gciql/*_state_measure_ballvisit_d50.yaml` |

The phase-2 recipe is: `measure_fp_absorbing: false` (drop absorption),
`measure_stopgrad_psi: true` (freeze the backward map carried from phase 1),
`actor_q_center: true` (the occupancy carries a large shared baseline that otherwise
dilutes the greedy signal), `measure_ortho_coef: 0.0`, plus the warm start.

## 9. Results on the record

| arm | cube-single | cube-double | scene |
|---|---|---|---|
| phase 1 (first-passage, no ortho) | 0.967 | 0.549 | 0.484 |
| phase 2, hindsight columns | ~0.93 | 0.53 | **0.927** |
| phase 2, batch columns | 0.952 | 0.358 | 0.342 |

Phase 2 with batch columns REGRESSES against its own phase-1 start on both multi-object
environments. Hindsight columns rescue scene outright (0.484 -> 0.927) and roughly hold
on cube-double. On the zero-shot axis the spec-14 harness found zero-shot tracking native
goal-conditioned success within 0.06 on all six multi-env checkpoints, with scene
hindsight at 0.872 — the best zero-shot number in the project.

This is why the two-phase + hindsight variant is still the best method *overall*: it is
the only successor measure with strong numbers on all three environments. Density beats
it on cube-single (0.929 one-stage vs ~0.93 two-stage) but has never been run on
cube-double or scene.

## 10. Two-phase gotchas

- **Orthonormalization is toxic here.** With ortho on, cube-single first-passage goes
  0.967 -> 0.225. Every measure config in this family sets `measure_ortho_coef: 0.0`.
- **The FrozenPsi trap.** The `*_ballvisit_*_d50.yaml` configs are the phase-2 recipe and
  set `measure_stopgrad_psi: true`, which is correct only WITH a warm start. Copying one
  of them for a from-scratch run freezes `psi` at random init and floors the cell for a
  reason that has nothing to do with the hypothesis being tested.
- **Batch columns are dead on conjunctive environments.** Freezing the actor holds the
  parent number, but unfreezing decays to the control anyway (ledger §9.16j/k) — the
  batch-column fixed point itself damages the actor. Do not spend runs re-testing this.
- **Never add pessimism to a measure cell.** Min-aggregation took cube-single 0.42 ->
  0.240 (T7).
- **Success is blind to measure validity.** A cell can score well on goal-reaching while
  its region-mass ordering is poor, and vice versa. Quote the calibration read
  (`eval_measure_g4_spearman.py`) alongside any success number.

---

# Shared gotchas — read before changing either method

## 11. Comparing methods across the two eval harnesses

The two objectives are read out by different scripts — `eval_zeroshot_rewards.py` for the
bilinear/z path (normal FB, GCIQL, both two-phase stages) and `eval_density_zeroshot.py`
for the density head. Before putting their numbers in one table, know this:

**The `region` family is not constructed the same way on every env.**
`eval_zeroshot_rewards.py` draws hindsight centers only when the env is multi-object
(`if "region" in families and multi`, ~line 1168); on cube-single it falls back to legacy
FIXED probe centers `R1/R2/R3` from `env_geometry`. `eval_density_zeroshot.py` always
draws hindsight centers. Consequences:

| column | cube-single | cube-double / scene |
|---|---|---|
| `dirac` (the env's 5 task goals) | comparable | comparable |
| `region` | **NOT comparable** — 3 fixed centers vs 4 hindsight-drawn | comparable (both hindsight) |
| `dense` | density harness only | density harness only |

A concrete near-miss: on cube-single, two-phase phase-2 scores region 0.688 and density
scores 0.27. That reads as a 2.5x gap and is not a result — they were scored on different
reward sets. The dirac column on the same checkpoints (0.875 vs 0.77) is the comparable
one, and it overlaps within per-seed spread.

If you need a cube-single region comparison, put both methods on the same centers first
(either teach the density harness the fixed probe centers, or let the zsr harness take
the hindsight path on single-object envs) — do not compare the two as they stand.

Also check before comparing: `--episodes`, `--seed`, `--relabel-size`, and the decode
(`--z-inference retrieval --retrieval-k 1` vs `--decode retrieval --retrieval-k 1`) must
match, and the JSON `meta` block of every run records all of them for exactly this
reason.

## 12. Lessons that cost real time

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
