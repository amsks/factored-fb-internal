# ExoRL and HumEnv: what ran, and what it shows

State as of 2026-08-25 17:00 CDT, branch `density-fb`. Every number here was recomputed from
the run directories on the date of writing, not copied from a prior table. Paths are given so
each one can be re-derived.

## How to read these tables

**T3 and T4 are different evaluations. Never read one against the other.** T3 withholds the
environment's goal and infers the task from reward samples (`eval.csv` carries `inferred_cond*`
columns); T4 evaluates through the goal frame, which hands the policy the target. T4 is the
easier task by construction.

`scripts/exorl_report.py` **defaults to `--steps 500000`**. On a 1M-step root that silently
reports the 500k row. Pass `--steps` explicitly.

All ExoRL data is the RND exploration buffer, 10k episodes per domain, under
`$SCRATCH/exorl_data/exorl/datasets/{point_mass_maze,walker,cheetah}/rnd/buffer`. Jobs need
`EXORL_DATA_ROOT` and `EXORL_CACHE_ROOT` set or they look in `$HOME` and die after allocation.

---

## 1. ExoRL — the four arms

All offline, 3 seeds, `actor.alpha=0`. `density_fb` runs `ddpgbc_density`; `fb`/`psm`/`rldp`
run stock `ddpgbc`. The actor choice is not interchangeable — `critic/density_fb.yaml` sets
`actor_q_center=true`, and stock `ddpgbc` divides by the uncentered leg and throws the centered
one away.

| arm | root under `$SCRATCH/` | steps | what it adds |
|---|---|---|---|
| **T3** | `factored-fb-exorl/t3` | 500k | our shapes, our defaults; 4 methods × 3 domains |
| **T4** | `factored-fb-exorl/t4` | 500k | online, goal-frame eval |
| **paper-HP** | `factored-fb-exorl-paper` | 1M | Touati Table 1 optimisation knobs: batch 1024, discount 0.98, tau 0.01, smoothing 0.2/0.3, lr_actor 1e-4 |
| **arch** | `factored-fb-exorl-arch` | 1M | + reference widths: F 1024×1, actor 1024×1, z_dim 100 on walker |
| **arch-shape** | `factored-fb-exorl-arch-shape` | 1M | + B `[256,256]`, the real Y structure (`measure_density_archi=motivo`), `b_norm`, and lr 3e-4 → 1e-4 |

Reference points: Touati et al. report walker walk 760 ± 19 and cheetah run 247 ± 9 on this
buffer. The maze random-policy floor is 18.9 (`scripts/launch_exorl.sh:58`); the RND behaviour
policy's own cheetah return is 36.5 (`scripts/launch_exorl_arch.sh:11`). **The walker and
cheetah random floors are not recorded in any file in this repo** — they should be.

### T3, 500k, zero-shot reward route

Return, mean ± 95% CI over 3 seeds.

| domain | density_fb | fb | psm | rldp |
|---|---|---|---|---|
| point-mass-maze | **751.7 ± 11.3** (succ 1.000) | 144.6 ± 176.0 | 0.34 ± 0.48 | 129.7 ± 53.4 |
| walker | **495.9 ± 80.1** | 26.8 ± 4.9 | 389.6 ± 329.6 | 198.3 ± 237.9 |
| cheetah | 33.9 ± 39.8 | 63.8 ± 70.2 | 14.7 ± 27.6 | 56.0 ± 46.5 |

`density_fb` is the only method with no dead seed anywhere. Every other method has one: psm
scores 0.34 on maze, fb sits below the random floor on all three walker seeds. Cheetah
discriminates nothing at 500k under any actor — every method's interval covers every other's.

Two further T3 findings from the same root:

* **Centered vs uncentered normalizer is a variance effect, not a mean shift.** Maze
  `density_fb` scores 751.7 ± 11.3 with `ddpgbc_density` and 681.7 ± 154.0 with stock `ddpgbc`
  — the same ~760 ceiling, but the uncentered arm has a 524.5 seed and the centered one does
  not.
* **`flowbc` is the wrong actor here.** Its BC term clones the RND explorer. Every `flowbc`
  cell on maze (7.1 / 12.4 / 15.1 / 11.4) sits at or below the 18.9 random floor.

### The 1M arms, walker and cheetah

| cell | T3 @500k | paper-HP @1M | arch @1M | arch-shape @1M | reference |
|---|---|---|---|---|---|
| walker density_fb | 495.9 ± 80.1 | 627.4 ± 67.6 | 123.0 ± 23.7 | **756.1 ± 45.6** | 760 ± 19 |
| walker fb | 26.8 ± 4.9 | 67.3 ± 50.0 | 259.9 ± 158.2 | 309.1 ± 189.7 | 760 ± 19 |
| cheetah density_fb | 33.9 ± 39.8 | 126.6 ± 26.3 | 54.0 ± 22.4 | 111.7 ± 30.8 | 247 ± 9 |
| cheetah fb | 63.8 ± 70.2 | 53.9 ± 26.0 | 75.0 ± 73.5 | 21.7 ± 36.6 | 247 ± 9 |

Per-seed returns for the arch-shape column: walker density_fb 716.2 / 755.5 / 796.7; cheetah
density_fb 136.1 / 82.4 / 116.6; walker fb 147.9 / 482.5 / 297.0; cheetah fb 2.9 / 3.0 / 59.1.

### What we are seeing

**Walker replicates. Cheetah does not.** `arch-shape` walker `density_fb` at 756.1 ± 45.6 is
the first cell on this branch that lands on a published ExoRL number (760 ± 19), with a tight
interval and no dead seed. Cheetah on the same arm reaches 111.7, still under half of 247 ± 9.

**The arch → arch-shape jump moved four variables at once**, so the walker gain cannot be
attributed to the Y structure alone. Confirmed from the two `config.json` files: B hidden dims
`null` → `[256,256]`, `measure_density_archi` `null` → `motivo`, `measure_density_b_norm`
`null` → `true`, and `lr` 3e-4 → 1e-4. Separating these is one run each.

**500k understates badly; the curves are still climbing.** Paper-HP cheetah `density_fb` goes
27.5 ± 41.2 at 500k → 126.6 ± 26.3 at 1M; arch-shape walker goes 728.8 ± 84.6 → 756.1 ± 45.6
and arch-shape cheetah 78.5 ± 38.3 → 111.7 ± 30.8. Any conclusion drawn from a 500k cell about
cheetah is unsafe.

**Our `fb` baseline replicates nothing and is unstable everywhere.** Its best cell is
309.1 ± 189.7 (arch-shape walker) against a 760 reference, and it carries a collapsed seed in
almost every cell — 2.9 and 3.0 on arch-shape cheetah, 98.5 on arch walker, 0.0 on T3 maze.
The intervals are wider than the gaps between arms, so the apparent arch gain for `fb` is not
usable. Since `fb` is the reference method here, the shortfall is a pipeline property, not a
`density_fb` property.

**T4 online: the offline conclusion does not transfer.** Removing the BC term, which is correct
offline, collapses online. Maze goes 414.8 ± 136.8 (`flowbc`) → 0.07 (`alpha=0`); cheetah
4.6 → 0.81; walker 21.7 → 27.9. Online has no fixed dataset, so the BC term is the only thing
anchoring the policy to its own buffer, and `alpha=0` lets an untrained critic drive collection
from step one. Neither setting is right — a small non-zero alpha is untested. Online also has
no exploration algorithm at all (`env/online.py` is policy plus isotropic action noise;
`online_explore_goal_prob` defaults to 0.0), which is why locomotion collapses online under any
actor while the maze works.

---

## 2. HumEnv — the humanoid goal evaluation

### Setup

The SMPL humanoid from HumEnv / Meta Motivo, trained **offline** on the released FB-CPR replay
buffer (`facebook/metamotivo-S-1`, 16,650 episodes / 5,000,000 transitions), converted by
`scripts/convert_humenv_buffer.py`. Observations are 358-dim, actions 69-dim — 9× and 14× the
cube stack.

The metric is `humenv.bench.GoalEvaluation`, run post-hoc over checkpoints by
`scripts/humenv_goal_eval.py`, on 12 episodes (4 goals × 3 rollouts). The hit kernel uses the
benchmark's own pose block, `measure_hit_spec="0:214:2.548"`: `0:214` is the block
`humenv/bench/utils/metrics.py::distance_smpl` slices, and 2.548 is calibrated by
`scripts/calibrate_hit_spec.py` so the random-pair hit rate matches cube-single's own (0.0133
either way).

**The FB-CPR checkpoint is an oracle, not a baseline.** It is online-trained on 30M interactive
steps with AMASS regularisation. Comparing an offline method against it asks a distillation
question, not a win/loss one.

### Arms and results

All under `$SCRATCH/factored-fb-humenv/`, critic `density_fb`, actor `ddpgbc_density`.

| arm | steps | seeds | success | proximity | distance |
|---|---|---|---|---|---|
| `d128` (defaults, d=128) | 500k | 1 evaluated | 0.000 | 0.00005 | 9.41 |
| `motivo` (Motivo HPs) | 1M | 3 | 0.000 | 0.0017 | 9.93 |
| `motivo-shape` (+ Y structure) | 1M | 3 | 0.028 | 0.0035 | 10.34 |
| *FB-CPR checkpoint (oracle)* | *—* | *—* | *0.750* | *0.818* | *2.18* |

Per-seed success on `motivo-shape`: 0.000 / 0.000 / 0.083 — the single non-zero comes from one
goal (`episode_14042_300`) on seed 2, 1 of 3 rollouts. Files:
`goal_eval.jsonl`, `goal_eval_motivo.jsonl`, `goal_eval_motivo_shape.jsonl`.

The `motivo` arm's overrides, from its own `config.json`: `discount=0.98 tau=0.01
batch_size=1024 lr=1e-4 lr_f=1e-4 lr_b=1e-5 lr_actor=1e-4 value_hidden_dims=[1024,1024]
measure_density_b_hidden_dims=[256] measure_density_dim=256 norm_obs=true
measure_density_z_rand_frac=0.8 actor.hidden_dims=[1024,1024] actor.const_std=true
actor.boot_noise_std=0.2 actor.boot_noise_clip=0.3`. `motivo-shape` adds
`measure_density_archi=motivo actor.motivo_archi=true measure_density_b_norm=true`.

Deliberate deviations from Motivo, to be stated wherever a number from this arm is reported:
1M updates rather than their ~3M; γ=0.98; offline on the frozen buffer rather than 30M online
steps with AMASS regularisation; and everything FB-CPR-objective-side is gone by definition
(ortho_coef, reg_coeff, q_loss_coef, pessimism penalties, expert_asm_ratio, relabel_ratio,
seq_length 8, the discriminator).

### What we are seeing

**Flat.** Distance is ~10 across every arm against the oracle's 2.18, and it does not improve
from 500k to 1M, from d=128 to d=256, or with Motivo's hyperparameters. Proximity moves from
5e-5 to 3.5e-3 — three orders below the oracle's 0.818, and small enough to be noise at 12
episodes.

**The architecture fix that rescued ExoRL walker does nothing here.** The same
`measure_density_archi=motivo` change that coincides with walker going 123 → 756 moves humanoid
success from 0.000 to 0.028 and distance the wrong way, 9.93 → 10.34.

**Nothing has been isolated yet.** Two knobs had to be added for the parity arm at all: `_group_lr`
in `impls/agents/__init__.py`, because `lr_f`/`lr_b` were previously merged in doing nothing;
and `norm_obs` in `impls/configs/critic/density_fb.yaml`. Whether the flat result is a scale
problem, a conditioning problem, or a residual porting problem is not yet distinguished by any
run.

---

## 3. Open, and not done

* **The per-task zero-shot sweep never ran.** Training is reward-free — `rewards` is read only
  inside `infer_cond` — so one checkpoint serves every reward task of a domain.
  `scripts/reeval_checkpoint.py` has `--infer_cond_episodes`, `--eval_on_inferred_cond` and
  `--override exorl_task=<t>` for exactly this. It is verified on one checkpoint
  (`t3_zeroshot.jsonl`: a walker `density_fb`/`flowbc` seed-0 checkpoint trained under `walk`
  scores walk/run/stand/flip = 203.2 / 87.9 / 385.0 / 236.7 at 10 episodes each) and has never
  been swept across the 36 T3 runs. Every cell above uses only `reach_top_left` / `walk` / `run`.
* **What separates `fb`'s 309 maze seed from its 0.02 seed is unexplained.** Q inflation does
  not account for it. All three fb maze seeds inflate `q_mean` to 1396.5 / 1273.6 / 1231.7 at
  500k while returning 0.0 / 309.2 / 124.5 — the inflation is identical across the good and the
  dead seed. And psm's two `q_mean` ≈ 96 seeds fail at 0.0 and 0.2 return, so a low `q_mean`
  does not protect either. Values read from each run's `train.csv`.
* **The arch-shape four-variable change is unseparated** (see above).
* **`--infer_cond_episodes` is statistically weak on point-mass-maze.** Only ~0.08% of RND
  states fall inside `reach_top_left`'s full-credit radius, so a uniform M=256 draw contains a
  rewarded state ~18% of the time; the rest of the time the index is the neutral fallback, and
  `infer_cond_noop_reason` does not warn because it only catches exactly zero.
* **Nothing from 2026-08-24 or 08-25 is committed.** The launchers that carry the reasoning for
  every arm above — `scripts/launch_exorl.sh`, `launch_exorl_paper.sh`, `launch_exorl_arch.sh`,
  `launch_humenv.sh`, `launch_humenv_motivo.sh`, `scripts/exorl_report.py`, plus the whole
  HumEnv env and eval path — are untracked.
