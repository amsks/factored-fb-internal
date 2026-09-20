# ExoRL audit: our FB and Density-FB vs the published numbers (2026-09-17)

Four read-only sub-audits, full reports in `docs/notes/2026-09-17-exorl-audit/`:
`audit_fb_impl.md` (FB code and hyperparameters vs Touati's `controllable_agent` and td_jepa's
metamotivo FB), `audit_dfb_exorl.md` (the 2M / 5-seed Density-FB runs), `audit_exorl_pipeline.md`
(dataset, relabelling, task inference, eval vs the official protocol), `audit_published_numbers.md`
(exact tables from Touati 2023, HILP, RLDP, TD-JEPA).

## Where we stand (paper-protocol scorer, 10 episodes, 50k reward samples, final checkpoint)

| domain mean | ours Density-FB 2M (5 seeds) | ours FB reference cfg 1M (3 seeds) | Touati FB | TD-JEPA's FB | RLDP-paper FB |
|---|---|---|---|---|---|
| walker (stand/walk/run/4th) | 634 | 777 | 654 (flip) | 811 (spin) | 595 (flip) |
| cheetah | 232 | not run | 513 | 673 | 586 |
| quadruped | 569 | not run | 606 | 596 | 569 |
| point-mass-maze (4 reach) | 302 | not run | goal-given, n/a | 8 tasks, 6x margin, n/a | 427 |

Walker 4th task: ours and TD-JEPA use spin, Touati/HILP/RLDP use flip. TD-JEPA's maze uses a 6x
success margin and 8 tasks, so its maze column is not comparable to anyone's. Published protocols
differ from ours in steps (Touati 1M, HILP 1M, TD-JEPA 2-3M, RLDP 5M), seeds, and inference
samples (5k-10k vs our 50k).

## 1. The pipeline is not the problem
Dataset (first 5000 RND episodes, same as all references), transition alignment, per-task reward
relabelling (byte-identical task definitions; stored reward equals our relabelled reward to 1e-3),
task inference (uses the relabelled reward of the scored task, never the constant fallback), and
eval (10 episodes, mean action, final checkpoint) all match the official code. Minor deviations:
5% validation hold-out, fixed reset seeds, 50k instead of 10k inference samples. None depresses
returns.

## 2. FB: the implementation is a faithful port; the DEFAULT config is what collapses
- Loss, targets, stop-gradients, z mixing, sphere projection and the reward-inference estimator
  match td_jepa term for term (`impls/critics/fb.py:133-152` vs `agents/fb/agent.py:246-271`).
- At the reference configuration (`scripts/launch_walker_fbref.sh`) walker is AT OR ABOVE published
  FB at 1M: 920 / 804 / 400 / 985.
- Our default `fb.yaml` on walker: 357 / 49 / 16 / 853 (500k). At the same 500k budget the
  reference config already reads 588 on walk vs 36 for the default, so this is configuration, not
  budget. Ranked culprits: the 512x4 left encoder to a 50-d sphere in front of F (reference: a 256-d
  linear projection), batch 256 vs 1024, target rate 0.005 vs 0.001, tower shapes. `fb_tdj` adds
  ortho 1000 (reference sweep 0.1-10) and drops further.
- Remaining deviations of the reference-config run from td_jepa: 1M vs 3M steps, actor lr 3e-4 vs
  1e-4, actor reads raw obs instead of the encoder output, an extra LN+tanh+Dense in the 0-layer
  encoder (`impls/utils/networks.py:66-74`), hard clip instead of a straight-through clamp on the
  noisy actor action (`impls/actors/ddpgbc.py:52,91-94`). No wrong sign/target/normalisation found.
- Consequence: every FB baseline row on ExoRL must come from the reference config. Only walker has
  it; cheetah, quadruped and maze FB-reference runs do not exist.

## 3. Density-FB: three mechanisms, none of them noise
Inference is stable (independent 50k draws agree to cosine >= 0.993 in every cell tested), the
logs are clean, and walker is flat from 250k in all five seeds. The gaps come from training.

1. **The training-task reward is nearly flat on the locomotion domains.** The hit radius was set
   so 1.3% of random state pairs fall inside it, on the full observation block. In 17-78
   dimensions that radius is only 1.5-3.4x below the median pairwise distance (velocity dims
   carry >95% of it), so the Gaussian bump the task encoder is fit to spreads over 277/1024
   candidates on walker, 205 on cheetah and 879 on quadruped, versus 60 on cube-single. Half the
   actor/encoder rows carry almost no task signal; the sphere-draw rows do the work. The critic
   itself is unaffected (one-hot immediate term).
2. **The actor maximises a linear proxy of the reward in the 50-d sphere-normalised feature
   basis.** That proxy explains 73-80% of walker stand/run reward and 14-26% of the maze reward
   in the learned basis; cheetah's speed-thresholded rewards are only 44-47% linear even in raw
   observations. Domain ranking of our returns follows this: quadruped (rewards linear in obs,
   R2 >= 0.996) at published level, walker 10-50% low, cheetah at a quarter. The exact-reward
   "kernel" actor readout used on cube was never run on ExoRL (`latent_centered` in every config).
3. **Maze seed 3 is a collapsed TD3 actor** (constant action (-1,-1) for every state and task
   vector from the first checkpoint; BC weight 0). Other maze seeds are bang-bang controllers
   whose reach for a given z decides a cell (0 / ~300 / ~650) deterministically.
4. **The 2M "FB reference recipe" bought nothing.** The 1M runs (tau 0.005, [512,512,512] value
   tower) scored higher on walker (820/699/244 vs 715/641/204). Target rate 0.001, the [1024]
   tower, sphere fraction 0.5, dim 50, goal mix and the proxy reward were copied from FB's audit
   or from cube with no ExoRL ablation. Actor Q shrinks monotonically after 250k while the measure
   loss is flat.

## What this licenses
- FB baseline rows: reference config only; run cheetah/quadruped/maze at that config.
- Density-FB on ExoRL: (a) port the exact-reward actor readout; (b) set the hit kernel by its
  concentration over the batch (target ~60/1024 as on cube) or restrict it to position/posture
  dims, not by random-pair hit rate on the raw full-obs block; (c) ablate the sphere fraction and
  BC weight (maze collapse); (d) revert to the 1M / tau 0.005 recipe as the base.

## Fix applied (2026-09-17): radius re-calibrated by training-reward support, Phase 3 queued
User decision: radius only (actor reward stays the linear proxy; sphere fraction, target rate,
tower, discount unchanged), so the comparison against the 2M cells has one lever.

`scripts/calibrate_hit_spec.py` gained a `--target_support` mode: bisect the radius until the
actor's training-task reward (the Gaussian bump around a random goal, softmax-normalised over a
batch of 1024 distinct candidate next-states) has a given effective support (exp of its entropy).
Tests: `tests/scripts/test_calibrate_hit_spec.py`. The target is cube-single's own support under
its published spec, measured on `~/.ogbench/data/cube-single-play-v0.npz`: **85.6 / 1024**.

| domain | block | old radius (hit-rate) | support at old | new radius (support 85.6) | median pair dist / new radius |
|---|---|---|---|---|---|
| walker | 0:24 | 15.9 | 325 | 9.5603 | 5.3 |
| cheetah | 0:17 | 7.5 | 299 | 4.6159 | 5.4 |
| quadruped | 0:78 | 21.087 | 921 | 6.2467 | 5.1 |
| maze | 0:2 | 0.03 | 64.5 | unchanged | (cube: 6.1) |

Phase 3 = 20 jobs (4 domains x 5 seeds), 1M steps, appended to `logs/exorl_dfb/manifest.tsv` by
`logs/exorl_dfb/gen_phase3.sh`; the running `dfbsched` scheduler picks them up when Phase 2
(OGBench top-up) finishes. Save dir `exp/exorl_dfb_sup86/<domain>/density_fb_z`, logs
`logs/exorl_dfb/<domain>_dfbz_sup86_s<k>.log`. Smoke probe (quadruped, 300 steps) ran clean with
the quoted spec recorded in config.json.

Scoring when done: `scripts/exorl_paper_eval.py` at 50k inference samples on `params_1000000.pkl`
-> `exp/paper_eval_exorl_sup86_1M.jsonl`; compare with the 2M runs' own 1M in-run eval rows and
the 2M paper-eval rows in `exp/paper_eval_exorl_dfb_2M.jsonl`.

## Phase 3 RESULT (2026-09-17 16:40Z): support-calibrated radius is a NEGATIVE
Paper protocol (10 episodes, 50k reward samples, final checkpoint), mean over 5 seeds of the
per-task return. Rows: `exp/paper_eval_exorl_sup86_1M.jsonl` (new) vs `exp/paper_eval_exorl_dfb_2M.jsonl`.

| domain mean | new radius, 1M | old radius, 2M |
|---|---|---|
| walker | 553 | 634 |
| cheetah | 212 | 232 |
| quadruped | **140** | 569 |
| maze | 273 | 302 |
| all | 294 | 434 |

Quadruped collapsed on every task and every seed (stand 177 vs 795, jump 135 vs 609); walker lost
~100 on stand/walk/spin; cheetah and maze unchanged within noise. Budget is not the confound:
walker/cheetah were flat from 250k in the 2M runs, and the 2M runs' own 1M in-run quadruped-walk
evals (226-496) already sit far above the new runs' (90-367).

Reading: sharpening the goal bump on the full-observation block makes the encoded training tasks
"reach this exact velocity configuration", which is further from the velocity/posture rewards the
eval asks for, not closer; and the radius also divides the backward tower's input
(`_density_queries`, impls/critics/density_fb.py:320-325), so quadruped's B saw inputs 3.4x
larger. The audit's finding 1 (flat bump) is therefore NOT the lever. The remaining candidate
from the audit is finding 2: the actor climbs a linear proxy of the reward in the 50-d feature
basis, which explains under half of cheetah's reward and a quarter of maze's. Runs kept under
`exp/exorl_dfb_sup86/` for reference; do not quote them as Density-FB numbers.

## Cheetah / maze: kernel or hyperparameters? (2026-09-17 23:00Z, reference-implementation check)
Question: the two remaining domains (cheetah 232 vs published FB 586-673; maze 302 vs published FB 427).
Both are our Density-FB against PUBLISHED numbers; nothing else in the table is ours.

What the references do per domain (td_jepa `launch_fb_dmc.py:107-114`, RLDP `rldp_train_dmc.py:57-85`,
HILP README, Touati `fb_ddpg.py`):
- cheetah: NOTHING domain-specific in any reference. Same lr 1e-4, discount 0.98, tau, batch as walker.
  So cheetah is not a "missing per-domain hyperparameter"; it is either our pipeline on cheetah or the method.
- point-mass: EVERY reference that runs it changes the actor lr: td_jepa FB `lr_actor 1e-6` + discount 0.99;
  RLDP-paper's FB baseline `lr_actor = lr_b = 1e-6` + discount 0.99 (Touati: z_dim 100, discount 0.99, goal-set z).
  Our maze Density-FB ran `lr_actor 3e-4` (300x), and the audit's maze failure is exactly a TD3 actor
  saturated at the action bound (seed 3 constant (-1,-1), seed 4 99.9% |a|>0.99 bang-bang). Hyperparameter.
- The kernel is NOT the maze lever: maze uses the benchmark's own 0.03 ball, bump concentration 51/1024
  (cube-like), and the sup86 refit left maze unchanged (273 vs 302). On cheetah the sharper radius also
  changed nothing (212 vs 232), so bump sharpness is not the cheetah lever either.

Pipeline evidence on cheetah is MISSING: no method has been run on cheetah in our pipeline at a config
known to work. The only other cheetah runs (`exp/exorl-official{,-r2}/dmc-cheetah/rldp_official`, 4 seeds, 5M)
are the INERT-ortho RLDP (`rldp_ortho_inert: true`) and score run 14 / walk 65 / run_back 4 / walk_back 17
(`exp/paper_eval_r2.jsonl`) -- the same inert build scores 421/127/80/144 on walker. Density-FB (60/303/96/469)
is 4-5x ABOVE that build on cheetah, so nothing yet separates "pipeline" from "method" on cheetah.

Discriminating runs LAUNCHED 22:52Z, tmux `exorl_disc`, `scripts/launch_exorl_disc.sh`, logs `logs/exorl_disc/`,
1M steps, all 8 GPUs:
  (a) FB reference config (the one that hit published walker 920/804/400) on cheetah, seeds 0-2
      -> `exp/exorl_fbref/dmc-cheetah_fb_ddpgbc_seed_*`. ~500 = pipeline fine, method is the gap; ~230 = pipeline.
  (b) FB reference on maze at reference maze settings (discount 0.99, lr_actor 1e-6), seeds 0-2
      -> `exp/exorl_fbref/dmc-point-mass-maze_fb_ddpgbc_seed_*`. Target: RLDP-paper FB 427.
  (c) Density-FB Phase-1 recipe on maze with lr_actor 1e-6 (only change), seeds 0-1
      -> `exp/exorl_dfb_mazelr/`. Target: > 302 with no saturated actor.
Score with `scripts/exorl_paper_eval.py` (fb has no INFERENCE_SAMPLES entry -> 50k) on params_1000000.pkl.

## Discriminating runs RESULT (2026-09-18 13:10Z; paper protocol, 10 eps, 50k samples, params_1000000)
`exp/paper_eval_exorl_disc*.jsonl`; scorer layout via symlinks in `exp/exorl_disc_score/`.

| cell | run | task returns | domain mean | published reference |
|---|---|---|---|---|
| Density-FB maze, lr_actor 1e-6 (2 seeds) | `exp/exorl_dfb_mazelr` | TL 942 / TR 685 / BL 799 / BR 646 | **768** | RLDP 671, PSM 514, Lap 501, FB 427, HILP 308; our 2M at 3e-4: 302 |
| FB reference, maze (disc .99, lr_actor 1e-6, 3 seeds) | `exp/exorl_fbref/dmc-point-mass-maze_*` | TL 731 / TR 329 / BL 414 / BR 4 | 370 | RLDP-paper FB 427 (same per-task shape: BR ~0) |
| FB reference, cheetah (3 seeds) | `exp/exorl_fbref/dmc-cheetah_*` | run 136 / run_b 166 / walk 495 / walk_b 668 | 366 | Touati 513 (1M), HILP-paper 412 (1M), RLDP 586 (5M); Density-FB 2M: 232 |

Reading:
- MAZE = HYPERPARAMETER, closed. The one reference maze setting we never applied (actor lr 1e-6) takes
  Density-FB from 302 to 768, above every published method; both seeds agree within 5% on three of four
  tasks and no actor saturates. Seeds 2-4 launched (round 2) for the five-seed table.
- CHEETAH pipeline is fine: the FB reference config gets 366 at 1M (0.7-0.9x the 1M published FB numbers,
  cheetah keeps rising to 3-5M in every reference), while Density-FB sits at 232 at 2M -> a METHOD gap on
  cheetah, not a pipeline gap and not the hit kernel (sup86 radius changed nothing).

Round 2 LAUNCHED 13:15Z, tmux `exorl_disc2`, `scripts/launch_exorl_disc2.sh`, 1M, logs `logs/exorl_disc/`:
  maze lr 1e-6 seeds 2,3,4 -> `exp/exorl_dfb_mazelr`;
  (d) cheetah lr_actor 1e-4 (reference actor lr; ours was 3e-4) seeds 0-2 -> `exp/exorl_dfb_cheetahlr`;
  (e) cheetah actor reward `kernel` (exact bump on hindsight rows) seeds 0-1 -> `exp/exorl_dfb_cheetahkernel`.
Score as before with `--task-set paper` (cheetah task set is identical in both sets).

## Round 2 RESULT (2026-09-18 16:30Z; paper protocol, 1M, exp/paper_eval_exorl_disc2_*.jsonl, *_disc_mazedfb.jsonl)
| arm | domain mean (pop std) | per task |
|---|---|---|
| cheetah, actor lr 3e-4 -> 1e-4 (2M reference recipe, 1M), 3 seeds | **292** (35) | run 95 / run_b 137 / walk 421 / walk_b 516 |
| cheetah, exact-reward (kernel) actor readout, 2 seeds | 238 (27) | 64 / 108 / 308 / 471 |
| maze, actor lr 1e-6, all 5 seeds | **692** (88) | BL 688 / BR 548 / TL 774 / TR 758 |
Reading: the reference actor lr is a real lever on cheetah (+60 over the 2M cells at half the budget; FB-ref in
the same pipeline = 366), the kernel readout is not (238 vs 232). Maze at five seeds = 692, still above every
published method (RLDP 671). Table: point-mass cell -> 692.2 (pop std); cheetah cell held at 232 until the
20-job 1M-recipe + lr 1e-4 batch (tmux `exorl_1mrecipe`, exp/exorl_dfb_1mrecipe/) is scored -- that batch
supersedes walker/cheetah/quadruped/maze cells with one recipe, one actor lr policy, five seeds.

## 1M-recipe + reference actor lr batch RESULT (2026-09-19 00:50Z) -- THE TABLE CELLS
exp/exorl_dfb_1mrecipe (tau 0.005, value [512,512,512], B [256,256] motivo norm, lr 1e-4, lr_actor 1e-4 / maze 1e-6,
1M, 5 seeds), paper protocol, exp/paper_eval_exorl_1mrecipe.jsonl, mean (pop std) over per-seed domain means:
| domain | this batch | previous best cell | published FB / best published |
|---|---|---|---|
| walker (stand/walk/run/spin) | **724.6** (41.2): 843/744/327/985 | 681 (3 seeds, lr 3e-4) | 811.5 / 811.5 FB |
| cheetah | **301.4** (45.4): run 102 / run_b 126 / walk 401 / walk_b 577 | 232 (2M) | 672.7 / 688.7 TD-JEPA |
| quadruped | **655.9** (15.9): jump 733 / run 460 / stand 904 / walk 527 | 569 (2M) | 595.6 / 694.8 HILP |
| point-mass (4 reach) | **751.5** (74.1): BL 739 / BR 521 / TL 944 / TR 801 | 692 (lr 1e-6, 2M recipe) | 513.0 / 513.0 FB |
One recipe, one budget, five seeds on every domain; tab:manipulation ExoRL row updated to these cells.
Levers that got here: actor lr (references' per-domain values), the 1M recipe (tau .005 + 3-layer value tower).
Not levers: hit-radius sharpness (sup86), exact-reward actor readout, backward-norm removal (exp/exorl_dfb_bnorm0,
score at 1M pending, not expected to matter). Remaining gap to FB is walker/cheetah only.

## Single-lever sweep RESULT (2026-09-19 16:40Z; 1 seed, 1M, exp/paper_eval_exorl_levers.jsonl; base = 1M-recipe batch seed 0: walker 744 / cheetah 274)
| lever | walker | cheetah |
|---|---|---|
| sphere (measure_density_z_rand_frac=1.0) | 763 (+19) | 340 (+66) |
| qnorm (actor.q_normalize=true) | 778 (+35) | 293 (+18) |
| tower1024 (value_hidden_dims=[1024]) | 618 (-125) | 381 (+106) |
| tau001 | 707 (-36) | 340 (+66) |
| headmean (measure_density_actor_head_reduce=mean, new knob) | 702 (-42) | 339 (+65) |
| disc099 | 684 (-60) | 336 (+62) |
| temp2 (actor_temp 2) | 633 (-111) | 211 (-63) |
| zdim100 | 749 (+6) | 269 (-5) |
Shared levers: sphere + qnorm. Cheetah-only levers: tower1024, tau001, headmean, disc099. Round 2 = sphere+qnorm at
5 seeds (walker, cheetah) + 1-seed cheetah stacks (scripts/launch_exorl_combo.sh, exp/exorl_combo/).
