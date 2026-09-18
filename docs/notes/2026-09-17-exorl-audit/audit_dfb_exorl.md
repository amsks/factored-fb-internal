# Audit: Density-FB (`density_fb_z`) on ExoRL, 2M / 5 seeds (2026-09-17)

Read-only audit of `exp/exorl_dfb/<domain>/density_fb_z/*` scored in `exp/paper_eval_exorl_dfb_2M.jsonl`.
All numbers below were computed from the run directories, the logs, the dataset files, and one
diagnostic re-inference on the final checkpoints (script in this scratchpad: `diag_z.py`,
logs `diag_maze.log`, `diag_loco.log`). No repo file was modified.

## 0. What the pipeline actually does (so the findings have a referent)

Training (`impls/critics/density_fb.py`):
- The critic regresses a successor measure as a softmax classifier over the batch's 1024 next
  states: target = (1-γ)·one-hot(own next state) + γ·(target softmax at the actor's next action)
  (`stage_loss`, density_fb.py:952-1010; `measure_density_hard=true` so the immediate term is
  one-hot, density_fb.py:1000-1006). **The hit radius does not enter the critic's TD target at all
  under the hard immediate term.**
- Task vectors for training rows: half the rows get a task vector encoded from a goal-reaching
  reward (`_delta_phi`: a Gaussian bump of the hit radius around the row's own future state,
  softmax-normalised over the 1024 candidates, density_fb.py:354-374; encoded by the centered ridge
  in density_fb.py:708-770); the other half get a fresh uniform draw on the sqrt(50) sphere
  (`_z_mix_masked`, density_fb.py:772-782; fraction 0.5, config.json). The radius enters ONLY
  through this training reward and through the scaling of the backward tower's input
  (`_density_queries`, density_fb.py:320-325).
- The actor (TD3-style, `impls/actors/ddpgbc.py:88-105`, BC weight 0) maximises
  q = Σ_j w_j(s,a,z)·r_z(u_j) with r_z(u) = (B_slow(u) − μ)·z/√50, the linear proxy
  (`z_actor_reward_rows`, density_fb.py:1059-1081, mode `latent_centered`; `q_fn` 1098-1140).
- Eval (`impls/main.py:478-492`, `scripts/reeval_checkpoint.py:149-170`): 50,000 rows are drawn
  from the training set; their `rewards` column is the re-simulated task reward
  (`keep_dataset_rewards`, `impls/utils/datasets.py:363-396`; `env/exorl_env.py:436-498`), aligned
  to next states; the task vector is the centered ridge solve of that reward on the slow backward
  tower's features, projected to the sqrt(50) sphere (`infer_cond`, density_fb.py:1250-1288). The
  actor then rolls on that z with the goal path disabled (`InferredCondAgent`,
  impls/utils/eval_cond.py). `inferred_cond_norm` is sqrt(50)=7.071 by construction; the cosine
  to the all-ones vector is the constant-reward-fallback detector (eval_cond.py:73-89).

## 1. Per-seed failure analysis

Training-time `eval.csv` evaluates ONE task per domain (the domain default `exorl_task`: walker
walk, cheetah run, quadruped walk, maze reach_top_left), 10 episodes, with the same 50k-sample
inference as the paper protocol (job scripts pass `--infer_cond_episodes 50000
--eval_on_inferred_cond`). Series (return at 0/250k/.../2M):

```
walker (walk)       s0 26 728 480 672 629 593 738 740 687
                    s1 24 674 644 618 447 406 458 690 523
                    s2 24 778 702 540 675 800 723 728 669
                    s3 23 621 606 562 604 696 679 711 567
                    s4 18 686 674 575 210 691 728 406 699
cheetah (run)       s0 32  84  78  75  95  72  75 110  78
                    s1  0  43  44  65  48  58  20  45  86
                    s2 12  66  82  86  62  71  40  74  16
                    s3 24  33  77  84   5  46  10  44 109
                    s4  0  49  51  57  67  95  93 133  71
quadruped (walk)    s0 175 411 513 150 286 343 385 402 328
                    s1 148 295 276 306 226 218 386 519 443
                    s2 159 294 108 369 481 317 439 404 516
                    s3 224 674 434 528 496 476 267 503 428
                    s4 197 659 401 451 443 462 406 544 629
maze (reach_top_left) s0 4 759 795 158 566 747 757 293 555
                    s1  0 841 757 751 730 758 381 753 326
                    s2  0 748 759 758 656 339 138 185 639
                    s3  0   0   0   0   0   0   0   0   0
                    s4  0 326  84 755 758 756  11 681 596
```

Reading:
- **Walker is flat from 250k onward** (all five seeds reach 620-780 walk at 250k and end at
  520-700). Nothing is still rising at 2M; 2M bought nothing over 250k. The paper-protocol
  walk scores (614/625/635/592/737) agree with the training-time 2M values to within eval
  noise, so walker has no dead seed and no late instability — it is uniformly ~100-200 below
  the FB reference (stand 715 vs 920, walk 641 vs 804, run 204 vs 400).
- **Cheetah run is dead in every seed at every checkpoint** (16-133; published FB ~250). Not a
  late collapse and not a seed effect. "cheetah walk seed 2 = 46" is one draw of a noisy cell:
  the same checkpoint scored 16 on run at the training-time 2M eval and 76 on run in the paper
  eval; seed 1 scored 19 run / 107 run_backward. The cheetah cells are dominated by eval-to-eval
  variance of a poor policy, not by one bad seed.
- **Correction to the per-seed lists in the brief** (they were in file order, not seed order).
  Sorted by seed 0..4 from `exp/paper_eval_exorl_dfb_2M.jsonl`:
  ```
  cheetah   run [78,56,19,76,71]  run_backward [81,107,46,79,167]  walk [335,317,46,450,367]  walk_backward [371,477,287,458,752]
  maze      bottom_left [326,291,644,0,327]  bottom_right [326,451,322,0,478]  top_left [609,288,665,0,693]  top_right [0,302,315,0,1]
  quadruped jump [392,677,664,575,734]  run [291,418,428,377,505]  stand [796,851,906,460,965]  walk [330,431,482,434,661]
  walker    run [190,247,176,197,211]  spin [975,971,977,978,977]  stand [766,783,714,518,794]  walk [614,625,635,592,737]
  ```
  So: **maze seed 3 is dead in all four tasks**; the other maze zeros are reach_top_right seeds 0
  and 4; cheetah seed 2 is the weak cheetah seed in three of four tasks (19 / 46 / 46); the
  460 stand and the 518 stand are both seed 3 (quadruped, walker).
- **Maze seed 3 is a collapsed actor, present from the first checkpoint.** Training-time
  reach_top_left is 0 at all nine evals and the ten per-episode returns are identical to 1e-68 at
  250k, 1.75M and 2M (`eval_episodes.jsonl`): the same trajectory every time. Probing the 2M
  actor directly (`diag_act.py`, `diag_act.log`): for 512 dataset states and four random task
  vectors it outputs exactly (-1, -1) for every state and every z (mean |a| = 1.000, std over
  states 0.000). Its actor Q mean never rose (0.09-0.16 vs 0.36-0.44 for the other seeds,
  train.csv). This is the TD3 actor with BC weight 0 saturating at the action bound; nothing at
  inference time can rescue it. Seed 4 is also 99.9% saturated (|a| > 0.99 on 99.9% of
  states) but state-dependent (std over states 0.83-1.0): a bang-bang controller, which is why
  its per-episode returns are so tightly clustered (583-616) and why a task vector either
  steers it to the goal or not at all.
- **The other maze zeros (top_right seeds 0, 4) and the 300-vs-650 split are per-cell task-vector
  outcomes, not noise between episodes.** Per-episode returns within a cell agree to ±5% and the
  training-time series of a healthy seed swings between ~750 and ~150-350 from checkpoint to
  checkpoint (seed 0: 795 → 158 → 566 → 747 → 293 → 555; seed 4: 758 → 11 → 681 → 596). The
  policy for a given z is nearly deterministic; what moves is which z the checkpoint gets and how
  the bang-bang actor responds to it. The reward-sample draw itself is not the source: re-inferring
  z from four independent 50k draws gives cosine ≥ 0.997 between draws on every maze cell tested
  (§6), so the instability is checkpoint-to-checkpoint drift of the actor's response to a stable z,
  not inference noise.
- **Quadruped stand seed 3 = 460**: training-time walk for seed 3 is normal (428 at 2M); one weak
  task cell, not a dead seed. Quadruped is the domain where the linear proxy is best matched
  (stand/run/jump rewards are linear in the observation, R² ≥ 0.996, see §4) and the only domain
  near the published band.
- **Would an earlier checkpoint score better?** Walker: 250k-500k checkpoints are as good as 2M
  (no gain, no loss). Maze: 250k-1.25M training-time evals are mostly 750+ for seeds 0/1/2/4 vs
  300-640 at 2M — the maze policies drift late (seed 2: 748→639, seed 1: 841→326 on the
  training-time series), so an earlier checkpoint would likely score higher, but the dominant
  effect is the inference draw, not the checkpoint.

## 2. Hit-spec sanity

Specs (job scripts `logs/exorl_dfb/gen_phase1.sh:17-22`; scorer `scripts/exorl_paper_eval.py:77-78`):
walker `0:24:15.9` = all 24 observation dims, L2 radius 15.9; cheetah `0:17:7.5` = all 17 dims;
quadruped `0:78:21.087` = all 78 dims; maze `0:2:0.03` = (x, y), the benchmark's own target size.
The quoted form was parsed correctly: every config.json records the string (e.g.
`"measure_hit_spec": "0:24:15.9"`), and the unquoted landmine was caught in the pre-launch probe
(`logs/exorl_dfb/PROBE_walker_s0.log`: `AssertionError: hit-spec '1455.9' looks like OmegaConf's
sexagesimal parse`), which is why one script per job was generated.

Calibration (`scripts/calibrate_hit_spec.py`; `env/domains/dmc-walker.yaml` header;
`logs/exorl_dfb/quadruped_calib.log`): the radius is the 1.33% quantile of the L2 distance between
random observation pairs from 200 RND episodes, i.e. the fraction of random pairs "inside the ball"
is matched to cube-single's 0.0133 under `19:22:0.4`. Recomputed here on 200 episodes per domain:

| domain | random-pair median L2 | radius | frac random pairs ≤ r | frac consecutive-step pairs ≤ r | 50-step-apart pairs ≤ r | median / radius |
|---|---|---|---|---|---|---|
| walker 0:24 | 50.8 | 15.9 | 0.0133 | 0.20 | 0.041 | 3.2 |
| cheetah 0:17 | 25.1 | 7.5 | 0.0135 | 0.67 | 0.058 | 3.4 |
| quadruped 0:78 | 32.1 | 21.1 | 0.0136 | 0.20 | 0.025 | 1.5 |
| maze 0:2 | 0.33 | 0.03 | 0.021 | 1.00 | 0.42 | 11 |
| cube-single 19:22 (reference) | 2.45 | 0.4 | 0.0136 | – | – | 6.1 |

Two things follow:
- The distance is dominated by unnormalised velocity dims (walker: 6 velocity dims carry 95% of
  the block variance; cheetah 97%). A "goal" in this kernel is a velocity configuration, and even
  consecutive states are usually outside the radius (walker 80%, quadruped 80%) — the radius has no
  physical meaning as "reaching a state".
- **Matching the hit rate does not match the kernel's shape.** The training-task reward is the
  Gaussian bump softmax-normalised over the 1024 candidates. Its concentration (effective number of
  candidates carrying the reward, on a 1024-candidate batch, 64 random goals):
  cube-single **60** of 1024 (top-1 mass 0.037); maze 51; cheetah 205; walker **277**;
  quadruped **879** (top-1 mass 0.002, i.e. essentially uniform). In 78 dimensions the pairwise
  distance distribution is so concentrated that its 1.3% quantile is only 1.5× below the
  median, so the exp(-½ d²/r²) kernel is nearly flat over the batch. On walker/cheetah/quadruped
  the encoder therefore fits its training task vectors to near-constant rewards, and the actor's
  training-time reward on the encoded rows is near-flat; the useful training signal comes from
  the sphere-draw rows only. Is matching a fixed hit rate sensible here? No: the ExoRL tasks are
  velocity/posture rewards, not goal-reaching; the training tasks the encoder and actor see are
  "reach a velocity configuration" bumps that are (i) flat over the batch and (ii) unrelated to
  the eval rewards' structure. The maze is the one domain where the spec is the benchmark's own
  success ball and the bump is as peaked as cube's.

## 3. Config diff

Recorded config: `exp/exorl_dfb/dmc-walker/density_fb_z/*_seed_0/config.json`.

(a) vs the FB reference (`scripts/launch_walker_fbref.sh`, `exp/walker_fbref/*/config.json`):

| knob | Density-FB 2M | FB reference | note |
|---|---|---|---|
| critic lr | 1e-4 | 1e-4 (lr_f, lr_b) | same |
| actor lr | 3e-4 | 3e-4 | same (actor yaml default) |
| batch | 1024 | 1024 | same |
| target rate | 0.001 (both towers) | 0.001 | same |
| discount | 0.98 | 0.98 | same |
| F/value tower | [1024]×1 (+ LN) | 1024×1 | same width |
| B tower | [256,256] motivo, sphere-normalised | 256×2 norm | same |
| latent dim | 50 | 256 (`L_dim`) / z_dim 50 | FB's feature dim is 256 |
| ortho loss | none | coef 1 | by design (not in Density-FB) |
| actor block | motivo, alpha 0, TD3 noise .2/.3, no q-normalise | identical | same |
| task-vector mix | 0.5 sphere / 0.5 encoded | 0.5 sphere / 0.5 goal-encoded (`train_goal_ratio`) | same idea |
| goal sampling | value goals 0.2 cur / 0.5 traj-geometric / 0.3 random; actor goals 1.0 traj | none (FB has no goal) | Density-FB only |
| steps | 2M | 1M (fbref) / 3M (published) | – |

(b) vs the OGBench recipe (`scripts/tdjtab_manifest.sh:116-122`, `scripts/launch_dfb_state.sh`):

| knob | ExoRL 2M | OGBench tdjtab | validated where |
|---|---|---|---|
| actor | ddpgbc (TD3, alpha 0) | flowbc | OGBench only for flowbc; ddpgbc is the ExoRL FB convention |
| critic lr | 1e-4 | 3e-4 | 1e-4 is FB's ExoRL value; the Density-FB 1M walker runs used 1e-4 too |
| target rate | 0.001 | 0.005 | **never validated for Density-FB**: 1M walker runs at 0.005 scored higher (below) |
| value tower | [1024] | [512,512,512] | same |
| batch | 1024 | 1024 | same |
| discount | 0.98 / 0.99 maze | 0.99 | ExoRL convention |
| hit spec | full-obs block, calibrated radius | object coords, env success radius | see §2 |
| sphere fraction | 0.5 | 0.5 | OGBench memory says sphere rows hurt cube; not tested on ExoRL |
| actor reward | latent_centered | latent_centered | same |
| hard immediate term | true | true | same |
| dim 50 | 50 | 50 | same |
| eval | 50k reward samples, 10 eps | 10k relabelled rows | protocol |
| steps | 2M | 1M | – |

What the extra steps and the reference recipe bought: the earlier 1M walker/cheetah runs
(`exp/paper_eval_exorl_1m.jsonl`; `exp/walker_arch/.../density_fb_z`, tau 0.005, value
[512,512,512], otherwise identical) scored walker 820 / 699 / 244 / 959 and cheetah 62 / 84 /
262 / 380. The 2M reference-recipe runs score 715 / 641 / 204 / 976 and 60 / 96 / 303 / 469.
Doubling the steps and moving to tau 0.001 + [1024] did not help walker (stand -105, walk -58,
run -40). The reference recipe was audited for FB (`docs/notes/2026-09-15-fb-reference-audit.md`)
and copied to Density-FB without a Density-FB check; the sphere fraction, hit spec, goal mix,
dim 50 and the linear-proxy actor reward were all carried from cube without any ExoRL ablation
(the only ExoRL-specific decisions on record are the full-observation block and the radius
calibration, `env/domains/dmc-walker.yaml`).

Benign log facts: walker seed 0 resumed at step 8476 from a SIGTERM-checkpointed pre-launch probe
with the identical command (`logs/exorl_dfb/dmc-walker_dfbz_s0.log:20`, `PROBE2_walker_s0.log`);
the other 19 runs started fresh. No NaN, no traceback, no retry in any of the 20 logs
(`scheduler_status.txt`: 20 × `DONE (rc=0)`).

## 4. Reward inference at eval and the linear proxy

- Samples: 50,000 uniformly drawn transitions; reward = the task reward re-simulated from stored
  physics at the next state (`env/exorl_env.py:436-498`, cached under `~/exorl_data/relabel_cache`).
- Fit: centered ridge of reward on the slow backward tower's features, λ = 0.01·tr(C)/50,
  covariance and mean are running EMA statistics stored in the checkpoint; result projected to
  the sqrt(50) sphere (density_fb.py:708-770). No encoder network, no reward-weighted mean.
- Actor at eval: rolls on z through F(s,a,z); the linear proxy is used only at training time to
  shape the actor. The eval policy is whatever the actor learned to do for that z.
- Witness columns: norm 7.071 in all 80 cells (sphere projection), cosine to the all-ones vector
  in [-0.31, 0.26] in all cells — no cell hit the constant-reward fallback (which would read 1.0).
  Per-domain mean |cos|: walker 0.07, cheetah 0.10, quadruped 0.12, maze 0.10. These say the
  vectors are non-degenerate; they say nothing about whether they are right.
- How linear are the task rewards? Least squares of reward on the raw observation (200k rows):
  walker stand 0.92, run 0.80, walk 0.56, spin 0.54; cheetah run 0.47, run_backward 0.44, walk
  0.47, walk_backward 0.44 (speed-thresholded rewards); quadruped stand 1.00, run 0.996, jump
  0.996, walk 0.81; maze 0.00-0.01 (sparse: 2-13% of rows nonzero). The domains rank the same way
  as the returns: quadruped (linear) ≈ published, walker (mostly linear) 10-50% below, cheetah
  (thresholded, R² < 0.5) at 25% of published. The fit in the learned 50-d feature basis is
  measured directly in §6.

## 5. Training logs

Clean: no NaN, no crash, no retry, hit spec parsed as a string in every config.json. One resume
(walker s0 from the probe's step-8476 checkpoint). Critic health from train.csv: walker measure
effective support ~28 of 1024 candidates from 250k to 2M (flat); cheetah ~13; quadruped 5-15;
maze 13-36. Actor Q mean decays monotonically after 250k in every walker/cheetah run (walker
0.27 → 0.13, cheetah 0.27 → 0.11) while the measure loss is flat — the actor's proxy reward is
shrinking over training, not the policy improving.

## 6. Diagnostic: re-inferring z from independent 50k draws on the final checkpoints

`diag_z.py` (scratchpad): restore `params_2000000.pkl` exactly as the paper scorer does, draw
K independent 50,000-row reward samples, solve z for each, report the minimum pairwise cosine
between the K task vectors, the least-squares R² of the task reward on the centered slow
backward features (the basis the linear proxy lives in), the correlation of the actor's proxy
reward (B−μ)·z/√50 with the true reward on the same rows, and the return of one/two episodes
rolled on each z.

| cell | cos_min between draws | R² reward ~ linear in B | corr(proxy, reward) | returns per z |
|---|---|---|---|---|
| maze s3 reach_top_left | 0.998 | 0.25-0.26 | 0.50-0.51 | 0, 0, 0 |
| maze s3 reach_top_right | 0.998 | 0.21-0.22 | 0.45-0.47 | 0, 0, 0 |
| maze s4 reach_top_left | 0.9996 | 0.23-0.25 | 0.48-0.50 | 647, 621, 630 |
| maze s4 reach_top_right | 0.999 | 0.14-0.16 | 0.37-0.39 | 5, 7, 5 |
| walker s0 stand | 0.996 | 0.79-0.80 | 0.89 | 772, 749, 783 |
| walker s0 run | 0.993 | 0.73 | 0.85 | 174, 185, 200 |
| cheetah s2 walk | 0.988 | 0.49 | 0.70 | 19, 48, 38 (2 eps each) |
| cheetah s2 run | 0.990 | 0.50 | 0.71 | 23, 7, 18 (2 eps each) |

Conclusions:
- **Inference is stable.** Independent 50k draws give the same task vector to cosine ≥ 0.993
  on every cell, and no cell is near the constant-reward fallback. The 50k-sample protocol is
  not the source of any variance; the maze cell values (0 / ~300 / ~650) reproduce from
  fresh draws.
- **The linear proxy is a poor model of the maze rewards and a mediocre one of walker's.** In
  the learned 50-d basis the maze reward is only 14-26% explained (its reward is a sparse bump at
  one corner; the backward tower was trained on velocity-free 2-d positions but is only 50-d and
  sphere-normalised); walker stand/run are 73-80% explained; cheetah walk/run only 49-50% (proxy correlation 0.70). The actor was trained to maximise the
  linear proxy under the measure, so 20-27% of the walker reward and 75-85% of the maze reward is
  invisible to what the actor optimised. Compare FB itself, whose task vector is the same kind of
  linear readout but whose backward features were trained with the reward-regression objective in
  mind (orthonormality, 256-d features in the reference configuration vs 50-d here).
- Maze seed 4 fails reach_top_right under a stable, non-degenerate z (returns 5-7 in three
  re-inferences, 1 in the paper eval): the bang-bang actor simply does not go there for that z.
  That is a training failure (the policy family the actor learned does not cover that corner),
  not an eval failure.

## 7. Ranked findings (most likely impact on the returns first)

1. **The training-task reward is nearly flat on the locomotion domains, so half the actor's and
   encoder's training rows carry almost no task signal.** The hit radius was set so that 1.3% of
   random state pairs fall inside it, but in 17-78 dimensions the pairwise distance is so
   concentrated that this radius is only 1.5-3.4× below the median distance, and the Gaussian bump
   the encoder is fit to spreads over 205 (cheetah), 277 (walker) and 879 (quadruped) of 1024
   candidates — vs 60 on cube-single with the same procedure (§2). The "goal-reaching" training
   tasks are velocity configurations 80% of consecutive states do not share. The successor
   measure itself is unaffected (one-hot immediate term), so this is an actor/encoder-training
   defect, not a critic defect.
2. **The actor optimises a linear proxy of the reward in a 50-d sphere-normalised basis, and on
   these tasks that proxy explains 73-80% (walker) and 14-26% (maze) of the reward** (§6);
   cheetah's speed-thresholded rewards are only 44-47% linear even in raw observations (§4). The
   domain ranking of the returns follows the linearity ranking (quadruped ≈ published; walker
   10-50% low; cheetah at 25%). The exact-reward readout used on cube ("kernel" mode) was not
   used here (`measure_density_z_actor_reward=latent_centered` in every config.json).
3. **Maze seed 3 is a collapsed TD3 actor (constant action (−1,−1) for every state and every
   task vector) from the first checkpoint on; the other maze cells are bang-bang policies whose
   reach is decided by a stable z** (§1, §6). BC weight is 0 (`actor.alpha=0.0`), so nothing
   anchors the actor to the data. Zero-shot inference is stable (cosine ≥ 0.993 across draws) and
   never falls back to the constant-reward vector.
4. **The 2M "reference recipe" did not help.** The 1M walker/cheetah runs with tau 0.005 and a
   [512,512,512] value tower scored higher on walker (stand 820 / walk 699 / run 244 vs 715 / 641
   / 204). Walker is flat from 250k in all five seeds; 2M steps bought nothing. tau 0.001, the
   [1024] tower, the 0.5 sphere fraction, dim 50, the goal-sampling mix and the proxy actor reward
   were all copied (from FB's audit or from cube) without an ExoRL ablation (§3).
5. Config, launch, and logs are otherwise clean: hit specs parsed as strings in every
   config.json, no NaN/crash/retry, one benign resume (walker s0 from a probe checkpoint at step
   8476), actor/critic learning rates and batch match the FB reference (§3, §5).
