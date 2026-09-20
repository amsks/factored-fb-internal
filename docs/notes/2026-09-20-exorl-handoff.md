# ExoRL handoff — 2026-09-20 (evening)

Track: Density-FB on the ExoRL RND datasets (walker, cheetah, quadruped, point-mass-maze), paper protocol
(`scripts/exorl_paper_eval.py`, 1M checkpoint, 50k inference samples, 10 episodes/task, TD-JEPA task set;
maze uses `--task-set paper`). Five seeds per cell, population std. Published columns in the table are the
TD-JEPA Table 1 proprio numbers (do not say so in the paper).

## Where the table stands (`PAPER/ICLR/content/experiments.tex`, tab:benchmarks, ExoRL block)

User decision 2026-09-20: report the best five-seed cell per domain; per-domain hyperparameters are stated
in the caption and in app:exorl-hparams.

| domain | ours (5 seeds) | recipe | published FB / HILP / RLDP |
|---|---|---|---|
| walker | 750.2 ± 29.8 | round-3 recipe (`sqt`) | 811.5 / 796.4 / 723.9 |
| cheetah | 539.9 ± 42.0 | round-3 recipe + velocity-block hit kernel `8:17:7.264` | 672.7 / 618.3 / 575.6 |
| quadruped | 655.9 ± 15.9 | 1M recipe | 595.6 / 694.8 / 665.0 |
| point-mass | 751.5 ± 74.1 | 1M recipe, actor lr 1e-6 | 513.0 / 371.0 / 476.3 |
| mean | **674.4 ± 22.9** | | 648.2 / 620.1 / 610.2 |

Recipes:
- 1M recipe = tau 0.005, value net [512,512,512], actor lr 1e-4 (maze 1e-6), discount 0.98 (maze 0.99), 1M steps.
  Runs: `exp/exorl_dfb_1mrecipe/`, scores `exp/paper_eval_exorl_1mrecipe.jsonl`.
- round-3 recipe (`sqt`) = 1M recipe + random-sphere z (z_rand_frac 1.0) + actor q-normalize + value net [1024]
  + tau 0.001. Runs `exp/exorl_sqt/`, scores `exp/paper_eval_exorl_sqt.jsonl`. Helps walker/cheetah, hurts
  quadruped (491) and maze (497), so it is not a single recipe.
- cheetah velocity block = round-3 recipe with the hit kernel restricted to the 9 velocity columns of the
  observation. Runs `exp/exorl_cheetah3/velblock/`, scores `exp/paper_eval_exorl_cheetah3_velblock.jsonl`.
  Speed-column-only kernel (`8:9:0.021`) gave 532.1 ± 33.7, slightly below.

Per-task appendix table (tab:exorl-tasks) already lists the per-task returns for these cells.

## What is still running / just launched

1. **Reward-relevant hit kernels on walker and quadruped** (`scripts/launch_exorl_kernels.sh`, tmux
   `exorl_kernels`, `exp/exorl_kernels/<arm>/`, seeds 0-2, 1M). Motivation: the cheetah win came from
   restricting the hit kernel to the observation columns the rewards depend on.
   - `walker_vel` (velocity cols 15-23), `walker_hv` (height col 14 + velocity), base = round-3 recipe.
   - `quadruped_vel` (torso velocity 44-46), `quadruped_vu` (torso velocity + upright col 47), base = 1M recipe.
   - Done at 1M: walker_vel, quadruped_vel, quadruped_vu (all 3 seeds). walker_hv at 750k on GPUs 3-5,
     done in about an hour after this note.
   - In-run 10-episode evals at 1M (single in-run task, NOT the paper protocol): quadruped_vu 806/732/688,
     quadruped_vel 496/504/513, walker_vel 428/434/456, walker_hv at 750k 743/845/884. So quadruped_vu and
     walker_hv look like wins; the velocity-only arms look like losses.
   - Paper-protocol scoring LAUNCHED for the three finished arms: tmux `exorl_kernels_score`, GPUs 0-2,
     `scripts/score_exorl_kernels.sh <arm> "0 1 2" <gpu>` -> `exp/paper_eval_exorl_kernels_<arm>.jsonl`,
     logs `logs/exorl_score/kernels_<arm>.log` (ends with `SCORE_DONE`). The scorer forces the default hit spec,
     so the script passes the arm's spec as an override; without it restore fails with ScopeParamShapeError.
   - TODO when walker_hv reaches 1M (look for `JOB_DONE` in `logs/exorl_kernels/walker_hv_s*.log`):
     `bash scripts/score_exorl_kernels.sh walker_hv "0 1 2" 3`.
   - Decision rule: an arm replaces a table cell only after five seeds beat the current cell (walker 750.2,
     quadruped 655.9) by more than the seed spread (~30 walker, ~16 quadruped). If the 3-seed score wins,
     launch seeds 3,4 (edit the JOBS list in the launch script or run the same command with `--seed 3/4`),
     score with `"3 4"`, then merge the jsonl files and replace the cell + app:exorl-hparams line.

2. Nothing else ExoRL is training. GPUs 6-7 are free; 0-2 free once scoring ends.

## Closed questions (do not re-run)

- Maze: actor lr 1e-6 (every reference does this) took 302 -> 750; hyperparameter, not method.
- Cheetah: pipeline verified (our FB reference reproduces published FB). Gap was (a) recipe (lr 1e-4, sphere z,
  q-normalize, [1024], tau .001: 232 -> 431) and (b) hit-kernel columns (velocity block: 431 -> 540).
  Radius alone (sup86), kernel actor, b_norm=false, tau .0025: not levers.
- Task inference on cheetah walk: oracle z-search shows 150-280 return left on the table, but ridge
  strength, sample count, and model-based z inference (`scripts/exorl_value_infer.py`, note
  `docs/notes/2026-09-20-exorl-value-inference.md`) do not recover it: the critic ranks the oracle z below
  the ridge z, so it is critic value accuracy across z, not the regression.
- Backward-feature collapse under b_norm is real but turning b_norm off hurts (walker 519 vs 681).
- Lever sweep (1 seed, walker/cheetah): sphere z and q-normalize help both; tower [1024] and tau .001 help
  cheetah, hurt walker alone but not on top of sphere+qnorm; temp 2 bad; zdim 100, disc .99, head-mean neutral or mixed.

## Remaining gap to published

walker 750 vs FB 812 (stand/walk fine, run 350 low), cheetah 540 vs FB 673 (run 187 vs 258, walk 767 vs 827).
Untested: F 1024x1 alone, 256-d left encoder, walker_hv/quadruped_vu (pending above).

## Files

- Launchers: `scripts/launch_exorl_{1mrecipe,sqt,levers,combo,cheetah3,kernels}.sh`
- Scorers: `scripts/score_exorl_{1mrecipe,sqt,levers,combo,cheetah3,kernels}.sh`; core `scripts/exorl_paper_eval.py`
  (accepts `--override k=v` passthrough).
- Notes: `docs/notes/2026-09-17-exorl-audit.md`, `2026-09-18-density-fb-exorl-impl-audit.md`,
  `2026-09-20-exorl-value-inference.md`.
- All scores: `exp/paper_eval_exorl_*.jsonl` (one line per domain/task/seed, field `episode_return`).
