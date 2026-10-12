# Density-FB: squared error vs cross-entropy (phase 1, 500k steps, seeds 0-2)

**Question.** Density-FB's critic predicts a softmax distribution over the batch's next
states and is fit to its TD target by cross-entropy. Does the cross-entropy matter, or would
squared error on the predicted probabilities do as well?

**Setup.** The only change is the loss (`--override measure_density_loss=sq`). Everything else
matches the paper's cross-entropy runs: config, seed and evaluation schedule. The checks
compared each squared-error run's composed config with its cross-entropy twin's
`config.json`. Two keys the older runs did not record also differ; both are set to the
code default those runs used.

- Models: the goal-conditioned model (`density_fb`, scored on `eval.csv`; the paper's
  Density-FB table column) and the reward-inference model (`density_fb_z`, scored on
  `eval_relabel.csv`, where the task vector is inferred from 10k reward samples).
- Tasks: cube-single, cube-double, scene.
- Seeds: 0-2. Runs stopped at 500k steps.
- Score: per run, the task-mean success averaged over the 300k/400k/500k evaluations.
  Cell = mean +- SE over seeds.
- Cross-entropy numbers are the same evaluations of the paper's runs: the HF logs
  amsks/factored-fb-checkpoints@iclr2027, seeds 0-2.

**Result (success %, mean +- SE, n=3).**

| model | task | squared error | cross-entropy |
|---|---|---|---|
| goal-conditioned | cube-single | 39.8 +- 5.1 | 90.3 +- 0.9 |
| goal-conditioned | cube-double | 18.0 +- 0.4 | 58.7 +- 3.7 |
| goal-conditioned | scene | 48.3 +- 1.7 | 61.4 +- 2.4 |
| reward inference | cube-single | 77.8 +- 1.0 | 76.4 +- 3.8 |
| reward inference | cube-double | 18.0 +- 2.7 | 20.7 +- 1.4 |
| reward inference | scene | 47.8 +- 3.5 | 49.1 +- 2.4 |

Per-seed values are in `phase1_500k_seeds012.csv`.

**Reading.**

- For the goal-conditioned model, cross-entropy is clearly better on all three tasks: +50,
  +41 and +13 points. A freshly trained cross-entropy cube-single run from 2026-09-25
  (`exp/fz_ablation_goal/softmax`, seeds 0-2) scores 93.6 at the same evaluations. So the
  cube-single gap is not an artefact of resumed or older baseline runs.
- For the reward-inference model the two losses tie on all three tasks (within about one SE).
  This agrees with the September cube runs (`exp/sqloss`, 1M steps: cube-single 78.2 vs 81.3,
  cube-double 30.4 vs 30.4).
- The two models share the critic and differ in the reward the actor maximises. The
  goal-conditioned actor averages the exact goal-reaching kernel under the predicted
  distribution, so it needs an accurate probability on the few next states that reach the
  goal. The reward-inference actor averages a linear reward (B_slow(u) - mu)^T z /
  sqrt(d). A plausible reading: cross-entropy fits the small, high-value probabilities that
  the kernel reward reads, while squared error on probabilities under-weights them. This is
  not yet tested directly.

**Status.** Phase 1 only: 3 seeds at 500k. Phase 2 extends the same runs to 1M steps and
adds seeds 3-4 (`bash scripts/launch_sq_ablation.sh 1000000 "0 1 2 3 4"`). It is then scored
like the paper table (`python scripts/score_sq_ablation.py --ce-exp <hf exp> --steps
800000 900000 1000000 --seeds 0 1 2 3 4`).

Runs: `exp/sq_ablation/` (checkpoints at 250k and 500k). Logs: `logs/sq_ablation/`.
Launcher: `scripts/launch_sq_ablation.sh`. Scorer: `scripts/score_sq_ablation.py`.
