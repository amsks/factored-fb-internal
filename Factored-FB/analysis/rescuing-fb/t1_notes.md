# T1 — goal-smoothed reranking: RESULT (pre-registered in PAPER/rescuing-fb/03-probes-spec.md)

*2026-07-16. Script `scripts/rescuing_fb/t1_goal_smooth_rerank.py`; raw JSON
`analysis/rescuing-fb/t1_goal_smooth_rerank.json`; run log `t1_run.log` (tmux `t1rerank`,
GPU 0). Protocol = Exp-1/C3: N=32 candidates/step from the shared frozen bilinear BC flow,
argmax Q̄, env.reset() starts, H=200, success = cube within 4 cm, 50 episodes/task × 5 tasks
(250/arm). Critics: bilinear-d50 flow @500k, MLP flow @1M (seed 0). Unit check confirmed vs
`cube_env.compute_observation`: obs[19:22] = (cube_xyz − c)·10 ⇒ σ = 0.2 / 0.4 obs units for
2 / 4 cm. Goal cloud fixed per (arm, task), seeded.*

## Result table (mean success over 5 tasks; per-task in the JSON)

| arm                    | bilinear-d50 | MLP   |
|------------------------|--------------|-------|
| K=1 (Exp-1 replication)| **0.924**    | **0.616** |
| K=8, σ=2 cm            | 0.864        | 0.548 |
| K=8, σ=4 cm            | 0.792        | 0.456 |
| K=32, σ=2 cm           | 0.912        | 0.580 |
| K=32, σ=4 cm           | 0.864        | 0.408 |
| K=32, random-goal ctrl | 0.148        | 0.040 |

Gate: K=1 PASS (bilinear 0.924 vs known 0.944–0.96; MLP 0.616 vs known 0.568–0.592, E=50 vs
25). Binomial SE per arm ≈ 0.02–0.03.

## Verdict vs the pre-registered predictions

1. **Pooling-TRUE prediction (MLP → ≥0.70 at K=32, σ=4 cm): REFUTED.** MLP goes the *wrong
   way*, monotonically in σ: 0.616 → 0.580 (2 cm) → 0.408 (4 cm) at K=32. Inference-time
   goal smoothing does not surface a latent pooled advantage — it actively destroys the
   MLP's usable signal. At fixed σ the MLP is ~flat in K (0.548→0.580 @2 cm; 0.456→0.408
   @4 cm), i.e. more MC samples of the smoothed value don't help either — the smoothed
   functional itself is worse, not noisier.
2. **Pooling-FALSE branch ("MLP flat in K") is the closer fit**, amended: flat in K,
   *declining* in σ. The MLP's action-advantage is goal-LOCAL and brittle — averaging Q over
   a 2–4 cm goal cloud replaces it with neighboring goals' (mis-ranked) advantages rather
   than denoising it. Consistent with the dGOAL −16 to −24 finding: even 2–4 cm goal
   perturbations are effectively distribution-shift for the MLP head.
3. **Bilinear ~invariance: HOLDS at 2 cm, mild real degradation at 4 cm.** K=32: 0.912
   (σ=2 cm, within noise of 0.924) vs 0.864 (σ=4 cm, ≈3 SE below). 4 cm = the success
   radius, so σ=4 cm smoothing mixes in goals whose success sets differ — some drop is
   geometric, not representational. K=8 is noisier (0.864/0.792), as expected from a
   cruder MC estimate of Q̄. Directionally as predicted: smoothing a rank-50 linear
   functional of ψ(g) barely moves the ranking.
4. **Random-goal control: CONFIRMED for both.** bilinear 0.924 → 0.148, MLP 0.616 → 0.040
   (≈ the 0.06 μ-floor). Goal-specificity carries essentially all of the reranking signal
   for both critics — there is no goal-agnostic "good action" prior doing the work.
   (Per-task quirk: bilinear ctrl task 4 = 0.70 while tasks 1,2,3,5 ≈ 0 — a random-goal
   average that happens to steer toward task 4's region; worth ignoring, mean is the arm.)

## Interpretation for §2.2

The advantage-pooling mechanism is NOT an inference-time property of the trained MLP: you
cannot recover the bilinear head's cross-goal pooling by Monte-Carlo smoothing the MLP's Q
over local goal clouds at eval. If pooling is the mechanism, it must act at TRAINING time —
the rank-d ψ(g) bottleneck as a parameter-sharing constraint that shapes φ(s,a) during TD —
which this probe cannot test (that is the T5 param-matched / two-tower control and the
training-time regularizer arm). The surviving readings after T1: (a) the MLP's per-goal
action ordering is largely absent/collapsed (rank-collapse reading), and (b) whatever
ordering it has is goal-brittle. T1 kills the cheap "smooth at eval" rescue for RG-FB: v0
should keep the structural bilinear/low-rank head rather than an eval-time pooling shim.

Deviations from spec: none material. Choices not pinned by the spec, logged: goal cloud
sampled once per (arm, task) (fixed smoothed critic per rollout, not per-step resampling);
K perturbed variants exclude the exact g; E=50 batched envs (spec's 50 episodes/task);
random-goal control drawn uniformly from the 1M dataset observations, per-task seeded.
