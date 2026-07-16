# Mechanism probes T1–T3 (spec, pre-registered) + T4–T9 queue

*2026-07-16. T1–T3 are forward-pass / rollout probes on existing checkpoints — no training.
Outputs → `analysis/rescuing-fb/` (one JSON + one short md each). Each probe states its
prediction BEFORE running. Shared harness: `scripts/atlas/{critics.py, rerank_eval.py,
rerank_audit_multienv.py}`, banks in `analysis/atlas/banks.npz`. Checkpoints:
`~/bridge_ckpt_backup/gciql_outputs/factored-fb/<group>/...`.*

## T1 — Goal-smoothed reranking (tests the ADVANTAGE-POOLING mechanism)

**Hypothesis (from 01-synthesis §2.2):** the MLP's action-advantage signal exists but is
per-goal noise (it never pools across goals); the bilinear head pools it by construction.

**Design.** Reranking as in Exp 1 (`rerank_eval.py`: at every env step sample N=32 candidates
from the frozen BC flow μ, execute argmax_a Q(s,a,g_eval)), except the scoring critic uses a
goal-smoothed value:

    Q̄_K(s,a,g) = (1/K) Σ_{j=1..K} Q(s,a,g_j),   g_j = g with cube-xyz dims (obs[19:22])
                                                  perturbed by ε_j ~ N(0, σ²I₃)

Arms (cube, C3 protocol, H=200, 50 episodes/task × 5 tasks, seed-0 checkpoints):
- critics: MLP (`GCIQL-State-Flow` 1M) and bilinear-d50 (`GCIQL-State-Flow-Bilinear-d50` 500k)
- K ∈ {1 (=Exp-1 replication), 8, 32}; σ ∈ {0.02, 0.04} (units: 1 obs unit = 10 cm ⇒ 0.2/0.4
  obs units — VERIFY the unit against `cube_env` before running; 0.04 m = the success radius)
- **Control arm**: K=32 with g_j drawn as RANDOM dataset goals (not local perturbations) —
  smoothing must be LOCAL to help; random-goal smoothing destroys goal-specificity and should
  hurt both critics.

**Pre-registered predictions.**
- Pooling mechanism TRUE: MLP rerank rises materially with K (0.57 → ≥0.70 at K=32,σ=0.04);
  bilinear stays within noise of 0.95 (its Q̄ ≈ Q by construction: smoothing ψ(g) over a local
  cloud barely moves a rank-50 linear functional).
- Pooling mechanism FALSE (MLP action info absent, pure rank collapse): MLP flat in K.
- Both critics degrade under the random-goal control.

**Output**: `analysis/rescuing-fb/t1_goal_smooth_rerank.json` (per arm: mean success, per-task,
n_episodes) + `t1_notes.md`.

## T2 — DR3 co-adaptation across training (tests the STABILIZER mechanism)

**Hypothesis (01-synthesis §2.3):** removing the ψ(g) bottleneck (bilinear_sg) opens a DR3
feature co-adaptation channel; its runaway growth is what q_mean-crossing-positive looks like.

**Design.** For each checkpoint, on a fixed dataset batch (B=4096 transitions, fixed RNG):

    coad(ckpt)  = E[φ(s,a)·φ(s′,a′_ds)] / sqrt(E[‖φ(s,a)‖²]·E[‖φ(s′,a′_ds)‖²])
    (a′_ds = the dataset's next action — behavior-consistent, no actor needed)
    plus: E[‖φ‖²], E[‖ψ‖²], q_mean, q_max, and for bilinear_sg also ψ([s,g]) stats.

φ definitions: bilinear/bilinear_sg → the explicit φ tower (per head, report head-0 and mean);
MLP → penultimate-layer activations of the Q-MLP via flax `capture_intermediates`.

Checkpoint series:
- **bilinear_sg** (`~/bridge_ckpt_backup/GCIQL-State-Flow-Bilinear-SG-d50/`, 3 seeds): every
  saved step available (at minimum 500k; if intermediates are absent, supplement with the
  training-log q_mean/value_loss curves from the run dirs and correlate the final-point coad
  across the 3 seeds — the 2 divergent seeds must show higher coad than the stable one).
- **bilinear + MLP exp5 retrains** (save_interval=50k, output under `/home/adityamohan/
  exp_frozen` / `exp5_retrain` — locate via `scripts/atlas/exp5_trajectory.py` and
  `analysis/atlas/exp5_trajectory.json`): 0→500k trajectory.

**Pre-registered predictions.** (a) bilinear_sg coad ≫ bilinear coad, and within bilinear_sg
the divergent seeds > stable seed; (b) coad(t) rises before/with q_mean's approach to 0 for
divergent runs; (c) bilinear and MLP coad stay flat-ish over 0→500k. Failure of (a) kills the
DR3 reading of the sg divergence (fall back to "no bound + no V-target" generic instability).

**Output**: `analysis/rescuing-fb/t2_dr3_stats.json` + `t2_notes.md` (+ a small matplotlib
figure `t2_coad_vs_step.png`).

## T3 — Off-support peak × domain (tests the EXTRACTION boundary condition)

**Hypothesis (01-synthesis §2.4):** the bilinear off-support peak is fatal only where the
behavior action-manifold is narrow (cube scripted play); antmaze's broad action coverage makes
DDPG+BC ascent effectively on-support — explaining bilinear+DDPGBC winning antmaze (0.837)
while losing cube (0.51).

**Design.** Three measurements, states = 512 random dataset states per domain (+ goals = the
domain's eval goals):

1. **Peak statistic** (Finding-1 replication, per domain): for the co-trained bilinear+DDPGBC
   agent, a_DDPG = actor(s,g); reference cloud B_ref = the dataset action at s + actions at the
   64 nearest-neighbor dataset states (whitened obs space). Report
   median[Q(a_DDPG) − max_{a∈B_ref} Q(a)] and frac[Q(a_DDPG) > max]. Cells:
   cube (`GCIQL-State-Flow-Bilinear-d50-ddpgbc`) — expects to REPLICATE ≈79%/+2.14;
   antmaze (`GCIQL-AntMaze-Medium-Bilinear-d50-DDPGBC`) — prediction: frac ≤ 0.25, median ≤ 0;
   scene (`GCIQL-Scene-Bilinear-d50-DDPGBC`) — exploratory, no prediction.
2. **Actor departure**: whitened NN distance of a_DDPG to B_ref (cube expects ≫ antmaze).
3. **Dataset action-coverage** (dataset-only, no model): distribution of NN-distance from
   uniform-random actions in [−1,1]^A to the dataset action set (per-state-conditional: among
   the 64-NN states' actions). Prediction: cube's conditional action manifold is much narrower
   than antmaze's (report median NN-dist ratio).

**Pre-registered predictions.** Peak replicates on cube; absent-or-benign on antmaze; coverage
ratio antmaze ≫ cube. If the peak is equally strong on antmaze, the coverage story is dead and
the DDPG+BC domain flip needs a different account (e.g. value-geometry demands of locomotion).

**Output**: `analysis/rescuing-fb/t3_offpeak_multienv.json` + `t3_notes.md`.

## Queue (not dispatched today)

- **T4** goal/z-conditioned vf on antmaze bilinear+flow (3 seeds) — RUN AFTER RG-FB v0 code
  lands (same knob). Prediction: recovers toward ≥0.6 from 0.24.
- **T5** param-matched MLP + late-fusion two-tower control (3 seeds each).
- **T6** zero-shot generality probe (see 02-rgfb-spec §4.2).
- **T7** FB min-vs-mean targets + pessimism (torch, 3 seeds).
- **T8** faithful one-step FB (torch, 3 seeds).
- **T9** rank of the true advantage from counterfactual rollouts (atlas banks + sim).
- **T10 — centered / advantage measure (the last measure-native rescue).** Every failed
  measure cell trains raw M, where the shared occupancy baseline M̄ swamps the action-
  contingent ΔM (~0.2% of Q variance). Train the measure loss on the μ-centered object
  ΔM(s,a,g) = M(s,a,g) − E_{ã∼μ}M(s,ã,g) (atlas program idea #2), with its own scale norm;
  Q for the actor = U_frozen-or-IQL + ΔM·z. Pre-registered claim boundary: if T10 AND T8 fail,
  the paper may say "no known form of measure-TD grounding supports cube manipulation"; if
  either works, the claim narrows to "the raw policy-conditioned measure loss fails" and the
  fix becomes a contribution. Context: measure-only works on antmaze (0.784 DDPGBC) — the
  failure is manipulation-specific (occupancy≈task on navigation, ≠ on manipulation).
