# T6 — zero-shot generality of reward→z→policy: results + verdict

*2026-07-16. Pre-registered spec: `PAPER/rescuing-fb/05-t6-generality-spec.md` (written before
any rollout). Probe: `scripts/rescuing_fb/t6_generality_probe.py`; raw:
`analysis/rescuing-fb/t6_generality_g4.json`. Checkpoint: G4 z-cond
(`~/bridge_ckpt_backup/gciql_outputs/factored-fb/gciql_g4_zcond/sd000_20260616_193331`,
d512, 1M, actor_cond='z', actor_vf UN-conditioned). 25 eps/cell, H=200, CPU-only, cores 0-47.*

## Harness gate: PASS

Anchors (dataset relabel reward, moment z, `match_g` norm): mean **0.912** (gate stage) /
**0.920** (full-stage rerun) ≥ 0.90 threshold; E1a anchor 0.955 (3 restore seeds) — in band.
Per task: 0.84/0.96/1.00/0.92/0.84. Bonus harness checks: synthetic indicator
1[‖cube−g‖<0.04] agrees with the dataset relabel reward EXACTLY (1.0 on all 5 tasks over the
50k sample), and the goal-free `match_data` norm anchors at **0.952** — as good as `match_g`
(the RG-FB goal-free norm rule is validated at d512 too).

## Results (PRIMARY norm = match_data; means over task-variants)

| Family | Success | Uniform control on same metric | Read |
|---|---|---|---|
| (e) anchors, 5 single goals | **0.952** (match_g 0.920) | 0.064 | works (gate) |
| (a) two-goal mixtures, 10 pairs | **0.260** | — | **FAILS** (< 0.7) |
| (b) lift-anywhere (exploratory) | 0.944 | **0.640** (sqrtd: 0.944) | **CONFOUNDED — uncertifiable** |
| (c) region-goals, 3 novel boxes | **0.672** final (0.917 any-step) | 0.003 | **WORKS** (≥ 0.5) |
| (d) uniform r≡1 | goals 0.064 / regions 0.003 / lift 0.640 | — | goal+region metrics valid; lift metric broken |

Per-family detail:

- **(a) pairs**: t2-t3 = 1.00 (deterministic per start: from t2 starts always to g3, from t3
  starts always to g2), t1-t5 = 0.66, t1-t2 = 0.46, everything else ≤ 0.24 with FIVE pairs at
  0.00–0.06. cos(z, ψ(g)) is high for BOTH members in every pair (0.77–0.82, vs 0.88–0.90 for
  anchors) — the inferred z is directionally "between" the two task vectors, and the z-actor
  mostly commits to neither. Moment inference is LINEAR (z_mix ∝ z_g1 + z_g2) while the optimal
  policy of a max-mixture reward is not: the blended z is generally OFF the trained single-goal
  z-manifold, exactly the §6 claim-taxonomy prediction (02-rgfb-spec).
- **(b) lift**: 0.944 looks great, but the negative control lifts 0.640 (match_data) / 0.944
  (sqrtd) — the agent's default z-agnostic behavior is grasp-and-hold, and the lift moment z is
  nearly the mean-ψ direction anyway (cos(z_lift, mean ψ) = 0.970; positives are 22% of the
  dataset). Pre-registered clause (d) triggers: the LIFT metric cannot discriminate task
  inference from default behavior on this dataset. No claim licensed either way.
- **(c) regions**: final-step-in-box 0.592/0.672/0.752 (R2/R1/R3), any-step 0.90–0.96, control
  0.003. Novel table positions that are NOT any eval-task goal are reached and settled on at
  0.6–0.75 — a real, uncontaminated extension beyond the 5 trained eval goals (the gap to
  anchors 0.95 is real too; some of it is the stricter final-step metric: any-step is 0.92).
- **Norm hygiene re-confirmed**: sqrtd collapses anchors t4/t5 to 0.36/0.08 and regions to
  0.05–0.26 final; every claim above is match_data.

## Verdict vs the pre-registered read

**As-coded verdict: REFUTED** — (a) = 0.26 < 0.7 fails both the CONFIRMED and PARTIAL branches.

**Honest caveat (the data fall outside the pre-registered taxonomy):** the REFUTED branch was
defined as "only (e) works", and that description is factually violated — (c) region-goals
work (0.672 ≥ the 0.5 bar) with a clean negative control. The taxonomy anticipated
"mixtures work, novel regions fail" (interpolation-within-hull) and we got the OPPOSITE:
novel single-region goals work, composite rewards fail.

## What claim the paper is licensed to make

- **Licensed**: *zero-shot goal-reaching* — reward samples → moment z → policy, with the
  goal-reaching prior, for **arbitrary dataset-supported cube-position goals** (not just the 5
  eval goals): novel 6 cm regions ≥ 0.06 m from every trained eval goal reach 0.67 final /
  0.92 any-step. This strengthens the v0 claim of 02-rgfb-spec §6 (task-prior-explicit
  zero-shot, prior = goal-reaching) — the prior covers the goal FAMILY, not a goal LIST.
- **NOT licensed**: "zero-shot RL" / behavior-foundation-model language. The first composite
  reward family off the single-goal manifold (two-goal max-mixtures) collapses to 0.26 because
  linear moment inference lands between trained task vectors; non-goal rewards (lift) cannot
  be certified on this dataset (control confound).
- **Constructive implication**: this is direct empirical motivation for the §6 v1 BFM path —
  task-mixture hindsight relabeling (train z at Σwᵢψ(gᵢ) with composite rewards) — the failure
  is a z-space COVERAGE problem at training time, not an inference-formula problem (cos
  diagnostics stay high).

## Caveats / follow-ups

- Ran on G4 (d512, 1M, actor_vf UN-conditioned). **Must be re-run on the RG-FB v0 checkpoint
  (d50, 500k, actor_vf_cond='z') at 500k as confirmation** — booked in DECISIONS.md.
- Single checkpoint, single restore; 25 eps/cell (pairs pooled over 2 variants = 50; regions
  and lift pooled over 5 variants = 125).
- Region-goal success used the pre-registered final-step metric; the any-step secondary is
  ~0.92, so the 0.67 partly reflects "settling" difficulty, not reaching.
