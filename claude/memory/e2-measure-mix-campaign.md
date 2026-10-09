---
name: e2-measure-mix-campaign
description: "E2 scale-controlled IQL+measure interpolation — GUARDRAIL PASSED (15/15 q_mean in [-21,-8] full-history), success 0.80-0.93 all cells = FLAT/coexist branch; 7/15 done @500k; ONLY the prize cross-eval remains"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0c042022-8040-4720-98dd-c66e745e506e
---

**E2 (does adding the FB successor-measure to a WORKING IQL critic destroy / coexist / help,
once the scale confound is removed?) — launched 2026-07-07.** Answers audit-20260706 §1
(18/18 raw mc* runs crashed via unbounded <F,z> inflation).

Code (TDD, committed on gciql-fb-bridge):
- `5b56475` — two default-OFF knobs in `measure_loss`: `measure_scale_norm` (divide TD terms
  by sg[mean(M^2)] → dimensionless gradient, the actor q_loss trick; via new pure staticmethod
  `_measure_terms`) and `measure_stopgrad_phi` (stop-grad phi → measure gradient shapes only
  psi). Refactor is bit-exact when scale_norm=False. 4 new tests + 30 measure-path tests green.
- `<matrix commit>` — 5 configs `configs/gciql/cube_single_state_gciql_flow_bilinear_d50_measuremix_*`
  + launcher `scripts/launch_e2_measure_mix.sh`.

Run matrix (15 = 5 cells × 3 seeds @500k), all: base ~0.90 IQL bilinear-d50 flow,
measure_only=false (grafted on IQL), measure_legs=critic, measure_ortho_coef=0, scale_norm=true:
| cell | measure_coef | stopgrad_phi | group |
|---|---|---|---|
| I-l001 | 0.01 | no | GCIQL-State-Flow-Bilinear-d50-MeasureMix-l001 |
| I-l01  | 0.1  | no | ...-MeasureMix-l01 |
| I-l1   | 1.0  | no | ...-MeasureMix-l1 |
| Q-l01  | 0.1  | yes| ...-MeasureMix-l01-sgphi |
| Q-l1   | 1.0  | yes| ...-MeasureMix-l1-sgphi |

**2026-07-07 ~11:00 status: GUARDRAIL PASSED + flat reading.** Full-history wandb scan of all
15 runs: training+validation critic/q_mean ∈ [−21.2, −8.4] at EVERY log point (spec band
[−100, 10]) → zero scale escapes, numbers certified interpretable. Caveat: grad_norm was
never logged, so the spec's grad_norm>500 criterion is unevaluable — cite guardrail as
q-band-only. Success @latest: l001 .83/.89/.91 · l01 .85/.90/.93 · l01-sgphi .88/.86/.90 ·
l1 .85/.85/.90 · l1-sgphi .80/.86/.87 vs base ~0.90 = the spec's "flat until λ≈1" branch
(measure coexists; old 18/18 collapse was scale, not toxicity; no small-λ improvement).
7/15 finished @500k, 8 at ~340–400k. **REMAINING: the prize cross-eval only** —
eval_zeroshot_inference.py (v0/v1) on each cell's ψ once all 15 finish.

Status: tmux `e2measure` (launcher waits on all 15), logs `/dev/shm/e2measure_logs/*.log`,
dispatch `/dev/shm/e2measure_dispatch.log`. All 15 launched healthy (0 crashes, no OOM).
Smoke @3k confirmed scale control: q_mean −6.7 / critic_loss 0.74 / grad_norm 36 (vs crashed
raw mc1.0: 38.6 / 104 / 1.4k).

OPS CAVEAT (my mistake — didn't `tmux ls` first): launched ON TOP of the still-running
`sotagap` campaign (scene DDPGBC + HIQL sg10). ~23 jax procs / 8 GPUs → contention. Non-fatal
(mem 9.6/23GB, no OOM/crash); ~7/15 progress fast (to 256k), ~8 crawl through their step-0 eval
behind sotagap. Self-resolves as sotagap's slow scene/HIQL runs finish. Just slower overall —
collect when all reach 500000. Check `for f in /dev/shm/e2measure_logs/*.log; do tr '\r' '\n'
<$f|grep -oE '[0-9]+/500000'|tail -1; done`.

EARLY NUMBERS (1 seed/cell, @100–200k, 2026-07-07 — NOT final, l001 λ=.01 not evaled yet):
scale control WORKS (q_mean ≈ −18…−20 = healthy IQL scale; closs ≈1.2–1.6; gnorm ≈40–70,
one cell l1_sgphi 260–316; measure_loss ≈ −0.02). Overall success (mean of 5 tasks):
l01 0.72–0.77 · l1 0.64–0.79 · l01_sgphi 0.73 · l1_sgphi 0.62–0.71. Vs plain-IQL baseline
0.90 / crashed raw measure 0.05–0.16 / plain FB 0.48. **Emerging read = "coexist /
compatible": scale-controlled measure does NOT destroy value structure — the 18/18 crashes
were a SCALE pathology, not an objective conflict.** Same hard-task profile (tasks 4,5 weak).

COLLECT (when all 15 reach 500000):
- progress: `for f in /dev/shm/e2measure_logs/*.log; do tr '\r' '\n' <$f|grep -oE '[0-9]+/500000'|tail -1; done`
- metrics+eval: read `/dev/shm/gciql_outputs/factored-fb/GCIQL-State-Flow-Bilinear-d50-MeasureMix-*/*/eval.csv`
  (per-task + overall success) and `.../train.csv` (training/critic/q_mean, /grad/norm, /measure/measure_loss).
- GUARDRAIL: flag any run with critic/q_mean outside [−100,10] or grad_norm>500 → scale-escaped, not interpretable.
- Dose-response (spec §E2): monotone-drop w/ λ = measure destroys value; flat-then-cliff = compatible-until-dominant;
  small-λ gain = useful aux reg; stopgrad-φ flat where shared-legs drops = conflict lives in φ (the consumed leg).

⚠️ PRIZE CROSS-EVAL — GOTCHA: E2 cells train **actor_cond='goal'** (exact-goal actor), so
`scripts/eval_zeroshot_inference.py` (rolls the actor on a d-dim z via sample_actions_with_z,
needs actor_cond='z') will NOT apply to the E2 checkpoints as-is. Two ways to do the cross-eval:
  (a) CHEAP, value-level: `scripts/infer_ablation_gciql.py --value-latent-dim 50` on each cell's
      ψ — AUC of V_z over reward states — tests whether measure shaping improved the backward
      map's z-inference geometry (no z-actor needed). Recommended first.
  (b) EXPENSIVE, actor-level: re-run the winning small-λ / stopgrad-φ cell(s) with actor_cond=z
      added, then eval_zeroshot_inference.py. Only if (a) shows a real ψ-geometry lift.
  Bar: E1a already got zero-shot 0.955 on the PLAIN critic ([[zeroshot-gciql-e1a-works]]) — so E2
  must "beat/help ψ", not "enable" zero-shot.

OPS for next session: `tmux ls` FIRST (sotagap may still be draining). E1b was dropped
(E1a made it moot). See [[zeroshot-gciql-e1a-works]], [[audit-sf-measure-20260706]].
