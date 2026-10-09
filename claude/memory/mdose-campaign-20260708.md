---
name: mdose-campaign-20260708
description: "mdose campaign DONE + COLLECTED + RE-PROBED (2026-07-08 ~16:30): WHY-ANSWER COMPLETE — (1) antmaze measure-only-ddpgbc 0.79 vs cube 9.9 = occupancy/domain mechanism confirmed; (2) sgphi cliff λ1→3 (succ .85→.15, critic aucE .81→.73, value-leg still healthy at λ3 = damage starts in critic ψ); (3) shared λ=10 fully absorbed (.92 succ, .96 aucE) = φ co-adaptation masks conflict at ≥10× dose; C5 zmix rescue REFUTED (.07/.165=D1). Full note: PAPER/fb-gciql-bridge/mdose-results-20260708.md. REMAINING: antmaze s0 straggler, tex integration (tex owned by parallel session)."
metadata: 
  node_type: memory
  type: project
  originSessionId: 942da339-fbce-4c5a-84ad-c7933b2fc811
---

**HANDOFF 2026-07-08 ~04:15 UTC.** The bridge-track session answering: **why does a Bellman/TD
constraint on the representation (successor measure) mess up bilinear GCIQL (0.90 scalar-TD)?**
Follows [[e2-psi-crosseval-conflict-finding]] (the ψ cross-eval that flipped E2's "inert" into
"conflict masked by scalar TD"). HIQL/sg10/antmaze-bc03 work = the OTHER (SOTA) track, not this.

## RESULTS (collected 2026-07-08 ~15:00-16:30 UTC) — full note PAPER/fb-gciql-bridge/mdose-results-20260708.md
succ/cos/aucE/auc0 (critic legs): base .90/.883/.912/.955 · sgphi λ3 .15/.92*/.727/.746 ·
sgphi λ10 .02 (1 q-viol) /.786/.646/.636 · sgphi λ30 .00 (3 q-viol, UNINTERPRETABLE) ·
shared λ10 .92/.931/.962/.960 · C5a .069 · C5b .165 (=D1) · antmaze measureonly-ddpgbc .79 (2 seeds; s0 pending).
*cos rises at λ3 while AUC collapses = ψ subspace collapse; AUC is load-bearing. Value-leg control
healthy at λ3 (.952) → damage ORIGINATES in critic ψ, propagates via Q-target chain by λ10.
Answer: (1) not intrinsic — antmaze 0.79 = occupancy/domain mechanism; (2) cube = ψ fixed-point
geometry conflict, dose-responsive; (3) scalar TD masks it via φ co-adaptation, ≥10× margin.
Probe data analysis/e2_psi_probe/mdose15.json. TEX INTEGRATION still to do (tex currently owned
by the parallel SOTA session — do it in a dedicated pass, source = the results note).

## Why-question scoreboard (what's answered vs open)
- NOT scale: E2 guardrail (q_mean band held, 15/15) — old 18/18 crashes were ⟨F,z⟩ inflation.
- NOT structure: B1 fbz-IQL 85 / two-tower 86 / B4 87 on the 90-critic.
- IS a ψ-geometry conflict: stopgrad-φ cells degrade dose-responsively (cos .893→.815,
  aucE .921→.810 @λ=1, all 6 sgphi seeds below base range); shared legs absorb it (φ compensates).
  Measure never HELPS ψ at any dose. Side datum: moment-z out-ranks exact ψ(g) (auc .955 vs .921).
- OPEN (this campaign): how deep the conflict goes (dose curve), whether it's occupancy-driven
  (antmaze flip?), whether FB's mixed z-sampling rescues the measure-only cells toward torch-FB 48.

## RUNNING: tmux `mdose` (launched 04:00 UTC; monitor armed in the old session, re-arm if resuming)
21 runs = 7 cells × 3 seeds @500k. Launcher `scripts/launch_measure_dose_antmaze.sh` (committed),
logs `/dev/shm/mdose_logs/*.log`, dispatch `/dev/shm/mdose_dispatch.log` ("ALL DONE" when the
wait returns). At 04:10: 9 runs stepping (antmaze ~110 it/s ETA ~1.5h; cube 36-65 it/s ETA 2.5-3.5h),
12 still in step-0 eval behind the parallel campaign (E2-like, self-resolves). Overall ETA ~6-9h.
| cell | wandb group | reads against |
|---|---|---|
| sgphi λ=3/10/30 | GCIQL-State-Flow-Bilinear-d50-MeasureMix-l{3,10,30}-sgphi | dose curve past λ=1; where does ψ/readout break completely? |
| shared λ=10 | ...MeasureMix-l10 | shared legs absorbed λ≤1 (succ .85–.93 vs base .90); absorption limit? |
| antmaze measure-only ddpgbc | GCIQL-AntMaze-Medium-Bilinear-d50-MeasureOnly-DDPGBC | cube MeasureOnly 9.9 / antmaze IQL-ddpgbc 0.88 anchors — occupancy (arm-dominated ρ) hypothesis |
| C5a zmix | ...Bilinear-d50-C5-Zmix | C4 stack + measure_z_random_frac 0.5 (z-coverage alone; actor_cond=goal) |
| C5b zmix+actorz | ...Bilinear-d50-C5-Zmix-ActorZ | C5a + actor_cond=z + actor_z_from_critic (z-consistent bootstrap a+~π(s+,z_i)); target torch-FB 48 |
Guardrail: critic/q_mean ∈ [−100,10] full-history. grad_norm is DOSE-LOADED in sgphi arms
(λ=30 smoke: gnorm 4.7k with healthy q_mean −37) — q-band only, do not disqualify on gnorm.

## Code state — ALL bridge-track code COMMITTED on gciql-fb-bridge (push not verified)
- `9d87bc2` scripts/e2_measure_psi_probe.py — ψ cross-eval probe: dataset-moment z=Σr·ψ(s) +
  cos + rank-AUC on BOTH legs. **Gotcha it fixed: E2 ran measure_legs='critic'; the old
  infer_ablation_gciql.py probes the VALUE net = wrong leg.** Results: analysis/e2_psi_probe/all18.json.
- `f10356e` measure_loss ddpgbc next-action fix (distrax distribution crashed target_critic concat;
  mirrors sf_loss pattern; flow path bit-exact) + 4 dose configs + antmaze measureonly config.
  Test: test_gciql_measure_scale.py::test_measure_ddpgbc_actor_next_action_trains_finite.
- `5e8c912` measure_z_random_frac (metamotivo sample_mixed_z: per-sample √d-sphere random z,
  same z both TD sides, z-consistent bootstrap under actor_cond=z) + GCFBValue z_vec kwarg
  (None ⇒ bit-exact) + C5a/C5b configs + launcher. Tests: tests/test_gciql_measure_zmix.py (3).
- Termination masking (audit caveat #2) traced and DROPPED: tM is stop-gradded ⇒ absorbing-goal
  masking only shifts the diagonal term by a gradient-constant; off-diag absorption measure-zero.
- KNOWN residual infidelity in C5b: actor loss still trains only at z=ψ(actor_goals); actor-side
  z-mix = follow-up knob if C5 moves the needle.
- Dirty working tree (tex, scripts/probes/phase_probe.py, sg10 configs) = the PARALLEL SOTA
  session's (its commits 0a275fd, 16c0355 interleave ours) — do not commit/revert those here.

## NEXT SESSION (in order)
1. `tmux ls` + progress: `for f in /dev/shm/mdose_logs/*.log; do tr '\r' '\n' <$f|grep -oE '[0-9]+/500000'|tail -1; done`
2. Collect success @500k (eval.csv per run under /dev/shm/gciql_outputs/factored-fb/<group>/, or wandb
   amsks/factored-fb by group, metric evaluation/overall_success, mean/3 seeds). Verify q-band guardrail
   full-history per run (train.csv training/critic/q_mean).
3. Re-probe ψ geometry on the new CUBE cells + 3 base seeds:
   `XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl /dev/shm/.venv-jax/bin/python scripts/e2_measure_psi_probe.py \
    --checkpoint-dirs <base sd000-002 + all 18 cube-cell run dirs> --checkpoint-step 500000 \
    --value-latent-dim 50 --out analysis/e2_psi_probe/mdose.json`
   (works for C5 cells too — GCFBValue is signature-compatible... CHECK: C5 has fb_forward_z=true ⇒
   load_agent must be given the right arch; if restore fails, extend load_agent cfg with
   fb_forward_z/project_z from the run's flags.json.)
4. Decision tree: antmaze measure-only high (≳0.5) ⇒ occupancy mechanism real (domain-flip);
   ≈cube-low ⇒ intrinsic objective conflict. C5 ≥ ~40 ⇒ fidelity gap was z-coverage, bridge closes
   from the measure side, residual 48→90 = objective — the paper punchline; C5 flat ⇒ z-coverage
   refuted, conflict story stands alone. sgphi dose curve + l10-shared ⇒ quantify the tug-of-war.
5. Synthesize into PAPER/fb-gciql-bridge/fb-gciql-bridge.tex (build: ~/.local/bin/tectonic) +
   update [[e2-psi-crosseval-conflict-finding]] and this file.

[[e2-psi-crosseval-conflict-finding]] [[e2-measure-mix-campaign]] [[audit-sf-measure-20260706]]
[[zeroshot-gciql-e1a-works]] [[always-run-in-tmux]] [[wandb-entity-amsks]]
[[host-identity-you-are-aws-gpu]] [[jax-gpu-packing-preallocate]] [[factored-fb-test-env]]
