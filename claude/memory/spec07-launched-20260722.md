---
name: spec07-launched-20260722
description: "Spec 07 (capability on FP-FB + renewal measure) written, implemented via workflow, ALL TESTS GREEN, campaigns LAUNCHED in tmux s07q (training) + s07r (renewal/eval)"
metadata: 
  node_type: memory
  type: project
  originSessionId: c8ce34ba-2682-4d2a-b4a3-6c479c9b6bf7
  modified: 2026-07-22T19:44:09.553Z
---

2026-07-22 (bridge track). Follow-on to [[firstpassage-measure-works-20260722]].

**Spec:** `PAPER/rescuing-fb/specs/07-fp-capability-and-renewal-measure.md` — capability
(reward semantics) on top of the FP-NoOrtho 0.967 win + renewal-factorization successor
measure (`M = Q_FP·R(g)`, R action-free). User decisions: target=capability; the 0.967
independent-eval verification gate is SEPARATE (not yet run — all reads provisional).

**Implemented (parallel workflow, 3 agents + verify; all green):**
- `task_prior_actor`/`task_prior_critic_mix` knobs in `tools/wandb_mode_shim/gciql_flow.py`
  (actor-only z-prior: Dirac/mixture/sphere 0.6/0.2/0.2, fully sg, knob-off bitwise identical);
  tests `tests/test_gciql_fp_taskprior_actor.py` 6/6.
- `scripts/rescuing_fb/renewal_head.py` (MC renewal + FP_mu + fidelity eval + infer_z);
  tests 15/15. ⚠ FP_mu censoring fraction is high — restrict M^mu claims accordingly.
- `scripts/rescuing_fb/eval_zeroshot_rewards.py` (6 task families ±GPI, renewal-vs-plain
  z-weighting); tests 18/18. Regressions (measure_axes, rgfb_v1, rgfb) all pass.
- NOT committed.

**⚠ Key discovery:** the 0.967 FP-NoOrtho cell runs `actor_cond='goal'` (raw goal obs), NOT
z. So (a) TPActor forces actor_cond=z ⇒ item-1 is a TWO-difference cell vs the anchor;
(b) inference-only eval on frozen ckpts routes z_r through a goal-proxy decode (weaker
claim). True z-native capability = the TPActor retrain.

**Launched 2026-07-22 ~09:27 (tmux):**
- `s07q`: tpactor s0/s1/s2 (GPU 2/3/4) + criticmix s0 (GPU 7), 500k, out
  `/home/adityamohan/exp_spec07`, logs `/dev/shm/spec07_logs/`. Groups
  `GCIQL-State-Flow-Measure-FP-NoOrtho-TPActor-d50` / `...-TPActorCriticMix-d50`.
- `s07r`: `spec07_renewal_driver.sh` CPU-only — renewal fits (psi+cube, +fp-mu, +eval) on
  the frozen FP-NoOrtho s0–s2 ckpts, then inference-only zero-shot (renewal vs plain, +GPI).
  Outputs `analysis/rescuing-fb/renewal_*.npz|json`, `zeroshot_infonly_s*_{renewal,plain}.json`.
- v1-taskprior comparison arm ALREADY DONE (3 seeds, prev queue):
  `/home/adityamohan/exp_queue/factored-fb/RGFB-Cube-Single-State-d50-v1-taskprior/`.

**Pre-registered reads** (in spec + config headers): Dirac ≥0.90 (retry probs 0.8/0.1/0.1
once); mixture no-GPI ≥0.60; +GPI ≥0.75; regions ≥0.672; dense within 5pts of Dirac;
criticmix PREDICTED to hurt Dirac. Stopping rule: if inference-only already clears the bars,
capability needs no new training.

**After collection:** results_map.py + CSV refresh ([[results-csv-pipeline]]), DECISIONS.md
entry, and the separate 0.967 verification gate before any claim.

## RESULTS (collected 2026-07-22 ~18:20 — runs finished in ~9h)

- **TPActor DIRAC BAND FAILED, decisively:** final 0.560/0.292/0.332 (mean 0.395) vs anchor
  0.967. Curves NEVER reach the anchor (0.41–0.57 from the first 50k eval, then oscillate
  0.3–0.6) ⇒ NOT slow coupled degradation — the forced z-actor form (actor_cond='z' +
  actor_z_from_critic, mutated at create; flags.json still shows 'goal' pre-mutation) is bad
  from the start. ⚠ TWO-difference cell: z-conditioning AND z-prior — the single-difference
  control (FP-NoOrtho + z-actor + Dirac-only z) is NOT yet run and decides attribution
  before the pre-registered 0.8/0.1/0.1 retry is worth anything.
- **CriticMix (1 seed) — the surprise:** tracked the ANCHOR early (0.84@50k, 0.96@100k,
  0.90–0.96 through 250k) then COLLAPSED to 0.33–0.67. Two readings: (i) the late collapse
  = the pre-registered linearity-conflict prediction, slow-acting; (ii) the early phase
  shows off-Dirac critic grounding initially makes the z-readout USABLE by the actor
  (reshapes the phi/psi span) — the z-actor CAN hit 0.96 in this cell. Both matter.
- **First s07r driver DIED in 2 min:** repo .venv has no jax (the e3 gotcha, again) + it
  re-downloaded the dataset. Fixed (PY=/dev/shm/.venv-jax/bin/python), relaunched as tmux
  `s07r2` 18:23 — renewal fits + inference-only zero-shot pending.
- results_map.py updated with both groups; CSV refresh backgrounded.
- **Next decisions:** (1) single-difference z-actor control before any retry; (2) criticmix
  needs 2 more seeds + train.csv forensics on the 300k collapse (what moved: measure loss,
  q scale, z norms?); (3) capability may have to route through inference-only + goal-proxy
  (+GPI) if z-native actors are dead in the FP corner.

## BULLISH-SM PIVOT (user directive 2026-07-22 evening) + WAVE 2 LAUNCHED

**User: aggressively pursue the successor measure; fix what's not working FOR the SM.**

Three findings that re-aim the program:
1. **ORTHO CONFOUND ACROSS THE WHOLE OCCUPANCY DIAL:** every floored measure cell
   (offdiag sweep w=0.1–0.5 all 0.02–0.08, GCIQLCorner 0.03, Expectile-FP 0.04) ran
   `measure_ortho_coef=1.0` — the proven toxin (FP 2×2: 0.967 no-ortho vs 0.225 ortho).
   The dial has NEVER been run deconfounded. (Checked the z-actor confound theory first —
   dead: all floored cells run actor_cond='goal' like the winner.)
2. **CellW0 IDENTITY ANOMALY:** cellw0-noortho ran with hindsight columns + target_min
   (identity conditions apparently met) yet scores 0.17 vs the theorem's ≡GC-IQL 0.904.
   Prime suspect: delta_M=(1-gamma)delta_Q ⇒ 100× smaller gradient scale at gamma=.99
   (run had measure_scale_norm=False). Forensics agent dispatched.
3. **RENEWAL FACTORIZATION VALIDATED (s0):** FP×R fidelity Spearman ≈0.51 overall,
   **uniform 0.60–0.63 across ALL horizon buckets incl. K>20**; FP-alone ≈0.06 with wild
   calib slopes ⇒ R carries ALL the occupancy structure, exactly as the factorization
   claims. FP_mu×R ≈0.51 too. Position-only R fits 13× worse than ψ-input R (cube mse 0.93
   vs psi 0.069) ⇒ R is NOT a pure cube-position functional.

**⚠ FORENSICS (CellW0 anomaly, agent report 2026-07-22): the exm (expectile-measure)
branch of gciql_flow.py BYPASSED `_measure_terms` ⇒ `measure_scale_norm` was silently a
NO-OP there; without ortho nothing pins the bilinear scale ⇒ divergence (q_abs→1e4-4e5) on
2/3 seeds; the bounded seed plateaus ~0.5 partly because the actor's q/sg|q| normalization
is not affine-invariant (M=1+(1-γ)Q offset dilutes the greedy signal ~3.8×). The ENTIRE
offdiag sweep = {diverged}∪{ortho-crushed}; no w was ever evaluated healthy. FIX APPLIED
(one hunk: normalize Lm/Lv by sg[mean(M_act²)] under measure_scale_norm; tests 6/6 green;
actor centering fix NOT yet applied — candidate follow-up). Cell W identity test only
checks algebra on random matrices + 2-step finiteness, not training dynamics.**

**Wave 2 (CORRECTED, tmux `s07w2b` exm cells + `s07w2c` criticmix/zctl; first launch
killed after forensics):** 8 exm jobs now run with `measure_scale_norm=true` override:
corner-NoOrtho ×3, Expectile-FP-NoOrtho ×2, CellW030-NoOrtho ×2 + CellW010-NoOrtho ×1
(the deconfounded dial), criticmix s1/s2, zctl (FP-NoOrtho + D1 z-actor, no prior) ×2.
Groups: `...-GCIQLCorner-NoOrtho-d50`, `...-Expectile-FP-NoOrtho-d50`,
`...-CellW030-NoOrtho-d50`, `...-CellW010-NoOrtho-d50`, `...-FP-NoOrtho-ZActor-d50`.
**tmux `s07cm200`**: criticmix_s0@200k zero-shot eval chained behind s07r2.
Inference-only zero-shot bars (s0/s1): dirac≥0.90 FALSE (bimodal per-task, goal-proxy
fragile), mixture+GPI 0.75 TRUE on 2/4 passes, regions FALSE, linear ≈ random control.
