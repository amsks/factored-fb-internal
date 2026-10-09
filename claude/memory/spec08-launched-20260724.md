---
name: spec08-launched-20260724
description: "Spec 08 (next experiments for the two-phase FB curriculum) — derivation merged into the note; E1/E2/E6 training + E3/E5 done; 3 new knobs + configs + launchers, all TDD-green, UNCOMMITTED"
metadata: 
  node_type: memory
  type: project
  originSessionId: d725d6db-21fb-4621-abc8-3b88a6e6a6b5
  modified: 2026-07-24T08:48:03.063Z
---

2026-07-24 (bridge track). Follow-on to [[wave2-collected-20260723]] (ball-visitation
0.952/0.928, (z,x) arm 0.904/0.960, dense 0.90/0.13 split) and
[[firstpassage-measure-works-20260722]]. User asked to (a) merge the `fb-to-2phase.tex`
derivation into `two-phase-successor-measure-note.tex` and (b) spec + implement the next
experiments. Then picked E3+E5+E6 to run now. Autonomous, long session.

**Docs (PAPER/ = untracked):**
- Note gained **§4 "From density-ratio TD to finite-region regression"** (Dirac density-ratio
  → un-anchored linear term → kernel substitution → separated-form). Compiles clean (tectonic).
- **`PAPER/rescuing-fb/specs/08-two-phase-next-experiments.md`** = 7 experiments E1–E7, fully
  grounded (knobs/configs/launch/eval), with a live Status section. READ THIS FIRST for detail.

**New code (all default-off bit-identical, TDD red→green):**
- `freeze_actor` (E2) in `gciql_flow.py` create(): optax.multi_transform masks
  `modules_actor`+`modules_actor_vf` to set_to_zero (mirror of freeze_critic_ckpt).
- `measure_z_random_frac_anneal` (E6) in `gciql_flow.py` measure_loss: ramps the bernoulli
  random-z prob 0→`measure_z_random_frac` over `anneal_ramp` (default 300k); branch stays
  static on the config max (jit-safe), only the prob is traced from `self.network.step`.
- `ridge_z` + `--z-inference ridge --ridge-lambda` (E3) in `eval_zeroshot_rewards.py`
  (GLS/whitened moment `(Sigma+lam I)^-1 b`, lam=ridge*tr(Sigma)/d; mirrors
  `evals/z_inference.infer_l1`).
- `scripts/rescuing_fb/r_action_independence.py` (E5): loads FP-Q + BallVisit-M critics
  (shared frozen ψ), computes ICC of R=M/Q given g, argmax_a Q vs M agreement, rank corr.
- **Tests green:** `tests/test_gciql_phase2_occupancy.py` 9 (7 + 2 anneal),
  `tests/test_eval_zeroshot_rewards.py` 21, `tests/test_gciql_measure_axes.py` 6.
- New configs: `cube_single_state_measure_ballvisit_zx_d50.yaml` (E1),
  `..._ballvisit_frozenactor_d50.yaml` (E2), `..._measure_fp_zx_d50.yaml` (E6).
- New launchers: `launch_spec08_e1.sh`, `_e2.sh`, `_e6.sh` (ARM=zfrac0|anneal),
  `run_spec08_e3_ridge.sh`; E5 driver at `/dev/shm/spec08_logs/e5_driver.sh`.
- **NOTHING COMMITTED** (branch shared with the other track; user chose to defer commit).

**LAUNCHED 2026-07-24 (tmux, out `/home/adityamohan/exp_spec08`, logs `/dev/shm/spec08_logs/`):**
- `s08e1` **E1** headline: phase-2 ball-visitation on the (z,x) form, `visz_s0/s1`,
  warm-start from `exp_spec07/.../FP-ZX-d50/zx_s{0,1}`. Recipe = ZX form (fb_forward_z,
  actor_cond=z, actor_z_from_critic, measure_z_random_frac=0.5) + measure_fp_absorbing=false
  + measure_stopgrad_psi=true + actor_q_center=true. Step-0 eval 0.90/0.936 = ZX anchor ✓.
  Group `...-BallVisit-ZX-d50`. **Payoff = direct-z region/linear eval when ckpts land.**
- `s08e2` **E2** frozen-B/frozen-actor `faz_s0/s1`, warm-start from FP-NoOrtho
  `exp_spec06/.../m_firstpassage_noortho_s{0,1}`; diagonal ball-visit + freeze_actor=true.
  Step-0 eval 0.964 = FP anchor ✓. Group `...-BallVisit-FrozenActor-d50`. The MISSING cell of
  the 2×3 (frozen/updated=ballvisit 0.952/0.928; updated/updated=occtr collapsed).
- `s08e6zf0` **E6** goal-only `zf0_s0/s1`, cold start (step-0 eval 0.0 expected).
  Group `...-FP-ZX-zfrac0-d50`. The 0.5 arm = reuse existing `zx_s0/s1`.
- `s08e6an` **E6** annealed `zfan_s0/s1`, WAITING at the MAX=6 gate (pipelines in when
  E1/E2/zf0 finish). Group `...-FP-ZX-zfracAnneal-d50`.
- `s08e3` **E3** forensic (dense, moment + ridge λ∈{1e-3,1e-2,1e-1}, ZX s0/s1) → RUNNING,
  writes `analysis/rescuing-fb/e3_dense_s{0,1}_{moment,ridge_lam*}.json`. ~done soon.

**RESULT IN — E5 (both seeds, `analysis/rescuing-fb/e5_action_indep_s{0,1}.json`):**
argmax_a Q vs M agreement **0.30/0.30**, R=M/Q **ICC-given-g 0.24/0.29**, within-goal CV
**1.3/1.8** ⇒ **strict pointwise action-independence of R is moderately REFUTED** (matches
§3's hedge that recurrence carries control info the split discards). ⚠ Computed at RANDOM
actions (Q near-floor ⇒ M/Q noise-inflated); NOT a contradiction of the renewal magnitude
claim (goal-averaged R(g), Spearman 0.60, is a smoother estimand). Follow-up: rerun E5 on
actor-proposed / high-Q actions to disentangle the noise.

**PENDING:** E3 numbers (imminent); E1/E2/E6 finish ~9h → E1 direct-z region/linear payoff,
E2 frozen/frozen verdict (isolates the policy–measure loop), E6 coverage comparison; **E4**
(kernel normalization for ∫r dM — region/linear — needs E1's phase-2 ckpts); **E7** (scene/
pixels, deferred; ⚠ eps=0.4 is HARDCODED in 3 places in gciql_flow.py, not a knob).
results_map.py + CSV refresh for spec08 groups NOT done ([[results-csv-pipeline]]); DECISIONS.md
no spec08 entry; COMMIT pending.

**How to apply:** the two-phase note is the paper's positive result; spec 08 is the program to
finish it. E1 (separated-form phase-2 + non-goal-reducible extraction) is the headline test —
watch its direct-z region/linear numbers. E5 already gives one reportable (negative-ish)
finding for §3.
