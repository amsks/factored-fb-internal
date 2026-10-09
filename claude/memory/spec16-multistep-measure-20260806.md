---
name: spec16-multistep-measure-20260806
description: "2026-08-06 spec16 IMPLEMENTED (UNCOMMITTED, tests green 6 new + 42 regression): multi-step successor-measure TD in jax gciql_flow -- measure_nstep=K (k-step window) + measure_nstep_lambda (sampled lambda-return), k=1 bit-identical to one-step, GCDataset supplies mstep_next_obs/mstep_k; derivation note PAPER/Multistep-FB/main.tex COMPILED (ICLR style); wave M1 pre-registered in spec, NOT launched"
metadata: 
  node_type: memory
  type: project
  originSessionId: 1ba5ea79-da0c-4583-a89e-0a11cfad5238
  modified: 2026-08-06T01:05:30.330Z
---

Bridge track, follows [[spec15-play-data-rsm-20260805]]. User direction chain:
RSM factored head rejected ("not using the successor measure") -> event-time
TD rejected ("no preprocessing") -> FINAL DESIGN = successor measure itself,
learned by k-step/lambda-return TD over raw trajectory windows + (stage 2,
NOT implemented) dueling D(g)*H head.

**Derivation** (`PAPER/Multistep-FB/main.tex`, compiles with tectonic, ICLR
2027 style copied from FB-Manipulation-ICLR): k-step operator is a gamma^k
contraction with k-invariant fixed point; error = gamma^k-shrunk bootstrap +
off-policy bias; DWELL-INVARIANCE LEMMA: bias <= E[gamma^T_int] * window mass
because object coords are frozen until the first interaction => play data
licenses long horizons exactly where one-step TD collapses (cd T_int~176 =>
gamma^T_int~0.17). Useful k in [median dwell, ~2x discount half-life] =
[50,150] at gamma=.99; geometric k = lambda-return, modulus
gamma(1-lambda)/(1-lambda*gamma). One-step collapse: T1 C ~ pbar + gamma*C
=> C* = dataset-average mass = the 9.16 dwell constant.

**Implementation (UNCOMMITTED):**
- `tools/wandb_mode_shim/gciql_flow.py`: knobs measure_nstep=0 /
  measure_nstep_lambda=0.0 in get_config; spec16 assert-lattice (requires
  measure_first_passage incl. ball-visitation via measure_fp_absorbing=False;
  refuses expectile/nhat_head/fb_forward_z/eps_multiscale/z_random_frac/
  encoder); static `_nstep_window_terms` (O, cont from [K,B,B] window hits,
  absorbing + visitation) + `_measure_terms_nstep` (diff = M - O -
  gamma^{k_i}*cont*tM_b, squared diag, scale_norm msq/msq); new branch in
  measure_loss fp path -- bootstrap at s_b = window[k_i-1], SAME rng noise as
  the one-step a_next so K=1 is bit-exact; info key measure_nstep_k_mean.
- `third_party/ogbench/impls/utils/datasets.py` GCDataset.sample: when
  measure_nstep=K>0 adds mstep_next_obs [B,K,obs] (= next_observations[idx+u-1]
  clamped at the trajectory final row; u=1 == batch nobs exactly) + mstep_k
  (min(draw, K, steps-to-final+1)); geometric draws when lambda>0. K=0 adds
  nothing (other agents unaffected).
- `tests/test_gciql_measure_nstep.py`: 6 tests -- default-off ignores fields,
  K=1 bit-identical (absorbing x visitation x tpd batch/hindsight), numpy
  reference for window terms, GCDataset shapes/clamping/geometric, lattice
  refusals, end-to-end K=3 with k_mean. ALL GREEN + 42 neighboring measure
  tests green (hit_spec, axes, scale, phase2_occupancy, campaign_v2).

**Wave M1 pre-registered (spec 16 §6), NOT LAUNCHED:** cube-single m1_base
(visitation from scratch = known-bad anchor) / m1_k64 / m1_lam (K=256,
lam=.99) / m1_fp_k64 (working phase-1 arm + nstep, must hold ~0.9+).
PRIMARY: multistep visitation must NOT pin q_abs to the dwell constant and
must beat the from-scratch floor. k-flatness ablation {16,64,256} = mechanism
test. M2 (cd/scene hindsight vs 9.16 anchors .53/.93) only if M1 passes.
Stage 2 = dueling dwell head, separate pre-registration.

Related: [[spec15-play-data-rsm-20260805]] (empirics + design chain),
[[p2-multienv-decay-mechanism-20260730]] (the collapse this fixes).
