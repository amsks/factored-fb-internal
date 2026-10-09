---
name: spec30-review-response-20260816
description: "2026-08-16 method review AUDITED 3-ways + spec 30 WRITTEN — TD InfoNCE equivalence CONFIRMED (hard mode; flagship is SOFT); ★ geometry RE-MEASURED: 'rank-1 collapse' was the softmax-GAUGE MEAN (33-103× signal; centered Cov eff_rank 6-21/50; centering alone rescues the point index); trained R-A encoder PROVABLY reward-shift-unstable (cos≈0 under r+1, centered encoder exactly invariant); .812 used the Dirac goal-code fallback"
metadata: 
  node_type: memory
  type: project
  originSessionId: 973481bf-5a6b-4ead-8080-c40f063e1e9d
  modified: 2026-08-16T22:24:37.107Z
---

Spec: `PAPER/Multistep-FB/specs/30-review-response.md`. ⚠ **REVISES the
mechanism WORDING of [[spec28-p0-z-canonicalization-20260814]] /
[[spec29-ra-dual-index-20260814]]**: all spec-28 measurements stand, but
"B collapses onto one direction" overstates — B→B+b is exact softmax gauge,
‖μ_B‖ = 33–103× the centered signal (z25 1087.8 vs 10.6), centered Cov(B)
eff_rank **6.3–20.8 / 50** vs uncentered 1.000–1.002. Centering alone
rescues the point index (pairwise cos .9995→.001–.004, AUC .427–.858→
.990–1.000, no in-sample inflation on disjoint split); ridge adds PR
(28–30 vs 6.6–19.8). Artifacts `results/spec30_geom/` + uncommitted script
`scripts/rescuing_fb/eval_spec30_geom.py`.

**Shift instability (severe, review §4.5):** the TRAINED R-A uncentered
encoder: cos(enc(kg), enc(kg+1)) = .035/−.000/−.044 — a policy-preserving
shift near-orthogonalizes the index. Centered encoder
(C+λI)⁻¹E[(B−μ)(r−r̄)] reads 1.000000 exactly (invariant to B+b, r+β, αr).
⇒ P1 = `dual_centered` encode mode + centered-POINT goal-code arm; ship
centered if ≥ uncentered−noise; seeds/cd only on the winner.

**TD InfoNCE (arXiv 2310.20141, ICLR 2024) — equivalence CONFIRMED
update-for-update** (their Eq. 10 = our hard categorical target; appendix
proves classifier = normalized discounted future-state distribution).
Residual delta verified: no task latent z, no reward→task inference, no
offline zero-shot regime, ℓ2-normalized towers. ⚠ Flagship cells are SOFT
(`measure_density_hard: False` incl. R-A screen) — hard is the ablation.
Positioning = cite-and-extend; exact TD InfoNCE arm ~80–120 lines.

**Other confirmed code facts:** plain FB's B sees FULL obs vs our φ(s′)
(dwell narrative describes φ-space, not the failing baseline); query
support = batch's own next states ⇒ dwell states are negatives in
proportion BY CONSTRUCTION (review's double-count point); .812 eval used
Dirac fallback ẑ∝(Σ̂+λI)⁻¹B̄(φ(g)) — `density_z_from_reward_samples` is
built+tested+trained-through but NEVER rolled out (= G29-3); kernels
pool-normalized (base measure = kernel×pool density); zero-shot de-biasing
has no exact tabular test; hard-mode duplicate-query false negatives
structurally real, uninstrumented.

**Review's own errors (rebuttal notes, CORRECTED by Rev-1 verification):**
SF2 = online sample-efficiency NOT zero-shot; DVFB = online unsupervised
NOT offline. ⚠ RETRACTED: the Finer-BFM E[rB]-vs-(E[φφᵀ])⁻¹E[rφ] contrast
IS in their §3.1 main text (our first fetch missed it) ⇒ covariance-inverse
task regression is established prior art; our novelty = softmax-gauge ×
task-inference interaction only. ⚠ CORRECTED: TD InfoNCE DOES do offline
GC-RL (D4RL AntMaze + GC-BC MSE regularizer App D.3; actor = contrastive
CE, Eq 11) — the delta is no reward-prompted latent family / no reward→
index inference / no validated arbitrary-reward control; its critic uses
ONE shared negative pool, terms differ only in positive-slot support.
NEW must-cite: **OpTI-BFM** (2510.20264, ICLR 2026 ORAL) — BFM task
inference as UNCENTERED reward regression, online feedback; our delta =
gauge + centered form. Real must-engage: TD InfoNCE, TD-JEPA (ICLR26 oral),
ZSOPL (2602.01962). Others real (TD Flows ICML25, Bellman Diffusion
2407.12163, InFOM ICLR26, Soft FB 2602.06769, One-Step FB 2602.11399).

**Rev-1 (reviewer assessment of spec30, all verified):** centered encoder
has NO M=1/Dirac limit (r₁−r̄=0 ⇒ m̂ᶜ=0) ⇒ point-prompt API (analytic
centered point prompt, never "Dirac M=1 limit") SPLIT from reward-sample
API (M≥256) + constant-reward fallback; encoder is TRANSLATION-gauge
invariant only (full linear gauge B→AB not fixed — ridge+sphere norm not
A-invariant); population exact / EMA approximate (same-schedule moments,
symmetrize, eig floor, one pinned base measure = batch next-state
marginal); SOFT variant = kernel-smoothed occupancy ν_K, integrates
K_ρ,ε r NOT r ⇒ hard mode is the exact interface for G29-3's sharp/max/
avoid families; P2.1 = TWO TD InfoNCE controls (paper-exact + matched-
extractor); P2.2 renamed goal-conditioned outcome-space LSIF (raw-g);
2×2 → 3×2 index factor {unc point, cent point, cent dual}; G29-3 adds
search-over-z oracle (encoder vs policy-family-coverage separation) and
FRE becomes mandatory if arbitrary-reward is headline; first GPU wave
parallel: P1-centered cs + paper-exact TDInfoNCE cd + LSIF cd.

**Spec 30 priority:** P5 paper edits (TD InfoNCE positioning, gauge-mean
restatement + invariance proposition, Dirac-fallback sentence, Option-B
dwell wording, scope sentence) → P1 centered screen → P2 exact TD InfoNCE
+ same-φ-FB-external-readout (privileged-φ confound killer) → seeded 2×2
{uncentered-point, centered-dual}×{latent, kernel} → G29-3-centered with
shifted rewards + M-curves. Not adopted: 5-seeds-now (3 now, 5 camera),
SF2/DVFB/InFOM as quantitative baselines, MC action-ranking as blocker.
