# Spec 25 — Porting FB's index-coupling mechanisms to the z-indexed density measure

Status: WRITTEN 2026-08-13. Parents: spec 24 + its audit (implementation
faithful; z-measure trains healthily but is index-INSENSITIVE: flat
candidate-score spread, E2a≈E2b≈.03 vs g anchors .549/.488/.177 on the
fixed v2 harness with a bit-exact control) + the FB/Meta-Motivo protocol
comparison (2026-08-13): plain FB's z works on ExoRL not because of the
z-sampling mixture (spec 24 copied that) but because of three coupling
mechanisms spec 24 did NOT port. This spec ports estimator-compatible
analogs of all three and retests. TDD; every knob default-off,
bit-identical to the spec-24 arm.

## 0. The three mechanisms and their ports

FB / Meta Motivo couple index to behavior three ways; the density
estimator got none of them in round 1:

**C1 — structural value sensitivity.** FB's task value IS F(s,a,z)ᵀz: the
model cannot ignore its index. Ours enters z only as an input feature the
softmax may ignore (and did: score spread .00086 max vs .00068 mean).
PORT: a structural logit tilt,

    ℓ_ij = F(s_i,a_i,z_i)ᵀB(ũ_j)/τ + β · z_iᵀB(ũ_j)/√d      (β fixed, not learned)

so w(s,a,z) is an exponential-family tilt of the base scores and is
z-sensitive by construction at every step, including step 0 — the
ignore-z fixed point stops being reachable. The CE/bootstrap still fits
π_z's occupancy; F(s,a,z) can reshape around the tilt but not delete it.
Knob `measure_density_z_tilt` (β, default 0.0 = spec-24 behavior;
round-1 dose β=1.0, one value, no sweep).

**C2 — a shaped B whose linear functionals form a reward family.** FB
gets this from orthonormalization; ortho is convicted on manipulation for
the scalar estimator and stays refused. ORTHO-FREE PORT: a supervised
reward-decodability auxiliary. Per batch, draw one synthetic kernel
reward r_c(u) = exp(−½‖(u−c)/ε‖²) at a random dataset center c, solve the
closed-form ridge v* = (B_UᵀB_U + λI)⁻¹B_Uᵀ r_c(U) on the batch queries
(differentiable through B), and add

    L_dec = coef · ‖B_U v* − r_c(U)‖² / n

Knob `measure_density_z_rdecode_coef` (default 0.0; round-1 dose 1.0,
λ=1e−4). This shapes span(B) to support exactly what E2a's
z_r = E[r·B] and the tilt's zᵀB need. (This is the training-time twin of
the P0 probe below.)

**C3 — index-consistent actor supervision.** FB trains its actor across
the whole z-manifold against the structural value; Meta Motivo adds a
per-z discriminator. Spec 24's actor instead had its imitation term
diluted (50% of rows clone toward random-sphere z's — a mapping no data
supports) and its index encoding drifts with target-B. TWO PORTS:
- `measure_density_z_bc_hindsight` (bool, default False): the flow-BC
  term conditions ONLY on hindsight z's (z of the row's own future
  coords — reachable and self-consistent, exactly what the g-recipe's
  actor_p_trajgoal=1.0 does implicitly); random-sphere z's still appear
  in the measure loss and the actor's q term (the manifold-coverage FB
  gets), just never in the imitation term.
- `measure_density_z_enc_tau` (float, default 0.0 = encode with target-B
  as in spec 24): when >0, z's are encoded by a separate slow EMA copy of
  B (τ_z, round-1 dose 0.0005 — 10× slower than the critic τ), making the
  index encoding quasi-stationary. The eval path MUST read the same EMA
  params from the checkpoint (train/eval convention test, rz-law lesson).

NOT ported, with reasons: ortho itself (convicted; C2 replaces it);
Meta Motivo's latent-conditional discriminator (needs on-policy rollouts;
offline analog is a research project — reserve, noted in §6).

## 1. P0 — B-decodability probe FIRST (eval-only, existing checkpoints)

Before any training: measure ridge-reconstruction R² of the three reward
families (sparse/dense/region indicators on the pool) from B(ũ_pool)
features, for the z24_cd checkpoints AND the g-indexed d18 cd checkpoints
(same pool, same ridge λ, P=16384). Script:
`scripts/rescuing_fb/probe_b_reward_decode.py`, reusing `build_pool` and
the family builders from `eval_density_zeroshot.py` (thin-wrapper rule;
v1-E2 lesson: a control is mandatory — include a raw-coordinate feature
baseline whose R² is known high for the dense family).
Readings (pre-registered):
- g-B decodes, z-B does not ⇒ C2 is binding; expect the rdecode arm to
  matter most.
- both decode ⇒ B-geometry is fine; C1/C3 carry the failure; rdecode may
  be dropped from the full stack if its solo arm is flat.
- neither decodes (but raw coords do) ⇒ B under CE never supports linear
  readout; C2 becomes the headline mechanism and E2a's formula was never
  viable without it.

## 2. Implementation contract (gciql_flow.py; delta on the spec-24 code)

- Knobs as in §0, all under the existing `measure_density_z` lattice
  (each asserts `measure_density_z`; tilt asserts β ≥ 0; enc_tau in
  [0, 0.1]; rdecode_coef ≥ 0). Defaults reproduce spec-24 bit-identically
  (trace-time branches; pinned by test).
- Tilt: applied in BOTH online and target logits (the target must live in
  the same family), and in the actor's q-path logits. `w̄⁺` uses the
  target towers' tilt with the same z.
- EMA encoder: registered as `density_b_zenc` params, initialized =
  density_b, updated in the `update` override with τ_z; `_density_z_encode`
  reads it when enc_tau > 0; `sample_actions` reads it from the restored
  checkpoint (it IS serialized — unlike the 17b protos, these are params;
  no file coupling).
- BC-hindsight split: the actor batch already carries per-row hindsight
  coords; rows whose z was sphere-drawn get their BC contribution masked
  and renormalized (mask test §4). q term unmasked.
- rdecode aux: closed-form ridge inside the loss (jnp.linalg.solve on the
  d×d gram; d=50 — cheap), centers drawn from the batch's own query rows,
  fresh per step, info keys `z_rdecode_r2`, `z_rdecode_loss`.
- NEW INSTRUMENT (gate, all arms): `density_w_ztv` = E[TV(w(s,a,z),
  w(s,a,z′))] over per-row resampled index pairs — the direct
  index-sensitivity meter the spec-24 wave lacked. Log unconditionally in
  z mode.

## 3. Pre-registered wave (after P0; venue logic from spec 24's audit)

Stage 1 — cube-single screens, 1 seed each, 500k, standing recipe:
- `z25_cs_tilt` (C1), `z25_cs_actor` (C3 both knobs), `z25_cs_dec` (C2),
  `z25_cs_full` (C1+C2+C3).
  Screen gate S1: `density_w_ztv` bounded away from 0 at 500k AND E1
  (goal-encoding z(g) conditioning) ≥ 0.5 (g-anchor ≈ .93; spec-24 z
  scored ~0 here on cd and ≈BC on pmm — any life is signal).
Stage 2 — cube-double, 3 seeds, the best single mechanism + `z25_cd_full`:
  gates G1: E1 within 0.15 of the g-anchor .736; G2: E2a/E2b through the
  v2 harness (control gate re-run first, mandatory) ≥ 0.35 on sparse
  (g-anchor .549); G3: E2a ≈ E2b within 0.05 (the analytic interface
  works) — G2 with E2b≫E2a still counts as "index revived, analytic
  assignment needs C2 dose".
Pre-registered NULL: all four screens flat on `density_w_ztv` ⇒ the
estimator's softmax normalization itself absorbs index information
regardless of coupling; the z program stops and the reward-embedding
actor (w_r conditioning, the spec-19 extension) becomes the only
interface route — write that spec instead of iterating here.

## 4. Tests (`tests/test_gciql_measure_density_z_coupling.py`)

1. all-defaults bit-identity to the spec-24 arm (param tree + loss).
2. tilt numpy reference (both towers + actor path; rows on simplex).
3. tilt non-ignorability: with F frozen at random init, TV(w(z1),w(z2))>0.
4. EMA encoder: init == B; τ_z-mix formula; z invariant to online-B
   perturbation; sample_actions reads the EMA params from a restored ckpt.
5. BC mask: sphere-z rows contribute zero grad to the BC term, hindsight
   rows unchanged; renormalization correct.
6. rdecode: closed-form ridge vs numpy; grads reach B; R² info in [0,1].
7. `density_w_ztv` logged in z mode, absent otherwise.
8. lattice refusals (each knob without measure_density_z; τ_z out of
   range). 30-update end-to-end with the full stack: finite, simplex.

## 5. Eval protocol (unchanged instruments, one addition)

E1 goal-encoding and E2a/E2b exactly as spec 24 §3 through the v2 wrapper
(`eval_density_z_rewards.py`), with its bit-exact control gate re-run per
session. Addition: report `density_w_ztv` and P0-style B-decode R² per
checkpoint next to every success number, so index-sensitivity and
B-geometry are quoted alongside outcomes (the spec-24 wave's blind spot).

## 6. Anti-scope

No β/τ_z/coef sweeps (one dose each, round 1); no ortho in any form; no
discriminator (reserve: an offline latent-conditional discriminator on
hindsight-relabeled trajectory segments is plausible but is its own
spec); no PMM cells (non-discriminating metric band, spec-24 audit); no
changes to the g-indexed paths (bit-identity is a test); no quoting any
"below BC" comparisons.
