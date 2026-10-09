# Zero-shot GCIQL + scale-controlled measure interpolation — design

_2026-07-06. Follows the experiment-trust audit (`PAPER/fb-gciql-bridge/audit-20260706.md`),
which established: (i) the G4 exact-z→inferred-z gap (0.93 → 0.07 on the same critic,
cf-AUC 0.92) has never had its fix properly tested (exp4's `actor_z_infer` was an artifact —
z ≈ 99% noise by construction); (ii) the IQL+λ·measure interpolation is an empty cell
(18/18 runs crashed via Q-scale blowup); (iii) B1 (`fb_forward_z` under IQL) already works
at 0.848. Domain: cube-single-play state, bilinear d50, flow actor, jax
(`tools/wandb_mode_shim/gciql_flow.py`), 3 seeds @500k unless noted._

**Goal.** Answer the project's two open questions with trustworthy cells:
**(E1)** can the ~0.90 IQL critic be made zero-shot (FB's API: reward samples → z → policy,
no fine-tune) by repairing the inference→actor path, without touching the training
objective? **(E2)** what does adding the successor-measure loss to a *working* IQL critic
actually do, once the scale confound is removed — actively destroy, coexist, or help?

---

## E1 — Zero-shot GCIQL (P1)

### E1a. Post-hoc inference repair on existing checkpoints (analysis only, run first)

No training. Locate G4 z-cond checkpoints (`gciql_g4_zcond`, actor_cond='z', @1M, n=9;
`~/bridge_ckpt_backup` holds the 1M-campaign params) and write an eval script
(`scripts/eval_zeroshot_inference.py`) that, per task:

1. Builds the dataset-based reward sample set (the G4 eval protocol's relabeled indicator
   reward over dataset states).
2. Computes z under three rules, all sphere-normed to √d afterward:
   - **v0 (baseline, must reproduce ~0.07):** `z = Ê_ρ[r·ψ(s')]` — the FB moment formula.
   - **v1 (regression/whitened):** `z = (Ê[ψψᵀ] + εI)⁻¹ Ê[r·ψ]`, ε ≈ 1e-3·tr/d. This is
     least-squares regression of r onto ψ — the correct dual vector when ψ is not
     orthonormal (GCIQL's ψ isn't, and ortho-as-a-loss kills the critic; whitening imposes
     the isometry with linear algebra instead).
   - **v2 (exact, upper anchor, must reproduce ~0.93):** `z = ψ(g)`.
3. Logs the diagnostic that dissociates the two failure stories: **cos(z_rule, ψ(g))** per
   task, alongside success via `sample_actions_with_z`.

Decision rule: v1 ≥ 0.6 → inference formula was the dominant break; E1b becomes
polish. v1 ≈ v0 ≈ 0.07 with cos ≈ 1 → pure actor conditioning-distribution shift → E1b is
the load-bearing fix. cos low for v0 but high for v1 → whitening story confirmed.
Cost: ~1 evening, CPU/1-GPU eval only.

### E1b. Inference-consistent actor training (`actor_z_infer` v2 — fix in place)

Rewrite `_infer_actor_z` (gciql_flow.py:321) and the exp4 path in `actor_loss`:

- **z construction (the fix):** per batch row i, `z_i = sphere(Σ_j k_ij ψ(s'_j) + ψ(g_i))`
  with `k_ij = exp(−‖s'_j − g_i‖² / (2h²))` over the batch's next-states (raw-obs distance;
  h set so the effective neighbor count is ~5–20; report `k_eff` in metrics). This models
  the *test-time* inference rule (average of ψ over the reward region) instead of exp4's
  10%-random mask (which summed ~102 random ψ's against 1 true one at B=1024).
- **Mix:** new knob `actor_z_infer_mix` (float = fraction of rows conditioned on the exact
  `ψ(g_i)`; default 0.5, per-row Bernoulli) — exact-z performance is preserved while the
  actor learns the inferred-z neighborhood. `actor_z_infer_mix=1.0` must reduce to the
  stock G4 actor loss exactly (test 3).
- **Q-term fix:** the actor's Q-guidance uses the critic at the **raw goal**
  (`critic(obs, goals, actor_actions)`), not the z-projected readout exp4 used — removes
  the detached-Q-scale pathology (exp4: actor q_mean −1.76 vs critic −19.2). Only the
  actor's *conditioning input* becomes noisy; its training signal stays exact.
- Everything else stock G4: IQL objective untouched, bilinear d50, actor_cond='z'.

Config: `cube_single_state_gciql_flow_bilinear_d50_zinfer_v2.yaml`, group
`GCIQL-State-Flow-Bilinear-d50-ZInferV2`. 3 seeds @500k.

TDD (new tests in `tests/test_gciql_zinfer.py`, jax-subprocess style):
(1) kernel-z with h→0 equals sphere(ψ(g)) (self-retrieval); (2) k_eff finite and >1 for
moderate h; (3) mix path: with mix=1.0 identical to stock G4 actor loss; (4) 5 updates
finite; (5) actor Q-term calls critic with raw goals (not z) — assert via shapes/monkeypatch.

**Success criteria:** in-training exact-z eval ≥ 0.88 (no regression vs G4 0.93@1M /
~0.90@500k); E1a-script zero-shot (v0 and v1) on the trained checkpoint: ≥ 0.6 = win,
≥ 0.85 = full repair ("zero-shot GCIQL" exists — the paper's constructive result).
Failure mode to watch: exact-z drops (inferred-z conditioning poisons the actor) →
raise mix to 0.75.

---

## E2 — Scale-controlled IQL + λ·measure interpolation (P1)

### Code prereqs (both default-OFF, TDD)

1. **`measure_scale_norm`** (bool): inside `measure_loss`, divide the TD terms by a
   stop-gradded scale so the gradient is dimensionless — `offdiag / sg[mean(M²)+1e-6]` and
   `diagt / sg[sqrt(mean(M²))+1e-6]` (exactly the trick the actor's q_loss already uses).
   Ortho term unchanged (already scale-bounded by ψ).
2. **`measure_stopgrad_phi`** (bool): stop-grad φ inside `measure_loss` (both online legs)
   so the measure gradient flows **only into ψ** — the quarantined variant: measure can
   shape the goal/readout geometry but cannot touch the value-critical forward leg.

Tests: (a) scale_norm on ⇒ measure_loss invariant (≤1e-4 rel) to multiplying φ,ψ params'
output by 10 (simulate via scaled features through a stub); (b) stopgrad_phi on ⇒
∂measure/∂φ-params ≡ 0, ∂measure/∂ψ-params ≠ 0; (c) both compose with `measure_coef>0` +
full IQL loss, 5 updates finite, and `critic/q_mean` stays within [−100, 10].

### Run matrix

Base = the 0.90 cell (bilinear d50, flow, IQL). All with `measure_scale_norm=true`,
`measure_legs='critic'`, `measure_ortho_coef=0`:

| Cell | measure_coef | stopgrad_phi | group suffix |
|---|---|---|---|
| I-λ.01 | 0.01 | no | `MeasureMix-l001` |
| I-λ.1 | 0.1 | no | `MeasureMix-l01` |
| I-λ1 | 1.0 | no | `MeasureMix-l1` |
| Q-λ.1 | 0.1 | yes | `MeasureMix-l01-sgphi` |
| Q-λ1 | 1.0 | yes | `MeasureMix-l1-sgphi` |

15 runs (5 cells × 3 seeds) @500k. Guardrail: any run with `critic/q_mean` outside
[−100, 10] or grad_norm > 500 at a log point is flagged as scale-escaped (the confound
returning) — its success number is not interpretable.

### Reading the outcome

- **Smooth monotone degradation with λ (scale healthy):** the measure gradient destroys
  task-relevant value structure *per se* — objectives conflict at the representation level
  (supports the projection-mismatch mechanism). Q1's "what brings in the gap" gets a
  dose-response curve.
- **Flat until λ≈1 then cliff:** compatible-until-dominant; the collapse in measure-only
  cells is "IQL signal removed," not "measure signal toxic."
- **Small-λ improvement:** measure as auxiliary regularizer — immediately interesting for
  the SOTA paper.
- **stopgrad-φ flat where shared-legs degrades:** the conflict lives in φ (the consumed
  forward leg) — and the quarantined variant becomes the candidate "measure head for
  zero-shot inference on a working critic": evaluate each cell's ψ with the E1a script
  (v0/v1 inference) to test whether small-λ measure shaping + repaired inference =
  zero-shot at ~0.9. **This cross-evaluation is the actual prize cell of the whole spec.**

---

## E3 — Corrected SF cell (P2, only if the SF claim is wanted back)

Minimal faithful SF: learn base `b(s)` + task head `w(g)` by reward regression
`L_rew = (⟨b(s'), w(g)⟩ − r)²` (relabeled indicator r), SF vector TD on φ against this
learned b, readout `Q = ⟨φ, w(g)⟩`; plus the scalar-reward-TD positive control on the same
(b, w) (≈ ref's `sf_simple` ≈ `zg_iql` 0.44). Deferred — B1 = 0.848 already provides the
constructive "z-indexed family under scalar TD works" result; write E3 only if a reviewer
needs the SF-specific control.

---

## Sequencing & cost

| Order | What | Cost | Blocks on |
|---|---|---|---|
| 1 | E1a eval script on G4 checkpoints | 1 evening, no training GPUs | checkpoint location |
| 2 | E2 code (2 knobs + tests) + launch 15 runs | 1 evening code + 1 GPU-evening | — |
| 3 | E1b code + launch 3 runs | 1 evening code + shares GPUs | E1a informs mix/urgency |
| 4 | E1a script re-run on E1b + E2 checkpoints (the prize cross-eval) | hours | 2,3 |

Ops (memory rules): tmux always; wandb entity `amsks`; jax venv `/dev/shm/.venv-jax`;
`XLA_PYTHON_CLIENT_PREALLOCATE=false`; never co-init 2 jax runs on one GPU (stagger
launches on the >1500MB-block rule).

## Doc fixes bundled with this work (no runs needed)

- `PAPER/PROJECT-MAP.md` §③: retract "collapse is in the SF critic/representation"
  (audit §2 corrected reading); add B1 = 0.848 to the results.
- `PAPER/fb-gciql-bridge/untried-design-space.md`: mark B1 and C2 done.
- `PAPER/fb-gciql-bridge/fb-gciql-bridge.tex`: add the off-diag-residual datum
  (6347 z-blind vs 266 z-cond) to the mechanism section; scale-pathology caveat on the
  measure band.
