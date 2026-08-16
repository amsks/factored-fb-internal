# Spec 30a — Implementation queue: first GPU wave (spec 30 Rev 1)

For a fresh implementing agent on this box. Authority = specs/30-review-response.md
(Rev 1); read it first, then specs/29-ra-dual-index.md and DENSITY-FB-EXPLAINED.md
§3.6/§4.4/§6.6. Code anchors: tools/wandb_mode_shim/gciql_flow.py — density_loss
2404–2642, dual encode 1046–1142 (`_density_z_encode`, `density_z_from_reward_samples`),
`_delta_phi` 1832–1856, `_density_queries` 1859–1868, eval `sample_actions` 3289–3304,
`sample_actions_with_z` 3309; tests test_gciql_spec29_dual_index.py; preflight evidence
results/spec30_geom/ (centered indices de-collapse, cos ~.003 — cite, do NOT rerun).

House rules (binding): TDD; every new knob default-OFF and bit-identical when off
(3 real update() steps + sample_actions, mutation-tested — mirror the spec-29 suite);
tests `.venv/bin/python -m pytest`; training in tmux (unique names — has-session
prefix-matches); JAX GPU venv `/dev/shm/.venv-jax` (volatile, recreate if wiped);
`XLA_PYTHON_CLIENT_PREALLOCATE=false`; verify datasets exist (paths in /dev/shm get
wiped); check `nvidia-smi` before packing (a second track shares this box — do not
kill its jobs); wandb entity `amsks` (verify); add every new group to
scripts/results_map.py immediately; COMMIT code+tests+spec before launching (no
Claude co-author line); artifacts/results stay untracked. Never quote mid-run
success; quote 500k AND mean≥350k.

Wave-1 = 4 training runs (1 seed each) + 1 CPU-only hygiene item. Anchors:
cs g .93 / BC .111 / R-A-uncentered .812; cd soft .736 / hard .680 / BC .019.

## Item 1 — `dual_centered` encoder + two cube-single arms (the P1 screen)

New encode mode `measure_density_z_encode=dual_centered`:
- EMA moments μ̂ and Ĉ (CENTERED second moment) as serialized param leaves, SAME
  EMA schedule as the existing uncentered Σ̂ (`density_b_zenc` pattern). Base
  measure for μ̂, Ĉ, r̄, m̂ᶜ = the batch next-state query marginal, used
  consistently. Symmetrize Ĉ each update; eigenvalue floor/jitter before solving;
  log min-eig(Ĉ), eff_rank(Ĉ), cond(Ĉ), and EMA-vs-current-batch sync residuals.
- TWO SEPARATE APIs (spec 30 Rev 1 — the centered encoder has NO M=1 limit):
  - point-prompt: z_point = √d·norm(B̄(φ(g)) − μ̂)  ("analytic centered point
    prompt"; never described as a Dirac/M=1 limit);
  - reward-sample: {(u_j,r_j)}_{j=1..M} → ẑ = √d·norm((Ĉ+λI)⁻¹ m̂ᶜ),
    m̂ᶜ = (1/M)Σ (B̄(u_j)−μ̂)(r_j−r̄). M=1 raises an error. Constant-reward
    fallback: ‖c‖ < ε_z ⇒ z_neutral; always log ‖m̂ᶜ‖, ‖c‖, Var[r].
- λ = 0.01·tr(Ĉ)/d default. Training rows: hindsight rows encode via the
  reward-sample path with r_j = kg over the row's query set; random-sphere rows
  unchanged (existing `z_hindsight` masking). Everything else matched to the
  spec-29 R-A recipe (z_rand_frac .5, enc_tau>0 required).
- Required unit tests beyond bit-identity: exact invariance B→B+b; exact
  invariance r→r+β; invariance r→αr after normalization; M=1 raises;
  constant-reward fallback; a centered sibling of
  `test_sigma_is_ema_uncentered_and_matches_spec28_pool`.

Arms (cube-single, 500k, s0), both with the KERNEL readout
(`measure_density_z_actor_reward=kernel`):
- `ra2_cs_centpoint`: conditioning = centered point prompts.
- `ra2_cs_centdual`:  conditioning = reward-sample prompts.
Eval rule (no-silent-shortcut): the centdual arm's native goal eval MUST go
through the reward-sample API with an M=256 kernel pool; record the point-shortcut
eval as a separate diagnostic column only. Mid-run instruments from step 0:
eval_z_action_ranking.py at eval intervals + z_cos_spread; KILL an arm flat at
actor-sensitivity ≈ 0 by 250k.

### Item 1 — IMPLEMENTED 2026-08-16 (`measure_density_z_encode=dual_centered`)

Shipped knobs (all default-OFF, bit-identity to the pinned pre-spec-30 commit proven over
3 real `update()` steps + `sample_actions` on the spec-24/25/28/29 and g arms):
`measure_density_z_encode='dual_centered'`, `measure_density_z_prompt ∈ {sample, point}`,
`measure_density_z_eval_pool_m` (256), `measure_density_z_cov_eig_floor` (1e-6·tr/d),
`measure_density_z_const_eps` (1e-6), plus Item 4's `measure_density_dup_instrument` /
`measure_density_dup_eps`. λ and the moment EMA reuse spec 29's
`measure_density_z_ridge_lambda_mult` / `_sigma_tau` (same schedule, as specified).

Implementation decisions the spec left open, recorded so the arms are readable later:
- **μ̂, Ĉ** = `ZCenteredMomentsState` param leaves (init μ=0, C=I), EMA'd in `update()`
  from the batch next-state query marginal at the PRE-step index tower, Ĉ centered on the
  batch mean with 1/M and symmetrized every update — the exact convention of
  `eval_spec30_geom.py::ridge_batch(centered=True)`, so the trained moments are the
  objects the preflight probes scored. Eigenvalue floor applied at solve time; λ from the
  UNFLOORED trace.
- **z_neutral = 1_d** (‖1_d‖ = √d, so the standing norm invariant survives the fallback).
- **Exact √d normalization** (`_sphere_or_neutral`), not `_sphere`: `_sphere`'s +1e-6
  denominator guard makes the map only approximately scale-invariant (a 100× downscale of
  r moved z by 5e-3 — measured), which would have broken the r→αr claim. The explicit
  constant-reward fallback replaces that guard.
- **Native goal eval** on the sample arm: `sample_actions` builds kg(goal, pool) over a
  FROZEN M=256 pool (`ZEvalPoolState`, written once on the first update from the batch's
  raw query coords, serialized as a param leaf) and encodes through the reward-sample API.
  It cannot fall back to the point prompt. Training rows still use the row's own query set
  (M = batch size = 1024), per this spec. The point shortcut is exposed separately as
  `sample_actions_point` / `density_z_point_prompt` for the diagnostic column.
- Item 4's irreducible-CE estimate is logged in soft mode too (as the hard-mode
  counterfactual), raw and (1−γ)-weighted — the weight the immediate term actually carries.

Tests: `tests/test_gciql_spec30_centered_index.py`, 11 green (bit-identity; both closed
forms; exact B→B+b invariance incl. the stale-μ̂ case; exact r→r+β and r→αr with the
uncentered arm as the moving control; M=1 raises; constant-reward fallback; the centered
EMA sibling; the no-silent-shortcut eval rule; lattice; end-to-end; duplicate instrument).
The shift/scale invariances are pinned at float32 round-off (cos > 1−1e-8), with the
per-component bound scaled to the cancellation |β| induces — the algebra is exact.

## Item 2 — paper-exact TD InfoNCE, cube-double, s0 (`tdi_cd_exact_s0`)

Per the audited delta list (~80–120 lines + lattice + tests):
- critic: their Eq. 10 — immediate term labels the dynamics next state, bootstrap
  term labels a MARGINAL draw soft-labeled by the stop-grad target softmax, ONE
  shared negative pool s^(2:N) drawn iid from the dataset future-state marginal
  (new query-source knob; our current support = batch next states);
- ℓ2-normalize both towers (their convention; temperature implicit);
- actor: goal-contrastive CE over the commanded goal vs marginal negatives
  (their Eq. 11) + MSE goal-BC regularizer (their App. D.3), their future-state
  goal-relabeling distribution;
- same φ outcome coordinates as Density-FB.
This is the PAPER-EXACT control ("how does the published method do here?").
The matched-extractor variant (their critic + our kernel readout/flow actor) is
the promotion wave — implement the knobs so it is one config away, but do not
launch it in wave 1.

## Item 3 — goal-conditioned outcome-space LSIF, cube-double, s0 (`lsif_cd_phi_s0`)

Revive the jax LSIF-style bilinear measure arm (measure_loss, gciql_flow.py:2003):
m(s,a,g,x) = F(s,a,g)ᵀB(φ(x)), raw-g conditioning, LSIF objective, actor scores
via EXTERNAL kernel reward integration q = E_x[m(s,a,g,x)·kg(x)] (mirror
`_density_actor_q` over m instead of w), same flow actor + BC machinery
(~40–80 lines). Matched to Density-FB in everything except LSIF-vs-CE and
unnormalized-vs-normalized. This is the privileged-coordinates confound control.

## Item 4 — P4 hygiene (CPU, alongside)

- Tabular ground-truth test: zero-shot de-biased readout vs exact
  (I−γP^π)⁻¹-based successor measure on a toy MDP, covering: measure start index,
  reward attachment convention, terminal/absorbing handling, the 1−γ
  normalization, the finite-pool correction, hard AND soft immediate kernels.
- Duplicate instrument: log per-batch exact- and ε-duplicate query fractions +
  estimated irreducible CE in hard mode (instrumentation only, no fix yet).

## Reporting

Deliver: commit hashes; test counts (all green); tmux session names + GPU
placement; wandb groups (+ results_map.py entries); watchers/monitors armed on
all four runs; and NO result claims before 500k. Promotion gates (spec 30):
3 seeds only for arms that survive — centered competitive with .812-level;
TD InfoNCE does not trivially reproduce the soft result; LSIF stays well below
Density-FB.
