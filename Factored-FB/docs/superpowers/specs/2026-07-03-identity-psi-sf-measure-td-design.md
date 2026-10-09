# Design: `identity_psi` — SF-style measure TD on φ with a frozen identity goal leg

**Status:** PENDING USER REVIEW (defaults chosen while user away; two decisions marked ⚠️ are overridable).
**Date:** 2026-07-03
**Branch:** gciql-fb-bridge
**Agent touched:** `tools/wandb_mode_shim/gciql_flow.py` (jax bilinear GCIQL + flow actor) + one config + one test file.

## Motivation

Across the FB↔GCIQL bridge, the *learned* successor-measure TD on the bilinear
representation collapses on cube (MeasureOnly 9.9, C4 16.5, MeasureSA 0.0-dead; all
≪ the ~90 that the identical architecture reaches under scalar IQL Q-TD). Separately,
on the **torch FB** agent, freezing the backward map to identity (`fixed_b=identity`,
`z=g`) under a **scalar IQL** critic doubled success (`zg_iql` ≈ 0.44 vs learned-B
0.21) — learned B is a bottleneck there.

This experiment runs the **missing cell**: freeze the goal leg to identity **and** keep
the **vector successor-measure (SF-style) TD** on φ, on the **jax bilinear GCIQL-flow**
substrate (so it sits apples-to-apples next to the other GCIQL bridge cells). It is the
on-substrate counterpart of the FB `zg_her` run (identity-B + measure ≈ 0.19, which was
*confounded*: identity B ⇒ linear reward g·s, q blew up to ~1460).

**Primary hypothesis / what we learn:** the learned-ψ measure runs may collapse because
φ and ψ co-adapt (two learnable legs shrinking toward each other). Freezing ψ=identity
removes that degree of freedom. Does φ stay healthy (no q blowup) — and does success move
off the 9–25 band — when the goal leg cannot collapse? Expected regime is the
linear-reward one (accepted, per user's "literal, run as-is"); the value is the clean
on-substrate diagnostic, not necessarily high goal-reaching success.

## The math

Bilinear critic today: `Q(s,a,g)=⟨φ(s,a), ψ(g)⟩/√d`, ψ a learned MLP, d=value_latent_dim.
The knob sets the goal leg to identity:

```
ψ(x) = x                               (param-free)
M[i,j] = ⟨φ(s_i,a_i), s'_j⟩ / √d       (B := ψ := I applied to successor obs s')
φ(s,a) ← diag-attract(s') + γ φ(s'_+, a'_+)     (off-diag TD + diagonal attraction)
actor: greedy on Q = ⟨φ(s,a), g⟩       (flow actor, measure_only)
```

This reuses the existing `measure_loss` critic-leg path (`gciql_flow.py:367-390`)
verbatim; the only change is that `ψ`/`B` is identity instead of `psi(·)`.

## Decisions

### ⚠️ Decision 1 — φ forward map (DEFAULT: task-agnostic, canonical SF)

- **Primary cell (default):** task-agnostic `φ(s,a)` — `fb_forward_z=False`. Goal enters
  *only* through the identity readout weight `g`. The honest SF object and the literal
  reading of the request. NB: this is the `MeasureSA`-shaped forward leg (which was
  0.0-dead under *learned* ψ) — testing it under *frozen* ψ=identity IS the hypothesis.
- **Optional second cell (one flag):** z-conditioned `φ(s,a,z=g)` — `fb_forward_z=True`.
  z-conditioning was the 0.0→25.5 difference in learned-ψ measure runs, so it may be the
  only variant that isn't dead. Provided as a sibling config; launch is the user's call.

### ⚠️ Decision 2 — run scope (DEFAULT: build + test only)

Implement the knob, config(s), and TDD-green tests. **Do not launch training.** User
launches the cube-single 3×500k run themselves (tmux, entity `amsks`).

## Implementation (contained to `gciql_flow.py` + config + test)

1. **New critic module** `GCIdentityPsiValue(GCBilinearValue)`:
   - `self.phi = MLP((*hidden_dims, obs_dim), activate_final=False, layer_norm=...)`
     — output dim = **goal/obs dim**, NOT d=50 (so ⟨φ, ψ(g)=g⟩ is well-formed).
   - `_psi(goals) = goals` (identity, param-free; skip goal_encoder).
   - `__call__` returns `v=(φ*ψ/√d).sum(-1)` with ψ=goals, plus `(v, φ, ψ)` under `info`.
   - Mirrors `agents/fb/fixed_backward.py:FixedIdentityBackward` (B(obs)=obs, z_dim=obs_dim).
2. **Config plumbing** in the critic-construction block (`gciql_flow.py:~524-542`):
   - `identity_psi: true` selects `GCIdentityPsiValue` (composes with `fb_forward_z` for
     the z-conditioned variant — φ then concatenates z=g).
   - Assert `value_latent_dim == goal_dim` (φ output dim must match the raw goal).
   - Force `actor_cond='goal'` under `identity_psi` (z=ψ(g)=g, so z-plumbing is moot and
     this avoids the z-dim mismatch guard at `:494-497`).
3. **Gate off the ψ ortho term** when `identity_psi`: ψ has no params, so `Cov=ψψᵀ` is an
   inert constant — set/require `measure_ortho_coef=0` and skip it (same gating the FB
   agent applies for a param-free B). No sphere / `project_z` (raw identity; matches FB's
   `norm_z=False`).

## Config

Base off `configs/gciql/cube_single_state_gciql_flow_bilinear_d50_C2_measure_twotower.yaml`:
- drop `fwd_two_tower`, `measure_ortho_coef`;
- add `identity_psi: true`, `measure_only: true`, `measure_legs: critic`, `measure_coef: 1.0`;
- set `value_latent_dim = <cube_single obs_dim>` (assert-checked at build);
- keep the flow actor, `actor_cond: goal`.
- Sibling config with `fb_forward_z: true` for the optional z-conditioned cell.

Filenames (proposed):
`cube_single_state_gciql_flow_identity_psi_measure.yaml` (+ `_fbz` sibling).

## Comparison anchors

- **FB `zg_her` ≈ 0.19** (same idea, FB substrate) — the primary anchor.
- **MeasureOnly 9.9 / C4 16.5** (learned-ψ measure cells, this exact substrate).
- **Diagnostic:** φ/q health — did freezing ψ stop the `q≈1460` blowup seen in FB zg_her?
- (Context: scalar-TD identity cell FB `zg_iql` ≈ 0.44 / `idB_dnorm` 65.3 — NOT this cell,
  but the contrast that motivates asking whether the *objective* or the *ψ collapse* is
  the killer.)

## Testing (TDD, `tests/test_gciql_identity_psi.py`)

- (a) `_psi` returns the goal tensor unchanged.
- (b) φ output dim == obs_dim; `Q=⟨φ,g⟩/√d` runs and is finite.
- (c) `measure_only + identity_psi` trains finite for a few steps; `measure/measure_loss`
  present; no ortho term contributes (ψ carries no grad).
- (d) the ψ leg exposes **no learnable params** (param count invariant to goal dim on ψ).
- (e) `identity_psi + fb_forward_z` (the sibling cell) also trains finite.

Env: `.venv-jax-cpu` / jax venv (`.venv/bin/python -m pytest` is torch-only; GCIQL tests
run under the jax venv — see memory `jax-venv-location`).

## Out of scope (YAGNI)

- `identity_norm` (normalized identity ψ) — FB showed it doesn't fix the measure blowup;
  add later only if the raw cell motivates it.
- Any change to `isfb_flow.py` or the torch FB agent.
- Scalar-IQL identity-ψ cell (that's the already-run FB `zg_iql`; not re-run on substrate
  unless requested).
```
```
