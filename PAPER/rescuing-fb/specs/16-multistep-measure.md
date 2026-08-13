# Spec 16 — Multi-step successor-measure TD (dwell-structured estimator)

Status: IMPLEMENTED 2026-08-06 (uncommitted; 6 new tests + 42 neighboring
measure tests green; wave M1 NOT launched). Derivation:
`PAPER/Multistep-FB/main.tex` (compiled).
Parent: spec 15 §3b/§3c. Constraint (user): NO data preprocessing — the
structure lives in the estimator; the sampler may only read raw trajectory
windows (same access class as hindsight goal sampling).

## 1. What is being changed, in one paragraph

The region successor-measure TD in `tools/wandb_mode_shim/gciql_flow.py`
(`measure_loss`, critic legs, hit-spec region paths) currently uses the
one-step target `hit + γ·tM(s_{t+1}, a⁺)`. We replace it, behind a
default-off knob, with the k-step target
`Σ_{u=1..k_i} γ^{u-1}·hit(s_{t+u}, col_j) + γ^{k_i}·tM(s_{t+k_i}, a⁺)`,
where k_i is fixed or geometric (λ-return), truncated at the episode
boundary. Fixed point is unchanged for every k (contraction, modulus γ^k;
see the note §2); what changes is where the dwell mass sits in the target:
observed instead of bootstrapped. Off-policy bias of the observed window is
gated by γ^{T_int} (dwell-invariance lemma, note §3) — largest safety
margin exactly on multi-object envs.

## 2. Knobs (all default-off, bit-identical when off)

| knob | type/default | meaning |
|---|---|---|
| `measure_nstep` | int, 0 | 0 = stock one-step path, byte-identical. N>0 = k-step window cap K. |
| `measure_nstep_lambda` | float, 0.0 | 0 = fixed k=K. λ∈(0,1): k_i ~ Geometric(1−λ) truncated at K (λ-return via sampled horizons). |
| `measure_dwell_head` | bool, false | STAGE 2 (separate wave): M = D(g)·H with D=exp(goal-only head), H=bounded bilinear. Not in the first implementation round. |

Assert-lattice (mirror the spec06/07 discipline):
- `measure_nstep>0` requires a hit-spec region path (`measure_first_passage`
  or region-visitation), `measure_legs='critic'`, `value_arch='bilinear'`.
- Refuse combination with `measure_eps_multiscale`, `measure_nhat_head`,
  `measure_expectile`, `fb_forward_z` (first round keeps the lattice small;
  `measure_testpoint_dist` batch AND hindsight both allowed — hindsight is
  the standing recipe).

## 3. Sampler change (`third_party/ogbench/impls/utils/datasets.py`, GCDataset)

When `measure_nstep = K > 0`, `sample()` additionally returns:
- `mstep_next_obs` [B, K, obs_dim]: observations at idxs+1 … idxs+K,
  gathered with `np.minimum(idxs+u, final_state_idx)` clamping (same
  `terminal_locs`/`searchsorted` machinery as `sample_goals`).
- `mstep_k` [B] int: realized horizon = min(K or geometric draw,
  final_state_idx − idx), ≥ 1. k_i=1 must reproduce the one-step target
  exactly (unit test).
- No other fields: the bootstrap state is `mstep_next_obs[i, k_i−1]`.

Frame-stack / pixel paths: out of scope round 1 — assert `encoder is None`.

## 4. Loss change (`measure_loss`, region branches)

For sample i, column j (col_states = nobs or vgoals per `tpd`):
1. `hits_w[u,i,j] = hit(mstep_next_obs[i,u], col_j)` via `_hit_matrix`
   vmapped over the K axis; mask u ≥ k_i.
2. **Visitation** (`fp=false` region path): observed
   `O[i,j] = Σ_u γ^u·hits_w` (γ-power convention matched to the existing
   one-step code, verified by the k=1 equivalence test); target
   `y = O + γ^{k_i}·tM_k[i,j]`, where `tM_k` uses `tphi` at
   `s_b = mstep_next_obs[i,k_i−1]` and `a⁺ ~ π(s_b, goal)` (same actor
   machinery as now, evaluated at the k-step state).
3. **First-passage** (`fp=true`): absorbing version. `notyet[u] =
   Π_{v<u}(1−hits_w[v])`; observed `O = Σ_u γ^u·hits_w[u]·notyet[u]`;
   bootstrap weight `γ^{k_i}·notyet[k_i]` (zero once the region was hit
   inside the window). k=1 reduces to the current absorbing target.
4. Everything downstream (`measure_scale_norm`, stopgrads, ortho, coef)
   unchanged.

Cost: one extra `_hit_matrix` per window step (cheap: slicing + norm) and
ONE extra target-critic forward at the k-step states (replaces the
existing one at nobs when the knob is on).

## 5. Tests (TDD, all in `tests/test_gciql_measure_nstep.py`, jax subprocess)

1. Default-off: config with `measure_nstep=0` → param tree byte-identical +
   `measure_loss` bit-identical vs baseline on a fixed batch.
2. k=1 equivalence: `measure_nstep=1` with mstep fields built from the
   batch's own nobs → loss equals the one-step path bit-exactly (both fp
   and visitation branches).
3. Hand-computed 3-step window on a 2-sample toy batch: observed mass and
   bootstrap weight match a numpy reference (fp and visitation).
4. Episode-boundary truncation: k_i clamps at the final state; no leakage
   across trajectories (constructed two-trajectory batch).
5. Geometric mode: with fixed rng, k_i distribution matches the truncated
   geometric; λ→0 degenerates to k=1.
6. Sampler: GCDataset with `measure_nstep=K` returns the new fields with
   correct shapes/clamping on a synthetic 2-episode dataset; K=0 returns
   no new fields (loader untouched).
7. Assert-lattice refusals fire.

## 6. Pre-registered cells (after tests green; d50, standing recipe elsewhere)

Wave M1 (cube-single, 4 cells × 1 seed, 500k, GPUs free now):
- `m1_base`: phase-2-style region VISITATION from scratch (known-bad
  anchor: §9.16 collapse / anneal-campaign floor).
- `m1_k64`: same + `measure_nstep=64`.
- `m1_lam`: same + `measure_nstep=256, measure_nstep_lambda=0.99`.
- `m1_fp_k64`: phase-1 first-passage + nstep=64 (does multistep preserve
  the working arm? must hold ≈0.9+ success).

Reads (pre-registered):
- PRIMARY: m1_k64/m1_lam q_abs does NOT pin to the dwell constant
  (cube-single: sits well below 43 with signed cross-state variation),
  and success does not collapse to the visitation-from-scratch floor.
- SECONDARY: set-level calibration vs Monte-Carlo region masses (L1fix
  harness) — multistep should be no worse than the two-phase λ=0 cells.
- k-flatness ablation {16,64,256} on the winner (mechanism test: sharp
  k-tuning falsifies the story).
- NULL: both multistep arms collapse like m1_base ⇒ the observed-term
  argument is wrong or the implementation buries it — stop, no wave M2.

Wave M2 (only if M1 primary passes): cube-double + scene, hindsight
test-points, vs the §9.16 hindsight anchors (.53/.93); then the dwell head
(stage 2) as its own pre-registered arm.

## 7. Open questions carried forward

- Readout algebra with λ>0: calibration is against the λ-return fixed
  point (note §5 caveat); quote λ next to any calibration table.
- Dwell head (stage 2) parameterization: D=exp(MLP(goal-block coords))
  vs per-column softplus gain; identifiability via H≤1 (sigmoid) — decide
  when speccing stage 2, after M1 evidence.
- Torch-stack port: only if the jax M1/M2 story lands (stack decision
  memory: jax is the method artifact).
