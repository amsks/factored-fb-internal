# RG-FB — Reward-Grounded Forward-Backward: method + implementation + eval spec

*2026-07-16. Everything here is assembled from cells already validated in this repo; the new
work is the single end-to-end agent, the actor-vf fix, and the zero-shot-only evaluation
protocol. Pre-registered before launch.*

## 1. Method

**One-line:** FB's zero-shot API grounded by hindsight scalar expectile TD instead of
policy-conditioned measure TD.

### 1.1 Objects

- Value features **φ(s,a) ∈ R^d** (per ensemble head), task encoder **ψ(g) ∈ R^d**
  (= FB's B, trained through the TD loss). d = 50.
- Critic: **Q(s,a,z) = φ(s,a)·z/√d** with z = ψ(g) during training. φ is NOT z-conditioned
  (validated: fbz-under-IQL 0.848 < plain bilinear 0.904).
- Value net: V(s,z) = φ_V(s)·z/√d (bilinear V, same ψ; matches GCBilinearValue value path).
- Actor: FQL-style — BC flow vector field + one-shot noise actor conditioned on
  **z = sg(ψ(g))** (validated: G4 z-cond 0.921 @500k, n=12).

### 1.2 Losses (all existing code paths unless marked NEW)

1. **Grounding**: GC-IQL expectile TD, expectile 0.9, γ=0.99, rewards 0/−1, masks at goal,
   OGBench goal mix (value 0.2/0.5/0.3; actor uniform future). Exactly the bilinear-d50 recipe.
2. **Actor**: BC flow-matching + one-shot distillation (bc_coeff=3.0 cube) + −Q term
   (scale-normed). **NEW: the BC vector field and the distillation rollout are z-conditioned**
   (`actor_vf_cond='z'`): vf(s, z, x_t, t) with z = sg(ψ(g)) — same conditioning as the
   one-shot actor. Rationale: audit smell #1 (goal-agnostic anchor); required for cross-domain
   portability (antmaze), harmless-to-positive expected on cube.
3. **Optional, default OFF for v0**: low-λ scale-normed measure auxiliary
   (`measure_coef≤1.0`, `measure_scale_norm=True`) to preserve an honest successor-measure
   reading of φψᵀ — E2 showed λ≤1 is performance-neutral (0.86–0.92). Turn on in v1 only if
   the successor-measure semantics claim needs it for the paper.

### 1.3 Zero-shot inference (test time; the FB API)

Given reward samples {(sᵢ′, rᵢ)} from the offline dataset (relabeled, r ∈ {0,1}):

    z_raw = Σᵢ rᵢ · ψ(sᵢ′)          (dataset moment E_ρ[r·ψ])
    z     = z_raw · (‖ψ(g_ref)‖-scale hygiene)

Norm hygiene (validated in E1a): the actor consumed RAW ψ(g) during training and ‖ψ‖ is
task-dependent (≠ √d, no project_z), so rescale z to the typical ‖ψ‖ scale — implement as
rescaling to the mean ‖ψ(s′)‖ over the relabel sample (goal-free version of E1a's 'match_g';
E1a showed the norm choice is worth ~0.29). relabel_size = 50,000. Report BOTH norms as in
`scripts/eval_zeroshot_inference.py`.

**The eval protocol is zero-shot ONLY**: the deployed policy never sees the goal observation,
only the inferred z. (Training-time eval may track exact-z success as a diagnostic anchor,
clearly labeled `eval/exact_z_success` vs `eval/zeroshot_success`.)

## 2. Implementation plan (jax, `tools/wandb_mode_shim/gciql_flow.py`)

All new knobs **default-off**; existing checkpoints must keep loading byte-identically.

1. **`actor_vf_cond ∈ {'none' (default, current behavior), 'goal', 'z'}`**:
   - 'goal': vf and distillation rollout take the raw goal concatenated (parity with the
     `awr_flow` branch's pattern, gf:592).
   - 'z': vf takes sg(ψ(goal)) via the existing `_value_psi` path (same z the one-shot actor
     gets). Assert lattice: `actor_vf_cond='z'` requires `actor_cond='z'`;
     `actor_vf_cond≠'none'` is incompatible with `value_arch` lacking a ψ (mlp) unless 'goal'.
   - Init plumbing: vf `ex` inputs sized accordingly (gf:1512 region); `_flow_rollout` and the
     BC loss (gf:627-633, 471-479) thread the condition.
2. **Tests** (`tests/test_gciql_rgfb.py`): (a) default `actor_vf_cond='none'` produces
   bit-identical loss to current code on a fixed batch; (b) 'z' and 'goal' shapes + gradient
   flow (no grads into ψ from the vf condition — sg verified); (c) assert-lattice cases;
   (d) existing gciql_flow tests stay green.
3. **Zero-shot eval script** `scripts/rescuing_fb/eval_rgfb_zeroshot.py`: generalize
   `scripts/eval_zeroshot_inference.py` to (a) any checkpoint/config (read knobs from the run
   dir), (b) the goal-free norm rule above, (c) all 5 singletask cube tasks × 50 episodes ×
   3 seeds, (d) JSON out to `analysis/rescuing-fb/rgfb_zeroshot_<group>.json`.
4. **Config** `configs/gciql/cube_single_state_rgfb_d50.yaml`: bilinear d50 anchor config +
   `actor_cond=z` + `actor_vf_cond=z`. v0 has NO measure aux.
5. **Launch** `scripts/rescuing_fb/launch_rgfb_v0.sh`: 3 seeds, 500k (+eval every 50k),
   GPUs from the free set {0,2,3,4,7}, one process per GPU,
   `XLA_PYTHON_CLIENT_PREALLOCATE=false`, tmux `rgfb`,
   `output_root=/home/adityamohan/exp_rgfb`, wandb entity `amsks`,
   group `RGFB-Cube-Single-State-d50-v0`.

## 3. Pre-registered reads (before results exist)

| Metric @500k, 3 seeds | Success band | Failure band |
|---|---|---|
| `eval/exact_z_success` (diagnostic anchor) | ≥0.88 (G4 replication: 0.921) | <0.80 → the vf-conditioning change hurt; fall back `actor_vf_cond='none'` |
| **`zeroshot_success` (headline)** | **≥0.90** (E1a anchor 0.955 was d512/1M — d50/500k may sit slightly lower) | <0.75 → norm hygiene or d50-vs-d512 gap; run the d512 fallback |
| zeroshot ≈ exact-z gap | ≤3 pts | >10 pts → inference-side problem, isolate with the E1a v0/v1/v2 ladder |

Anchors to beat: plain FB 0.42 (own harness) / 62.0 (TD-JEPA RGB FB); supervised bilinear
GCIQL 0.904; published GCIQL 0.68.

**Failure-isolation order** (if v0 misses): (1) rerun with `actor_vf_cond='none'` (isolates the
new knob); (2) d512 + 1M (matches E1a's exact operating point); (3) E1a inference ladder on the
v0 checkpoint (isolates inference vs training).

## 4. v1+ queue (after v0 reads)

1. **v1-measure**: + `measure_coef∈{0.1,1.0}` scale-normed stopgrad-φ auxiliary → does the
   successor-measure semantics come for free at d50 too?
2. **T6 generality probe** on the v0 checkpoint: synthetic rewards — two-goal mixtures
   r = max(r_g1, r_g2), lift-anywhere (cube z > h), avoid-region penalties. Decides
   "zero-shot RL" vs "zero-shot goal-reaching" claim scope. Uses only inference + rollouts.
3. **Cross-domain**: antmaze-medium (gates on T4/T3 reads: z-conditioned vf expected to fix
   the flow collapse; else selection-based extraction), then scene (gated on its own
   diagnosis), then cube-double.
4. **Baselines for the paper** (torch side): plain FB (exists), faithful one-step FB (T8 —
   z-conditioned F, a′∼dataset, AWR extraction; NOTE the repo's `onestep` zeroed z and does
   not count), VC-FB-style conservatism (T7: min-aggregation / pessimism_penalty>0), zher
   (exists), HILP if time permits.

## 5. Risks / honesty notes

- The E1a 0.955 was measured on the G4 checkpoint (d512, 1M, actor_cond=z, actor_vf
  UN-conditioned). v0 changes three things at once (d50, 500k, vf-conditioning) — hence the
  isolation ladder in §3.
- Zero-shot claim scope is decided by T6, not asserted.
- Param-count confound vs MLP (bilinear ≈2× trunk params) is a bridge/sota-paper control (T5),
  not an RG-FB blocker, but must be in the paper.

## 6. Claim taxonomy: RG-FB is NOT (yet) a behavior foundation model — pre-registered scope

v0 trains z only at ψ(g) for dataset goals: the trained task manifold is the hindsight
goal-reaching family. Zero-shot inference works for rewards whose moment-z lands on that
manifold (E1a); nothing is guaranteed off it. Therefore:

- **Licensed claim (v0)**: task-prior-explicit zero-shot RL — no test-time training, reward
  samples → z → policy, prior = goal-reaching. This fully covers the OGBench eval (which only
  tests the goal family). Do NOT print "behavior foundation model" for v0.
- **T6 decides the empirical scope** (goal mixtures / lift-anywhere / avoid-region on the v0
  checkpoint).
- **v1 BFM path — task-mixture hindsight relabeling (scalar-TD-native z-prior coverage)**:
  sample K goals + weights w, composite reward r = Σ wᵢ·1[s≈gᵢ], z = Σ wᵢψ(gᵢ)
  (moment-consistent by linearity), same expectile TD. Populates the z-space off the
  single-goal manifold with grounded targets (HER generalized from goals to tasks). Combined
  with the low-λ measure auxiliary (which supplies the measure semantics justifying
  z = E[r·ψ] for arbitrary r), this is the ingredient stack for a defensible BFM claim.
- Context: FB's own BFM validation is mostly goal-reaching + dense locomotion; every zero-shot
  method carries an implicit reward prior (Ollivier 2025, 2502.10792). RG-FB makes the prior
  explicit rather than narrower-in-practice.
