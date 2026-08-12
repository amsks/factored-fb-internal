# Spec 24 — `measure_density_z`: z-indexed Density-FB, tested on point-mass-maze

Status: WRITTEN 2026-08-12. Parents: spec 18 (density measure, g-indexed,
IMPLEMENTED), spec 23 P0 (ExoRL point-mass-maze RND anchors: g-indexed
density .315, stock GCIQL .242, BC-only .235, seed 0, 500k, gates G0a/G0b
passed), stack-divergence audit 2026-08-05 (z-conditioned actors floored
multi-object manipulation — which is why this experiment runs on
point-mass-maze, NOT cube). TDD; default-off bit-identical.

## 0. One paragraph

The density measure currently indexes its policy family by goals; FB
indexes by a latent task vector z. This spec swaps the index and nothing
else: z is sampled FB-style from the backward embedding of relabeled
future configurations (mixed with random directions), F and the actor
condition on z instead of g, and the CE/softmax estimator is byte-identical
otherwise. The question it isolates: is the density estimator a drop-in
replacement inside FB's FULL interface (z-family + reward-to-index
assignment), or is the goal indexing itself part of what works on top of
the estimator? Point-mass-maze is the deconfounded venue: it is FB's
native regime where z-machinery is known healthy, so a failure is
attributable to the estimator-x-z interaction rather than to z being
broken, and our own record says z-conditioning fails on manipulation for
unrelated reasons. Pre-registered readings against the P0 anchors:
matching the g-indexed .315 = full FB interface restored on the density
estimator; landing at the BC floor .235 = the goal index is load-bearing,
not just the estimator.

## 1. Definitions (pin exactly)

Let x = the hit-spec block coordinates (point-mass-maze: the agent xy —
reuse whatever hit_spec spec 23's P0 config records; read it, do not
re-derive), B̄ the TARGET backward tower, d = measure_density_dim.

    z(x)      = sqrt(d) * B̄(x̃) / ||B̄(x̃)||           (x̃ = eps-scaled block coords; stop-gradded)
    z_i       = z(x_{g,i})  with prob 1 - z_rand_frac  (x_{g,i} = block coords of value_goals row i)
                ~ sqrt(d) * Uniform(S^{d-1})  otherwise (fresh per row per step)

FB's convention: ||z|| = sqrt(d); the mix ratio is a knob (default
z_rand_frac = 0.5, FB's standard mix). z is an INPUT, never
differentiated; built from the target tower so the index does not chase
the online tower within a step.

Training (the only changes vs spec 18; everything else byte-identical):

    scores      f_ij = F(s_i, a_i, z_i)^T B(ũ_j)/τ        (z replaces g in F's input)
    bootstrap   a+_i ~ π̄(· | s'_i, z_i)                    (same z, target actor)
    CE target   t_i  = sg[(1-γ)κ_i + γ w̄(s'_i, a+_i, z_i)]  (κ soft or hard, unchanged)
    actor q     q_i  = Σ_j w_j(s_i, π(s_i,z_i), z_i) · ( B̃(u_j)^T z_i / sqrt(d) )

The actor line is the z-analog of the goal-mass objective: the reward
implied by index z is the linear function r_z(u) = B(u)^T z (FB's reward
model), and the actor maximizes its expectation under the predicted
occupancy — the density readout of r_z. B̃ here is the TARGET tower,
stop-gradded, matching the frozen-params discipline of the g-indexed
actor branch (grad-hygiene test §5.5). Everything stays inside the
standard flow-BC actor objective (density-fb.tex eq actor).

## 2. Knobs (`tools/wandb_mode_shim/gciql_flow.py`)

```python
config.measure_density_z = False   # spec 24: index the density policy family by FB-style
  # latent z instead of goals. z = sqrt(d)*normalize(target_B(goal block coords)) mixed with
  # random sphere draws; F and the actor condition on z (input width d, not obs width).
  # Requires measure_density; mutually exclusive with measure_density_behavior.
  # False = off, bit-identical.
config.measure_density_z_rand_frac = 0.5   # fraction of rows with uniform-sphere z
```

Lattice (extend the spec-18 block; this is a DELIBERATE, scoped
relaxation of the no-z rule — document the stack-divergence provenance
inline):

```python
if config.get('measure_density_z', False):
    assert config.get('measure_density', False)
    assert not config.get('measure_density_behavior', False)
    assert 0.0 <= float(config.get('measure_density_z_rand_frac', 0.5)) <= 1.0
    # scoped z-relaxation: the blanket z-knob refusals (fb_forward_z, actor_z_infer, ...)
    # stay refused -- this knob is the ONLY z path, and it exists for the spec-24
    # point-mass-maze question. Manipulation configs with this knob are out of scope
    # round 1 (stack-divergence 2026-08-05); refuse env_name containing 'cube' or
    # 'scene' at the LAUNCHER level (config comment), not in the agent.
```

Architecture: when on, F's example input is `[obs, actions, z(d)]` and the
actor's goal input is replaced by z (actor network width changes — mirror
how `measure_density_behavior` changes F's input; the actor needs the same
treatment here, which behavior mode did not because it dropped the q-term
instead). GCDataset untouched: z is built inside the losses from the
existing `value_goals`/`actor_goals` block coordinates.

## 3. Evaluation on point-mass-maze (the PMM eval)

All on the ExoRL point_mass_maze RND seed-0 buffer, exactly the spec 23 P0
protocol (read `configs/gciql/point_mass_maze_rnd_seed0_*.yaml` and the P0
harness; same episodes, horizon, eval machinery).

**E1 — goal reaching (the P0-comparable number).** Condition the actor by
direct goal encoding: z* = z(φ(g_eval)) for each eval goal. This is the
z-analog of direct goal conditioning; no reward samples needed. Quote
next to the anchors (g-density .315, GCIQL .242, BC .235).

**E2 — reward-to-index assignment, two arms on the reward families**
(sparse/dense/held-out-region built by the density zero-shot harness on
maze coordinates — the eval script generalizes; add maze hit_spec):
- E2a analytic: z_r = sqrt(d) * normalize( E_ρ[r(x) B̄(x̃)] ) over the
  relabel pool — FB's linear reward embedding, the honest test of the
  linear index on this estimator.
- E2b search: score candidate indices {z(u_j)} for pool samples u_j with
  the pool readout Q̂_r(s0, ·) and take the argmax — the z-space analog of
  the g-indexed retrieval decode.
Readout formula unchanged (pool of 16,384; P16384 protocol; deterministic;
quote family means, 25 episodes, 3 seeds).

**E3 — health.** density_ess_median, CE, k_self as in spec 18 G2; plus
`z_cos_spread` (mean pairwise |cos| of batch z's — an index-collapse
instrument; log it, gate below).

## 4. Pre-registered wave and gates

- `z24_pmm_dens_s{0,1,2}` — measure_density_z on the P0 density config
  (soft immediate, matching the P0 arm so the ONLY diff is the index),
  500k, standing recipe, distinct run_group
  `GCIQL-PointMassMaze-RND-Flow-Measure-DensityZ-d50`.
- Optional screen (1 seed) `z24_pmm_dens_hard_s0` if the soft arm passes G1.

Gates:
- G1 (interface): E1 goal success within 0.05 of the g-indexed anchor
  (.315) at 500k ⇒ full-FB-interface-restored reading. Between BC (.235)
  and that band ⇒ partial. At or below BC ⇒ goal indexing is load-bearing.
- G2 (health): spec-18 G2 instruments in healthy ranges AND z_cos_spread
  not → 1 (index collapse).
- G3 (index assignment): E2a within 0.05 of E2b family means ⇒ the linear
  z_r embedding survives the density estimator; E2b ≫ E2a ⇒ the index is
  fine but the ANALYTIC assignment is not (search required, FB's headline
  interface only partially restored). Both ≪ E1 ⇒ reward-to-index is the
  bottleneck, not the family.
- NULL: G1 fails ⇒ quote as a RESULT (the density estimator does not
  transfer to z-indexing even where z works for FB); do not patch with
  ortho/anchors (anti-scope).

Interpretation notes (write into the collection ledger): E1 vs anchors
separates family quality; E2a vs E2b separates index quality from
assignment quality. If everything passes, the paper claim upgrades from
"a normalized successor measure with a goal-indexed family" to "a drop-in
FB estimator"; run walker (spec 23 P1) with the z arm before making that
claim in writing.

## 5. Tests — `tests/test_gciql_measure_density_z.py` (mirror the spec-18 suite)

1. default-off bit-identity (param tree + total_loss; z knob off ≡ spec-18
   arm).
2. z construction: ||z_i|| = sqrt(d) for both mix branches; rand_frac=0/1
   degenerate cases; z built from TARGET B̄ (perturb online B, z invariant).
3. CE numpy reference with z replacing g (both soft and hard κ).
4. actor-q numpy reference: q = Σ w_j (B̃(u_j)^T z)/sqrt(d) at frozen
   params; grad(actor_loss) ≈ 0 over both towers and targets, > 0 over
   actor; grad(density_loss) ≈ 0 over actor and through z (z stop-grad).
5. E1 goal encoding determinism: z*(g) reproducible from a checkpoint.
6. Lattice refusals: without measure_density; with behavior mode;
   rand_frac out of range. Accepts base.
7. 30-update end-to-end: finite, simplex, no NaN, z_cos_spread < 1.

## 6. Anti-scope

No manipulation (cube/scene) z-configs in round 1 (stack-divergence); no
ortho or normalization additions to B for the z path (fp-noortho lesson —
if the sphere-normalized index needs B-conditioning to work, that is a
finding); no learned z-inference networks; no z_rand_frac sweep (one
value, 0.5); no changes to the g-indexed paths (bit-identity is a test);
walker/cheetah z arms gated on G1+G3 (spec 23 P1 owns those slots).
