# HIQL × bilinear — boundary mechanism (M6 addendum) — design

_2026-07-07. Addendum to `2026-07-07-ogbench-sota-iclr-design.md` (Act 2/3): why HIQL
barely benefits from bilinear factorization when GCIQL (+26pts) and GCIVL (+18–30pts) do.
Grounded in the goal_rep probe results below (`scripts/probe_hiql_goalrep.py`,
`analysis/hiql_goalrep/`)._

## The three puzzles (cube-single, sg10, 3 seeds @500k)

| cell | MLP | bilinear-d50 |
|---|---|---|
| native AWR | ~0.17 | ~0.21 |
| flow-both | ~0.26 | **~0.14** |

1. Native bilinear helps only +4pts (vs +26 GCIQL / +18–30 GCIVL).
2. With flow actors bilinear flips to the WORST cell (−12) — opposite of GCIQL/GCIVL.
3. HIQL is floored on cube regardless: published 15±3 (OGBench, near-worst method), our
   MLP sg10 reproduces it; no HIQL cell exceeds 0.36 while flat methods reach 0.79–0.93.
   Published context: HIQL owns antmaze-medium (96±1) and is mid-pack on scene (38±3) —
   the floor is CUBE-specific (free-object transport), not manipulation-general.

**Matched control that anchors everything:** GCIVL ≈ HIQL-low (same V-only expectile TD,
same AWR readout) but Schaul-form bilinear ψ(g), free-standing head → +18pts. The two
structural deltas to HIQL are (a) the ψ input form, (b) the shared goal_rep. Any
explanation must live in (a) or (b) — "AWR readout can't exploit a better V" is refuted.

## Probe results (evidence in hand, 11 ckpts @500k)

`goal_rep` (rep_dim=10, LengthNormalized; sole gradient source = value loss; actors
stop-grad it) probed on dataset (s,g) pairs at three horizons; metrics: participation
ratio (PR), held-out ridge R² rep→goal-cube-xyz, and Spearman(rep-space distances,
goal-cube-xyz distances) at fixed s ("navigability" of the high actor's output space):

| cell (success) | PR | R²goal sub | R²goal rnd | geomSp |
|---|---|---|---|---|
| native MLP (0.17) | 3.38 | 0.311 | 0.141 | 0.115 |
| native bilin (0.21) | 4.01 | 0.287 | 0.102 | 0.087 |
| flow MLP (0.26) | 3.52 | 0.313 | 0.149 | 0.119 |
| flow bilin (0.14) | 4.33 | 0.296 | **0.078** | **0.059** |

- **No rank collapse** (PR rises under bilinear) — H1's strong "rep starvation" form dead.
- **Bilinear warps GLOBAL rep geometry, spares local**: subgoal-scale decodability ≈
  untouched; distant-goal decodability −⅓; navigability −30–50%, worst in the worst cell.
- **All cells under-resolve the object**: best R²(goal cube) = 0.31 sub / 0.14 rnd. The
  10-dim rep is dominated by non-cube features everywhere (RLBrew interaction-variable
  thesis, hierarchy edition).
- Sanity: MLP rep metrics match across actor types (actors stop-grad the rep) ✓.
- Caveats: linear probes; flow-bilin n=2; flow-vs-native gap within bilinear may be seed
  noise (bilinear-vs-MLP direction is consistent across all 5 bilinear ckpts).

## Hypotheses (revised)

- **H1′ (mechanism for puzzle 2, supported):** bilinear training reshapes goal_rep into an
  inner-product-friendly embedding that degrades as a LONG-RANGE GENERATION space. The
  Gaussian high actor regresses a pointwise mean (insensitive); the flow high actor models
  a distribution over the warped manifold + √rep_dim renorm (sensitive) → flow×bilinear
  worst cell.
- **H2 (mechanism for puzzle 1, untested):** HIQL's `_BilinearRepValue` is the BVN form
  φ(s)·ψ(goal_rep([s;g])) — s on both sides ⇒ NOT low-rank over (s,g) ⇒ the structure that
  wins in GCIQL/GCIVL (Schaul ψ(g)) was never imposed. HIQL pays bilinear's cost without
  its constraint.
- **H3 (first-order on cube, established):** the binding constraint is hierarchical —
  goal_rep under-resolves the object/interaction variable at every horizon; value-side
  improvements are second-order. Predicts bilinear CAN help HIQL where it has headroom
  (antmaze 96, scene 38).
- H4 (AWR saturation): refuted by the GCIVL control; optionally confirm from wandb exp_a
  stats, no runs.

## Experiments

| id | cell | tests | cost | status |
|---|---|---|---|---|
| E-A | antmaze-medium HIQL {MLP, bilinear-d50} × 8 @1M, OGBench HPs (= pure defaults, sg25) | H1′/H2 vs H3: max headroom, no manipulation confound | ~16 GPU-h | **RUNNING** (tmux `amhiql`, groups `HIQL-AntMaze-Medium-{MLP,Bilinear-d50}`) |
| E-B | cube native bilinear, `rep_dim` 10→50 | is the warp a 10-dim bottleneck artifact? | 3s @500k, config-only | pending |
| E-C | cube Schaul-form HIQL: `V=φ(s)·ψ(g)` raw goal + `low_actor_rep_grad=True` (keeps goal_rep trained via the low actor since the value no longer trains it) | H2 directly; mirrors parent spec M3 in reverse | 3s @500k, small TDD knob | pending |
| E-D | cube two-head split: bilinear head for the HIGH query V(·,g), stock MLP head for value-TD/low path (MLP remains goal_rep's trainer) | H1′ surgically: factorization where goals are distant, rep trainer intact | 3s @500k, TDD (two-head value) | pending |
| E-E | goal_rep probe on E-A antmaze ckpts (+ scene HIQL ckpts when parent-spec grid 1b lands) | does the warp generalize; does it matter where HIQL works? | CPU-only, free | after E-A |

Decision rules:
- E-A bilinear ≥ MLP by a clear margin → puzzle 1 was H3 (ceiling hid the win); paper
  claim: "bilinear helps every method *with headroom*". E-A bilinear ≤ MLP → form/rep
  pathology (H1′/H2) is real and domain-general → E-C/E-D become P0.
- E-C recovers a GCIVL-sized win on cube → H2 confirmed; the paper's form-axis section
  (M3) gains the HIQL row. E-C ≈ option-A bilinear → form isn't it; H1′/H3 carry.
- E-D rescues flow×bilinear (≥ flow×MLP 0.26) → H1′ confirmed as "rep trainer" effect.
- E-B moves nothing → 10-dim bottleneck exonerated (strengthens H2 form story).
- Writing rule regardless: HIQL row in the SOTA table is annotated "method floored on
  cube (published 15±3); see mechanism section" — never presented as a bilinear failure.

## Code prerequisites (all TDD, default-OFF, vendored-tree edits mirror the existing
`value_arch` pattern in `third_party/ogbench/impls/agents/hiql.py`)

1. `bilinear_form: rep|goal` knob on `_BilinearRepValue` (E-C): `goal` feeds ψ the raw
   goal (bypasses goal_rep in the value only).
2. `value_split: shared|high_bilinear` knob (E-D): second value head for the high-actor
   advantage query; low advantage + TD loss stay on the MLP head. Target-network pair for
   both heads; high head trained on the same IVL expectile loss, queried only in
   `high_actor_loss`.
3. `rep_dim` is already a config field (E-B: config-only).
4. Configs + fresh `-v2`-style wandb groups per parent-spec hygiene rule.

## Non-goals

- No change to subgoal sampling, HGCDataset, or the stock MLP path (default-OFF knobs).
- No pixel path.
- Scene HIQL cells belong to the parent spec's grid 1b (E-E consumes their ckpts).
