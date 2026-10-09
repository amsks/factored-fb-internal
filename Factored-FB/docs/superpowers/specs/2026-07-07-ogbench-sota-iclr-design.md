# OGBench-SOTA bilinear → ICLR paper (Shape C) — design

_2026-07-07. Builds on `PAPER/ogbench-sota/audit-20260707.md` (trust audit: no code bugs;
param confound 2.06×; α confound on cube DDPG+BC; GCIVL flow-bilinear healthy batch =
0.792; HIQL sg10 rerun flipping positive; antmaze claim is method-relative)._

**Thesis (working title).** *Factorized values for offline goal-conditioned RL: when
bilinear wins, why, and when it doesn't.* Three acts: (1) a corrected conditional-SOTA
map at ICLR-grade protocol; (2) mechanism — generalization, spectrum, and the
Schaul-ψ(g)-vs-BVN-ψ(s,g) axis; (3) boundaries — the actor×domain flip and scene.

**Protocol rule (all Act-1 headline cells).** 8 seeds @1M, `evaluation/overall_success`,
eval protocol matched to OGBench (verify from the OGBench paper whether they average the
final k evaluation epochs; adopt exactly). Diagnostic/control cells: 4 seeds @500k.
**Fresh wandb group names with `-v2` suffix — never reuse a group** (audit anomaly #1/#2).
Ops: tmux, entity amsks, jax venv auto via `run_gciql.py`, no jax co-init.

---

## Act 1 — The corrected SOTA map (P0)

### 1a. Claims-hygiene reruns (fix audit confounds)
| Cell | Fix | Seeds |
|---|---|---|
| cube GCIQL bilinear-d50 + DDPG+BC, **α=1.0** | α confound (was 0.3 vs stock 1.0) | 8@1M |
| cube GCIVL bilinear + flow, clean relaunch | bad-batch pollution | 8@500k (GCIVL cells are 500k-scoped like their stock baselines) |
| cube **param-matched MLP** + flow: `value_hidden_dims=(768,768,768)` (≈2.25M ≈ bilinear) | the 2.06× param confound — THE reviewer-killer | 8@1M |
| cube **half-width bilinear**: hidden (360,360,360) (≈1.1M ≈ stock MLP) | same confound from the other side | 4@500k |
| HIQL sg10 MLP vs bilinear (extend in-flight sotagap cells) | sg25 bug | 8@1M |

### 1b. Complete the method × domain grid (rows currently `not run`)
| Domain | New cells (bilinear arm uses each method's best actor + stock actor) |
|---|---|
| antmaze-medium-navigate | GCIVL {stock, bilinear}; HIQL {MLP, bilinear} (at OGBench's tuned antmaze subgoal_steps — check hyperparameters.sh); GCIQL param-matched MLP + DDPG+BC |
| scene-play | GCIQL bilinear+DDPG+BC (in flight → extend to 8 seeds); GCIVL {stock, bilinear}; HIQL {MLP, bilinear} |
| **breadth** (P1) | cube-double-play GCIQL {stock, bilinear+flow}; humanoidmaze-medium-navigate GCIQL {stock, bilinear+DDPG+BC} — one manipulation + one navigation extension |

Rough compute: ~30 headline cells × 8 seeds ≈ 240 runs; cube ≈ 35 min/1M-run, antmaze ~1 h,
scene ~2 h → ≈ 4–6 days on 8 GPUs with a queue watcher. Trim rule if needed: breadth cells
drop to 4 seeds first.

### 1c. Claim calibration (writing rule, no runs)
- **Domain SOTA** language only where the bilinear cell beats the best *published number
  across all methods* for that task: currently cube (0.93–0.95 @1M vs 68). Antmaze is a
  *method-relative* win (published HIQL = 96). Scene TBD.
- **Related-work sweep (P0, no GPU):** deep-research pass over post-Oct-2024 literature
  reporting cube-single-play / antmaze-medium-navigate / scene-play state numbers
  (horizon-reduction/SHARSA, TRA, GC-flow methods, any 2025–26 OGBench-based papers), plus
  positioning vs BVN (Hong et al. 2022), Schaul UVFA, USFA, CRL, quasimetric RL. The SOTA
  table quotes the best *current* published number per task, not just OGBench-2024.

---

## Act 2 — Mechanism (P1)

### M1. Goal-split generalization (the BVN test, offline)
Offline analog of BVN's OOD-goal experiment: restrict relabeled goals during training to a
half-space (cube: goal cube x-position below median; antmaze: goal x below median), eval on
held-out-half goals. Cells: cube GCIQL {param-matched MLP, bilinear-d50} × {in-dist,
held-out} eval, 4 seeds @500k; antmaze same with DDPG+BC.
- _Code prereq (the one nontrivial engineering item):_ (i) a goal-predicate filter on the
  relabeled goal sampler (shim-level wrapper around GCDataset sampling — mask-and-resample
  goals by predicate; do NOT edit vendored datasets.py); (ii) custom eval-goal sets (drive
  the OGBench eval loop with a restricted/complementary task-goal list).
- _Prediction:_ if bilinear's win is generalization (BVN's story), its advantage *grows*
  on held-out goals; if the in-dist gap equals the OOD gap, the mechanism is optimization,
  and we can say something BVN couldn't.

### M2. Spectral probe (analysis only, checkpoints exist)
Sample Q(s,a,g) from the trained *MLP* critic on an (s,a)×g grid (dataset states/actions ×
dataset goals, ~2k×2k), SVD → effective rank & spectrum decay; likewise for the bilinear
critic (sanity: rank ≤ d). Compare to the empirical rank cliff (d5 dead, d50 fine ⇒
predicted effective rank ∈ (5, 50] on cube).
- _Reading:_ if the MLP's learned Q is already ≈rank-50, capacity was never the issue —
  bilinear is an optimization/generalization shortcut to a solution the MLP parameterizes
  but struggles to find. Repeat per domain (antmaze/scene checkpoints) → does intrinsic
  rank order domains the way the d-cliff does?

### M3. The form axis: ψ(g) vs ψ(s,g) (Schaul vs BVN)
New knob `psi_sg=True` in the GCIQL bilinear path (concat s into ψ's input; TDD). Cells:
GCIQL bilinear best-actor × {cube, antmaze, scene}, 4 seeds @500k.
- _Stakes:_ our winning form is BVN's *baseline* (they claim ψ(s,g) is required in complex
  envs — cube already contradicts that). Scene — multi-object, compositional goals, where
  bilinear currently *loses* (0.292 flow / DDPG+BC pending vs stock 0.516) — is exactly
  where BVN predicts ψ(s,g) matters. If ψ(s,g) rescues scene, Act 3's boundary becomes a
  mechanism win *and* a possible scene SOTA. HIQL's `_BilinearRepValue` (ψ∘goal_rep([s;g]))
  is already the BVN form — the grid becomes interpretable across methods.

### M4. Rank sweep off-cube
d ∈ {5, 50, 512} on antmaze (DDPG+BC) and scene (best actor), 3 seeds @500k → is the rank
cliff domain-dependent (intrinsic-dimension ordering)?

### M5. Task-embedding reading (mostly free, from project ③)
Cite/import: B1/B4/two-tower (FB structure survives on the IQL objective, 85–87), G4
(actor conditioned on z=ψ(g) → 0.93 — ψ is a functional task embedding; value-TD-trained
successor features). Optional new cell: G4-style z-conditioning on antmaze (2 seeds).

---

## Act 3 — Boundaries (P1/P2)

### A1. Actor×domain flip diagnostic
Flow wins cube (+20), dies on antmaze (−46). Cheapest discriminating sweep:
`bc_coeff ∈ {0.3, 1, 3, 10}` for antmaze bilinear+flow (2 seeds @500k) — is the flow
actor's collapse a BC-strength mistuning or intrinsic? Plus one diagnostic metric logged
everywhere: flow-BC distill MSE and Gaussian-actor log-prob on the same data (action
multimodality proxy). If bc sweep rescues antmaze flow → "actor needs per-domain BC
tuning" (weak boundary); if not → genuine actor×domain interaction (strong boundary,
report as practitioners' guidance).

### A2. Scene resolution
Sequenced: scene bilinear+DDPG+BC (in flight) → M3 ψ(s,g) on scene → if both fail, scene
is documented as the honest failure case with the M2 spectral analysis as the candidate
explanation (compositional goal structure ⇒ high intrinsic rank?).

### A3. Pixel (P2, optional)
One cell: cube pixel GCIQL {stock DrQ, bilinear DrQ} 3 seeds @500k (host-bound; keep to
one cell). Claim scope stays "state observations" if this slips.

---

## Code prerequisites (all TDD, default-OFF)
1. `psi_sg` knob (M3) — gciql_flow bilinear path.
2. Goal-predicate filter + custom eval-goal sets (M1) — shim-level, no vendored edits.
3. Configs: param-matched MLP (768³), half-width bilinear (360³), α=1.0 cube ddpgbc,
   `-v2` groups throughout.
4. Spectral probe script (M2) — checkpoint loader + SVD, no training code.

## Decision gates
- "Bilinear helps every method" goes in the abstract ONLY if HIQL sg10 holds at 8 seeds.
- Param-matched MLP within noise of stock MLP → confound dismissed in one table row; if it
  closes ≥half the gap → report both, claim shrinks to "factorization + params".
- M1 OOD gap > in-dist gap → lead the mechanism section with generalization; else lead
  with M2 optimization/spectrum.

## Sequencing
| Wk | Work |
|---|---|
| 1 | 1a hygiene reruns + 1b grid launches (queue watcher) · related-work sweep · M2 script on existing checkpoints · code prereqs 1–3 |
| 2 | M1 + M3 + M4 launches · A1 sweep · collect Act-1 @1M · refresh ogbench-sota.tex tables |
| 3 | Analysis + writing: mechanism figures (M1 curves, M2 spectra, rank-cliff-by-domain), boundary section, related-work table |
