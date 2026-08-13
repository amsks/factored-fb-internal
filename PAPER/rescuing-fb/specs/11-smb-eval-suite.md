# Spec 11 — SMB: a successor-measure benchmark on cube play data

**Purpose.** One frozen evaluation that (i) covers everything a goal-reaching algorithm
and a successor measure should each do, (ii) *separates* them by construction —
GC-IQL-class methods must fail the occupancy tiers structurally, the current two-phase
method must fail the full-measure tiers structurally — and (iii) defines the exact gap
any future modification (spec 10) is required to close. Every future iteration runs this
suite unchanged; no more per-campaign ad-hoc reward grids.

Design inputs: the six reward families already in the note (Table `tab:rewards`), the
7-gap capability list, spec 10, and the literature scrape (§7 below, filled from the
survey). Environment: `cube-single-play-v0` first (all machinery exists), with the task
JSON parameterized so cube-double/scene can be added via `env_geometry` later.

---

## 0. Two wings

- **Wing A — Control**: task specified by m=2000 reward-labeled dataset samples
  (current protocol); policy extracted zero-shot; batched rollouts. Primary metric:
  **achieved discounted reward vs an oracle reference** (γ=0.99, horizon 200), because
  success flags cannot score occupancy/duration/signed tasks; success flags kept
  secondary where defined.
- **Wing B — Measure calibration (no extraction, rollout ground truth)**: the wing no
  published FB eval has. For a *fixed* conditioning (goal g or task vector z), roll the
  checkpoint's own policy and compare its **predicted discounted occupancies** against
  the empirical discounted visitation computed from those same rollouts. One rollout
  scores *hundreds* of query regions simultaneously, so this is cheap and
  statistically strong. This is where "is it actually a measure?" is decided —
  independent of actor quality, decode quality, and reward-inference quality.

The two wings deconfound the classic ambiguity: a control failure can be inference,
actor, or object; a Wing-B failure is the object.

---

## 1. Wing A tiers (exact definitions)

Notation: c(x) = cube position (obs units, `obs[19:22]`, 1 unit = 10 cm); ε = 0.4
(4 cm); all rewards are functions of dataset states; oracle values are Monte-Carlo,
defined per tier.

**T0 — Native GC (sanity anchor).** The 5 OGBench `cube-single` tasks, goal handed to
the policy. 50 eps/task. Metric: overall success. Everything passes; frames the rest.

**T1 — Goal-reducible zero-shot.** Families `dirac` and `dense` (σ=8 cm) exactly as in
the note. Routes: retrieval-1 for goal-conditioned methods, direct-z for ZX. Metric:
success + achieved discounted reward.

**T2 — Candidate composition (either-of).** Family `mixture` (all 10 task-goal pairs),
GPI over the 2 candidates. Metric: success-at-either + achieved reward. Any Q(s,a,g)
method passes; this tier is a *floor*, not a discriminator.

**T3 — Occupancy-weighted choice ★ (the R-discriminator; NEW).**
Two rewarded ε-balls with scalar weights: r(x) = w_S·1[c(x)∈B(g_S)] + w_T·1[c(x)∈B(g_T)].
- g_S = a **stable** region: an on-table cube position (cube dwells indefinitely once
  placed → dataset post-arrival occupancy in the "settled" mode, R̂_S ≳ 30).
- g_T = a **transient** region: an elevated cube position (table z + 8 cm; the cube
  passes through during transport, cannot rest → "pass-through" mode, R̂_T ≲ 5).
- Weights chosen from the **dataset** renewal statistics so the two rankings disagree:
  w_T/w_S strictly between the reaching ratio and the occupancy ratio, i.e.
  E[γ^T_S]·w_S < E[γ^T_T]·w_T (reaching/retrieval ranking prefers T) but
  Q_S·R_S·w_S > Q_T·R_T·w_T (occupancy ranking prefers S). Default w_S=1, w_T=4,
  **validated and frozen by `smb_tasks.py` from MC stats before any checkpoint sees
  the task** (5 pairs across the workspace, sanity: R-gap ≥ 5× per pair).
- Extraction: GPI over the two candidate conditionings using each method's own readout
  (GC-IQL: reward-weighted Q_reach; measure methods: reward-integrated M̃).
- Metrics: achieved discounted reward / oracle (oracle = the better single-region
  policy's MC return, computed by rolling both dedicated policies); chose-correct-region
  rate.
- **Why the baseline must fail:** E[γ^T] contains no revisit mass; on our data the
  reaching term predicts occupancy at rank ρ≈0.06. A method whose object saturates at
  arrival is structurally blind to the w·R tradeoff. This is the tier the user asked
  for: "combination goal not possible with the baseline, since they are single-goal
  algorithms" — realized as an *occupancy-vs-arrival* combination, which is the honest
  discriminator (a plain either-of mixture is NOT one; GPI over Q solves it).

**T4 — Extended masses (region + union).**
- `region`: the existing 6 cm-box / 4 cm-ball families.
- `union3` (NEW): r = 1[c(x) ∈ B(g₁)∪B(g₂)∪B(g₃)], three balls spread across the
  workspace. Success = end in any ball; primary = achieved reward.
- Discriminator: retrieval collapses the set to one sample; measure integration should
  value the whole mass (nearest/easiest member chosen state-dependently).

**T5 — Signed rewards, corridor reach-avoid (REDESIGNED).**
r(x) = 1[c(x)∈B(g_A)] − λ·1[c(x)∈B_avoid], λ ∈ {1, 5}.
- B_avoid is placed **on the transport corridor**: `smb_tasks.py` picks the
  highest-dataset-density cube region on the straight line between the task start cube
  position and g_A (radius 5 cm). This makes the penalty *bind*: the old failure mode
  (time-in-B ≤ 0.003 for every policy including BC) is a task-design artifact.
- Controls shipped with the tier: (i) BC-floor policy time-in-B (does the corridor
  actually get crossed by default? must be ≥ 0.05 or the task is void); (ii) the
  λ=0 ablation (same A, no penalty) to price the detour.
- Metrics: achieved signed discounted reward; success-at-A; time-in-B; deflection =
  time-in-B(λ) / time-in-B(λ=0).

**T6 — Feature-linear rewards.** r(x) = ψ(x)ᵀz̃/√d, 3 random unit z̃, exactly the
existing family (the honest FB-native case). Metric: achieved discounted reward vs the
random-z control, plus (new) vs the **best-candidate control** (GPI over the 5 native
goal conditionings scored by the linear reward) — the second control separates "learned
something about z̃" from "any reasonable policy scores".

**T7 — Dwell tasks (metric variant, no new family).** Every T1/T3/T4 rollout is
additionally scored by discounted time-in-rewarded-region. For transient regions
arrival ≠ occupancy; this column is where reaching-only methods overreport.

---

## 2. Wing B tiers (measure calibration)

Protocol: for each conditioning c ∈ {5 native task goals} (as raw g, ψ(g), or z per
method), reset the singletask env, roll the checkpoint's own policy 25 episodes × 200
steps, record cube trajectories. Ground truth for any query set X:
ĝt(X) = mean over episodes of Σ_t γ^t 1[c(s_{t+1}) ∈ X]. Query battery (frozen in the
task JSON): 64 ε-balls at dataset cube positions (stratified over the workspace incl.
elevated), 32 random axis-aligned boxes (side 4–12 cm), 8 unions-of-2-balls.

**B1 — Diagonal calibration.** Predict M̂(s₀,a₀,g) for the *conditioned* ball only.
Metrics: Spearman across the 5 goals × resets; median relative error raw (occupancy
units) and after one global affine fit.
- Phase-2 should be calibrated raw (the ≈24 vs 24.7 scale match, made per-goal).
- Phase-1 / GC-IQL have E[γ^T]-affine objects: report their *raw* failure (they are not
  in occupancy units — this line quantifies "a goal-reaching value is not a measure")
  and their affine-fit rank (can they at least rank?; prediction: poorly, ρ≈0.06-style,
  wherever R varies).

**B2 — Off-diagonal / arbitrary-query calibration ★ (the full-measure test).**
Hold the conditioning fixed, sweep the whole query battery.
- Diagonal-form checkpoints **cannot express this** (query is tied to conditioning);
  they are scored by their only available surrogate — evaluating the critic at column
  x under conditioning x (the wrong policy) — and that mismatch IS the measurement of
  the diagonal-slice deficiency.
- ZX checkpoints predict F(s₀,a₀,z)ᵀB(x_q)/√d directly. Prediction: decent rank,
  uncalibrated level for non-ball queries (kernel mass ≠ density); the spec-10
  modification (N̂ ratio, region columns, additive B) is *required* to pass calibrated.
- Metrics: Spearman + relative error per query family (balls / boxes / unions), and
  additivity defect for unions: |M̂(X₁∪X₂) − M̂(X₁) − M̂(X₂)| / ĝt(X₁∪X₂) (balls
  disjoint by construction).

**B3 — Reward-integration accuracy (∫ r dM).** For each Wing-A task and each candidate
conditioning z: predicted Q̂_r(s₀,z) = Σ_j w_j r_j M̂(s₀,a₀,z,g_j) (w_j = plain 1/m
now; 1/N̂ after the modification) vs the MC discounted reward of π_z's own rollouts.
Metrics: rank correlation across (task, z) pairs — "does the integral rank policies?" —
and regression slope/R² — "is it a calibrated value?". This is the single number that
licenses or kills the ∫r dM claim, per method.

---

## 3. Pre-registered capability matrix

✓ = passes (tier-specific bar), ✗ = fails structurally (predicted), ~ = partial.

| Tier | GC-IQL bilinear (retrieval/GPI) | Phase-1 FP diag | Phase-2 vis diag | Phase-2 vis ZX | spec-10 target |
|------|------|------|------|------|------|
| T0 native | ✓ .93 | ✓ .97 | ✓ .94 | ✓ ~.84/.71 | ✓ |
| T1 dirac/dense | ✓ | ✓ | ✓ | ✓ | ✓ |
| T2 mixture GPI | ✓ | ✓ | ✓ | ✓ | ✓ |
| T3 occupancy-choice | **✗** | **✗** | **✓** | ✓ | ✓ |
| T4 region/union | ~ .73–.80 | ~ | ~ | ? | ✓ ≥ .85 |
| T5 corridor signed | ✗ | ✗ | ✗ | **✗** | ✓ deflection |
| T6 linear | ✗ | ✗ | ✗ | **~ unresolved** | ✓ beats both controls |
| B1 diag calib | **✗ raw** | ✗ raw | **✓** | ✓ | ✓ |
| B2 arbitrary query | ✗ inexpressible | ✗ | **✗ inexpressible** | ~ rank-only | **✓ calibrated** |
| B3 ∫r dM | ✗ | ✗ | ~ diag-only | ~ | **✓** |

The three bolded columns are the story: T3+B1 = what a measure buys over the
goal-reaching baseline (2-phase ✓, GC-IQL ✗). T5/T6/B2/B3 = what the *full* measure
buys over the current 2-phase (2-phase ✗/~, modification must turn exactly these ✓ and
nothing else may regress). If T3 does NOT separate GC-IQL from phase-2, or B2/B3 do not
separate ZX from the modification, the suite has failed its purpose and gets redesigned
before any method iteration.

## 4. Checkpoint rows (existing, 500k unless noted)

| Row | Group / runs |
|-----|--------------|
| GC-IQL anchor | `exp_spec08/.../GCIQL-State-Flow-Bilinear-d50` |
| Phase-1 FP diag | `exp_spec06/.../Measure-FirstPassage-NoOrtho-d50/m_firstpassage_noortho_s{0,1,2}` |
| Phase-2 vis diag | `exp_spec07/.../Measure-BallVisit-FrozenPsi-d50/vis_s{0,1}` |
| Phase-1 ZX | `exp_spec07/.../Measure-FP-ZX-d50/zx_s{0,1}` |
| Phase-2 vis ZX | `exp_spec08/.../Measure-BallVisit-ZX-d50/visz_s{0,1}` |
| (optional) frozen-actor vis | `exp_spec08/.../BallVisit-FrozenActor-d50/faz_s{0,1}` |

FB-torch (0.42 anchor) is out of scope for v1 (different stack); its published numbers
are cited as context only.

## 5. Implementation plan

Frozen artifact first, harness second, runs third. All under
`scripts/rescuing_fb/`, tests under `tests/`, TDD, `.venv/bin/python -m pytest`.

1. **`smb_tasks.py`** — builds `analysis/rescuing-fb/smb_cube_single_v1.json` ONCE from
   the dataset: T3 pairs (settled vs pass-through regions from post-arrival MC stats,
   the renewal_head machinery; weight validation), T4 unions, T5 corridor centers
   (density argmax on the start→goal segment), Wing-B query battery. The JSON is the
   benchmark; it is committed and never regenerated silently (a `--check` mode verifies
   reproducibility).
2. **`eval_smb.py`** (Wing A) — wraps/extends `eval_zeroshot_rewards.py` machinery
   (loaders, batched rollouts, GPI): new families `occhoice`, `union3`, `corridor`;
   achieved-discounted-reward primary metric everywhere; T6 best-candidate control;
   per-tier JSON out.
3. **`eval_smb_measure.py`** (Wing B) — rollout recorder + query-battery scorer +
   per-method predictor adapters (diag critic, ZX critic, GC-IQL Q); outputs the
   B1–B3 metrics.
4. **`run_smb.sh`** — loops rows × tiers, tmux, one GPU; ~hours not days (all
   eval-only).
5. Post-run: capability-matrix table auto-emitted (`smb_matrix.py`), added to
   `results/` CSVs via `results_map.py`, DECISIONS.md entry.

## 6. The tabular companion (for the note)

A minimal MDP that exhibits the structures the method builds in, with everything exact
(linear solves) — see `scripts/rescuing_fb/tabular_playpen.py` and the numbers it
emits. The **playpen MDP**: a 7×7 "table" gridworld plus a 7-cell "shelf" row.
Dynamics: 5 actions (LRUD+stay); table cells inside a 2×2 **bowl** are sticky (move
attempts leave with prob 0.3 — the cube-at-rest analog); **shelf** cells eject to the
table row below on the next step regardless of action (the cube-in-air analog);
elsewhere deterministic with slip 0.05. Behavior policy: uniform random (play). γ=0.95.
Demonstrations, in the note's order:
1. **Occupancy = reaching × recurrence, exactly** (singleton goals; Markov-chain
   renewal identity), with the play-data statistic reproduced: recurrence variance
   dominates M's magnitude (rank(R,M) high, rank(Q,M) low) while argmax_a M = argmax_a Q
   at the sticky cells and *disagrees* at shelf cells — the E5 refutation in miniature.
2. **The diagonal slice is not the measure**: the T3 occupancy-choice task solved
   exactly — arrival ranking picks the shelf, occupancy ranking picks the bowl; and
   slice-sum vs fixed-z valuation shown to disagree.
3. **Estimator face-off at low rank** (d=8 < |S|=56, same F,B, same offline data,
   minibatch SGD): density-TD without ortho (scale trace), with ortho (goal-ranking
   distortion), vs absorbing→visitation regression (converges, calibrated). Reported
   honestly — whatever the tabular run shows is what goes in the note.

## 7. Literature alignment (survey completed 2026-07-26)

Scraped protocols: FB'21, "Does zero-shot RL exist?"'23, Blier-Ollivier'21, FB-CPR/Meta
Motivo, HILP, PSM, TD-JEPA, SF+GPI/USFA, OGBench, FRE, VC-FB, Soft-FB-for-general-
utilities. Key conclusions incorporated into SMB:

- **Wing B is empty territory.** Across the entire lineage there are ZERO quantitative
  occupancy-prediction, density-calibration, or ∫r dM̂-vs-MC metrics — every paper
  routes through rollout reward. The only partial exceptions: Soft-FB reports a
  correlation between M̂-estimated and true performance on didactic envs (because its
  inference *plans through* M̂), and VC-FB plots log-Q vs realized IQM. B1–B3 are the
  first systematic version. (PSM — whose object IS the measure — never evaluates it as
  one.)
- **Signed/avoid is claimed, never scored**: FB'21's reach-while-avoid exists only as
  contour-plot demos; the SF line's negative weights are event-avoidance, never spatial
  regions. T5 is the first scored spatial version. Compromise composites: one
  quantitative instance ever (FB-CPR appendix, 74% of TD3). Dwell/path: only FRE's 3
  path-following tasks. T3/T7 extend these.
- **Protocol anchors adopted for comparability**:
  1. **Headline normalization = % of the matched supervised baseline** (Touati'23's
     "% of offline TD3"; ours: % of the GC-IQL bilinear-d50 anchor trained on the same
     data — already our table convention).
  2. **m=10⁴ reward-labeled samples** is the canonical budget (we use 2000; add a
     10⁴ arm for citability). TD-JEPA converts OGBench goals to reward tasks and finds
     a **+1 reward shift materially changes inference** — include the shift as a
     robustness probe (a calibrated ∫r dM must be shift-equivariant:
     Q_{r+c} = Q_r + c/(1−γ); Wing B3 tests this for free).
  3. **True-z vs inferred-z ablation** (new cross-cutting axis A-z): every Wing-A tier
     runs once with the oracle conditioning (ψ(g*) / best candidate) and once with the
     inferred one. No paper isolates the inference stage; TD-JEPA's shift quirk and
     FRE's FB-collapse-on-sparse-goals (0.0) both indicate it matters. This decomposes
     every control failure into inference vs object/actor.
  4. **Inference-route menu**: moment, ridge (ours), retrieval (HILP-G precedent),
     exp-tilted weighting (FB-CPR's z ∝ E[exp(10r)·B·r] — worth one arm), and
     **z-search ranked by the learned measure itself** (Soft-FB precedent — exactly
     spec-10's route, now citable).
  5. **Reporting**: IQM + bootstrap CIs over the task×seed grid (HILP/rliable);
     OGBench's internally-computed graded `sum(successes)` is recorded as the
     partial-credit column (the benchmark flag discards it — free signal).
- **USFA's candidate-set ablation** maps to our GPI axis: report each tier with
  candidate sets {inferred z only} vs {+ native goals} vs {+ refined z} — how much GPI
  rescues the representation is itself a finding.
