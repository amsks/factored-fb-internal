# What Density-FB can do that FB cannot — one perspective (2026-09-04)

Status: synthesis note, written after the cube bottleneck probe
(`docs/specs/2026-09-04-cube-bottleneck-mechanism-probe.md`,
`analysis/cube_bottleneck/cube-single-play-v0/RESULTS.md`). Each claim carries its
evidence and its status: **shown** (direct measurement), **consistent** (no direct test,
fits every measurement), **open** (the direct test was null or is pending).

## 1. The data property

OGBench play: the arm moves every step; the goal-relevant object does not.
Cube stationary on 48% of cube-single transitions (median run 44 steps), 72% on
cube-double, drawer/window 91% on scene (DENSITY-FB-EXPLAINED §2.1). The object
coordinates are 3 of 29 state dims and are **low-variance under the data
distribution ρ**; the task lives entirely in that slice.

## 2. The two objectives, as trained (from the code)

**FB** (`impls/critics/fb.py:121-150`), per batch of `(s_i, a_i, s'_i)`, task vector `z`:

    m_ij = F(s_i, a_i, z)ᵀ B(s'_j)                       (rank-d, unbounded scalar)
    L_FB = ½ mean_{i≠j} ( m_ij − γ m̄_ij )²  −  mean_i m_ii  +  c · ortho(B)
    ortho(B): ½ mean_{i≠j} (B_iᵀB_j)² − mean_i ‖B_i‖²    (B whitened under ρ)
    Q(s,a,z) = F(s,a,z)ᵀ z,   z = E_ρ[ r(x) B(x) ]  (goal: z ∝ B(g))

This is least-squares density-ratio estimation of `dM^π/dρ` over **full next states**:
the diagonal term is a Dirac at the row's own successor `s'_i` (arm included), the
off-diagonal term regresses the bootstrapped ratio against every other successor, and
the task is read out through a **linear ρ-moment of B**.

**Density-FB** (`impls/critics/density_fb.py:857-905`), goal `g` enters F directly,
query set `U` = the batch's own next states, `φ` = object coordinates:

    p(u_j | s_i, a_i, g) = softmax_j ( F(s_i,a_i,g)ᵀ B(u_j) / τ )        (a distribution)
    k_ij = softmax_j ( −½ ‖(φ(s'_i) − φ(u_j)) / ε‖² )                    (object kernel)
    t_ij = (1−γ) k_ij + γ p̄(u_j | s'_i, a'_i, g)                          (on the simplex)
    L_D  = − mean_i Σ_j t_ij log p(u_j | s_i, a_i, g)                     (cross-entropy)
    Q(s,a,g) = Σ_j p(u_j | s,a,g) · k(g, u_j)                              (∈ [0, 1])

Derivation in one line: the normalized successor distribution
`p^π(·|s,a) = (1−γ) Σ_t γ^t P(s_t = · | s,a)` satisfies
`p^π = (1−γ) P(s'|s,a) + γ E_{s',a'} p^π(·|s',a')`; measure it on object outcomes through
the kernel `k`, restrict the support to the sampled query set, and fit by cross-entropy.
`Q` is then `(1−γ)` times the discounted probability that the object reaches the goal
under the goal's own policy — a bounded, goal-specific value.

Three things differ, and they are separable:
(a) **what is predicted** — object outcomes vs full states;
(b) **how the task enters** — direct goal conditioning + kernel readout vs a linear
    ρ-moment index `z`;
(c) **how the fit is scored** — cross-entropy on a normalized distribution vs squared
    error on an unbounded ratio; and B trains *through* (c) with target (a), with no
    ρ-whitening term.

## 3. Where FB fails, and where it does not (cube, all measured)

| what | result | status |
|---|---|---|
| Acquisition | state FB grasps ≈ as often as GCIQL; from the grasp-adjacent states its Q ranks the gripper-open action last, AUC .91–.95 | shown (phase diagnostic; probe Test 2) |
| Scale | predicted total mass 87–105 vs true 100; ratio across phases .94–.98 | shown (probe; refutes "inflate scale") |
| Held start | handing FB the secured cube: Δ = +.02/+.10/+.03; 53% of failures put the cube back on the table mid-transport (median step 13–18), 47% still hold it 22 cm from the goal; gripper command at t=0 identical to dataset and Density-FB | shown (probe Test 1, R-HELD2) |
| B's resolution of the task variable | held-vs-free separability AUC .75 (PSM .94); R²(B→cube-goal distance) .39; no goal-centered transport basin | shown (task-distribution probe; fb_cube_failure Finding 3) |
| The index | z(g) across goals at cosine .998; AOSS mean-B direction 98%; centering alone rescues the point index; forcing Density-FB through z collapses it (.017 vs .736) | shown (spec28, spec30, spec24) |
| Ortho term | toxic on cube (first-passage no-ortho .967 vs with-ortho .225); helpful on ExoRL walker (RLDP live ortho → 62–94% of published) | shown |
| Coordinates alone | B fed cube xyz: FB .11→.34; same-φ LSIF on cube-double .300 vs categorical .736 | shown (1 seed for the LSIF control) |
| Dwell/imbalance as the *loss* mechanism | tabular fork-and-dwell null twice; FB's bilinear critic flat under dwell weight | **open** |

## 4. The perspective

**FB's failure on manipulation is goal-blindness after acquisition, caused by every
task-carrying quantity being routed through the low-variance object slice that a
ρ-driven estimator discards.** Concretely:

1. The Dirac target on full `s'` and the ρ-whitening ortho term make B a
   ρ-variance embedding: its d directions go to what moves every step, the arm and
   gripper. That is why acquisition works (gripper/contact are high-variance and
   well-resolved) and why the object-goal relation is not (AUC .75, R² .39).
2. `z = E_ρ[r B]` for a sparse object reward is a small signal on top of B's common
   direction, so goals collapse into one index and the actor's value is nearly
   goal-invariant. After the grasp there is no goal-specific gradient to follow: the
   policy carries the cube to a play-like location or lets go when the arm's own motion
   prior takes over. This is the 47%/53% split from a held start, and the missing
   "transport basin" in the value field.
3. The squared-ratio loss fits the dominant outcome (object where it already is) and
   spends little on the rare outcome (object moved); this is why coordinates alone
   recover only .34 and the same-φ LSIF control stops at .30. **This link is the one
   without a passing direct test**; keep it as a candidate factor in the paper.

**Density-FB removes each of the three at its source, and the removals are
separable by the existing controls:**

- (a) The target is a kernel over object outcomes, while B still consumes the full
  state and trains through the cross-entropy. Over a query set whose rows share object
  positions but differ in arm pose, the softmax can only lower the loss by making
  `B(u)` invariant to the arm and discriminative in the object. Anti-collapse comes
  from contrast over *outcomes*, not from whitening under ρ. Evidence: full-state
  Density-FB .91 ≈ privileged .94; no ortho term needed; the ortho toxin/benefit
  reversal between cube and walker is exactly (a) in both directions.
- (b) The goal enters F directly and the readout is a kernel expectation, bounded and
  goal-specific; the reward interface uses the centered solve. Evidence: spec24
  (Density-FB through z collapses to .017), spec28/30 (centering rescues z), and the
  probe: Density-FB's Q is flat over the grasp action (AUC .50) and it still succeeds,
  because success is decided by *where it carries the cube*, not by the grasp.
- (c) Cross-entropy on the simplex: bounded per-row gradients `p − t`, relative
  weighting, a target that cannot drift in scale. Evidence: the .300→.736 estimator
  share on cube-double (1 seed). Mechanism still open.

**What Density-FB can do that FB cannot, in one sentence:** predict *where the object
will be relative to a goal, under that goal's own policy*, from data in which the
object mostly does nothing — and that is precisely the quantity transport needs and
acquisition does not.

## 5. Extrapolation: what the density objective does elsewhere

The retrofits (`rldp_density.py`, `hilp_density.py`, `psm_density.py`) transplant
**only (c)**: their immediate target is `one_hot(i)` on the row's own *full-state*
successor, B/φ is stop-gradded (learned by the SP rollout + ortho, or by Hilbert
distances), and the z index is kept with `q = Σ_j p_j B(u_j)ᵀz`. Their results are the
cleanest confirmation of the decomposition:

| arm, cube-single | full state | object coords |
|---|---|---|
| Density-RLDP | .23 | .82 |
| Density-PSM | .17 | .88 |
| PSM | .12 | .94 |
| FB | .11 | .34 |
| HILP / HILP-density | .08 | .09 |
| **Density-FB** | **.91** | **.94** |

- (c) alone, on a full-state outcome with a ρ-driven frozen representation, does
  nothing (.17–.23): normalizing a distribution over the wrong outcome variable
  normalizes the wrong thing.
- Handing the retrofits object coordinates supplies (a) by construction (B cannot
  see the arm) and defuses (b) (z over an object-only B is no longer mean-dominated);
  they then land within .06–.12 of Density-FB. The estimator is *sufficient* once the
  outcome variable is right; Density-FB is the one variant that gets the outcome
  variable right on its own, because B trains through (a).
- HILP is inert under everything: temporal distance in play data is a function of arm
  motion, and on the object-only input it has almost no signal because the object is
  static most of the time. No estimator on top can repair a representation objective
  that has no gradient on the task variable.

Beyond manipulation, the same reading predicts the nulls we already have: on ExoRL
walker the state *is* the outcome and every dimension moves, so (a) and (b) coincide
with what FB does and Density-FB replicates FB (756 vs 760); on AntMaze the goal
variable is the agent itself and Density-FB matches GCIQL (.829 vs .837). The density
objective earns its keep exactly when the reward-relevant variable is a low-variance,
mostly-static slice of a state dominated by irrelevant motion — objects, drawers,
buttons (scene .64 vs 0) — and is neutral otherwise.

## 6. What this changes in the paper, and the one test still owed

- Drop "control bottleneck / precise actions" as the mechanism. Keep the setting
  ("interaction-conditioned successor structure") and state the mechanism as
  post-acquisition goal-blindness from ρ-driven outcome representation and index.
- Keep (c) as the estimator change with its measured share; do not claim a scale or
  dwell mechanism for it until a direct test passes.
- Owed test (cheap, same harness): at held mid-transport states, with each arm's
  **own** policy defining the label, does FB's goal-conditioned Q rank actions that
  move the cube toward the goal, and does its value differ across goals at all?
  Prediction from this note: FB's Q is near goal-invariant there; Density-FB's is not.

---

## 7. CORRECTION (same day, after pushback) — §4–5 over-attribute; the separator is the estimator alone

Three facts I had wrong or under-weighted, each checkable in the tree:

1. **The shipped full-state cell uses a ONE-HOT successor target, not the object kernel.**
   `impls/configs/critic/density_fb_state.yaml:22` has `measure_density_hard: true`
   (default since 2026-08-16; every table cell ran with it). So `density_fb_state` = .91
   with B on the full observation *and* an immediate target that is the row's own
   full-state successor. Object coordinates enter only through the goal reward at readout
   (`k(g, u_j)` over the cube slice) and the centered-z reward samples. §4(a)'s claim that
   "the kernel target teaches B the object" is **withdrawn**. Also: soft (kernel) and hard
   (one-hot) targets give the same numbers (tab:main), so the kernel is never load-bearing.
2. **Goal vs z is not the separator.** Density-FB through the *centered* z index reaches
   .82 on cube-single (spec29/30) against .93 goal-conditioned; the uncentered z port at
   .017 (spec24) is a fixable index defect, not a property of the estimator. And plain FB
   with a clean goal entry — `fb_oc`, B on cube xyz only, so `z(g)` is purely the goal's
   cube position — still sits at .34. §4(b) is a real z-interface finding (centering) but
   **not what separates FB from Density-FB**.
3. **Object coordinates in B are not the separator either.** Density-FB full-state .91 vs
   privileged .94; FB with them .34. §4(a)'s "outcome variable" share is `.11→.34`
   (cube-single) / `.005→.30` (cube-double, same-φ LSIF with the kernel readout). Real,
   but the minority.

**What is left, and it is one thing.** Across every matched control the single variable
that carries the gap is the successor *estimator*: cross-entropy on a normalized
categorical distribution over sampled successors, with B trained through it, in place of
squared error on the unbounded occupancy ratio.

| matched control | occupancy-ratio fit | normalized / bounded fit |
|---|---|---|
| cube-double, same φ, same kernel readout, same actor (1 seed) | LSIF .300 (TD-InfoNCE .576) | CE .736 |
| cube-single, full-state B, one-hot successor target | FB .11–.23 | Density-FB .91 |
| cube-single, bridge bilinear critic, no ortho, goal-conditioned actor (3 seeds, 2026-07-22) | discounted occupancy .004/.516/.004 | **first-passage .956/.964/.980** |
| same, with ortho | — | first-passage .225 |

The third row is the one I had not put next to Density-FB before, and it is decisive for
the mechanism question: an *unnormalized squared-loss* bilinear estimator works on cube
once its target is **bounded** (first-passage probability ≤ 1 instead of occupancy mass
that piles up wherever the object dwells). Two independent fixes — bound the target
(first passage) or normalize it (categorical simplex) — both recover ≥.9; the unbounded
occupancy target does not, with or without the ortho term.

**So the "dwell" story is right, at one specific level.** The object dwells on 48–72% of
transitions, so the true occupancy target at the object's current configuration is of
order `1/(1−γ)` while it is near zero elsewhere. With a bilinear function approximator and
a squared loss, fitting that peak dominates; the final *total* mass comes out calibrated
(probe: 87–105 vs 100) but the *allocation* across object outcomes — where transport
needs it — does not. This is a function-approximation/optimization effect: the tabular
fork-and-dwell test (exact fixed points) was null, which is consistent, not contradictory
— the v2 finding was that the channel is optimization rate / sample coupling, not the
fixed point. The ortho term makes it worse because it whitens B under ρ (arm-dominated),
which is why ortho is a toxin on cube and a benefit on walker.

**On "bottleneck".** If bottleneck means *the task variable is a small, mostly-static slice
of the state*, that is exactly the dwell condition above and the word is fine as a name
for the setting. It is not a control bottleneck (held start does not rescue FB; FB's Q
ranks the grasp action at .95) and it is not a coverage/corridor bottleneck in the maze
sense (held transitions are 30% of the data; FB works on mazes). A maze analogue would
need the agent to *sit still* most of the time, not a narrow passage — and the tabular
version of that was null, so the effect needs function approximation to appear. The
paper should say "the goal-relevant variable dwells", not "bottleneck".

**Extrapolation, corrected.** The retrofits transplant the normalized estimator but (i)
freeze the representation (SP rollout + ortho; Hilbert distances) and (ii) keep an
*uncentered* point/moment z readout. At full state they fail (.17–.23); with object
coordinates they reach .82–.88. Those two defects are confounded in the existing cells;
the cell that separates them is `rldp_density` + centered index (or + B trained through
the CE). The prediction from this note: the centered index recovers most of it, because
that is what it did for Density-FB's own z path (.017 → .82).

**Owed experiments (in order of cost):** (1) eval-only: centered z inference on the
existing plain-FB cube checkpoints (the AOSS spec's H2, never run here); (2) eval-only:
rldp_density with the centered index; (3) training: the loss ladder on cube-double at
fixed φ and readout — LSIF, LSIF with the target clipped/bounded, first-passage, CE — to
pin which property of the target (boundedness vs normalization) carries the gap.
