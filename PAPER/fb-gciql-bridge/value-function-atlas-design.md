# Value-Function Atlas — design (rev 2)

**Date:** 2026-07-14
**Status:** design approved through review round 2. Phase −1 SOLVED (prefix-replay). Two items
carry open sub-decisions, flagged inline.

## 1. The two questions

One anomaly, one win — **different questions**:

| | flow actor | DDPG+BC |
|---|---|---|
| bilinear d50 | **0.947** | 0.557 |
| MLP | 0.747 | 0.747 |

- **The win** — bilinear+flow > MLP+flow → **supported ranking + actor consumption** (Primary A).
- **The reversal** — bilinear+flow ≫ bilinear+DDPG+BC → **exploitability** (Primary B).
- **What bilinear actually learns** → **oracle value-content alignment** (Primary C).

(OOD-grid C3 numbers. Paper 3-seed/500k: bilinear 91.6, MLP 71.3.)

## 2. The object

    A^m_s[i,j] = Q_m(s, a_i, g_j) − E_{ã~μ(·|s)}[ Q_m(s, ã, g_j) ]

μ = one frozen, Q-independent BC flow, shared by all critics. `E_μ[Q]` from a **fixed**
μ-sample set at each *s*, identical across critics. Centering gives `E_{a~μ}[A] = 0`.

**Grounding (verified, `datasets.py:242-244`).** `gc_negative=True`, `masks = 1 − successes`.
Let `N_g` = number of pre-success transitions receiving reward −1 (`γ^∞ = 0` if the goal is
never reached). Then

    Q(s,a,g) = −E[ (1 − γ^{N_g}) / (1 − γ) ].

Wording: *larger values correspond to a larger expected discounted hitting score, which jointly
reflects hit probability and time-to-hit.* (Not "sooner AND higher-probability" — a policy can
trade the two.)

**Non-identifiability.** `φ→Rφ, ψ→R^{−T}ψ` leaves Q unchanged. We SVD the **evaluated operator**,
invariant up to sign and rotation within exactly-degenerate singular subspaces.

**Not a claim.** `rank(A^bilinear) ≤ d` is architecture, not a finding. Report the **MLP's
effective rank** as the comparison; state the bilinear bound as architecture.

**Standardization (fixed).** After μ-centering, each state–critic matrix is divided by **one
positive global scalar** (its Frobenius norm). No per-action or per-goal normalization — those
change the operator and can erase the goal-common feasibility component. All comparison
statistics (argmax, singular vectors, rankings, modal topology) are invariant to this positive
scaling only — **not** to arbitrary `A → cA` with `c < 0`.

## 3. μ — the behavioral reference distribution

Verified goal-independent in both stacks: `actor_vf(obs, x_t, t)` takes no goal in jax `fql_flow`
(`gciql_flow.py:550`) or torch `flow_bc` (`agents/fb/flow_bc/agent.py:79`); only `awr_flow` is
goal-conditioned.

**μ is not "exact support".** A continuous flow assigns nonzero density across most of the
bounded action box, so its literal support is not the object we mean. Precise statement:

> μ(a|s) is the **Q-independent behavioral reference distribution** to which the flow actor is
> distillation-anchored. Its **high-density sample region** provides the operational notion of
> behavioral reference used here. `B_μ(s)` = the **μ-reference action set** (samples from μ), not
> "the support".

`mu_flow.py` must **prove** Q-independence (parameter-path test, per `scripts/fb_grad_decomp.py`),
not assume it.

**Support/reference distance (fixed).** Regularized whitening —

    Σ_{μ,λ}(s) = (1−λ)·Σ_μ(s) + λ·diag(Σ_μ(s)) + εI

— because grasp actions can occupy a near-singular local subspace even in 5-D. Report **relative**
distance, normalized against the μ-cloud's own leave-one-out spread so states of different
conditional action entropy are comparable:

    d_rel(a; s) = min_k ‖Σ_{μ,λ}^{−1/2}(a − a_k^μ)‖₂  /  median_i min_{k≠i} ‖Σ_{μ,λ}^{−1/2}(a_i^μ − a_k^μ)‖₂

**Sensitivity (pre-registered):** goal-conditional reference `μ(a|s,g)`, the hindsight-conditional
behavior policy from the data. Centering and the reference subset become **column-specific**. A
goal-marginal cloud can misclassify an action as referenced because it fits some *other* goal.

## 4. Claims

### Primary A — supported ranking AND actor consumption (explains the win)

Restrict to `B_μ(s)`. Four steps — ranking quality alone is not enough; the deployed actor must
consume it:

1. **Ranking.** Spearman `ρ_m` between `A^m_s(a,g)` and the oracle `J^MPC_μ(s,a,g)` over `B_μ`.
2. **Regret.** `J^MPC_μ(s, a*_{μ,bil}, g) > J^MPC_μ(s, a*_{μ,MLP}, g)` (best μ-reference action per
   critic).
3. **Actor lands in the good basin.** `A^bil_s(a_flow^bil, g)` sits in a high percentile of the
   bilinear values over `B_μ(s)`.
4. **Realized closed-loop.** `J^MPC_μ(s, a_flow^bil, g) > J^MPC_μ(s, a_flow^MLP, g)` — the actual
   flow action bilinear takes beats the actual flow action MLP takes.

Steps 3–4 are **part of Primary A**, not marked points. Without them, better ranking may be real
but incidental to closed-loop performance.

**Factorial (from the OOD grid — the win is a GOAL-shift effect).** Start bank matches the grid:
ID-start = dataset states, OOD-start = `env.reset()`; ID-goal = hindsight, OOD-goal = eval task.

| cell | start | goal | bilinear | MLP | prediction on ρ |
|---|---|---|---|---|---|
| C1 | ID | ID | 0.867 | 0.980 | **ρ_MLP ≥ ρ_bil** |
| C4 | OOD | ID | 0.773 | 0.787 | ρ_MLP ≈ ρ_bil (within equivalence margin δ) |
| C2 | ID | OOD | 0.930 | 0.823 | ρ_bil > ρ_MLP |
| C3 | OOD | OOD | 0.947 | 0.747 | ρ_bil > ρ_MLP, **largest gap** |

Real risk: **C1 predicts the MLP wins**, C4 predicts a tie (requires an equivalence interval δ,
§10 — "fail to reject" is not "tie"). The claim is compositional generalization to recombined
goals, amplified by start shift — not generic robustness.

**Split-oracle reality (OPEN sub-decision).** Prefix-replay branching (§7) is exact only for
states *we* generate. Reset states (C3/C4) qualify. **Dataset states (C1/C2) do not** — their
generating reset seed/task is unknown, so the oracle there falls back to `qpos/qvel` restore with
a residual. Resolution order:
  1. **Measure** the residual: contact-rich (grasping) continuation, full-state and cube-xyz drift
     vs the 4cm criterion. If ≪ threshold, approximate restore is fine for C1/C2 — the split is
     invisible. *(This is the test botched in rev 1 — random actions never touch the cube; must use
     grasp actions and measure full state.)*
  2. If it fails: substitute **μ-rollout states matched to the dataset distribution** for C1/C2
     (fully prefix-replayable; cost: μ's on-manifold states, not literal dataset rows).
  3. Last resort: C1/C2 become **critic-only** (report `A^m` and the MLP-wins contrast; no oracle
     regret there).

### Primary B — actor exploitability (explains the reversal)

    G_exploit^m(s,g) = [ A^m(a*_all, g) − A^m(a*_μ, g) ] / ( sd_{a~μ} A^m(a,g) + ε )

Value available **only by leaving the μ-reference set**. Compared against the best μ-reference
action, not the mean (a multimodal flow's mean can itself be off-reference; mean value is 0 by
centering).

**Exploitation requires all three** — distance alone is novelty, not exploitation:

1. **Reference departure** — `d_rel(a_ddpgbc | bilinear) > d_rel(a_ddpgbc | MLP)`
2. **Basin alignment (fixed def):**

       align = ⟨a_ddpgbc − a*_μ, a*_all − a*_μ⟩_{Σ^{−1}} / (‖a_ddpgbc − a*_μ‖_{Σ^{−1}} · ‖a*_all − a*_μ‖_{Σ^{−1}})

   Pre-register **positive cosine alignment AND** `d(a_ddpgbc, a*_all) < d(a*_μ, a*_all)`.
3. **Predicted–realized disagreement** — `A^bil(a_ddpgbc) > A^bil(a*_μ)` **but**
   `J^MPC_μ(a_ddpgbc) < J^MPC_μ(a*_μ)`.

**Ordering (full 2×2):** `G_exploit(C1) ≈ G_exploit(C4) < G_exploit(C2) ≤ G_exploit(C3)`.

**Interpolation plot.** λ∈[0,1]: `A^m_s((1−λ)a*_μ + λ a_ddpgbc, g)` vs `J^MPC_μ` on the same path.
Learned critic rising while oracle falls = exploitation, visibly.

### Primary C — oracle value-content alignment (what bilinear learns)

Build the oracle's own preference operator on the μ-reference set:

    A^oracle_s(a,g) = J^MPC_μ(s,a,g) − E_{ã~μ}[ J^MPC_μ(s,ã,g) ]

**Pre-registered content is quantitative alignment**, not mode-naming:

- Correlation of `A^m` with `A^oracle` over `B_μ`; principal angles between each critic's leading
  mode subspace `span(V_m)` and the oracle's `span(V_oracle)`.
- Prediction: `align_bil > align_MLP` (and > FB, SF).

**Descriptive (illustrates, does not pre-register):** sort all critic atlases by the **oracle's**
action order (once, shared), display side by side: `A^oracle, A^bil, A^MLP, A^FB, A^SF`. SVD the
oracle; name modes (goal-common feasibility → directional steering) as illustration.

Rationale for the split: the MLP scores 0.71 — it does **not** "fail to recover manipulation
structure". Pre-registering "MLP distorts the modes" would likely read as a failed prediction
("MLP recovers them slightly less cleanly"). The survivable claim is the alignment *metric*; the
mode stories illustrate it.

### Exploratory — multi-modality

Connected components in the top-*q*% of standardized `A` on a shared k-NN action graph
(§10 fixes q, k, min component size). Negative result does not undermine A/B/C.

## 5. Two atlases, weighted goals

**Do not** SVD over `B_all` for the semantic atlas — `B_global`/`B_perturb` are far from the flow
actor's region and can dominate singular modes, so you'd visualize extrapolation.

- **Semantic / Primary A & C:** `A^m_{s,μ} = A^m_s|_{B_μ × G}`.
- **Exploitability / Primary B:** `A^m_{s,all} = A^m_s|_{B_all × G}` — off-reference maxima,
  interpolation, `G_exploit` only.

**Goal categories have different cardinalities** (5 eval / hindsight / dense grid). A plain SVD
weights by how many columns each category has. Either analyze categories **separately**, or use
explicit category **quadrature weights** in a weighted SVD. Otherwise the dense grid silently
defines "important goal structure".

## 6. Cross-method comparability

Scale-invariance de-risks the torch methods: argmax location, `d_rel`, modal count, principal
angles are all invariant to positive scaling; only `G_exploit` is a value and it's z-scored.

**FB z-route.** Reward-inference is per-*reward* (`infer_z(obs, reward)`, `agents/base.py:50`), so
`z_g = E_ρ[r_g(s')·B(s')]` is computable per bank goal.
- Primary (A/B/C quantitative): reward-inferred `z_g` (FB's deployed, stronger route).
- Sensitivity (labelled): `z = project_z(B(g))`. Report both for the 5 eval goals.
- **Identifiability check (required):** per dense-grid goal, record #positive matches, effective
  sample size, nearest reference-state distance, `‖z_g‖`. **Exclude or separately label** goals
  whose `z_g` is not identifiable from the fixed reference batch — else many `z_g` collapse to
  near-identical vectors and FB looks artifactually flat.

**SF cells.** Main: `simplesf-arch` (0.380, strongest, recognizable). Mechanistic control: joint
vector-TD SF with `λ_ortho ∈ {0,1}` — **not 1000** (`iql-ortho1000`=0.072 shows ortho-1000 alone
kills a good critic; needs one new run). `sf-joint-ortho1k` (0.120) → appendix, confound labelled.

## 7. Phase −1 — SOLVED: reset + prefix-replay

`J^MPC_μ` must **branch** (many actions from one state) → needs exact state reconstruction.
**Snapshot/restore was never made bit-exact** (`_target_effector_pose` is an IK-target integrator
outside `mjData`, not in the dataset; residual `set_control` divergence 1.5e-3 unresolved).

**Prefix-replay solves it (verified, 0.0 divergence including counterfactual branching):**

1. `env.reset(seed, options=dict(task_id=...))` — **always pass `task_id`**; a bare
   `reset(seed=)` samples a random task from the global RNG (`manipspace_env.py:304`, the only
   global-RNG call) and is **not reproducible**.
2. Replay a fixed action prefix to the target state.
3. Execute the counterfactual first action, then the planner.

This reconstructs `mjData` + `_target_effector_pose` + controller state + step counter + every
hidden variable, because they're all produced by the same deterministic action sequence. Cost:
each branch re-simulates its prefix.

**Consequence (see §4 split-oracle):** exact for reset-derived states (C3/C4, all P1–P3 semantic
states we roll ourselves); **not available for literal dataset rows** (C1/C2) — their generating
prefix is unknown.

## 8. `J^MPC_μ` — privileged, behavior-constrained control oracle (NOT ground truth)

Not "ground truth" and no `*` — MPPI is approximate and the behavior constraint is a construction.

**Continuation must be behavior-CONSTRAINED, and this is where I differ from the review.** A pure
flow-prior planner (`a_t ~ μ`, flat selection) measures value under the *behavior* policy = `A^μ`,
which we established last round is the **wrong** target (IQL trains toward the in-support
*optimal*, not the behavior value). The fix keeps both properties:

> **Flow-proposed, critic-free-return-selected MPPI.** Propose continuation sequences from the
> frozen μ (stay on-manifold); **select** by realized discounted goal-hit return (`sup` over the
> behavior-constrained class), **not** by any learned critic. On-manifold *and* aimed at the
> in-support optimum, while remaining critic-neutral.

Wording: *`J^MPC_μ` is a privileged simulator-based, behavior-constrained control oracle aligned
with the in-support action-ranking problem IQL is intended to solve.* **Do not** equate it with
IQL's expectile target.

**Fixed planner config (§10):** horizon H, #sampled sequences, replan schedule, temperature,
terminal handling, and **#replicates** (a single noisy MPPI estimate corrupts Spearman rankings
among near-equivalent reference actions — replicate and report oracle noise).

**Sensitivity:** unconstrained MPPI, labelled as measuring **physical recoverability** only.

**Budget.** `J^MPC_μ` on: a `B_μ` subsample (Primary A); `a*_{μ,m}`; each deployed flow &
DDPG+BC action; `a*_{all,m}`; the `a*_μ → a_ddpgbc` path. **Not** the whole global bank. Explicit:
`B_μ` + actor points establishes *"the DDPG+BC action exploits the critic"*; `a*_{all,m}` is
required for *"the critic's global off-reference max is spurious"* — different claims.

## 9. Components

| module | does | notes |
|---|---|---|
| `banks.py` | build + freeze banks | deterministic; **always pass `task_id`** |
| `mu_flow.py` | locate μ; **prove** Q-independence | parameter-path test |
| `critics.py` | uniform `Q(s,A,G)→[n_a,n_g]` | jax 4 GCIQL; torch FB (reward-inferred `z_g` + coverage check), SF |
| `prefix_replay.py` | reset+prefix branching + identity test | Phase −1 |
| `oracle_j.py` | flow-proposed, return-selected MPPI | uses `prefix_replay` |
| `build_atlas.py` | `A^m_s` (two atlases) | |
| `analyze.py` | Primaries A/B/C + exploratory | paired bootstrap |
| `figures.py` | oracle-sorted side-by-side, Δ-sheets, interp | Δ-sheets oracle-gated (§13) |

## 10. Pre-registered constants (fill before implementation)

`|B_μ|`, `|B_perturb|`, `|B_global|`; covariance shrinkage λ, ε; local-density estimator; goal
category counts + weights; equivalence margin δ for `ρ_MLP ≈ ρ_bil`; top-q% and k and min
component size for multi-modality; MPPI H / #sequences / temperature / replan / terminal /
#replicates. **"C4 predicts a tie" needs δ, not failure-to-reject.**

## 11. Statistics

One result per `(seed, s, g)`; paired/hierarchical bootstrap over seeds and states. **Do not**
treat actions within a cell as independent. **Seed-pairing caveat:** pair only where seeds have a
real correspondence — jax-vs-torch or independently-generated runs are **not** paired just because
the integer label matches.

## 12. What is / isn't oracle-gated

**Gated** (needs `J^MPC_μ`): Primary A steps 1–2 & 4, Primary B condition 3, all of Primary C, the
Δ_s(g) sheets (`a⁺`/`a⁻` chosen by the oracle).
**Ungated:** evaluated critic matrices, SVDs, `d_rel`, Primary A step 3, actor/argmax locations,
Phase 0A. These proceed regardless of oracle status.

## 13. Order of work

1. **Phase −1** — DONE (prefix-replay verified).
2. **Split-oracle residual test** (§4) — decides whether C1/C2 use dataset states or μ-rollouts.
3. **Phase 0A** (branch gate) — four actor `d_rel` on byte-identical `(s,g)`; if
   `d_rel(a_ddpgbc|bil) ≯ d_rel(a_ddpgbc|MLP)`, the exploitation account of the reversal dies.
4. **Phase 0B** (branch gate) — matched continuations; if distant bilinear+DDPG+BC actions aren't
   systematically worse than the best reference action, exploitation dies even if distance held.
5. banks → critics → atlases → oracle → A/B/C → figures.

Phases 0A/0B gate only the exploitation branch (Primary B). Primary A, C, and the atlas proceed
regardless.

## 14. Open

- Split-oracle sub-decision (§4) — pending the residual test.
- One `λ_ortho ∈ {0,1}` vector-TD SF run (§6).
- All §10 constants.
