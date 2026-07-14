# SF experiments — design inheritance, variants tried, and the untried remainder

All SF cells live in `tools/wandb_mode_shim/gciql_flow.py` (`sf_feature_td=True`
replaces the IQL critic loss with a successor-feature vector TD on φ; `sf_loss`
at line 617). Configs: `configs/gciql/cube_single_state_gciql_flow_sf_feature_td*.yaml`.
Tests: `tests/test_gciql_sf_feature_td.py`.

Reference implementation being reproduced: raymondchua `simple_successor_features`.
**Re-read from source 2026-07-10** (`…/scratchpad/simple_successor_features/agent/`); two claims
previously recorded here were wrong.

The repo has TWO agents, and we ported the wrong one as our "SF":

| reference agent | what it actually does | our analogue | score |
|---|---|---|---|
| `sf_simple` — **their method** | `Q=⟨ψ_SF(s,a), z⟩` contracted FIRST, then an ordinary **scalar** Bellman backup `target = r + γ·min(Q1',Q2')` with the true reward and clipped double-Q. Basis (a learned net) trained by that critic loss. `z` is a **free `nn.Parameter` vector** fit by linear reward regression `r ≈ ⟨φ_basis(s',a'), z⟩` — under `no_grad`, at `lr_task=1e-5`, every 10 steps. | **bilinear GCIQL** (scalar reward-TD on `Q=⟨φ(s,a),ψ(g)⟩`, learned goal embedding in place of a per-task free `z`) | **0.90** |
| `sf_canonical` — **their baseline** | **vector** SF-TD `SF ← b(s) + γ·SF_target`, min-**L2-norm** head selection for the target, base `b(s) = F.normalize(obs)` (the raw obs!), φ task-conditioned via `cat[obs, z]`, `z` L2-normalized, and **`reward` is passed into `update_critic` and never used**. | **our `sf_feature_td`**, closest to the `identity`+`fb_forward_z` cell | **0.08–0.16** |

So **the paper's "simple" recipe is: do NOT run a TD on the representation.** Keep the SF
factorization for the readout, contract with z, and run a plain scalar reward-TD. We have been
running their working method all along — it is our 0.90 control — and their collapsing baseline
as the treatment. Our band reproduces the reference's own ordering.

Two corrections to earlier notes:
1. `sf_simple.py:322` regresses the **task vector**, not the basis (the basis is `no_grad` there).
   The reference's reward model is therefore **asymmetric**, `⟨φ_basis(s',a'), z⟩`, between a net
   and a free vector — it has no ψ=0 stationary point. Our symmetric `⟨ψ(s),ψ(g)⟩` does, which is
   exactly the saddle that killed E3 v1 and forced `sf_reward_shift`. Item 5 below is not an
   optional extra; it is what the reference actually does.
2. `audit-20260706.md` says the canonical baseline "at least optimizes the right reward."
   It does not — `sf_canonical` has **no reward term anywhere**. Its implicit reward is
   `⟨normalize(s), z⟩`, the same linear proxy we flagged as a confound. The reference's own
   baseline shares our identity-base defect.

Also present in `sf_canonical` and absent from ours: the **min-L2-norm double target** (item 4)
and L2-normalization of the base and of z.

---

## 1. Design decisions inherited from GCIQL (identical to the 0.90 control)

Everything below is shared bit-for-bit with the bilinear-GCIQL positive control
(`GCIQL-State-Flow-Bilinear-d50`, 0.90 @1M), so none of it can explain an SF-vs-GCIQL gap.

**Data / goal sampling (GCDataset — this IS HER):**
- value goals: 20% current state / 50% same-trajectory future (geometric) / 30% random state
- actor goals: 100% same-trajectory future
- rewards hindsight-relabeled per sampled goal: `0 if s'==g else -1` (`gc_negative`)
- bootstrap goal-masked (`masks`=0 at goal) — applied inside the SF target
  (`sf_loss`: `target = b(s) + γ·mask·φ_target(s',a')`)

**Architecture / readout:**
- bilinear form `Q(s,a,g) = ⟨φ(s,a), ψ(g)⟩/√d` (GCBilinearValue), d=50
  (d=28=obs-dim in the identity-base cells)
- ensemble of 2 heads, clipped double-Q (min) for the actor target

**Optimization / TD plumbing:**
- Adam lr 3e-4, batch 1024, discount 0.99, target network with τ=0.005
- TD target fully stop-gradded; next action `a' ~ π(s', g)` from the current actor
  (relabeled goal), clipped to [-1,1]

**Actor (unchanged from the bridge's flow cell):**
- FQL flow actor: BC flow matching + Q-max distillation, `bc_coeff=3.0`, `flow_steps=10`,
  goal-conditioned on the raw goal obs (`actor_cond=goal`)
- greedy on the SF readout Q=⟨φ,ψ(g)⟩; no critic grads through the actor Q-term
- DDPGBC (stock gaussian) variant run as the no-flow-actor control

**Eval:** exact-goal OGBench eval, 50 episodes, cells @500k, 3 seeds.

**What is deliberately NOT inherited:** the IQL expectile machinery (value net,
expectile loss, AWR) — `sf_feature_td` is an SF-ONLY critic; the scalar Bellman
backup is precisely what gets replaced by the vector TD
`φ(s,a) ← b(s) + γ·φ(s',a')`.

---

## 2. SF variants tried

### 2a. The 8-cell grid (flow actor), @500k ×3 seeds — band 0.08–0.16

Axes: conditioning × base × ortho. All collapsed to ~the same band; best cell
FrozenCond ≈ 0.157. (Anchor: same architecture + scalar TD = 0.90.)

| axis | values | knob |
|---|---|---|
| φ task-conditioning | agnostic φ(s,a) / conditioned φ(s,a,z), z=sg(ψ(g)) | `fb_forward_z` |
| base b(s) | identity b(s)=s (d=28) / frozen random ψ (d=50) | `sf_base` |
| anti-collapse | none / ortho ‖E[φφᵀ]−I‖ λ=1000 | `sf_ortho_coef` |

### 2b. DDPGBC controls (no-flow-actor), 4 cells @500k — all ≤ flow

IdentityB 0.027 / Frozen50 0.013 / IdentityCond 0.071 / FrozenCond 0.127
→ the flow actor is NOT the bottleneck; the collapse is in the SF critic/representation.

### 2c. E3 — corrected SF with a reward-grounded learned base (`sf_base=learned`)

ψ trained ONLY by bilinear reward regression `r̂(s,g)=⟨ψ(s),ψ(g)⟩/√d` on the
(relabeled) batch rewards; φ ONLY by the vector TD. Fixes the confound from the
2026-07-06 audit (earlier SF cells had NO reward anywhere in the loop and an
inexpressible linear-proxy readout).

- **v1 RewReg** (`6dde39e`): collapsed to the ψ=0 saddle — {−1,0} rewards ×
  SYMMETRIC bilinear reward model (grad ∝ ψ vanishes at 0 where the majority class
  is already fit). Killed @260k. Saddle artifact, not the E3 answer.
- **v2 RewReg-Shift** (`sf_reward_shift=1.0`, regress on r+1∈{0,1}; `8afc7fc`):
  saddle escaped — sf_reward_loss ≈ 0.027 ≪ 0.21 class prior, r̂_mean ≈ positive
  rate → the reward IS expressible in the base this time. Success still 0.0 at
  100k/200k/300k (3 seeds). Killed @~333k. **Both v1 and v2 carried the
  arriving-state reward bug below — neither is citable; the cell must be re-run.**
- **Reward-semantics bug (2026-07-09 audit point 2, FIXED).** The regression predicted
  `batch['rewards']` from ψ(s′), but GCDataset sets `successes = (idxs == value_goal_idxs)`
  and derives BOTH `rewards = successes − 1` and `masks = 1 − successes` from the CURRENT
  index — and the SF target's immediate feature is `b(s)`. So the old code was off-by-one
  against reward, mask, and base simultaneously. Now `r̂(s,g)=⟨ψ(s),ψ(g)⟩/√d`, which also
  pins `‖ψ(g)‖² = r_at_goal + shift` at the goal. (The arriving-state alternative would
  require retargeting the TD to `b(s′)` and relabelling on `idxs+1` — not mixed.)
- **v3 RewReg-Shift-Cond** (`fb_forward_z=true`): audit point 3. `mask` depends on the
  sampled goal, so a task-AGNOSTIC φ(s,a) must average over mutually inconsistent
  goal-conditioned terminations. Conditioning φ on z=sg(ψ(g)) makes the backup internally
  consistent. Config added; not yet run. The other remedy the audit offers — a
  goal-independent continuation mask (the *behaviour*-SF test) — is unimplemented and
  changes the fixed point, so it stays a separate cell.

---

## 3. Not yet added (the open design space)

Ordered roughly by how much each would strengthen the "SF representation-TD fails"
claim if it also fails.

1. **Re-run E3 @500k with the corrected reward semantics** — both prior E3 runs carried
   the ψ(s′) off-by-one. Not a new design; highest priority. Two cells now:
   `..._rewreg_shift` (agnostic φ) and `..._rewreg_shift_cond` (φ(s,a,z), audit point 3).
1b. **Scalar IQL TD + frozen/identity ψ (the readout-vs-TD crux).** ✅ *implemented
   2026-07-09, not yet run.* Normal GCIQL recipe, bilinear, but ψ(g) frozen-random (d50)
   or ψ(g)=g (d28). New `psi_base` knob lifts the base-class dispatch out of the
   `sf_feature_td` branch and applies it symmetrically to the value net AND the critic
   (freezing only the critic would leave V(s,g) a trained bilinear head). Guards refuse
   `mlp`, `sf_feature_td`, `fb_forward_z`, `fwd_two_tower`, `isfb`, and a mismatched d
   under identity. Tests: `tests/test_gciql_psi_base.py` (4). Configs:
   `..._bilinear_d50_frozen_psi`, `..._bilinear_identity_psi`, plus `..._bilinear_d28`
   (rank control — the identity map pins d=28, so without it "ψ is raw obs" is confounded
   with "rank dropped 50→28").

   If scalar-TD + frozen-ψ ≈ 0.9 while SF + frozen-ψ ≈ 0.1 on the SAME frozen leg, the SF
   failure is pinned on the vector TD itself; if it also fails, the frozen readout was part
   of SF's handicap and the SF cells never isolated the TD. Torch priors: idB_dnorm 65.3
   (z=g but F still goal-conditioned) vs idB_decond_dnorm **2.7 collapse** (true φ(s,a)·g/√d
   under scalar TD) — the identity arm is the decond variant, so a collapse replicates torch.

   **EARLY RESULT @100k (2026-07-09, campaign still running) — BOTH ARMS COLLAPSE.**
   | cell | @100k | @200k | @300k | @400k |
   |---|---|---|---|---|
   | bilinear-d50 (control) | 0.688 | 0.752 | 0.920 | 0.876 |
   | bilinear-d28 (rank control) | 0.660 | 0.816 | 0.848 | 0.872 |
   | **frozen-ψ d50** | **0.016 / 0.028** (n=2) | — | — | — |
   | **identity-ψ d28** | **0.008 / 0.012** | 0.020 / 0.024 | 0.012 / 0.012 | 0.012 / 0.040 |

   Rank is exonerated (d28 tracks d50 seed-for-seed, |Δ|≈0.03). Neither arm blows up
   numerically — Q stays in the usual [−20, −10] band. But they fail for DIFFERENT reasons,
   and the difference is the finding:

   - **frozen-ψ UNDERFITS the Bellman backup**: critic_loss 4.9–7.0 vs the control's 1.3–1.9,
     q_mean −10.0…−11.3 vs −18.5…−19.7 (values compressed toward a shorter effective horizon).
     A fixed random goal basis cannot express Q(·,·,g).
   - **identity-ψ FITS the backup and still has no policy**: critic_loss 1.5–1.7 and q_mean
     ≈ −17.4, i.e. statistically indistinguishable from the control — yet success is 0.01.
     The value regression is healthy while its argmax over `a` is useless. This is the audit's
     "linear proxy r=⟨s,g⟩ has the wrong argmax" argument, now demonstrated under the SCALAR TD.

   **Consequence for §2a/2b: the 8-cell SF grid never isolated the vector TD.** Every cell in it
   used a frozen or identity ψ, and that choice ALONE collapses the critic under the ordinary
   scalar backup. The 0.08–0.16 band cannot be read as evidence against representation-level TD.
   The only SF cells that can speak to the vector TD are the learned-base ones (E3).
1a. **Joint-ψ SF + FB orthonormalization (cells A and C).** ✅ *designed 2026-07-10, spec =
   `docs/superpowers/specs/2026-07-10-joint-sf-ortho-design.md`; not implemented.*
   Train ψ by the SF vector TD **itself** (un-stop-grad the base `b(s)=ψ(·)`) and keep it
   non-degenerate with FB's ortho λ=1000 + `project_z` sphere norm, instead of freezing it
   (§2a, now known confounded) or grounding it by reward regression (E3).

   **This is very nearly the FB ablation, and the spec proves how nearly.** With
   `Δᵢ = φ(sᵢ,aᵢ) − γφ_tgt(s′ᵢ,a′ᵢ)`, the joint-SF loss expands to
   `‖Δ‖² − 2⟨Δ,ψ(s)⟩ + ‖ψ(s)‖²`, while FB's measure loss gives `½ΔᵀCΔ/d − ⟨Δ,ψ(s′)⟩/√d` plus
   `ortho = ½‖C−I‖²_F − d/2` with `C := E_ρ[ψψᵀ]`. **FB's diagonal term IS the SF base term**;
   ortho drives `C→I`, equating the two quadratics; `project_z` makes the norm terms constant.
   Three residual differences survive:
   (i) the **estimator** — FB pairs every `(sᵢ,aᵢ)` with every `ψ(s′ⱼ)` (B×B, residual in the
   learned metric `C`), the SF loss uses only the diagonal `i=i` in the identity metric;
   (ii) the **base index** `ψ(s)` vs `ψ(s′)`; (iii) the **absorbing mask**, which FB lacks
   (`measure_loss` has zero references to `batch['masks']`, and orthos `ψ(next_observations)`
   — the ρ marginal, not the HER goals).

   **Cell A** removes (ii) and (iii) by adopting FB's conventions wholesale — `b=ψ(s′)`, no
   mask, `fb_forward_z`, `project_z`, ortho λ=1000 on `ψ(s′)` — so it is identical to
   `C-fbz-projz-ortho1000` (**0.231 ± 0.030**) in every respect *except the estimator*. That
   isolates the per-sample-vs-matrix axis the whole SF program exists to test and has never
   tested. Filling in the 2×2:

   |  | reward-grounded ψ | ortho-grounded ψ |
   |---|---|---|
   | measure estimator (B×B) | *empty* | **0.231 ± 0.030** (C-fbz-projz-ortho1000) |
   | SF estimator (per-sample) | 0.000 / 0.036 (E3) | **0.120 ± 0.016** (cell A) |

   **RESULT (2026-07-10, 3/3 seeds @500k). The estimator matters: cell A = 0.120 ± 0.016 vs the
   matched B×B measure cell's 0.231 ± 0.030.** Non-overlapping CIs; the per-seed deltas are
   negative on all three seeds (−0.116 / −0.072 / −0.144). Everything else was held identical —
   `fb_forward_z`, `project_z`, ortho λ=1000 on ψ(s′), base ψ(s′), no mask, same actor, same
   data — so the ONLY difference is the test matrix: `½ΔᵀIΔ` (per-sample diagonal) vs `½ΔᵀCΔ/d`
   (B×B pairs). At the ortho optimum C→I these share a fixed point, so this is an **estimator /
   gradient-density effect, not a different objective**: in the measure loss every `ψ(s′ⱼ)`
   receives gradient from all B forward vectors, while cell A's `ψ(s′ᵢ)` learns only from its own
   sample — a ~B× sparser signal into ψ.

   Learning curves (both flat after 100k, like the whole measure family):

   | cell | 100k | 200k | 300k | 400k | 500k |
   |---|---|---|---|---|---|
   | bilinear GCIQL (scalar TD) | 0.733 | 0.824 | 0.883 | 0.907 | **0.904 ± 0.019** |
   | FB measure (B×B) | 0.201 | 0.240 | 0.232 | 0.165 | **0.231 ± 0.030** |
   | **A: joint-ψ SF (per-sample)** | 0.088 | 0.103 | 0.091 | 0.121 | **0.120 ± 0.016** |
   | **C: + absorbing mask** | 0.084 | 0.121 | 0.103 | 0.096 | **0.125 ± 0.017** |

   Ortho was healthy throughout: `|ψ| = 7.071 = √d` exactly, `‖C−I‖²_F ≈ 2.2–2.7` against
   `‖C‖²_F ≈ 52` (a ~5% deviation). So ψ *was* whitened and the peaked-kernel premise held;
   the cell's shortfall is not an ortho failure.

   **Cell C is a null.** 0.125 ± 0.017 vs A's 0.120 ± 0.016, per-seed Δ = +0.000/+0.000/+0.016.
   Absorption (`Q = γ^T`, first-hitting) versus a visitation density (`Σγᵗ1[sₜ=g]`) makes **no
   difference** on cube. That removes one hypothesis from the board.

   Note cell A lands squarely inside the OLD confounded SF band (0.08–0.16). Training ψ by the
   TD and whitening it with FB's ortho did not lift the per-sample SF cell out of that band.

   **Cell C** keeps the absorbing mask with a z-conditioned φ, so `Q = γ^T` (discounted
   first-hitting time) rather than `Σγᵗ1[sₜ=g]` (visitation density). No FB counterpart; asks
   whether absorption is the better goal-reaching object. Coherent *only* with `fb_forward_z`,
   since a goal-dependent mask cannot sit in a goal-agnostic recursion.

   Ortho is not optional: un-stop-gradding the base makes `φ≡0, ψ≡0` a **global** minimum of
   the TD. The `−tr(C)` term makes the origin a strict local *maximum* (repelling), unlike the
   *attracting* ψ=0 saddle that killed E3 v1. `project_z` makes radial collapse impossible
   outright; the off-diagonal term then only has to prevent angular collapse.

   Nice consequence (spec §4.2): with sphere-normed ψ, `r̂_g(s) = √d·cos∠(ψ(s),ψ(g))`, which is
   `√d` at `s=g` and which the off-diagonal ortho drives to 0 off-goal — so
   `r̂_g(s) ≈ √d·1[s=g]`. **The peaked kernel E3 buys by supervised regression, this cell buys
   by whitening.** Same target, two routes: supervision vs geometry.

   New knobs: `sf_joint_psi`, `sf_base_index ∈ {current, next}`, `sf_mask ∈ {goal, none}`,
   `sf_psi_ortho_coef` (on **ψ** — distinct from the existing `sf_ortho_coef` on **φ**, which
   at λ=1000 drove cells to 0.001–0.004). Five guards encode the algebra.
1d. **GC-SimpleSF — the missing (scalar TD + explicit linear reward model) cell.** ✅ *designed
   2026-07-10, spec = `docs/superpowers/specs/2026-07-10-gc-simple-sf-design.md`; not implemented.*

   **The unification.** Write the SF Bellman residual as a vector,
   `Δ(s,a) = φ(s,a) − b(s′) − γ·φ̄(s′,a′) ∈ ℝᵈ`. Then

   - canonical SF (our `sf_feature_td`): `L = ½‖Δ‖² = ½ΔᵀIΔ`
   - **Simple SF**: its scalar TD residual is `Q − (r + γQ̄) = (φ − b(s′) − γφ̄)ᵀw = Δᵀw`,
     **assuming** its reward factorization `r = b(s′)ᵀw`. So `L = ½(Δᵀw)² = ½Δᵀ(wwᵀ)Δ`.
   - FB / measure graft: `offdiag = ½·E_{i≠j}⟨Δᵢ, ψ(s′ⱼ)⟩²/d = ½ΔᵀCΔ/d`, `C := E_ρ[ψψᵀ]`.

   **Same residual, three PSD test matrices: `wwᵀ` (rank 1), `I` (rank d), `C` (rank d → I under
   ortho).** And the measured ordering is monotone in rank:

   | objective | test matrix | rank | @500k |
   |---|---|---|---|
   | bilinear GCIQL | scalar / rank-1 | 1 | **0.904 ± 0.019** |
   | FB measure (B×B) | `C` | d | 0.231 ± 0.030 |
   | joint-ψ SF (per-sample, cell A) | `I` | d | 0.120 ± 0.016 |

   Corollary: FB's ortho is **semantic**, not merely anti-collapse. With
   `M(s,a,ds′) = ⟨φ,ψ(s′)⟩ρ(ds′)`, the successor feature of base `b` is `Ψ_b = E_ρ[bψᵀ]·φ`;
   taking `b = ψ` gives `Ψ_ψ = C·φ`, so **φ IS the successor feature of ψ iff C = I**.

   **The cell.** `Q = ⟨Ψ(s,a,z), z⟩/√d`, `z = ψ(g)`, scalar (expectile) TD with the TRUE reward,
   and ψ trained ONLY by `r̂ = ⟨sg[b], ψ(g)⟩/√d` regressed on `batch['rewards']`. The reference's
   trunk split is what makes this work: `h = trunk([s,a,z])`, `b = h/‖h‖` (zero parameters,
   gradient only via the trunk from the TD), `Ψᵢ = headᵢ(h)`. Because `b ≠ ψ`, the reward model is
   **asymmetric** and has no ψ=0 saddle — unlike E3's symmetric `⟨ψ(s),ψ(g)⟩`, which is precisely
   why E3 v1 collapsed and needed `sf_reward_shift`. No shift is needed here.

   Two cells, one knob apart, on the same new architecture:
   `simplesf-arch` (ψ learned end-to-end) isolates the **architecture** against bilinear GCIQL's
   0.904; `simplesf-grounded` (ψ's only signal is `L_rew`) isolates the **grounding** against
   `simplesf-arch`. Rung 3 is the paper's cell.

   **Prediction:** rung 2 ≈ 0.84–0.90 (`B1` says z-conditioning costs ~0.06); rung 3 far below,
   plausibly 0.0–0.2 — because restricting ψ's gradient is catastrophic under an otherwise
   identical scalar backup (end-to-end 0.904 → frozen 0.036 → reward-only-with-vector-TD 0.003).
   If rung 3 lands near 0.90, reward-grounding is the mechanism and the objective axis needs
   rewriting.

1e. **Task-weighted successor moments — the replay norm vs the task norm.** ✅ *designed
   2026-07-10, spec = `docs/superpowers/specs/2026-07-10-task-weighted-moments-design.md`;
   not implemented.* From the research note: *"FB minimizes density error in the replay norm, but
   control needs reward-projected error in the task norm."*

   **The reduction.** With `Δ(s,a,z) = φ(s,a,z) − γφ̄(s′,a′,z)`, every objective we run is
   `δᵀMδ` plus a linear Dirac term, for a PSD test matrix `M`:

   | objective | `M` | @500k |
   |---|---|---|
   | canonical SF / joint-ψ cell A | `I` | 0.120 ± 0.016 |
   | FB measure | `C_ρ = E_{s′∼ρ}[ψψᵀ]` | 0.231 ± 0.030 |
   | Simple SF / GCIQL family | `wwᵀ` (rank 1) | 0.904 ± 0.019 |
   | **the note's proposal** | **`C_R = E_{r∼R}[z_r z_rᵀ]`** | ← 1e |

   The moment map `z_r = E_ρ[r·ψ]` collapses on goal indicators: `z_g ∝ ψ(g)`, hence
   **`C_R = C_ρ` with the test states drawn from the task distribution ν (`value_goals`) instead
   of ρ (`next_observations`)**. `measure_loss` currently tests against ρ.

   **The trap.** FB whitens `ψ(s′)` under ρ *and* tests under ρ, so at the ortho optimum
   `C_ρ → I` and its own norm degenerates to the identity — which is exactly why the FB measure
   and canonical SF share a fixed point and differ only by estimator (0.231 vs 0.120). **The
   task-norm has content only if the ortho distribution ≠ the test distribution.** We whiten
   under ρ and test under ν. Whitening under ν too would rebuild canonical SF (predicted ≈ 0.12).

   **The cells.** N1 `C-TaskNorm` (test states ν, full B×B) and N2 `C-TaskProj` (test states ν,
   diagonal — project each `Δᵢ` onto its own `zᵢ = ψ(gᵢ)`). The Dirac term stays on `ψ(s′ᵢ)` in
   both: it is the reward, and letting it follow the test set turns the diagonal into `Q − γQ̄`
   and deletes it. N2's *stationary point* (not its loss) is GCIQL's backup with the learned
   kernel `r_kernel = ⟨ψ(s′),ψ(g)⟩/√d` in place of the true indicator — the midpoint between the
   measure (0.231) and GCIQL (0.904).

   **Already-run parts of the note.** Its §10 core objective (`z_ω = N(E_ρ[r_ω ψ])`,
   `Q = F(s,a,z_ω)ᵀz_ω`, reward-TD + expectile V) is, for goal indicators, `fb_forward_z` under
   IQL = **B1 = 0.844 ± 0.008**. Its §8.4 scalar gating ≠ its §9 task projection: SG-ISFB
   reweights *rows* by the advantage while its measure TD stays a full B×B loss against ψ(nobs),
   and it scores **0.145–0.227** — no rescue. Its §10 auxiliary is E2-measure-mix, neutral at
   λ≤1 (0.857–0.917).

   **Reading it.** N1 ≈ 0.90 vindicates §6; N1 ≈ 0.23 refutes it. N2 ≈ 0.90 ⇒ the *rank* of the
   test matrix is the failure; ≈ 0.23 ⇒ the *kernel reward* is; ≈ 0.12 ⇒ the *diagonal estimator*
   dominates (the cell-A effect) and N2 says nothing about the reward.

1f. **Annealed SM finetuning from a converged value geometry.** ✅ *designed 2026-07-10, spec =
   `docs/superpowers/specs/2026-07-10-annealed-sm-finetune-design.md`; not implemented, queued.*

   Every measure-vs-IQL result we have is a **cold start**. Warm-start from bilinear GCIQL
   (**0.904 ± 0.019**, `params_1000000.pkl`, 10 seeds on disk) and grow `λ(t)` on the SM loss:
   `L = L_IQL + λ(t)·L_SM`, linear ramp over 300k then **hold** 100k, with a short 10k LR warmup.
   Dense evals (every 25k) make each run a **success-vs-λ dose curve**.

   This separates two explanations the cold-start grid cannot:
   **fixed-point incompatibility** (the measure's optimum is not a good value function, so it
   destroys the geometry however you approach it) vs **optimization path** (the cold collapse is an
   artifact of learning the measure from scratch).

   Cells: `Anneal-Zero` (λ≡0 control, validates restore + isolates "400k more IQL"),
   `Anneal-SgPhi` (0→10, crosses the **known cold cliff**: sgphi λ1 = 0.857 → λ3 = **0.176** →
   λ10 = 0.024), `Anneal-Shared` (0→100; shared legs absorb even λ=10 at 0.917, so its cliff is
   unknown).

   **Two code facts forced the design.** (i) `restore_agent` deserializes the whole `TrainState`, so
   `network.step` comes back as **1,000,001** — a schedule keyed on it would start fully ramped.
   (ii) `optax.adam(learning_rate=<callable>)` swaps `EmptyState` for `ScaleByScheduleState`, so a
   constant-LR checkpoint **cannot restore into a scheduled optimizer**. Both dissolve via a prep
   script that writes a finetune-ready checkpoint: anchor params, *fresh* `opt_state` matching the
   scheduled `tx`, `step = 0`. Fresh Adam moments are deliberate — restored moments are tiny at
   convergence, so the first SM gradient would be amplified by `1/√v`, manufacturing a spurious
   "incompatible" result.

   The two ramps do not confound: at `t=10k` (LR at full value) `λ = λ_max/30 ≈ 0.33` for sgphi,
   an order of magnitude below its cliff. The whole dose-response happens at constant LR.

   **PRE-REGISTERED:** `critic/q_mean` must stay in **[−21, −8]**. Raw measure grafts blew the Q
   scale and crashed 18/18; every surviving E2 cell sat in that band. Outside it, the cell is a
   scale blowup and its success number is uninterpretable.

   Reading: SgPhi survives past λ=3 ⇒ the cold collapse was an optimization path. SgPhi collapses
   near λ≈3 anyway ⇒ the **fixed point itself** is incompatible with a good value geometry — the
   strongest form of the bridge's thesis.

1c. **Goal-independent continuation mask** (the *behaviour*-SF test) — the second remedy
   the 2026-07-09 audit offers for point 3. `sf_loss` currently bootstraps through
   `mask = 1 − 1[idxs == value_goal_idxs]`, which is goal-dependent. A behaviour-SF φ would
   use the trajectory's own termination instead. Changes the fixed point, so it is a
   distinct cell from the v3 conditioned one, not a substitute. Not implemented.
2. **FB's mixed z-sampling on the conditioned cell** (conditioned-SF + z-mix).
   FB trains with ~50% random √d-sphere task vectors; `measure_z_random_frac`
   exists but is wired ONLY into `measure_loss` (gciql_flow.py:564–582) — `sf_loss`
   always uses z=ψ(vgoals). Needs: thread `z_vec` through `sf_loss` the way
   measure_loss does + one config. Mitigating priors: agnostic cells have no z at
   all yet fail identically, and z-mix rescued nothing on the measure side
   (C5 = 0.07/0.165).
3. **`project_z` / √d-sphere norm on ψ and z under SF** — FB pins ‖B‖=‖z‖=√d;
   never enabled in any SF config (knob exists).
4. **Min-L2-norm double-SF target** — reference fidelity gap #2 from the audit;
   deliberately deferred from E3 ("one change at a time"). Our min is scalar
   (min over head Qs), theirs picks the vector of the min-norm head.
5. **Asymmetric reward model for E3** — r̂=⟨ψ_a(s'), ψ_b(g)⟩ (two embeddings)
   avoids the symmetric saddle *structurally* instead of via the +1 shift;
   would also decouple "base features" from "task embedding".
6. **Actor on z instead of raw goal under SF** (`actor_cond=z` + `actor_z_from_critic`)
   — FB's actor sees z, ours sees the raw goal obs. Tested on the measure side (D1),
   never on SF.
7. **Domain flip: SF on antmaze** — the mdose move that separated occupancy from
   objective for the measure (antmaze measure-only 0.79 vs cube 9.9). No SF cell
   has run on antmaze; if SF-antmaze works, the SF story becomes domain-conditional
   like the measure story.
8. **Inferred-z eval on SF cells** — E1a's dataset-moment readout (z=E[r·ψ])
   recovered 0.955 on the IQL critic; never pointed at an SF ψ. Cheap
   (eval_zeroshot_inference.py on existing ckpts — but note SF runs currently
   save no params; needs save_interval set).
9. **n-step / λ SF backups, ortho×rewreg combinations, sf_coef/reward_coef sweeps**
   — unexplored hyperparameter space; only worth touching if some cell first
   leaves the ~0.1 floor.

*Companion notes: `audit-20260706.md` (the E3 rationale), `untried-design-space.md`
(bridge-wide), `config-matrix-gciql.md` (per-config deltas).*
