# Task-weighted successor moments on the GCIQL measure graft

**Date:** 2026-07-10
**Branch:** `gciql-fb-bridge`
**Living doc:** `PAPER/fb-gciql-bridge/sf-design-space.md`
**Source:** research note, "the measure loss learns the wrong geometry for control in cube"
**Status:** design approved, not implemented

---

## 1. The reduction

Write the FB/SF residual as a **vector**, `Δ(s,a,z) = φ(s,a,z) − γ·φ̄(s′,a′,z) ∈ ℝᵈ`. Then every
objective in our program is `δᵀMδ` for a PSD test matrix `M`, plus a linear Dirac term:

| objective | `M` | @500k |
|---|---|---|
| canonical SF (`sf_feature_td`, joint-ψ cell A) | `I` | **0.120 ± 0.016** |
| FB measure (`measure_loss`) | `C_ρ = E_{s′∼ρ}[ψψᵀ]` | **0.231 ± 0.030** |
| Simple SF / GCIQL family | `wwᵀ` (rank 1, task direction) | **0.904 ± 0.019** |
| **the note's `L_task-moment`** | **`C_R = E_{r∼R}[z_r z_rᵀ]`** | *this spec* |

The note's contribution is *which* `M`. Its §6 thesis: **FB minimizes density error in the replay
norm; control needs reward-projected error in the task norm.**

### 1.1 Instantiating `C_R` on OGBench's reward family

The moment map is `z_r = E_ρ[r(s)·ψ(s)]`. For a goal indicator `r_g(s) = 1[s=g]` this collapses:

```
z_g = ρ(g)·ψ(g)  ∝  ψ(g)        ⇒        C_R = E_{g∼ν}[ψ(g)ψ(g)ᵀ]
```

So **`C_R` is `C_ρ` with the test-state distribution swapped from ρ to the task distribution ν.**
`measure_loss` currently draws its B×B test states from `next_observations` (ρ). The note's §9,
applied to goal-reaching, is exactly: draw them from `value_goals` (ν) instead.

### 1.2 Why the ortho distribution must NOT move with it

FB orthonormalizes `ψ(s′)` under ρ **and** tests against `ψ(s′ⱼ)` drawn from ρ. At the ortho
optimum `C_ρ → I`, so FB's own norm degenerates to the identity — which is precisely why the FB
measure and canonical SF share a fixed point and differ only by estimator (0.231 vs 0.120,
measured 2026-07-10).

**Therefore the task-norm has content only if the ortho distribution ≠ the test distribution.**
We whiten under ρ (unchanged, FB-faithful) and test under ν. Whitening under ν as well would drive
`C_ν → I` and rebuild canonical SF — predicted ≈ 0.12, a null by construction.

### 1.3 What is NOT new

- **The note's §10 core objective** — `z_ω = N(E_ρ[r_ω ψ])`, `Q = F(s,a,z_ω)ᵀz_ω`, reward-TD with
  expectile `V` — is, for goal indicators (`z_g ∝ ψ(g)`), exactly `fb_forward_z` under the IQL
  backup: cell **B1 = 0.844 ± 0.008**. Already run. It "keeps the part that works," and our grid
  agrees. Its novelty needs `R` broader than goal indicators, which cube does not provide.
- **The note's §8.4 policy coherence / scalar gating** is not the same as its §9 task projection.
  SG-ISFB (`isfb_flow.py`) already reweights *rows* by the advantage
  (`wbar[None,:,None] * (M − N)²`) while its measure TD stays a full B×B loss against `ψ(nobs)`.
  Measured: **0.145 – 0.227** — the measure band, no rescue. Sample reweighting ≠ test-direction
  reweighting.
- **§10's auxiliary variant** (`L_IQL + λ·L_task-moment`) is E2-measure-mix with a different
  auxiliary. E2 showed the measure auxiliary is neutral where stable (λ≤1: 0.857–0.917) and
  destructive with stop-grad-φ above it (λ=10: 0.024). Low information per GPU-hour.

So the untested content is exactly two knobs: **the test-state distribution** and **the rank of
the test set**.

---

## 2. The loss, decoupled

FB's loss is

```
L = E_{i, s⁺∼ρ}[ (M(sᵢ,aᵢ,s⁺) − γM̄)² ]  −  2·E_i[ M(sᵢ,aᵢ,s′ᵢ) ]
```

The first term samples **test states**; the second is the **empirical transition** — the Dirac
that plays the role of the reward. The current batch estimator packs both into one B×B matrix
(`_measure_terms`) only because the test states *are* the next-states. Once the test set moves to
`value_goals` they decouple, and the loss must be computed in two pieces:

```
psi_o  = psi(next_observations)         # Dirac term + ortho.  ALWAYS ~rho.
tpsi_o = target_psi(next_observations)
psi_t  = psi(test_goals)                # test set.  ~nu when measure_test_states='goals'
tpsi_t = target_psi(test_goals)

diff_t[i,j] = <phi_i, psi_t[j]>/sqrt(d)  -  gamma * sg[<tphi_i, tpsi_t[j]>/sqrt(d)]

offdiag = 0.5 * mean_{i!=j} diff_t[i,j]^2        # 'full'  -> 0.5 * Delta^T C_nu Delta / d
        = 0.5 * mean_i     diff_t[i,i]^2         # 'task'  -> 0.5 * Delta^T (z z^T) Delta / d

dirac_i = <phi_i, psi_o[i]>/sqrt(d) - gamma * sg[<tphi_i, tpsi_o[i]>/sqrt(d)]
diag    = - mean_i dirac_i                       # the reward term. ALWAYS on psi(s'_i).

L = offdiag + diag + measure_ortho_coef * ortho(psi_o)
```

**Getting this wrong is silent.** If the Dirac term is allowed to follow the test set to `ψ(g)`,
then `diff_t[i,i] = Q_i − γQ̄_i` and the reward term vanishes — the cell would optimize a
reward-free objective and its number would be uninterpretable.

With `('rho','full')`, `psi_t = psi_o` and the two pieces are the off-diagonal and diagonal of the
same matrix: **bit-identical to today's loss**.

### 2.1 What N2's fixed point is

N2 is *not* literally a squared TD error. It is FB's quadratic-plus-linear form restricted to one
test direction. But its stationary point is a scalar TD. Minimizing
`½(Q − γQ̄)² − ⟨F, ψ(s′)⟩/√d` over `F`, and projecting the gradient onto `z` (with `‖z‖² = d`
under `project_z`):

```
(Q − γQ̄)·d = ⟨ψ(s′), z⟩ = √d · r_kernel      ⇒      Q = r_kernel/√d + γ·Q̄
r_kernel(s′,g) = ⟨ψ(s′), ψ(g)⟩/√d
```

So N2's fixed point is **GCIQL's backup with the learned kernel `r_kernel` in place of the true
indicator `1[s=g] − 1`** — the exact midpoint between the FB measure (0.231) and bilinear GCIQL
(0.904).

---

## 3. The two cells

Both are `measure_only` cells, identical to `C-fbz-projz-ortho1000`
(`configs/gciql/cube_single_state_gciql_flow_bilinear_d50_C_proj.yaml`, **0.231 ± 0.030**) except
for the test set: `fb_forward_z=true`, `project_z=true`, `measure_ortho_coef=1000`,
`measure_legs=critic`, flow actor, d=50.

| cell | `measure_test_states` | `measure_test_rank` | isolates |
|---|---|---|---|
| anchor *(exists)* | `rho` | `full` | — (**0.231 ± 0.030**) |
| **N1 `C-TaskNorm`** | `goals` | `full` | the **norm** (ρ → ν) |
| **N2 `C-TaskProj`** | `goals` | `task` | the **rank** (B×B → diagonal) |

### 3.1 Two deliberate choices

**The Dirac term stays on `ψ(s′ᵢ)` in both cells.** It is the reward. Moving it to `ψ(gᵢ)` would
turn the diagonal into `Q − γQ̄` and delete the reward (see §2).

**N1 excludes `j = i` from the cross-pairs.** FB excludes it because `M(sᵢ,aᵢ,s′ᵢ)` also appears
in the Dirac term. That collision disappears once the test set is goals — but `gᵢ` remains
strongly correlated with row `i` (GCDataset draws 20% of `value_goals` as `sᵢ` itself and 50% as a
same-trajectory future state), so including `j=i` would bias the estimator toward the on-trajectory
directions. Excluded, matching FB's convention.

---

## 4. Knobs and guards

```python
config.measure_test_states = 'rho'    # 'rho' (next_observations, FB-faithful) | 'goals' (value_goals, the task distribution nu)
config.measure_test_rank   = 'full'   # 'full' (B x B cross-pairs, i != j) | 'task' (diagonal: project each Delta_i onto its own z_i = psi(g_i))
```

Guards in `create()`:

1. `measure_test_rank == 'task'` ⇒ `measure_test_states == 'goals'`. The ρ diagonal **is** the
   Dirac term; projecting there would double-count it.
2. `measure_test_states != 'rho'` or `measure_test_rank != 'full'` ⇒ `measure_only` **or**
   `measure_coef > 0` (the knobs only affect `measure_loss`).
3. Either knob off-default ⇒ `measure_legs == 'critic'` (the value-legs branch has no goal leg).
4. Either knob off-default ⇒ **not** `sf_feature_td` and `value_arch != 'simplesf'` (each replaces
   the loss entirely).

---

## 5. Tests (TDD, in `tests/test_gciql_task_norm.py`)

1. **Backward compat, decisive.** `('rho','full')` reproduces today's `measure_loss` value
   exactly, hand-computed from the single-matrix `_measure_terms` formula.
2. **`('goals','full')` term-by-term.** offdiag from `ψ(value_goals)` (i≠j), Dirac from
   `ψ(next_observations)` diagonal, ortho from `ψ(next_observations)` — all three computed by hand
   and compared against the logged info keys.
3. **The whitening/test split is real.** Perturb `value_goals` only: `measure_offdiag` must change
   while `measure_orth` is **unchanged**. This is the assertion that pins "whiten under ρ, test
   under ν"; if the ortho followed the test set, `C_ν → I` and the cell would silently be
   canonical SF.

   **Do NOT also assert the Dirac term is goal-invariant.** Under `fb_forward_z` the forward map is
   `φ(s,a,z)` with `z = ψ(g)`, so perturbing the goals legitimately changes `φ` and hence
   `dir_i = ⟨φᵢ, ψ(s′ᵢ)⟩`. Only the ortho reads `ψ(next_observations)` alone. That the Dirac uses
   `ψ(s′ᵢ)` and not `ψ(gᵢ)` is asserted separately by test 5.
4. **`('goals','task')` at γ=0.** `L = ½·mean(Qᵢ²) − mean(dirac_i) + λ·ortho`, hand-computed,
   where `Qᵢ = ⟨φᵢ, ψ(gᵢ)⟩/√d`.
5. **The Dirac term never follows the test set.** Under `('goals','task')`, assert
   `measure_diag == −mean_i dirac_i` computed from `ψ(s′ᵢ)` and **not** from `ψ(gᵢ)`; assert the
   two differ (guards against a vacuous test).
6. **Guards** reject `('rho','task')`, the non-measure configs, `measure_legs='value'`,
   `sf_feature_td`, and `value_arch='simplesf'`; both valid cells construct.
7. **Existing suites stay green** (`sf_feature_td` 16, `psi_base` 4, `sf_joint_psi` 9,
   `fb_forward_z` 2, `measure_scale` N).

---

## 6. Configs

- `configs/gciql/cube_single_state_gciql_flow_bilinear_d50_C_tasknorm.yaml`
  → `run_group: GCIQL-State-Flow-Bilinear-d50-C-TaskNorm`
- `configs/gciql/cube_single_state_gciql_flow_bilinear_d50_C_taskproj.yaml`
  → `run_group: GCIQL-State-Flow-Bilinear-d50-C-TaskProj`

Both copy `..._C_proj.yaml` verbatim and add the knobs. 500k steps, 3 seeds, `eval_episodes=50`.

**Distinct wandb groups, never reused** — `results_map.py` keys on the group; relaunching into an
existing group silently averages good and bad seeds (this bit us on the E3 re-run, 2026-07-09).

---

## 7. Diagnostics

`measure_loss` already logs `measure_loss`, `measure_offdiag`, `measure_diag`, `measure_orth`.
Add `measure_qz_mean` (= `mean_i diff_t[i,i]` before squaring) for the `task` cell.

Reading them:
- `measure_orth → −d/2` in the B→∞ limit iff `C_ρ = I`. (Exactly, at B=1024, d=50 the whitened
  floor is `d(B−d)/(2(B−1)) − d = −26.20`; see `scripts/campaign_monitor.py::c_minus_i_sq`.)
  It must stay at the ρ-whitened floor in both cells — the ortho did not move.
- `measure_diag` growing unbounded is **benign** for a successor measure (the PSM finding);
  it is not divergence. The flow actor normalizes its Q-term by `sg[|q|.mean()]`.
- If `measure_offdiag` collapses to ~0 in N2 while success stays low, the single test direction is
  being fitted exactly and the failure is elsewhere (rank/estimator, not optimization).

---

## 8. Anchors, and what each outcome means

| cell | @500k |
|---|---|
| bilinear GCIQL (true reward, scalar TD) | **0.904 ± 0.019** |
| `B1` = `fb_forward_z` under IQL (= the note's §10) | 0.844 ± 0.008 |
| FB measure, ρ-norm, B×B *(the anchor)* | **0.231 ± 0.030** |
| `C1-fbz-ortho1000` (no sphere) | 0.255 ± 0.052 |
| SG-ISFB (advantage-reweighted **rows**) | 0.145 – 0.227 |
| joint-ψ SF cell A (identity norm, diagonal) | 0.120 ± 0.016 |

**N1** ≈ 0.90 vindicates the note's §6: the replay-vs-task norm *is* the failure.
**N1** ≈ 0.23 refutes it: reweighting test directions by task relevance changes nothing.

**N2** attributes the remaining failure:
- ≈ 0.90 ⇒ the **rank** of the test matrix is the problem (fitting the full density).
- ≈ 0.23 ⇒ the **kernel reward** `⟨ψ(s′),ψ(g)⟩/√d` is the problem, not the rank.
- ≈ 0.12 ⇒ the **diagonal estimator** dominates and we have rediscovered the cell-A effect
  (gradient density into ψ), which would make N2 uninformative about the reward.

The third outcome is the one to watch: N2 confounds rank with estimator, exactly as cell A did.
If N2 ≈ 0.12, the follow-up is N1-with-diagonal-estimator (`('rho','task')` is guarded off, so it
would need a fourth cell) to separate them.

---

## 9. Cost and sequencing

6 runs (2 cells × 3 seeds) @500k. GPUs are free (`psibase` and `sfjoint` complete).

After collection: add both groups to `scripts/results_map.py`, run
`compile_results_csvs.py --refresh`, and fold into `PAPER/fb-gciql-bridge/sf-design-space.md`
alongside the `M ∈ {I, C_ρ, C_ν, wwᵀ}` table.

## 10. Out of scope

- Reward families `R` richer than goal indicators (cube provides none; would need synthesis).
- The note's §10 auxiliary `L_IQL + λ·L_task-moment` (= E2-mix, already neutral at λ≤1).
- Torch FB (`agents/fb/`). The user chose the GCIQL measure graft; the same two knobs would port,
  but the anchors and the HER goal distribution ν live here.
- Moving the ortho to ν (predicted null by §1.2; worth one cell only if N1 surprises).

One change at a time.
