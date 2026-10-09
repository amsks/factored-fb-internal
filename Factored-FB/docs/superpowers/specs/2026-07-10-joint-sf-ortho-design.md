# Joint-ψ successor-feature cells with FB orthonormalization

**Date:** 2026-07-10
**Branch:** `gciql-fb-bridge`
**Living doc:** `PAPER/fb-gciql-bridge/sf-design-space.md`
**Status:** design approved, not implemented

---

## 1. Motivation

Every SF cell run so far trains φ by a vector TD while ψ is either fixed
(`identity`/`frozen`) or trained by an auxiliary reward regression (`learned` + E3).
ψ has **never** been trained by the SF loss itself.

Two results force the question:

1. **The 8-cell SF grid is confounded.** The `psi_base` control (2026-07-09) ran the
   ordinary scalar IQL TD with the *same* crippled goal leg: frozen ψ → 0.016/0.028 @100k,
   identity ψ → 0.028 @500k, against a rank-matched control at 0.850. A frozen or identity ψ
   destroys the critic *even with the scalar backup*. So the grid's 0.08–0.16 band says
   nothing about representation-level TD.
2. **E3 (reward-regressed ψ) is the only cell that ever isolated the vector TD**, and it
   currently reads 0.000 (agnostic) / 0.036 (conditioned) with the reward demonstrably
   fitted (`sf_reward_loss ≈ 0.02` vs a 0.21 class prior).

The remaining degree of freedom is to let the **TD itself** train ψ, holding it
non-degenerate with FB's orthonormality penalty rather than with a reward regression.

### 1.1 Why this is not simply the FB ablation (and where it is)

Write `Δᵢ = φ(sᵢ,aᵢ) − γ·φ_tgt(s′ᵢ,a′ᵢ)`. The joint-SF loss expands as

    ‖Δᵢ − ψ(sᵢ)‖²  =  ‖Δᵢ‖²  −  2⟨Δᵢ, ψ(sᵢ)⟩  +  ‖ψ(sᵢ)‖²

and the FB measure loss (`_measure_terms`, `Mᵢⱼ = ⟨φᵢ, ψ(s′ⱼ)⟩/√d`) as

    offdiag = ½·E_{i≠j}⟨Δᵢ, ψ(s′ⱼ)⟩²/d = ½·ΔᵢᵀCΔᵢ/d       C := E_ρ[ψψᵀ]
    diag    = − E_i ⟨Δᵢ, ψ(s′ᵢ)⟩/√d
    ortho   = ½‖C‖²_F − tr(C)  =  ½‖C − I‖²_F − d/2

Term by term:

| SF term | FB term | relation |
|---|---|---|
| `‖Δ‖²` | `½ΔᵀCΔ/d` | identical **when C = I**, up to scale |
| `−2⟨Δ, ψ(s)⟩` | `−⟨Δ, ψ(s′)⟩/√d` | same form; differs in base index and coefficient |
| `+‖ψ(s)‖²` | `−tr(C)` inside ortho | opposite sign; **both constant under `project_z`** |

FB's *diagonal* term **is** the SF base term. Ortho at λ=1000 is exactly what drives `C → I`,
making the first rows coincide; `project_z` makes the third row inert on both sides.

So with ortho + sphere the joint-SF cell **collapses onto the FB successor-measure objective**,
leaving exactly three residual differences:

1. **The estimator.** FB couples every `(sᵢ,aᵢ)` to every `ψ(s′ⱼ)` in the batch (B×B outer
   product) and measures the residual in the learned metric `C`. The SF loss uses only the
   diagonal `i=i` and measures it in the identity metric. Same fixed point at `C=I`;
   different Monte-Carlo estimator of it.
2. **The base index**: `ψ(s)` vs `ψ(s′)`.
3. **The absorbing mask** `m_g(s)`, which FB does not have.

**This is the point of the experiment.** Difference (1) is the per-sample-vs-matrix axis the
SF program exists to test and has never actually tested. Differences (2) and (3) are removed
by construction in cell A (below), and made the *sole* variable in cell C.

### 1.2 The 2×2

|  | reward-grounded ψ | ortho-grounded ψ |
|---|---|---|
| **measure estimator** (B×B) | *empty* | **0.231 ± 0.030** (`C-fbz-projz-ortho1000`) |
| **SF estimator** (per-sample) | 0.000 / 0.036 (E3) | **cell A ← this spec** |

If cell A ≈ 0.23, the estimator is irrelevant and the SF story dissolves into the FB story —
a useful negative. If cell A ≈ 0.0 while the measure cell is 0.23, the per-sample vector TD is
uniquely destructive, which is a genuine finding.

---

## 2. The mask, and why it must go in cell A

`m_g(s) = 1[s≠g]` is **goal-dependent** while `φ(s,a)` is goal-agnostic. A goal-dependent
scalar coefficient in front of the bootstrap is precisely the step where `ψ(g)` fails to factor
out of the recursion, so a goal-agnostic joint-SF cell *with* the mask is algebraically
incoherent.

The mask also changes the object being computed. Unrolling with `T` = first hitting time of `g`:

- **no mask:** `Q = Σₜ γᵗ 1[sₜ=g]` — a discounted **visitation density**. This is FB's object.
- **mask:** `Q = γ^T` — a discounted **first-hitting time**.

The base index follows from the same choice. FB's diagonal term is `⟨Δᵢ, ψ(s′ᵢ)⟩`: the successor
measure counts *arrival at s′*, so `b(s) = ψ(s′)`. This is the **opposite** of E3's convention,
correctly so — E3's current-index base is forced by `rewards`/`masks` being tied to the current
index (`datasets.py:242-244`, where `rewards = −masks` exactly), whereas the ortho cell has no
reward and no mask at all. The 2026-07-09 audit's rule was *do not mix* the two conventions.
Cell A adopts the arriving-state convention wholesale; cell C adopts the current-index one.

Verified against the code: `measure_loss` contains **zero** references to `batch['masks']`, and
it orthonormalizes `B := ψ(next_observations)` — the ρ marginal, not the HER goal distribution.
(Our standalone `ortho_loss` uses `value_goals`; it is the odd one out, and is not used here.)

---

## 3. The two cells

Both live in `gciql_flow.py::sf_loss` on the GCIQL trunk. No new agent, no new file. The critic
class is the existing `GCLearnedBValue` (ψ trainable).

|  | **A — matched measure** | **C — absorbing / first-hitting** |
|---|---|---|
| `sf_base_index` | `next` → `b = ψ(s′)` | `current` → `b = ψ(s)` |
| `sf_mask` | `none` (m ≡ 1) | `goal` (m = 1[s≠g]) |
| `Q` computes | `Σ γᵗ 1[sₜ=g]` (density) | `γ^T` (hitting time) |
| `fb_forward_z` | true | true (**required**) |
| `project_z` | true | true |
| `sf_psi_ortho_coef` | 1000 | 1000 |
| ortho evaluated on | `ψ(next_observations)` | `ψ(next_observations)` |
| reads against | `C-fbz-projz-ortho1000` = **0.231 ± 0.030** | cell A |

Cell C's ortho is deliberately on `ψ(s′)` too, so the ortho term is *identical* across cells and
the recursion is the only difference.

Cell A is the estimator-axis control: identical to `cube_single_state_gciql_flow_bilinear_d50_C_proj.yaml`
in every respect except that the B×B measure loss is replaced by the per-sample vector TD.
Cell C has no FB counterpart; it asks whether a hitting-time value is the better goal-reaching
object, and it is coherent only because `fb_forward_z` gives φ the task the mask depends on.

---

## 4. Loss

```python
zarg  = (value_goals,)                    # fb_forward_z: phi(s,a,z), z = sg(psi(g))
b_goals = nobs if sf_base_index == 'next' else obs
_, phi, _   = critic(obs, value_goals, acts, *zarg, params=grad_params)
_, _,   b   = critic(obs, b_goals,     acts, *zarg, params=grad_params)   # psi(s') or psi(s)
if not sf_joint_psi:
    b = stop_gradient(b)                  # today's behaviour, bit-identical
m     = batch['masks'] if sf_mask == 'goal' else 1.0
_, tphi, _  = target_critic(nobs, value_goals, a_next, *zarg)

resid = phi - b - gamma * m * stop_gradient(tphi)
L_td  = mean(resid ** 2)

# FB orthonormality on psi, per ensemble head then averaged (same formula as measure_loss)
Cov   = einsum('hid,hjd->hij', psi_o, psi_o)          # psi_o = psi(next_observations)
o_off = 0.5 * sum((Cov * off)**2, axis=(1,2)) / sum(off)
o_diag = -mean(diagonal(Cov), axis=1)
L_ort = mean(o_off + o_diag)

L = sf_coef * L_td + sf_psi_ortho_coef * L_ort
```

`sf_joint_psi=False` reproduces today's loss exactly — one code path, no fork.

`psi_o` is always `ψ(next_observations)`. In **cell A** (`sf_base_index='next'`) `b` *is* that
tensor, so it is reused for the ortho term — one forward, not two. In **cell C**
(`sf_base_index='current'`) `b = ψ(obs)` and `psi_o` needs its own
`critic(obs, nobs, acts, *zarg)` call. Cell C therefore costs one extra critic forward per step;
this is accepted so that the ortho term is identical across the two cells.

### 4.1 Why the ortho is not optional

Un-stop-gradding the base creates a degenerate **global** minimum: `φ ≡ 0, ψ ≡ 0` gives
`resid ≡ 0`, hence `L_td = 0`. Nothing in the TD prevents it.

`L_ort` at `ψ=0` gives `C=0`, so `L_ort=0` and `∇L_ort=0` — the origin remains stationary — but it
is a strict local **maximum** along every direction, since near zero `L_ort ≈ −E‖ψ‖² < 0`. The
`−tr(C)` term inflates the norm and repels collapse. Contrast E3 v1, where the ψ=0 saddle was
*attracting* because the `{−1,0}` majority class was already fit there.

Under `project_z` the radial collapse is impossible by construction (`‖ψ‖² ≡ d`, so `tr(C) = d` is
constant and the diagonal term is inert). The sphere kills the radial collapse; the off-diagonal
term kills the angular one (all ψ pointing the same way). Two failure modes, two remedies.

### 4.2 The peaked kernel, for free

With `b = ψ = ψ̄` (sphere-normalized), the implicit reward of the SF telescope is

    r̂_g(s) = ⟨ψ̄(s), ψ̄(g)⟩/√d = √d · cos∠(ψ̄(s), ψ̄(g))

which is `√d` at `s=g`, and which the off-diagonal ortho term drives to 0 for independent
`s ≠ g`. So at the ortho optimum `r̂_g(s) ≈ √d · 1[s=g]` — the goal indicator, up to scale.

E3 buys this peaked kernel with a supervised regression on `batch['rewards']`; this cell buys it
with a whitening constraint on ρ. **Same target, two routes** — supervision vs geometry.

---

## 5. Knobs

New config entries in `get_config()`, all default-OFF so every existing cell is unchanged:

| knob | type | default | meaning |
|---|---|---|---|
| `sf_joint_psi` | bool | `False` | un-stop-grad the base so the SF TD trains ψ |
| `sf_base_index` | `'current'` \| `'next'` | `'current'` | `b(s) = ψ(s)` vs `ψ(s′)` |
| `sf_mask` | `'goal'` \| `'none'` | `'goal'` | use `batch['masks']` vs `m ≡ 1` |
| `sf_psi_ortho_coef` | float | `0.0` | λ on FB orthonormality of ψ |

**`sf_psi_ortho_coef` (on ψ) is a different knob from the existing `sf_ortho_coef` (on φ).**
Conflating them would be easy and disastrous: φ-ortho at λ=1000 drove those cells to 0.001–0.004.
The new knob is named to make the leg explicit.

`project_z` already exists but is only passed to `GCFBValue`/`GCTwoTowerCritic`
(`gciql_flow.py:906`); it must be threaded into `GCLearnedBValue`.

### 5.1 Guards (asserts in `create()`)

These encode the algebra so a misconfigured cell fails loudly rather than producing an
uninterpretable number — the failure mode this program keeps repeating.

1. `sf_joint_psi` ⇒ `sf_base == 'learned'` **and not** `sf_reward_regress`
   (exactly one gradient path into ψ).
2. `sf_joint_psi` ⇒ `sf_psi_ortho_coef > 0`
   (else `φ≡0, ψ≡0` is a global minimum, not a bad local one).
3. `sf_mask == 'goal'` ⇒ `fb_forward_z`
   (a goal-dependent absorption coefficient cannot sit in a goal-agnostic recursion).
4. **not** (`sf_base_index == 'next'` **and** `sf_mask == 'goal'`)
   (mixes the arriving-state and current-index conventions — the exact class of bug fixed in
   E3 on 2026-07-09).
5. `sf_base == 'learned'` ⇒ `sf_reward_regress` **or** `sf_joint_psi`
   (a learned ψ with no training signal). Replaces the current stricter assert.

---

## 6. Tests (TDD — written and failing before implementation)

In `tests/test_gciql_sf_joint_psi.py`, jax-subprocess style, matching
`tests/test_gciql_sf_feature_td.py`.

1. **γ=0, cell A, decisive.** Target collapses to `ψ(s′)`, so
   `L == mean((φ − ψ(s′))²) + λ·L_ort`, both terms computed by hand.
2. **Ortho identity.** Numerically verify `L_ort == ½‖C − I‖²_F − d/2` for a random batch.
   This checks the *mathematics*, not merely the code.
3. **Mask genuinely ignored** under `sf_mask='none'`: randomize `batch['masks']`; loss unchanged.
   (Under `sf_mask='goal'` the same randomization must change it.)
4. **ψ trains under joint, and only via the TD.** With `sf_joint_psi=True`, ψ params move after
   an update; with `False` they do not. `sf_reward_regress` is off in both.
5. **`project_z`**: `‖ψ(·)‖ = √d` for every row; consequently `o_diag` is constant across batches.
6. **Backwards compatibility.** Default config → `sf_loss` returns exactly today's value
   (compare against the stop-grad/current-index/goal-mask expression computed by hand).
7. **All five guards** reject their bad combinations and accept the two valid cells.

---

## 7. Configs

- `cube_single_state_gciql_flow_sf_joint_ortho1k.yaml`
  → `run_group: GCIQL-State-Flow-SF-JointPsi-Ortho1k`
- `cube_single_state_gciql_flow_sf_joint_ortho1k_absorb.yaml`
  → `run_group: GCIQL-State-Flow-SF-JointPsi-Ortho1k-Absorb`

Both: `value_arch=bilinear`, `value_latent_dim=50`, `bc_coeff=3.0`, flow actor,
`sf_feature_td=true`, `sf_base=learned`, `sf_joint_psi=true`, `sf_psi_ortho_coef=1000`,
`fb_forward_z=true`, `project_z=true`, 500k steps, 3 seeds, `eval_episodes=50`.

Cell A adds `sf_base_index=next`, `sf_mask=none`.
Cell C adds `sf_base_index=current`, `sf_mask=goal`.

**Distinct wandb groups, never reused.** `results_map.py` keys on the group; relaunching a
corrected cell into an existing group silently averages good and bad seeds (this bit us on the
E3 re-run, 2026-07-09).

---

## 8. Diagnostics

Log under `sf/`: `psi_norm_mean`, `ortho_off`, `ortho_diag`, `phi_absmax`, `target_absmax`,
`q_mean`.

Known hazards and how to read them:

- **Radial ψ→0 collapse** — impossible under `project_z`, but log `psi_norm_mean` anyway; it
  must sit at `√50 ≈ 7.07`.
- **Angular collapse** — watch `ortho_off`. If it plateaus high, ψ has not decorrelated and the
  peaked-kernel argument (§4.2) never took hold; the cell then reads differently.
- **Unbounded measure growth in `q_mean`** — the PSM work established this is *benign* for a
  successor measure (linear diagonal term, no upper bound). It must **not** be misread as
  divergence. The flow actor normalizes its Q-term by `sg[|q|.mean()]`, so it is scale-invariant.

---

## 9. Anchors

| cell | @500k |
|---|---|
| bilinear GCIQL, scalar TD (control) | 0.904 ± 0.017 (13 seeds) |
| `tau0.5` (no improvement operator) | 0.920 ± 0.016 |
| `C1-fbz-ortho1000` (measure, no sphere) | 0.255 ± 0.052 |
| **`C-fbz-projz-ortho1000` (matched baseline for A)** | **0.231 ± 0.030** |
| `C4-fb-rebuild` (measure + sphere + ortho **1.0**) | 0.165 ± 0.030 |
| E3 rewreg-shift (corrected, running) | 0.000 (1 seed) |
| E3 rewreg-shift-cond (corrected, running) | 0.036 (2 seeds) |
| SF 8-cell grid (**confounded**, retracted) | 0.08 – 0.16 |
| SF + **φ**-ortho λ=1000 | 0.001 – 0.004 |

---

## 10. Cost and sequencing

6 runs (2 cells × 3 seeds) @500k. Eight GPUs currently carry 8 stragglers from the `psi_base`
campaign; this queues behind them or packs at 2/GPU.

After collection: add both groups to `scripts/results_map.py`, run
`compile_results_csvs.py --refresh`, and fold the result into
`PAPER/fb-gciql-bridge/sf-design-space.md` (§2, and the 2×2 in §1.2 above).

## 11. Out of scope

- FB's mixed z-sampling (`measure_z_random_frac`) threaded into `sf_loss` — design-space item 2.
- Min-L2-norm double-SF target — item 4.
- Asymmetric reward model `⟨ψ_a(s′), ψ_b(g)⟩` — item 5.
- The empty top-right 2×2 cell (measure estimator + reward-regressed ψ).
- A shared-frozen-ψ control for the `psi_base` frozen arm (separate, unrelated gap).

One change at a time.
