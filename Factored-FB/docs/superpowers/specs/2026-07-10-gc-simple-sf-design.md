# GC-SimpleSF: the missing (scalar TD + reward-grounded task vector) cell

**Date:** 2026-07-10
**Branch:** `gciql-fb-bridge`
**Living doc:** `PAPER/fb-gciql-bridge/sf-design-space.md`
**Reference:** raymondchua `simple_successor_features`, `agent/sf_simple.py` (read from source)
**Status:** design approved, not implemented

---

## 1. Why

Our bridge grid has an empty cell. Laying the three objectives side by side:

| | scalar TD | vector SF-TD | explicit linear reward model |
|---|---|---|---|
| bilinear GCIQL | ✅ | ✗ | ✗ |
| E3 (corrected learned-basis SF) | ✗ | ✅ | ✅ |
| **Simple SF (Chua et al.)** | ✅ | ✗ | ✅ |

Nothing we have ever run occupies the third row.

### 1.1 The unification that makes this precise

Write the SF Bellman residual as a **vector**, `Δ(s,a) = φ(s,a) − b(s′) − γ·φ̄(s′,a′) ∈ ℝᵈ`.

- **Canonical SF** (our `sf_feature_td`) fits every coordinate: `L = ½‖Δ‖² = ½ΔᵀIΔ`.
- **Simple SF** fits only the scalar `Q = ⟨Ψ, w⟩`. Assuming its reward factorization `r = b(s′)ᵀw`,
  the TD residual collapses:
  `Q − (r + γQ̄) = φᵀw − b(s′)ᵀw − γφ̄ᵀw = Δᵀw`, hence `L = ½(Δᵀw)² = ½Δᵀ(wwᵀ)Δ`.
- **FB / our measure graft**: `offdiag = ½·E_{i≠j}⟨Δᵢ, ψ(s′ⱼ)⟩²/d = ½ΔᵀCΔ/d`, `C := E_ρ[ψψᵀ]`.

**Same residual, three PSD test matrices — `wwᵀ` (rank 1), `I` (rank d), `C` (rank d, → `I` under
ortho).** Note where the reward-linearity assumption enters: it is exactly what lets `r` cancel
against `b(s′)ᵀw`. Without it, Simple SF's scalar loss is not a projection of the SF residual.

Also: FB's ortho is *semantic*, not merely anti-collapse. With `M(s,a,ds′) = ⟨φ,ψ(s′)⟩ρ(ds′)`,
the successor feature of base `b` is `Ψ_b = E_ρ[bψᵀ]·φ`. Taking `b = ψ` gives `Ψ_ψ = C·φ`, so
**φ is the successor feature of ψ iff C = I**.

### 1.2 Measured ordering (2026-07-10, all 3-seed @500k)

| objective | test matrix | rank | score |
|---|---|---|---|
| bilinear GCIQL | `wwᵀ`-like (scalar) | 1 | **0.904 ± 0.019** |
| FB measure, B×B | `C` | d | **0.231 ± 0.030** |
| joint-ψ SF, per-sample (cell A) | `I` | d | **0.120 ± 0.016** |

The more Bellman equations imposed per sample, the worse. GC-SimpleSF tests whether the rank-1
row *needs* the explicit reward model, or whether GCIQL's direct use of the true reward already
subsumes it.

### 1.3 The claim under test

Simple SF's `w` receives gradient **only** from reward regression (`task.detach()` is passed to
both `update_critic` and `update_actor`; `sf_simple.py:283,288`). Our results show that
restricting ψ's gradient is catastrophic under an otherwise identical scalar backup:

| ψ's training signal | TD | score |
|---|---|---|
| end-to-end (value + critic losses) | scalar | 0.904 |
| end-to-end, with Ψ(s,a,z) (`B1` = `fb_forward_z` under IQL) | scalar | 0.844 ± 0.008 |
| none (frozen random ψ) | scalar | 0.036 |
| none (identity ψ(g)=g) | scalar | 0.012 |
| reward regression only (E3) | **vector** | 0.003 |

**Prediction: rung 3 lands well below rung 2, plausibly 0.0–0.2.** If it instead lands near 0.90,
the reward-grounding mechanism wins outright and the bridge's objective axis needs rewriting.
That is the point of running it.

---

## 2. Architecture

One new value class, `GCSimpleSFValue`, transcribing the reference's trunk split
(`sf_simple.py:51-84` FeatureNet, `:87-123` CriticSF):

```
h(s,a,z)  = trunk([s, a, z])       ∈ ℝᵈ      # single trunk, NOT ensembled
b(s,a,z)  = h / (‖h‖ + eps)        ∈ ℝᵈ      # reward BASIS; zero parameters of its own
Ψᵢ(s,a,z) = headᵢ(h)               ∈ ℝᵈ      # twin heads (ensemble=2), as CriticSF.Q1/Q2
z         = ψ(g)                              # task vector; sphere-normed when project_z
Qᵢ        = ⟨Ψᵢ, z_used⟩ / √d
```

`z_used = stop_gradient(z)` when `simplesf_ground_psi`, else `z`. `info=True` returns
`(v, Psi, z, b)` with **`z` never detached in the info tuple** — the reward loss needs its
gradient.

**The load-bearing property:** `b` and `Ψ` are *different vectors off a shared trunk*, and `b`
has no parameters. Its only gradient arrives through the trunk, from the TD. This is what makes
the reward model `⟨b, ψ(g)⟩` **asymmetric** — a network against a task vector — so it has no
`ψ = 0` stationary point. Our E3 tied both legs to one ψ network, producing the symmetric
`⟨ψ(s),ψ(g)⟩` whose gradient vanishes at zero; that saddle is what killed E3 v1 and forced
`sf_reward_shift`. **No shift is needed here.** With `‖b‖ = 1` and `‖z‖ = √d`, `r̂ ∈ [−1, 1]`,
which already covers the reward range `{−1, 0}`.

For the value net (`actions=None`, `ensemble=False`): `h_V(s,z) = trunk([s, z])`, one head.

Implementation notes:
- `trunk = MLP((*hidden_dims, latent_dim), activate_final=False, layer_norm=layer_norm)` — our
  `MLP` applies `Dense → gelu → LayerNorm` per layer with no activation on the final layer, so
  `h` is a raw pre-activation vector, matching the reference's final `nn.Linear(hidden, sf_dim)`.
  (Deviation: reference uses `LayerNorm → Tanh`; we keep our gelu/LN order. Held constant across
  both rungs, so it cannot explain a rung-2-vs-rung-3 difference.)
- `head = MLP((hidden_dims[-1], latent_dim))`, `ensemblize(·, 2)` when `ensemble` — matching
  `CriticSF.make_q()`'s `Linear(sf_dim, hidden) → ReLU → Linear(hidden, sf_dim)`.
- `psi = MLP((*hidden_dims, latent_dim))` as in `GCBilinearValue`.

---

## 3. The two cells — one knob apart

| rung | cell | ψ(g)'s gradient | isolates | anchor |
|---|---|---|---|---|
| 1 | bilinear GCIQL *(exists)* | value + critic losses | — | **0.904 ± 0.019** |
| 2 | **`simplesf-arch`** | value + critic losses | the **architecture** | vs rung 1 |
| 3 | **`simplesf-grounded`** *(the paper's cell)* | **reward regression only** | the **grounding** | vs rung 2 |

Rung 3 adds, for **each** of the value net and the critic:

```
L_rew = ( ⟨sg[b(·)], ψ(g)⟩/√d  −  r_g(s) )²
```

and detaches `z` inside `value_loss` / `critic_loss`, so `L_rew` is ψ's sole gradient path.
Rung 2 has neither. Everything else — data, HER mixture, flow actor, `bc_coeff=3.0`, γ=0.99,
target network, τ — is identical across all three rungs.

---

## 4. Three deliberate deviations from the reference

**(a) Backup: GCIQL's expectile, not the paper's DDPG min-twin.** Otherwise rung 2 would differ
from rung 1 by *architecture + backup* and isolate nothing. Safe because `tau0.5` — which strips
the improvement operator entirely — scores **0.920 ± 0.016**, so the operator is empirically free
on this task. Rung 3 is therefore "the paper's method with our backup," and the writeup must say
so. (The reference's target is `r + γ·min(Q̄₁,Q̄₂)` with `a′ ~ π(s′)`, `sf_simple.py:346-353` —
policy evaluation with pessimism, not a max.)

**(b) Reward index: `b` at the current index `(s,a)`, not `(s′,a′)`.** GCDataset ties `rewards`
**and** `masks` to `idxs == value_goal_idxs` (`datasets.py:242-244`; note `rewards = −masks`
exactly). The reference's `(s′,a′)` reflects its own transition convention. Mixing the two is the
class of bug fixed in E3 on 2026-07-09, so we adopt the current-index convention wholesale.

**(c) Two task vectors, not one.** Value and critic are separate flax modules with separate goal
legs. Grounding only the critic's ψ would leak goal information back through `V(s′,g)` into the
critic target. So each net's ψ is detached from its own losses and grounded by its own `L_rew`
against its own `b`. The reference has a single global `w`; we have `ψ_V(g)` and `ψ_Q(g)`.

---

## 5. Knobs and guards

```python
config.value_arch = 'mlp'              # extended: 'mlp' | 'bilinear' | 'simplesf'
config.simplesf_ground_psi = False     # rung 3: psi trained ONLY by L_rew (z detached in V/Q)
config.simplesf_reward_coef = 1.0      # weight on L_rew (summed over the value net and critic)
# reuses: project_z (sphere-norm on z), value_latent_dim (d=50)
```

Guards in `create()`:

1. `simplesf_ground_psi` ⇒ `value_arch == 'simplesf'`.
2. `value_arch == 'simplesf'` ⇒ **not** any of `sf_feature_td`, `measure_only`, `measure_coef>0`,
   `fb_forward_z`, `fwd_two_tower`, `isfb`, `sf_joint_psi` — each selects its own critic class or
   loss and would silently override.
3. `value_arch == 'simplesf'` ⇒ `psi_base == 'learned'` (the `psi_base` dispatch is for
   `GCBilinearValue`; the two must not compose silently).
4. `value_arch == 'simplesf'` ⇒ `actor_cond == 'goal'` (the flow actor sees the raw goal, as in
   every other cell).

---

## 6. Tests (TDD, written and failing before implementation)

In `tests/test_gciql_simplesf.py`, jax-subprocess style, matching `tests/test_gciql_psi_base.py`.

1. **Shape and readout.** `Q = ⟨Ψ, z⟩/√d` computed by hand; `Ψ` has twin heads `[2,B,d]`; the
   value net has one head `[B,d]`.
2. **`‖b‖ = 1`** for every row, and **`b` is not collinear with `Ψ`** — guards against the
   circular `b = φ` degeneracy (which would make `L_rew` demand `Q(s,a,g) ≈ r_g(s)`).
3. **Grounded**: ψ params receive gradient from `L_rew` and **exactly zero** from
   `value_loss + critic_loss` (via `jax.grad`, assert `< 1e-8`).
   **Ungrounded**: ψ params receive gradient from `value_loss + critic_loss`, and
   `simplesf_reward_loss` is absent from `info`.
4. **The trunk receives TD gradient in BOTH rungs** — `b`'s only training signal. Assert the
   trunk params move after an update in each rung.
5. **`L_rew` exactness**: equals `mean((⟨sg[b], ψ(g)⟩/√d − batch['rewards'])²)` summed over the
   value net and the critic, computed by hand, at the **current** index.
6. **No saddle by construction**: with `ψ = 0`, `∂L_rew/∂ψ ≠ 0` (contrast the symmetric E3 form,
   whose gradient vanishes there). Assert the gradient norm at a zeroed ψ is `> 0`.
7. **Guards** reject all four bad combinations and accept both valid cells.
8. **Backward compatibility**: `value_arch ∈ {mlp, bilinear}` produces bit-identical losses; the
   existing suites stay green.

---

## 7. Configs

- `configs/gciql/cube_single_state_gciql_flow_simplesf_arch.yaml`
  → `run_group: GCIQL-State-Flow-SimpleSF-Arch`
- `configs/gciql/cube_single_state_gciql_flow_simplesf_grounded.yaml`
  → `run_group: GCIQL-State-Flow-SimpleSF-Grounded`

Both: `value_arch=simplesf`, `value_latent_dim=50`, `bc_coeff=3.0`, flow actor, `project_z=true`,
500k steps, 3 seeds, `eval_episodes=50`. Grounded adds `simplesf_ground_psi=true`,
`simplesf_reward_coef=1.0`.

**Distinct wandb groups, never reused** — `results_map.py` keys on the group; relaunching into an
existing group silently averages good and bad seeds (this bit us on the E3 re-run, 2026-07-09).

---

## 8. Diagnostics

Log under `simplesf/`: `reward_loss`, `rhat_mean`, `rhat_std`, `b_norm_mean` (must be 1.0),
`z_norm_mean` (must be `√50 ≈ 7.071` under `project_z`), `psi_grad_norm`.

Reading them:
- `reward_loss` against the class prior. With `r ∈ {−1,0}` and ~79% of relabelled rewards equal
  to −1, a model that predicts the mean scores ≈ 0.166. **A plateau near the prior means the
  reward is not linearly readable in `b`** — which is the reference's load-bearing assumption
  failing, and would make rung 3 uninterpretable as a test of grounding.
- `b_norm_mean ≠ 1` ⇒ the normalization is not reaching the basis.
- Flat `psi_grad_norm` in rung 3 ⇒ ψ is receiving no signal at all; check the detach.

---

## 9. Anchors

| cell | @500k |
|---|---|
| bilinear GCIQL (scalar TD, learned ψ) | **0.904 ± 0.019** |
| `tau0.5` (no improvement operator) | 0.920 ± 0.016 |
| `B1` `fb_forward_z` under IQL — already Ψ(s,a,z) | 0.844 ± 0.008 |
| `zg_iql` (ψ(g)=g) | 0.44–0.51 |
| psi_base frozen ψ | 0.036 |
| psi_base identity ψ | 0.028 |
| FB measure (B×B) | 0.231 ± 0.030 |
| joint-ψ SF cell A (per-sample) | 0.120 ± 0.016 |
| E3 (reward-regressed ψ + vector TD) | 0.003 |

---

## 10. Cost and sequencing

6 runs (2 cells × 3 seeds) @500k. `psibase` has 3 E3 stragglers; `sfjoint` is complete and its
GPUs are free. Launch on the freed devices; do not disturb the stragglers.

After collection: add both groups to `scripts/results_map.py`, run
`compile_results_csvs.py --refresh`, and fold the numbers into
`PAPER/fb-gciql-bridge/sf-design-space.md` together with the `wwᵀ / I / C` unification (§1.1).

## 11. Out of scope

- The reference's DDPG min-twin backup as a separate cell (see §4a; `tau0.5` says it is free).
- A single shared task vector across V and Q (§4c).
- `lr_task = lr/10` and `update_task_every_step = 10` from `sf_simple.yaml` — we use one learning
  rate and update every step. If rung 3 collapses, this is the first HP to revisit.
- The empty 2×2 corner (measure estimator + reward-grounded ψ).
- Mixed z-sampling, min-L2-norm double target (design-space items 2 and 4).

One change at a time.
