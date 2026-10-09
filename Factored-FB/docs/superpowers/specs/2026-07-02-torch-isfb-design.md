# SG-ISFB on the canonical torch FB agent — design

**Date:** 2026-07-02
**Branch context:** `gciql-fb-bridge` (new code lands under `agents/fb/isfb/`)
**Spec source:** `PAPER/fb-modified/isfb.tex` (Scalar-Gated In-Sample Forward–Backward)

## Motivation

An audit of the existing JAX ISFB implementation (`tools/wandb_mode_shim/isfb_flow.py`,
built on `GCIQLFlowAgent`) found it is **not faithful FB**: it silently drops two
load-bearing, *matched* FB design decisions — (a) the training z-distribution
(canonical FB trains a 50/50 mix of random-sphere z and z=B(goal); the JAX bridge
only ever uses z=B(value_goals)), and (b) reward-weighted zero-shot z-inference
`z=E[r·B(s')]` (only present in an external eval arm). The JAX substrate was built
to be GCIQL, so making it byte-faithful to FB would mean re-deriving FB inside it.

**Decision:** reimplement SG-ISFB in PyTorch on top of the canonical FB agent
(`agents/fb/`), where "standard FB" already exists and is validated. The **only**
additions over standard FB are the isfb.tex pieces. Every incidental FB decision
(z-mix sampler, √d sphere on z and B, `ortho_coef=1.0`, EMA 0.005, F concat arch,
obs normalizer, orthogonal init, reward-weighted eval inference) stays canonical by
construction.

### Consequence accepted

The `16→87` (C4→B4) gap in isfb.tex is a **JAX/GCIQL-substrate** phenomenon. On
faithful torch FB the plain successor-measure objective already gets **~48%** on
cube (our anchor), not 16%. The comparison band therefore moves to the torch FB
family and is re-established natively on this substrate.

## Scope (this pass)

isfb.tex **Experiment 1 only** (exact-goal), with the **full variant set**
(`coord`, `scalar_gate_hard`, `scalar_gate_awr`). Exp2 (full-support z) and Exp3
(inferred-z actor) are deferred until Exp1 lands. This matches the spec's own
staging ("Do not start with the full zero-shot multi-task setting").

## Architecture

Mirror the existing `agents/fb/flow_bc/` subclass pattern. New files only; the
canonical `agents/fb/agent.py` and `agents/fb/model.py` are **not modified**.

### `agents/fb/isfb/model.py` — `ISFBModel(FBFlowBCModel)`
Inherits the flow actor, `_forward_map` (F), `_backward_map` (B), `project_z`
(√d sphere), obs normalizer, encoders. Adds exactly one module:

- `_value_forward_map` (F_V): a **second `ForwardArchiConfig` instance, identical
  architecture to `_forward_map`**, always called with a **zeroed action**. This
  is arch-faithful (same z-injection, same widths, same `num_parallel=2` ensemble),
  so `M=⟨F,B⟩` and `N=⟨F_V,B⟩` live in comparable spaces and reuse the exact FB
  forward code path. *(Chosen over a bespoke `(obs,z)→ℝ^d` MLP.)*

`_prepare_for_train` also deep-copies `_target_value_forward_map` (EMA target for
the N⁻ bootstrap).

### `agents/fb/isfb/agent.py` — `ISFBAgent(FBFlowBCAgent)`
Keeps the flow actor and `update_actor` **unchanged** (isfb.tex §4: the actor is
inherited and goes greedy on `q=⟨F(s,a,z),z⟩/√d`, which the flow Q-term already
does). Changes:

- Override the critic update `update_fb → update_isfb` (the SG-ISFB loss below).
- Add `_value_forward_map` params to the **forward optimizer** (`lr_f=1e-4`,
  matching canonical FB's optimizer grouping).
- EMA `_target_value_forward_map` at `f_target_tau=0.005` alongside F/B/left_enc.
- New config: `measure_insample`, `alpha_n`, `alpha_m`, `awr_beta`, `awr_wmax`
  (expectile `τ` and `ortho_coef` already exist on FBAgent).

## `update_isfb` loss (isfb.tex Eqs. 1–5, 16–20)

Reuses FB's own `fb_successor_terms` / `off_diag` helpers verbatim for the measure
terms and ortho. Per batch, on the online nets:

```
M_ij = ⟨F(s_i,a_i,z_i), B(s'_j)⟩ / √d
N_ij = ⟨F_V(s_i,z_i),   B(s'_j)⟩ / √d
q_i  = ⟨F(s_i,a_i,z_i), z_i⟩ / √d
v_i  = ⟨F_V(s_i,z_i),   z_i⟩ / √d

L_V  = expectile_τ(sg q_i − v_i)                       # only action-selection operator
A_i  = sg(q_i − v_i)                                    # scalar advantage
w_i  = hard: τ·1[A>0] + (1−τ)·1[A≤0]
     | awr:  clip(exp(β·A), 0, w_max)
     | coord: (no scalar gate; see below)
wbar = w / (mean(w) + ε)
L_N  = mean_i wbar_i · (sg M_ij − N_ij)²                # weighted vector regression
     | coord: expectile_τ(sg M_ij − N_ij) elementwise   # pre-refinement baseline
Δ_ij = M_ij − γ · sg N⁻_{i',j}                          # target F_V, B at s'; NO a'~π
L_M  = fb_offdiag(Δ) + fb_diag(Δ)                       # FB's own two terms
L_⊥  = ortho(B)                                         # unchanged, coef 1.0

loss = 1[insample≠coord]·L_V + α_N·L_N + α_M·L_M + ortho_coef·L_⊥
```

**The single substantive divergence from `update_fb`:** the target measure uses
`N⁻ = ⟨target_F_V(s'), target_B⟩` instead of `⟨target_F(s', a'~π), target_B⟩`.
This is exactly `measure_bootstrap=value_measure`. No next-action is sampled in the
critic update (assertable in tests). `coord` mode omits `L_V`/`A`/`w` and fits N to
M with an elementwise expectile (isfb.tex Eq. coord_expectile).

Stop-gradients: `sg` on `q` in L_V, on `A`, on `M` inside L_N, on the target
branch (N⁻), and z detached from B (inherited from `sample_mixed_z`).

## z-conditioning and the experiment ladder

- **Exp1 exact-goal** = `train_goal_ratio=1.0` → `sample_mixed_z` always returns
  z=B(goal). No new sampling code; the existing FB sampler is reused.
- Exp2 (full-support, `train_goal_ratio<1`) and Exp3 (inferred-z actor) are
  out of scope here but require no structural change — Exp2 is a config flip; Exp3
  would train the actor on `reward_inference` z (already implemented on FBModel).

## Comparison band (torch FB family, this substrate)

| Cell | Config | Role |
|---|---|---|
| C4-analog | plain torch FB (FlowBC, d50) | failed/weak measure baseline (~48%) |
| coord-ISFB | `measure_insample=coord` | does any expectile measure help? |
| SG-ISFB hard | `measure_insample=scalar_gate_hard` | coherent IQL-style operator |
| SG-ISFB awr | `measure_insample=scalar_gate_awr` | smoother extraction |
| B4-analog | torch FB `critic_mode='iql'` | target band |

**Open item (resolved to default):** if a torch FB-IQL (`critic_mode='iql'`) cube
number does not already exist, the implementation plan includes running it (cheap —
existing knob) so the B4-analog band is real on this substrate.

## Testing (TDD, `.venv/bin/python -m pytest`)

Unit tests written before wiring:
1. Shapes of M, N, q, v (`[num_parallel, B, B]` and `[num_parallel, B]`).
2. `L_V` equals the IQL expectile of `(sg q − v)`.
3. Hard-weight branch recovers the τ / (1−τ) split by sign of A.
4. AWR-weight branch matches `clip(exp(βA),0,wmax)` then normalization.
5. `coord` branch = elementwise expectile of `(sg M − N)`, no scalar gate.
6. **N⁻ bootstrap uses target F_V, not a policy next-action** — assert
   `update_isfb` makes no actor sampling call.
7. Ortho term identical to canonical FB's.
8. One smoke train step on a **random** batch (all-zeros batch is rank-degenerate
   and NaNs — a known gotcha from the JAX version).

## Non-goals

- No changes to `agents/fb/agent.py` / `model.py`.
- No Exp2/Exp3 machinery.
- No pixel path.
- No new z-sampling or eval-inference code (reuse FB's).
