# GCIQL→FB morph ladder — design + experiment spec

_2026-06-14. Reciprocal of the FB→IQL program: start from the method that works
(GCIQL) and impose FB's restrictions one at a time to find the single one that
collapses it. Companion to `2026-06-11-fb-psm-crl-gap-analysis-and-program.md` and
`2026-06-11-interaction-weighted-fb.md` (the reconciliation table this operationalizes)._

## Motivation

Every FB-side intervention lands **below plain FB (0.42)**: FB-IQL critic 0.20,
IWR 0.23, zHER 0.35, q_loss_coef 0.34. Importing IQL's *objective* into FB's *form*
does not help — evidence that the **bilinear form `Q=F·z` is the bottleneck, not the
value target**. The dual test isolates this directly: take working GCIQL
(0.89 cf-AUC / 72.3 success) and walk it toward FB one restriction at a time.

The reconciliation table (`interaction-weighted-fb.md §1.5e`) predicts the break
point: moving *down a column* (MLP value → bilinear, goal coupling held fixed) is
what drops cf-AUC 0.89 → 0.50. So the prediction is **G3 (the inner-product form)**.

This is the clean version of the `morph_readout.py` attempt, which was KILLED for a
measurement flaw (scored a value-with-no-policy on cross-policy grasps → all cells
≈0.55, not comparable to GCIQL's same-policy 0.89). The fix here: **every rung is a
full agent with its own actor, scored by end-to-end success**; cf-AUC is a secondary
readout measured on **each rung's own on-policy rollouts**.

## The ladder

Everything is held equal to GCIQL except the one named change per rung.

| rung | conditioning | critic form | status |
|---|---|---|---|
| G0 GCIQL | raw goal `g` | MLP `Q(s,a,g)` | anchor — exists (72.3 / cf-AUC 0.89) |
| G1 + flow actor | raw goal `g` | MLP | anchor — `gciql_flow`, exists (72.3) |
| **G2** | **`z=B(g)`, low-dim** | MLP `Q(s,a,z)` | **NEW** — isolates the z-bottleneck |
| **G3** | `z=B(g)`, low-dim | **bilinear `F(s,a)·B(g)/√d`** | **NEW** — isolates the inner-product |
| G4 | zero-shot `z=E[r·B]` | bilinear | cite `z_inference_ladder` (fragile) |
| G5 | — | + measure loss | = FB endpoint (cite FB-IQL 0.20 / plain FB 0.42) |

**Held fixed across G0–G3:** IQL expectile value/critic loss; **exact goal at eval**
(no zero-shot inference — this is the key separation from FB-IQL, which kept the
fragile inference path); flow-BC actor (= the G1 baseline, so the actor is constant
and already exonerated); same data and goal-sampling distribution; `z_dim = 50`
(FB parity).

**Deltas:** G1→G2 = raw goal vs low-dim `z=B(g)`. G2→G3 = MLP head vs inner-product.
G3 carries the bottleneck too (`GCBilinearValue.psi` *is* the low-dim embedding), so
the G2→G3 contrast attributes any collapse specifically to the inner-product form.

### Why G3 is NOT redundant with FB-IQL (0.20)

FB-IQL was bilinear-but kept two confounds G3 removes: (1) **ortho whitening** on B
(a measure-objective remnant), and (2) **zero-shot z-inference at eval** (shown
fragile by `z_inference_ladder` — z is a policy-family index). G3 drops both: no
ortho, and the exact goal is handed to the critic at eval (GCIQL's native path).
So G3 tests the bilinear *form* in isolation:
- **G3 collapses** ⇒ the inner-product is the wall, independent of inference/ortho.
- **G3 ≈ 0.89** ⇒ the form is fine; FB's failure is the inference + measure-objective,
  not the bilinear factorization. (Would reopen the inference leg as the real lever.)

## Implementation

Knobs on the JAX GCIQL agent (`third_party/ogbench/impls/agents/gciql.py`),
**default-off → byte-identical stock GCIQL**. Follows the `gciql_flow` injection
pattern (`tools/wandb_mode_shim/gciql_flow.py`) for the flow actor.

- `critic_form: 'mlp' | 'bilinear'` — `'bilinear'` swaps `GCValue` → the reference
  `GCBilinearValue` (`utils/networks.py:327`, `Q=φ(s,a)ᵀψ(g)/√d`) for both the value
  and critic modules, `latent_dim = z_dim`. No new math — reuses OGBench's own module.
- `goal_z_dim: int | None` — when set (G2), route the goal through a small **plain MLP
  backward encoder B** (mirrors FB's backward map; NOT the pixel-oriented `GCEncoder`)
  to a `z_dim`-vector `z=B(g)`, and feed `(s, z)` to the MLP value/critic. For G3,
  `GCBilinearValue.psi` already supplies this bottleneck, so `critic_form='bilinear'`
  implies it.
- Flow-BC actor reused from the `gciql_flow` injection for all rungs (constant actor).

**Decisions on the two open points (defaulted):**
(a) G2's B is a **plain low-dim MLP** (FB-backward-map analog), not `GCEncoder` —
    keeps the bottleneck the only change and avoids pixel wiring on a state task.
(b) G4/G5 stay **cited** from existing results; optionally re-runnable in-agent later
    as a same-agent confirmation, but out of scope for the decisive read (YAGNI).

## Measurement

- **Primary — eval success.** Directly comparable to plain FB 0.42 and GCIQL 72.3.
- **Secondary — cf-AUC** via `counterfactual_value_probe`, measured on **each rung's
  own on-policy rollouts** (same-policy grasps). This is the explicit fix for the
  `morph_readout` cross-policy flaw; the value is always scored against the policy
  that produced the states.
- **Dissociation read:**
  - G2 ≈ 0.89 & **G3 collapses** ⇒ inner-product form is the binding restriction
    (confirms the thesis from the GCIQL side).
  - **G2 already collapses** ⇒ the low-dim z-bottleneck is the binding restriction.
  - G3 ≈ 0.89 ⇒ the form is fine; reopens inference/measure-objective as the lever.

## Runs

- 3 seeds × 500k, state `cube_single`, jax venv (GPU), `XLA_PYTHON_CLIENT_PREALLOCATE=false`
  to pack seeds per GPU, in tmux.
- wandb `entity=amsks`, groups `gciql_morph_g2` / `gciql_morph_g3`.
- Promote a clear winner to 10×1M only if it clears the noise floor.

## Tests / faithfulness

- Guard: `critic_form='mlp', goal_z_dim=None` reproduces stock GCIQL bit-for-bit
  (the default-off contract).
- Unit: `GCBilinearValue` critic output shape + finiteness; B-encoder emits `z_dim`;
  injection parity test mirroring `tests/test_gciql_flow_injection.py`.
- G3 reuses the reference `GCBilinearValue` ⇒ no new value-function math to audit.

## Scope / caveats

- State-only; pixel is out of scope.
- A null result (G2 and G3 both ≈ GCIQL) would contradict the thesis and is itself a
  publishable finding — the ladder is falsifiable in both directions.
- The flow actor (G1) is the constant baseline; this ladder says nothing new about the
  actor (already exonerated at 72.3).
