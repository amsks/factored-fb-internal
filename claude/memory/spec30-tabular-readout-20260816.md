---
name: spec30-tabular-readout-20260816
description: "2026-08-16 spec30 Item4 tabular ground truth: the zero-shot de-biased readout is CORRECT — reproduces P(I−γP^π)⁻¹·r to 9.4e-14 relative (asserted 1e-11) in the exact-arithmetic case, both TD fixed point and readout checked. NO defect. But three previously-unstated limits, and one bites us: double_q min-over-heads sub-normalizes w (30% low in test), and that IS the shipped default in eval_density_zeroshot.py — so every tab:zeroshot number went through the one path the proof does NOT cover"
metadata:
  node_type: memory
  type: project
  originSessionId: 576b53a0-d404-4a89-9380-6fece089b102
  modified: 2026-08-16T22:58:32.633Z
---

Measure track. Spec 30a Item 4. Tests `tests/test_spec30_tabular_readout.py`
(27: 26 numpy + 1 jax-gated), helper `scripts/spec30_tabular_mdp.py`.

**The readout under test** (DENSITY-FB-EXPLAINED §3.5):
`Q̂_r(s₀,a) = ( r(x₀) + (w − κ(x₀|U_P))ᵀ r(U_P) ) / (1−γ)`

**★ NO DEFECT.** With a hard kernel, pool = whole state space and κ a true Dirac,
the shipped `compute_density_primary` reproduces `P[s,a,:](I−γP^π)⁻¹·r` to
**9.4e-14 relative** on four unrelated rewards, asserted at 1e-11 (not a loose
tolerance). Both halves checked: the shipped TD target `(1−γ)k+γw̄` is iterated
to its population fixed point (= ν exactly in hard mode, = νK in soft) and THAT
is fed to the shipped readout. First time this formula has been checked against
exact linear algebra; several published numbers flow through it.

**Conventions now pinned** (previously unwritten): the measure starts at the
SUCCESSOR x₁ (`Q_std = r(x₀) + γ·Q̂`, both directions tested); reward attaches to
states the measure lands on; ν is a probability vector so the single `1/(1−γ)`
is the entire rescale (pinned at 4 discounts); `density_loss` carries NO terminal
mask, so an absorbing self-loop is the only consistent semantics.

**The de-biasing pair is load-bearing and now regression-guarded.** It cancels
term-by-term in float64; dropping either half leaves exactly `∓r(x₀)/(1−γ)` =
**61% of the true value at γ=.99**, and for a box-indicator reward drives a mass
estimate NEGATIVE. That is exactly the G4 false-null geometry from
[[spec19-ors-implemented-20260806]], now a permanent test.

**⚠ THREE LIMITS — behaviour, not bugs, but never stated before:**
1. The pair corrects immediate-term **smoothing**, NOT pool **support
   truncation**. With x₀ in the pool κ is Dirac, the pair is identically 0, and
   truncation error passes straight through uncorrected.
2. Soft-mode exactness requires `r − Kr` constant over the visited support
   (proved exact there). Otherwise it extrapolates a single-state bias with
   ~100× amplification.
3. **`double_q` min-over-heads sub-normalizes w** (readout 30% low in the test
   case), so the exactness claims do NOT cover that path.

**⚠ WHY LIMIT 3 MATTERS HERE:** `double_q: bool = True` is the DEFAULT in
`scripts/rescuing_fb/eval_density_zeroshot.py` (`w_h.min(axis=0)`, ~line 267),
so **every tab:zeroshot cell was computed through it**. `min` of two simplex
rows does not lie on the simplex — mass < 1 — and the deficit varies with
(s,a,g), so it is not a uniform rescale. Mitigations before panicking: the table
reports ROLLOUT success (goal chosen by argmax of Q̂_r), not the readout value,
so a monotone-per-candidate scaling would not move it; and every Density-FB arm
used the same path, so within-table comparisons are consistent. **Untested: does
a varying sub-normalization change the DECODE ranking?** Cheap decisive check —
rerun one cell with `double_q=False` and see if the number moves. NOT YET RUN.

Related: [[spec19-ors-implemented-20260806]], [[spec18-density-measure-20260806]],
[[spec30-review-response-20260816]].
