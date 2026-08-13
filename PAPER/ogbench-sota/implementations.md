# implementations.md — experiments specified by `phenomenon.tex` (ogbench-sota)

*2026-07-20. Concrete specs for the final section of `phenomenon.tex` (why the bilinear form
works). Ordered by information-per-GPU-hour. Substrate for all cells: the standard bridge
substrate (cube-single-play state, GC-IQL, 3 seeds @500k, flow actor unless stated;
`tools/wandb_mode_shim/gciql_flow.py` knob conventions: default-off, param-tree-preserving,
tests before launch). Every group → `scripts/results_map.py` + CSV refresh + DECISIONS entry.*

## 1. T5a — late-fusion control (decides Layer 1; the highest-value cheap cell)

**Arch knob** `value_arch='latefusion'`: two towers bit-identical to `bilinear` d50
(φ(s,a) ∈ R^50, ψ(g) ∈ R^50), readout = 2-layer 512 MLP on concat [φ; ψ] instead of the
inner product. Same twin heads, same V head treatment (latefusion V too), same everything
else.
**Protocol:** (i) full training run, 3 seeds; (ii) frozen-critic rerank N-sweep
(N ∈ {1,8,32,128}) with the shared behavior flow — the rerank number is the primary read,
success secondary.
**Pre-registered prediction (phenomenon Pred. 1):** rerank @N=32 ≤0.65 (MLP band), NOT
0.91–0.96 (bilinear band). If late-fusion reranks in the bilinear band, Layer 1
(gradient-pooling-through-the-linear-readout) is FALSIFIED — book it and strip Layer 1 from
the claim; Layers 0/2/3 carry the explanation alone.

## 2. T5b — parameter-matched MLP (closes the 2.06× confound)

`value_arch='mlp'` with hidden widths scaled ×1.44 (≈2.06× value-head params, matching
bilinear d50's two towers). 3 seeds + rerank.
**Prediction:** stays in the MLP band on both success (~0.71) and rerank (~0.6) — parameters
were never the mechanism. Any material lift = the confound was real; re-normalize every
bilinear-vs-MLP claim in both papers against this cell.

## 3. srank / effective-rank trajectories (checkpoint-only, CPU)

Script `scripts/rescuing_fb/t5c_srank.py`: on existing checkpoints (bilinear d50 50k→500k,
MLP 50k→1M, the T2 checkpoint set), compute srank(δ=0.01) and participation ratio of (a)
φ(s,a) features (bilinear) / penultimate layer (MLP), on a fixed 4096-transition batch;
plus the cf-probe's U/A decomposition per checkpoint if cheap.
**Prediction:** MLP penultimate srank collapses over training (implicit
under-parameterization) while bilinear φ srank stays ≳ task-family rank (~28); the collapse
curve should time-align with where the MLP's rerank ordering goes flat. Either outcome
sharpens Layer 1's "absent, not noisy" clause.

## 4. Peak-geometry probe + norm-capped φ (decides Prediction 2)

**4a (probe, CPU):** extend `scripts/rescuing_fb/t3_offpeak_multienv.py`: along rays
a_ref → a_DDPG at the 512 probe states, record ‖φ(s,a)‖ and Q; test whether the peak's
Q-gain is norm-carried (corr(ΔQ, Δ‖φ‖) high on cube, absent on antmaze).
**4b (intervention):** `phi_norm_cap` knob: hard cap ‖φ‖ ≤ c (c = p99 of on-support ‖φ‖,
computed online with sg) via φ → φ·min(1, c/‖φ‖). Retrain cube bilinear+DDPG+BC, 3 seeds.
**Predictions:** 4a positive on cube only; 4b shrinks peak frac 0.639 → ≤0.2 AND lifts
bilinear+DDPG+BC from 0.51 toward the flow band; flow-actor run within noise of 0.904
(the cap must not cost the ordering). 4b failing while 4a is positive = the cap is the wrong
surgery (try spectral norm on the φ trunk); both failing = residual-relocation reading is
wrong (Layer 3's extraction rule still stands on its own numbers).

## 5. Bounded ψ(s,g) (the BVN regime-inversion test)

`value_arch='bilinear_sg_bounded'`: ψ(s,g) leg with tanh output bound (scale = p99 of
trained ψ(g) norms) — targeting exactly the T2 diagnosis (no output bound + no V-target +
local pockets). 3 seeds.
**Reads:** stabilizes at bilinear level ⇒ the online/offline BVN inversion is an
optimization artifact; claim narrows to "ψ(g) is sufficient and safe." Stabilizes at MLP
level ⇒ state-conditioning also destroys the pooling (Layer 1 evidence). Still diverges ⇒
the pocket mechanism needs the V-target arm too.

## 6. T9 — rank of the true advantage (simulator, CPU-heavy)

From the atlas counterfactual banks: at ~200 probe states × goal grid, estimate
A*(s,·,g) by K-action counterfactual rollouts (the cf-probe machinery, actor-independent
action set = dataset-NN cloud); assemble the (s,a) × g advantage matrix; report ε-rank
spectrum.
**Prediction:** ε-rank(A*) ≤ ~28 at ε = the success-relevant resolution (matches the d≥28
flat band). A high-rank result would break Layer 0's sufficiency story for the *advantage*
(and imply the win is even more Layer-1-dependent) — either way it calibrates the d floor.

## 7. Scene split-head (extends Layer 0 to compositional families)

`value_arch='bilinear_split'` on scene: Q = Σ_k φ_k(s,a)·ψ_k(g_k)/√d with per-factor goal
towers (k = scene's object/door factors, from the env's goal layout), d_k = 16 each. 3
seeds, both actors.
**Reads:** recovers ≥ the MLP's 0.52 ⇒ scene was a span failure of one product (Layer 0
extended, paper gains the "factored task families need factored heads" rule). Stays ≤0.30 ⇒
scene failure is not span — escalate to the shared-rep audit (HIQL-style probe of what the
scene value loss trains).

## Cross-references

- The two-paper positioning sentence lives in phenomenon §6 item 7; RG-FB v1 spec =
  `PAPER/rescuing-fb/specs/06-phenomenon-implementations.md` item 4.
- Confound ledger this file retires when run: 2.06× params (item 2), BVN pairing (item 5),
  scene untested axis (item 7) — all currently listed as caveats in `ogbench-sota.tex`.

## 8. Bilinear-Q + MLP-V (which leg carries the bilinear win?) — added 2026-07-21

`value_arch='bilinear_mlpv'`: bilinear critic Q=phi(s,a).psi(g)/sqrt(d) with a PLAIN-MLP value
head V(s,g) (GCValue). Every other bilinear cell bilinearizes V and Q symmetrically; this is
the only clean isolation of V's bilinearity (dueling uses MLP V but changes Q's form too).
Pairs with T5a late-fusion: latefusion asks "is the inner PRODUCT load-bearing?"; this asks "is
V's bilinearity load-bearing, or only Q's low-rank goal bottleneck?". 3 seeds, 500k, flow actor
+ frozen-critic rerank. Assert-lattice: MLP V has no psi, so refuse actor_cond='z'/measure/z
paths. Tests: `tests/test_gciql_sota_archs.py::test_bilinear_mlpv_split_v_and_q`.
**Pre-registered read:** stays in the bilinear band (~0.90 / rerank ≥0.9) ⇒ V-bilinearity is
irrelevant, the win is Q's goal bottleneck alone (pure Layer 0); drops to the MLP band (~0.71 /
rerank ~0.6) ⇒ symmetric bilinearization is a real mechanism (the value_loss comment about V
"confounding the readout" is load-bearing, and both legs need the bottleneck).
Group `GCIQL-State-Flow-BilinearMLPV-d50`; config `cube_single_state_bilinear_mlpv_d50.yaml`.
