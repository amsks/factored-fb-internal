# G4 Spearman — adversarial verification of `eval_measure_g4_spearman.py`

**VERDICT: ARTIFACT.** The model readout subtracts `κ(x₀)` while zeroing its
paired `r(x₀)` term, injecting a per-box offset of `−1[x₀∈box]/(1−γ) = −100·1[x₀∈box]`
into a quantity whose true scale is 0–100. Boxes containing `x₀` are exactly the
high-MC boxes, so the injected term is anti-correlated with the ground truth.
With the term removed, all 6 checkpoints **pass** the pooled 0.65 gate.

## Reproduction

Instrumented copy of the script (scratchpad `g4_verify.py`, repo file untouched),
same flags, separate process: pooled Spearman **0.160 / 0.166 / 0.174** (density
s0/1/2) and **0.132 / 0.144 / 0.177** (partition s0/1/2) — bit-identical to the
author's JSONs. Determinism (item 6) **confirmed**, across processes.

## The bug (item 2)

Spec 18 §1 / spec 17b: `Q̂_r = ( r(x₀) + (w − κ(x₀))ᵀr ) / (1−γ)`. The comment in
the spec is explicit that `r(x₀) − κ(x₀)ᵀr` is a **de-biasing pair** ("exact-r at
x₀ replaces the smoothed estimate"); the two cancel, leaving `wᵀr/(1−γ)`, which is
exactly the MC object `Σ_k γ^k 1[x_{1+k}∈box]` (both count from the next state).

The script calls `compute_density_primary(0.0, w, kappa_x0, r_pool, γ)` /
`compute_Qr_from_dots(0.0, p·z_r, φ(x₀)·z_r, γ)` — `r_x0=0` kills the first half of
the pair and keeps the second. Verified numerically: `author == mm_w − κ_box/(1−γ)`
to 3e-14. For `r = 1_box` the correct `r(x₀)` is not 0, it is `1[x₀∈box]`.

Damage is monotone in box size: `frac(x₀∈box)` = 0.02 / 0.05 / **0.32** at
mult 1/2/4. At mult 4 the author's `model_mass_mean` is **−0.21** against
`mc_mean = 24.9` — the symptom he saw and attributed to "numerical noise". It is
not noise; it is a subtracted indicator. Correct readout mean is **32.9** vs MC 24.9
(the ~1.3× is MC horizon truncation, γ²⁰⁰=0.134, plus ε-kernel box widening) — i.e.
the corrected object is calibrated in magnitude too, not merely in rank.

## Item 4 — probe sanity floor/ceiling (decisive control)

| control fed as "model mass" | pooled Spearman |
|---|---|
| the MC estimate itself | **1.000** |
| monotone transform `log1p(mc)` | **1.000** |
| pure noise | 0.011–0.034 |
| perfect ordering on `mc>0`, arbitrary within the `mc==0` ties | 0.751–0.772 |

The MC side, the box geometry, and the Spearman function are **sound**. The last
row is the realistic ceiling: 0.65 is reachable but the headroom is only ~0.10.

## Item 3 — `degenerate_frac ≈ 0.60`

"Degenerate" = MC mass exactly 0 (box never entered): 524/864. `measure_readout_head.spearman`
uses `np.argsort(np.argsort(x))` — **ordinal ranks, no tie averaging** — so those
524 points get an arbitrary permutation of ranks fixed by array order. Tie-corrected
(`scipy.rankdata`) values are ~0.05 higher; this is a real but second-order defect.

Non-degenerate subset (`mc>0`, n≈340), tie-corrected:

| ckpt | author readout | corrected (`wᵀ1_box/(1−γ)`) |
|---|---|---|
| dens s0/s1/s2 | 0.371 / 0.382 / 0.398 | **0.801 / 0.796 / 0.802** |
| part s0/s1/s2 | 0.327 / 0.369 / 0.364 | **0.791 / 0.805 / 0.783** |

## Corrected gate read

Using the author's own `spearman` and his own pooled statistic, readout =
`wᵀ1_box/(1−γ)` (density) / `pᵀ1_box/(1−γ)` (partition):

| ckpt | author | corrected pooled | tie-corrected | gate ≥0.65 |
|---|---|---|---|---|
| d18_cs_dens_s0 | 0.160 | **0.684** | 0.778 | PASS |
| d18_cs_dens_s1 | 0.166 | **0.702** | 0.789 | PASS |
| d18_cs_dens_s2 | 0.174 | **0.676** | 0.767 | PASS |
| d17_cs_part_s0 | 0.132 | **0.692** | 0.790 | PASS |
| d17_cs_part_s1 | 0.144 | **0.687** | 0.784 | PASS |
| d17_cs_part_s2 | 0.177 | **0.713** | 0.777 | PASS |

The self-consistent full-Q variant (restore `r(x₀)=1[x₀∈box]`, keep `−κ(x₀)`) gives
0.61–0.65 pooled / 0.73 tie-corrected — also ~4× the author's number. The residual
gap to `w`-only is the hard-indicator-vs-smoothed-kernel mismatch, ×100 amplified.

## Item 5 — per-multiplier (corrected, tie-corrected, all 6 ckpts)

| mult (half-width) | author | corrected |
|---|---|---|
| 1.0 (0.4) | 0.24–0.30 | **0.41–0.48** |
| 2.0 (0.8) | 0.38–0.42 | **0.59–0.64** |
| 4.0 (1.6) | 0.24–0.29 | **0.83–0.85** |

**Pooling is not innocent.** Box size drives both MC and model mass, so pooling
injects a common size factor that *inflates* the rank correlation: the pooled 0.68
sits above every individual column except mult=4. Within-start median Spearman
(removes any cross-start normalization concern) tells the same story: 0.48 / 0.59 /
0.87 corrected. Honest read: **ordering is strong at coarse scale, mediocre at the
finest scale**, and the pooled PASS is partly a resolution artifact. Any future G4
read should state which column is gated.

## Item 1 — policy match (checked, no defect found)

- Both objects bootstrap `a⁺ ~ actor(s′, g)`, a one-shot stochastic flow draw
  (`density_loss` / `delta_partition_loss`). The MC rollout calls
  `sample_actions(..., temperature=0.0)`, but for `actor_kind='flow'` the flow
  branch does `del temperature` and draws fresh `N(0,I)` noise from the seed — so
  the MC policy **is** the bootstrap policy. (The `temperature=0.0` in the script
  reads like determinism and is not; harmless here.)
- Goal conditioning matches: the measure is trained on `g = value_goals` with
  `value_p_randomgoal = 0.3`, so a uniformly-random dataset row as `g` is in the
  training goal distribution, and the same `g` is used for `a₀`, the rollout, and
  the readout.
- Teleport uses the npz `(qpos,qvel)` + `mj_forward`; MC accumulates
  `Σ_t γ^t 1[x_{1+t}∈box]`, matching the next-state convention. Minor residual
  biases (horizon-200 truncation ≈13%, early `term/trunc` break on the singletask
  goal) are monotone/rare and do not affect the verdict.

## What must NOT be concluded

The "both measures order occupancy no better than chance" claim is **withdrawn**.
It does not license spec 19 §3's "fix the measure before touching shaping". The
weaker, defensible statement is: *coarse-scale occupancy ordering is good
(0.83–0.85 at half-width 1.6); fine-scale ordering is moderate (0.41–0.48 at
half-width 0.4)*. Whether ORS shaping needs fine-scale ordering is the real question.

Fix is one line per branch: pass `r_x0 = 1[x₀∈box]`, or drop the `−κ(x₀)` term and
read `wᵀ1_box/(1−γ)` directly. Nothing in this session was edited, committed, or launched.
