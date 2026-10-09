# T2 — DR3 feature co-adaptation across training (2026-07-16)

Probe: `scripts/rescuing_fb/t2_dr3_stats.py` (jax CPU) → `t2_dr3_stats.json`;
figure `t2_coad_vs_step.png` (`scripts/rescuing_fb/t2_plot.py`); training-log curves
`t2_train_curves.json`. Fixed batch B=4096 (seed 20260716) from the raw ogbench
`cube-single-play-v0.npz`; a′ = the dataset next action; goals sampled once with the
training value-goal mixture (0.2 cur / 0.5 traj-geom / 0.3 random). φ = the explicit
φ-tower (bilinear, bilinear_sg) / penultimate LayerNorm activations re-applied from
params (mlp; nn.vmap blocks capture_intermediates; reconstruction check max|Δq|<1e-3).
Both ensemble heads agree to <0.01 everywhere; numbers below are head-means.

    coad = E[φ(s,a)·φ(s′,a′)] / sqrt(E‖φ(s,a)‖² · E‖φ(s′,a′)‖²)

## Numbers

| series | steps | coad | E‖φ‖² | E‖ψ‖² | probe q_mean | probe q_max |
|---|---|---|---|---|---|---|
| bilinear exp5 sd0 | 50k→500k | 0.9898–0.9914 (flat) | 475→560 | 483→505 | −17.8→−19.6 | +3.2…+4.5 |
| bilinear ref sd000 | 500k, 1M | 0.9905, 0.9909 | 552, 567 | 514, 529 | −19.6, −20.0 | +3.6, +5.4 |
| mlp exp5 sd0 | 50k→500k | 0.968→0.941 (mild decline) | 550→427 | — | −27.6→−32.8 | +0.3…+0.6 |
| **sg s0 (stable, 0.692)** | 500k | **0.960** | 115 | **1083** | −34.2 | +0.2 |
| **sg s1 (diverged, 0.284)** | 500k | **0.981** | 94 | **2355** | −23.0 | **+778** |
| **sg s2 (diverged, 0.000)** | 500k | **0.978** | 121 | **3789** | **+8.2** | **+1177** |

bilinear_sg extra (ψ takes [s,g], so ψ(s,g)·ψ(s′,g) is a second shared-s channel):
psi_coad s0 = 0.9857, s1 = 0.9924, s2 = 0.9960 — **monotone in divergence severity**,
as is E‖ψ‖² (1083 < 2355 < 3789) and q_max (0.2 < 778 < 1177).

Deviation from spec: the sg runs saved only `params_500000.pkl`
(`save_interval=500000`), so the per-step sg coad trajectory does not exist; per the
pre-registered fallback we use the training-log q_mean/value_loss curves (train.csv)
for the time axis and the cross-seed comparison at the final step. Divergence
identification from train.csv: s1 value_loss 0.24→3.4, q_max→762; s2 value_loss
1.5, training q_mean crosses positive (+7.9 @500k); s0 stays value_loss ≤0.19,
q_max≈0. Note `training/critic/q_mean` in train.csv is the TD *target*
(r+γ·mask·V(s′)); the probe q_* are the critic's own Q(s,a,g), min over heads.

## Verdict vs pre-registered predictions (03-probes-spec §T2)

- **(a) bilinear_sg coad ≫ bilinear coad — REFUTED.** Stock bilinear sits at 0.990,
  *above* every sg seed (0.960–0.981). The normalized φ statistic is near-saturated
  for all archs (bootstrap-adjacent (s,a),(s′,a′) are one control step apart, so any
  smooth feature map scores ≈1); it has almost no dynamic range left for a runaway
  to show up in. The within-sg half of (a) holds directionally: divergent seeds
  0.981/0.978 > stable 0.960 (margin +0.02, 3 seeds — suggestive, not decisive).
- **(b) coad rises with q_mean→0 — NOT TESTABLE on φ as registered** (no sg
  intermediate ckpts), and at the final step φ-coad does not order the two divergent
  seeds by q_mean. However the ψ-side quantities do, monotonically (see below).
- **(c) bilinear + MLP flat over 0→500k — CONFIRMED.** bilinear 0.990±0.001 with
  slow benign ‖φ‖² growth (+18%); MLP drifts mildly *down* (0.968→0.941) with
  shrinking features. No co-adaptation runaway in either; 500k→1M ref points flat.

## Reading

The strict DR3-on-φ mechanism claim fails: the divergence signature is **not** in
the forward tower. It lives on the leg that received the state — ψ(s,g): divergent
seeds carry 2.2–3.5× the stable seed's E‖ψ(s,g)‖² (and ~5–7× stock bilinear's),
their ψ(s,g)·ψ(s′,g) alignment rises monotonically with divergence severity, and
Q=φ·ψ/√d inherits the blow-up (q_max +778/+1177 on a ≤0-bounded target). So
"removing the ψ(g) bottleneck opens a shared-s co-adaptation channel" survives in a
modified form — the channel is the state-conditioned ψ leg's norm/alignment runaway,
not growing φ(s,a)·φ(s′,a′) dot products — but the pre-registered headline statistic
refutes the literal DR3 reading. Per the spec's failure clause, the sg divergence
should be booked as "no output bound + no V-target + a state-conditioned leg with
unbounded norm growth", with the seed-monotone ψ statistics as the mechanism trace.
Caveat: final-step-only, 3 seeds, one divergence event per seed; a re-run of sg with
save_interval=50k would make (b) testable properly.
