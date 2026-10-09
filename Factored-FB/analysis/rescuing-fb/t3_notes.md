# T3 — off-support peak × domain: verdict

*2026-07-16. Probe: `scripts/rescuing_fb/t3_offpeak_multienv.py` →
`analysis/rescuing-fb/t3_offpeak_multienv.json`. Spec: `PAPER/rescuing-fb/03-probes-spec.md` §T3;
hypothesis: `01-synthesis-deep-audit.md` §2.4. 512 random dataset states × 5 eval goals per
domain; co-trained bilinear+DDPGBC seed-0 checkpoints @500k; reference cloud B_ref(s) =
dataset action at s + actions of the 64-NN dataset states (per-dim-whitened obs). No BC flow
involved (the Finding-1 instrument used μ-flow samples; agreement is expected to be
qualitative, not exact).*

## Numbers

| | cube (A=5) | antmaze (A=8) | scene (A=5) |
|---|---|---|---|
| peak median [Q(a_DDPG) − max_ref] | **+1.64** | **−0.80** | −1.05 |
| frac [Q(a_DDPG) > max over B_ref] | **0.639** | **0.003** | 0.014 |
| peak IQR | [−0.23, +6.47] | [−1.90, −0.39] | [−2.32, −0.48] |
| actor departure (whitened NN dist, LOO-normalized) | **3.34** (q75 8.35) | **0.82** (q75 0.95) | 0.80 (q75 1.01) |
| coverage: cond NN-dist /√A (uniform probes) | 0.504 | 0.625 | 0.474 |
| … uniform-null /√A (same set size, same A) | 0.291 | 0.401 | 0.291 |
| … **narrowness vs uniform** (cond/null; 1 = uniform-broad) | **1.73** | **1.56** | 1.63 |
| … narrowness, marginal set | 1.22 | 1.18 | 1.15 |
| eval success anchor (this ckpt) | 0.648 @900k | 0.784 @400k | 0.20 @400k |
| DDPG+BC α | 0.3 | 0.3 | **1.0** |

Per-goal cube peak fracs 0.57–0.70 (uniform across tasks — not a single-task artifact);
antmaze 0.000–0.008.

## Verdict vs pre-registered predictions

1. **Cube replication — PASS (sanity gate).** frac 0.639 / median +1.64 vs Finding-1's
   0.79 / +2.14. Qualitative replication with a *more generous* reference (a dataset-NN action
   cloud spans more of the play data than 256 μ-flow samples at one state, so max over B_ref is
   higher and both statistics shrink). Gate threshold frac ≥ 0.4 cleared; the antmaze read is
   trusted.
2. **Antmaze absent-or-benign — CONFIRMED, decisively.** Prediction was frac ≤ 0.25, median ≤ 0;
   measured frac 0.003, median −0.80. The co-trained DDPG+BC action is *below* the best
   behavior-reference action essentially always, and the actor barely leaves the behavior cloud
   (whitened departure 0.82 — inside the cloud's own LOO spread, vs cube 3.34, q75 8.35).
   The off-support Q-maximum that DDPG+BC climbs on cube does not exist (or is not reachable)
   on antmaze. This is the extraction boundary condition the domain flip needed:
   bilinear+DDPGBC wins antmaze (0.837) because there its Q-ascent stays on-support.
3. **Coverage antmaze ≫ cube — FAILED (the pre-registered *explanation* is unsupported).**
   Raw per-dim conditional NN-dist is actually *higher* on antmaze (0.625 vs 0.504), but that is
   an action-dimension artifact (NN distances at fixed set size grow with A; uniform-null 0.401
   vs 0.291). After null-normalization, cube's conditional action manifold is only **1.11×**
   narrower than antmaze's (1.73 vs 1.56 narrowness) — nowhere near the ≫ gap the coverage
   story requires, and both are modestly narrower than uniform. The dataset-only action-coverage
   statistic cannot explain a 0.639 → 0.003 peak flip.
4. **Scene (exploratory).** No off-support peak (frac 0.014, median −1.05), departure 0.80.
   Scene's bilinear+DDPGBC failure (0.20) is therefore NOT the cube peak mechanism — consistent
   with the synthesis treating scene as a separate unexplained domain. Caveat: the scene cell
   trained with α=1.0 (vs 0.3 cube/antmaze), so its low departure is partly BC-pinning by
   construction; the cube-vs-antmaze comparison is α-matched and clean.

## Reading

The **phenomenon** transfers exactly as hypothesized (peak: cube yes, antmaze no ⇒ the DDPG+BC
domain flip is explained by where the off-support peak exists), but the **mechanism candidate**
pre-registered for it (broad vs narrow dataset action coverage, measured in raw action space)
is refuted: cube and antmaze have similar conditional action-coverage after dimension
correction. Per the spec's own fallback, the peak's domain-specificity needs a different
account — the surviving candidate from 01-synthesis §2.3/Cost-A is value-geometry: the
low-rank/quasimetric residual must relocate *somewhere*, and where it lands depends on the
domain's Q-landscape (locomotion's smooth, densely-supported (s,a)→s′ geometry vs
manipulation's contact discontinuities), not on marginal action-space width. Candidate probes
if pursued: peak statistic vs distance-to-support in (s,a) jointly (not a alone), or the
effective-rank/residual-norm instruments from `ckpt_static_metrics.py` run per domain.

Consequence for the program: unifying extraction around selection/flow (01-synthesis §6.1)
remains the right move — the peak is real and domain-dependent, so gradient-ascent extraction
is unsafe in general; and "fix the dataset coverage" is NOT a viable alternative route.
