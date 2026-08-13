# Offline RL for Autoscaling — how we got to the current result

**Status report, 2026-07-23.** Covers the decisions of roughly the last two weeks (the
evaluation audit through the current results and in-flight runs). Figures are in
`figures/` and regenerate from logged data with `figures/make_figures.py`.

## The problem and the pitch

Autoscalers decide, every few minutes, how much compute a service gets. Production
systems use simple hand-tuned rules; reinforcement learning (RL) could do better, but
letting an RL agent experiment on a live service is not acceptable. Our premise: the
deployed rule controller already produces, as a by-product, exactly the data offline RL
trains on — logs of situations, scaling actions, and outcomes.

**Central result:** a policy trained purely from such logs — no interaction with the
system — beats the logging controller on 80% of held-out test windows, beats direct
imitation of the logs, and matches a hand-tuned reference controller that took explicit
domain insight to build. **Central pitch:** this works under three conditions, each of
which we isolated experimentally, and each of which an operator can check *before*
training. When one is missing, we can show exactly how it fails.

1. The logs must contain *alternatives* (different actions in comparable situations).
2. The objective must have *headroom* (something to improve — here, cost).
3. The training must be *sound* (two specific failure modes found and gated).

---

## Step 1 — Make the comparison fair (evaluation audit)

**Problem.** An audit of our evaluation pipeline found that learned policies and rule
controllers were not scored under identical conditions: they were warmed up
differently, scored by different formulas, failed windows were silently dropped, and
one agent family's actions were being mistranslated on their way to the simulator. Any
of these alone was enough to reorder methods.

**What we did.** Fixed all of it, then froze the evaluation: one test suite of 266
three-hour windows from held-out real traffic traces (spanning quiet, borderline, and
overloaded regimes), every policy through the identical code path, and all comparisons
made per-window (policy A vs. policy B on the *same* window, then aggregated) with
standard significance tests.

**Intuition.** Before arguing about algorithms, make sure a difference in score can
only come from the policy. All previously reported comparisons were discarded rather
than patched.

## Step 2 — Make training trustworthy (the value-estimate gate)

**Problem.** The learner's internal value estimates ("how good is this action here")
have a hard arithmetic ceiling implied by the reward definition. Checkpoints trained
before 2026-07-21 exceeded that ceiling by up to seven orders of magnitude — the
training had quietly diverged, and diverged runs happened to look like plausible but
static policies.

**What we did.** Added standard stabilizers (capping the training targets at the
ceiling, gradient clipping, conservative twin estimates) and made the ceiling an
automatic acceptance gate: a run whose estimates violate the bound is discarded, not
reported. All older checkpoints were invalidated.

**Intuition.** A model whose internal quantities violate arithmetic cannot support any
claim, favorable or not. The gate makes this a property checked by the pipeline, not by
vigilance.

## Step 3 — Fix the data recorder

**Problem.** The episode generator did not fully reset the controller between recorded
episodes, so the logged "production behavior" was more over-provisioned and less
responsive than the controller actually is.

**What we did.** Fixed the reset, regenerated every dataset (the `rate_v3` generation),
and attached an automatic datasheet (size, action usage, capacity-vs-load statistics)
to every dataset so this class of problem is visible at a glance.

## Step 4 — Why the controller's own logs are not enough

**Problem.** With fair evaluation and sound training, RL trained on the controller's
own logs still did not beat imitation. This turned out to be a property of the data,
not a bug.

**Intuition.** The production controller is a deterministic rule: in a given situation
it always takes one action. Its logs therefore never show what *would have happened*
under a different action. A learner can only rank actions it has seen outcomes for; with
no alternatives in the data, the best it can do is copy the controller. (Control theory
and offline-RL theory both state versions of this formally.)

**Evidence.** Figure 2: the own-logs dataset uses 82 of 252 possible scaling actions,
and at any given traffic level offers ~10 effective choices; allocated capacity is
statistically unrelated to load. We also ran a controlled experiment holding the
algorithm and every training setting fixed while swapping only the dataset: on data
with decoupled capacity, the same learner produces a policy whose capacity tracks load
almost perfectly; on the controller's own logs it stays effectively static.

![dataset support](figures/2_dataset_support.png)

## Step 5 — The fix: variation that operators already have

**Problem.** The textbook cure — random exploration — is exactly what production
forbids.

**What we did.** Generated training logs using only perturbations that production
fleets already exhibit: (a) the controller run under different setpoint tunings
("operators retune"), (b) switching among several standard controllers
("deployments"), and (c) small, bounded, correlated action perturbations of the kind
hysteresis and estimation error already cause. We also built a fully synthetic
high-variation dataset as a comparison point. Each dataset must pass two measurable
acceptance gates *before* any training: enough per-situation action variety in every
traffic regime, and no single behavior so dominant that copying it is already optimal.

**Intuition.** The identifiability requirement says variation must exist; it does not
say the variation must be random. Realistic variation turned out to be not only
acceptable but more sample-efficient than the synthetic kind (higher final score from
~40% less data).

## Step 6 — The result

All numbers: cost-aware objective, 266 held-out windows, identical harness, single
training seed (pilot scale; multi-seed rerun in flight).

| Policy | Mean score | Better than ScaleX on |
|---|---|---|
| **IQL on realistically-excited logs** | **7.22** | 80% of windows |
| IQL on synthetic-excitation logs | 6.71 | 73% |
| Hand-tuned capacity oracle | 6.37 | 73% |
| Step-scaling rule | 5.40 | 66% |
| Target-tracking rule | 5.36 | 78% |
| Behavior cloning (imitation) | 4.02 | 64% |
| ScaleX (logging controller) | 3.00 | — |

![aggregate](figures/3_aggregate_performance.png)

Per-window, against each baseline class (the honest comparison — same window, both
policies):

![histograms](figures/4_delta_histograms.png)

- vs. the logging controller: better on 80% of windows (statistically decisive).
- vs. imitation on the same data: better on 78% — the gain is from valuing outcomes,
  not just from better data.
- vs. the hand-tuned oracle: a statistical tie (41% win rate). The claim is not that
  RL beats hand-tuning; it is that RL *recovered* hand-tuned performance from logs
  alone, without the domain insight the oracle's designer needed.

**The policies genuinely react.** Figure 1 shows the learned policy on three individual
test windows: scaling up for a spike and back down after it, scaling up and holding
under sustained heavy demand, and holding a small allocation under quiet traffic —
while the production rule parks at roughly full capacity in all three. 112 of 260
windows show the learned policy actively moving its allocation; on the remainder it
correctly holds a lean constant because demand is flat.

![reactivity](figures/1_reactivity_examples.png)

**Full metrics for the same three windows.** For each example window, the complete
picture on one page — allocation with demand, rejected requests, latency against the
0.8 s target, and CPU utilization — with per-policy average allocations in the legend.
Colors are identical across this report and the paper (blue = IQL, orange = imitation,
black = ScaleX, pink dashed = hand-tuned oracle), and request patterns are always drawn
as steel-blue bar histograms — the same convention as the evaluation harness's
per-window figures, frozen project-wide.

![full example: spike](figures/6_full_example_0001_1034_v1_mu10_cm2_w0.png)

On the spike window, note the oracle's failure mode: it under-provisions through the
second half and pays 20–40% rejected requests and multi-second latencies, while the
learned policy serves the same demand cleanly at a third of ScaleX's capacity.

![full example: sustained](figures/6_full_example_0013_915_v0_mu10_cm4_w0.png)

![full example: quiet](figures/6_full_example_0060_1439_v0_mu0p1_cm1_w0.png)

On the quiet window all four policies deliver identical service (zero rejections,
latency 40× below target); they differ only in what it costs — 544 average OCU for the
production rule vs. 12 for the learned policy.

**Mechanism.** On a typical quiet window the learned policy holds ~12 capacity units
where the production rule holds ~544, at zero rejected requests and comparable latency
for both — the win is right-sizing, not a different service trade-off. Two honest
caveats: on borderline-load windows the learned policy accepts somewhat higher latency
in exchange for the large capacity reduction (the objective prices this trade
explicitly, and we report the latency series rather than hiding it in the score); and
the hand-tuned oracle occasionally cuts capacity too deep and takes a service hit that
the learned policy avoids.

**Where RL pays — and where it doesn't.** The win exists on the cost-aware objective,
which has headroom. On pure service-quality objectives (latency-only, rejection-only)
the over-provisioned rule is already essentially optimal, everyone ties near the
ceiling, and imitation is the right tool. This is condition 2, confirmed empirically.

![where RL pays](figures/5_where_rl_pays.png)

## Results update (landed 2026-07-23, after the sections above were written)

Four run sets came back; three change the picture, one flagged its own failure.

1. **The ladder is now complete and clean.** Retraining the own-logs and retunes
   models with the stabilized recipe replaced the two provisional rungs. The ladder
   now climbs monotonically: own logs 4.12, retunes 5.61, synthetic excitation 6.71,
   realistic excitation 7.22 — and on the first two rungs the RL policy and imitation
   are level (differences of +0.07 and +0.21). This sharpens the story: adding data
   variety lifts everyone; only the full excited pool lets value-based selection pull
   ahead of imitation. The paper's headline numbers are unchanged.
2. **Objective re-targeting (one model, any objective) is not yet free.** The
   first-cut generalist methods match everyone on the service-quality objectives but
   give up about two score units to the specialist on the cost objective (best
   generalist 5.18 vs. specialist 7.22). One method failed its own built-in safety
   check — it scored below the guaranteed floor its theory promises — which means an
   estimation bug, not a valid result; it is under investigation. This is exactly why
   the check was pre-registered.
3. **Combined cost+service objectives:** the policy trained on the excited pool leads
   the cost+latency column (12.42 vs. best rule 11.57); on cost+rejection the
   CPU-tracking rule still leads (14.51 vs. 13.94). Mixed outcome, worth one line in
   the meeting.
4. **The history-conditioned (12-frame) evaluations all failed to score** — every
   window, including the control arms, which points to the evaluation harness not
   applying the frame-stacking configuration rather than to the policies. The
   memoryless control cells did score and behave exactly as predicted: removing the
   forecasting controller's data changes the memoryless policy by a statistically
   insignificant −0.16. The 12-frame re-evaluation needs one harness fix and a rerun.

## State of the project (2026-07-23)

**Done.** Fair frozen evaluation protocol; stabilized training with automatic
acceptance gates; regenerated datasets with datasheets and pre-training gates; the
pilot-scale headline result above; the paper rewritten around the
success-under-conditions narrative with per-window mechanism figures.

**In flight.**
- Multi-seed rerun of the full ladder (removes the single-seed caveat), plus the
  stabilized re-scores of the two remaining ladder rungs.
- History-conditioned agents (the policy sees the last hour, not one snapshot):
  a controlled 2×2 experiment testing whether *anticipatory* scaling becomes learnable
  when — and only when — the data also contains a forecasting controller's actions.
- Two combined cost+service objectives training across all datasets (16 runs).
- A new algorithm variant tailored to this setting ("Roster-GPI"): one training run
  that serves *any* objective weighting, built from per-controller value estimates with
  a formal guarantee of never being worse than the best individual controller in the
  data at each state. Implemented and unit-tested; training next.

**Next decisions.** Full-scale data generation (all training traces), then the
multi-seed paper numbers; the history and Roster-GPI results decide whether the paper
gains a fourth condition (policy memory matched to data content) and a second
algorithm.

---

*Everything here regenerates from logged artifacts: figures from
`figures/make_figures.py`; the paper-grade versions of these plots live in
`PAPER/ICLR-New/paper/figures/` with statistics in `numbers.json`.*

