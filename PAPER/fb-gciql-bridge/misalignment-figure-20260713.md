# The occupancy–task misalignment, measured (2026-07-13)

**Question:** show, model-free and "without a doubt," WHY the successor-measure /
SF objectives fail on cube while scalar reward-TD works — and what that implies
for correcting FB's zero-shot control given the fixed dataset, with grasp as the
known failure mode.

**Figure:** `analysis/misalignment/occupancy_task_misalignment.png`
**Script:** `scripts/plot_occupancy_task_misalignment.py` (reruns from the raw
OGBench npz; no learned model, no hyperparameters anywhere in the loop).
**Data:** `~/.ogbench/data/cube-single-play-v0.npz` (1000 eps × 1000 steps),
`antmaze-medium-navigate-v0.npz`. Cube obs layout used: `obs[19:22]` cube xyz
(units: 1 = 10 cm — positions are decimeter-scaled), `obs[18]` binary grasp/
contact flag (semantics inferred empirically, see caveats), `obs[12:15]` effector.

## The revision this measurement forced

The earlier narrative — "occupancy is arm-dominated; the interaction slice is
low-mass in ρ" — is **wrong at the level of raw mass** and must not be used:

- the cube is **held on 45.5% of dataset steps** (P(cube moves >5mm/step | flag)
  = 0.905 vs 0.004 — the flag is a razor gate, and it is an *observed* dimension);
- naive per-step normalized motion share and marginal variance share do NOT
  separate the domains in the needed direction (antmaze xy per-step motion share
  is 0.002; cube dims carry 26% of marginal variance).

What survives measurement is sharper. The misalignment is **temporal**, plus one
**unsupported boundary**.

## Panel A+B — the misalignment is temporal

Conditional K-step displacement of the task variable, gated on the present mode
(cube: grasp flag; antmaze: best observable split found, speed median):

| K (steps) | cube gate ratio E[disp\|held]/E[disp\|free] | antmaze best-gate ratio |
|---|---|---|
| 1–2 | **~130–150×** | ~2.5× |
| 5 | 55× | 2.1× |
| 10 | 7.8× | 1.4× |
| 20 | 2.8× | 1.4× |
| 50–200 | **1.0×** | 1.3× |

(P-version at K=1: P(cube displaced >2cm | held)=0.29 vs 0.000; by K=50 both
0.92+ — the scripted play policy re-grasps, so long-horizon occupancy forgets
the gate.)

Overlay the discount kernel: a γ=0.99 successor measure puts **82% of its
discounted mass at K>20, where the gate ratio is 1.0**; action selection is
decided at K≤5, where the future is 55–150× mode-determined. The measure is not
failing to fit its object — **its object genuinely does not need the grasp bit
under the data policy**. Control does. On antmaze the short- and long-horizon
structure are the same smooth diffusion at every horizon (no observable exceeds
~2×), which is exactly why measure-only scores 0.79 vs the 0.837 anchor there
and 0.10 vs 0.90 on cube (mdose domain flip).

Subtlety that strengthens the claim: the K=50 ratio of 1.0 holds under the DATA
policy (which always re-grasps). For a policy that fails to re-grasp, the
long-horizon future IS gated — but the data cannot exhibit that counterfactual,
and a max-free density TD has no mechanism to construct it. The expectile TD
does: hindsight relabeling ties every backup to the trajectory's own realized
future through its own contact events.

## Panel C — the failure boundary is unsupported

Of **12,144** grasp releases in 1M transitions, **12,127 are deliberate places**
(cube ends ≤3.5cm above table) and **17 are mid-air drops** (ρ-mass 1.7e-5;
P(drop | held) = 3.7e-5 per step). The stable-vs-slipping distinction —
precisely the eval failure mode — is a value cliff supported by 17 transitions.
No reweighting of any objective can amplify what is essentially absent from the
data. Corollary: staying inside behavioral support is mandatory on cube, which
is why the BC-anchored flow actor is load-bearing there (bilinear+DDPGBC 0.51 <
MLP+DDPGBC 0.70 < bilinear+flow 0.90) and why nothing should be asked to rank
grasp-stability counterfactuals.

## Proposed 4th panel (model-side, connects data → learned scores)

Panels A–C say the incentive is absent; Panel D should say the learned models
behaved as the incentive predicts. Either: (i) existing interaction-variable
probes (FB AUC 0.75 vs PSM 0.94 vs scalar-TD critic 0.91–0.96, moment-z 0.955);
or (ii) bridge-state score histograms — dataset states labeled by whether the
grasp survives the next k steps; GCIQL's Q separates the labels, F·z does not.

## Corrections for FB zero-shot (ranked by evidence, each with its derivation)

1. **Proven: swap the grounding, keep the whole FB API (~0.95 available).**
   Bilinear φᵀψ trained by hindsight scalar reward-TD supports FB's zero-shot
   protocol as-is: moment z = E_ρ[r·ψ(s′)] on relabelled offline transitions →
   0.955 (E1a); full FB skeleton survives on this signal (B4 = 87.1). Missing
   piece = **inference-consistent actor** (G4's exact-z→inferred-z collapse
   93→7): condition the actor during training on z computed by the moment rule
   from relabelled minibatch rewards + ‖B(g)‖ rescale, so the train-time and
   test-time conditioning distributions match. Untested; highest-value single
   run on the board.
   *Derivation:* the bridge ladder proved the binding factor is the grounding
   (D1), and E1a proved the inference API is intact on a reward-TD critic — so
   the only broken link left in the zero-shot chain is the actor's conditioning
   distribution, which G4 isolated as an independent success-level bottleneck.
2. **In-family (keep the measure), each targeted by a specific panel:**
   (a) **short-horizon measure head** — a second measure at γ≈0.8 (mean
   lookahead 5 steps, inside the gate window), actor ranks on a blend.
   *Derivation:* Panel B — the gate is only visible at K≤10, and a γ=0.99
   measure puts 82% of its mass at K>20 where the gate ratio is 1.0; a γ=0.8
   kernel puts its mass exactly where the gate lives, so the short head is
   *forced* to resolve the grasp bit to fit its own objective. Distinct from
   the absorbing-mask null (cell C changed the object but kept γ=0.99, i.e.,
   kept looking at the wrong horizons).
   (b) **mode-indexed forward map** — condition F on the observed grasp bit
   obs[18] (or a two-headed F switched by it).
   *Derivation:* Panel A — the gate variable is not hidden, it is a literal
   observation dimension, so no inference machinery is needed; the fbz
   precedent (task-indexing F cut the measure's own TD residual 6347→266, 24×)
   shows indexing is how this architecture resolves conditional structure.
   Caveat carried forward: fbz also showed fitting well is insufficient for
   control (25.5), so (b) only makes sense combined with (a) or with a
   grounding fix.
   (c) **in-sample max** (ISFB-coord): the only in-family lift ever observed
   (16.5→33). *Derivation:* the data-policy caveat in Panel A/B — the K=50
   gate ratio of 1.0 holds only because the scripted policy re-grasps; a
   max-free TD inherits that policy's indifference to the gate, while an
   improvement operator is the only in-family mechanism that can prefer
   futures the data policy merely happened to realize.
   Honest prior for the (a)+(b)+(c) stack: it lifts but plateaus in FB's own
   band (~35–50) unless the reward enters ψ's training — every result in the
   program says the readout must be shaped by the task signal.
3. **Inference-side fixes, no retraining (cheap, partially proven):** compute z
   from the *dataset* moment (never on-policy rollouts — the on-policy version
   was the 93→7 artifact), on arriving states with the correct index
   convention, rescaled to ‖B(g)‖ rather than √d (the actor consumes raw-B
   scale; whitening is refuted, E[BBᵀ] has κ≈1e14). For transport goals the
   reward-bearing moment samples are placements, i.e., held-mode states, so the
   moment already points at the right mode — no reweighting needed.
4. **What Panel C forbids (a constraint on all of the above):** with 17 drop
   transitions in 1M, nothing may be asked to *rank grasp-stability
   counterfactuals from data* — that boundary is unlearnable at this dataset.
   Consequence: corrections must aim the value at selecting among in-support
   grasp-maintaining actions, and the BC anchor is load-bearing (it is why the
   flow actor beats DDPG+BC on cube: bilinear+DDPGBC 0.51 < MLP+DDPGBC 0.70 <
   bilinear+flow 0.90). Any "fix" that loosens the BC constraint on cube is
   predicted to fail regardless of the critic.
5. **Speculative (flagged as such): two-timescale factorization.** Accept the
   division of labor the data implies: Q_z(s,a) = F_long(s,z)·z (task
   progress / transport, measure-trainable — this is the part the occupancy
   objective CAN do) + F_short(s,a,z)·z (grasp maintenance, short-γ or
   TD-grounded). *Derivation:* Panel B says the two objectives are each
   correct at different horizons; rather than forcing one object to serve
   both, split the value by timescale. No in-house evidence yet; this is the
   research bet, not a recommendation.
6. **Refuted — do not re-walk:** task-norm reweighting (N1 0.20 / N2 0.067),
   z-mix (C5), zHER, decouple-actor-z, whitening, per-sample estimator patch,
   absorbing mask, and warm-starting the measure from converged good geometry
   (anneal iqlfade: success collapses in lockstep with iql_w → the measure's
   own fixed point is bad on cube).

## Appendix — reasoning trace (how this note was produced)

Recorded so the analysis can be audited and the method reused. Four stages.

### A.1 How the prompt was parsed

The request was two questions: *"how can we, without a doubt, show this
[occupancy–task misalignment] visually"* and *"how can we correct FB to do
zero-shot control better on the eval tasks, given this dataset restriction —
the failure mode is grasp."* Each phrase was treated as a constraint:

- **"without a doubt"** ⇒ the plot must be immune to the standard rebuttals a
  reviewer would raise: mistuned hyperparameters, a bad FB implementation, an
  unlucky architecture. The only class of plot immune to all three is one
  computed from the **raw dataset + the task definition alone, with no learned
  model anywhere in the loop**. That single inference fixed the design space
  before any data was touched: model-side evidence (probe AUCs, learned-score
  histograms) could only be a *supplementary* panel, never the headline.
- **"given this dataset restriction"** ⇒ corrections requiring new or
  different data are out of bounds; every proposal must work on
  cube-single-play as it exists.
- **"the failure mode is grasp"** ⇒ both the figure and the fixes must connect
  mechanically to grasp acquisition/maintenance, not to generic performance.
- **Zero-shot** (from "correct FB to do zero-shot control") ⇒ a hidden
  negative constraint: any fix that abandons test-time task inference is not a
  corrected FB but a different algorithm; fixes must preserve the API
  z = E[r·B] with no fine-tuning.

### A.2 Measurement before design (and what it overturned)

The prevailing story going in — from the bridge tex and prior session notes —
was "occupancy is arm-dominated; the interaction slice is low-mass in ρ." Rule
applied: **before building a figure on a claim, measure the one number the
figure hinges on.** If the dataset had, say, 40% object-motion transitions,
the "low-mass" figure would be indefensible — so that number was computed
first, before any figure code existed.

The first measurements *broke the story*: per-step normalized motion share and
marginal variance share do not separate cube from antmaze in the needed
direction (antmaze xy carries only 0.002 of per-step motion; cube dims carry
26% of marginal variance), and the cube turns out to be held on 45.5% of
steps. Two intermediate results were also physically impossible — a "2 m cube
displacement over 50 steps" on a tabletop, and a suspiciously exact
P = 0.500 — which triggered inspection of the raw array (per-dimension
ranges) instead of more summary statistics. That inspection decoded the real
layout: positions are decimeter-scaled (the 10× unit error behind the
impossible numbers), obs[18] is a binary flag, and the per-step cube
displacement distribution is bimodal (q50 = 0.01 mm vs q70 = 1.3 cm — parked
vs carried), which is what makes the 5 mm motion threshold insensitive.

### A.3 The inference chain to the temporal story

With correct units, three measurements assembled the revised mechanism:

1. **The gate is razor-sharp at one step** — P(cube moves | flag) = 0.905 vs
   0.004 — so the interaction mode fully controls the task variable's
   *immediate* future, and the mode is observable.
2. **The gate washes out with horizon** — the conditional K-step displacement
   ratio decays ~150× → 55× → 7.8× → 1.0 across K = 1→50 — because the
   scripted play policy re-grasps, so long-horizon occupancy forgets the gate.
   The same statistic on antmaze never exceeds ~2× at any K under any split
   tried: there, short- and long-horizon structure are the same object.
3. **Putting 1+2 against the discount kernel**: a γ=0.99 successor measure
   places 82% of its mass at K>20 (gate ratio 1.0), while action selection is
   decided at K≤5 (ratio 55–150×). Conclusion: the measure is not failing to
   fit its object — *its object genuinely does not require the grasp bit under
   the data policy*. Control does. This upgraded the story from spatial
   ("interaction is rare") to **temporal** ("interaction matters only at
   horizons the objective barely weighs"), and it predicts the already-known
   domain flip (measure-only: antmaze 0.79 vs cube 0.10) rather than merely
   being consistent with it.
4. **The drop count** (17 mid-air drops vs 12,127 deliberate places in 1M)
   was measured because "the failure mode is grasp" demanded a data-side
   account of grasp *failure* specifically — and it supplied the second,
   independent misalignment: the boundary the eval fails on is essentially
   absent from the data.

### A.4 Why the figure took this form, and where the corrections came from

Panel choices follow the inference chain one-to-one: Panel A shows the gate
exists and is binary on cube / absent on antmaze (the two conditional
displacement curve-pairs); Panel B is the argument itself — gate
informativeness vs horizon with the discount mass and the action-selection
window as shaded regions (shading, not a second y-axis); Panel C shows the
unsupported boundary on a log-mass axis with the drop bar flagged as the eval
failure mode. All three are dataset-only by the "without a doubt" constraint;
the model-side evidence (probe AUCs 0.75/0.94/0.96) is deliberately demoted to
a proposed fourth panel whose role is to show the learned models behaved as
the incentive predicts — connecting, not carrying, the argument.

Every correction in the ranked list above was then derived by pairing one
measured fact with one prior result from the program (the *Derivation* lines):
the grounding swap comes from D1 + E1a closing every link except the actor's
conditioning (G4); the short-γ head comes directly from Panel B's mass
argument; the mode-indexed F from Panel A's observation that the gate is a
literal obs dimension plus the fbz fitting precedent; the in-sample max from
the data-policy caveat (a max-free TD inherits the re-grasping policy's
indifference); the BC-anchor constraint from Panel C's 17 transitions; and the
refuted list from the program's own negative results, included so no fix is
re-proposed. The method throughout: no correction is proposed unless it is
targeted at a *measured* mechanism and priced against an *existing* number.

## Caveats before the paper cites this

- **obs[18] semantics are inferred empirically** (binary; P(carried|flag)=0.905
  vs 0.004). Verify against the OGBench manipspace env source (likely gripper-
  closed or contact indicator) before naming it "grasp flag" in print.
- Thresholds used: cube motion >5mm/step (per-step displacement is bimodal:
  q50=0.01mm vs q70=1.3cm, so the split is threshold-insensitive); K-step
  displacement >2cm (cube) / >0.5m (antmaze); drop = release with cube z >3.5cm.
- Antmaze "best gate" = speed median split; other candidate splits (position,
  joint config) were not exhaustively searched — the claim is "no observable we
  found," strengthened if a search over splits still caps near ~2×.
- Related but distinct from `task-distribution-analysis` (representation-side
  probes); this note is purely data-side.
