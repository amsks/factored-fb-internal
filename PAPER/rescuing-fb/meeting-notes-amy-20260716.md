# Meeting notes — why bilinear GCIQL works on cube, and what we built from it

*Prepared 2026-07-16. Plain-language walkthrough of the reasoning. All numbers are success
rates on OGBench cube-single (state-based), 3–13 seeds unless noted.*

---

## 1. The setup, in one paragraph

We want zero-shot RL for manipulation: train once on unlabeled robot-play data, then solve any
new task at test time without retraining. FB (forward-backward) is the standard method for
this. On cube manipulation it gets **0.42**. The best supervised goal-reaching baseline
(GCIQL) gets **0.72**. While dissecting the gap, we made one change to GCIQL — replace its
single value network with a **bilinear form**: one network embeds the (state, action), another
embeds the goal, and the value is the match (dot product) between the two. That one change
jumped GCIQL to **0.90**, well past the published state of the art (0.68) — but only when
paired with a **flow policy** (one that imitates dataset actions and nudges them toward higher
value). With a standard gradient-climbing policy the same critic got *worse* (0.51 vs 0.70).
Explaining that became the project.

## 2. How we reasoned through it, step by step

**Step 1 — Is FB's problem its architecture or its training signal?**
We built every hybrid between FB and GCIQL: same networks, swap only the training objective.
Result: the architecture doesn't matter (all FB's structural pieces work fine under GCIQL's
objective, 0.85–0.90); the **objective is the whole gap** (FB's objective on the identical
network: 0.10–0.34). Strongest version: start a network *at* the good solution and train it
with FB's objective only — it destroys the solution (0.93 → 0.05). FB's training signal
doesn't just fail to find the answer; it doesn't even want the answer.

**Step 2 — Is FB's "read the task from rewards" idea also broken?**
No. We ran FB's test-time recipe (estimate the task vector from reward-labeled data, hand it
to the policy) on top of the well-trained GCIQL critic: **0.955** — identical to being told
the goal exactly. FB's interface is fine; only its training signal is broken.

**Step 3 — Is the bilinear win about generalizing to new situations?**
We built an evaluation grid separating "new start states" from "new goals." Findings:
(a) FB fails even with everything in-distribution (0.43 while grasping the cube 100% of the
time — the failure is *steering after grasping*, not generalization); (b) the bilinear form's
edge is specifically on **new goals** — novel goals cost it nothing, while the standard
network loses 16–24 points on them.

**Step 4 — Is the bilinear critic just a more accurate value estimator?**
No — it's *less* accurate. We measured how well each critic's value predicts actual future
success from in-hand states: standard network **0.81**, bilinear **0.39 (below chance)**. The
winning critic is the worse predictor. So the win is not accuracy.

**Step 5 — Then what is it? Test the critic's *choices*, not its numbers.**
New test: freeze the trained critic, and at every step sample 32 candidate actions from a
plain imitation policy, executing whichever the critic scores highest. If the critic's value
numbers are meaningless but its *ranking* of actions is good, this plays well.
Result: bilinear-as-ranker plays at **0.91–0.96** — as well as its fully trained policy. The
standard network as ranker: **0.57–0.59**. FB as ranker: gets *worse* the more candidates you
give it (its scores actively mislead about actions).
**This is the core finding: the bilinear critic's product is a near-perfect ordering of
realistic actions; its value numbers are garbage, and that's fine, because ordering is all
the policy consumes.**

**Step 6 — Why does the gradient-climbing policy lose with this critic?**
The bilinear critic has a blind spot: it assigns inflated scores to certain *fabricated*
actions far from anything in the dataset (it rates them above every realistic action 79% of
the time). A gradient-climbing policy walks straight into that trap (we verified with freshly
trained policies — it's a property of the critic, not of co-training). The flow policy is
anchored to imitating dataset actions, so it physically never visits the trap. That explains
the sign flip: bilinear + climb = 0.51, bilinear + flow = 0.95.

**Step 7 — Today's three targeted tests sharpened the story.**
- *Is the standard network's action signal just noisy?* We averaged its scores over clouds of
  nearby goals to denoise. It got **worse**, not better — and a 2–4 cm goal shift already
  degrades it. Its action signal isn't noisy; it's **absent**. The standard network satisfied
  its training loss almost entirely with "how far is the goal," learning nearly nothing about
  "which action helps."
- *Why is the bilinear form stable?* We tried letting the goal network also see the state.
  2 of 3 runs blew up (values ran to +10¹³ on a quantity that must stay ≤ 0), and the blow-up
  lives precisely in the leg that was given the state. With the goal-only bottleneck, any
  score inflation must go through globally shared features and gets corrected by the rest of
  the data. **The bottleneck is what keeps training stable.**
- *Is the "trap" universal?* No — on maze navigation the same critic has no trap (fabricated
  actions beat real ones 0.3% of the time vs 63.9% on cube), which is why the gradient
  policy is fine there. (We also ruled out "cube's dataset actions are just narrower" — the
  coverage difference is only 1.1×, far too small.)

## 3. The clear answer

**Why does bilinear GCIQL + flow do so well on cube?** Because three things fit together:

1. **The training signal is sound.** GCIQL trains on "did you get closer to this goal" using
   only actions that actually appear in the dataset — a stable, well-posed learning problem.
   (FB's signal — match the long-run visitation statistics of the data — is provably unstable
   and spends almost all of its attention on where the *arm* goes rather than which *action*
   helps, which is why FB fails here.)

2. **The bilinear value form spends its capacity on the right thing.** Which-action-is-better
   is a tiny part of the value signal (<1% of its variance); a single big network fits the
   dominant "how far am I" part and learns essentially nothing about actions. The bilinear
   form can't do that: one shared action-side network must serve *every* goal at once, so the
   tiny action signal from millions of (state, goal) pairs accumulates in one place. The
   result is a near-perfect action ranking — bought by sacrificing accurate value numbers
   (which nothing downstream needs), free generalization to new goals (a new goal is just a
   new weighting of the same shared features — exactly what the cube benchmark tests), and
   training stability (the goal-side network can't inflate scores locally). Its one flaw is
   inflated scores on fabricated far-from-data actions.

3. **The flow policy consumes exactly the good part.** It only ever asks the critic to rank
   realistic actions — where the bilinear critic is near-perfect — and never visits the
   fabricated-action trap, which is the only place the critic is wrong. A gradient-climbing
   policy does the opposite, which is why it loses with this critic on cube.

And cube is the domain where all three trades pay off simultaneously: the benchmark charges
for new goals (free for this form), success hinges on step-by-step action choice after
grasping (ranking, not value accuracy), and the play data makes the imitation anchor a good
one. On mazes the actor-side conditions differ, which is why the recipe doesn't transfer
as-is — that transfer is exactly what we're engineering now.

## 4. What we built from this (status as of today)

- **RG-FB**: keep FB's zero-shot interface (task read from reward samples; no goal ever shown
  to the policy), replace its broken training signal with GCIQL's. First end-to-end run
  today: **zero-shot 0.825 ± 0.014** — roughly 2× FB, above the published *supervised*
  GCIQL (0.717) — with the inference step now verified essentially lossless (2.7-point gap
  to being told the goal exactly). A control isolating the remaining ~8 points vs the 0.90
  supervised anchor is training now.
- **Generality probe** (on the earlier large checkpoint): zero-shot works for *any* goal
  position within the data (0.67 on goal regions never in the eval set) — the claim is
  "zero-shot goal-reaching," not yet "zero-shot for arbitrary rewards" (two-goal composite
  tasks fail at 0.26; we know why — those task vectors were never seen in training — and the
  fix, training on mixed-goal tasks, is queued).
- **Rescuing FB's own objective**: a "centered" version of FB's loss that gives the
  action-relevant part its own learning signal is running now — as an *add-on* to the working
  recipe it trains healthily (0.71–0.78 mid-training, first time an FB-style loss hasn't hurt
  on cube); as a standalone it needed a scale fix (relaunched today with FB's own
  normalization anchors). This experiment decides whether we can say FB's objective is
  *fixable* or only *replaceable*.

## 5. What we'd claim in a paper, in one sentence each

1. FB-style zero-shot RL fails on manipulation because of its training objective, not its
   architecture or its task-inference interface — and the objective's failure is specific to
   manipulation (it works on mazes, where "where you go" *is* the task).
2. A goal-factored (bilinear) value head turns goal-conditioned TD learning into an action-
   ranking machine: state-of-the-art manipulation when extracted by imitation-anchored
   policies, with measured reasons for when it fails (gradient extraction, and domains where
   its off-data trap appears).
3. RG-FB: FB's zero-shot interface on this grounding matches supervised performance on cube
   zero-shot — with an honest, measured boundary of the claim (goal-family tasks yes,
   composite rewards not yet).
