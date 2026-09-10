# FourRooms tabular demo: FB squared loss vs Density cross entropy at a sticky bottleneck

2026-09-09. Code: `analysis/fourrooms_estimator/run.py` (numpy exact math, jax/optax for
the one learned fit). Data: `analysis/fourrooms_estimator/gap.csv` and `occupancy.csv`.
Figure: `PAPER/ICLR/figures/fig_fourrooms_estimator.{pdf,png}`.

## Setup

Standard 11x11 FourRooms grid, 104 free cells, four rooms joined by four single-cell
doorways cut into an interior wall. Four actions (up, down, left, right), deterministic,
bumping a wall means stay. The behavior policy that generates the dataset is uniform
random over the four actions at every state. The bottleneck is the doorway D between the
top-left and top-right rooms. A sticky rule with parameter k sits only at D: the action
that carries the agent into the next room (a_cross, "right") always works, no matter k.
Each of the other three actions (a_fail) keeps the agent at D with probability 1 minus
1/k, and otherwise behaves normally. k=1 means no stickiness. Two of the three a_fail
actions (up, down) are already blocked by a wall at D regardless of k, so stickiness only
changes the behavior of the one action that would otherwise carry real signal (moving
back into the room the agent came from). The reward used for the headline table is 1 on
every cell of the top-right room and 0 elsewhere. We also test a single target cell deep
in that room, and a second policy (see below).

## Exact quantities

Everything except the two fits is exact linear algebra, no sampling, no training. We
build the per-action transition matrices, average them under the behavior policy to get
one big transition matrix, and get its stationary distribution rho by power iteration.
rho is the state distribution the dataset would show if you just watched the random
policy run forever. The successor measure M(s,a,s') is the discounted sum over time of
the probability of being at s' if you start at s, take action a once, then follow the
policy: M = P(a) times the discounted resolvent (I minus gamma P)^-1. This is the exact
object both estimators are trying to reproduce with a low rank model. We checked every
row of M sums to 1/(1-gamma), which it should since it is a discounted sum of
probabilities.

FB fits a rank-d model to the ratio M(s,a,s') divided by rho(s'), a squared error loss
weighted by rho(s'). This is a linear regression problem and its exact rank-d optimum is
a truncated singular value decomposition, no training needed. The value it reports for a
reward is the usual FB dot product between a state-action feature and a reward-weighted
target feature. Density fits a rank-d model too, same feature shapes, but the model
outputs a full probability distribution over next states through a softmax, and it is
fit by minimizing cross entropy against the true next-state distribution, trained with
Adam. Both models see exactly the same information; the only difference is the loss they
are optimized under, squared error on a ratio versus cross entropy on a probability.

## Sanity checks

At the full rank d equal to the number of states, FB reproduces the true action gap
exactly (to floating point precision) and Density gets within 0.05 percent of it, which
is what a correctly implemented pair of estimators should do since neither one has a
capacity excuse left at full rank. At k=1, with no stickiness at all, the true gap
between crossing and failing at the door is still large and clearly nonzero, so the
bottleneck experiment starts from a state where the two actions are genuinely different,
not from a degenerate setup.

## Gap table (room reward, gamma = 0.98)

The normalized action gap compares the value difference between crossing the door and
failing to cross it. A value of 1 means the fit sees exactly the same gap as the true
measure. A value of 0 means the fit cannot tell the two actions apart at all. A negative
value means the fit has the ranking backwards.

FB (squared loss):

| k \ d |   4   |   8   |  16   |  32   |
|-------|-------|-------|-------|-------|
|   1   | 0.910 | 0.912 | 0.945 | 0.978 |
|   5   | 0.644 | 0.694 | 0.730 | 0.932 |
|  10   | 0.455 | 0.504 | 0.564 | 0.907 |
|  20   | 0.272 | 0.316 | 0.407 | 0.875 |
|  40   | 0.132 | 0.171 | 0.290 | 0.855 |

Density (cross entropy):

| k \ d |   4   |   8   |  16   |  32   |
|-------|-------|-------|-------|-------|
|   1   | 0.917 | 0.950 | 0.976 | 0.993 |
|   5   | 0.707 | 0.865 | 0.939 | 0.982 |
|  10   | 0.570 | 0.832 | 0.923 | 0.976 |
|  20   | 0.458 | 0.801 | 0.911 | 0.971 |
|  40   | 0.349 | 0.799 | 0.900 | 0.971 |

At gamma = 0.9, room reward, rank 4, FB's gap goes slightly negative at k = 10, 20 and 40
(values -0.007, -0.057, -0.032): the fit picks the wrong action as best. Density does not
flip sign anywhere under the room reward, at either discount. Under the sparser
single-cell target reward both estimators get noisier and each shows a handful of sign
flips at various k and d, FB more often than Density; that reward is a weaker signal for
this comparison and we report it only as a secondary check.

## Second experiment: directed occupancy versus the dataset marginal

We also ran the direct analogue of the usual "predicted occupancy collapses toward the
training distribution instead of the goal" plot. We built a goal-seeking policy that
takes the shortest path toward a cell deep in the target room, with 10 percent random
action noise, and asked what each fit predicts the next-state distribution looks like
three steps before the door, if the agent takes the correct forward action and then
keeps following the goal-seeking policy. We compared each fit's predicted row to the
true row and to rho, the dataset's own state distribution, using total variation
distance. The plain reading of "FB's row looks like the dataset distribution instead of
the true one" is not literally true here: FB's distance to the true row is small (1 to 5
percent) and its distance to rho stays large (about 80 percent) at every k and d we
tried, same as Density and same as the true row's own distance to rho. Total variation
distance to rho cannot separate a good fit from a bad one in this setup, because rho is
spread thinly over the whole grid and any reasonably concentrated successor row, true or
fitted, is far from it in this metric regardless of quality. What does hold is that FB's
distance to the true row is consistently 2 to 4 times larger than Density's, at every k
and d we tested, and FB's predicted row is visibly more spread out over the room in the
figure than Density's or the true row. The fraction of mass landing in the target room
overall is nearly identical for all three fits here, about 93 percent, so that particular
summary number does not discriminate at this starting state.

## Verdict, eight sentences

Yes, FB's gap collapses toward zero as k grows and recovers only slowly as rank d grows,
while Density's gap stays much closer to 1 across the same sweep. FB's gap does not
survive at small rank and large k: at rank 4 and k = 40 it falls to 0.13, and at the
lower discount it actually flips sign, so the fit prefers the wrong action outright.
Density degrades too as k grows, since a higher rank is genuinely needed to resolve a
sharper bottleneck, but it degrades much less: at rank 8 it still holds around 0.80 where
FB has fallen to 0.17. This is the predicted failure mode: a squared loss on a ratio
under-penalizes getting a rare, small-probability transition wrong, and the doorway's
"successful crossing" transition is exactly that kind of rare event once k is large, so
the squared loss estimator smooths right over it. The cross-entropy estimator does not
have this problem because it penalizes probability errors on a log scale, so a large
relative error on a small true probability still contributes a large loss. The second,
independent experiment is more mixed: the strong version of the claim, that FB's
predicted occupancy is literally closer to the dataset distribution than to the true one,
is not supported by total variation distance in our setup, because the dataset
distribution is too diffuse for that metric to be discriminating either way. The weaker
and still meaningful version holds throughout, FB's occupancy prediction is consistently
several times further from the truth than Density's, and visibly smoother across the
room, so the two experiments agree in direction even though only one of them supports the
sharpest phrasing of the original claim. We are reporting this mixed result plainly
rather than only keeping the version that lines up with the hoped-for story.

## Trained tabular agents: TD training versus the closed-form optimum

We also trained both estimators the ordinary reinforcement learning way instead of
fitting them in closed form, to check whether the closed-form story survives real
optimization. The setup: a fixed uniform random policy, no actor, no goal or task
conditioning, just two tables of numbers, F of shape (state times action, rank) and B
of shape (state, rank), for each method. A slowly moving copy of each table (updated a
tiny step toward the live table every step, Polyak averaging with a rate of 0.005)
supplies the bootstrap target, exactly like a target network in ordinary Q learning. For
each stickiness k and each of 5 seeds we sampled a fresh 100,000-step rollout of the
random policy under the sticky-door dynamics, then trained on batches of 256 sampled
transitions plus 256 sampled next states used as candidates, for 20,000 Adam steps at
learning rate 3e-4, rank 8, discount 0.98. FB's loss is the usual squared loss on the
same ratio target as the closed form, computed with a bootstrapped target instead of the
exact one and no orthonormality penalty. Density's loss is cross entropy against a
target that mixes the true immediate transition with a bootstrapped guess from the
target network, the same structure as the closed-form fit but learned online.

The result: trained Density stays close to its own closed-form best, but trained FB
does not come close to its own closed-form best at any k. At rank 8, gamma 0.98, room
reward, the closed-form gaps are FB 0.91 / 0.69 / 0.50 / 0.32 / 0.17 and Density 0.95 /
0.86 / 0.83 / 0.81 / 0.77 for k = 1, 5, 10, 20, 40. The trained versions, mean over 5
seeds with a 95 percent interval, are Density-FB 0.78 / 0.85 / 0.82 / 0.79 / 0.68 and FB
0.08 / 0.13 / 0.28 / 0.31 / 0.06 (full numbers and per-seed values in
`gap_trained.csv`). Density's trained number tracks its closed-form number within about
10 to 20 percent at every k. FB's trained number is far below its own closed-form
number everywhere, most dramatically at k = 1 where the closed-form fit is nearly
perfect (0.91) but the trained fit reaches only 0.08, about a twelfth of what is
achievable at that same rank. FB's trained gap is also far noisier across seeds than
Density's: at k = 20 the 95 percent interval runs from about -0.55 to 1.18, meaning some
seeds land far above the closed-form value and some land badly negative, while Density's
interval at the same k is a tight 0.68 to 0.90. So the closed-form collapse-with-k story
is not an optimization artifact, since Density recovers a good fraction of its target by
gradient descent while FB does not, but it is also not the whole story: FB's practical,
trained behavior is even worse than the closed-form squared-loss optimum predicts, and
noisier, which points at the squared loss itself being harder to optimize with a
bootstrapped target, on top of being a worse fit once optimized. If trained FB had
matched its SVD optimum we would say the collapse is purely a loss-shape property; since
it falls well short of that optimum instead, we report plainly that TD training adds a
second, separate handicap for FB beyond the one the closed-form comparison already shows.

## Action-value heatmap at the bottleneck

We also ran a version of the door experiment with a 2D continuous action instead of 4
discrete actions, to look at the actor's action-selection problem directly rather than
only at a scalar gap. The action is a point in [-1, 1] squared, discretized to a 15 by
15 grid (225 actions). Away from the door, an action moves the agent to whichever
neighboring cell is closest to its direction if the action's length is at least 0.5,
otherwise the agent stays. At the door, only actions with a_x at least 0.5 and a_y
within 0.25 of zero (the "crossing cone", a narrow wedge of 12 of the 225 actions
pointing through the doorway) succeed for certain; every other action is a failed
crossing and is subject to the same sticky rule as before. Reward, discount, and both
estimators are exactly as in the rest of this note, with rank 8 as the default and rank
16 added in the CSV, and a rank sweep from 2 to 32 at k = 20. Figures:
`PAPER/ICLR/figures/fourrooms/action_value_door.{pdf,png}` (each panel colored by its
own min and max, so shape is comparable but not magnitude),
`action_value_door_shared.{pdf,png}` (the same panels, but now each column, one k,
shares one color scale taken from the true panel, so a fading cone is visible directly
instead of being hidden by rescaling), and `action_value_door_d.{pdf,png}` (rank sweep
at k = 20). Numbers in `action_value_door.csv`, raw Q surfaces in
`action_value_door.npz`.

The normalized gap at rank 8 falls with k for FB (0.90 / 0.71 / 0.57 / 0.44 / 0.41 for k
= 1, 5, 10, 20, 40) while Density stays high (0.94 / 0.95 / 0.96 / 0.96 / 0.97), the same
pattern as the 4-action case. Despite that shrinking margin, FB's best action (the white
dot) stays inside the crossing cone at every k from 5 to 40, and so does Density's; the
shared-color-scale figure shows why this is still worth checking rather than assuming,
because FB's cone visibly fades into the surrounding color as k grows even though it
stays the brightest cell. At k = 1 both fits' best action, and the true best action,
fall just outside the strict cone boundary; this is a tie, not a failure, since with no
stickiness many actions near the cone give literally identical value once the agent is
already at the door, and argmax just picks one member of a tied set. The rank sweep at
k = 20 shows the same recovery-with-rank pattern as the rest of this note: FB's
normalized gap goes from -0.03 at rank 2 (a genuine sign flip) up to 0.94 at rank 32,
while Density stays between 0.83 and 0.99 across the same range. Rank 2 is also the one
case in this experiment where FB's best action leaves the crossing cone entirely (true
and Density's best action stay inside the cone at every rank tested). So the same story
holds under a continuous, higher-dimensional action space: FB's ranking is fragile
exactly where the environment is fragile, and it needs more rank than Density to recover
it.
