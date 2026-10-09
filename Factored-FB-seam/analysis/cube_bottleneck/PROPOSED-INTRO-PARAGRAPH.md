# Proposed replacement for `PAPER/ICLR/content/introduction.tex:10-27`

**NOT APPLIED.** The spec (deliverable 3) asks for a proposed paragraph conditional on the
reading, left for you to accept or reject. `introduction.tex` is untouched.

## Why a replacement is owed

The current paragraph makes three claims. The probe tests all three on cube, and two of them
do not survive.

| claim in the current text | what the probe measured |
|---|---|
| "this loss of precision arises at *control bottlenecks*" | **Not supported.** Removing the bottleneck at test time — starting the episode with the cube already secured, from a recorded in-distribution state — leaves plain FB essentially where it was: Δ = +0.020 / +0.100 / +0.032 across seeds, 0.21 → 0.26, against Density-FB's 0.99. (R-HELD2, fired on 3/3 seeds.) |
| FB "must simultaneously represent... potentially allocating insufficient capacity" to the post-bottleneck region | **No support, and one direct negative.** FB's predicted total mass is 87–105 against the true 1/(1−γ) = 100 and moves less than ±8% between pre-contact, at-contact and held states (row-mass ratio 0.94–0.98; R-MEAS4 needed > 3 and does not fire). Whatever FB gets wrong, it is not inflating or starving mass by phase. |
| \name{} "makes the consequences of precise bottleneck actions easier to distinguish" | **Contradicted in the measured direction.** At grasp-adjacent states, plain FB separates securing from non-securing actions *better* than Density-FB: within-state AUC 0.55/0.78/0.75 on the measure and 0.95/0.95/0.91 on the Q readout, against Density-FB's 0.44/0.55/0.64 and 0.55/0.50/0.50. The best bottleneck discriminator in the whole set is `fb` at `ortho_coef` 1.0 (AUC 0.91), which is also the **worst controller** (0.096). |

The last row is the load-bearing one: bottleneck discrimination and control performance are
*anti-correlated* across these four arms, so a mechanism that explains the win by better
bottleneck discrimination is explaining it with a quantity that runs the wrong way.

## What the evidence does support

FB's failures from a held start are failures of *retention*: 49–56% of them end with the cube
back on the table, and FB holds the cube for 70–75% of the opening 50 steps against
Density-FB's 81–86%. The difference between the estimators shows up **after** acquisition, in
maintaining and transporting a grasp already established — which is where the 2026-05 phase
diagnostic put it, and consistent with S2 having helped none of PSM, RLDP or GCIQL.

## Proposed text

> We hypothesize that this loss of precision arises not at the moment of acquisition but in
> \emph{sustaining} an interaction once it has been established: the regime in which an
> agent must keep a grasp intact while transporting the object to a commanded configuration.
> Unlike classical TD learning, which compresses an action's consequences into a scalar
> value for a given reward, SM methods must represent the full distribution of possible
> future states. Once the cube is held, that distribution is broad and weakly structured —
> every downstream placement remains reachable, and the successor predictions of actions
> that retain the grasp differ from those that drop it only in how the diffuse post-grasp
> region is weighted, not in which states appear in it. Unnormalized regression onto
> individual occupancy values supplies little pressure to get that weighting right: each
> future state carries its own small residual, and the aggregate distinction between keeping
> and losing control is spread thinly across all of them. We find empirically that plain FB
> reaches and even acquires the cube about as often as methods that succeed, and that
> handing it an already-secured cube does not close the gap — it drops what it is given.
> The same structure arises beyond manipulation wherever a policy must hold a condition
> rather than reach one: maintaining balance across a locomotion cycle, or keeping a
> vehicle within a lane while it advances.

**Two caveats to weigh before accepting it.** (i) This is cube-single at 500k, three seeds,
one architecture pair; the claim is about *these* estimators on *this* environment, and the
paragraph above is written to be defensible at that scope. (ii) The retention account is
supported by the failure decomposition, but the probe did not run the *positive* control for
it — an intervention that removes the retention difficulty and predicts the gap should close.
If you want the paragraph to assert a mechanism rather than a location, that control is the
next experiment, and I would keep the wording as a location claim until it exists.
