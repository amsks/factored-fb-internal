# Identifying the Dominant Mechanism Behind Bilinear GCIQL in Cube Manipulation

## Goal

We want to determine which physical and value-function structure is primarily responsible for the bilinear critic's advantage in `cube-single`.

The leading candidate explanations are:

1. **Grasp acquisition:** the bilinear critic better identifies actions that establish control of the cube.
2. **Retained control or viability:** it better identifies actions that preserve a stable, steerable grasp.
3. **Directional action--goal composition:** it better matches local cube motion to the direction of the requested goal.
4. **Scripted pick-and-place primitives:** it learns the phases and action patterns of the play-policy data generator rather than a general manipulation structure.

These candidates are correlated in ordinary trajectories. A good grasp often also preserves control, and an action that preserves control is often needed before directional transport. Correlation with a learned value mode is therefore not enough to identify a dominant explanation.

The central strategy is:

\[
\boxed{
\text{necessity}
+
\text{sufficiency}
+
\text{environmental intervention}
}
\]

A candidate is called dominant only when it passes all three tests.

---

## 1. What would count as a dominant explanation?

Let

\[
A_{\mathrm{bil}}(s,a,g)
=
Q_{\mathrm{bil}}(s,a,g)
-
\mathbb{E}_{\tilde a\sim\mu(\cdot\mid s)}
Q_{\mathrm{bil}}(s,\tilde a,g)
\]

be the centered action--goal preference operator of bilinear GCIQL.

For a candidate mechanism \(h\), construct a physically grounded candidate operator

\[
T_h(s,a,g).
\]

We then ask three questions.

### 1.1 Sufficiency

Does the part of the bilinear value operator aligned with \(T_h\) preserve most of the bilinear advantage?

\[
A_{\mathrm{keep}\text{-}h}
=
P_{\mathcal T_h}A_{\mathrm{bil}},
\]

where \(P_{\mathcal T_h}\) projects onto the candidate subspace.

If actions ranked using \(A_{\mathrm{keep}\text{-}h}\) retain most of the bilinear improvement over the MLP, then the candidate is largely sufficient.

### 1.2 Necessity

Does removing that component destroy most of the advantage?

\[
A_{\mathrm{remove}\text{-}h}
=
A_{\mathrm{bil}}
-
P_{\mathcal T_h}A_{\mathrm{bil}}.
\]

If the bilinear advantage disappears after removing \(h\), then the component is necessary.

### 1.3 Environmental intervention

Does removing the corresponding physical difficulty from the environment also remove the bilinear advantage?

This is the strongest test because it intervenes on the proposed cause rather than only interpreting a learned subspace.

### 1.4 Pre-registered dominance threshold

Call candidate \(h\) dominant only if:

- keeping only \(h\) preserves at least \(80\%\) of the bilinear-over-MLP gain;
- removing \(h\) destroys at least \(80\%\) of that gain;
- the corresponding environmental intervention substantially reduces the bilinear-over-MLP gap;
- competing candidates do not satisfy all three conditions.

The \(80\%\) threshold is illustrative and should be fixed before running the experiment.

---

## 2. Physically grounded candidate operators

These candidate objects should be constructed from simulator counterfactuals, not from another flexible learned probe.

### 2.1 Directional transport

For a state \(s\) and candidate action \(a\), measure the short-horizon cube displacement

\[
\Delta x_{\mathrm{cube}}(s,a).
\]

For goal \(g\), define the normalized goal direction

\[
d_g(s)
=
\frac{
g-x_{\mathrm{cube}}(s)
}{
\left\|g-x_{\mathrm{cube}}(s)\right\|+\epsilon
}.
\]

The directional candidate is

\[
T_{\mathrm{dir}}(s,a,g)
=
\Delta x_{\mathrm{cube}}(s,a)^\top d_g(s).
\]

This asks:

> Does action \(a\) move the cube in the direction required by goal \(g\)?

A richer version keeps three separate fields:

\[
T_x(s,a,g)=\Delta x_x(s,a)d_{g,x}(s),
\]

\[
T_y(s,a,g)=\Delta x_y(s,a)d_{g,y}(s),
\]

\[
T_z(s,a,g)=\Delta x_z(s,a)d_{g,z}(s).
\]

### 2.2 Grasp acquisition

From a pre-contact state, execute \(a\), then use a fixed short continuation and measure

\[
p_{\mathrm{grasp}}(s,a)
=
\Pr(
\text{secure control is acquired within }H
\mid s,a
).
\]

The goal-common candidate operator is

\[
T_{\mathrm{grasp}}(s,a,g)
=
p_{\mathrm{grasp}}(s,a).
\]

This asks:

> Does the action place the robot into a state where manipulation can begin?

Because nearly every placement goal requires grasp acquisition, this component should be positive across a broad goal region.

### 2.3 Retained control or manipulation viability

From an in-hand state, estimate

\[
p_{\mathrm{hold}}(s,a)
=
\Pr(
\text{the cube remains controlled for }H
\mid s,a
).
\]

Define

\[
T_{\mathrm{hold}}(s,a,g)
=
p_{\mathrm{hold}}(s,a).
\]

A stronger viability candidate measures how many goals remain reachable after the action:

\[
T_{\mathrm{viable}}(s,a)
=
\frac{1}{|\mathcal G|}
\sum_{g\in\mathcal G}
\mathbbm{1}
[
g\text{ remains reachable after }a
].
\]

This asks:

> Does the action preserve useful future control over the cube?

This is related to empowerment, but it is restricted to control that matters for the benchmark's goal family.

### 2.4 Scripted pick-and-place primitives

Construct a candidate basis corresponding to the play-policy phases:

\[
\text{approach},
\quad
\text{close},
\quad
\text{lift},
\quad
\text{translate},
\quad
\text{release}.
\]

For example, define a fixed action-template score

\[
T_{\mathrm{script},k}(s,a)
=
-\left\|
a-a_k^{\mathrm{template}}(s)
\right\|^2.
\]

This tests the less flattering explanation:

> The bilinear critic may be learning the repeated phases of the scripted data generator rather than a general physical action-effect basis.

---

## 3. The decisive environmental interventions

The candidate mechanisms should be separated by interventions that remove one difficulty while preserving the others.

### 3.1 Secure in-hand initialization

Start every episode from a standardized state in which the cube is already securely held.

This removes:

- approach;
- grasp acquisition.

It retains:

- grasp retention;
- directional steering;
- placement.

#### Prediction

- If the bilinear advantage largely disappears, grasp acquisition is a major cause.
- If the advantage remains, grasp acquisition cannot be the dominant global explanation.

### 3.2 Weld the cube to the gripper after contact

After the grasp criterion is satisfied, temporarily weld the cube to the gripper.

This removes:

- slipping;
- dropping;
- marginal grasp instability.

It retains:

- approach;
- contact acquisition;
- arm motion;
- directional transport;
- goal placement.

#### Prediction

- If the bilinear advantage disappears, retained control or grasp viability is dominant.
- If it remains, the main advantage is not stable grasp retention.

This is the cleanest intervention for the retained-control hypothesis.

### 3.3 Rotated or reflected workspace

Rotate the robot--cube configuration, action coordinates, and goals together.

A genuine local-effect--goal-direction factorization should transform consistently:

\[
A(Rs,Ra,Rg)
\approx
A(s,a,g).
\]

#### Prediction

- Directional action--goal fields should rotate or reflect with the environment.
- A dataset-script basis need not preserve this equivariance.

### 3.4 A different play-policy generator

Generate a second offline dataset that solves the same pick-and-place tasks using different:

- approach trajectories;
- wrist orientations;
- grasp strategies;
- phase durations;
- transport paths.

#### Prediction

- If the basis represents physical action effects, its dominant goal fields should remain similar.
- If it represents the original scripted policy, its subspace and performance advantage should change substantially.

---

## 4. The expected phase-dependent answer

A single factor may not dominate the entire episode.

The strongest current hypothesis is a phase-dependent hybrid basis:

\[
\begin{aligned}
\mathrm{P0/P1}:&
\quad
\text{grasp acquisition and general manipulation feasibility},
\\
\mathrm{P2}:&
\quad
\text{retained control and steerability},
\\
\mathrm{P3}:&
\quad
\text{local cube displacement}\times\text{goal direction}.
\end{aligned}
\]

This still gives one coherent explanation:

\[
\boxed{
\text{acquire control}
\rightarrow
\text{preserve control}
\rightarrow
\text{use control directionally}
}
\]

The experiment should therefore report dominance separately at each phase before claiming one global mechanism.

---

# Reusing analyses from *Is Manipulation Hard for Successor Measures?*

Several analyses in the current paper can be reused directly. Others can be adapted as supporting evidence.

## 5. Phase decomposition: directly reusable and central

The paper already separates rollouts into:

1. approach;
2. control acquisition;
3. maintain control;
4. transport.

This is exactly the right framework for testing phase-dependent dominance.

### Reapplication

Run the same phase decomposition for:

- the original bilinear and MLP agents;
- secure-in-hand evaluation;
- welded-cube evaluation;
- policies induced by the keep-only and remove-only value operators.

The important comparison is how each intervention changes the terminal failure profile.

### Example conclusions

- If welding removes the bilinear advantage specifically by eliminating maintain-control failures, retained control is the dominant factor.
- If the gap remains but transport failures change with rotated goals, directional composition is more important.
- If secure in-hand starts remove the gap before transport begins, grasp acquisition is dominant.

This reuses the paper's strongest diagnostic almost unchanged.

## 6. Counterfactual in-hand value ranking: directly reusable for retained control

The paper resets the simulator to in-hand grasp states and asks whether the method's value ranks those states by eventual success under its own policy.

This is the most directly reusable analysis for the retained-control hypothesis.

### Reapplication

Use the same in-hand states and labels, but compare:

- full bilinear value;
- MLP value;
- \(T_{\mathrm{hold}}\);
- \(T_{\mathrm{viable}}\);
- bilinear value with the hold component removed;
- hold-only bilinear component.

The main questions are:

\[
\operatorname{AUC}(T_{\mathrm{hold}},Y)
\]

and

\[
\operatorname{AUC}
(
A_{\mathrm{remove}\text{-}\mathrm{hold}},
Y
).
\]

### Decisive pattern

Retained control becomes the dominant explanation if:

- \(T_{\mathrm{hold}}\) ranks eventual success almost as well as the full bilinear value;
- the hold-only component preserves most of the ranking;
- removing it reduces the bilinear ranking toward the MLP or FB level;
- welding the cube removes both the ranking advantage and the performance gap.

The existing paper already shows that GCIQL strongly ranks in-hand states while FB, RLDP, and PSM do not. The new analysis can determine whether this strength comes specifically from retained-control information.

## 7. Representation--value ladder: reusable with a different ladder

The paper compares:

\[
\text{raw state}
\rightarrow
\text{representation readout}
\rightarrow
\text{learned scalar value}.
\]

This distinguishes representation bottlenecks from scalar-readout bottlenecks.

### Reapplication

Construct a candidate--value ladder:

\[
\text{physical candidate}
\rightarrow
\text{candidate subspace of }A
\rightarrow
\text{full bilinear value}.
\]

For retained control, for example:

1. physical score \(T_{\mathrm{hold}}\);
2. projection of the bilinear operator onto the hold candidate;
3. full bilinear value;
4. bilinear value after removing the hold component.

This analysis shows where the candidate information enters and whether the final scalar ranking uses it.

### Limitation

The ladder alone is not causal. It becomes strong only when paired with the keep/remove intervention and the welded-cube environment.

## 8. State-support analysis: reusable as a confound control

The paper compares the distance to offline data for successful and failed pickup/carry states.

### Reapplication

Use support diagnostics to ensure that an apparent candidate effect is not only an action-support artifact.

Measure:

- state support at each phase;
- action distance to the frozen BC-flow cloud;
- support of actions selected by full, keep-only, and remove-only operators.

### Purpose

Suppose the hold-only critic appears strong but selects much more supported actions. Then its benefit may partly arise from conservatism rather than retained-control semantics.

The support analysis cannot identify the dominant mechanism, but it can rule out a major confound.

## 9. Backward-embedding geometry probe: adaptable for directional composition

The paper asks how well backward embeddings predict cube-to-goal distance and shows that coarse geometry alone does not explain maintained control.

The same logic can be applied to the bilinear value modes.

### Reapplication

For each leading goal mode \(v_k(g)\), test whether it corresponds to:

- horizontal goal direction;
- vertical goal direction;
- cube-to-goal distance;
- goal height;
- a broad goal-common field.

For each leading action mode \(u_k(s,a)\), compare it with:

- cube displacement;
- end-effector displacement;
- grasp acquisition;
- grasp retention.

### Interpretation

This can show whether the dominant bilinear modes are directional, contact-related, or goal-common.

### Limitation

This remains a semantic probe. It should support, not replace, the necessity/sufficiency and environment-intervention tests.

## 10. Local successor gap: reusable as a formal framework

The paper defines a local successor gap between nearby acquired-control states with different eventual success.

This can be adapted into a local action-preference gap:

\[
\Gamma_{\epsilon}^{A}(g)
=
\sup_{
\substack{
(s,a_1),(s,a_2)\\
\|a_1-a_2\|\leq\epsilon
}
}
\left|
A(s,a_1,g)
-
A(s,a_2,g)
\right|.
\]

Candidate-specific versions can be defined:

\[
\Gamma_{\epsilon}^{\mathrm{hold}},
\qquad
\Gamma_{\epsilon}^{\mathrm{dir}},
\qquad
\Gamma_{\epsilon}^{\mathrm{grasp}}.
\]

This provides a theoretical connection between:

- small local action changes;
- different contact or transport outcomes;
- large goal-value differences.

It is useful for the paper's formal framing, but it is not itself the decisive empirical test.

## 11. Per-goal success and terminal outcomes: directly reusable

The appendix already reports per-goal success and exact terminal outcomes.

### Reapplication

Report each intervention and critic-surgery result per goal.

This matters because different mechanisms may dominate for different targets:

- elevated goals may emphasize lifting and grasp retention;
- lateral goals may emphasize directional steering;
- goals requiring longer transport may emphasize viability.

A mechanism that appears dominant only after averaging over goals should not be called globally dominant.

---

# Minimal decisive experiment sequence

## Step 1: secure in-hand evaluation

Remove approach and grasp acquisition.

- Gap disappears: grasp acquisition is the leading candidate.
- Gap remains: move to retained control versus direction.

## Step 2: welded-cube evaluation

Remove slipping and dropping.

- Gap disappears: retained control is dominant.
- Gap remains: directional composition or script structure remains live.

## Step 3: rotated-workspace evaluation

Test whether the bilinear goal fields rotate consistently.

- Equivariance holds: directional action--goal composition is strongly supported.
- Equivariance fails: scripted primitives or other task-specific correlations become more plausible.

## Step 4: keep/remove candidate value surgery

Run local counterfactual action selection using:

- full bilinear operator;
- hold-only;
- direction-only;
- full minus hold;
- full minus direction.

Measure the simulator outcome under the same continuation.

## Step 5: alternative play-policy dataset

Use this only if the script-basis explanation remains plausible after the first four steps.

---

# What a decisive conclusion would look like

## Outcome A: retained control dominates

Evidence:

- the bilinear advantage remains from secure in-hand starts;
- it disappears when the cube is welded;
- \(T_{\mathrm{hold}}\) matches bilinear action and state rankings;
- hold-only value preserves most of the gain;
- removing the hold component destroys it;
- directional fields remain secondary.

Conclusion:

> Bilinear GCIQL's dominant advantage in Cube comes from allocating a reusable value direction to preserving stable, steerable control of the object.

## Outcome B: directional composition dominates

Evidence:

- the advantage remains under secure in-hand and welded-cube evaluation;
- the leading value fields rotate with the workspace;
- directional-only value preserves most of the gain;
- removing directional fields destroys it.

Conclusion:

> Bilinear GCIQL's dominant advantage comes from factoring local cube-motion effects from global goal direction.

## Outcome C: grasp acquisition dominates

Evidence:

- the advantage largely disappears from secure in-hand starts;
- grasp-only value preserves most of the pre-contact gain;
- removing grasp-acquisition fields destroys the advantage at P0/P1.

Conclusion:

> Bilinear GCIQL mainly improves the early interaction decision that establishes useful object control.

## Outcome D: scripted primitives dominate

Evidence:

- the basis changes substantially under a new data-generating controller;
- rotation equivariance is weak;
- template-phase scores explain the value modes better than physical action effects;
- performance degrades when training trajectories change despite unchanged manipulation physics.

Conclusion:

> The bilinear factorization primarily compresses the repeated skill structure of the play-policy dataset rather than a policy-independent manipulation geometry.

---

# Final recommendation

The strongest reuse from the existing paper is:

1. **phase decomposition** for locating where each candidate matters;
2. **counterfactual in-hand value ranking** for retained control;
3. **the representation--value ladder**, adapted into a candidate--value keep/remove ladder;
4. **state-support gaps** as a control;
5. **per-goal terminal outcomes** for checking whether dominance varies by target.

The new contribution beyond the current paper is the combination of:

\[
\boxed{
\text{candidate-specific value surgery}
+
\text{environmental intervention}
}
\]

The existing paper shows where manipulation methods fail and whether their values rank interaction states. The proposed experiment can identify which physical value component causes the bilinear critic's advantage.