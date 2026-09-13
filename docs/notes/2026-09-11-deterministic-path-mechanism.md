# Why the categorical model preserves control on the deterministic path

Date: 2026-09-11. Interpretation and objective controls for the
[path theorem](2026-09-11-deterministic-path-counterexample.md) and its
[spherical extension](2026-09-11-deterministic-path-spherical-counterexample.md).
This note uses their exact target and explicit factors; it does not report a
new training experiment.

The strongest supported explanation is a change in **prediction geometry**.
Two linear successor features and two logit features describe different sets
of probability matrices. The categorical family can fit the broad spatial
variation and the interior goal peak simultaneously. Cross-entropy supplies
a useful global certificate, but squared loss on this same softmax family
also preserves the required decisions. This example is not a CE-only result.

## What the two features can describe

The exact normalized successor matrix is

\[
N=\frac1{85}\begin{pmatrix}
43&22&12&8\\22&33&18&12\\12&18&33&22\\8&12&22&43
\end{pmatrix}.
\]

Rows are indexed by immediate successor, not source state. Thus the important
decision from state 0 compares rows 1 and 0 at goal column 1. The true gap is
\(11/85>0\).

Linear FB constrains the successor prediction itself to rank two. In the
unconstrained reference calculation, the squared optimum retains the constant
direction and one broad left-to-right direction. The latter is monotone and
predicts a larger goal value at endpoint 0 than at interior goal 1. Other
rank-two FB features can represent the goal's values exactly, but their errors
elsewhere are larger. The spherical theorem proves that failure also occurs
under the specified spherical constraint and finite penalty; it does not
identify that optimum with the unconstrained SVD solution.

Density instead restricts the logits to rank two. The explicit candidate uses

\[
x=(-1,-1/2,1/2,1),\qquad e=(1,-1,-1,1),\qquad B_j=(x_j,e_j),
\]

and forward rows

\[
F=\begin{pmatrix}-.8&.1\\-.4&-.23\\.4&-.23\\.8&.1\end{pmatrix}.
\]

The first feature describes position; the second distinguishes endpoints from
interior states. For immediate successor 1, the forward row $(-.4,-.23)$
produces logits

\[
(.17,.43,.03,-.63).
\]

The negative second coefficient favors the interior, while the negative first
coefficient favors its left member. State 1 therefore has the largest logit.
The resulting goal probabilities for rows 0 and 1 are approximately
$0.2747761$ and $0.3587023$, preserving the correct initial choice.

More generally,

\[
q_i(j)\propto \exp(F_{i1}x_j)\exp(F_{i2}e_j).
\]

Exponentiation combines the two preferences multiplicatively; normalization
produces positive unit-mass rows without a learned constant logit direction.
This candidate has rank-two logits but rank-four probabilities. Its exact
features are a feasible explanation, not a claim that every fitted optimum
learns these particular coordinates.

For a strictly linear rank-two probability model with exact unit row mass,
normalization leaves only a one-dimensional affine variation of its rows.
Softmax can use both logit coordinates for relative preferences because a
common row-logit shift has no effect. FB does not impose exact row mass, but
its squared objective still pays for errors in the large common mass component.
Simply rescaling an unconstrained squared-regression target by $1-\gamma$
does not change its optimal subspace; that scalar operation is not the same
as replacing linear densities with normalized exponentials.

## The same categorical candidate can be spherical and whitened

The explanation does not require giving Density less restricted backward
features. Put $a=\sqrt{2/5}$, $c=(0,1/3)$, and

\[
S=\operatorname{diag}(2a,3a/2),\qquad
B'=(B-\mathbf1c)S
=a\begin{pmatrix}-2&1\\-1&-2\\1&-2\\2&1\end{pmatrix},
\qquad F'=FS^{-1}.
\]

Every row of $B'$ has squared norm $5a^2=2$, and

\[
\tfrac14(B')^{\mathsf T}B'=I.
\]

This is the uncentered second moment used by the FB orthogonality term, not
the covariance after subtracting the feature mean. The transformed logits obey

\[
F'(B')^{\mathsf T}=FB^{\mathsf T}-(Fc)\mathbf1^{\mathsf T}.
\]

Only a row constant was subtracted, so all categorical probabilities are
unchanged. The same candidate therefore satisfies the spherical constraint
and attains the orthogonality penalty's minimum. Relaxing those backward
constraints is unnecessary for the feasible categorical approximation used
in this example. The original existence proof is for the unrestricted logit
family; the objective comparisons below also apply to any attained global
minimum in the matched constrained family.

## What cross-entropy contributes, and what it does not establish

The candidate's mean KL is $0.003304199257608$. At a global categorical
optimum, identical target rows have identical predictions by conditional
strict convexity in centered logits. Combining the duplicated source weights,
any wrong or tied choice at state 0 costs at least $0.004806898167084$ in
KL; at state 2 the bound is $0.02019950996663$. The candidate beats both
bounds, so every global optimum must enter the goal at both states. Those
decisions imply optimal return $25/9$ from the specified start.

For a fixed target $p$, the cross-entropy gradient with respect to logits
is $q-p$. Locally,

\[
D_{\rm KL}(p\Vert q)\approx\frac12\sum_j\frac{(q_j-p_j)^2}{p_j}.
\]

Thus it measures allocation errors relative to target probability, rather
than the uniform-reference absolute squared errors in the FB approximation.
This is a description of the loss, not evidence that its relative weighting
is necessary for the control improvement in this graph.

## Squared loss on the same softmax family also succeeds

For the very same categorical candidate, including its spherical and whitened
factorization above, the four-row squared error is

\[
E_q=\sum_{i,j}(N_{ij}-q_{ij})^2
=0.00587903196817<0.005880.
\]

For the eight source rows with weights $1/8$ and reference $1/4$, the
corresponding objective is one half the sum of their row errors. If duplicate
targets share predictions, it equals the four-row expression above. A wrong
or tied goal ordering between rows 1 and 0 requires, from those two goal
coordinates alone,

\[
E_q\ge\frac12(11/85)^2=0.00837370242215.
\]

For rows 1 and 3, which determine the decision at state 2, the analogous bound
is $\tfrac12(21/85)^2=0.03051903114187$. Both exceed the feasible candidate.

This also rules out wrong choices at any global minimum with **eight freely
parameterized forward rows**. Conditional uniqueness is not needed:

1. Fix the backward features of a global minimum. Each forward row must minimize
   its own squared error. Duplicate target rows consequently have the same
   minimum error, even if their predicted distributions differ.
2. If a decision is wrong, take its two offending forward rows and copy each to
   the other source rows having that same target. Copy one minimizer for every
   remaining target as well. This preserves the total objective, backward
   constraints, and the wrong choice.
3. The resulting model has only four distinct prediction rows and violates
   the four-row lower bound above. Its objective cannot be globally minimal.

The argument is unchanged by adding a nonnegative excess orthogonality penalty:
copying forward rows leaves that penalty unchanged, and the feasible candidate
has zero excess penalty. Therefore CE is not essential to this example's
global control separation. Linear versus softmax prediction geometry remains
the decisive distinction demonstrated by this control.

The squared-error value is obtained by direct evaluation of the existing
explicit factors. Unlike the two principal counterexample verifiers, this
note does not add a separate rational-interval implementation for that decimal;
the displayed comparison has a margin greater than $0.00249$.

## Boundaries of the explanation

The exact target has probability rank four and centered-log rank three, so
both rank-two families misspecify full prediction. The linear family can
nevertheless represent the goal's values: its failure concerns what its
objective selects, not an inability to represent this control task.

The categorical TD target is ((1-\gamma)e_{s'}+\gamma\bar q(s',a')\), where
the second term changes during learning. The global categorical certificates
here concern the exact successor target, not global convergence of that TD
procedure. The spherical note separately treats an idealized reduced FB
potential with exactly equilibrated forward factors and instantaneous targets;
that does not supply a categorical TD convergence theorem. The experiment uses
a fixed uniform continuation policy and exact reward integration, with no
learned actor or reward-inference error.
