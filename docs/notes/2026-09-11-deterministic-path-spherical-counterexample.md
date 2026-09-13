# A global spherical-FB counterexample on the deterministic path

Date: 2026-09-11. Exact rational certificate, independently audited. This extends
the [four-state path theorem](2026-09-11-deterministic-path-counterexample.md) to
FB's spherical backward features and a finite, soft orthogonality penalty.

At dimension two, with two exact-target FB heads and orthogonality coefficient
1000, **every global spherical-FB approximation optimum stays outside the goal**.
Its return from the start is zero. The same spherical class contains a positive,
unit-mass representation that predicts every value for this goal exactly. Every
global rank-two-logit categorical optimum achieves the optimal return (25/9)
from that start. This theorem concerns exact-successor approximation; it does
not assert that every TD training run converges to this optimum.

The [mechanism note](2026-09-11-deterministic-path-mechanism.md) explains the
spatial and interior features, gives a probability-preserving transformation
of the categorical candidate to spherical, whitened features, and shows that
squared loss on the softmax family also preserves control. The result does
not identify cross-entropy alone as the necessary change.

## Graph, target, and objective

States 0 through 3 form a path. Left and right move deterministically to the
adjacent state; outward actions at the endpoints stay there. Start at state 0
and reward every arrival at state 1, which is not terminal. Discount is (4/5).
The fixed continuation policy chooses either action uniformly. All eight source
state-action pairs have equal weight, giving the natural uniform next-state
reference. The normalized successor target, indexed by immediate successor, is

\[
N=\frac1{85}\begin{pmatrix}
43&22&12&8\\22&33&18&12\\12&18&33&22\\8&12&22&43
\end{pmatrix}.
\]

For a source pair ((s,a)), use row (f(s,a)) of (N), where (f) is the
deterministic transition map. The critic integrates the specified goal reward
against its successor prediction. A negative right-minus-left goal difference
at state 0 makes its greedy policy stay there forever.

Constrain each row of (B\in\mathbb R^{4\times2}) to norm (\sqrt2), and set

\[
C=B^{\mathsf T}B/4,\qquad
\operatorname{eig}(C)=1\pm\delta,\qquad0\le\delta\le1.
\]

When (B) has full rank, its orthogonal column-space projector is
\(\Pi=B(B^{\mathsf T}B)^{-1}B^{\mathsf T}\).
At a global joint approximation optimum, every unconstrained forward row
minimizes its squared loss for this (B). Both heads therefore have prediction
(N\Pi), with common error

\[
E=\|N-N\Pi\|_F^2=\operatorname{tr}(N^2)-\operatorname{tr}(N^2\Pi).
\]

This elimination of the forward factors is essential. The bounds below apply
to global joint minimizers, whose forward factors must be conditionally optimal;
they are not a lower bound for every arbitrary correct-control pair (F,B).
These FB successor predictions need not have unit row mass.

The [implemented orthogonality term](../../impls/critics/fb.py) is

\[
\tfrac12\operatorname{tr}(C^2)-\operatorname{tr}(C)
=\tfrac12\|C-I\|_F^2-1=\delta^2-1.
\]

Unnormalized successors are (5N). Each head's exact-target regression loss
is (25E/2); two heads contribute (25E), while the regularizer is counted
once. At coefficient 1000, subtracting the common regularizer minimum and
dividing by 25 gives exactly

\[
J=E+40\delta^2.
\]

A one-head critic with coefficient 500 has the same relative objective. This
is a finite soft penalty, not a hard constraint (C=I). The coefficient is
material; this proof does not establish the same result at coefficient one.
The stronger coefficient is specified in
[the modern table critic configuration](../../impls/configs/critic/fb_tdj.yaml).

## A candidate constrains every global optimum

Take (B=\sqrt2U), with

\[
U=\begin{pmatrix}1&0\\255/257&32/257\\0&1\\-32/257&255/257\end{pmatrix}.
\]

The identity (255^2+32^2=257^2) gives unit row norms, and
(U^{\mathsf T}U=2I), hence (C=I). Its exact objective is

\[
J=E=\frac{149734}{1856825}=0.08063980181223<\frac{81}{1000}.
\]

Its goal gap is (-1168/21845<0), but we do not assume this candidate is the
global optimum. Every global optimum has (J<0.081), hence (\delta<1/20),
because (40(1/20)^2=0.1). Rank-deficient (B), with (\delta=1), cannot
belong to this sublevel. A global minimum exists: spherical (B) lie in a
compact set, (C) is uniformly nonsingular on this sublevel, and the minimized
objective and optimal forward factors are continuous there.

The spherical row norms imply

\[
\Pi_{jj}=\tfrac14 B_j^{\mathsf T}C^{-1}B_j
\in\left[\frac1{2(1+\delta)},\frac1{2(1-\delta)}\right],
\qquad
\left|\Pi_{jj}-\frac12\right|<\frac1{38}.
\]

Thus the near-whitening property used next follows from the actual penalty
and a feasible objective value.

## Exact certificate for a uniformly negative action gap

Let (A=N^2), (d=N_{1,:}-N_{0,:}), and

\[
D=\tfrac12(de_1^{\mathsf T}+e_1d^{\mathsf T}),\qquad
g=\operatorname{tr}(D\Pi)=(N\Pi)_{1,1}-(N\Pi)_{0,1}.
\]

Use the rational constants

\[
\mu=43/50,\quad z=(-1/10,-1/20,17/100,0),\quad
\beta=13/100,\quad v=(357,469,573,478)^{\mathsf T}/1000.
\]

They certify

\[
H=A+\mu D+\operatorname{diag}(z)\preceq\beta I+vv^{\mathsf T}.
\]

All 15 principal minors of the rational difference are positive. In particular,
its four leading principal minors are

\[
\frac{1662761}{289000000},\quad\frac{5008221}{144500000000},\quad
\frac{2817179}{14450000000000},\quad\frac{7937679}{7225000000000000}.
\]

Sylvester's criterion proves positive definiteness without numerical eigenvalue
tolerances. For a rank-two orthogonal projector, (0\preceq\Pi\preceq I)
and (\operatorname{tr}\Pi=2), so (\operatorname{tr}(H\Pi)\le2\beta+\|v\|^2).
Using the projector-diagonal bound, and retaining the action-gap term, gives

\[
\begin{aligned}
E&\ge\operatorname{tr}(A)-2\beta-\|v\|^2
+\tfrac12\sum_jz_j-\frac{\|z\|_1}{38}+\mu g\\
&=\frac{580561507}{5491000000}+\frac{43}{50}g
=0.105729649790566\ldots+0.86g.
\end{aligned}
\]

Every global minimizer has (E\le J<81/1000), hence

\[
g<\frac{81/1000-580561507/5491000000}{43/50}
=-\frac{135790507}{4722260000}
=-0.0287554067332\ldots<0.
\]

This excludes correct or tied initial choices at **every global optimum**.
The proof does not select a sampled numerical minimum or depend on learning
rate or training duration. Starting at state 0, each such optimum stays there
and receives zero return.

## Spherical capacity is sufficient for this reward

Let (p_i=N_{i,1}), and choose backward rows

\[
B=\begin{pmatrix}1&1\\1&1\\1&-1\\1&-1\end{pmatrix},
\qquad F_i=(1,4p_i-1).
\]

The rows have norm (\sqrt2), the backward second moment is (I), and

\[
\widehat N_{i,:}=(p_i,p_i,1/2-p_i,1/2-p_i).
\]

Since (0<p_i<1/2), these are positive unit-mass rows with exact goal values.
Native FB reward readout agrees: (z_r=\mathbb E_\rho[B r]=(1/4,1/4)), so
(F_i^{\mathsf T}z_r=p_i). Scaling the forward rows by five restores the
unnormalized values. This predicts all action values for this goal exactly;
it makes no assertion about representing every reward's value simultaneously.
Its zero-penalty objective is

\[
J=E=\frac{1228}{7225}=0.169965397923875\ldots,
\]

which is larger than the failing candidate's objective.

## Additional result for the expected FB field after forward equilibration

The exact-target result above is not the only connection to FB. On this
reversible graph, one can also identify and certify a potential for a specific
reduction of the actual expected FB semi-gradient field. This reduction
requires the forward factors to be exactly at their TD fixed point for the
current backward features, and target networks to equal the current networks.
It does not describe general simultaneous updates with finite learning rates
or target lag.

Let (R\in\mathbb R^{8\times4}) be the deterministic source-to-next-state
matrix, (W=I_8/8), and (G=(I-\gamma T)^{-1}=5N). Then
\(R^{\mathsf T}WR=I_4/4\). For full-rank (B), the forward TD equilibrium is

\[
F=RGB C^{-1},\qquad FB^{\mathsf T}=4RG\Pi.
\]

After this substitution, the expected backward semi-gradient of one FB head,
excluding the orthogonality penalty, is

\[
-\frac14(I-\Pi)GB C^{-1}
=-\frac12\nabla_B\operatorname{tr}(\Pi G).
\]

The equality uses symmetry of (G) and
\(\nabla_B\operatorname{tr}(\Pi G)=2(I-\Pi)GB(B^{\mathsf T}B)^{-1}\).
Projection onto the spherical tangent spaces preserves this equality. Thus
for two heads the reduced field is gradient descent on

\[
-\operatorname{tr}(\Pi G)+1000\delta^2.
\]

Adding a constant and dividing by five gives the reduced potential

\[
J_{\rm red}=\operatorname{tr}(N)-\operatorname{tr}(N\Pi)+200\delta^2.
\]

This uses (N), whereas the oracle squared objective uses (N^2). The two
objectives are not silently identified. The same spherical, whitened candidate
above has reduced potential (0.365850308995193\ldots<183/500=0.366).
Every global minimum of this potential therefore has (\delta<1/20) and
the same projector-diagonal bound as above. Its existence follows from the
same compact-sublevel argument.

A second exact rational certificate uses

\[
\mu=5/4,\quad z=(-9/100,0,13/50,0),\quad\beta=3/8,
\quad v=(239,435,559,409)^{\mathsf T}/1000.
\]

All 15 principal minors of
\(\beta I+vv^{\mathsf T}-[N+\mu D+\operatorname{diag}(z)]\)
are strictly positive, as checked with exact fractions. Applying the same
trace argument to (E_{\rm red}=\operatorname{tr}(N)-\operatorname{tr}(N\Pi))
gives

\[
E_{\rm red}\ge\frac{31324279}{80750000}+\frac54g
=0.387916767801857\ldots+1.25g.
\]

Since (E_{\rm red}\le J_{\rm red}<0.366), every global minimizer of this
reduced TD potential has

\[
g<\frac{0.366-0.387916767801857\ldots}{1.25}
=-\frac{1769779}{100937500}=-0.017533414241486\ldots<0.
\]

This reduced family can also implement optimal control with its forward
factors at TD equilibrium. Take (B_j=(1,s_j)), where (s=(-1,1,1,-1)).
It is spherical and whitened, and the equilibrated prediction has goal column
\((0.2,0.3,0.3,0.2)\). Its greedy policy enters the goal from states 0 and 2
and moves left from state 3, giving optimal control. Its reduced potential
is (10/17=0.588235294117647\ldots), larger than the failing candidate's.
These predictions need not equal the true fixed-continuation values to induce
optimal control.

This additional theorem concerns global minimizers of the reduced potential.
It does not prove convergence of the coupled forward/backward dynamics, target
EMA, Adam, or sampled training. It also does not provide a global theorem for
the categorical TD objective: the Density certificate remains an exact-target
approximation result. An independent finite-difference audit of the reduced
field is saved in
[the audit script](../../analysis/spherical_symmetric_certificate_audit.py).

## Categorical certificate and scope

The graph and target are unchanged, so the
[original categorical certificate](2026-09-11-deterministic-path-counterexample.md#an-explicit-categorical-candidate-and-a-global-control-certificate)
applies. Its explicit rational rank-two-logit candidate has mean KL
(0.003304199257608), below the wrong-or-tied bounds (0.004806898167084)
and (0.02019950996663) for states 0 and 2. Every global categorical optimum
therefore enters the goal at both states and achieves optimal return (25/9)
from start 0. The proof includes existence of a finite optimum and the
duplicate-row argument for combining source weights. It does not need to
certify the unused state 3. If two categorical heads share (B), conditional
strict convexity also makes their predictions identical at a global
exact-target optimum, so the same certificate applies to their average loss.

Both full-target classes remain misspecified: (N) has rank four, and its
row-centered logarithm has rank three. Density backward features are
unrestricted, matching that implementation's default. This is not loss-only.

The theorem concerns exact-successor approximation. Actual TD substitutes a
target-network prediction for future occupancy, yielding a changing-target
procedure. The [TD report](2026-09-11-deterministic-path-td.md) includes spherical
runs that learn correct control and runs that fail. These recoveries do not
contradict the approximation theorem; the theorem does not prove that every
trained spherical critic fails. A fixed uniform continuation policy and exact
reward integration are used throughout, with no learned actor, task encoder,
or reward-inference protocol.

## Reproduction

```bash
../Factored-FB/.venv/bin/python scripts/didactic/verify_deterministic_path_spherical.py
../Factored-FB/.venv/bin/python scripts/didactic/verify_deterministic_path.py
```

The [spherical verifier](../../scripts/didactic/verify_deterministic_path_spherical.py)
checks exact spherical norms, second moments, candidate projection/error,
all 15 positive principal minors, the uniform action-gap bound, and the feasible
control representation. It also verifies the separate reduced-TD-potential
certificate with its own 15 exact positive principal minors and equilibrated
control witness. The [original verifier](../../scripts/didactic/verify_deterministic_path.py)
checks the MDP, source/reference weighting, full-target ranks, policy returns,
and rational categorical KL bounds. Both end with `"all_assertions_passed": true`.
Neither performs training or parameter search.
