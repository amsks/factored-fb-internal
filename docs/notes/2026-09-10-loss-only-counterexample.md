# A finite-capacity, loss-only counterexample

This construction separates representational capacity from the representation
selected by a population objective. The same class of stochastic rank-two
probability matrices can implement correct control, yet its global squared-loss
optimum chooses the wrong action in both contexts. Every global cross-entropy
optimum over that same class chooses correctly. No training trajectory is involved.

## Construction and scope

There are two equally likely contexts, two equally sampled actions, and four
outcomes. After one action the episode ends; set the discount to zero and reward
arrival at the third outcome. Write the transition probabilities as

\[
P_{x,a}=p_0+xv+aw,\qquad x,a\in\{-1,+1\},
\]

with

\[
\begin{aligned}
p_0&=(.49,.49,.01,.01),\\
v&=(.0196,-.0196,-.0008,.0008),\\
w&=(.001,-.001,.009,-.009).
\end{aligned}
\]

All probabilities are positive. The true probability matrix has rank three, and
its row-centered log-probability matrix also has rank three. Both rank-two
probability and rank-two-logit classes are therefore misspecified for full
prediction. Nevertheless, the feasible approximation

\[
Q^{\rm good}_{x,a}=p_0+aw
\]

has probability rank two and admits logits of rank at most two: there are only
two distinct rows. It reproduces **every reward's action-value difference**,
because the omitted term depends only on context. In particular, the correct
action's goal advantage is .018 in both contexts.

Use the explicitly independent reference \(\rho=(1/4,1/4,1/4,1/4)\), rather than
the transitions' natural next-state marginal \(p_0\). With zero discount, the
[FB population TD loss](../../impls/critics/fb.py) is, up to an additive constant,

\[
\tfrac12\mathbb E_{x,a}\sum_j
\frac{(P_{x,a,j}-Q_{x,a,j})^2}{\rho_j},\qquad
Q=FB^\top D_\rho,\quad D_\rho=\operatorname{diag}(\rho).
\]

Thus uniform reference gives ordinary squared probability error. There is no
bootstrap or exact-successor-versus-TD substitution in this example.

## Global squared optimum chooses incorrectly

The truncated SVD is the global rank-two squared optimum. Its retained singular
values are separated from the discarded one; its rows already sum to one and
its entries are positive. It is therefore also the unique optimum within the
**stochastic** rank-two class. Its predicted goal advantage is

\[
\widehat Q_{x,+}(3)-\widehat Q_{x,-}(3)
=-3.52682571\times10^{-5}
\]

in both contexts: strict reversal, not a tie. The sign also follows from a
two-dimensional eigensystem. In the paired outcome directions, its top direction
is proportional to \((1,-\kappa)\), where

\[
0<\kappa<\frac{.00000668}{.00030352}<\frac{.001}{.009}.
\]

Its goal coordinate is negative while its projection of the action direction
is positive. The resulting policy earns .001 on average; the correct policy
earns .019, a factor of 19 higher.

## Cross-entropy optimum must choose correctly

For the feasible \(Q^{\rm good}\), mean forward KL is

\[
\mathbb E_{x,a}D_{\rm KL}(P_{x,a}\Vert Q^{\rm good}_{x,a})
=.001169123764.
\]

Any prediction choosing the wrong action, or tying, in either context has a
larger KL lower bound. Aggregate outcomes into goal versus non-goal and apply
data processing. If the true goal probabilities are \(p_+>p_-\), the minimum
Bernoulli KL subject to reversed predicted ordering occurs when both predictions
equal \(m=(p_++p_-)/2\). Accounting for the two rows' weights gives

\[
\tfrac14[\operatorname{kl}(p_+\Vert m)+
\operatorname{kl}(p_-\Vert m)]\ge .002214551837.
\]

This exceeds the feasible upper bound. Consequently every global cross-entropy
optimum in the same stochastic rank-two class selects both actions correctly.
The same certificate covers the rank-two-logit density class because it also
contains \(Q^{\rm good}\). This is a loss-only separation within the shared
probability class; the additional density-class result is separate.

## Limits and reproduction

FB backward outputs are unrestricted here. Orthogonality regularization may be
retained: weighted whitening realizes the SVD solution while attaining the
regularizer's global minimum. The production spherical output constraint is
not covered. Production minibatches also use next-state candidates; replacing
the independent uniform reference by the natural marginal \(p_0\) restores
correct squared-fit action ordering in this example.

Values use exact reward integration. No learned actor, reward encoder,
reward-inference evaluation, default production comparison, persistent-state
mechanism, or manipulation claim follows from this construction.

Run with Python, NumPy and SciPy:

```bash
python scripts/didactic/verify_loss_only_counterexample.py
```

The [verification script](../../scripts/didactic/verify_loss_only_counterexample.py)
checks positivity, normalization, ranks, singular separation, strict action
reversal, KL bounds, representability, and reference sensitivity. Its `strict`
record is this example; `baseline` records a simpler tied-action control.
