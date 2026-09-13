# A population objective counterexample with enough dimension for control

Date: 2026-09-10. Mathematical construction, with numerical checks of explicit
matrices; this is not a trained-agent result.

At dimension two, the unique population squared-loss FB prediction selects the
wrong action in both contexts, although another feasible FB representation
preserves action-value differences for **every reward**. A categorical successor
model of the same latent dimension has an objective certificate for correct
control. The separation concerns the linear versus exponential model classes:
it does not establish an advantage unique to cross-entropy.

**MDP and sampling.** There are two contexts \(x\in\{-1,+1\}\), two actions
\(a\in\{-1,+1\}\), and four terminal outcomes, indexed \(0,1,2,3\). The reward
is paid on arrival; \(\gamma=0\). Define

\[
e_1=\tfrac12(1,-1,1,-1),\qquad e_3=\tfrac12(1,-1,-1,1),\qquad p_0=\tfrac14\mathbf1,
\]
\[
v=.1e_1-.005e_3,\qquad w=.01e_1+.05e_3,\qquad P_{xa}=p_0+xv+aw.
\]

The four rows, ordered \((++,+-,-+,--)\), are

\[
P=\begin{pmatrix}
.3275&.1725&.2825&.2175\\
.2675&.2325&.3225&.1775\\
.2325&.2675&.1775&.3225\\
.1725&.3275&.2175&.2825
\end{pmatrix}.
\]

Every transition has positive probability. Sample source rows uniformly. Their
actual outcome marginal is also uniform, \(\rho_j=1/4\). The diagnostic reward
is \(r_j=\mathbf1\{j=2\}\). Thus
\(Q_r(x,+)-Q_r(x,-)=2w_2=-.04\): action \(-1\) is strictly optimal in both
contexts.

**The FB population optimum fails.** The critic predicts the successor density
ratio \(m_{xa,j}=4P_{xa,j}\). With dimension-two factors, let
\(\widehat P_{xa,j}=F_{xa}^{\top}B_j/4\). Its population objective is

\[
L_{\rm FB}=\mathbb E_{xa}\mathbb E_{j\sim\rho}
  (F_{xa}^{\top}B_j-4P_{xa,j})^2
  =\|\widehat P-P\|_F^2.
\]

This is the density-ratio squared objective analyzed in the original
[FB supplement, Appendix B.4, equations 36–37](https://proceedings.neurips.cc/paper_files/paper/2021/file/003dd617c12d444ff9c80f717c3fa982-Supplemental.pdf).
Here it is also the population TD objective, up to an additive sampling-variance
constant: with zero discount there is no bootstrap target.

Write \(c_1=.1x+.01a\), \(c_3=-.005x+.05a\), and \(s_i=2e_i\). Uniform rows
give \(\mathbb E[c_1c_3]=0\),
\(\mathbb E[c_1^2]=.0101>.002525=\mathbb E[c_3^2]\).
The singular values of \(P\) are \(1,\sqrt{.0404},\sqrt{.0101},0\).
Consequently its unique best rank-two approximation is

\[
\widehat P^{\rm FB}_{xa}=p_0+c_1e_1,
\qquad L_{\rm FB}^{*}=.0101.
\]

It is realized by \(B_j=(1,s_{1j})\), \(F_{xa}=(1,2c_1)\), with
\(\|B_j\|=\sqrt2\) and \(\mathbb E_\rho[BB^\top]=I\). Backward norm and
covariance constraints therefore preserve this optimum. Its goal action gap is
\(+.01\), so both decisions are wrong.

**The dimension can preserve all control information.** Set

\[
B=\frac1{\sqrt{26}}\begin{pmatrix}6&4\\-6&4\\-4&6\\4&6\end{pmatrix},
\qquad F_{xa}=\left(\frac{a\sqrt{26}}{50},\frac{\sqrt{26}}5\right).
\]

These satisfy the same backward constraints. Their positive, unit-mass
prediction is \(\widehat P^{\rm ctrl}_{xa}=(.2,.2,.3,.3)+aw\), hence

\[
\widehat P^{\rm ctrl}_{x,+}-\widehat P^{\rm ctrl}_{x,-}
=2w=P_{x,+}-P_{x,-}.
\]

For every reward vector, exact reward integration therefore produces the true
action-value difference. Native FB readout is identical:
\(F^\top z_r=\widehat P r\), with \(z_r=\mathbb E_\rho[Br]\).
But this feasible solution has \(L_{\rm FB}=.0801>.0101\). The objective prefers
discarding useful action information despite its representability.

**A categorical certificate.** A normalized successor critic uses the target
\((1-\gamma)e_{s'}+\gamma\bar q(\cdot\mid s',a')\); its actor maximizes the
predicted expected reward. This is the finite-outcome form of the successor
cross-entropy objective in
[Temporal Difference Flows, equations 2–4](https://openreview.net/pdf?id=j6H7c3aQyb).
At \(\gamma=0\), minimize \(H(P)+\mathbb E_{xa}\mathrm{KL}(P_{xa}\|q_{xa})\).
The following candidate has dimension-two logits:

\[
q_{xa,j}=\tfrac14(1+2c_1s_{1j})(1+2c_3s_{3j}),
\quad B_j=(s_{1j},s_{3j}),\quad
F_{xa}=(\operatorname{atanh}(2c_1),\operatorname{atanh}(2c_3)).
\]

It obeys the same backward constraints, satisfies
\(q_{xa}-P_{xa}=.00495xa(s_1\odot s_3)\), and has average KL
\(0.0002064994354\).
To certify every categorical optimum, let \(p_+<p_-\) be the true goal
probabilities at a context. Coarsening each outcome distribution into
goal/non-goal cannot increase KL. A wrong or tied decision requires goal
predictions \(u\ge v\). Convexity gives

\[
\min_{u\ge v}\{\mathrm{kl}(p_+\|u)+\mathrm{kl}(p_-\|v)\}
=\mathrm{kl}(p_+\|m)+\mathrm{kl}(p_-\|m),\quad m=(p_++p_-)/2.
\]

The minimum occurs at the tie \(u=v=m\). Including row weights \(1/4\), any
wrong decision costs at least \(0.0004742086049\) for \(x=+1\), or
\(0.0006318215591\) for \(x=-1\). The feasible candidate beats both bounds;
therefore every global categorical optimum chooses correctly in both contexts.

**Scope and loss control.** Both families misspecify the full distribution:
\(P\) and its row-centered log-probability matrix have rank three. The candidate
\(q\) has rank-two logits but rank-four probabilities. In fact, squared loss
over this softmax class also succeeds: its feasible error is
\(16(.00495)^2=.00039204\), whereas any wrong goal ordering requires at least
\((.04)^2/2=.0008\) from those two goal entries alone. Thus this example does
not isolate cross-entropy from model geometry.

Zero discount removes the distinction between exact-target approximation and
population TD here. It proves neither a multistep manipulation failure nor a
guarantee for a learned actor or the default latent reward encoder. Control uses
exact reward integration. Reproduce the algebra and bounds with
[the verification script](../../scripts/didactic/verify_objective_counterexample.py).

A [separate loss-only construction](2026-09-10-loss-only-counterexample.md)
compares KL and squared error within the same stochastic rank-two class. That
comparison uses an independently chosen uniform reference and unconstrained
backward outputs; those assumptions differ from this construction's natural
reference and fixed backward norms.
