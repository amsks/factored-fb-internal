---
name: rldp-ortho-dead-regularizer-20260831
description: 2026-08-31 ExoRL walker RESULT -- the official RLDP repo's ortho regularizer contributes zero gradient; running it live at the published coef 1.0 recovers 62-94% of published (from 14-42%)
metadata:
  type: project
---

The official RLDP repository sets `ortho_coef = 1.0` and then contributes **no gradient**
from it (`orth_loss_diag` is constant `d` under `b.norm=True`; the off-diagonal term is
multiplied by zero). Runs 1 and 2 replicated that faithfully, which is why they never
reproduced the paper's numbers.

**Walker, 5M, n=4, full protocol (50 episodes, 50k inference samples):**

| arm | ortho | eff.rank/512 | stand | walk | run | flip |
|---|---|---|---|---|---|---|
| run 2 | inert (official-literal) | 26 | 365 | 107 | 67 | 125 |
| `rldp_orthoa` | live, coef 1.0 | 181 | 828 | 490 | 225 | 347 |
| `rldp_orthob` | live, coef 1e-4 | 152 | 148 | 41 | 25 | 28 |
| RLDP Table 1 | | | 878 | 791 | 325 | 492 |

**Three findings that overturned my own earlier reasoning — do not re-derive them:**

1. **Effective rank does NOT predict control.** `orthob` de-collapsed to 152 dims (vs 26)
   and is the *worst* arm, below the inert baseline. Rank proves the collapse is real;
   it says nothing about whether the policy works.
2. **The ortho/SP gradient-balance ratio is not the operative variable.** I built `orthob`
   at a balanced 1.09 to mimic td_jepa's working 0.56 and called `orthoa`'s 1.09e4
   pathological. `orthoa` is the arm that works. Do not design arms from that ratio.
3. **Collapse did NOT cause the action-blind critic.** `orthob` de-collapses and still
   ranks actions at chance. Two real but separate problems.

**Cheap screen that worked:** `scripts/rldp_action_sensitivity.py` sign test
`P[Q_dataset > Q_negated]` at 2.5M called the final 5M ordering exactly, with no rollouts
(orthoa .614/.646 above chance; orthob .465/.490 and run 2 .493 at/below).

**Still open:** gate not passed — `walk` at 0.62 of published with 3x the published seed
spread; cheetah/quadruped unrun for these arms. HILP staged but not launched.

Related: [[exorl-validation-campaign-20260828]]
