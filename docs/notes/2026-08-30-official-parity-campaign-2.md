# ExoRL official-parity campaign, run 2 (`rldp_ortho_inert: true`) — GATE STILL FAILS

Rung 2 of the spec sec.4 ladder. Run 1 (paper Eq-7 ortho) failed 0/8 with a measured
cause: phase 1's ortho gradient exceeded the self-predictive gradient by 6.1e4, so B never
learned dynamics. Run 2 flips to the code-exact inert ortho, making phase 1 SP-only.

Verdict: **the fix is real and measurable, and it is not sufficient. Gate fails 0/8.**
Walker improved on every task; cheetah did not improve and on balance regressed.

## 1. What the fix definitively achieved

Three independent confirmations, all pre-registered before the run:

| check | run 1 | run 2 |
|---|---|---|
| phase-1 ortho/SP gradient ratio | 6.1e4 | **1.7e-5** |
| phi predictive vs no-motion baseline | not measured | **0.031** (32x better than static) |
| B collapse (per-dim std / participation ratio) | not measured | **0.628 / 15.3 of 512** (not collapsed) |
| phase-2 `fb_loss` at step 2,045,000 | 13,330 (doubling every ~5k) | **234.7** (flat 120-256) |

The phase-2 divergence is gone: 57x lower at the same offset, and flat across the full 3M
rather than exponential. Phase 1 now learns a genuinely predictive representation.

## 2. The table (10 episodes, matching run 1's so the two are comparable)

| domain | task | run 1 | RUN 2 | random floor | official RLDP | r2/official |
|---|---|---|---|---|---|---|
| walker | stand | 226.0 | **421.3 +- 145.4** | 126.16 | 877.69 +- 45.03 | 0.48 |
| walker | walk | 99.2 | **126.5 +- 93.3** | 33.92 | 790.94 +- 67.55 | 0.16 |
| walker | run | 38.5 | **80.4 +- 23.2** | 22.65 | 324.85 +- 54.57 | 0.25 |
| walker | flip | 90.2 | **144.0 +- 76.3** | 50.07 | 491.64 +- 37.30 | 0.29 |
| cheetah | run | 16.6 | **14.0 +- 21.9** | 3.68 | 236.31 +- 20.75 | 0.06 |
| cheetah | run_backward | 20.3 | **3.4 +- 1.7** | 20.15 | 322.08 +- 39.28 | 0.01 |
| cheetah | walk | 104.0 | **64.8 +- 96.0** | 18.39 | 895.31 +- 49.84 | 0.07 |
| cheetah | walk_backward | 84.7 | **17.5 +- 10.3** | 100.75 | 984.76 +- 0.85 | 0.02 |

walker n=3 (walker-s1 lost ~1.3M steps to a CUDA fault and finished late; re-score when it
lands), cheetah n=4.

## 3. The domains split, and that is the finding

**Walker improved on all four tasks (1.3-2.1x) and is now 3-4x its random floor on every
one.** In run 1 walker was within noise of the floor on 3 of 4. So the ortho fix bought
real representation quality — the agent is now clearly doing something — just not enough
to approach the published cells.

**Cheetah did not improve and mostly regressed**, and two tasks are BELOW their random
floor: `run_backward` 3.4 vs floor 20.15, `walk_backward` 17.5 vs floor 100.75.

A snapshot of the training task alone would have missed this: walker's mean across the
first finished cells looked indistinguishable from run 1, and only the 4-task table showed
the consistent walker gain.

## 4. Hypothesis for cheetah, NOT established

The two worst cells are the two BACKWARD tasks, and they are the two that require behavior
opposite to the tasks sharing their reward structure. Cheetah is also the domain RLDP
Table 4 labels SPARSE while walker is DENSE, which matters for a reward-moment readout
z = E[r B]: with rewards near zero almost everywhere, the moment is dominated by a handful
of rows and is correspondingly noisy. Either could explain a domain-specific failure that
a representation fix does not touch. Neither is tested.

## 5. Where the ladder goes

Rung 2 fails, so the remaining recorded ambiguities (spec sec.4) are:
  * `steps-3M` — the CODE default is 3M representation + 3M FB
    (examples/rldp_train_dmc.py:142); this campaign used the paper text's 2M+3M. This is
    the largest single unexplored knob and the cheapest to justify.
  * eval-50 vs eval-10 — will not move a 0.02 ratio.
  * pointmass gamma 0.99 — pointmass was not run in either campaign.

Outside the recorded list, one measured observation deserves a look before more compute:
`bc_log_prob` falls monotonically through phase 2, reaching -1.7e8 on walker-s0. Official
RLDP's actor is unanchored TD3 (alpha=0, no BC), so this is expected rather than a bug,
but it means the policy drifts arbitrarily far from the behavior data — and an offline
method whose actor leaves the data distribution is a plausible cause of exactly this
pattern (representation fine, returns poor, worse on the sparser domain).
