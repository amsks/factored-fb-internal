---
name: tdjepa-vs-rldp-published-20260831
description: TD-JEPA (arXiv 2510.00739) Table 1 reproduces RLDP's own numbers (1.07x) but gets HILP 2.5x HIGHER than RLDP's baseline column -- RLDP under-reports its baselines and its ranking flips
metadata:
  type: reference
---

TD-JEPA paper: https://arxiv.org/abs/2510.00739 (code: `~/git/baselines/td_jepa` @ 840a745).
Its Table 1 is ExoRL RND, same data as RLDP's Table 1, **per-domain averages over 4 tasks**.
Reports mean +- **standard error** (RLDP reports **std** -- error bars are NOT comparable).

**Domain means, RLDP's paper vs td_jepa's reimplementation:**

| method | walker | cheetah | quadruped | mean(3) |
|---|---|---|---|---|
| RLDP paper's RLDP | 621.3 | 609.6 | 612.8 | 614.6 |
| td_jepa's RLDP | 723.9 | 575.6 | 665.0 | **654.8 (1.07x)** |
| RLDP paper's HILP | 348.1 | 193.5 | 289.8 | 277.1 |
| td_jepa's HILP | 796.4 | 618.3 | 694.8 | **703.2 (2.54x)** |
| RLDP paper's FB | 594.7 | 586.3 | 568.6 | 583.2 |
| td_jepa's FB | 811.5 | 672.7 | 595.6 | **693.3 (1.19x)** |

**RLDP's own method reproduces (1.07x); its BASELINES do not.** HILP is 2.5x low, FB 1.2x low.
Not a task-set artifact: cheetah and quadruped have *identical* task lists in both, and still
show HILP 3.19x and 2.40x. (Walker differs -- td_jepa uses `spin`, RLDP uses `flip`, verified
different rewards in `metamotivo/envs/dmc_tasks/walker.py`: spin = angmomentum>=10 alone;
flip = angmomentum>=5 combined with standing.)

**The ranking flips.** td_jepa full average (4 domains): TD-JEPA 661.2 > FB 648.2 >
BYOL-g 645.4 > HILP 620.1 > ICVF 619.3 > BYOL 618.6 > **RLDP 610.2** > Laplacian 591.1.
RLDP is second-to-last there; in its own paper it wins. Its headline claim rests on weak
baselines.

**Consequence for our gates:** never gate HILP against RLDP Table 1's HILP column -- it is
2.5x low. Use HILP's own paper or td_jepa. Also note td_jepa numbers are best-of-sweep
(lr_b x ortho_coef, 6 configs x 5 seeds) and use 10 eval episodes / 10k inference samples
at z_dim 50, vs our 50 episodes / 50k samples at z_dim 512.

Related: [[rldp-ortho-dead-regularizer-20260831]], [[exorl-validation-campaign-20260828]]
