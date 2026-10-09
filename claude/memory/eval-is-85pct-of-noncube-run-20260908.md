---
name: eval-is-85pct-of-noncube-run-20260908
description: 2026-09-08 luis measured 16.7 h of eval per non-cube 1M run (85% of the run, 9 of 11 passes never read); eval is MuJoCo-physics-bound not GPU-bound, so GH200 ≈ RTX 3090 on scene/puzzle/antmaze
metadata:
  type: project
---

**The dominant cost of the tdjtab campaign is eval, not training, and most of it is thrown away.**

luis measured on scene-play-v0 (RTX 3090, protocol settings `eval_episodes=20`,
`eval_relabel_episodes=10`, `eval_interval=100000`, 5 tasks): Route-A pass **61 min**, Route-B pass
**30 min** ⇒ 91 min per full pass × 11 passes = **16.7 h of eval per non-cube 1M run**, against ~3 h
of training. **Eval is 85% of the run and 9 of the 11 passes are never read** — the table quotes the
final step and Route B is final-step-only by the spec's gate 2.

Every earlier estimate in the campaign doc was wrong on this: the "5–8 h/run" budget for
scene/puzzle/antmaze was extrapolated from cube runs (200-step episodes) and misses ~3× for the
non-cube envs (scene 750, puzzle 500, antmaze 1000 steps per episode).

| eval schedule | non-cube run | 98-run campaign | makespan @44 slots |
|---|---|---|---|
| `eval_interval=100000` (current) | 19.7 h | 1753 h | 40 h |
| final step only | 6.0 h | 607 h | **14 h** |

Proposed (posted for challenge, NOT yet committed): `--eval_interval 1000000`, keep
`--save_interval 250000`. Table cells still produced in-run at 1M on both routes; curves become a
post-hoc rescore (`scripts/relabel_rescore.py --exp-root exp/tdjtab`). Open: does any table/figure
read a non-final eval row; does the step-1 eval fire regardless of `eval_interval`.

**Corollary — eval is MuJoCo-physics-bound, NOT GPU-bound.** vista: **32.8 s/episode on a GH200**.
luis: **36 s/episode on an RTX 3090**. Same number. A GH200's ~1.4× training advantage applies only
to the ~15% of a non-cube run that trains. So "biggest block to the fastest GPU" is the wrong rule:
luis (36 concurrent, no SU accounting, **PreemptMode=OFF** so the other project cannot evict us) takes
the 84 non-cube runs; evalbox takes the 20 eval-light cube cells + post-hoc rescoring (192 threads,
CPU-bound eval); vista takes **cube only** — its advantage is real there and wasted on non-cube.
This inverts the user's vista>luis>evalbox priority; flagged, not overridden.

**Vista SUs ENCUMBER, they do not just spend** (`ASC26008`, partition `gh`, 1 GPU/node): a submission
is refused if requested walltime × nodes summed over everything queued exceeds the balance, so a 3 h
run at `-t 12:00:00` locks 12 SU. Hence "~260 SU compute, ~960 SU encumbrance at 12 h wall".
**Balance 17 SU** (2026-09-08 00:10Z), down from 89 against only ~7.6 node-hours run — unreconciled
72 SU drop, vista flagged rather than guessed. Buys ~3 cube-single runs. Expiry 2027-01-07.

Both sites independently confirmed **98 to launch + 6 already at 1M = 104** ⇒ the 6 cube `density_fb`
resumes are DONE. Soft arms (32 runs) cost ~184 h under the cut schedule ≈ 4 h extra makespan, so
136 becomes nearly free if the eval cut lands. Detail in §10 of the campaign doc.
See [[three-site-topology-20260907]], [[tdjtab-delivery-and-evalbox-capacity-20260907]].
