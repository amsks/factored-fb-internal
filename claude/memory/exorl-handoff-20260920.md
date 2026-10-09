---
name: exorl-handoff-20260920
description: "2026-09-21 ExoRL kernel arms RESOLVED at 5 seeds: quadruped_vu 760.5±20.4 REPLACES quad cell (>all published), walker_hv 773.5±7.8 HELD OUT (+23 within ~30 spread, walker stays 750.2); table+hparams updated, mean now 700.5±23.1 vs FB 648.2; Fig3 y-axis shortened + 600dpi, Fig4 y-axis Gr/Gr->'recovery'"
metadata: 
  node_type: memory
  type: project
  originSessionId: a7ce80c8-00fe-40c1-9170-e275620481cc
  modified: 2026-09-21T07:54:22.353Z
---

Handoff note: docs/notes/2026-09-20-exorl-handoff.md (full status, recipes, closed questions, commands).
Full history of the ExoRL job: [[exorl-cheetah-maze-disc-20260917]], [[exorl-audit-20260917]].

Table state (tab:benchmarks ExoRL block, per-domain recipes per user ruling 2026-09-20): walker 750.2±29.8
(round-3 `sqt` recipe), cheetah 539.9±42.0 (round-3 + velocity-block hit kernel 8:17:7.264), quadruped
655.9±15.9 (1M recipe), point-mass 751.5±74.1 (1M recipe, actor lr 1e-6); mean 674.4±22.9 vs FB 648.2.
Table also got benchmark means on the header rows (2026-09-20 ~20:30Z); DMC mean-row spreads corrected to
propagated values (FB 5.8 / HILP 10.0 / RLDP 16.0).

OPEN 2026-09-20 ~21:00Z: reward-relevant hit-kernel arms (scripts/launch_exorl_kernels.sh, exp/exorl_kernels/,
seeds 0-2). walker_vel / quadruped_vel / quadruped_vu done at 1M; walker_hv at 750k (GPUs 3-5, ~1 h).
In-run 1M evals (not paper protocol): quadruped_vu 806/732/688 (vs cell 656), quadruped_vel ~504, walker_vel
~440 (loss), walker_hv @750k 743/845/884 (vs cell 750). Paper scoring launched: tmux `exorl_kernels_score`,
GPUs 0-2, new scripts/score_exorl_kernels.sh <arm> "0 1 2" <gpu> -> exp/paper_eval_exorl_kernels_<arm>.jsonl
(passes the arm's hit spec override; scorer otherwise forces the default spec). NEXT: score walker_hv when
JOB_DONE appears in logs/exorl_kernels/walker_hv_s*.log; if an arm beats its cell by > seed spread at 3 seeds,
run seeds 3,4, rescore, replace the cell + app:exorl-hparams line.

RESOLVED 2026-09-21: paper-protocol 5-seed scores (exp/paper_eval_exorl_kernels_<arm>.jsonl, all 5 seeds each):
quadruped_vu 760.5±20.4 (jump 785.9 / run 527.1 / stand 972.8 / walk 756.2) PASSES (+105 >> ~16 spread) ->
REPLACED the quadruped cell; now beats FB/HILP/RLDP outright. quadruped_vel 580.8 loss. walker_hv 773.5±7.8
and walker_vel 561.6 loss: walker_hv wins the mean but +23 < ~30 spread and error bars overlap (~1.7 sigma),
so by the pre-registered rule walker STAYS 750.2 (round-3 recipe). tab:benchmarks: quadruped 655.9->760.5 (bold,
HILP unbolded), ExoRL mean 674.4->700.5±23.1; app:exorl-hparams quadruped clause rewritten (hit kernel cols
44-46 r0.11 + col 47 r0.001). Figs regenerated: analysis/single_bottleneck/fig3_family.py (Fig 3, run with arg
`2` for rank2; y-labels "value / exact" + "visits / exact", dpi 600) and run.py --figures-only (Fig 4
fig_sb_ablation_trained y-label now "recovery"). All UNCOMMITTED on density-fb.

**Why:** the cheetah win came from restricting the hit kernel to reward-relevant observation columns; the same
lever holds on quadruped (torso velocity + upright) but not walker at five seeds.
**How to apply:** treat in-run evals as direction only; only paper-protocol five-seed cells go in the table;
respect the "beat by > seed spread" rule even when the mean is higher (walker_hv is the example that failed it).
