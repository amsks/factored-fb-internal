# Handoff to luis: baseline seeds 3,4 for the OGBench main table (FB / HILP / RLDP)

Prepared on evalbox 2026-09-18. Purpose: bring the three baseline rows of the paper's main
table (PAPER/ICLR/content/experiments.tex, tab:manipulation, OGBench block) from three seeds
to five, matching Density-FB, whose seeds 3 and 4 finished on evalbox this week
(density_fb_z and density_fb_z_state on cube-single, cube-double, scene, puzzle-3x3; see
docs/notes/2026-09-18-ogbench-five-seed-aggregate.md). Nothing else in the table changes.

## What to run
File: `handoff/baseline_seeds34/commands.txt` -- 24 commands, one per line, each expects
`$GPU` set (same convention as scripts/tdjtab_manifest.sh / scripts/launch_tdjtab.sh).
    3 critics (fb_tdj, hilp_tdjc, rldp) x 4 envs (cube_single, cube_double, scene_play,
    puzzle_3x3) x seeds {3, 4}, 1M steps, eval every 100k, Route-B relabel eval in-run
    (eval_relabel_size=10000, shift=1.0, episodes=10). No code changes needed; the commands
    are the manifest lines for the same critic/env with --seed 3/4.
Per-run cost is the tdjtab figure: cube-single ~3.5 h, cube-double ~7 h, scene/puzzle ~4-5 h
at one run per GPU.

Where they land (deliberately two roots, so each seed sits beside seeds 0-2 of the SAME arm):
  * fb_tdj and hilp_tdjc: `--save_dir exp/tdjtab` -> exp/tdjtab/<env>-play-v0_<critic>_flowbc_seed_{3,4}
    (the density seeds 3,4 are there too). hilp_tdjc cube_double carries the cube-double hit
    spec override (measure_hit_spec=19:22:0.4,28:31:0.4) exactly as the seed 0-2 runs did; it is
    inert for training (backward_input=state) but keeps config.json identical across seeds.
  * rldp: `--save_dir exp/ortho1 --override ortho_coef=1`. The RLDP row in the main table is the
    ortho_coef=1 arm (RLDP's own default, exp/ortho1, launched by scripts/launch_ortho1.sh), NOT
    the manifest's ortho_coef=0.01 arm that sits under exp/<env>_rldp_flowbc_seed_{0,1,2} on the
    hub. Seeds 3,4 must carry the same coefficient, and putting them in exp/tdjtab would make
    them indistinguishable by path from the 0.01 arm. If you prefer a single root, change the
    save_dir but keep `--override ortho_coef=1`; config.json records the coefficient either way.

## Idempotence
Every command carries --auto_resume: re-running after a fault resumes from the latest
params_*.pkl in the run dir, and a run that already has params_1000000.pkl is a no-op for
training (eval only re-runs if you ask for it). Safe to resubmit the whole file.

## Report back
Either status/<run-dir>.json per run (scripts/tdjtab_handback.py --run <dir> writes the status
file and uploads params_1000000.pkl, config.json, eval.csv, eval_relabel.csv to
amsks/factored-fb-checkpoints; for the rldp runs pass the exp/ortho1 dir), or a mailbox post
naming which of the 24 finished. The table cell is the mean over the last three eval passes of
eval_relabel.csv (make_results_table.py:score_run), so eval_relabel.csv is the file that matters.

Note: commands.txt is also in the git bundle handoff/rebalance/factored-fb-rebalance.bundle (path handoff/baseline_seeds34/commands.txt).
