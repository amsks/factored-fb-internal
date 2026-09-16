# Density-FB 5-seed campaign — ExoRL then OGBench (2026-09-16, overnight)

Goal: the paper's Density-FB columns at 5 seeds on both ExoRL and OGBench. RLDP / HILP / FB
numbers come from the TD-JEPA paper, so only Density-FB is trained here.

Driven by one scheduler in tmux `dfbsched`:
`logs/exorl_dfb/scheduler.py` reads `logs/exorl_dfb/manifest.tsv`, runs 2 jobs per GPU on 8
L4s (16 slots), re-reads the manifest at each phase boundary, skips any job whose final
checkpoint already exists (every job passes `--auto_resume`), retries a crashed job twice.
Status: `logs/exorl_dfb/scheduler_status.txt`. Per-job logs: `logs/exorl_dfb/<name>.log`.
Job scripts (one file each, so the hit-spec keeps its quotes): `logs/exorl_dfb/jobs/`.

## Phase 1 — ExoRL Density-FB (running now)
`density_fb_z`, 2M steps, 5 seeds, 4 domains = 20 runs. Config = the post-FB-audit walker
reference recipe (value tower [1024], tau 0.001, B [256,256] Motivo normalized, lr 1e-4,
Motivo TD3 actor, alpha 0, q_normalize off) plus the reward-inference eval
(`--infer_cond_episodes 50000 --eval_on_inferred_cond`) — the TD-JEPA protocol, NOT a goal
Dirac. (The earlier `exp/walker_dfbref` runs evaluated on z=0, so their returns are not the
comparable number; that is why the infer flags are mandatory here.)

Per domain: discount, hit-spec (calibrated — random-pair hit rate matched to cube-single 0.0133):
- dmc-walker            0.98   0:24:15.9
- dmc-cheetah           0.98   0:17:7.5
- dmc-quadruped         0.98   0:78:21.087   (calibrated tonight; block was blocked before)
- dmc-point-mass-maze   0.99   0:2:0.03

Save dir `exp/exorl_dfb/<domain>/density_fb_z`. Final eval for the paper number is the
50-episode paper-exact eval (`scripts/exorl_paper_eval.py`), a separate rescore after training.

## Phase 2 — OGBench Density-FB top-up (auto-starts when Phase 1 is fully done)
Top both existing arms up to 5 seeds (adds seeds 3,4), 4 manipulation envs, flowbc, 1M,
per-env object-coord hit-spec, `measure_density_hard`, `latent_centered`, Route-B relabel eval
= 16 runs. Faithful copies of `scripts/tdjtab_manifest.sh` / `scripts/launch_dfb_state.sh`,
only the seed changed.
- with object coordinates:    `density_fb_z`        -> `exp/tdjtab`   (backward tower reads the object-coord block)
- without object coordinates: `density_fb_z_state`  -> `exp/dfb_state` (backward tower reads full state)
Envs: cube_single, cube_double, scene_play, puzzle_3x3 (antmaze excluded — out of scope).

CHECK ON WAKING: (a) the with/without object-coordinate mapping above matches intent;
(b) cube `density_fb_z_state` seeds 0–2 currently live under `exp/hub/exp/` (HF-synced), while
the new seeds 3,4 write to `exp/dfb_state` — the 5-seed aggregation for that cell must read
both locations. Correct now if wrong: `dfbsched` re-reads the manifest before Phase 2 starts,
so editing `logs/exorl_dfb/manifest.tsv` (or the job scripts) before Phase 1 ends is enough.

## Watch / control
- `tmux attach -t dfbsched` ; `tail -f logs/exorl_dfb/scheduler_status.txt`
- Stop everything: `tmux kill-session -t dfbsched` then `pkill -f impls.main` (auto_resume
  continues from the last checkpoint on relaunch).
- Rough timing: ~3 h per 1M at 1 job/GPU; at 2/GPU expect Phase 1 ≈ 13–16 h, Phase 2 shorter (1M).
