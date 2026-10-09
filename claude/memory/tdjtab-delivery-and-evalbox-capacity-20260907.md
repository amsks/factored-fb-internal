---
name: tdjtab-delivery-and-evalbox-capacity-20260907
description: 2026-09-07 tdjtab campaign is DELIVERED (pushed @6cc2952, relabel blocker resolved, resume ckpts on HF); evalbox = 8x L4, ~35 runs/day; split proposed to cluster on the HF mailbox
metadata:
  type: project
---

The 167-run tdjtab campaign (TD-JEPA-protocol OGBench proprio table) is no longer blocked on
delivery. Three items from the 2026-09-07 prep handoff went stale within hours:

1. **Code is committed AND pushed** — seam `density-fb` @ `6cc2952` ("updated impls for large run
   deployment") == `origin/density-fb`, tree clean. Carries the campaign doc, 7 non-cube domain
   yamls, `scripts/tdjtab_manifest.sh` (167 ids) + `launch_tdjtab.sh`, non-cube relabel rewards,
   `hf_manifest.json`. The "no push credentials on this box" note is obsolete.
2. **Blocker §5 item 1 (Route-B relabel cube-only) is RESOLVED** — `env/ogbench.py` dispatches on
   task-info key (cube `goal_xyzs` / antmaze `goal_xy` / puzzle `goal_button_states` / scene `goal`);
   `tests/impls/test_ogbench_relabel_families.py` + `test_ogbench_relabel.py` = **40 passed**
   (verified, 73 s). `$RELABEL` need not be dropped on scene/puzzle/antmaze. Not yet smoke-tested
   in-run (unit tests cover the reward fns, not the eval path).
3. **All 12 resume checkpoints are on HF** (`amsks/factored-fb-checkpoints`): cube-double
   `rldp`/`rldp_density` seeds 0-2 have `params_500000.pkl`. Stronger: `density_fb` and
   `density_fb_state` on BOTH cube envs already have **`params_1000000.pkl`** from the cluster's
   24/24 Part B handback ⇒ 6 of the 12 resumes (+6 optional state) are likely already DONE —
   check `measure_density_hard: true` in their `config.json` before spending GPU-h.

**evalbox capacity** (this box, `ip-172-31-37-188`): 8 × NVIDIA L4 24 GB (all idle), 192-thread
2× EPYC 7R13, 728 GiB RAM, 287 GB free on `/`, `/dev/shm` 365 GB (venv + dataset mirror, volatile).
No scheduler — tmux only. All 9 table-env datasets present in `~/.ogbench/data/`. At 1 run/GPU and
~5.5 h/run: **≈35 runs/day, ≈190 GPU-h/day; all of tier 1 ≈ 4.7 days**. 2 runs/GPU untested.

Capacity + delivery status appended as **§7 of
`Factored-FB-seam/docs/notes/2026-09-07-tdjepa-table-needed-runs.md`** (uncommitted) and posted whole
to the HF mailbox as `mailbox/evalbox/2026-09-07T23-25-00Z.md` (+ scope update `...T23-46-25Z.md`) (`scripts/mailbox.py post --from evalbox`).
**SCOPE CUT 2026-09-07 23:40 UTC — antmaze NAVIGATE ONLY.** The backup dataset mirror
`zhouzypaul/ogbench_datasets` (flat, 44 npz at root) lacks antmaze-{medium,large}-stitch and
antmaze-medium-explore, so those 3 envs are dropped rather than shipped. Table = **6 envs**
(cube-single, cube-double, scene-play, puzzle-3x3, antmaze-medium-navigate, antmaze-large-navigate);
**tier 1 = 104 runs, not 167** (63 dropped); ≈550-600 GPU-h ≈ 3 days on evalbox's 8 GPUs. The 3 domain
yamls stay in the repo unused. Exclusion regex for every launcher call:
`^(antmaze_medium_stitch|antmaze_large_stitch|antmaze_medium_explore)/`.
Partition (§8.1, three disjoint blocks 36+20+48): **A evalbox** =
`(scene_play|puzzle_3x3|antmaze_(medium|large)_navigate)/(fb_tdj|hilp_tdjc|rldp)/s[0-2]`;
**B evalbox** = all 20 cube cells (12 of them resumes); **C cluster** = same 4 non-cube envs ×
`(density_fb|density_fb_z|hilp_density_tdjc|rldp_density)` when SUs land (cluster at 89 SU, needs ~260;
tiers 2-3 held). **Nothing launched on either side; the partition must be confirmed on
the mailbox first or run ids collide.** See [[tdjtab-campaign-prep-20260907]], [[partb-campaign-handoff-20260906]].
