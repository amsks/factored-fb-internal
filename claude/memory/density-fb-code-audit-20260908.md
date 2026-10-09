---
name: density-fb-code-audit-20260908
description: 2026-09-08 full code+results audit of the seam density-fb branch: training path CLEAN (loss, TD target, actor reward, index, resume with opt_state); 766 pass / 5 stale fails; 2 eval bugs (density_fb goal-arm Route B zero-goal; HILP-family Route A goal-blind) + make_results_table.py::score_run scores the LAST TASK not the task mean (unused for tdjtab cells); per-cell verdict table in scratchpad report
metadata:
  type: project
---

**Fact.** Audit (seam worktree, HEAD 6cc2952 + uncommitted) found no defect in Density-FB training. Tests on CPU jax: 766 passed, 5 failed (stale `update_stats` noop assertion; missing `hilp_density.yaml`; `density_fb x ddpgbc_density_cpr` raises KeyError not ComboError), 8 ExoRL env errors (box mujoco pairing). Clean cells recomputed: privcoord Density-FB .910 state / .943 / .053 / .640; tdjtab fb_tdj cube-double 1.6, rldp 2.2, density_fb Route A 68.6, density_fb_z 74.9/81.3 and 33.9/30.4. INVALID: density_fb Route B 11.1/8.9 ([[density-fb-goal-arm-route-b-zero-goal-20260908]]); all HILP-family Route A cells incl. privcoord HILP/Density-HILP rows ([[route-a-goal-blind-bug-20260908]]); Density-HILP .74 (Route B) stands.

`scripts/make_results_table.py::score_run` keys on step only, so per-task OGBench eval.csv rows overwrite each other (cube-double density_fb: .686 correct vs .52 wrong) and KNOWN_CRITICS drops density_fb/rldp/hilp/fb_tdj. tdjtab cells used a correct scratch scorer, so no reported number is affected; fix + test added 2026-09-08 (uncommitted). Unconfirmed: `main.py` overwrites config.json before restore, so a ckpt resumed with a different `measure_density_hard` would silently switch target; Route B relabel draws share the global NumPy RNG with GCDataset.

**Why:** the user asked "is our code bug free / are Density-FB results corruption free" before deciding how to fix underperformance.

**How to apply:** trust Route A Density-FB and all density_fb_z numbers; quote Route B only from critics with a live index (inferred_cond_norm = sqrt(50)); do not cite make_results_table.py as the last-3 estimator until the fix is committed. Full report: session scratchpad report_codeaudit.md (2026-09-08).
