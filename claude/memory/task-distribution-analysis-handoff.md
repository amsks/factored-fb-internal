---
name: task-distribution-analysis-handoff
description: "DONE — FB-vs-PSM training task-distribution analysis: built, run on all 20 seeds; PSM resolves the interaction variable far better than FB"
metadata:
  node_type: memory
  type: project
  originSessionId: 6787d218-60cd-4a6d-ad25-71a042116338
---

**DONE (2026-06-08).** Built + run on all 20 seeds (fb_state_plain ×10, psm_flowbc_state ×10).
Spec: `PAPER/specs/2026-06-07-fb-vs-psm-task-distribution-analysis.md`. Code:
`evals/task_distribution.py` (pure logic, 8 unit tests in `tests/test_task_distribution.py`)
+ `scripts/task_distribution_probe.py` (driver, `--quick`/`--m`). Outputs:
`analysis/task_distribution/{fb_state_plain,psm_flowbc_state}/s*.json` + `aggregate.json` +
`summary.md`.

**Headline results:**
- **Probe 1 (shared goal pool, 50k next-states):** 45% cube-involved (grasped 10.5% + lifted
  34.6%), 55% arm-only. Both methods see the SAME goal distribution (same buffer, uniform
  sampling, 50/50 random/relabel mix; FB `z=project_z(B(s'))` dim 50, PSM `z=project_z(phi(s'))`
  dim 128) — so any difference is REPRESENTATIONAL, not data access.
- **Probe 2:** separability AUC of cube-involved vs free in z = **PSM 0.94 vs FB 0.75**. PSM's
  phi resolves the interaction variable far better; FB's B is near-full-rank (PR 48.9/50, from
  ortho_coef=1000) but interaction-blind. PSM PR 68.5/128.
- **Probe 3:** eval tasks moderately OOD for both; **FB per-task coverage tracks per-task
  success** (task1 worst cos 0.46 / maha 35.8 → 0% success; task5 best → 80%).

**Key impl gotcha (fixed):** the handoff spec's obs layout was WRONG — obs is NORMALISED, so
effector (obs[12:15]) and cube must be compared IN THE OBS FRAME (cube at obs[19:22], contact
obs[18]); lift uses physics cube-z (real metres, physics[14:17]). Mixing obs-effector with
physics-cube gives garbage distances. Dataset domain string is the resolved env id
`cube-single-play-v0`, not `cube_single`. Run analysis with `PYTHONPATH=. .venv/bin/python`.
See [[fb-state-run-name-map]], [[encmatch-ablation-confound]].
