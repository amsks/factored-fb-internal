---
name: wandb-group-naming-convention
description: wandb group naming for factored-fb runs; torch FB-family pixel = <method>_pixel; how FB-pixel runs got lost
metadata: 
  node_type: memory
  type: project
  originSessionId: 86447647-071e-49fb-ae40-21a8ab6b35cd
---

wandb group naming convention for the `amsks/factored-fb` project (see [[wandb-entity-amsks]]).

**PIXEL runs (renamed 2026-06-09) follow `<Method>-Pixel-<Encoder>-<Actor>`:**
- `FB-Pixel-DrQ-Flow` (5), `PSM-Pixel-DrQ-Flow` (10), `RLDP-Pixel-DrQ-Flow` (10) — torch FB-family
- `GCIQL-Pixel-DrQ-DDPGBC` (5), `GCIQL-Pixel-IMPALA-DDPGBC` (5)
- `GCIVL-Pixel-DrQ-AWR` (5), `GCIVL-Pixel-IMPALA-AWR` (13)

Token rules: Method = FB/PSM/RLDP/GCIQL/GCIVL; Encoder = DrQ or IMPALA (`impala_small`→IMPALA); Actor = Flow (flowbc agents) / DDPGBC (gciql) / AWR (gcivl, uses alpha+expectile). CRL and onestep-FB have NO pixel runs.

**STATE baselines + FB variants renamed (2026-06-09) to `<Method>-State-<Actor>[-Variant]`:**
`FB-State-Flow` (plain), `FB-State-Flow-L256`, `FB-State-Flow-GoalCond` (v1), `FB-State-Flow-USFA` (v2), `FB-State-Flow-EncMatch`, `PSM-State-Flow`, `RLDP-State-Flow`, `CRL-State-Flow`, `OneStepFB-State-Flow`, `GCIQL-State-DDPGBC`, `TDMPC2-State-Planner`.

**HP-sweep / ablation groups renamed (2026-06-09) to `<Method>-State-Flow-<Category>-<HPvalues>`:** category token = Sweep / FBmatch[-LrCheck] / Tuned / Pess0; then ONLY that run's own single HP values (NEVER ranges) — e.g. `PSM-State-Flow-Sweep-Ortho1-LrPhi1e-5`, `PSM-State-Flow-FBmatch-Tau0.01-LrPhi1e-4`, `PSM-State-Flow-Pess0-LrPhi1e-5`, `RLDP-State-Flow-Tuned-Ortho0.001-LrB1e-5`. HP token format: ortho_coefXpY→OrthoX.Y, lr_phi→LrPhi, lr_b→LrB, target_tau→Tau. Category tokens ARE kept (disambiguate Sweep vs FBmatch sharing ortho1/lr).

**Still NOT renamed (left as-is):** other-domain groups (`antmaze-*`, `covbal_*`, `cube-single-ortho*`, `cube-single-onestep-*`, `cube-single-video-replay-*` — mostly killed/crashed, not finished), `psm_isolation_300k`.

**Pending (rename once finished, same scheme):** the 3 RUNNING groups → `psm_state_tuned__ortho_coef1__lr_phi1e-5`→`PSM-State-Flow-Tuned-Ortho1-LrPhi1e-5`; `psm_state_orthohi__ortho_coef100__lr_phi1e-5`→`PSM-State-Flow-OrthoHi-Ortho100-LrPhi1e-5` (and ortho1000 likewise).

**NEVER renamed running groups** (rule: skip any group with a `running` run, even mixed): as of 2026-06-09 those were `psm_state_tuned__ortho_coef1__lr_phi1e-5` (mixed 4 running/6 finished), `psm_state_orthohi__ortho_coef100__lr_phi1e-5`, `psm_state_orthohi__ortho_coef1000__lr_phi1e-5`. Rename them only once finished.

2 dead step=1 visual junk runs remain in the old `factored-fb-gciql` group — intentionally NOT folded into `GCIQL-State-DDPGBC`.

**Why this matters:** The 5 FB-on-pixels runs (`visual-cube-single__fb_flowbc__s{0..4}__full1M_amsks`, DrQ, ~1M, final success ~0.352±0.065) were launched with **`group=None`** and were invisible to any group-keyed survey → caused a wrong "no FB pixel runs exist" conclusion before being regrouped.

**How to apply:** Always set `wandb_group` at launch. When auditing "what did we run", group-by-`r.group` MISSES ungrouped runs — also scan by `config.agent` + `config.obs_type`. NB: 2 dead step=1 visual runs still sit (untouched) inside the `factored-fb-gciql` STATE group — junk, ignore them.
