---
name: handoff-bridge-10seed-1m-campaign
description: "READ FIRST on resume (2026-06-22): the 10-seed×1M bridge-table completion campaign is RUNNING in tmux bridge1m — status, output locations, and exact post-run steps to finish the paper table."
metadata: 
  node_type: memory
  type: project
  originSessionId: 543e113e-d1a1-4497-92ba-1bcf831ec8df
---

**Goal:** promote the FB↔GCIQL bridge ablation grid in
`PAPER/fb-gciql-bridge/fb-gciql-bridge.tex` from interim **n=3 @500k** to
**n=10 @1M** so the table is publication-accurate. Extends [[handoff-iwr-program]].

**STATUS — DONE (2026-06-24):** training finished (`ALL DONE` in dispatch log, 125
params_1000000.pkl). @1M per-seed extracted (GCIQL eval.csv + FB wandb summary
`eval/reward/eval/success`); G4 zero-shot eval re-run on 9 seeds via
`scripts/eval_zcond_gciql.py` (→ `analysis/eval_zcond_gciql_1m/s*.json`):
**exact-z 93.0±1.9, inferred-z 6.7±3.1** (n=9). `fb-gciql-bridge.tex` fully refreshed
to @1M (grid + 3 inline tables + G4 table + all prose + framing) and PDF rebuilt.
Key @1M shifts: bridge d512 88→**94.5±1.3** (n=8); d50 **92.6**, d128 **93.9**, d5 6→**14.3**;
stock GCIQL **67.4**, FlowBC **72.8**; τ.5 **88.9**, τ.95 **90.6**, singleQ **90.2**,
gaussian-actor **56.6**; zHER **32**, q-loss **38.8**, FB-IQL-critic **23.6**; CRL-flow **25.8**,
CRL-gauss **16.8**. Story holds & tightens. **5 stragglers left n=8/9** (d512 sd003/sd007,
FlowBC sd003, G4 sd005, CRL-flow sd003 — exited <1M; user chose to finish at n=8/9, NOT relaunch).
Note: the old "~9pt ψ-bottleneck drop" (G4 exact 86 vs bridge 88) collapsed to ~1.5pt at 1M
(93 vs 94). Extraction script: scratchpad `extract_1m.py`; results `results_1m.json`.
Minor open item: q-loss CI (38.8±6.2) still overlaps plain FB (43±6); means clearly ordered.

**(orig) RUNNING (launched 2026-06-22 22:48 UTC):** tmux **`bridge1m`** = the dispatcher
`scripts/launch_bridge_10seed_1M.sh` (ran with `DRY_RUN=0 SLOTS_PER_GPU=3`). **125 jobs**
(33 resume + 92 fresh), **24 concurrent** (cap = 8 GPU × 3). Dispatch log
`/dev/shm/bridge1m_dispatch.log`; per-job logs `/dev/shm/bridge1m_logs/<tag>.log`.
ETA ~13h → **expect done ~2026-06-23 ~12:00 UTC**. As of 23:22: 24/125 launched
(first wave), 11 resume seeds already hit 1M (the +500k extensions finish fast).

**Is it done?** `tmux ls | grep bridge1m` (gone ⇒ finished) · dispatch log should reach
`125` `gpu` lines and end with `ALL DONE` · `find /dev/shm/gciql_outputs -name params_1000000.pkl | wc -l`.

**⚠️ BACKUP WAS SKIPPED (user deferred).** Everything is in **volatile /dev/shm** — a
reboot kills all in-flight runs AND the resume checkpoints. `scripts/backup_bridge_ckpts.sh`
is ready and safe to run mid-campaign (just rsyncs). Nag the user to run it.

**Where outputs land (read @ step 1000000):**
- GCIQL/JAX → `/dev/shm/gciql_outputs/factored-fb/<group>/sd*/eval.csv` col
  `evaluation/overall_success`. Groups: `GCIQL-State-Flow-Bilinear` (d512), `-d50/-d128/-d5`,
  `-d50-tau05/-tau095/-singleq/-ddpgbc`, `gciql_g4_zcond`, `crl_state_ddpgbc`,
  `factored-fb-gciql` (stock), `GCIQL-State-Flow` (FlowBC), `factored-fb-crl-flowbc`.
- FB/torch → `/dev/shm/factored-fb_outputs/<group>_s*/` + logs; metric `eval/reward/eval/success`.
  Groups `fb_state_zher`, `fb_state_iql` (q-loss), `fb_state_iqlcritic_fix`.
- Special cases: d512 = 5 existing @1M + fresh seeds 5–9; CRL-Gaussian = 10 fresh (had no ckpt);
  FB resume uses TOTAL `num_train_steps=1000000`, GCIQL resume uses REMAINING `train_steps=500000`
  (+`restore_epoch=500000`+`exp_name=<dir>`, shim offsets steps). G4 in-loop eval = exact-z;
  exact+inferred-z need `scripts/eval_zcond_gciql.py` after.

**POST-RUN — the task to finish (locked convention = 95% CI, Student-t, t(n-1)·s/√n,
from UNROUNDED per-seed):**
1. Re-extract @1M overall_success per seed (all 10), every row — same extractors used this
   session (eval.csv for GCIQL, `eval/reward/eval/success` for FB; PSM-o1000 came from wandb).
2. mean ± 95% CI from unrounded values.
3. Update the grid + the 3 inline tables (§3.6 interventions, §"improving rep", §G4) + prose
   in `fb-gciql-bridge.tex`; **relabel "@500k"→"@1M"**; rebuild PDF
   (`cd PAPER/fb-gciql-bridge && /tmp/tectonic -X compile fb-gciql-bridge.tex`; re-fetch tectonic
   musl from gh releases if /tmp wiped).
4. Re-examine prose claims at the tighter n=10 CIs (the 43→86 ladder, dissociations).

**KNOWN ISSUE to revisit after @1M:** at n=3 the 95% CIs are wide enough that some claims have
**overlapping CIs** — e.g. §3.6 "every intervention lands below plain FB" (plain FB 43±6 vs
zHER 35±17, IWR 23±23). Survivors: actor ladder 43±6→72±4→88±2, G4 86±11 vs 8±21. The 10-seed
refresh is exactly what resolves this. User was OFFERED a prose-softening pass but deferred it.

**Doc state NOW:** `fb-gciql-bridge.tex` was FULLY converted to 95% CI this session (grid added
with CRL rows + all inline tables + prose), PDF rebuilt. Those are still the INTERIM @500k
numbers — the campaign replaces them with @1M. §1 headline table left as-is (already 95% CI,
peak-IQM/1.96×SE — a different estimator, intentionally).

**Files created this session:** `scripts/launch_bridge_10seed_1M.sh`,
`scripts/backup_bridge_ckpts.sh`, `configs/gciql/cube_single_state_gciql_flow_bilinear_d{50,128,512}.yaml`.
Spec: `PAPER/specs/2026-06-22-bridge-table-10seed-1M-campaign.md`.

**Cleanup done this session:** killed the 10 paused PSM ortho100 procs (12-day SIGSTOP'd,
abandoned) + 8 stale tmux sessions (crl_state, onestep, rldp, sweep_psm_lrcheck, sweep_psm_tau01,
sweep_rldp_fbmatch, psm_orthohi_1M, psm_fresh_rebal). **Only `bridge1m` remains.** See
[[tmux-sessions-registry]]. Conventions: [[wandb-entity-amsks]], [[jax-gpu-packing-preallocate]],
[[always-run-in-tmux]], [[gciql-flow-agent]].
