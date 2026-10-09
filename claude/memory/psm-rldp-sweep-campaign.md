---
name: psm-rldp-sweep-campaign
description: "RUNNING — PSM & RLDP state hyperparameter sweep (tuning parity with FB); how to resume: summarize grids → pick winners → 1M×10 tuned runs"
metadata:
  node_type: memory
  type: project
  originSessionId: 6787d218-60cd-4a6d-ad25-71a042116338
---

**Goal:** close the tuning asymmetry — FB-state was swept (`ortho_coef×lr_b` → `fb_state_plain`
winner), PSM & RLDP were not. Sweep both on STATE, 3 seeds/combo @ 500k for selection, then the
winner per method at 1M×10 seeds (the numbers PAPER reports). Spec+plan:
`PAPER/specs/2026-06-08-psm-rldp-hyperparameter-sweep{,-plan}.md`.

**Faithfulness audit (done):** our PSM is a bit-exact replication of the reference
(`~/git/baselines/PSM/agent/psm.py`); only deviation is the FlowBC actor. PSM's `ortho_coef` was
a DEAD knob (reference hardcodes `1*orth_loss` in the phi update, `0*` in SF) — we MADE IT LIVE
(`self.ortho_coef * orth_loss`, default 1.0 = bit-exact). RLDP's `ortho_coef`/`lr_b` were already
live. So PSM ortho sweep is an intentional ablation beyond the paper (paper fixes it at 1).

**Code (committed on `feat/psm-integration`, NOT yet pushed):**
- `c40a8dc` PSM ortho_coef live + `tests/test_psm_ortho_coef_live.py`
- `3356d3b` `scripts/run_grid_sweep.sh` (generic 2-knob launcher) + `4589504` fix (venv python
  `$REPO/.venv/bin/python`, exit-code propagation — bare `python` is rc=127 in tmux non-login shell)
- `9a2e07c` `scripts/sweep_lib.py` + `scripts/summarize_sweep.py` + `tests/test_sweep_lib.py`
  (NB: test loads sweep_lib by FILE PATH to dodge `tests/scripts/` shadowing repo-root `scripts/`;
  do NOT switch pytest to importlib mode — it breaks 10 modules' collection. Baseline = 9
  pre-existing collection errors, mostly missing pandas, not regressions.)

**Status: LAUNCHED 2026-06-08 08:07.** tmux `sweep_psm` + `sweep_rldp`, all 8 GPUs, `JOBS_PER_GPU=2`
(~4 jobs/GPU, ~0.4GB each). 18 PSM + 18 RLDP @ 500k. Grids:
- PSM `agent=psm_flowbc`: `ortho_coef∈{0.1,1,10} × lr_phi∈{1e-4,1e-5}`, group `psm_state_sweep__*`
- RLDP `agent=rldp_flowbc`: `ortho_coef∈{0.001,0.01,0.1} × lr_b∈{1e-4,1e-5}`, group `rldp_state_sweep__*`
BOTH launchers share ONE grid dir **`logs/grid_20260608_080705`** (launched same second) — that's
fine, summarize filters by knob name. Masters `logs/sweep_{psm,rldp}_master.log`. NB JOBS_PER_GPU=2
→ 16 slots/launcher, so 4 of the 36 jobs queue until slots free (expected).

**TO RESUME (when runs reach `[step 500000]` / `pgrep -fc train.py` → 0 / tmux sessions exit):**
1. Grid dir is `logs/grid_20260608_080705` (both methods; `ls -dt logs/grid_*` to confirm).
2. Summarize + pick winners (CPU; same dir for both, summarizer filters by knob):
   `PYTHONPATH=. .venv/bin/python scripts/summarize_sweep.py logs/grid_20260608_080705 --knob-a ortho_coef --knob-b lr_phi --out analysis/sweep/psm_state`
   `PYTHONPATH=. .venv/bin/python scripts/summarize_sweep.py logs/grid_20260608_080705 --knob-a ortho_coef --knob-b lr_b   --out analysis/sweep/rldp_state`
3. Launch winners at 1M×10 (mirror fb_state_plain) via `scripts/run_grid_sweep.sh` with single-value
   KNOB_*_VALS, `SEEDS='0 1 2 3 4 5 6 7 8 9'`, `TRAIN_STEPS=1000000`, `JOBS_PER_GPU=2`,
   `RUN_GROUP_BASE=psm_state_tuned` / `rldp_state_tuned`. Run in tmux ([[always-run-in-tmux]]).

**UPDATE 2026-06-08 ~20:30 — FB-incidentals matched, lr re-check launched.** Investigation of
FB-vs-PSM (representation stabilization) surfaced that PSM differs from FB on incidental HPs that
affect Q estimation. Decided to match three to FB in `configs/agent/psm.yaml` (reference PSM values
in parens): `z_dim 50` (128), `discount 0.99` (0.98), `target_tau 0.005` (0.01). Bit-exact test
unaffected (passes explicit kwargs, doesn't read the yaml). `z_dim`↔`max_log_seed` are independent
(proto binary-z is `max_log_seed=16`, separate from continuous z_dim), so 128→50 is safe. RLDP NOT
matched (separate base; revisit if 3-way fairness wanted). Because `z_dim` 128→50 changes capacity,
the original sweep's lr_phi=1e-5 winner is stale → launched a **quick lr_phi re-check** (tmux
`sweep_psm_lrcheck`: ortho_coef=1, lr_phi∈{1e-4,1e-5}×3 seeds @500k) at the new basis. When it
finishes: pick lr_phi winner, set it as `psm.yaml` default, THEN launch the 1M×10 tuned runs.

**UPDATE 2026-06-09 — sweeps DONE, winners picked, defaults committed+pushed.** All grids finished
(no train.py procs, GPUs idle, logs end `[train] Done.` @500k). Aggregate 500k eval/success (mean/3 seeds):
- PSM (z128 orig basis): `ortho1×lr_phi1e-5` = **0.340** (BEST PSM overall) > ortho10×1e-5 0.247 > ortho1×1e-4 0.180; ortho_coef matters (0.1→1→10 ≈ 0.13→0.34→0.25, peak at ref value 1).
- PSM FB-matched basis (z50/0.99/τ0.005): best `ortho1×lr_phi1e-5` = 0.267 → FB-matching COST ~7pts. τ ablation: τ0.005 (0.267) ≈ τ0.01 (0.253), τ barely matters; lr_phi=1e-5 wins everywhere.
- RLDP: winner `ortho0.001×lr_b1e-5` ≈ **0.200**. RLDP-fbmatch (ortho1000/lr_b1e-4 at FB's point) = 0.140 → self-predictive head HURTS at FB's operating point (clean negative result).
- Reference: FB-state plain ≈ 0.44–0.60, still ahead of all PSM/RLDP (the paper's point).

**DECISION (user, 2026-06-09): PSM default = z128 ORIGINAL-basis winner**, NOT FB-matched. Chose PSM's
genuine best (0.34) over apples-to-apples fairness (0.27). `configs/agent/psm.yaml` now: `z_dim=128,
discount=0.98, target_tau=0.01, ortho_coef=1, lr_phi=1e-5` (only `lr_phi` deviates from reference).
Commit `6e20dd9`. **Branch `feat/psm-integration` IS pushed** (origin tip `6e20dd9`; the old "not yet
pushed" note was stale — remote already had it + a `chore(gitignore)` commit `25b0d0b`, rebased onto).

**UPDATE 2026-06-09 07:20 — 1M×10 TUNED RUNS LAUNCHED.** tmux `psm_tuned` + `rldp_tuned`, 10 jobs
each @1M, all 8 GPUs (JOBS_PER_GPU=2). PSM = `ortho_coef=1 lr_phi=1e-5` (z128 default, the committed
psm.yaml winner); RLDP = `ortho_coef=0.001 lr_b=1e-5`. Groups `psm_state_tuned__*`/`rldp_state_tuned__*`,
grid `logs/grid_20260609_072020`, masters `logs/sweep_{psm,rldp}_tuned_master.log`, output dirs
`outputs/<group>__s<seed>/`. ~10h/run expected. ALSO DONE this session: crash-recovery **resume
feature** (commits 2ad1f9c..9cdd61d on feat/psm-integration, NOT yet pushed) — `resume=true`/
`resume_from`, deterministic run-name output dirs, `RESUME=1` launcher flag; state+pixel+GCIQL smokes
green. If a tuned job dies, relaunch the same launcher with `RESUME=1`. Open: push feat/psm-integration.

**(superseded) TO RESUME — launch the 1M×10 tuned runs** (the numbers PAPER reports): PSM winner = z128 default above;
RLDP winner = `ortho0.001×lr_b1e-5`. Use `scripts/run_grid_sweep.sh`, single-value KNOB_*_VALS,
`SEEDS='0..9'`, `TRAIN_STEPS=1000000`, `JOBS_PER_GPU=2`, groups `psm_state_tuned`/`rldp_state_tuned`,
in tmux. **Open:** monitoring cadence (ping vs /loop) — user deferred, revisit at launch.
See [[tmux-sessions-registry]], [[encmatch-ablation-confound]], [[wandb-entity-amsks]], [[lrcheck-monitoring-protocol]].
