---
name: handoff-bridge-knobs-session
description: "READ FIRST: master handover. 2026-06-30 update at top = current state (measure-on-GCIQL reciprocal experiment running); 2026-06-29 below = the bridge-knobs/HIQL/GCIVL launch."
metadata: 
  node_type: memory
  type: project
  originSessionId: 29d24434-2cf7-47b2-89cc-f7b4407c6f44
---

## ===== 2026-06-30 UPDATE (read this first; supersedes the 06-29 section below) =====

**Branch is now `gciql-fb-bridge`** (off feat/psm-integration). Pushed commits: `8d35fab`
(knobs+HIQL/GCIVL bilinear+paper), `3833331` (bilinear_v, gcivl_flow, measure_coef/legs/only).
**UNCOMMITTED:** the `measure_ortho_coef` addition to `gciql_flow.py` (the FB ortho term) — fold
in once the run proves not-broken.

### RUNNING NOW (tmux: `measonly`, `r3bilinv`, `cksync`)
- **`measonly`** — THE KEY EXPERIMENT (reciprocal of the bridge): pour FB's successor-measure
  objective INTO bilinear-GCIQL. `gciql_flow` value_arch=bilinear d50, flow actor, `measure_only=true`
  (drops IQL value/critic — critic trained ONLY by the state-action SM), `measure_legs=critic`,
  `measure_coef=1.0`, **`measure_ortho_coef=1000`**. 3 seeds @500k, group `GCIQL-Bil-MeasureOnly-O1000`,
  logs `/dev/shm/measonly_logs/`. ⚠️ Pull `GCIQL-Bil-MeasureOnly-O1000` eval.csv @500k.
- **`r3bilinv`** — bilinear-V test (`fb_bridge_r3_bilinearv`): r3 (decond+learnedB+d_norm) + bilinear_v,
  3 seeds @500k. Early ~46 (1 seed) ⇒ bilinear-V NOT closing 57→90. Logs `/dev/shm/r3bilinv_logs/`.
- **`cksync`** — backup watcher → ~/bridge_ckpt_backup.

### ⚠️ THE MEASURE-ON-GCIQL STORY (be careful here — user flagged it, correctly)
Reciprocal of the FB→GCIQL bridge. Implemented FB measure loss on GCIQL's bilinear critic
(`gciql_flow.measure_loss`): off-diag TD + diagonal attraction (AUDITED == FB's `fb_successor_terms`)
+ **ortho** on psi (added late). Knobs: measure_coef, measure_legs (critic=state-action SM faithful /
value=state SM), measure_only (drop IQL), measure_ortho_coef.
- **measure_only NO-ortho collapsed to 8 (<plain FB 34)** → NOT conclusive; it was missing FB's ortho
  (load-bearing stabilizer, FB uses 1000). That run KILLED.
- **NOW re-running WITH ortho=1000.** Two readings: recovers ~34 ⇒ ortho was the issue (then need a
  balanced-scale run for any value-blind claim); still collapses ⇒ likely over-whitened.
- **SCALE CAVEAT:** our measure has a 1/sqrt(d) FB does NOT, so at λ=1000 ortho dominates ~5000:1 (more
  than FB's balance). If over-whitened, NEXT run = drop the 1/sqrt(d) on the measure (M=<phi,psi>) so
  λ=1000 is apples-to-apples with FB. DO NOT make a "measure is value-blind" claim until this is clean.

### RESULTS @500k this session (3 seeds, FlowBC actor, cube; the ladder)
CRL 30 · plain FB 34 · **FB-IQL goal_match r0=59** · +decond r1=58 · +d_norm r2=65 · +both r3=57 ·
stock GCIQL+flow 71 · bilinear-GCIQL 90. (FB ladder details [[fb-to-gciql-bridge-knobs]].)
- **idB (z=g) FINAL:** idB_dnorm (identity B) **65.3** = MATCHES r2 learned-B 65 ⇒ **learned B is NOT the
  lever; identity z=g ties it at convergence** (my 250k "identity hurts" read was a transient). idB_decond
  (identity+decondition) **2.7** = collapsed (goal-blind: task-agnostic phi + raw-g readout; the learned psi
  IS load-bearing when phi is de-conditioned).
- **gcivl_flow** (GCIVL bilinear + goal-cond flow actor, NEW agent) **79.2** vs GCIVL-AWR **67.5** ⇒ flow
  helps GCIVL +12 (v1 was a 0% bug — goal-agnostic velocity field; fixed v2 = goal in velocity field). [[hiql-gcivl-bilinear]]
- HIQL-bilinear ~16 (untuned/short-horizon), GCIVL-bilinear-AWR 67.5.

### FINDINGS (careful framing for the paper)
- **FB→GCIQL: the OBJECTIVE is the lever** — measure→IQL-goal-match doubles FB (34→59), actor+form fixed.
- Structural knobs FLAT: decondition phi, d_norm, bilinear-V none close 59-65→90; learned-B≈identity (65).
- **GCIQL→FB (measure-on-GCIQL): IN PROGRESS, not concluded.** The dramatic 90→8 was an ortho artifact;
  the faithful (ortho=1000) run + a balanced-scale control are needed before any claim.
- Paper already has the 500k ladder + 2-tier split + spine revision (FB-IQL ~59 not value-blind-24).
  PENDING: write the measure-on-GCIQL reciprocal ONLY after it's clean; r3_bilinearv final; design-audit section.

### NEXT STEPS
1. Pull measonly+ortho1000 @500k → decide recover-vs-collapse → if needed, balanced-scale run (drop 1/sqrt d).
2. Pull r3_bilinearv final (3 seeds). 3. Commit the measure_ortho_coef addition. 4. Paper writeup when measonly clean.

## ===== 2026-06-29 (original launch session) =====

**Session goal:** build + launch the FB→GCIQL bridge knob ladder, add bilinear to HIQL/GCIVL,
keep everything safe. All launched 2026-06-29 ~08:20. Branch `feat/psm-integration`.

## RUNNING (3 tmux campaigns — check `tmux ls`)
1. **`bridgeknobs`** — 15 FB runs (5 variants ×3 seeds @500k), `scripts/launch_bridge_knobs.sh`.
   Logs `/dev/shm/bridge_knobs_logs/<tag>.log`; wandb amsks/factored-fb `fb_bridge_<tag>`.
   Variants: a0_cubexyz · r0_base · r1_decond · r2_dnorm · r3_allon (see [[fb-to-gciql-bridge-knobs]]).
   @08:30 r0_base_s0 at step 16k, 15/15 training. ETA ~2-3h.
2. **`hiqlbil`** — 6 jax runs (hiql_bilinear_d50 ×3 + gcivl_bilinear_d50 ×3 @500k),
   `scripts/launch_hiql_gcivl_bilinear.sh`. Logs `/dev/shm/hiql_gcivl_bil_logs/`. HIQL hierarchical
   eval ~12s/ep (slow). See [[hiql-gcivl-bilinear]].
3. **`cksync`** — backup watcher (every 10min) mirroring the 2 above off volatile /dev/shm →
   `~/bridge_ckpt_backup`. `scripts/sync_running_campaigns.sh`. (ckpts land only @500k, save_interval=500000.)

**Done check (FB):** `grep -l train/q_loss /dev/shm/bridge_knobs_logs/*.log`; pull `success/overall` @ final.
**Collect:** re-extract @500k seeds {0,1,2} via `scratchpad/extract_seeds.py` (set step=500000) +
the jax eval.csv. Compare vs priors: bilinear-GCIQL 92, FB-IQL-critic 24, zg_iql 44, stock+flow 73, plain FB 48.

## DONE this session
- **Implemented (TDD, all green; torch):** FB knobs `decondition_F`, `d_norm`, `critic_goal_mix`,
  `actor_goal_mix`, `iql_reward=goal_match` (GCIQL dataset reward: success=goal is cur state).
  Tests `tests/test_decondition_dnorm.py` + `tests/test_goal_mix.py`; 33 FB-family green; real train.py
  smoke OK (iql_r_pos≈p_cur=0.2 confirms reward fires right). Details [[fb-to-gciql-bridge-knobs]].
- **Implemented (jax):** `value_arch=bilinear` on HIQL (custom `_BilinearRepValue`, option A keeps the
  state-dependent subgoal rep) + GCIVL (clean GCBilinearValue swap). Both jax-smoke-pass. [[hiql-gcivl-bilinear]]
- **Paper** `PAPER/fb-gciql-bridge/fb-gciql-bridge.tex`: added decouple + z=g rows; **plain-FB anchor
  corrected 43→48** (wandb-confirmed, was stale — [[bridge-plain-fb-anchor-correction]]); **split the grid
  into Part A (3-seed dev) + Part B (10-seed final)** with extracted per-seed numbers; PDF rebuilt.
- **Backups:** ALL prior /dev/shm checkpoints → `~/bridge_ckpt_backup` (166G; added phase2_out/fb_outputs/
  zg_cube_out this session). Prior campaigns (decouple/phase2/sanity/zg) all DONE + results in the paper.

## PENDING / NEXT
1. **Monitor the 3 campaigns to completion**, then collect the @500k 3-seed comparison table (FB ladder
   answers: does decondition_F/d_norm/goal-mix move FB-IQL toward bilinear-GCIQL? + HIQL/GCIVL bilinear-vs-mlp).
2. **OPEN PAPER TASK (user-requested):** add a "what changes from one algorithm to another" section to
   fb-gciql-bridge.tex — the per-knob design-decision audit (FB → GCIQL+Flow and → bilinear-GCIQL+Flow:
   critic struct / objective / sampling / actor / optim). The full audit was worked out this session.
3. **Known UNIMPLEMENTED knob gaps** (FB→bilinear-GCIQL not fully faithful): bilinear V head (V stays MLP),
   min(q1,q2) vs FB's mean-of-2-heads, actor-cond z vs raw g. Implement if the ladder leaves a residual gap.
4. **Commit the code** (uncommitted): `agents/fb/agent.py`, `configs/agent/fb.yaml`, `train.py`,
   `third_party/ogbench/impls/agents/{hiql,gcivl}.py`, new `tests/test_{decondition_dnorm,goal_mix}.py`,
   new `scripts/launch_{bridge_knobs,hiql_gcivl_bilinear}.sh` + `sync_running_campaigns.sh`.
   (NB `agents/fb/model.py` + `fixed_backward.py` show modified but are PRE-EXISTING uncommitted, not this session.)
   Conventions: [[commit-no-claude-coauthor]], [[factored-fb-test-env]] (`.venv/bin/python -m pytest`).

## GOTCHAS
- launch_bridge_knobs.sh bug FIXED: was missing `train.py` (all 15 failed instantly first try). OK now.
- `save_interval=500000` ⇒ NO mid-run checkpoints; a /dev/shm wipe mid-run loses in-progress training
  (cksync can't help until 500k). Relaunch w/ smaller save_interval only if a reboot is expected.
- OpenGL "Failed to load libOpenGL.so.0" INFO lines in logs are harmless (eval env).
- jax venv `/dev/shm/.venv-jax` (volatile); jax runs use XLA_PYTHON_CLIENT_PREALLOCATE=false to pack/GPU.
- HIQL/GCIVL bilinear = STATE path only (no visual encoder); jax side smoke-verified, no unit tests.
Relates: [[handoff-iwr-program]], [[tmux-sessions-registry]], [[wandb-entity-amsks]], [[always-run-in-tmux]].
