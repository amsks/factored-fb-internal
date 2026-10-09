---
name: tmux-sessions-registry
description: Registry of active/known tmux sessions on the GPU box and their purpose — check at session start
metadata: 
  node_type: memory
  type: reference
  originSessionId: 6787d218-60cd-4a6d-ad25-71a042116338
---

Running list of tmux sessions on the Factored-FB GPU box and what they're for.
Per [[always-run-in-tmux]], update this whenever you launch or retire a session.
At session start, run `tmux ls` and cross-check against this list (verify GPU use
with `nvidia-smi`; a session can linger after its run finished).

**CURRENT STATE (2026-08-26): only the two Claude sessions remain. GPUs idle (no compute apps).**
- **`opus`**, **`fable`** — the two parallel Claude Code sessions in the repo (measure track + the other track). Not campaigns; do not kill.
- KILLED 2026-08-26 (all idle leftovers, no GPU work): `cksync` (58d old, `scripts/sync_running_campaigns.sh` ckpt backup),
  `s30watch1` (`watch_spec30_p1.sh`, spec30 P1 arms — finished, WAVE-1 finals already collected),
  `s30i23watch` (`watch_s30_items23.sh`, spec30a Items 2+3 isolation baselines — finished at 500k).
  Watcher logs kept at `/dev/shm/spec30_p1_logs/watch.log` and `/dev/shm/spec30_i23_logs/watch.log` (volatile, /dev/shm).

---
**CURRENT STATE (2026-07-08 ~04:15). THREE live campaigns (two sessions sharing the box):**
- **`mdose`** (bridge track, launched 04:00) — 21 runs @500k: MeasureMix sgphi λ{3,10,30} +
  shared l10, antmaze MeasureOnly-DDPGBC, C5a/C5b zmix; round-robin all 8 GPUs ~2-3/GPU,
  logs `/dev/shm/mdose_logs/`, launcher scripts/launch_measure_dose_antmaze.sh. Full handoff:
  [[mdose-campaign-20260708]].
- **`e2probe`** — FINISHED (ψ cross-eval, results analysis/e2_psi_probe/all18.json); safe to kill.
- plus the parallel SOTA session's `bothbil` + `ambc03` below.

**(2026-07-07 evening, the parallel SOTA-track session's) two campaigns:**
- **`bothbil`** (launched ~evening) — HIQL v4 both-bilinear (native + flow-both × 3 seeds @500k),
  1/GPU on GPUs 0-5, logs `/dev/shm/bothbil_logs/`, groups HIQL-State-{Native,Flow-both}-BothBil-d50-sg10,
  commit f752725. See [[handoff-hiql-bilinear-20260707]] §v4.
- **`ambc03`** (launched right after) — antmaze flow reruns @ td_jepa-correct bc_coeff=0.3
  (GCIQL flow bilinear+MLP) + coverage (GCIVL-flow-bilinear, HIQL-v3-flow), 12 runs @500k
  round-robin all 8 GPUs, logs `/dev/shm/ambc03_logs/`, commit 0a275fd. See handoff §ambc03.
- `hiqlmech`/`amhiql`/`e2measure`/`sotagap` ALL FINISHED (sessions may linger; GPUs verified idle
  before the two launches above). Old lingering sessions (bridgeabl/bridgeends/.../untried) still safe to kill.

**Earlier (2026-07-07 ~12:15).** **`hiqlmech` LAUNCHED** (12 runs: E-B rep50 /
E-C schaul / E-D highbil native+flow × 3 seeds @500k, GPUs 0/1/2/4, 3/GPU;
logs `/dev/shm/hiqlmech_logs/`; groups HIQL-State-Native-Bilinear-d50-{rep50,schaul}-sg10,
HIQL-State-{Native,Flow-both}-HighBil-d50-sg10; commit a828ee8). Smoke was killed mid-final-eval
on user request (all 4 configs had already built+trained clean; logs error-free).
E2 measure-mix: ALL 15 FINISHED — prize cross-eval unblocked. `amhiql` at ~25–35% of 1M.
See [[handoff-hiql-bilinear-20260707]].

**Earlier (2026-07-07 ~10:00).** THREE live campaigns:
- **`amhiql`** (2026-07-07 ~09:45) — antmaze-medium HIQL MLP-vs-bilinear-d50, 2 cfgs × 8 seeds
  @1M (spec grid-1b protocol, OGBench-faithful HPs = pure agent defaults incl. sg=25;
  hyperparameters.sh:116 only sets default alphas). `scripts/launch_antmaze_hiql.sh`, GPUs 3/5/6/7
  (4 runs/GPU), logs `/dev/shm/amhiql_logs/`, groups `HIQL-AntMaze-Medium-{MLP,Bilinear-d50}`
  (16/16 confirmed on wandb amsks). Antmaze ~1h/1M solo → few hours packed. Purpose: H1/H2-vs-H3
  discriminator for the HIQL-bilinear brainstorm (HIQL has max headroom on antmaze, published 96).
- **`e2measure`** + **`sotagap`** — see below; as of ~10:00 e2measure 7/15 done @500k (rest 340–400k),
  sotagap cube HIQL sg10 cells done/finishing, scene ddpgbc 3 runs ~100k/500k (~2 days left, GPUs 0-2).

**Earlier state (2026-07-07 ~00:45).** `tmux ls` shows many lingering sessions
(bridgeabl/bridgeends/bridgewatch/cksync/decpl2/fbz/isfb/ortho/phase2/twotower/untried —
mostly old/done, verify with nvidia-smi before reuse/kill). Two live campaigns matter:
- **`e2measure`** (2026-07-07 00:43) — E2 scale-controlled IQL+measure interpolation, 15 runs
  @500k (`scripts/launch_e2_measure_mix.sh`, launcher `wait`s on all). Logs
  `/dev/shm/e2measure_logs/*.log`, ckpts `/dev/shm/gciql_outputs/factored-fb/GCIQL-State-Flow-Bilinear-d50-MeasureMix-*`.
  Full state + collect recipe: [[e2-measure-mix-campaign]]. Early: scale control works
  (q_mean ≈ −19), success 0.6–0.8 @100–200k. ~7/15 progressing, ~8 queued behind contention.
- **`sotagap`** (2026-07-06 21:59) — the OTHER campaign (scene DDPGBC + HIQL sg10, [[sota-bilinear-paper]]),
  STILL DRAINING; its scene/HIQL runs are the contention slowing e2measure. Do NOT kill without owner OK.
- LESSON (my mistake this session): I launched e2measure WITHOUT `tmux ls` first → oversubscribed
  the GPUs on top of sotagap (~23 jax/8 GPUs). Non-fatal (no OOM/crash) but slowed everything.
  ALWAYS `tmux ls` + `nvidia-smi` before launching.

---
**CURRENT STATE (2026-06-29 ~08:30): 5 sessions — 3 NEW campaigns + 2 lingering-done.**
Full handover: [[handoff-bridge-knobs-session]].
- **`bridgeknobs`** (08:21) — 15 FB knob-ladder runs @500k×3seeds; `scripts/launch_bridge_knobs.sh`;
  logs `/dev/shm/bridge_knobs_logs/`. (FB→bilinear-GCIQL knob A/B; [[fb-to-gciql-bridge-knobs]].)
- **`hiqlbil`** (08:17) — 6 HIQL/GCIVL bilinear-d50 jax runs @500k×3seeds;
  `scripts/launch_hiql_gcivl_bilinear.sh`; logs `/dev/shm/hiql_gcivl_bil_logs/`. ([[hiql-gcivl-bilinear]].)
- **`cksync`** (08:29) — backup watcher (10-min) → `~/bridge_ckpt_backup`; `scripts/sync_running_campaigns.sh`.
- **`decpl2`, `phase2`** — DONE campaigns (results in the bridge paper), sessions linger; safe to kill.
- Earlier `bridge1m` (10-seed×1M) finished long ago; all prior /dev/shm ckpts backed up (166G). History below is RETIRED.

---
**RETIRED / historical (all killed or finished as of 2026-06-22):**

**Active / known sessions** (as of 2026-06-18):

- **fb_ddpgbc / crl_ddpgbc / fbiql_ontraj_flow / fbiql_ontraj_ddpgbc** (launched 2026-06-18 08:31) —
  flow-vs-noflow + on-trajectory FB-IQL study, 3 seeds @500k each (12 runs), wandb `amsks/factored-fb`.
  All on cube_single state, ortho_coef=1000, bc_coeff=3 (cube domain default), lr=1e-4, z=L=50.
  • `fb_ddpgbc` → group `fb_state_ddpgbc`: plain FB (`agent=fb`, TD3+BC = no-flow DDPG+BC actor), GPUs 0/1/2.
  • `crl_ddpgbc` → group `crl_state_ddpgbc`: stock OGBench CRL no-flow (`cube_single_state_crl.yaml`, jax), GPUs 3/4/5.
  • `fbiql_ontraj_flow` → group `fb_iql_ontraj_flow`: FB-IQL on-traj (`agent=fb_flowbc q_loss_coef=1
    train_goal_mode=traj train_goal_ratio=1.0`), GPUs 6/7/0.
  • `fbiql_ontraj_ddpgbc` → group `fb_iql_ontraj_ddpgbc`: same but `agent=fb` (no-flow DDPG+BC), GPUs 1/2/3.
  Launch scripts `/dev/shm/launch_iwr/{A,C,D}*.sh` + crl/state.sh; outs `/dev/shm/launch_iwr/{A,B,C,D}.out`;
  seed logs `/dev/shm/<group>_<TS>/seed_*.log`. **Interpretation note:** FB-IQL here = the *measure-loss*
  recipe (`q_loss_coef=1`, measure kept) NOT `critic_mode=iql` — because `critic_mode=iql` computes z directly
  from the goal and IGNORES `train_goal_ratio`, so "ratio=1 on-trajectory measure approximation" only bites in
  the measure path. Pull at 500k like the rest ([[handoff-iwr-program]]). The Jun 10-16 sessions below are finished/idle.

**Active / known sessions** (as of 2026-06-16):

- **g4_s0 / g4_s1 / g4_s2** (launched 2026-06-16 19:33) — G4 success-level objective isolation
  ([[handoff-iwr-program]]): GCIQL bilinear IQL value/critic + FB FlowBC actor conditioned on
  `z=stop_grad(value.psi(goal))` (`actor_cond=z`), eval zero-shot. Seeds 0/1/2 on GPUs 0/1/2,
  500k steps, `XLA_PYTHON_CLIENT_PREALLOCATE=false`. Config `cube_single_state_gciql_flow_g4.yaml`
  (d=512), wandb `amsks/factored-fb` group `gciql_g4_zcond`. Logs `/dev/shm/g4_zcond_s{0,1,2}.log`,
  ckpts under `/dev/shm/gciql_outputs/.../gciql_g4_zcond/sd00{0,1,2}_*/params_500000.pkl`. In-loop
  eval reports EXACT-z success; run `scripts/eval_zcond_gciql.py` after for exact+inferred-z. ETA ~hours.
  NB the June 10-11 sessions below are almost certainly finished/idle — verify with nvidia-smi before reuse.

**Active / known sessions** (as of 2026-06-11):

- **psm_o1000_ext** + **fb_plain_ext** (launched 2026-06-11 06:00) — +500k EXTENSIONS to 1.5M.
  `psm_o1000_ext`: PSM ortho1000 10 seeds resume `step_1000000.pt`→1.5M (run_grid_sweep RESUME=1 TRAIN_STEPS=1500000,
  GPUs 0-3), group `psm_state_orthohi__ortho_coef1000__lr_phi1e-5`, logs `logs/grid_20260611_060034/`.
  `fb_plain_ext`: **fb_state_plain** (=the paper's FB baseline, fb_flowbc ortho_coef=1000 lr_b=1e-4, 10 seeds) resume
  `step_1000000.pt`→1.5M via per-seed `hydra.run.dir=/dev/shm/factored-fb/runs/20260607_070225__fb_state_plain__sN`
  + resume=true (launcher `/tmp/fb_plain_ext.sh`, GPUs 4-7), group `fb_state_plain`, logs `/dev/shm/fb_plain_ext_<TS>/`.
  Both resume from the exact 1M step (no wandb out-of-order gap). ETA ~7h. NB **ortho100 RE-PAUSED** (state T) on its
  GPUs — `kill -CONT` to resume it later, or it stays frozen.

**Active / known sessions** (as of 2026-06-10):

- **psm_orthohi_1M** + **psm_fresh_rebal** (launched 2026-06-10 16:29, rebalanced 22:17) — PSM orthohi
  promotion to 1M×10 ([[psm-readout-ablation]]). `psm_flowbc ortho_coef∈{100,1000} lr_phi=1e-5`, 10 seeds → 1M,
  group `psm_state_orthohi__ortho_coef{100,1000}__lr_phi1e-5` (entity amsks). `run_grid_sweep.sh`, ~650MB/job.
  **Layout after rebalance:** resume s0–s2 still on GPUs 1,3 (session `psm_orthohi_1M`, from `step_400000.pt`);
  fresh s3–s9 KILLED at 200k and relaunched (session `psm_fresh_rebal`, `RESUME=1 GPU_IDS="0 2 4 5 6 7"
  JOBS_PER_GPU=3`) from `step_200000.pt` → spread 3→6 GPUs to ~2× throughput. Rationale: aggregate
  throughput per GPU is ~constant (~48 steps/s) so packing doesn't help — only GPU count does; fresh was
  the long pole at 5/GPU. Logs `logs/grid_20260610_162911/` (resume) + `logs/grid_20260610_221753/` (fresh);
  stdout `/tmp/psm_orthohi_resume.out`, `/tmp/psm_fresh_rebal.out`. ETA ~12h from rebalance.
  **2026-06-10 ~22:50 — `ortho100` PAUSED via `kill -STOP` (all 10 in state T) to prioritize `ortho1000`.**
  ortho100 holds GPU mem but no compute; `ortho1000` (10 jobs) now ~1-2/GPU, ETA ~9h (gated by fresh s3+s9
  sharing GPU2). To resume ortho100: `kill -CONT` its PIDs (resumes in place, zero loss) once ortho1000 done.
  NB resume-from-older-ckpt artifact: ortho1000 s4/s7 were killed at ~260k but resumed from `step_200000.pt`,
  so wandb shows them frozen at ~260k (out-of-order step rejection) until they re-climb past 261k — cosmetic, self-heals.
  **2026-06-11 01:05** — ortho1000 s1 finished (1M); freed GPU1 used to migrate ortho1000 **s9 → GPU1** (session
  `psm_s9_gpu1`, resume from `step_400000.pt`), leaving s3 solo on GPU2 — both long-pole seeds now solo (~45 steps/s),
  long pole ~3.7h. Watcher #1 retired. **Watcher #2** (`/tmp/wait_all_o1000.sh`, bg) still armed → `kill -CONT` the 10
  paused ortho100 when all ortho1000 hit 1M. Reminder: ortho100 is still PAUSED (state T) and MUST be CONT'd or killed.

- **probe_suite** (launched 2026-06-10 22:20) — paper full probe suite on tuned PSM+RLDP 1M×10
  ([[psm-rldp-probe-suite]]). CPU-only (won't touch training GPUs), `scripts/run_psm_rldp_probe_suite.sh`,
  70 jobs N_PAR=24. Logs `logs/probe_suite_<TS>/`, stdout `/tmp/probe_suite.out`.

- **gciql_flow_state** + **gciql_flow_pixel** (launched 2026-06-09 23:44) — GCIQL+flow-actor ablation
  ([[gciql-flow-agent]]). state: 10 seeds @1M, group `GCIQL-State-Flow`; pixel: 5 seeds @500k DrQ,
  group `GCIQL-Pixel-DrQ-Flow`. Both on GPUs 0-4 (3 runs/GPU), launched with
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` to pack ([[jax-gpu-packing-preallocate]]). Launch logs
  `/dev/shm/gciql_flow_{state,pixel}.launch.log`, seed logs `/dev/shm/GCIQL-{State-Flow,Pixel-DrQ-Flow}_<TS>/seed_*.log`.
  NB the earlier psm_tuned/psm_pess0 sessions are gone; only psm_orthohi (GPUs 5-7) still ran at launch.
  **2026-06-10 status:** 8/10 state seeds hit 1M & exited clean; pixel 0/2/4 ~80%, finishing. TWO seeds
  CRASHED mid-run with `JaxRuntimeError: CUDA_ERROR_STREAM_CAPTURE_INVALIDATED` (transient GPU graph-capture
  fault, not a code bug): **state seed_3** (died 01:44 @400k — NO checkpoint, `save_interval=1000000`) and
  **pixel seed_1** (died 07:31 @200k — has 50k-interval ckpts). Both need relaunch to complete the campaign.

- **psm_tuned** + **rldp_tuned** (launched 2026-06-09 07:20) — the PAPER 1M×10 tuned runs.
  psm_tuned: `agent=psm_flowbc ortho_coef=1 lr_phi=1e-5` (z128 default basis, PSM sweep best
  0.34@500k), group `psm_state_tuned__ortho_coef1__lr_phi1e-5`. rldp_tuned: `agent=rldp_flowbc
  ortho_coef=0.001 lr_b=1e-5` (RLDP sweep best), group `rldp_state_tuned__ortho_coef0p001__lr_b1e-5`.
  Each 10 seeds @1M via `scripts/run_grid_sweep.sh` (JOBS_PER_GPU=2, entity amsks). Shared grid dir
  `logs/grid_20260609_072020`; masters `logs/sweep_{psm,rldp}_tuned_master.log`. Deterministic output
  dirs `outputs/<group>__s<seed>/` (NEW run-name layout from the resume feature). To monitor: compare
  vs FB@1M=0.52 / @500k(s0-2 matched)=tie; vs paper RLDP@1M=0.234. Relaunch-resume if a job dies:
  re-run the same launcher with `RESUME=1` (see [[psm-rldp-sweep-campaign]], `docs/resuming_runs.md`).

- **Older sweep sessions (DONE 2026-06-08, can be killed):** sweep_psm, sweep_rldp, sweep_psm_lrcheck,
  sweep_psm_tau01, sweep_rldp_fbmatch — all finished (`[train] Done.`), GPUs were idle before this launch.

- **sweep_rldp_fbmatch** (launched 2026-06-08 22:38) — RLDP with its FB-part HPs set 1-to-1 to
  FB's winner: `ortho_coef=1000, lr_b=1e-4` (vs RLDP's tuned 0.01/1e-5), keeping RLDP's
  `horizon=5`+predictor+SP loss. 3 seeds @500k on idle GPUs 6,7 (pinned via CUDA_VISIBLE_DEVICES,
  launched directly NOT via run_grid_sweep.sh so it wouldn't stack on the busy PSM GPUs 0-5).
  Tests whether the self-predictive head helps/hurts at FB's operating point (B is now
  double-regularized: ortho1000 + SP, and B is detached from the measure loss). Group
  `rldp_state_fbmatch__ortho_coef1000__lr_b1e-4`, logs `logs/grid_rldp_fbmatch_20260608_223837/`.
  Compare vs: FB itself (fb_state_plain ≈0.50) and RLDP-tuned (old sweep, lr_b1e-5/ortho0.01 ≈0.20).
  NB FB cube-state ortho_coef=1000 (fb.yaml bare default 1.0 is overridden at launch).

- **sweep_psm_tau01** (launched 2026-06-08 20:59) — target_tau ABLATION at the new FB-matched
  basis: z_dim=50 + discount=0.99 (new) but target_tau=0.01 (OLD). ortho=1, lr_phi∈{1e-4,1e-5}×3
  seeds @500k = 6 jobs. Isolates target_tau by comparing vs `sweep_psm_lrcheck` (identical but
  tau=0.005). Grid `logs/grid_20260608_205946`, group `psm_state_fbmatch_tau01__*`, master
  `logs/sweep_psm_tau01_master.log`. So the tau-pair is: lrcheck(tau0.005) vs tau01(tau0.01).

- **sweep_psm_lrcheck** (launched 2026-06-08 20:33) — quick lr_phi re-check (ortho_coef=1
  fixed, lr_phi∈{1e-4,1e-5} × 3 seeds @500k = 6 jobs) at the NEW FB-matched PSM basis. We
  edited `configs/agent/psm.yaml` to match FB incidentals: `z_dim 128→50`, `discount 0.98→0.99`,
  `target_tau 0.01→0.005` (reference values noted inline; bit-exact test unaffected, it passes
  explicit kwargs). This invalidated the original sweep's lr_phi=1e-5 pick (made at z_dim=128),
  hence the re-check. Group `psm_state_fbmatch_lrcheck__*`, master `logs/sweep_psm_lrcheck_master.log`.
  Next: pick lr_phi winner → set psm.yaml default → 1M×10 tuned. See [[psm-rldp-sweep-campaign]].

- **sweep_psm** / **sweep_rldp** — PSM & RLDP state hyperparameter sweep (tuning
  parity with FB). Each: 3×2 grid × 3 seeds = 18 runs @ 500k steps, concurrent on
  GPUs 0–7 (`JOBS_PER_GPU=2`, ~4 jobs/GPU, ~0.4GB each). PSM grid `ortho_coef∈{0.1,1,10}
  × lr_phi∈{1e-4,1e-5}` (note: ortho_coef was MADE live, see [[encmatch-ablation-confound]]
  sibling commit `c40a8dc`); RLDP `ortho_coef∈{0.001,0.01,0.1} × lr_b∈{1e-4,1e-5}`. wandb
  groups `psm_state_sweep__*` / `rldp_state_sweep__*` (entity amsks). Launched 2026-06-08
  08:07. Logs `logs/grid_<TS>/*.log` + masters `logs/sweep_{psm,rldp}_master.log`. Next:
  `scripts/summarize_sweep.py` per grid → winner → 1M×10 (`*_state_tuned`). Spec/plan:
  `PAPER/specs/2026-06-08-psm-rldp-hyperparameter-sweep{,-plan}.md`.

- **fb_state_encmatch** (FINISHED 2026-06-07 ~11:57) — param-matched state-encoder
  ablation; all 5 seeds have `final.pt`. Result: 0% eval — a design confound, NOT a
  bug (see [[encmatch-ablation-confound]]).
- **fb_state_plain** (FINISHED 2026-06-07 13:50) — plain FB baseline; all 10 seeds
  `final.pt`, eval ~0.44–0.60. Unblocked the FB-vs-PSM task-distribution analysis (done).

- **crl_state**, **onestep**, **rldp** — older sessions (created May 25–28).
  Appeared idle at the encmatch launch (all GPUs 0% util, no `train.py` procs),
  so likely finished runs whose sessions were never killed. Verify before reusing
  GPUs or killing.
