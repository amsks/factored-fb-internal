---
name: handoff-iwr-program
description: "MASTER HANDOFF (2026-06-12): state of the gap-analysis/IWR program — what's running, what's done, exact next steps. Read FIRST on session resume."
metadata: 
  node_type: memory
  type: project
  originSessionId: b798f8b4-b3ec-4145-b887-98d31d022aab
---

**The program:** why FB/PSM/CRL fail at manipulation + fixes. Full narrative ALREADY WRITTEN
in `PAPER/fb-iwr/fb-iwr.tex` (the investigation paper, beginner-accessible, all numbers) and
`PAPER/specs/2026-06-11-fb-psm-crl-gap-analysis-and-program.md` (working log with verdicts) +
`PAPER/specs/2026-06-11-interaction-weighted-fb.md` (IWR/zHER derivations). Read those for ALL
results; this note is logistics. Code committed locally on `feat/psm-integration`
(0ab965d..0173f7d, 4 commits 2026-06-12; push blocked — no GitHub auth on box).

**Core verdicts (details in the tex/spec):** failures post-control-acquisition; basis fine
(PSM lift 0.81) / value blind (cf-AUC≈0.5 vs GCIQL 0.89) / actor exonerated (GCIQL-Flow 72.3) /
inference fixes fail (z = policy index, decoupled semantics) / MLP readout on FB goals = chance
0.50 ⇒ **goal COUPLING carries the controllability signal** / gradient audit: FB measure term
active yet lift-blind = persistence-ordering is the objective's preference; ortho1000 = whitening
protection. Mechanism: capacity ∝ persistence × ρ-mass; traj-future goals = (1−γ)M^β coupling =
CRL's positives; renewal decomposition across contact set.

**RUNNING (all in tmux, survive chat closure):**
- `fb_iwr` — fb_state_iwr + fb_state_iwr_wz (IWR ±weight_z), 3 seeds @500k, GPUs 4-7,
  logs `/dev/shm/fb_iwr_20260611_*/`. ~done 2026-06-12 ~03:00.
- `psm_z50_o1000` — 2×2 factorial PSM z_dim=50 cell, 10 seeds @1M, GPUs 0-3 (~475k at 23:50).
- `fb_zher_queue` — WATCHER inside tmux: waits for psm_z50 final.pt 10/10 (or procs gone), then
  auto-launches `fb_state_zher` 3 seeds @500k on GPUs 0-2 (`/tmp/launch_zher_when_free.sh`).
- ~~`readout_traj`~~ **DONE 2026-06-12: NEGATIVE — MLP+traj = 0.446/0.495/0.460 ≈ chance,
  same as uniform.** Coupling alone (frozen basis, relabel-z conditioning) does NOT recover the
  signal. Next factorial (fb-iwr.tex §5.3 update): morph control→GCIQL one ingredient at a time:
  (i) raw obs input vs frozen φ; (ii) exact-goal g conditioning vs relabel-z; (iii) encoder
  trained END-TO-END with the coupled TD loss (vs frozen basis). The flipping ingredient is THE
  answer. NB zHER can still help success via the B3 z-semantics repair even if cf-AUC flat.
- `fb_z128_probes` — task-distribution + z-decode probes on the DONE fb_z128 cell (CPU);
  writes `analysis/task_distribution/{summary.md,z_decode_summary.md}`; `/tmp/z128_probes_done`
  marker when finished.
- ortho100 (psm_state_orthohi ortho_coef=100) still PAUSED (SIGSTOP, state T) — decide: kill or
  `kill -CONT`. Holding GPU memory only.

**RESULTS 2026-06-12 morning (all in fb-iwr.tex §5.3):**
- **IWR: dissociation CONFIRMED, success kill.** Basis: lift-R² 0.02→0.79, separability
  0.74→0.89 (exactly as derived — ρ′ re-orders spectrum, at z50!). Success: 42.0→22.7
  (+weight_z: 16.7). PR dropped 49→31 (basis concentrated on interaction).
- **fb_z128 capacity control:** basis partly capacity-confounded (separability 0.905, lift 0.30
  at d=128) BUT success 35.3 — WORSE than z50's 50.0, lands at PSM-z128's level (36.7).
- **Pattern across all cells: success ANTI-correlates with basis interaction-encoding**
  (plain-z50 lift 0.02/succ 50 → z128 0.30/35 → PSM 0.81/37 → IWR 0.79/23). No in-family
  consumer for interaction info; improving representation alone perturbs the working BC-position
  policy. Candidate proximate cause of IWR drop: B whitened under ρ′ but eval z inferred under ρ
  (GLS mismatch — could test L1 inference on fb_state_iwr ckpts). MLP+traj retrofit also chance.
  ⇒ binding constraint = END-TO-END coupling of representation+value+task (GCIQL's ingredient).

- **IWR+GLS (L1) on IWR ckpts: +13.3 / +19.7** (21.0→34.3, 11.0→30.7) — proximate cause of IWR
  drop = inference-measure mismatch (B white under ρ′, inference under ρ). All 3 ladder regimes
  now theory-consistent. Principled fix: measure-consistent inference z=E_ρ[w·r·B] (weight the
  relabel batch by interaction_weights — `reward_inference(weight=)` hook exists; ~5-line change
  in evaluator/agent eval path, NOT yet implemented). IWR+GLS 34.3 still < plain 42 ⇒ basis still
  has no value-consumer; end-to-end coupling remains the binding constraint.

**MORPH FACTORIAL launched 2026-06-12 ~05:00** (tmux `morph`, `scripts/morph_readout.py`,
CPU): morphs frozen-basis control→GCIQL. Axes --input phi|raw, --cond z|goal (cond=goal uses
env info['goal'] per task), --goal-mode uniform|traj. IQL expectile V/Q on the chosen rep, eval
cf-AUC on stored grasps via V(input(s),cond). Cells running: M1 raw/goal/traj (=GCIQL value
rebuild, endpoint — SMOKE 0.54→0.58 rising, FIRST cell off chance!), M2 phi/goal/traj (frozen
rep test), M3 raw/goal/uniform (traj-coupling test). 3 seeds ea. Out `analysis/morph_readout/
<in>_<cond>_<gm>/sN.json`. Watcher byuv6i37x. (Bug fixed: .norm(-1)→.norm(dim=-1).)
Read: M1≈0.89 & M2≈0.5 ⇒ representation MUST be end-to-end (the structural conclusion);
M1≈0.5 ⇒ harness can't reproduce GCIQL, investigate expectile/arch.

**NEXT STEPS (in order):**
1. Collect MLP+traj final (`analysis/readout_control/s{0,1,2}_traj.json` — NB s0_traj.json may
   be a 3k-step smoke leftover until overwritten) → fill form×coupling table in fb-iwr.tex §5.3.
2. Collect IWR @500k: success from `/dev/shm/fb_iwr_*/fb_state_iwr*_s*.log` (grep
   eval/reward/eval/success) vs fb_state_plain@500k (~0.45-0.50); MECHANISM endpoint: add
   fb_iwr/fb_iwr_wz to `scripts/task_distribution_probe.py` METHODS (+symlink run dirs into
   /dev/shm/factored-fb/runs/) and run z-decode → did lift-R²(B) rise from 0.02?
3. When psm_z50 done: add to METHODS, rerun both probes → complete 2×2 (algorithm × z_dim)
   table; fb_z128 probes already running.
4. zHER results (auto-launched): success + cf-AUC probe (counterfactual_value_probe --method fb
   on its checkpoints) → the bilinear+traj cell.
5. Test dissociation prediction (fb-iwr.tex §5.3): IWR→basis, zHER→value/success. If both hold
   → run COMPOSITION (IWR + zHER + weight_z); promote winners to 10×1M + full probe suite.
6. Paper upkeep: fb-iwr.tex table cells; optionally bib. rlbrew paper already has PSM=ortho1000.

**Key paths/conventions:** probes CPU-only `CUDA_VISIBLE_DEVICES= PYTHONPATH=. .venv/bin/python`,
`--mujoco-gl egl`; checkpoints: use `step_1000000.pt` for 1M-comparable numbers (final.pt = 1.5M
for fb_state_plain + psm_o1000 after extensions!); fb_state_plain runs
`/dev/shm/factored-fb/runs/20260607_070225__fb_state_plain__sN`; psm_o1000
`/dev/shm/factored-fb_outputs/psm_state_orthohi__ortho_coef1000__lr_phi1e-5__sN`; IWR knobs in
`configs/agent/fb.yaml` (reweight_mode=interaction, alpha master switch); zHER =
`train_goal_mode=traj`. Tests: `tests/test_{z_inference,iwr_weights,traj_goals,fb_reweight}.py`
(28 green). /dev/shm is VOLATILE — pull anything precious before reboots.

**2026-06-12 ~15:10 UPDATE:**
- MORPH KILLED (measurement flaw: scored a value-with-no-policy on PSM's cross-policy grasps;
  M1≈M2≈M3≈0.55 at 75k, all flat, not comparable to GCIQL's same-policy 0.89).
- zHER DONE: 42/34/28 mean 34.7 vs plain FB 42.0 — trajectory goals + measure loss HURTS
  success too (like IWR). Changing goal dist within FB's measure framework doesn't help.
- **FB-IQL LAUNCHED** (tmux `fb_iql`, group `fb_state_iql`, GPUs 2/3/7, 3 seeds @500k):
  fb_flowbc + train_goal_mode=traj + q_loss_coef=1.0 + ortho1000. Key: q_loss(471) >> measure(31)
  AND ortho grad doesn't touch F ⇒ forward map F trained PRIMARILY by goal-reaching value-TD
  (next_Q from target-F + policy action; SARSA-ish, BC-anchored). The bilinear Q=F·z the actor
  steers on is now a goal-conditioned TD value on trajectory goals — the in-FB recipe, NO code.
  Logs /dev/shm/fb_iql_<ts>/. Read vs plain 42.0; if up → bilinear FB fixable in-setup.
  If flat/down → need clean expectile-IQL critic REPLACING measure loss (V-net + expectile,
  ~60 lines in update_fb; design in this note's NEXT STEPS / discussed).

**2026-06-14 UPDATE — FB-IQL results + critic_mode=iql divergence bug FIXED:**
- **In-FB recipe** (`fb_state_iql`, q_loss_coef=1.0, measure loss ON; `/dev/shm/fb_iql_20260612_150544`):
  DONE @500k = 0.38/0.32/0.32 → mean **0.34** (peak 0.43), returns −154/−163/−169. BELOW plain
  FB 0.42. Up-weighting goal-TD on the bilinear doesn't help — lands with IWR(0.23)/zHER(0.35).
- **Clean critic_mode=iql** (commit 128c4a7, replaces measure loss; `/dev/shm/fb_iqlcritic_20260614_080910`):
  COLLAPSED — peaked ~0.24 @100k then →0.00, returns at −200 floor. ROOT CAUSE (debugged, confirmed
  vs OGBench GCIQL ref): bootstrap used reward {0,+1} + masked by env `terminated` flag (never set in
  cube offline data) instead of goal-reached ⇒ fixed point 1/(1−γ)=100, V/Q ramp linearly (0→43@14k),
  actor collapses. **FIX (commit f083a76, on feat/psm-integration):** extract `iql_goal_target(success,
  Vn, discount)` = (success−1)+γ·(1−success)·Vn, byte-faithful to GCIQL (datasets.py masks/rewards +
  gciql.py backup). Verified: 60k smoke V_mean bounded & converging NEGATIVE (−8.7) not +100; 11/11
  test_fb_iql (3 new), 39/39 FB-family. **CAVEAT:** smoke success still 0.00 @50k — but broken run was
  also only 0.06@50k, didn't peak till 100k; 60k too short to judge. Bilinear may still be value-blind
  (cf-AUC≈0.5) ⇒ a correct-but-uninformative value would confirm the thesis, not refute it.
**2026-06-15 — GCIQL→FB direction (the reciprocal of FB-IQL) — MAJOR:**
- **The bilinear FORM is NOT the wall.** `gciql_flow` already has `value_arch=bilinear` (Q=φ(s,a)·ψ(g)/√d,
  the SAME form as PSM/FB), IQL-trained, exact-goal eval, flow actor. Existing run `GCIQL-State-Flow-Bilinear`
  (5 seeds×1M, d=512, wandb amsks/factored-fb) = **0.93 overall_success** — BEATS the MLP critic (`GCIQL-State-Flow`
  = 0.73). An FB-shaped bilinear value trained by IQL with the exact goal essentially solves cube_single.
- **Phase diagnostic on G3** (`scripts/phase_probe_gciql.py` = adapted phase_probe_crl; 4 good seeds — sd003 failed,
  no ckpt; out `analysis/phase_probe_gciql_bilinear/`). S0 funnel: reached 100 → secured 96.5 → **success 95.0**;
  residual fails GRASP 3.5% / transport 1.5%. vs FB-family which all COLLAPSE at transport: PSM 88→**29**
  (fail@transport 59%), RLDP 85→**26** (59%), PSM-o1000 80→**38** (41%). **G3 closes the post-control-acquisition
  (transport) gap that defines FB's failure.** Funnel fig `PAPER/rlbrew/figures/fig_funnel_comparison.png` +
  `scripts/plot_funnel_comparison.py`. (NB the per-method `overall.md` "Readout" is a hardcoded template — wrong
  for G3; trust the parquet-computed composition.)
- **CONFOUND (not yet split):** G3=0.95 vs PSM=0.29 differ on THREE axes — objective (IQL-value vs measure),
  conditioning (exact goal vs zero-shot z=E[rB]), rank (d=512 vs ~50). Phase probe shows WHERE (transport), not WHICH.
- **CONDITIONING ARM DONE — inference is NOT G3's edge.** `scripts/infer_ablation_gciql.py` (value-level cf-AUC
  on the bilinear value net: z_exact=ψ(goal) vs z_inferred=E[r·ψ(s')], FB's L0 rule; AUC scale-invariant in z so
  no projection issue; faithful reward = cube qpos[14:17] dist<0.04 via env rollouts — NB obs[19:21]=cube xyz but
  per-axis scale/offset unrecoverable, so MUST use privileged qpos, dataset has none). 4 seeds (d=512): cf-AUC
  **exact 0.922±0.011 vs inferred 0.915±0.010 — IDENTICAL.** G3's value ranks goal-reaching states equally well
  under FB-style zero-shot inference; the same inference leaves FB at ~0.5. ⇒ conditioning/inference is NOT the
  differentiator. Form exonerated (G3=0.93) + inference exonerated ⇒ by elimination the binding ingredient is the
  OBJECTIVE (IQL value-TD vs measure-matching), pending the rank axis. Caveats: value-level (success-level needs
  z-cond actor rebuild = backburnered G4); on-policy state dist (FB's 0.5 was on ρ relabel batch). Out
  `analysis/infer_ablation_gciql/s{0,1,2,4}.json`.
- **BRIDGE RANK SWEEP — DONE 2026-06-15, RANK EXONERATED.** All 6 runs finished @500k (wandb amsks/factored-fb,
  groups `GCIQL-State-Flow-Bilinear-d{50,128}`; ckpts `/dev/shm/gciql_outputs/.../params_500000.pkl`):
  **d50 mean 0.899** (0.92/0.912/0.864), **d128 mean 0.899** (0.872/0.912/0.912) vs **d512 0.928** (5 seeds).
  At FB-parity rank d=50 the GCIQL bilinear value STILL scores ~0.90 — 10× rank cut costs ~3 pts. ⇒ rank is NOT
  G3's crutch. **All three axes now eliminated: form (0.93), inference (cf-AUC 0.92≈0.92), rank (d50≈d512).**
  By elimination the binding ingredient is the OBJECTIVE (IQL value-TD vs measure-matching). The paper's spine =
  FB-IQL(0.20)↔G3(0.93) bridge; reciprocal already confirms (correct critic_mode=iql on FB bilinear = 0.20,
  measure-trained rep is value-blind). NEXT: write this into fb-iwr.tex; success-level objective-isolation
  (z-cond actor on IQL-value rep, the backburnered G4) is the only remaining arm.
- **Gotchas:** run_gciql.py runs in TORCH .venv (has hydra), dispatches main.py to `/dev/shm/.venv-jax` via
  `storage=shm` (jax_python); phase_probe needs JAX_PLATFORMS=cpu (5 parallel GPU jax = OOM) + pandas/pyarrow
  (uv pip install into /dev/shm/.venv-jax, done); `pkill -f phase_probe_gciql` self-matches the shell → exit 144;
  set `save_interval≤train_steps` or no checkpoint is saved.
- IMPLEMENTATION (G2 MLP-bottleneck rung + spec `PAPER/specs/2026-06-14-gciql-fb-morph-ladder.md`) is BACKBURNERED —
  the spec's G2-vs-G3 'form' framing was partly mooted by G3=0.93 (form exonerated); revise to the
  FB-IQL(0.20)↔G3(0.93) bridge before resuming.

- **DONE — fixed critic_mode=iql full 3-seed @500k** (`/dev/shm/fb_iqlcritic_20260614_iqlfix_full`,
  group `fb_state_iqlcritic_fix`): success final 0.26/0.16/0.18 → **mean 0.20** (peak 0.27), returns
  −163/−178/−173. Value bounded & stable to 500k (V≈−10.3, no collapse). VERDICT: the fix kills the
  divergence (collapsing 0.03 → stable 0.20 that learns & holds), BUT a correctly-trained goal-cond
  IQL value on the bilinear Q=F·z still LOSES to plain FB 0.42 — thesis-CONFIRMING: bilinear is
  value-blind, correct value ≠ better policy; binding constraint = end-to-end coupling (GCIQL's
  ingredient), not the value-target form. Lands with IWR(0.23)/zHER(0.35)/q_loss_coef(0.34): nothing
  inside the FB bilinear framework recovers controllability. Fix = commit f083a76.

**2026-06-16 — G4 (success-level objective isolation) BUILT + LAUNCHED:**
- The last open arm. Everything pinning "objective is the lever" was VALUE-level (cf-AUC). G3's 0.93
  uses an EXACT-goal actor; FB uses a zero-shot z-cond actor. G4 puts the IQL-trained bilinear rep into
  FB's full zero-shot pipeline (bilinear + z-inference + z-cond actor) and measures SUCCESS. Stays ~0.9 ⇒
  objective confirmed at the policy level; collapses ⇒ a success-level conditioning effect invisible to ranking.
- **Impl (committed on `feat/psm-integration`):** one default-off flag `actor_cond: 'goal'|'z'` on the
  `gciql_flow` agent (`tools/wandb_mode_shim/gciql_flow.py`). `'z'` requires `value_arch='bilinear'`; actor
  conditions on `z=stop_grad(value.psi(goal))` (B := value net's psi, single shared backward map; stop-grad
  keeps the rep purely IQL-trained). KEY contract fix found via smoke: OGBench's vendored eval loop hands the
  actor the RAW goal, so `sample_actions` maps goal→psi internally (in-loop eval thus reports EXACT-z success);
  `sample_actions_with_z` feeds a precomputed z for the inferred-z arm. Tests `tests/test_gciql_zcond.py` (8,
  incl. stop-grad-into-value); 13/13 gciql-family green. Spec `PAPER/specs/2026-06-16-g4-success-level-
  objective-isolation{,-plan}.md`.
- **Eval:** `scripts/eval_zcond_gciql.py` — per task, exact-z (raw goal→psi) + inferred-z (z=E[r·psi(s')],
  relabel batch from exact-z rollouts, faithful cube-qpos reward thresh 0.04) → SUCCESS. Out
  `analysis/eval_zcond_gciql/s{seed}.json`. Run: `JAX_PLATFORMS=cpu MUJOCO_GL=egl /dev/shm/.venv-jax/bin/python
  scripts/eval_zcond_gciql.py --checkpoint-dir <run_dir> --checkpoint-step 500000 --seed <s>
  --value-latent-dim 512 --episodes 50 --out analysis/eval_zcond_gciql/s<s>.json`.
- **RUNNING:** tmux `g4_s{0,1,2}`, GPUs 0/1/2, 3 seeds @500k, wandb amsks/factored-fb group `gciql_g4_zcond`,
  logs `/dev/shm/g4_zcond_s{0,1,2}.log` (see [[tmux-sessions-registry]]). Launched 2026-06-16 19:33, ETA ~hours.
- **DONE 2026-06-16 ~23:17 — G4 RESULT = sharp DISSOCIATION** (`analysis/eval_zcond_gciql/s{0,1,2}.json`):
  **exact-z $z=\psi(g)$: 86.0±3.5** (83/84/91) — z-cond actor on the IQL rep nearly matches the 94.8 bridge
  (~9pt cost of routing goal through ψ bottleneck + z-cond vs raw goal). **inferred-z $z=E[r\psi(s')]$: 8.1±7.0**
  (2/4/18) — COLLAPSES. So objective+form+rank+z-cond COMPOSE at the policy level (50→86), but FB's zero-shot
  inference→actor is an INDEPENDENT success-level bottleneck that value-ranking (cf-AUC 0.92 inferred) MISSED.
  = the predicted "collapse ⇒ success-level conditioning effect invisible to value-ranking" branch.
  **CAVEAT (confound, not yet split):** the G4 actor was trained ONLY on exact-goal z=ψ(g), so inferred-z is
  OOD for the actor (relabel pos-frac ≈2%); the 8.1 conflates "FB inference fragile" with "actor never saw
  inferred z." CLEAN NEXT CONTROL = train the actor across the inferred-z distribution (canonical FB samples z
  from random-reward readouts) → separates the two. Written into `PAPER/fb-gciql-bridge/fb-gciql-bridge.{tex,pdf}`
  (the standalone synthesis doc, 2pp; compiled via /tmp/tectonic musl). Two-layer conclusion: (i) objective sets
  whether the rep's value is controllable (50→86); (ii) inference→actor is a separate zero-shot bottleneck.
- **2026-06-17 — RANK FLOOR found (d=5 breaks GCIQL).** Extended the bilinear rank sweep to d=5
  (`GCIQL-State-Flow-Bilinear-d5`, 3 seeds @500k, exact-goal/raw actor, config
  `configs/gciql/cube_single_state_gciql_flow_bilinear_d5.yaml`): **d=5 = 6±2** (4/6/8; 2% already at
  100k vs ~70% for d≥50) — COLLAPSE. Full sweep @500k: d5 6±2 / d50 90±3 / d128 90±2 / d512 88±2.
  ⇒ capacity cliff between d=5 and d=50, but FAR below FB's range (d=50-128), so rank still NOT FB's
  crutch. Written into `PAPER/fb-gciql-bridge` rank section. NB watcher glob gotcha: `*Bilinear-d5*`
  also matches `d50` (substring) — scope finds to the exact dir.
- The standalone doc `PAPER/fb-gciql-bridge/` is the self-contained writeup (objectives math + 3 modifications
  M1/M2/M3 + all wandb-pulled numbers + G4). Build: `cd PAPER/fb-gciql-bridge && /tmp/tectonic -X compile fb-gciql-bridge.tex`
  (tectonic musl binary in /tmp, volatile; re-download from gh releases if gone; needs ~/.cache/tectonic/formats/).
