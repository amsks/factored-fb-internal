# rescuing-fb — running log (append-only)

## 2026-07-16 — program start
- Folder created; synthesis moved in as `01-synthesis-deep-audit.md`; specs `02` (RG-FB) and
  `03` (probes T1–T3) pre-registered BEFORE any probe ran.
- GPU state at dispatch: GPU1 = other track dueling, GPU5/6 = other track audit; free {0,2,3,4,7}.
  Live other-track tmux: `dueling`, `audit`, `exp23`, `exp5` — DO NOT TOUCH.
- Dispatched (parallel agents): T1 goal-smoothed rerank, T2 DR3 stats, T3 off-peak×domain,
  RG-FB v0 implementation (+ 3-seed cube launch in tmux `rgfb`, output_root=~/exp_rgfb).

## 2026-07-16 — T3 off-support peak × domain: DONE (phenomenon confirmed, coverage explanation refuted)
- Probe `scripts/rescuing_fb/t3_offpeak_multienv.py` (CPU jax, no BC flow; B_ref(s) = dataset
  action at s + 64-NN dataset actions, whitened obs) → `analysis/rescuing-fb/
  t3_offpeak_multienv.json` + `t3_notes.md`. 512 states × 5 eval goals, seed-0 DDPGBC ckpts @500k.
- **Cube sanity gate PASS**: peak frac>0 = 0.639, median +1.64 (vs Finding-1's 0.79/+2.14 with
  the μ-flow cloud — expected shrink under the more generous dataset-NN reference).
- **Antmaze prediction CONFIRMED decisively**: frac 0.003 (≤0.25 predicted), median −0.80 (≤0
  predicted); actor departure 0.82 (inside the behavior cloud) vs cube 3.34. The off-support
  peak DDPG+BC climbs on cube does not exist on antmaze ⇒ explains the bilinear+DDPGBC domain
  flip (antmaze 0.837 win / cube 0.51 loss).
- **Coverage sub-prediction FAILED**: after dimension-correcting with a same-size uniform null
  (A=5 vs 8 confound), cube's conditional action manifold is only 1.11× narrower than antmaze's
  (narrowness 1.73 vs 1.56) — raw action-space coverage cannot explain a 0.639→0.003 flip. The
  peak's domain-specificity needs a value-geometry account (Cost-A residual relocation), not a
  data-coverage one; "broaden the data" is not a viable fix, unified selection-based extraction
  remains the road.
- **Scene exploratory**: no peak (frac 0.014, median −1.05) — scene's DDPGBC failure (0.20) is a
  different mechanism. Caveat: scene cell trained at α=1.0 (cube/antmaze 0.3), so its low
  departure is partly BC-pinning; cube-vs-antmaze is α-matched.

## 2026-07-16 — T2 DR3 co-adaptation: DONE (headline φ-statistic REFUTED; runaway lives on the ψ(s,g) leg)
- Probe `scripts/rescuing_fb/t2_dr3_stats.py` (jax CPU; fixed B=4096 batch, a′=dataset next
  action, training value-goal mixture) → `analysis/rescuing-fb/t2_dr3_stats.json`,
  `t2_coad_vs_step.png`, `t2_notes.md`. 25 ckpts: exp5 bilinear+mlp 50k→500k, ref bilinear
  sd000 @500k+1M, 3 sg seeds @500k (sg saved final-only ⇒ pre-registered fallback: train.csv
  curves + cross-seed final comparison; s0 stable 0.692, s1 div 0.284, s2 div 0.000).
- **(a) REFUTED**: normalized coad bilinear 0.990 > sg 0.960–0.981 > mlp 0.93–0.97 — sg is NOT
  ≫ bilinear; the statistic is near-saturated for every arch (adjacent transitions ⇒ smooth
  features ≈1). Within-sg half holds directionally (divergent 0.981/0.978 > stable 0.960).
- **(c) CONFIRMED**: bilinear flat 0.990±0.001 (also 500k→1M), mlp mildly declining — no runaway.
- **The actual divergence trace is on ψ(s,g)**, the leg that got s: E‖ψ‖² 1083 (stable) → 2355 →
  3789 (diverged), ψ(s,g)·ψ(s′,g) coad 0.9857 → 0.9924 → 0.9960, q_max +0.2 → +778 → +1177 —
  all monotone in divergence severity, while E‖φ‖² stays small (~100). φ towers stay sane.
- **Booking**: "ψ(g) bottleneck = stabilizer" survives in modified form — removing it opens a
  norm/alignment runaway on the state-conditioned ψ leg, not DR3's growing φ·φ′ dot products.
  Per the spec's failure clause, book the sg divergence as no-output-bound + no-V-target + an
  unbounded state-conditioned leg. If the mechanism figure is wanted for the paper, rerun sg
  with save_interval=50k to make prediction (b) (coad tracks q_mean→0) properly testable.

## 2026-07-16 — T10 centered measure PROMOTED (user call) + spec pre-registered
- User intuition: centered measure is worth a real shot. Spec written BEFORE implementation:
  `04-centered-measure-spec.md`. Two arms: T10-a (IQL baseline + centered-measure advantage,
  coexistence read, success ≥0.85) and T10-b (measure-only: behavior-anchored M̄ leg via SARSA
  bootstrap [contraction] + centered ΔM leg via π bootstrap; ≥0.5 = measure-native rescue).
- Claim boundary (joint with T8 one-step FB) pre-registered in the spec §3.
- Key risk encoded: the scalar-TD dueling analog (Arch 11) FLOORED — T10 normalizes the actor
  q-term by sg|Q_full| and the actor consumes U+ΔM, never ΔM alone; read dueling_v2's result
  before implementing if available.
- Dispatch: BLOCKED on gciql_flow.py ownership until the RG-FB v0 agent completes; T10
  implementation agent goes out immediately after.

## 2026-07-16 — RG-FB v0 IMPLEMENTED + LAUNCHED (3 seeds, tmux `rgfb`, GPUs 2/3/4)
- **Code** (`tools/wandb_mode_shim/gciql_flow.py`): new knob `actor_vf_cond ∈ {'none','goal','z'}`
  (default 'none' = bit-identical stock behavior, byte-identical param tree ⇒ every existing
  checkpoint keeps loading). 'z' feeds the BC flow vector field AND the distillation rollout
  `[obs, sg(ψ(goal))]` — the SAME task vector the one-shot z-actor consumes (`_actor_z`);
  'goal' concatenates the raw goal (awr_flow parity). New helper `_vf_obs` (gf:~514); threaded
  at the BC flow-matching loss and the `_flow_rollout` distillation target; vf example inputs
  sized in `create()`. Assert lattice: 'z' requires `actor_cond='z'` (transitively bilinear),
  excludes `actor_z_infer`; any conditioning requires `actor_kind='flow'` + state obs.
- **Tests** (`tests/test_gciql_rgfb.py`, 4 tests, all pass): (a) 'none'/knob-absent configs give
  identical param trees + bitwise-equal update() losses on a fixed batch, and `_vf_obs` is a
  passthrough; (b) 'z': vf input widened by d, actor_loss sends ZERO grad into value net AND
  critic (sg verified via jax.grad), vf goal-dependent, e2e update + sampling finite; (b') 'goal'
  works on the mlp arch; (c) 4 assert-lattice refusals. Full existing suite: **126 passed,
  2 failed — both PRE-EXISTING/environmental** (missing `scripts/run_gciql_visual.sh` on disk;
  `rliable` not installed in .venv for a figures test); no gciql_flow test failed.
- **Eval script** `scripts/rescuing_fb/eval_rgfb_zeroshot.py`: any-run-dir (arch knobs read from
  the run's flags.json), moment inference z=Σ r·ψ(s′), GOAL-FREE norm rule ‖z‖ = mean ‖ψ(s′)‖
  over the relabel sample ('match_data') + 'sqrtd' both reported, exact-z arm clearly labeled
  as diagnostic; JSON → `analysis/rescuing-fb/rgfb_zeroshot_<group>.json`. **Smoke-tested on a
  real G4-zcond d512 checkpoint** (task1, 2 eps): exact=1.0, zeroshot match_data=1.0, cos=0.90,
  goal-free norm 33.7 vs ‖ψ(g)‖ 35.6 — pipeline + old-checkpoint restore verified.
- **Launch**: config `configs/gciql/cube_single_state_rgfb_d50.yaml` (bilinear d50 anchor +
  actor_cond=z + actor_vf_cond=z, NO measure aux, eval every 50k, save every 100k), script
  `scripts/rescuing_fb/launch_rgfb_v0.sh`. 3 seeds (0/1/2) on GPUs 2/3/4, one proc each,
  XLA_PYTHON_CLIENT_PREALLOCATE=false, tmux **`rgfb`**, wandb amsks/factored-fb group
  **RGFB-Cube-Single-State-d50-v0**, output_root=/home/adityamohan/exp_rgfb (persistent).
  flags.json verified: bilinear/d50/actor_cond=z/actor_vf_cond=z/bc_coeff=3/measure_coef=0,
  goal mix 0.2/0.5/0.3, gc_negative. ~70-90 it/s ⇒ ~2-3h to 500k.
- **First-steps health (verified)**: q_mean −13…−20 (negative, bounded ✓), critic/value losses
  O(1), bc_flow ~0.16-0.20, distill ~0.01; seed 1 in-training exact-z success @50k = **0.548**
  (healthy ramp toward the G4 0.921 anchor).
- **How to check progress**: `tmux attach -t rgfb`; logs `/dev/shm/rgfb_logs/rgfb_v0_s{0,1,2}.log`;
  curves `~/exp_rgfb/factored-fb/RGFB-Cube-Single-State-d50-v0/rgfb_v0_s*/{train,eval}.csv`
  (eval col = exact-z diagnostic). At 500k run:
  `JAX_PLATFORMS=cpu MUJOCO_GL=egl /dev/shm/.venv-jax/bin/python scripts/rescuing_fb/eval_rgfb_zeroshot.py
   --run-dirs '~/exp_rgfb/factored-fb/RGFB-Cube-Single-State-d50-v0/*' --checkpoint-step 500000
   --episodes 50 --relabel-size 50000`. Then add the group to `scripts/results_map.py` + refresh CSVs.
- **Pre-registered reads (spec §3, binding)**: exact-z ≥0.88 success / <0.80 fail (fall back
  actor_vf_cond='none'); **zeroshot ≥0.90 headline** / <0.75 fail (norm hygiene or d50-vs-d512 →
  d512 fallback); zeroshot−exact gap ≤3pts / >10pts → E1a inference ladder. Anchors: plain FB
  0.42, supervised bilinear GCIQL 0.904, published GCIQL 0.68. Failure-isolation order:
  (1) vf_cond=none rerun, (2) d512+1M, (3) E1a ladder on the v0 ckpt.
- Deviation from spec: save_interval=100k (spec silent; midway ckpts enable the isolation
  ladder without a rerun). gciql_flow.py ownership released after this entry.

## 2026-07-16 — T6 generality probe: DONE (as-coded REFUTED, but outside the pre-registered taxonomy — novel region-goals WORK, composite rewards FAIL)
- Spec pre-registered BEFORE rollouts: `05-t6-generality-spec.md`; probe
  `scripts/rescuing_fb/t6_generality_probe.py`; raw `analysis/rescuing-fb/t6_generality_g4.json`
  + `t6_notes.md`. **Ran on G4 (d512/1M, actor_cond=z, actor_vf UNconditioned) — MUST be
  re-run on the RG-FB v0 checkpoint @500k as confirmation.** CPU-only, cores 0-47 (user cap),
  tmux `t6`, 25 eps/cell H=200, batched lockstep envs.
- **GATE PASS**: anchors (dataset reward, moment z, match_g) 0.912/0.920 ≥ 0.90 (E1a 0.955
  band). Goal-free `match_data` norm anchors 0.952 — RG-FB's norm rule validated at d512.
  Synthetic 1[dist<0.04] ≡ dataset relabel reward exactly (agreement 1.0).
- **(a) two-goal max-mixtures 0.260** (only t2-t3=1.0, five pairs ~0). cos(z, both ψ_g)
  stays high (0.77–0.82) ⇒ the failure is the actor consuming an OFF-MANIFOLD blended z, not
  broken inference geometry: moment inference is linear, the mixture's optimal policy is not.
- **(c) novel region-goals 0.672 final / 0.917 any-step** (controls 0.003) — zero-shot reaches
  6cm boxes ≥6cm away from every trained eval goal. The goal-reaching prior covers the goal
  FAMILY, not the eval-goal list.
- **(b) lift-anywhere 0.944 UNCERTIFIABLE**: uniform-reward control also lifts 0.640
  (match_data) / 0.944 (sqrtd) — default grasp-and-hold confound; cos(z_lift, mean psi)=0.970.
  Pre-registered clause (d) triggers for the lift metric only; goal/region metrics clean
  (uniform: own-goal 0.064, regions 0.003).
- **Claim scope for the paper (binding)**: license *zero-shot goal-reaching for arbitrary
  dataset-supported cube-position goals*; do NOT print "zero-shot RL"/BFM. Empirically
  motivates the §6 v1 BFM path (task-mixture hindsight relabeling = z-space coverage fix).
- sqrtd norm collapses anchors t4/t5 (0.36/0.08) — norm hygiene (E1a) re-confirmed at d512.

## 2026-07-16 — T1 goal-smoothed reranking: DONE (inference-time pooling REFUTED; control confirmed)
- Probe `scripts/rescuing_fb/t1_goal_smooth_rerank.py` (Exp-1/C3 batched-rollout protocol,
  N=32 shared-flow candidates, 50 eps/task × 5 tasks, seed-0 ckpts: bilinear-d50 flow @500k,
  MLP flow @1M; K goal variants batched into one critic pass) → `analysis/rescuing-fb/
  t1_goal_smooth_rerank.json` + `t1_notes.md` + `t1_run.log`. Unit VERIFIED vs
  `cube_env.compute_observation`: obs[19:22]=(xyz−c)·10 ⇒ σ = 0.2/0.4 obs units = 2/4 cm.
- **K=1 gate PASS**: bilinear 0.924 / MLP 0.616 (known 0.944–0.96 / 0.568–0.592; E=50 vs 25).
- **Pooling-TRUE prediction REFUTED**: MLP never rises — it FALLS monotonically in σ
  (K=32: 0.616 → 0.580 @2cm → 0.408 @4cm) and is ~flat in K at fixed σ. Eval-time MC goal
  smoothing destroys the MLP's goal-local advantage instead of denoising it (consistent with
  dGOAL −16…−24 brittleness). Pooling-FALSE branch survives, amended "flat in K, declining in σ".
- **Bilinear ~invariant at 2 cm** (K=32 0.912 vs 0.924), mild real dip at 4 cm (0.864, ≈3 SE;
  4 cm = success radius, partly geometric). Directionally as predicted for a rank-50 ψ(g) head.
- **Random-goal control CONFIRMED both**: bilinear 0.148, MLP 0.040 (≈ μ-floor 0.06) — the
  rerank signal is goal-specific for both critics. (Quirk: bilinear ctrl task4=0.70, others ~0.)
- **Booking for §2.2**: if advantage pooling is the mechanism it acts at TRAINING time (the
  ψ(g) bottleneck as a cross-goal parameter-sharing constraint on φ during TD), not as a
  property recoverable by eval-time smoothing — T5 (param-matched/two-tower) is the discriminating
  control. Kills the cheap "smooth at eval" shim for RG-FB; keep the structural low-rank head.
- Deviations: none material. Unpinned choices logged in t1_notes.md (goal cloud fixed per
  (arm,task); K variants exclude exact g; control = uniform dataset obs, per-task seeded).

## 2026-07-16 — RG-FB v0 training DONE (500k, crash-free); zero-shot eval RUNNING
- All 3 seeds reached 500k (~1h20 wall each); q_mean −19.3/−19.7/−20.4 (bounded ✓); ckpts every
  100k in `~/exp_rgfb/factored-fb/RGFB-Cube-Single-State-d50-v0/rgfb_v0_s{0,1,2}/`.
- **In-training exact-z diagnostic @500k (50 eps): 0.844 / 0.820 / 0.824, mean 0.829** — GRAY
  ZONE vs the pre-registered bands (success ≥0.88, fail <0.80). Not booked as a fail: the
  binding call uses the eval-script reads; note the G4 anchor 0.921 was d512 (v0 is d50, and
  the supervised bilinear-d50 anchor is 0.904 with a goal-actor, not z).
- Headline zero-shot eval launched in tmux `rgfb` (CPU): eval_rgfb_zeroshot.py, 5 tasks × 50
  eps × 3 seeds, log `/dev/shm/rgfb_zeroshot_eval.log`, JSON →
  `analysis/rescuing-fb/rgfb_zeroshot_RGFB-Cube-Single-State-d50-v0.json`. Reads to apply on
  completion: zeroshot ≥0.90 success / <0.75 fail; gap vs exact ≤3pts / >10pts (spec §3).
- CORRECTION (same day): the eval launch above was a DUPLICATE of the main session's identical
  eval (core-capped, log `rgfb_zeroshot_500k.log` in its scratchpad) and was killed by the main
  session — not a failure. The main session owns the headline zero-shot read + its DECISIONS
  entry, and has launched the pre-registered isolation control (actor_vf_cond=none, 3 seeds,
  group RGFB-Cube-Single-State-d50-vfnone-ctl, tmux `rgfb_ctl`) given exact-z 0.829 < 0.88.

## 2026-07-16 — RG-FB v0 HEADLINE + T10 mid-read + T10-b2 relaunch (main session)
- **RG-FB v0 zero-shot @500k: 0.825 ± 0.014** (exact-z 0.852; gap 2.7 pts ≤ the pre-registered
  3-pt band ⇒ **inference is CLEAN**; sqrtd norm again collapses, 0.365 — norm hygiene
  re-confirmed). Above published GCIQL 0.717, ~2× FB 0.42, below the 0.90 target; shortfall is
  training-side. Isolation control `actor_vf_cond=none` (3 seeds, group
  RGFB-Cube-Single-State-d50-vfnone-ctl, tmux rgfb_ctl, GPUs 2/3/4) RUNNING — attributes the
  gap to the vf knob vs the d50-z-actor combo.
- **T10-a (aux) @150k: 0.708–0.780, healthy**, tracking just under the bilinear anchor. No
  dueling-style floor — a centered measure objective coexists with scalar grounding.
- **T10-b (only) KILLED @~160k**: floored 0.06–0.08 with a diagnosed **scale runaway**
  (q_mean +1e8→+3.3e13, norms→1e14, M²→1e27; b_fidelity fine 0.09). Cause: scale-norming BOTH
  measure legs made the objective scale-invariant; with no scalar TD to pin absolute scale the
  diag-attraction inflates without bound. NOT the honest test of centering — claim boundary
  not settled by this cell. The T2 early-warning telemetry caught it exactly as designed.
- **T10-b2 (anchored) RELAUNCHED**: new knob `centered_scale_anchors` (default-off; 6/6 t10
  tests green, running T10-a unaffected) = FB's own anchors restored — sphere-projected ψ
  (‖ψ‖=√d) everywhere + RAW (un-normalized) regression on the contractive SARSA base leg +
  ortho(ψ)=1.0. Config `cube_single_state_t10b2_anchored_d50.yaml`, group
  GCIQL-State-Flow-CenteredMeasure-Only-Anchored-d50, 3 seeds staggered on GPU 7, tmux t10:b2.

## 2026-07-16 — meeting-notes doc: figures + number audit + Socratic restructure
- `meeting-notes-amy.tex` rebuilt as a Socratic why-chain (10 rungs + bedrock of open ends);
  compiled clean to `meeting-notes-amy.pdf`. Six figures in `figs/` (validated dataviz palette,
  fixed entity colors: bilinear blue / MLP green / FB pink): anchors, rerank N-sweep (the
  mechanism figure), T1 smoothing, OOD grid, RG-FB curves (live from eval.csvs), T3 peak.
- Number audit vs source CSVs — 3 corrections: B1 fbz-under-IQL 0.848→**0.844**
  (bridge_grafts.csv), measure-objective range 0.10–0.34→**0.08–0.34**, E2 λ≤1 band
  0.86–0.92→**0.86–0.90**. All other stated numbers verified against results/*.csv,
  analysis/rescuing-fb/*.json, findings-stream.
- **vf-none control update (from the fig pull)**: s1 0.816@500k, s2 0.888@500k, s0 0.816@350k.
  Control exact-z ≈ 0.84–0.85 vs v0's 0.829 ⇒ the vf-z knob costs ~2 pts (not the whole gap);
  the rest is the d50-z-actor combo vs the d512 G4 anchor (0.921). Ladder next step if we
  chase the last points: d512 rerun.

## 2026-07-16 — T10-b2 (anchored measure-only) FIRST EVAL: 0.232 @50k, seed 0
- ~3× the historical measure-only ceiling (0.076–0.088) at 1/10 of the budget; un-anchored
  version floored at 0.06. In the pre-registered "partial" band already. One seed, M-scale
  still decelerating-but-growing — watch 100k+ for plateau vs blow-up. Seeds 1/2 ramping.
- If ≥0.5 @500k: the measure-native rescue is real (centering + scale anchors + split
  bootstrap = the fix). If 0.3–0.5 plateau: partial rescue, residual = missing counterfactual
  support in ρ (the temporal-misalignment cap).

## 2026-07-16 — split into two LIVING documents
- `understanding-bilinear.tex` (161KB pdf): full Socratic chain over every experiment with
  conference-level training details; bedrock = 6 open questions ONLY about why bilinear Q
  works, each with a queued experiment (T5 late-fusion/param controls, peak-geometry norm
  probe, scene, T9 true-advantage rank, bounded-psi(s,g) regime test, srank trajectories);
  related-work survey incl. VIP/LIV, METRA, HILP, BVN, CRL, QRL/IQE/MRN, linear/low-rank
  MDPs, DR3/rank-collapse, HL-Gauss/SimbaV2, one-step/IDQL/SfBC/FQL. A verified-citation
  sweep agent is appending additions.
- `fb-variants-and-grounding.tex` (120KB pdf): implementation descriptions of every FB
  variant (torch + jax grafts + PSM context), the 5 annealing cells with the three lessons
  (curricula cross cliffs not fixed points; scalar TD is the load-bearing anchor; measure
  hurts through phi not psi), and the reward-grounding recipe with 4 instances (RG-FB
  measured, grounded-PSM proposed, T10 centered measure running incl. b2 0.221@50k,
  curriculum-for-constraint-injection).

## 2026-07-20 — T10 CLOSED (both arms) + T7/T8 launched (the last measure-side arms)

**T10-b (anchored measure-only) — REFUTED.** Finished 07-17, uncollected until now. 500k,
3 seeds: **0.068 / 0.060 / 0.104**. Pre-registered bands (spec 04 §3): ≥0.5 headline,
0.2–0.5 partial, **<0.2 arm dead** — this is the dead band, and it is the same 0.076–0.088
ceiling the *uncentered* measure objective already reached. The 0.18–0.26 spike at 50k was
transient: all three seeds decay monotonically after 100k. Reading: centering + scale
anchors + split bootstrap buys a better-behaved optimization *path* but converges to the
same bad fixed point. Consistent with the anneal iql-fade result (the measure's own fixed
point is bad on cube).

**T10-a (aux) — misses its bar too.** 500k finals **0.812 / 0.824 / 0.832 → ~0.823**, vs a
pre-registered ≥0.85 and the plain bilinear-IQL anchor **0.904**. So the centered-measure
advantage does not merely fail to help, it costs ~8 points. Note 0.823 ≈ RG-FB v0 (0.825)
≈ vf-none control (~0.84): a measure-taught advantage adds nothing over scalar grounding.
This keeps the sign of the standing E2 finding (the measure never HELPS) — centering
changed the magnitude, not the sign.

⇒ Per spec 04 §3, with both T10 arms down the strong claim now hinges on **T8**.

**Launched T7 + T8** (tmux `t78`, commit 04da9d6, 3 seeds × 1M, ortho=1000/lr_b=1e-4 to
match the anchors; script `scripts/rescuing_fb/launch_t7_t8.sh`):
- `fb_state_t7_min` — `fb_pessimism_penalty=0.5`. **Verified numerically that with
  num_parallel=2 this is EXACTLY min-aggregation** (mean − 0.5·|p0−p1| ≡ min), so the
  port's audited mean-target deviation is now a dose-response (0 = current mean,
  0.5 = min, 1.0 = beyond-min) rather than one guessed point. Anchor: plain FB 0.42 @1M.
- `fb_state_t7_extra` — `fb_pessimism_penalty=1.0` (beyond-min, VC-FB-style).
- `fb_state_t8_zcond` — `onestep=true onestep_z_cond=true`. **The z-zeroed one-step arm was
  already run and is already refuted at 0.058** (10 seeds, matched ortho1000/lr_b1e-4,
  bridge_grafts.csv:107), so only the zcond arm is new; 0.058 is its matched control.

**On T8's design.** `onestep` conflated two independent choices: the SARSA dataset-action
bootstrap (uncontested, a γ-contraction) and zeroing z into F. The deep audit (spec 01)
reads Zheng et al. 2602.11399 as keeping F z-conditioned; `agent.py`'s own comment defends
z-zeroing on the grounds that the behavior policy's successor features are genuinely
task-independent. Both readings are defensible and this repo has a track record of
subagent claims later retracted, so the disputed bit was made an **ablation** (new
default-off `onestep_z_cond`) instead of picking a side in code.

**Pre-registered reads.** t7_min ≫ 0.42 ⇒ the mean-aggregation optimism was a real cause and
the FB-failure claim must be restated against a conservative FB; t7_min ≈ 0.42 ⇒ optimism is
not the bug, closing the VC-FB confound. t8_zcond ≫ 0.058 ⇒ z-conditioning was the missing
piece and one-step FB is a live baseline; t8_zcond ≈ 0.058 ⇒ one-step FB fails under *both*
readings, and with T10 dead the "no known form of measure-TD grounding supports cube
manipulation" claim is licensed.

**Env note:** `/dev/shm/factored-fb/datasets` had been wiped (volatile tmpfs). Restored from
`~/bridge_ckpt_backup/datasets/cube-single-play-v0` (176M, 1000 episodes). This had also been
silently failing 12 launcher tests.

## 2026-07-20 — T7 + T8 COLLECTED: both DEAD ⇒ the strong negative claim is LICENSED
- All 9 runs reached 1M crash-free (logs `/dev/shm/fb_state_t{7_min,7_extra,8_zcond}_2026*/seed_*.log`;
  groups added to `results_map.py`, CSVs refreshed, audit clean).
- **T7-min (pessimism 0.5 ≡ exact min-aggregation): 0.24/0.28/0.20 @1M → 0.240** (0.193 @500k)
  vs plain-FB anchor 0.42 @1M. **T7-extra (1.0): 0.16/0.18/0.34 → 0.227** (0.147 @500k).
  Pre-registered read: t7 ≈/< anchor ⇒ **optimism is NOT the bug — conservatism HURTS plain FB**
  (dose-response flat-to-worse). The VC-FB confound is closed; no §3 cell needs a conservatism arm.
- **T8-zcond (one-step FB, F z-conditioned): 0.02/0.06/0.12 @1M → 0.067** (0.053 @500k) ≈ the
  z-zeroed matched control 0.058 (10 seeds). One-step FB fails under BOTH readings of Zheng et al.
  Note 0.067 ≈ the BC-flow μ-floor (~0.06): evaluating μ's successor measure and greedifying its
  z-readout adds nothing over BC on this data — consistent with the misalignment mechanism.
- **⇒ Per spec 04 §3 (pre-registered, joint over T10-a/T10-b/T8): the claim "no tested form of
  measure-TD grounding supports cube manipulation" is licensed.** Direction-II cells (first-passage,
  expectile-measure, hindsight test points) are now optional mechanism work, not claim-blockers.
- Synthesis of the whole record written as `phenomenon.tex` (this folder) + companion
  `PAPER/ogbench-sota/phenomenon.tex`; implementation specs `specs/06-phenomenon-implementations.md`.

## 2026-07-20 — spec 06 IMPLEMENTED (code + tests), pending launch
All 8 spec-06 items are implemented as ready-to-launch code + tests (no GPU runs yet). The
knobs are default-off ⇒ v0/anchors bit-identical, param-tree-preserving, assert-lattice, and
the measure/RG-FB test suites pass (`tests/test_gciql_measure_axes.py`, `tests/test_gciql_rgfb_v1.py`,
plus the 20-test measure regression set: no regression). Every new config builds + runs
`update()` cleanly on real cube dims (obs_dim=28, cube xyz at obs[19:22], verified vs
`cube_env.compute_observation`: xyz_scaler=10 ⇒ ε=4cm=0.4 obs units).
- **Direction I (method chain).** item1 GPI eval `scripts/rescuing_fb/t6_gpi_eval.py` (CPU);
  item2 `configs/gciql/cube_single_state_rgfb_d512.yaml`; item3 rerank
  `scripts/rescuing_fb/rgfb_rerank_eval.py` (CPU); item4 v1 task-prior knobs
  (`task_prior`/`task_prior_probs`/`task_prior_mix_k`) + configs `..._rgfb_v1_taskprior_d{50,512}.yaml`.
- **Direction II (mechanism cells).** item5 `measure_first_passage`; item6 `measure_expectile`
  (+`measure_v_phi` tower, `measure_expectile_tau`); item7 `measure_testpoint_dist='hindsight'`.
  Configs `..._measure_{firstpassage,expectile_fp,expectile,hindsighttp}_d50.yaml`.
- **item8** post-hoc measure readout head `scripts/rescuing_fb/measure_readout_head.py`
  (jax RG-FB head vs torch plain-FB native F·B; asserts checkpoint hash unchanged).
- Groups registered in `scripts/results_map.py`; launcher `scripts/rescuing_fb/launch_spec06.sh`
  (DRY_RUN=1 default). **Pre-registered reads applied verbatim per campaign when results land**
  (spec 06): item2 zero-shot ≥0.88 explains the d-gap / ≤0.825 promote item3; item3 ≥0.90 =
  "zero-shot 0.90 via selection"; item4 Dirac ≥0.80 non-reg, two-goal mixture ≥0.60/0.30–0.60/≤0.30;
  item5 ≥0.30 toxic-axis / ≈0.10 axis-C exonerated; item6 ≫0.10 = missing bootstrap / floor = axis-B
  exonerated; item7 ≫0.10 validates weighting half / null = weighting alone not the toxin. Method-chain
  stop: zero-shot ≥0.90 via any of 2/3/4 ⇒ freeze and write.

## 2026-07-20 — spec 06 implementation REVIEWED: 5 findings, all fixed; arms now launchable
- Review pass over the other session's spec-06 implementation (all 8 items). Verdict: eval
  scripts (1/3/8), d512 config (2), expectile-measure (6) clean; v1 (4) had a FATAL bug;
  5/7 had object-level mis-specifications. All fixed same day; 33/33 tests pass across the
  seven gciql-adjacent suites (rgfb, rgfb_v1, measure_axes, measure_scale, t10, task_norm,
  measure_zmix).
- **F1 CRITICAL (item 4)**: sampled z was fully stop-gradded ⇒ ψ never trained under
  `task_prior='mix'` — the agent would have been the frozen-random-ψ row (0.061). Root
  cause was a wrong sentence in spec 06 itself; spec corrected + code fixed (ψ-grads flow
  through z for Dirac/mixture arms; sg only on z_actor, p3 reward features, mixture renorm
  scale, TD targets). Test inverted to ASSERT ψ receives gradient.
- **F2 (item 4)**: mixture reward {0,1} clashed with Dirac gc_negative {−1,0} at
  near-identical z → r_mix = hit−1. **F4 (item 4)**: Dirac-arm actor consumed value_goals
  (30% random) → z_actor rebuilds Dirac rows from ψ(actor_goals), preserving the D8 split.
- **F3 (item 5)**: first-passage kept FB's linear diagonal attraction (unbounded pull; fixed
  point ≠ E[γ^T]) → diagonal now a squared residual when `hit` is supplied.
- **F5 (item 7)**: hindsight diagonal positives included 30% random goals (samples from no
  policy's future) → diagonal now from ψ(actor_goals); unmasked expectile+hindsight refused
  by the assert lattice.
- Ready to launch per the spec ordering: CPU scripts (items 1/3/8) anytime; `launch_spec06.sh`
  arms d512v0 → v1d50 → measure. Pre-registered reads unchanged.

## 2026-07-20 — PROGRAM RE-CENTERED (user decision): success = a SUCCESSOR MEASURE that works on cube
- GPI over goal candidates ran and HELPS — but it is out of scope for this direction. RG-FB,
  GPI, selection extraction = control anchors/baselines only; no further effort. The only
  acceptable success: a measure-native objective (matrix M(s,a,s⁺) over a test-point field,
  actor consumes only its readouts, zero-shot z entry) in the GCIQL band on cube.
- Spec 06 updated with the binding "measure-native" definition + the measure ladder.
- **In flight:** rung 1 (4 single/double-axis measure cells, tmux `s06gpu`, launched 22:18
  WITH today's 5 review fixes) + rung 2 **FULL-CORNER cell** (fp+expectile+hindsight, all 3
  axes at once — new config `cube_single_state_measure_gciqlcorner_d50.yaml`, smoke-tested,
  tmux `s06corner`, GPUs 4/7/0, 3 seeds × 500k, group GCIQL-State-Flow-Measure-GCIQLCorner-d50).
- Early rung-1 reads @50–200k: all in 0.00–0.10 (too early; anchors cross 0.5 only ~150–250k).
- Queued if rungs 1–2 land ≤0.5: rung 3 short-γ two-timescale measure (γ_short≈0.8 head +
  blend — strict occupancy semantics, targets the K≤5 gate window directly), rung 4
  mode-indexed φ (obs[18]), rung 5 torch FB-FP port for the claim-grade agent.
- Bars: ≥0.85 headline / 0.5–0.85 real / 0.2–0.5 partial / ≤0.2 dead → next rung.

## 2026-07-21 — CELL W: the existence theorem + the guaranteed-win successor measure (LAUNCHED)
- **Theorem (the win the direction needed):** with 0/−1+mask, `E[γ^T_hit] = 1+(1−γ)Q`;
  substituting into the first-passage measure residual gives IDENTICALLY
  `δ_M = (1−γ)·δ_Q^GCIQL`. So the first-passage successor measure with hindsight-PAIRED
  columns + in-sample expectile backup IS bilinear GC-IQL (0.904) up to affine
  reparameterization — same fixed point, same gradients up to scale. **A successor measure
  that works on cube exists by construction.** The negative record only ever concerned the
  revisit-counting occupancy estimator at ρ-weighting. Identity verified numerically in
  `test_cellw_knobs_and_affine_identity` (6/6 measure-axes tests pass; defaults guard the
  running corner arm bit-identically).
- **Why the corner cell may still miss:** its uniform B×B matrix makes the per-row task
  weighting ~99.6% UNPAIRED-random columns (axis A re-entering through the estimator);
  GC-IQL's mix is 30% random. New knobs `measure_offdiag_weight` (1.0=corner / 0.3≈GC-IQL
  mix / 0.0=exact equivalence) + `measure_expectile_source='target_min'` (anchor's
  double-DQN split).
- **LAUNCHED:** `cellw0` (w=0, 3 seeds, tmux `s06cellw`, GPUs 4/7/0 packed — the existence
  proof run). `cellw03` (w=0.3 — the measure-native candidate with real off-diagonal matrix
  supervision) queued for free GPUs. Groups GCIQL-State-Flow-Measure-CellW{0,03}-d50.
- **Renewal decomposition (occupancy completion, spec'd):** M_occ ≈ M_fp·R(B) with R
  (s,a)-independent recurrence mass — fit R as a readout after a Cell W win and the agent
  carries the FULL occupancy measure as a product; the identity is also the one-line
  explanation of the whole failure record (joint estimation spends capacity on R).
- Note: other session is concurrently implementing ogbench-sota arch cells (latefusion,
  bounded-ψ) in gciql_flow.py — edits coexist; measure-axes + rgfb-v1 suites green post-merge.

## 2026-07-21 — spec-06 campaign COLLECTED: axis C is REAL (FP 0.225); Cell W confounded by
## the ortho scale (0.071 ≈ the 0.072 ortho-1000 fingerprint); scale-fix reruns LAUNCHED
- All 500k arms done; CSVs refreshed (audit clean). Verdicts per the pre-registered bands:
  - **FirstPassage (axis C alone): 0.188/0.272/0.216 → 0.225 ± 0.035** — band 0.10–0.30
    "partial", upper half. 2.5–3× the measure ceiling and SUSTAINED at 500k (NOT the T10-b2
    transient — no decay 200k→500k). First measure-native cell to hold above the ceiling.
  - **HindsightTP (axis A alone): 0.095** → null at ceiling (weighting alone is not the toxin
    — the §3.3 read applied verbatim).
  - **Expectile unmasked (axis B alone): 0.103** → at ceiling, but DO NOT print "axis B
    exonerated": the exm cells carry the ortho-scale confound below.
  - **Expectile-FP 0.039 / Corner 0.035 / CellW0 0.071** — all POISONED by one diagnosed
    confound: in M-units the TD loss carries (1−γ)²=1e-4 scale (δ_M=(1−γ)δ_Q), while
    measure_ortho_coef=1.0 stayed O(1) ⇒ effective ortho λ ~1e3–1e4 ≡ the known ortho-1000
    catastrophe (0.072 on IQL). CellW0's 0.071 is a fingerprint match. The equivalence
    theorem is NOT falsified — the cell violated the program's own scale-identifiability
    rule (ortho not rescaled across the change of units). Own-goal, logged as such.
  - **RG-FB d512 (anchor track): zero-shot 0.828 ± 0.023**, exact 0.840, gap 1.2 (clean) —
    ≈ d50's 0.825 ⇒ pre-registered read: d is NOT the v0 gap. Rerank on RG-FB:
    BELOW_ANCHOR (oneshot N=1 floor 0.867 beats all rerank cells; declines with N ⇒
    off-support peak at inferred z). GPI: d50 gate 0.824 (below 0.9, PARTIAL, mixtures
    0.260→0.536); G4 d512 gate 0.928 PASS, mixtures 0.260→0.752 (GPI_FIXES_COVERAGE).
    Direction parked per the re-center; numbers recorded as anchors.
- **LAUNCHED (tmux `s06fix`, GPUs 2/3/4/7/0 packed, 3 seeds × 2 arms):**
  `CellW0-NoOrtho` (ortho=0 — the TRUE equivalence run; fp reward pins the scale so ortho=0
  is identified) and `FirstPassage-NoOrtho` (FP hit 0.225 DESPITE a ~100× oversized ortho —
  the clean axis-C read, upside open). Configs `*_noortho_d50.yaml`.
- Next after these land: expectile-fp + cellw03 at ortho=0; then the ortho dial (1e-4..1e-2,
  scale-matched) if the paper wants the ortho-scale mechanism figure.

## 2026-07-23 — wave-2 COLLECTED (occupancy dial dead DECONFOUNDED) + wave-3 relaunch + phase-2 occupancy LAUNCHED
- **Wave-2 verdicts (all no-ortho + scale-norm fix):** Corner-NoOrtho 0.000/0.012/0.004,
  Expectile-FP-NoOrtho 0.000/0.000, CellW030-NoOrtho 0.000 — the ortho-confound excuse is
  DEAD for the occupancy/expectile dial. Expectile+hindsight ON TOP of the 0.967 FP recipe
  kills it outright ⇒ first-passage semantics (plain, tau-less) is the necessary ingredient.
  Four toxins now separately demonstrated: ortho (0.967→0.225), occupancy/revisit semantics,
  z-in-critic (criticmix collapses on all 7 seeds after tracking the anchor mid-training),
  z-actor (TPActor 0.395, attribution control pending).
- **Inference-only zero-shot on the frozen 0.967 ckpts:** plain-z dirac 0.416/0.408/0.568
  (bimodal); mixtures 0.94–1.00 on ALL seeds; renewal z-weighting ≈ no dirac change;
  **RETRIEVAL goal-decode s0: 0.680 default → 0.888 @k=1 → 0.904 @k=16** — near-anchor
  zero-shot with NO z-actor. k-sweep pending on s1/s2 (defaults 0.680 / 0.456).
- **4 wave-2 jobs segfaulted during the first CPU eval** (12-job contention, 20 s/it):
  zctl_s0/s1 (the single-difference z-actor control), w01no_s0, w03no_s0. RELAUNCHED on the
  idle box (tmux `s07w3`, `launch_spec07_wave3.sh`, MAX=6). criticmix s3/s4 also died at
  step 1 — not relaunched (5 surviving seeds already decide the collapse).
- **Phase-2 occupancy program launched (tmux `s07p2`, `launch_phase2_occupancy.sh`):**
  - New knobs in gciql_flow.py, both default-off bit-identical (tests
    `test_gciql_phase2_occupancy.py` 3/3 + measure_axes/measure_scale/warmstart 13/13):
    `measure_stopgrad_psi` (freeze the backward map: sg every online psi column in
    measure_loss; phi-only training) and `actor_q_center` (q-loss normalized by
    sg[mean|q−mean(q)|] — affine-offset invariant; the wave-2 forensics dilution fix).
  - **B centered ×3:** CellW0-NoOrtho + scale_norm + actor_q_center — the theorem cell
    (δ_M=(1−γ)δ_Q ≡ bilinear GC-IQL) with the last diagnosed defect removed. Recovers ~0.90
    ⇒ existence cell realized; still floors ⇒ the identity anomaly needs a new suspect.
  - **C1 frozen-ψ occupancy ×3:** warm-start ALL modules from the matching FP-NoOrtho 500k
    ckpt, freeze ψ, train φ by the plain FB occupancy TD. Works ⇒ occupancy TD fails by
    CORRUPTING the representation; floors ⇒ the occupancy target itself is unlearnable on
    play data. Either way a mechanism claim.
  - **C2 train-ψ control ×2:** same warm-start, ψ trainable (the basin-survival read).
  - Pre-registered bands: ≥0.85 works / 0.5–0.85 partial / ≤0.2 dead (vs FP anchor 0.967,
    GC-IQL 0.904, occupancy cold-start floor ~0.10).
- Still open: the 0.967 independent-eval verification gate; retrieval k-sweep s1/s2;
  results_map/CSV refresh for the wave-2/3/phase-2 groups; spec07 code uncommitted.

## 2026-07-23 — BALL-VISITATION MEASURE COMPLETE: the occupancy successor measure that works
- **Density-TD occupancy on frozen ψ: 3-seed divergence confirmed** (q_abs 154→5e4; success
  0.98→0.42@100k on s0/s1; s2 reproduces, 0.652@50k). The linear Dirac-diagonal is the
  divergence channel — independent of representation quality. s0/s1 killed at ~100k
  (evidence in train.csv), slots given to the repair.
- **THE REPAIR WORKS: ball-visitation TD (`measure_fp_absorbing=false`) on frozen FP ψ,
  warm-started, actor_q_center on — BOTH SEEDS COMPLETE 500k: finals 0.952 / 0.928, full
  curves 0.89–0.99, no instability anywhere, critic scale plateaus ≈24 ≈ the MC renewal
  target 24.7 (correct measure, not a surrogate).** Group
  `GCIQL-State-Flow-Measure-BallVisit-FrozenPsi-d50` (tags vis_s0/vis_s1).
- **Verification gate CLOSED:** independent fresh-seed re-evals of FP-NoOrtho ckpts:
  s0=0.972 (100 eps), s1=0.984 (50 eps), s2 pending; plus 6 incidental step-1 warm-start
  evals 0.96–0.99. The 0.967 was no artifact.
- **Retrieval k-sweep CLOSED: k=1 is the robust decode** — dirac means 0.888/0.960/0.968
  across seeds (k=16 unstable 0.54–0.90).
- **Centered CellW0 (actor_q_center on the w=0 identity cell): 0.000 flat through 150k
  (s1) — the centering fix is NOT rescuing the expectile-form identity cell.** Not final
  (anchor crosses 0.5 at 150–250k) but trending dead; the divergence/dilution story of the
  exm branch likely has a further unmodeled defect. Ball-visitation supersedes it as the
  occupancy vehicle.
- **Research note (external audience):** `two-phase-successor-measure-note.tex` — standard
  terminology, both figures (pipeline block diagram + ball intuition), new §2 on what the
  GC baseline gets right (bilinear = the same rank-d factorization; affine identity
  E[γ^T]=1+(1−γ)V), final numbers folded in. Compiles clean (tectonic, 0 overfull).
- Pending: verify s2; occtr control; zctl/w01/w03 (wave-3, ~09:20+); w0cent to 250k;
  zero-shot eval ON the phase-2 (visitation) ckpts; results_map/CSV refresh; commit.

## 2026-07-23 — external review absorbed; note REFRAMED; z-actor attribution FLIPPED; (z,x) arm launched
- **Review's central critique accepted:** we learn the DIAGONAL SLICE g↦M^{π_g}(s,a,B(g)),
  not the (z,x)-separated M^{π_z}(s,a,dx) — summing slices mixes continuation policies, and
  ball-mass integration ≠ ∫r dM (kernel overlap + ρ(g) dependence). This is WHY dense/region/
  linear fail structurally. Note retitled "A Two-Phase Curriculum for Goal-Conditioned
  Visitation Values...", scope paragraph added, new §object + §dense, off-by-one fixed
  (fixed point is E[γ^{T−1}], affine identity exact under that convention), factorization
  demoted to empirical hypothesis (cube-position-only region ⇒ arrival-mode dependence),
  capacity claim narrowed to the reaching slice, "exact in the limit" softened, reward-free
  framing qualified (cube coords + 4cm radius = task structure), estimator-diagnosis
  isolation caveat added. Compiles clean.
- **zctl COLLECTED — ATTRIBUTION FLIPPED: the z-actor is FINE.** FP-NoOrtho + D1 z-actor
  (single difference): 0.960/0.996 @500k ≥ goal-conditioned anchor. TPActor's 0.395 was the
  z-PRIOR (mixture/sphere arms), not z-conditioning. Unblocks π(a|s,z_r) for dense rewards.
- **Occupancy-dial FINAL: w01no/w03no = 0.000 flat** (dial fully dead, deconfounded).
  **occtr (train-ψ) control: 0.968@1 → 0.072/0.056 @50k on both seeds** — trainable ψ makes
  density-TD collapse FASTER than frozen ψ. **w0cent: 0.000 through 200k — the actor-centering
  fix does NOT rescue the expectile identity cell** (its anomaly needs a new suspect; parked).
- **(z,x)-SEPARATED ARM LAUNCHED (user directive: crack dense rewards):** relaxed the spec06
  single-axis assert so fb_forward_z + measure_z_random_frac compose with the goal-reaching
  residual (rows = policy index z_i ∈ {ψ(g), random directions} with z-consistent bootstrap,
  columns = query x_j) — the sampled M(s,a,z,x)=F(s,a,z)ᵀB(x). Tests updated (measure_axes
  6/6; lattice test now asserts the new composition rules). Cold-start phase-1 form,
  2 seeds, group `GCIQL-State-Flow-Measure-FP-ZX-d50` (zx_s0/s1); smoke 300 steps clean.
  If stable → phase-2 visitation on top → z_r consumed directly by F and π.
- verify_s1 formal: 0.984→0.988 (two eval rows); s2 running. Zero-shot families on the
  finished vis ckpts running (visz, retrieval k=1).

## 2026-07-23 — (z,x)-SEPARATED PHASE-1 COMPLETE: stable at anchor
- **zx_s0/zx_s1 (FP-ZX-d50) finished 500k: 0.904 / 0.960**, curves 0.74–0.99 (one noise dip,
  diagnostics flat), q scale bounded ~1.0 throughout. The separated form — F(s,a,z) rows with
  z ∈ {ψ(g), random directions} 1:1, z-consistent bootstrap π(·|s',z), z-actor — trains from
  COLD START at anchor level. The criticmix late-collapse did NOT appear: grounding z through
  the matrix residual (vs the scalar mixture TD) is the stable form.
- Verification gate final: 0.972/0.988/0.984 (3 fresh-seed re-evals). Controls final: occtr
  collapsed both seeds; w0cent 0.000 through 350k (killed, verdict banked).
- Direct-z zero-shot eval on the zx ckpts LAUNCHED (mode=z — no goal decode). Next after it:
  visitation phase-2 on the zx form; then non-goal-reducible reward families.
- Note updated (§dense records the zx result; verify triple in footnote). Compiles clean.

## 2026-07-26 — SMB v1 COLLECTED: the capability-ladder eval ran end-to-end on 12 ckpts
- **Suite**: spec11 (`specs/11-smb-eval-suite.md`); frozen artifact `smb_cube_single_v1.json`
  (T3 pairs validated on dataset: R_stable/R_transient = 8.8–9.8×); harness
  `smb_common/smb_tasks/eval_smb/eval_smb_measure` + 36 tests green; outputs
  `analysis/rescuing-fb/smb/smb_{A,Aoracle,B}_{row}.json` (36/36, all exit 0).
  Mid-flight fix: Wing-B envs now `terminate_at_goal=False` (stock termination truncated
  gt occupancy at first arrival — gt_own 1.1 → 16–51 after fix).
- **T3 occhoice (headline)**: anchor chose-correct ≈ chance (0.35/0.61/0.43) with
  G/oracle 0.17–0.45 → **the discriminator works: a reaching value picks the wrong
  region**. But phase-2 diag is only ~chance too (0.44/0.51); ZX rows best (0.62–0.64).
  The learned readouts under-resolve R at s0 vs the 9× dataset gap → modification target,
  not the pre-registered clean ✓. (Oracle-z T3 arm is trivial — hands the correct region;
  ignore its 0.9s.)
- **T5 corridor SURPRISE — signed rewards partially work TODAY on measure rows**:
  corridor binds (timeB_λ0 0.04–0.13); deflection timeB_λ1/λ0: zx 0.41, visz 0.31, vis
  0.50 vs **anchor 2.3 (worse with penalty)**. Ahead of pre-registration; check
  success-at-A retention before claiming.
- **Wing B**: anchor B1 units-broken as predicted (pred −40 vs gt>0; B3 ρ≈0). fp B1
  rel-err ≈ 1.0 (reaching units, predicts none of the mass) as predicted. **visz (ZX
  phase-2) is the best-calibrated object**: B1 err 0.48–0.51 ρ 0.33–0.37, B2 balls
  ρ 0.31 / slope 0.27 (4× under), B3 ρ 0.65–0.76. Boxes ρ 0.07 + union additivity defect
  ~0.96 → **arbitrary sets + additivity fail exactly as pre-registered** (spec-10 L7/L8
  target). T6 linear: weak/unresolved everywhere (as predicted).
- **⚠ UNVERIFIED ANOMALY before any claim**: vis (diag phase-2) B1 predictions are
  NEGATIVE at reset states (−1.6…−6.3 vs gt 16–51; vis_s1 Spearman −0.57) despite
  training-time scale ≈24 ≈ MC. Either the harness readout path (psi source / a0 / min-head)
  or a REAL reset-state OOD miscalibration. Audit = rerun readout at dataset (s,a) states;
  decides bug vs finding.
- Pending: results_map/CSV refresh for smb outputs; T5 success-at-A pull; B1 audit; commit
  harness + spec + playpen (all uncommitted).

## 2026-07-26 (cont.) — B1 audit CLOSED (finding, not bug) + T5 surprise RETRACTED
- **B1 audit (vis_s0, 512 dataset pairs)**: readout path CORRECT. Own-successor ball
  diag mean 73.5 / median 96.5 (dwell scale ✓); random-goal off-diag mean 2.4 / median
  −0.4. ⇒ **REAL FINDING: the diagonal ball-visit critic is occupancy-calibrated only
  near its hindsight support; its far-field (incl. reset states) collapses to ≈0/negative**
  — a proximity-calibrated object, not a measure over the (s,g) product. ZX phase-2's
  better far-field (B1 err ~0.5) fits: random-z training spreads support. Spec-10's
  target, now measured. (First audit pass had a script bug — compared permuted copies of
  the same goal set; corrected before any conclusion.)
- **T5 retraction**: ZX "deflection" is task destruction, not avoidance — success-at-A
  collapses with λ (zx 0.89→0.24→0.06; visz 0.80→0.03): negative reward weights corrupt
  z_r and the policy stops reaching A. Only vis_s0 shows a genuine trade-off (succA
  0.94→0.78 with timeB 0.126→0.063), unreplicated on vis_s1. Also corridor_binds on only
  1/5 tasks per row (thin base — widen the corridor gate in v1.1). T5 verdict returns to
  the pre-registration: **no current method does reach-while-avoid**.
- Matrix artifact: `analysis/rescuing-fb/smb/smb_matrix_v1.csv` (+ smb_t5_retention.csv).
- Commit deliberately left to the user (harness+specs+playpen all uncommitted).

## 2026-07-26 — FULL SUCCESSOR MEASURE (eval-layer, spec 10 L1+L2) BUILT + GATED on SMB
- **Construction (no retraining)**: over the ZX phase-2 ckpts, m̂=M̃/N̂ with N̂ = the
  dataset's own ε-kernel mass (pure KDTree count); set masses = density integrals
  (additive BY CONSTRUCTION); Q̂_r = ∫r dM; z-inference = 1/N̂-corrected moment;
  z-selection = measure-ranked candidates (Soft-FB-style). `full_measure.py` (8 tests) +
  SMB routes `--predictor density`, `--z-inference cmoment`, `--z-select measure`.
  Commits 5f2e74d (SMB+playpen) + 03d64de (full measure).
- **Gate results (visz_s0/s1; ablations isolate each knob; zx = units control):**
  - **T6 linear RESOLVED**: full route beats BOTH controls on BOTH seeds (5.84 vs
    1.52/3.46; 1.32 vs −2.09/−1.41). First time this family clears its controls.
  - **Union additivity defect 0.96 → 0.19/0.35** (mechanism, as constructed).
  - **B3 ∫r dM calibration slope 25× → 0.35–0.41** (r² 0.41–0.43, ρ 0.66); B1 err
    0.48→0.26–0.31. Gate band 0.5–2 near-missed; 60× improvement.
  - **T5 task destruction FIXED by measure-select**: succA under λ retained 0.77–0.83
    (cmoment-only control keeps the old collapse 0.02–0.04) — the selector rejects
    corrupted z. Genuine avoidance still absent (timeB unchanged).
  - **T3 required one iteration**: raw density route chose-correct 0.15–0.22 — the
    1/N̂ importance tail overweights thin-support (elevated) regions. Winsorized q_r
    (`--density-cap-q`): cap95 no effect, **cap80 → chose 0.58/0.65, G/oracle
    0.74/0.93** (beats the no-density baseline 0.69/0.82 on both seeds); cap60 sweep
    point pending. Chose-gate 0.7 not yet cleared; G/oracle is.
  - **zx (phase-1) control**: density layer cannot rescue reaching units (B1 err 0.98,
    B3 slope 13) — phase-2 visitation training is what carries occupancy. As registered.
  - Remaining open: B2 boxes ρ 0.05/0.31 (kernel-average of member balls still crude —
    the trained-region-columns / additive-B arms, spec 10 L7/L8, are the proper fix).
- Files: `analysis/rescuing-fb/smb/smb_fm_*.json`, `smb_fm2_t3_*.json`.
- **cap60 = cap80 exactly** (chose 0.58/0.65, G/or 0.74/0.93): the winsorization has
  saturated — the residual T3 choice error is no longer the importance tail but the
  critic's own R-resolution at thin support (needs trained arms: spec10 L7/L8 region
  columns or the phase-3 renewal head). cap-q default 0.8 is safe (insensitive 0.6–0.8).

## 2026-07-27 — CAMPAIGN V2 COLLECTED: actor-z arm ADOPTED, multi-scale arm inconclusive
- 4×300k warm-started from visz@500k (tmux s10v2, exp_spec10); knobs default-off
  bit-identical (test_gciql_campaign_v2 9/9, regressions 28/28); gciql_flow.py still
  uncommitted (mixed-track dirty state).
- **Arm B `actor_z_greedy_frac=0.3` (azB) — the best full-SM checkpoint to date, ADOPT**:
  native 0.84/**0.916** (s1 improved over parent 0.712); **T3 chose 0.79/0.74 ✓ gate**
  (G/or 0.86/0.86) though T3 wasn't its target; **T6 ✓ both seeds** (3.93 vs 1.96/3.58;
  1.32 vs −2.20/−0.97); boxes 0.35/0.33 (best yet, gate 0.4 missed); B3 ρ 0.60/0.63 slope
  0.37/0.42. T5-deflection clause NOT met (sA retained 0.83–0.90, timeB flat) — avoidance
  still absent everywhere.
- **Arm A `measure_eps_multiscale` (msA) — seed-split, do not adopt yet**: s1 spectacular
  (T3 chose 0.72 ✓, G/or **0.99**, union defect 0.24) but s0 fails T3 (0.54) with native
  cost (0.64), and s1 regresses T6 below controls (−2.93 vs −0.80). ⚠ its box mechanism
  was NEVER exercised: the gate readout doesn't query per-scale (eps_gain lives only in
  the training loss path) — wiring a per-scale readout + matching-ε N̂ into the density
  predictor is the designated next step before any multi-scale verdict.
- SM scoreboard after v2 (azB): goals ✓, mixtures ✓, **occupancy-choice ✓**, linear ✓,
  unions additive ✓, ∫r dM rank ρ~0.6 slope ~0.4, boxes 0.35 (open), avoidance absent
  (open — likely data-support, the play data contains no detours).
- Files: `analysis/rescuing-fb/smb/smb_v2_{A,Bd}_*.json`; groups ...ZX-{MultiEps,ActorZ}-d50.

## 2026-07-27 — PHASE 2B COLLECTED: the full measure is TRAINED end-to-end (no KDTree)
- `measure_nhat_head` knob (softplus MLP on sg[critic ψ(g)], plain regression on the
  hit-matrix column mean, coefficient 1.0, gradient-isolated; 13/13 tests after fixing a
  TEST-SCAFFOLD bug — get_config's measure_coef=0.0 default zeroes the measure loss in
  total_loss even under measure_only; real yamls set 1.0, implementation was never wrong).
- Wave: nh_s0/s1 = azB@300k + 150k with nhat+actor_z_greedy (group ...ActorZ-Nhat-d50);
  native 0.808/0.928; nhat loss 3.3e-5→1.6e-5, N̂ mean ≈0.014 (true ball-mass range).
  ⚠ save_interval=100k ⇒ gate ran at params_100000 (same-ckpt comparison unaffected).
- **GATE: trained denominator ≡ KDTree within noise, and BETTER on calibration** —
  B1 err 0.34/0.26 (vs 0.39/0.34); B2 balls/boxes equal; **B3 slope 0.46/0.53 (vs
  0.35/0.42) — s1 crosses the 0.5 calibration band, first cell ever**; T3 chose 0.70/0.79
  ✓ gate both routes, s1 G/oracle 1.00; T6 ✓✓ both routes. Blemish: s1 union additivity
  defect 0.71 trained vs 0.32 KDTree (head smoothing at union members; single-seed, open).
- **Claim licensed: the full successor measure = pure network weights, trained entirely in
  phase 2 by bounded regressions (M̃ visitation TD + N̂ column-mean regression), dataset-
  free at readout, reproducing and slightly beating the eval-layer construction.**
- Uncommitted (user reconciliation needed): gciql_flow.py (mixed-track), campaign-v2/2b
  test file, eval hooks (density_trained/cmoment_trained), launch_spec10_v2.sh.
