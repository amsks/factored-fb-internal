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
