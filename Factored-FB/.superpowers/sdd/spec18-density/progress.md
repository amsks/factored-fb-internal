# SDD ledger — plan: PAPER/Multistep-FB/specs/18-density-measure.md

BASE: 862c9c6a368623adca7e6eb174d1c93aa4984c2f (spec17 fully committed)
Policy: no agent commits, no staging (shared branch; the OTHER track has 4 dirty files in scripts/rescuing_fb — eval_smb.py, eval_smb_measure.py, smb_common.py, t3_offpeak_multienv.py — NEVER `git add -u` a directory, I already made that mistake once and had to reset)
GPU state: part_s1 (spec17) still training on GPU0 until ~500k; GPUs 1-7 free.

Wave 1 dispatched (parallel, disjoint files):
- A: gciql_flow.py measure_density (§4) + tests/test_gciql_measure_density.py (§5) + 2 configs (§7)
- B: scripts/rescuing_fb/eval_density_zeroshot.py + tests (§6)
- C: OWED from spec17 — results_map.py entries + CSV refresh for the d17_* groups (§7 last para)

⚠ spec17 INCIDENT #2: d17_cs_part_s1 crashed AGAIN at 230k, same
  jax.errors.JaxRuntimeError: INTERNAL: CUDA error: Failed to end stream capture:
  CUDA_ERROR_STREAM_CAPTURE_INVALIDATED, raised from a device_get at logging time.
  First crash was on GPU4 (70k), second on GPU0 (230k) => NOT a bad card; XLA CUDA-graph/command-buffer
  capture is the failure mode. Other 5 cells ran 500k clean on the same code.
  WORKAROUND baked into launch_spec17_cs.sh and to be used for spec18: XLA_FLAGS=--xla_gpu_enable_command_buffer=
  (verified accepted by jax 0.10.0 on this box; disables command buffers, costs a little speed).
  Relaunched 11:38 tmux `s17cs2` GPU0 from scratch (comparable to seeds 0/2). Dead dirs preserved:
  d17_cs_part_s1.crashed_70k_cuda_1043 and .crashed_230k_cuda_1137.
  Its evals before dying were consistent with the other seeds (~0.86 @200k), so the third seed is confirmatory.

Wave 1 dispatched: A=agent code+tests+2 configs, B=eval_density_zeroshot+tests, C=results_map/CSV refresh for d17_*.

Task C (CSV refresh, OWED from spec17): DONE. results_map.py += GCIQL-CubeSingle-Flow-Measure-{Delta,DeltaPartition}-d50
  routed to CSV2 bridge_grafts.csv, matching the existing cube-single measure-variant convention.
  Compiled rows match my independently-measured ground truth EXACTLY: delta mean .2067 (.152/.264/.204),
  partition mean .884 n=2 (.884/.884), s1 correctly excluded.
  FINDING (agent's, worth keeping): the compiler pulls from wandb BY GROUP and accepts a run only if its
  logged step is in [490k,510k] -- that window is the ONLY safeguard against a crashed run being averaged in
  as a completed seed. All 3 dead part_s1 wandb runs (45k/70k/230k) fell outside and landed in partial_seeds,
  so the behaviour was correct HERE, but a crash landing inside the window would be silently accepted.
  Compiler semantics deliberately NOT modified.
  ⚠ scripts/results_map.py NOT COMMITTED: it already carried 218 lines of the OTHER track's uncommitted work
  (mb_*/fr_* ICLR baselines, nf_*/tz_* stack-divergence audit). Committing it would sweep in their in-flight
  edits -- the exact mistake I made once today with `git add -u`. Left dirty for that track to commit.

Task A (agent code): DONE. Controller-verified by reading density_loss :1869-1931 -- raw rows -> _delta_phi
  (kernel), eps-scaled rows -> density_b (tower), NO double-scale (the spec's called-out trap); targets called
  without params (frozen); a_plus stop-gradded; softmax over the QUERY axis. total_loss :2297-2306 explicitly
  NOT gated by measure_coef (the 17b CRITICAL-1 lesson carried forward). update override :2405-2421 soft-updates
  both towers. Agent-reported 56/56 green; my independent full-suite run in flight.
  New helper `_raw_block_cols` (not spec-named) added to avoid triplicating the raw-slice; behaviour matches spec.

Controller prep (launch-ready, NOT launched pending review):
  - 3 K-sweep configs written: cube_single_state_measure_delta_partition_K{64,512,1024}_d50.yaml (17b code,
    only protos path + run_group differ; each carries the pre-registered read in its header).
  - protos generated + chmod 444 + digests logged: K64 / K512 / K1024 (K1024 sha 02a6ff92...).
  - scripts/rescuing_fb/launch_spec18_cs.sh: 7 cells on GPUs 1-7 (GPU0 RESERVED for the spec17 part_s1 re-run),
    XLA_FLAGS=--xla_gpu_enable_command_buffer= on every job. DRY_RUN clean.
Review (opus, launch gate): DISPATCHED.

Task B (eval): DONE. scripts/rescuing_fb/eval_density_zeroshot.py + tests/test_eval_density_zeroshot.py (19 green).
  Agent caught a real pre-test bug: density_weights' head-axis AUTO-DETECTION was ambiguous ([H,d] with H=2 heads
  is indistinguishable by shape from [N,d] with N=2 queries) -> now requires an explicit leading head axis.
  Finding on the spec's "=" vs "~=" linearity split: under a plain closed-form ridge the SECONDARY readout is ALSO
  exactly linear (~1e-14); the difference materializes as CONDITIONING (ill-conditioned B + small ridge => SECONDARY
  error ~1000x PRIMARY). Demonstrated in a test rather than asserted.
  Unverified until a ckpt exists: everything past restore (build_pool/density_query/decode/rollout/main).

FINAL REVIEW (opus): FINDINGS but NO functional defect. Independently verified (not trusted): CE axis is the query
  axis (batch-axis alternative gives a different number => the check is live), t rows on the simplex, bootstrap on
  TARGET towers at (s',a+,g), eps-scaling split correct on all 5 paths, exact-zero grads by its own jax.grad,
  measure_coef not gating (confirmed by gradient), no new pytree field (save/restore round-trips), section 8 clean.
  Off-path bit-identity checked the STRONG way: working tree vs HEAD:gciql_flow.py in a parallel shim dir, 3 off
  configs, comparing loss + every info key + leaf count + post-update param hash + rng state. 6 of 7 injected
  mutants killed by the suite.
  LAUNCH BLOCKER (fix round 1 dispatched): density_ess logs a MEAN over rows x heads but gate G2 pre-registers a
  MEDIAN -- a bimodal or one-head collapse reads healthy. Same class as spec17's msq-computed-and-discarded.
  Also fixing: lattice gap (dueling/task_prior_critic_mix not refused; dueling branch precedes density in
  actor_loss => CE trains while the actor maximizes an untrained critic), 3 tautological tests (deleting the
  stop_gradient around t leaves all 10 green!), stale --k 256 comment in my K-sweep configs.
  RULED no-change (calibration, not defects): k_self ~= 0.057 and kernel ESS ~= 63 at eps=0.4/M=1024 is the HEALTHY
  baseline (batch-size dependent -- do NOT read 0.06 as coverage failure); tau=1 at d=50 gives a sharp init
  (ESS ~24/1024) recovering to ~65 in 20 updates => temperature/rank are the first suspects if G1 misses, per
  spec section 3, NOT something to schedule away (section 8).
  NOT reviewed (own pass owed; gates G3/G4 depend on them): eval_density_zeroshot.py + its tests, launch_spec18_cs.sh.

⚠ scripts/rescuing_fb/measure_region_density.py (04:35) and tests/test_smb_density.py (07:35) are the OTHER
  track's untracked files, NOT mine. Do not commit them. Commit explicit file lists, never directories.

Fix round 1 DELIVERED + controller-verified in source: density_ess_median + density_ess_h0/h1 :1937-1940 (the
  G2 gate stat now exists); lattice refuses value_arch='dueling' + task_prior_critic_mix :2989-2994; 3 tests
  de-tautologized; K-sweep config provenance corrected.
  HONEST CAVEAT accepted (agent's, verified reasoning): deleting stop_gradient(t) from density_loss STILL flips
  no test -- because TrainState.__call__ already severs the target towers from grad_params (neither call passes
  params=grad_params), so t never touches trainable params with or without the wrapper. The sg is redundant
  belt-and-braces, identical to 17b's p_tgt. No test CAN detect its removal; the new test validates the invariant
  via a reconstruction that would go live if the calling convention were ever violated.
Controller-independent suite: 90 passed (density 11, delta 8, partition 12, eval_density 19, partition_zs 22,
  eval_delta 18) in 352s.
COMMIT 74e861d (explicit file list -- NOT `git add <dir>`; results_map.py, measure_region_density.py and
  test_smb_density.py deliberately left to the other track).

WAVE LAUNCHED 2026-08-06 12:54 tmux `s18cs`, OUTROOT=/home/adityamohan/exp_spec18, 7 cells GPUs 1-7, 500k,
from scratch, XLA CUDA-graph capture disabled on every job:
  d18_cs_dens_s{0,1,2}   density PRIMARY
  d18_cs_hard_s0         hard-immediate ablation (k = I)
  d18_cs_partK{64,512,1024}_s0   17b K-sweep control
  protos digests: K64 6c9598a0... K512 7dd3649d... K1024 02a6ff92... (all chmod 444)
spec17 wave now 6/6: part_s1 finished (450k eval .876; s0/s2 .884) => partition = .884/~.88/.884 on 3 seeds.
STILL OWED: spec17 25-episode battery on all 6 ckpts; G4 Spearman for part K256; BC-only control; own review
  pass on eval_density_zeroshot.py + launch_spec18_cs.sh (gates G3/G4 depend on them).

FIRST READS @450k-500k (spec18; NOT all final):
  GOAL SUCCESS (training loop, 5 tasks x 50 eps):
    d18_cs_dens_s0 .944 (450k) | s1 .976 (400k) | s2 .940 (450k)     <- DENSITY
    d18_cs_hard_s0 .984 (450k)                                        <- HARD ABLATION = BEST
    d18_cs_partK64 .556 | K512 .872 | K1024 .868 (all 500k); K256 = .884 (spec17)
  Anchors: part K256 .884 | fp-noortho .967 (previous project best) | delta .15-.26 | from-scratch band .004-.5
  => G1 PASSES decisively (target was ~= part's .884; density is .94-.98, AT OR ABOVE the .967 anchor).
  => G2 PASSES hugely: ess_median 87.7/93.2/89.9 vs gate >= 4; per-head h0/h1 within ~1.5 of each other (no
     one-head collapse -- exactly what the mean-vs-median fix was added to be able to see). k_self .056-.059 =
     the reviewer's predicted HEALTHY baseline (~.057), NOT coverage failure.
  ⚠ PRE-REGISTERED EXPECTATION REVERSED: spec section 7 says "density ~= part, hard floors => the eps-kernel
     smoothing of the immediate term is load-bearing". The OPPOSITE happened -- hard (k = I, NO smoothing) is
     the best cell in the wave (.984), with ess 7.0 / median 5.4 (still >= 4) vs density's ~97. So the smoothing
     is NOT load-bearing; the sharper one-hot immediate target is at least as good and probably better.
  K-SWEEP is NOT flat: K64 .556 << K256 .884 ~ K512 .872 ~ K1024 .868. So prototype count IS load-bearing below
     ~256 and saturates above it. The "discretization is arbitrary" answer is therefore "arbitrary above a
     threshold", not "irrelevant" -- and density/hard beat the saturated plateau anyway.
  NOT YET RUN: G3 (pool sweep) and G4 (Spearman) -- both need eval_density_zeroshot.py on the finished ckpts.

FINAL 500k (all 7 cells) + eval-noise caveat:
  goal success: dens s0 .896 s1 .928 s2 .964 | hard .876 | partK64 .556 K512 .872 K1024 .868 | (K256 = .884)
  ⚠ MY EARLIER MID-RUN READ WAS WRONG: at 450k hard showed .984 and I reported it as best + "smoothing is not
    load-bearing". At 500k hard is .876. Last-4-eval means: hard .956 (1 seed), density .924 (3 seeds, .903-.947),
    part .874 (K>=256). Adjacent evals swing .05-.12 on 250 episodes => SINGLE-EVAL COMPARISONS ARE UNRELIABLE.
    What survives: density > partition, 3 seeds, both K>=256 values. What does NOT: hard > density (1 seed, noise).
  G2 final: ess_median 87.3/89.5/91.3 (density) and 5.14 (hard) -- all >= 4 => PASS. k_self .057-.059 (healthy);
    hard k_self = 1.000 by construction.
  K-sweep NOT flat: K64 .556 << K256 .884 ~ K512 .872 ~ K1024 .868 => prototype count IS load-bearing below ~256,
    saturating above. "Arbitrary above a threshold", not "irrelevant".

ZERO-SHOT 25-episode (results/spec18_eval/, results/spec17_eval_25ep/):
  density  s0 dirac .67 dense .64 region .27 | s1 .75/.67/.28 | s2 .89/.86/.26
  hard     s0 dirac .86 dense .86 region .47
  partition(17b) s0 .77/.86/.18 | s1 .67/.69/.04 | s2 .74/.62/.18
  delta(17a)     s0 dirac .42 region .11 | s1 .39/.21 | s2 .34/.05
  => spec18 section 7's DIRECTIONAL read CONFIRMED: region-family zero-shot improves over part's .17-.20 --
     density .26-.28, hard .47. Removing the fixed support pays off exactly where a finite prototype set has to
     interpolate. On point-goal families (dirac/dense) density ~= partition, NOT better.

GATE G3 (per-task success drift 4096->16384 <= .05): FAILS as written. dens_s2 dirac t2 .880 -> 1.000 (.120),
  t5 .800 -> .840, t1/t3/t4 unchanged.
  ATTRIBUTION IS CLEAN: I re-ran the IDENTICAL command and got bit-identical numbers on every task at both pool
  sizes => the eval is DETERMINISTIC, rollout noise is exactly ZERO, so the .120 is 100% readout. And both moves
  are IMPROVEMENTS with more samples => under-converged Monte-Carlo integration at P=4096, NOT instability.
  Q-drift agrees: dirac t1 Q = 9.64 (P=256) -> 18.25 -> 18.70 -> 19.88 (16384), drift_vs_maxP 10.24/1.63/1.18/0.
  Follow-up running: P in {16384, 32768, 65536} to find where it IS converged (relabel-size 120000 for the cap).
  Do NOT reinterpret the gate to make it pass -- report "fails at 4096, converges at larger P" if that is what holds.

GATE G3 RESOLVED: FAILS at the script default P=4096 (max per-task drift .120 vs P=16384), PASSES at P>=16384
  (drift .000 on ALL 5 dirac tasks, 16384 -> 32768). => the readout converges by 16384; 4096 is under-sampled.
  This CONFIRMS spec 18's core claim (query count = MC sample size, more samples = less integration error) and
  identifies the correct eval operating point. Fix is to move the eval default P, NOT to touch the method.
  ⚠ pool CONTENT also matters slightly at fixed size: P=16384 with --relabel-size 60000 gives t1 .96 where
    --relabel-size 50000 gave 1.00 (mean .896 vs .888). Pin the pool draw seed + relabel-size when quoting.
  ⚠ eval OOM-KILLED at P=65536 (728G box): the SECONDARY ridge readout scales badly in pool size. Practical
    ceiling ~32768 as written. Not a blocker (16384 suffices) but a real limit to record.
  ⚠ the eval is fully DETERMINISTIC (bit-identical repeats) -- good for reproducibility, and it is what let me
    attribute the 4096 drift 100% to the readout with zero rollout contamination.
STILL OWED: G4 (Spearman ordering) for both part K256 and density; BC-only control; my own review pass on
  eval_density_zeroshot.py + launch_spec18_cs.sh; results_map/CSV entries for the d18_* groups.
