# SDD ledger — plan: PAPER/Multistep-FB/specs/19-ors-program.md (+19a, +19b, 20)

BASE: a8f1df8e61718b6b626025c877b3d44f9e78d4ee
Policy: no agent commits; commit EXPLICIT FILE LISTS only (other track has dirty files in scripts/rescuing_fb/ + results_map.py + train.py; I already swept theirs in once by accident).
GPUs: all 8 free (spec17+spec18 waves complete).
User ask: implement + start runs on cube-single AND cube-double.
SCOPING: per spec 19 section 1 the only runs launchable now are P1.1 (19a behavior-density). P1.3 extractors are
  HARD-GATED behind P1.2 calibration (Spearman >= .65) which needs P1.1 ckpts. Do NOT launch extractors early.

Wave 1 (parallel, disjoint):
- A CRITICAL PATH: 19a = gciql_flow.py measure_density_behavior + datasets.py next_actions + 3 configs + tests
- B P0.1 G4 Spearman calibration read (d18 density + d17 part K256) + P0.3 CSV/results_map for d18_* groups
- C 19b section 0.1 ORS paper transcription note + section 1 calibration probe script (NO gciql_flow.py edits)
Wave 2 (serialized on gciql_flow.py, after A): launch cs+cd; then 19b extractor + P0.2 BC-only control knob.

Controller prep: scripts/rescuing_fb/launch_spec19_p1.sh written (P1.1 behavior density).
  ENVS default "cs cd" = the user's ask; SEEDS default 0 per spec 19 section 1 (3 seeds on cs only AFTER the
  P1.3 gate). Fails LOUD if a config is missing (verified: refuses right now, configs land with 19a).
  XLA CUDA-graph capture disabled on every job. Header states explicitly that the actor in this mode is pure BC
  and its evaluation/overall_success is NOT a method result -- the run's product is the frozen (F_D,B_D) towers.

P0 DEBTS CLOSED (agent B):
  P0.3 CSV: 5 spec-18 run_groups added to results_map.py (append-only, other track's entries untouched);
    all 5 compiled numbers match my ground truth EXACTLY (density .896/.928/.964, hard .876, K64/512/1024
    .556/.872/.868).
  P0.1 G4: new script scripts/rescuing_fb/eval_measure_g4_spearman.py (translates the valid-measure campaign's
    torch protocol, PHASE2-LEDGER section 9.15 / torchfb_full_measure.py part B, to the jax stack; reuses
    env_geometry + the shipped density/partition readouts + measure_readout_head.spearman live).
    RESULT: **G4 FAILS FOR BOTH METHODS, ALL 6 CKPTS** -- density .160/.166/.174, partition K256 .132/.144/.177,
    gate >= .65. Threshold NOT adjusted. Bit-identical on repeat.
    Author's own caveats: pooled-vs-per-multiplier is ambiguous in the spec text (both fail); degenerate_frac
    ~= .60 from cube-single's "random" probe policy; the new script has no unit tests.
  ⚠ CONSEQUENCE IF REAL (spec 19 section 3): "P1.2 gate fails => the behavior-density occupancy is miscalibrated
    in ordering => fix the measure BEFORE touching shaping". G4 is the g-conditioned twin of that diagnostic, so
    a real null here is a strong prior that P1.2 will also fail -- i.e. the ORS shaping signal would be built on
    an occupancy model whose ordering is near-chance.
    NOT implausible on the record: task success has ALREADY been observed to be BLIND to measure validity
    (spec-10 L1 work, ledger section 9.15) -- a .93-success policy with near-chance region ordering is exactly
    that dissociation again.
  VERIFICATION DISPATCHED (opus, adversarial): policy-match between MC rollout and the measure's policy is the
    likeliest way the number is wrong; also demanded a DECISIVE control (feed the probe a known-correct readout,
    must score ~1.0; feed noise, must score ~0), the non-degenerate-subset Spearman, and per-multiplier values.
  P1.1 launch is NOT blocked by this: P1.1 produces the artifact P1.2 needs, and P1.2 is the gate that would
    catch the defect before any extractor GPU time is spent.

19b section 0.1 TRANSCRIPTION DONE (paper fetched via pymupdf + released code via curl, NOT summarized):
  (a) r_W REPLACES batch['rewards'] for non-terminal transitions, BUT the success-index transition is
      force-set to 1.0 regardless of r_W (Appendix B.3 TD-target eq + released datasets.py/gciql.py).
  (b) NOT unscaled: Appendix B.7 Table 10 gives PER-TASK divisors (cd 0.25 => 4x, scene 0.75 => 1.33x).
      ⚠ the public code has a single hardcoded 0.75x with no per-task hook -- a real paper/code GAP, flagged
      not resolved. Our config knob ors_reward_scale must therefore be set PER ENV, not left at 1.0.
  (c) same gamma symbol throughout; numerically matches our envs (0.99/0.99).
  (d) ⚠ Theorem 1 (optimality preservation) needs A.1-A.4: determinism, one-step reachability, strict potential
      monotonicity, layer-monotone behavior policy. NONE empirically audited by the paper, and A.1 is NOT
      literally true for cube contact dynamics. => we may not claim their optimality guarantee transfers to our
      setting; say so wherever ORS results are written up.
  Files: PAPER/Multistep-FB/specs/19b-ors-transcription.md; scripts/rescuing_fb/ors_calibration_probe.py (P1.2
  gate probe, reuses _delta_phi/_density_queries/towers + env_geometry.HitSpec + the shipped spearman);
  tests/test_ors_calibration_probe.py 34 green.

⚠ SELF-INFLICTED FRAGILITY FOUND: tests/test_gciql_measure_density.py::test_default_off_bit_identical now FAILS.
  It compares the working tree against `git show HEAD:gciql_flow.py` -- a design I asked for in the spec-18
  round-2 review to make the off-path test non-tautological. It breaks whenever the working tree legitimately
  carries uncommitted gciql_flow.py work (exactly now, with 19a in flight). Needs to be made robust (pin a
  baseline copy, or skip when the tree is dirty) rather than left as a false alarm that trains people to ignore
  a red suite.

19a COMPLETE + LAUNCHED. Controller-verified END-TO-END (not just unit tests): a 300-step run through the REAL
  entrypoint logs density_sarsa_subst = .00195 = 2 substituted rows / 1024 batch ~= 1/traj_len, exactly as spec
  19a section 1 predicts; k_self .058 = the spec-18 healthy baseline. Without the forcing the key would be ABSENT.
  Forcing mechanism verified sound by reading: datasets.py:225 reads the flag PER SAMPLE (not captured at init);
  main.py builds GCDataset at :64 with the caller's config object, create() mutates it at :78 (both the local
  dict(config) copy AND orig_config), training loop samples at :96 => sees True. get_config default False at :3734.
  COMMIT c5348cf. LAUNCHED 22:24 tmux `s19p1`, OUTROOT=/home/adityamohan/exp_spec19, seed 0:
    d19_cs_bdens_s0 (GPU0), d19_cd_bdens_s0 (GPU1). ENVS="cs cd" = the user's ask; scene NOT launched.

G4 NULL WITHDRAWN -- IT WAS AN ARTIFACT (adversarial verification I commissioned):
  the readout zeroed r(x0) but KEPT its paired -kappa(x0). Spec 18 section 1 defines those as a de-biasing pair
  that CANCELS, leaving w^T r/(1-gamma) = exactly the MC object; dropping half injected -100*1[x0 in box] into a
  0-100 quantity, anti-correlated with truth, damage growing with box size (frac(x0 in box) .02/.05/.32).
  Verified author == mm_w - kappa_box/(1-gamma) to 3e-14. DECISIVE CONTROL: MC-vs-itself -> 1.000,
  monotone transform -> 1.000, noise -> .011-.034 => the MC side and rank machinery were always sound.
  CORRECTED (tie-corrected / ordinal), ALL 6 PASS >= .65:
    density   s0 .778/.684  s1 .789/.702  s2 .767/.676
    partition s0 .790/.692  s1 .784/.687  s2 .777/.713
  Corrected magnitudes agree too (32.9 vs MC 24.9) => calibrated in SCALE, not just rank.
  ⚠ HONEST READ, per-multiplier: mult 1.0 -> .41-.48, 2.0 -> .59-.64, 4.0 -> .83-.85. Box size drives BOTH
    variables so pooling sits above every column but the largest. Coarse-scale ordering strong; FINE-scale at the
    native eps=0.4 only MODERATE -- and fine scale is the ORS-relevant one (r_W is built from eps-scaled
    distances). Quote the per-multiplier line next to any pooled number.
  Also fixed: ~60% ties + ordinal argsort ranks capped even a perfect-on-nonzero model at ~.77 => now
    scipy rankdata. 8 tests added pinning BOTH controls so this readout-bug class cannot recur silently.
  COMMIT bf70d20.

FIRST TRAINING READS: cs @70k ce 6.55 (vs log M = 6.93), ess_median 459/1024, k_self .060, sarsa_subst .0010
  | cd @10k ce 4.52, ess_median 35.4, k_self .496, sarsa_subst .0029.
  ⚠ WATCH cs: CE near log M and ESS ~ M/2 = a DIFFUSE, near-uniform behavior occupancy. Legitimate reasons exist
    (unconditioned occupancy is broader than goal-conditioned; only 70k/500k). But if it stays there, r_W becomes
    a near-constant shaping signal -- the density analogue of collapse. P1.2 is the gate that catches it.
    cd is much sharper (k_self .496 vs .060: a 6-D conjunctive space separates points more at the same eps).

Dispatched (build-ahead, NOT launched): 19b section 2 ORS extractor + P0.2 BC-only control + repair of the
  fragile test_default_off_bit_identical (my own spec-18 review made it compare against `git show HEAD`, so it
  red-flags whenever the tree legitimately carries uncommitted gciql_flow work).

P1.2 GATE **PASSED** on cube-single: state-axis pooled Spearman = 0.8448 (n=50000) vs threshold 0.65.
  Action-axis diagnostic 0.8395 at 0.1445 coverage (not the gate).
  => SETTLES my diffuse-density concern: CE 6.53 vs log M 6.93 and ess_median ~468/1024 is HIGH ENTROPY but NOT
     a constant r_W. w varies across (s,a) enough to track the dataset's empirical discounted future distance at
     .845. Entropy of w was the wrong instrument; the gate was the right one.
  ⚠ RIGOR NOTE: I first ran the gate at pool 16384 but the shipped config trains at 4096 -- i.e. I would have
    validated a different signal than I ship. Re-ran at 4096: 0.8450 (vs 0.8448 at 16384) => r_W's ORDERING is
    insensitive to pool size, unlike the spec-18 eval readout's absolute Q (which needed P>=16384). Gate now
    validates exactly what training uses. Both JSONs kept.

P1.3 cube-single ORS extractor LAUNCHED (gate-compliant: cs passed, cd not yet measured so cd extractors NOT
  launched). GPUs 2/3/4, seeds 0/1/2, 500k, ~125 it/s (~70min).
  pool results/ors_pool/cube_single_K4096_s0.npy sha256 cac29ced42a0... chmod 444 (mean NN dist .0896).
  ⚠ LAUNCH GOTCHA: hydra `+agent_overrides.ors_reward_ckpt=` FAILS because the key already exists in the config
    as a placeholder -- must use `++` (add-or-override). First 3 launches exited 1 in <1s; caught by checking
    exit status, NOT by the log grep (which matched nothing and printed a misleading "no errors").
    Lesson: grep for SUCCESS markers, not just error strings, when confirming a launch.

P1.3 cube-single ORS COLLECTED (500k, 3 seeds): .692 / .704 / .820, mean .739
  (last-4-eval means .699/.759/.816 -> .758; within-seed last-4 spread ~.10-.12, the usual 250-episode noise).
  r_W healthy throughout: r_mean -1.66/-1.69/-1.73, std .83-.88, and ors_reward_mean > ors_r_mean confirms the
  success-row forcing to +1.0 is firing.
  GATE P1.3 cs = "within +-.05 of the density arm" (density .896/.928/.964, mean .929) => -.19 => **cs LOSS**.
  Spec 19 section 3 pre-registered: "a cs LOSS is a red flag for the shaping signal, not for ORS-the-idea --
  check the P1.2 calibration first." P1.2 PASSED at .845. So the tension is: ordering is good, extraction is not.
  ⚠ CONFOUND SPECIFIC TO THIS ARM: cs `ors_reward_scale = 4.0` is INFERRED (the paper's Table 10 has NO
    cube-single row) -- and with success rows pinned at +1.0 while r_W ~ -1.7, that constant sets the entire
    balance between sparse signal and shaping term. cd/scene scales ARE transcribed (4.0 / 1.333), so the
    DISCRIMINATORS rest on verified constants and cs is the weakest arm to conclude from. Do not read this as
    "ORS fails" -- read it as "the arm with the guessed constant lost".
  Candidate follow-up (NOT run): cs scale ablation (1.0 / 1.333 vs 4.0) to test whether the inferred constant
    explains the gap. Cheap, well-motivated, and it would decide between "wrong constant" and "shaping doesn't
    help where extraction was never binding".

P0.2 BC-ONLY CONTROL LAUNCHED (GPUs 2/3/4, 3 seeds not 1 -- spec says one run, but the comparison is against a
  3-seed arm with ~+-.10 eval noise, so a single seed would be weak evidence; deliberate, documented upgrade).
  Config is the density arm with ONLY actor_bc_only=true (q-term zeroed, critic still trains) => a gap isolates
  "the measure reaching the actor" from "bc_coeff=3.0 carrying cube-single".
  This is the missing baseline for BOTH .929 (density) and .739 (ORS) -- until it lands, neither is attributable.
  Launch confirmed by SUCCESS markers + running-job count this time, not by absence of error strings.

P0.2 BC-ONLY CONTROL COLLECTED (500k, 3 seeds): .132 / .124 / .076, MEAN .111.
  Config = the density arm with ONLY actor_bc_only=true (critic/towers still train; the ACTOR ignores q).
  => THE CLONING TERM ALONE GIVES .111 ON CUBE-SINGLE. Every cube-single number in this program is therefore
     attributable to the learned signal reaching the actor, by a wide margin:
       BC-only        .111   (bc_coeff=3.0 alone)
       delta (17a)    .207   (+.10)
       ORS (19b)      .739   (+.63)
       partition K256 .884   (+.77)
       density (18)   .929   (+.82)
       fp-noortho     .967   (project anchor)
  ⚠ THIS REFRAMES THE CUBE-SINGLE ORS "LOSS": ORS is NOT failing -- it clears the cloning floor by +.63. It is
    simply below the density arm's direct measure readout. "Shaping works, but reading the measure directly
    works better on cs" is the honest statement, NOT "ORS fails".
  Closes the standing debt that made every prior cs number unattributable (spec 19 section 1 P0.2, and the
  caveat I had been repeating since the spec-18 collection).

HEADLINE ARM FIRST READ -- cube-double ORS (s2 at 500k; s0/s1 at 420k/430k, still running):
  s0 .256 (last5 mean ~.334) | s1 .204 (~.296) | s2 .352 (~.340)
  Anchors: cd phase-1 .549 (THE GATE) | phase-2 hindsight .53 | phase-2 batch collapse .358 | BC floor ~.06-.08.
  => ORS lands ~.30-.35: ABOVE the multi-object cloning floor (so shaping IS doing real work) but BELOW the
     phase-1 anchor .549 AND below hindsight .53 -- i.e. it does NOT clear the headline gate.
  ⚠ THIS IS A WELL-CONTROLLED NEGATIVE, not a confounded one: the cd arm used the VERIFIED Table 10 scale (4.0)
    and its P1.2 calibration was the program's BEST (.970). The two available "it's a plumbing/constant problem"
    escapes are therefore BOTH excluded for cd.
  ⚠ eval noise on cd is large (a seed's last 5 evals span .16-.39) -- quote last-k means, never a single eval.
  ⚠ MISSING CONTROL: there is no cd BC-only run, so "shaping is doing work" on cd leans on the remembered
    multi-object BC floor (~.06-.08) rather than a matched control. Worth one run before this is written up.

CS SCALE ABLATION COLLECTED (seed 0, 500k): scale 1.0 -> .784 | scale 1.333 -> .732 | scale 4.0 (the shipped
  inferred value) -> .692/.704/.820 mean .739.
  => the INFERRED CONSTANT IS NOT THE EXPLANATION for the cs shortfall. Varying it over a 4x range moves success
     within noise and gets nowhere near density's .929. So the cs gap is NOT a mis-scaled reward; it is that
     reading the measure directly beats shaping on an env where extraction was never the bottleneck -- which is
     exactly what spec 19 section 3 predicted for cs ("P1 ties cs; extraction is not binding there"), except it
     ties LOW rather than tying.

PROGRAM-LEVEL READ (pre-registered, spec 19 section 3): "P1 wins cd/scene without hindsight => dense occupancy
  shaping replaces hindsight grounding => the paper's constructive answer to 9.16." That did NOT happen on cd.
  Both calibration gates passed, so the shaping SIGNAL is sound; what fails is the EXTRACTION step converting a
  well-ordered occupancy into control on a conjunctive env. Do not retire ORS on one env -- scene is the other
  discriminator and is not yet run -- but the cd result is the program's main bet missing its headline gate.

CUBE-DOUBLE ORS FINAL (500k, 3 seeds): .396 / .344 / .352 -- MEAN .364 (last-4 mean .361).
  vs anchor phase-1 .549 => **MISS**. Also below phase-2 hindsight .53, and lands essentially AT the phase-2
  batch-collapse level (.358).
  PER-TASK SIGNATURE (all 3 seeds agree): task1 .84-.86 but tasks2-5 .06-.48. ORS solves the first task well and
  degrades sharply on the rest -- a specific, reproducible pattern worth explaining, not just an average.
  Controls that make this a CLEAN negative: verified Table 10 scale (4.0), best-in-program calibration (.970),
  3 seeds. "Wrong constant" and "bad shaping signal" are both excluded.

CS SCALE ABLATION KILLED MY OWN LEADING HYPOTHESIS: scale 1.0 -> .784, 1.333 -> .732 vs shipped 4.0 mean .739.
  A 4x range in the constant moves success within noise and nowhere near density's .929 => the cs shortfall is
  NOT a mis-scaled reward. Retract the "the arm with the guessed constant lost" framing I gave earlier; the
  correct statement is that direct measure readout beats shaping on cs, where extraction was never binding.

CD BC-ONLY CONTROL LAUNCHED (3 seeds, new configs/gciql/cube_double_state_bconly_d50.yaml) -- closes the gap
  where the cd "shaping still does work" claim rested on a REMEMBERED floor (~.06-.08) rather than a matched run.
SCENE deliberately NOT launched: it is the other discriminator and where published ORS is strongest (80+-4), so
  it matters MORE now that cd missed -- but the user scoped this request to cube-single and cube-double, and
  starting a new env is a scope widening, not a gap-closure. Offered, awaiting a decision.
