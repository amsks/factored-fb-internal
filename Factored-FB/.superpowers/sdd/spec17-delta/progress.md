# SDD ledger — plan: PAPER/Multistep-FB/specs/17-delta-measure.md (+17a, +17b)

BASE (dirty tree, spec-16 uncommitted): 2b442e5d8a9b20785cef5b10cdf09b5a663c71ef
Worktree: NONE (in-place on gciql-fb-bridge; spec-16 measure_nstep is uncommitted and load-bearing for 17a lattice)
Policy: no commits, no staging (shared branch, other track dirty)

Wave 1 dispatched (parallel, disjoint files):
- Task 17a (sonnet): gciql_flow.py measure_delta + tests/test_gciql_measure_delta.py + 5 configs
- Task 17b-tests (sonnet): scripts/rescuing_fb/make_delta_prototypes.py + tests/test_gciql_measure_delta_partition.py (tests-first, 9/10 expected red) + partition config
Wave 2 (queued, serialized on gciql_flow.py): Task 17b-agent — §2 of spec 17b, make wave-1 tests green.

Anchors verified by controller for the 17b brief:
- actor_loss flow q-term: gciql_flow.py:1040 ; actor_q_center normalizer: :1057
- total_loss measure_only branch: :2072
- update/target_update inherited from third_party/ogbench/impls/agents/gciql.py:157/:147 ; train loop calls agent.update(batch) at third_party/ogbench/impls/main.py:97 (spec 17b 2.8 confirmed)
- module registration blocks (center_b/eps_gain/nhat_head): :2921-2944 ; params['modules_target_critic']=... at :3118 ; ex_goals bound at :2214 ; `import copy` at :12 ; ensemblize imported at :22

Task 17a: COMPLETE (uncommitted, per policy). Controller-verified against spec 17a line-by-line:
  - get_config knob :3238 ; measure_loss sentinel :1400 ; delta block :1525-1534 (hit_c from batch['observations'] -> off-by-one convention CORRECT) ; info keys :1723-1725 ; assert-lattice :2634-2655 (all 10 refusals present)
  - 5 configs correct: cs has measure_stopgrad_psi:false (the FrozenPsi trap avoided), eval_interval 10000 on cd/sc + 50000 on cs, 5 distinct run_groups, hit_specs match bases
  - agent-reported: 8/8 delta tests + 26/26 neighborhood green; sabotage check confirmed 3 tests fail without hit_n-hit_c
  - deviations (accepted): test 5 split in two; test 2 uses a _measure_terms capture instead of re-deriving the stochastic bootstrap; cs run_group name chosen (spec left it unnamed) = GCIQL-CubeSingle-Flow-Measure-Delta-d50

Task 17a: controller-independent verification: `.venv/bin/python -m pytest tests/test_gciql_measure_delta.py -q` -> 8 passed in 78.63s. CONFIRMED.
Task 17b-tests: COMPLETE. scripts/rescuing_fb/make_delta_prototypes.py + tests/test_gciql_measure_delta_partition.py (10 tests) + configs/gciql/cube_single_state_measure_delta_partition_d50.yaml.
  green 2/10 (default_off_bit_identical, prototype_script); red 8/10 all confirmed expected-red (missing symbols, per-test mapping in task-17b-tests-report.md).
  carried risks -> handed to 17b-agent: (i) tests 3/7 assume the actor_loss rng-split sequence is not shifted (use fold_in); (ii) test 8 only proves *some* AssertionError -- needs per-refusal re-verification after 2.7 lands.
Task 17b-agent: DISPATCHED (sonnet), serialized on gciql_flow.py after 17a.
Controller-independent regression (post-17a, pre-17b-agent): nstep/hit_spec/axes/scale/zmix -> 26 passed in 281.51s. CONFIRMED.

Task 17b-agent: COMPLETE. gciql_flow.py only. Controller-verified by reading: delta_protos field :753, _delta_phi :1306, delta_partition_loss :1769 (p_tgt=(1-g)phi(x')+g*p_bar, next-state convention, sg'd), total_loss wiring :2170, actor q-term :1029 (frozen params, no grad_params), update override :2267, lattice :2767-2799, protos load :3131.
  Deviations (accepted): compound asserts split into named ones; actor branch is an elif in the existing chain; 17b's mutual-exclusion assert unreachable for measure_delta=True (17a's block fires first) -- correct but dead for that input.
Controller-independent FULL suite (post-17a+17b): delta_partition 10 + delta 8 + nstep 6 + hit_spec 6 + axes 6 + scale 5 + zmix 3 = 44 passed in 461s. CONFIRMED.

Launch prep (controller): results/delta_protos/cube_single_K256_s0.npy generated [256,3] float32, mean NN dist 0.2616 vs eps 0.4. scripts/rescuing_fb/launch_spec17_cs.sh written, DRY_RUN clean (6 jobs, GPUs 0-5, from scratch, no warm start).
User decisions: wave = both spec-17 cs arms x 3 seeds x 500k. Commit scope = agent + spec16/17 tests + delta configs + datasets.py mstep; train.py EXCLUDED (other track).
Final review: DISPATCHED (opus) -- gate for commit + launch.

FINAL REVIEW (opus): FINDINGS -- 2 launch blockers, both controller-verified:
  C1 measure_coef default 0.0 x total_loss:2178 => partition CE weighted by ZERO. delta_head frozen at random init; run would log a plausible fake NULL on gate G3. (spec 17b section 6's "drop measure_coef" instruction is the trap.)
  C2 delta_protos pytree data field => flax.serialization.from_state_dict(agent, load_dict['agent']) at flax_utils.py:198 raises Missing field on EVERY pre-17b checkpoint => breaks main.py --restore_path and ~14 eval harnesses incl. the spec-14 L2 multi-reward battery. Spec 17b 2.2's "old checkpoints keep loading" is FALSE.
  H3 partition test base_cfg omitted measure_coef => target-soft-update test degenerated to old==old and passed with the update() override DELETED.
  M4 msq computed and discarded => gate G2 has no instrument. M5 delta targets are event-rate-scale vs ballvisit dwell-mass-scale (science property, NOT fixed by ruling). L6 unreachable 17b assert (ruled: leave). L7 double weight knob (subsumed by C1 fix). L8 stale test docstring.
  Verified CORRECT by review: 17a hit_c/hit_n sources + the J=(1-g)M-h identity under absorbing=False; hindsight column consistency; 17b p_tgt simplex + next-state convention; grad hygiene both directions; default-off bit-identity; all 6 configs incl. the FrozenPsi trap; make_delta_prototypes; anti-scope; eps scaling; log(p+1e-8) safety.
Fix round 1/5: DISPATCHED to the original 17b implementer (C1, C2, H3, M4, L8; L6+M5 ruled no-change). COMMIT AND LAUNCH ARE BLOCKED until this closes.
Fix round 1/5 delivered (C1 total_loss partition branch de-gated from measure_coef + new test #11; C2 delta_protos -> nonpytree_field hashable tuple + new test #12, hand-verified against the real cd_fp_s0 ckpt; H3 target-soft-update test rewritten to be override-sensitive; M4 measure_delta_msq key; L8 docstring). Agent-reported 20/20 + 26/26 green.
Scoped re-review: DISPATCHED (opus).

Re-review (fix round 1): ALL ADDRESSED. Evidence: delta_head grad 3352 under the shipped yaml; real cd_fp_s0 pkl restores; 1 XLA compile / 12 steps (static field does not retrigger tracing); uint32-equal proto round-trip; H3 override-deleted -> FAIL 1.55e-6, restored -> PASS 7.45e-9.
  3 new items, none launch-blocking: MED protos not serialized => the partition a 17b ckpt was trained on lives only in the .npy (MITIGATED: file + backup chmod 444, sha256 b4e13e9d... logged by the launcher and recorded here); LOW stale docstring; LOW tautological test-12 assert; LOW measure_coef silently inert on the 17b path.
Fix round 2/5 (the 3 LOWs): delivered. test 12 now restores an ON-knob agent from a DIFFERENT proto file's state dict and asserts delta_protos survives unclobbered + _delta_phi bit-identical; create() now refuses measure_coef>0 under measure_delta_partition.

COMMIT 79a9c6b (round-1 state): gciql_flow.py + datasets.py + 3 test files + make_delta_prototypes.py + launch_spec17_cs.sh + 6 delta yamls. train.py excluded (other track).

WAVE LAUNCHED 2026-08-06 03:50 tmux `s17cs`, OUTROOT=/home/adityamohan/exp_spec17, 6 jobs GPUs 0-5, from scratch, 500k, seeds 0/1/2:
  d17_cs_delta_s{0,1,2}  cube_single_state_measure_delta_d50
  d17_cs_part_s{0,1,2}   cube_single_state_measure_delta_partition_d50
  First reads @15-35k: part entropy 5.545(logK) -> 3.99 (s0@35k), p_max 0.118 vs uniform 0.0039 => the head IS training (the C1 bug would have pinned it at logK).
    delta arm: event_rate ~0.0013, hit_level ~0.0148 (events ~9% of dwell mass, as designed), msq 0.34-1.70 => G2 instrument live and bounded away from 0.

COMMIT 370938f (round-2 LOWs): real ON-knob restore test + create() refuses measure_coef>0 under 17b.

EVAL-SIDE (post-wave pieces, in flight):
- eval_zeroshot_rewards.py `--delta`: DONE round 1 (10 tests; refusals both directions + --force escape; off-path passthrough returns the raw fn object unchanged). AUDIT FINDING (controller-verified): `q_at_z` (t6_gpi_eval.make_q_at_z) is the ONLY raw critic-measure readout in the harness -- dirac/region/dense/linear/reachavoid read env success post-rollout or psi(g) for z-inference, none needs conversion (columns unchanged by delta; the actor already argmaxes correctly since J preserves the action argmax). Consequence: `mixture --gpi` is cube-single-only, so the cd/sc delta arms never touch the converted site at all.
  ROUND 2 SENT: the conversion used a nearest-state PROXY for h; exact is the same-weights weighted hit, Sum_j w_j h_{g_j}(x0) = hit_indicator(obs,S,blocks) @ w, with w from moment_z (retrieval = k-hot; ridge = push-through alpha=(PsiPsi^T/n+lam I)^-1 w, else proxy + recorded flag). Matters ONLY for cross-z GPI candidate ranking -- which is exactly what --delta exists to make correct.
  Also flagged by the agent: the harness has PRE-EXISTING run-to-run nondeterminism at fixed --seed, unrelated to this change. Do not mistake it for a regression later.
- eval_partition_zeroshot.py + 13 tests: DONE round 1 (Q_r=(r(x0)+(p-phi(x0)).z_r)/(1-gamma) vs hand-derived + the constant-reward c/(1-gamma) identity; phi imported live from _delta_phi; protos sha256-verified against the pinned digest and cross-checked vs agent.delta_protos post-restore; gamma/hit_spec from flags.json).
  ROUND 2 SENT: its rollout was Q_r-greedy over noise-sampled candidates = NOT the spec-14 protocol (retrieval-k1 decode), so the numbers would not have been comparable to the spec-14 table. Adding --decode retrieval (default) + qr_greedy, decode mode recorded in the JSON.
  UNVERIFIABLE until a ckpt lands (first save 100k): end-to-end restore + rollout sanity for both scripts. Neither agent fabricated a result.

FIRST PRE-REGISTERED READS @150k-260k of 500k (NOT final):
- 17b PARTITION: success 0.744 / 0.884 / 0.804, p_entropy 3.45-3.64 vs the G3 threshold 0.9*log256=4.991.
  => G3 (success>=0.5 AND entropy<4.991) PASSING ON ALL 3 SEEDS well before 500k. This is spec 17's PRIMARY
  cube-single claim: a one-stage occupancy-semantics method that trains an actor at all. p_max 0.123-0.141 vs
  uniform 0.0039. OPEN CONTROL: cube-single from-scratch anchors span 0.004-0.5 (unstable), so 0.74-0.88 is
  above the band -- but a bc_coeff=3.0 flow-BC-only control on cube-single has NOT been run in this wave and
  would be the clean attribution. Do not claim the measure caused it without that control.
- 17a DELTA: success 0.052 / 0.064 / 0.084 @100k = BC floor, and msq decayed 0.34-1.70 (@25k) -> 0.018-0.028
  (@150k), ~50x toward zero while event_rate stays ~0.0011 and hit_level ~0.014.
  => this is EXACTLY spec 17 section 4's pre-registered signature: "msq -> 0 with flat loss + success -> BC
  floor = ZERO-FLUX COLLAPSE (the relocated constant). Reported as: centering relocates, does not fix."
  Read stands unless 150k->500k reverses it (decay is monotone so far). G2 therefore looks FAILING for 17a.

EVAL-SIDE round 2 both DONE:
- eval_zeroshot_rewards.py --delta: exact weighted h = Sum_j w_j h_{g_j}(x0) via compute_z_weights() (moment / retrieval k-hot / ridge push-through alpha=(PsiPsi^T/n+lam I)^-1 w, gated RIDGE_EXACT_MAX_N=2000 then proxy fallback recorded as h_construction: proxy_decode). w recomputed PER mixture cell (r, hence w, differs per cell). 18 tests.
  ⚠ SCOPE CORRECTION (agent's finding, honest): within ONE gpi_select call z_r is FIXED, so h is constant across candidates and the argmax is provably invariant to exact-vs-proxy. The exact-h fix therefore matters for QUOTING Mhat values (calibration / L2 numbers), NOT for today's within-call action selection. My cross-z-ranking rationale in the round-2 brief was too strong; the fix is still required, its scope is narrower. A constructed ranking-flip case exists only when the proxy is steered onto a zero-weight decoy.
- eval_partition_zeroshot.py: --decode retrieval (DEFAULT, mirrors eval_zeroshot_rewards' actor_cond='goal' deployment: same candidate pool/rng, decode g_hat once, respects --retrieval-k, fixed-goal rollout) | qr_greedy (original). Candidates scored by the same Q_r algebra with a_j~actor(x0,g_j) at a canonical seed-0 reset x0. decode/retrieval_k/n_candidates recorded in the JSON. 22 tests incl. a source-inspection check that the two decodes call disjoint function sets.

EVAL RUNNER scripts/rescuing_fb/run_spec17_eval.sh committed; SMOKE-VERIFIED END-TO-END on the two partition
ckpts already at 500k (this closes the "unverifiable until a ckpt lands" gap for the 17b readout):
  d17_cs_part_s0  dirac 0.82 | dense 0.86 | region 0.17
  d17_cs_part_s2  dirac 0.80 | dense 0.70 | region 0.20      (10 episodes = PROVISIONAL; protocol is 25)
  => the 17b arm transfers zero-shot to point-goal-like reward families (dirac/dense) but NOT to extended
     regions (0.17-0.20). Re-run at --episodes 25 once all 6 runs land, then add both run_groups to
     scripts/results_map.py and refresh the CSVs.
  JSON meta records: gamma 0.99, hit_spec 19:22:0.4, protos path+sha256 b4e13e9d..., the Q_r formula, decode
  retrieval k=1, n_candidates 8, horizon 200, seed 0. Nothing quotable without its convention.

⚠ INCIDENT 2026-08-06: d17_cs_part_s1 CRASHED at 04:06 at step ~70k with
  jax.errors.JaxRuntimeError: INTERNAL: CUDA error: Failed to end stream capture:
  CUDA_ERROR_STREAM_CAPTURE_INVALIDATED
  = XLA/CUDA infrastructure fault, NOT a code bug (the other 5 cells with the same code ran to 500k clean).
  Dead run preserved at .../d17_cs_part_s1.crashed_70k_cuda_1043 + /dev/shm/spec17_logs/d17_cs_part_s1.crashed.log.
  CONTROLLER ERROR: my completion monitor tested `pgrep -fc 'main.py.*exp_spec17'`, and the monitor's OWN shell
  command line contains that literal pattern => self-match => the count never fell below 1 => the stall branch
  could never fire. It idled ~6.5h before I noticed by hand. Fix: the relaunch monitor keys on
  params_500000.pkl + train.csv staleness (>1200s), no pgrep at all. Do not write a pgrep-based liveness check
  whose pattern appears in its own command line.
  RELAUNCHED 10:43 tmux `s17cs1`, GPU0, from scratch (no resume path in this launcher).

FINAL 500k READS, 5/6 cells (part_s1 rerunning):
  goal-based (training loop, 5 tasks x 50 eps):
    delta   s0 0.152  s1 0.264  s2 0.204        (vs fp-noortho anchor 0.967; inside the 0.004-0.5 unstable band)
    part    s0 0.884  s2 0.884                  (identical means are COINCIDENCE - per-task profiles differ:
                                                 .90/1.00/.98/.72/.82 vs .98/.98/.98/.68/.80; each = 221/250)
  zero-shot multi-reward (10 eps, provisional; protocol is 25):
    delta   s0 dirac .44 region .11 | s1 dirac .38 region .23 | s2 dirac .36 region .05
    part    s0 dirac .82 dense .86 region .17   | s2 dirac .80 dense .70 region .20
  GATES: G3 (17b) PASSES on every seed - success .884 >= .5 AND p_entropy 3.44 < 4.991.
         G2 (17a) FAILS - msq .34-1.70 (25k) -> .018-.028 (150k) -> .0042/.0044/.0100 (500k), ~100-400x decay to
         zero with event_rate flat at .0011 = spec 17 section 4's ZERO-FLUX COLLAPSE verbatim
         ("centering relocates, does not fix").
         G4 (Spearman >= .65 ordering) NOT YET COMPUTED.
  --delta conversion validated end-to-end on real delta critics (exact_weighted h, gamma .99, hit_spec 19:22:0.4).
