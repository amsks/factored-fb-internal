# Value-Function Atlas — execution decisions

## Split-oracle route for C1/C2 (spec §4) — RESOLVED: dataset states

Task 3 gate, 2026-07-14. 16 contact-rich dataset rows (obs[18]>0.5, cube lifted,
>=20 in-episode steps remaining), qpos/qvel restore, replay recorded actions for 20 steps,
compare to recorded continuation.

- cube-xyz drift: median 0.0024 m, p90 0.0030 m, max 0.0044 m
- full-obs drift: median 0.068, max 0.175 (this is the ARM, via the unrestorable
  `_target_effector_pose` IK integrator; it does not move the cube)
- 16/16 rows under the 0.04 m success threshold

**Decision: C1/C2 use literal dataset states with qpos/qvel restore.** The unrestorable IK
integrator perturbs the arm observation but the cube (what the 4cm criterion measures) stays
within ~4mm over a 20-step horizon. The μ-rollout fallback is NOT needed.

Caveat carried forward: the arm-obs residual means any *arm-dependent* oracle claim at C1/C2
should be read with the ~0.07-0.18 obs perturbation in mind; cube-placement claims are safe.

## Critic registry + checkpoint step policy (Batch 2) — RESOLVED

Checkpoints live on the PERSISTENT disk at
`~/bridge_ckpt_backup/gciql_outputs/factored-fb/<group>/` (NOT /dev/shm, which is volatile RAM
and had been cleared). Identity confirmed by `agent_name` in flags.json, not the group label.

| critic          | group                              | agent_name  | arch     | actor  | STEP USED |
|-----------------|------------------------------------|-------------|----------|--------|-----------|
| bilinear_flow   | GCIQL-State-Flow-Bilinear-d50      | gciql_flow  | bilinear | flow   | **500k**  |
| bilinear_ddpgbc | GCIQL-State-Flow-Bilinear-d50-ddpgbc| gciql_flow | bilinear | ddpgbc | **500k**  |
| mlp_flow        | GCIQL-State-Flow                   | gciql_flow  | MLP      | flow   | **1M**    |
| mlp_ddpgbc      | factored-fb-gciql                  | gciql (stock)| MLP     | gaussian ddpgbc | **500k** |
| simplesf        | GCIQL-State-Flow-SimpleSF-Arch     | gciql_flow  | SF       | flow   | **500k**  |
| fb              | factored-fb_outputs/*fb_flowbc*    | torch FB    | FB       | flowbc | final     |

Step policy: 500k for all EXCEPT mlp_flow, which saved no 500k intermediate (only 1M, every
seed). Using its 1M is a CONSERVATIVE confound: the MLP side of "the win" comparison gets 2x
the training and still loses to 500k bilinear, so the bilinear advantage is understated, not
inflated. Recorded per user decision 2026-07-15.

Notes: mlp_ddpgbc (stock `gciql`) needs a SEPARATE jax loader path (not gciql_flow); its seeds
start at sd001 (no sd000) -- use the three lowest seeds consistently. simplesf secured to
persistent disk (was only in volatile /dev/shm).

## jax runtime (Batch 2) — VERIFIED via end-to-end smoke test

- Atlas jax code MUST run under `/dev/shm/.venv-jax/bin/python` (jax 0.10.0). The repo `.venv`
  has NO jax. This venv is on VOLATILE RAM -- gone on reboot; rebuild if missing.
- Required sys.path inserts (from oodgrid_run_jax.py:18-21), in this order:
  `R, R/scripts, R/third_party/ogbench/impls, R/tools/wandb_mode_shim`  (R = repo root).
  Without `tools/wandb_mode_shim` on path, `import gciql_flow` fails on `ogbench_flow`.
- Load: `restore_agent(GCIQLFlowAgent.create(0, zeros(1,28), zeros(1,5), cfg), CK_DIR, step)`
  where cfg = get_config() updated with flags.json['agent'] intersecting keys, encoder=None.
- VERIFIED call signatures on a real bilinear_flow @500k checkpoint:
  - `agent.network.select('critic')(obs[B,28], goals[B,28], actions[B,5]) -> (q1[B], q2[B])`
  - `agent.network.select('actor_vf')(obs[B,28], actions[B,5], t[B,1]) -> vel[B,5]`
  - `agent.sample_actions(observations=, goals=, seed=PRNGKey, temperature=0.0) -> [B,5]`
- Run with `XLA_PYTHON_CLIENT_PREALLOCATE=false`. CPU-only eval via `CUDA_VISIBLE_DEVICES=""`.
- TEST SPLIT: test_config/prefix_replay/split_oracle run under `.venv`; test_mu_flow/critics/
  build_atlas must run under the jax venv (they import jax). Document per-module.

## Batch 2 built artifacts (2026-07-15)

- `analysis/atlas/banks.npz` (1.8 MB): 4 phases x 32 states, A_mu [4,32,256,5], built from
  bilinear_flow's mu (228s). All phases filled 32/32 — P3 (transport) reached in ~3 episodes,
  so the spec's "mu rarely reaches transport" risk did NOT bite (this mu is a competent grasper).
- End-to-end atlas verified on live checkpoints: ||A||_F=1.0; over the 5 eval goals bilinear
  top-sv 0.969 < mlp 0.980 (bilinear more mode-distributed — a first multi-modality signal,
  n=1 state). rank=5 is the n_g ceiling here; use the 64-goal grid for real effective-rank.

## Phase 0A gate (Task 9) — PASSED. Exploitation branch (Primary B) SURVIVES.

Median whitened distance to the mu-reference action cloud (P0+P2, 8 states x 5 eval goals):

| actor / critic  | bilinear | MLP  |
|-----------------|----------|------|
| DDPG+BC         | **17.9** | 1.76 |
| flow            | 2.37     | 2.00 |

branch_alive = True (d_ddpgbc_bil 17.9 > d_ddpgbc_mlp 1.76, ~10x).

Reading: the bilinear critic HAS attractive off-reference extrema -- the DDPG+BC actor finds
and chases them (17.9), while the flow actor is distillation-anchored and stays put (2.37). The
MLP critic has no such extrema, so its DDPG+BC actor also stays near (1.76). This is the
mechanism for the actor reversal (bilinear+flow >> bilinear+ddpgbc). Phase 0B (oracle-based
causal confirmation: are those distant actions actually WORSE?) is pending the oracle subsample.

## Task 10 Primary A — code done; run in progress; two issues found by testing

Oracle discrimination requires H=40 (not smoke's H=12): at H=40 J varies across first-actions
for EVAL goals (std ~1.7; good actions reach the target, bad ones don't). C2/C3 (eval goals)
are runnable now; a real run (C2,C3 x 4 states x 16 acts x H40 x 2 reps, ~1h) is backgrounded
-> analysis/atlas/primaryA_evalgoals.json.

ISSUE for C1/C4: goals_hindsight were RANDOM dataset rows -> unreachable from the state ->
oracle J constant (std 0) -> no ranking signal. FIX (committed in banks.py): per-state
REACHABLE hindsight goals = observations mu reaches by rolling forward from each state
(goals_hindsight_ps [4,K,NH,28]). Requires a banks REBUILD (~20 min of mu rollouts) and a
primaryA update to read per-state hindsight goals before C1/C4 can run. NOT yet done.

## Oracle-free atlas analysis (2026-07-15) — multi-modality REFUTED; difference is in CONTENT

32 states/critic, 64-goal grid (rank not n_g-capped), mu-reference actions. No simulator.

| critic         | eff_rank    | top_sv_share | basins |
|----------------|-------------|--------------|--------|
| bilinear_flow  | 6.65 ±1.68  | 0.500        | 1.00   |
| mlp_flow       | 6.81 ±1.41  | 0.486        | 1.00   |
| simplesf       | 6.99 ±2.02  | 0.479        | 1.00   |
(FB skipped: torch not in jax venv -> needs a torch-venv run.)

bilinear vs MLP goal-mode principal angles (top-3): [53.7, 71.5, 83.3] deg.

FINDINGS:
1. eff_rank + mode concentration are the SAME across bilinear/MLP/simplesf (within SD). The
   earlier n=1 teaser (bilinear top_sv 0.969 < mlp 0.980 -> "more mode-distributed") does NOT
   survive at n=32 (0.500 vs 0.486). NO spectral-shape difference.
2. ALL unimodal (basins=1.00). => the multi-modality SECONDARY hypothesis is REFUTED. The
   actor reversal is NOT "Gaussian averages across modes" -- there is only one mode. Consistent
   with Phase 0A: the mechanism is OFF-SUPPORT extrema, not multiple in-support modes.
3. Principal angles [53.7,71.5,83.3] => bilinear and MLP span DIFFERENT goal-preference
   structure despite identical spectral shape. Same rank, different CONTENT.

CAVEAT: principal angles need a null baseline (same-critic different-seed = noise floor;
random rank-3 in 64-D ~ orthogonal) before calling 53.7 deg "different." NEXT.

## Primary A oracle run (2026-07-15) — INCONCLUSIVE (broken oracle discriminator, NOT a tie)

2h 3min run. Result: C2 bilinear rho=0.011 mlp=-0.029; C3 bilinear=-0.005 mlp=0.022. All ~0.

DO NOT read this as "bilinear and MLP rank equally." It is a NULL from a broken instrument.
Root cause VERIFIED: the oracle J floor-clusters. Over 8 actions at C2: J = [-33.1 x5,
-29.35 x2, -28.77 x1] -> only 3 distinct values, 5/8 at the floor. Spearman over 60%+ tied
data is ~0 regardless of the critic.

WHY: the oracle continuation is GOAL-AGNOSTIC mu. It doesn't steer toward goal g, so whether
the cube reaches g within H=40 depends on luck, not the first action -> most actions -> floor
(never reach g) -> J is mostly ties. The sparse discounted-hitting reward has no gradient for
the majority of actions.

FIX (next): graded J = -(min cube-to-goal distance over the rollout), so EVERY action gets a
continuous score reflecting how close behavior-continuation got to g. No floor, Spearman
meaningful. Cheap swap in oracle.py. Also worth: speed up (2h is too slow -- batch the mu
continuation, or shorten H with the graded reward which needs less horizon).

UNAFFECTED (these do NOT use the oracle J ranking, still solid):
- Phase 0A off-support exploitation (17.9 vs 1.76) -- the actor-reversal mechanism.
- Oracle-free: multi-modality refuted, principal angles [53.7,71.5,83.3].

## Angle calibration + off-reference critic test (2026-07-15) — both oracle-free

### Principal-angle calibration (top-3 goal modes, deg; 0=identical, 90=orthogonal)
| pair                     | mode1 | mode2 | mode3 |
|--------------------------|------:|------:|------:|
| bilinear vs bilinear (seed) | 17.5 | 38.1 | 70.4 |
| MLP vs MLP (seed)           | 22.2 | 41.6 | 62.7 |
| **bilinear vs MLP**         | **53.7** | 76.2 | 85.8 |
| random vs random (ceiling)  | 71.0 | 79.9 | 86.9 |

=> The content difference is REAL: on the dominant mode, bil-vs-mlp 53.7 deg is far above the
same-critic seed floor (~20 deg) and below the random ceiling (71 deg) -- partial shared
structure but a genuine divergence, not seed noise. Modes 2-3 are ~random (no shared structure).

### Off-reference critic test (bilinear_ddpgbc vs mlp_ddpgbc, own co-trained critic+actor)
| critic          | Q(a_DDPG) - max_ref | frac>0 | ||grad_a Q|| |
|-----------------|--------------------:|-------:|-------------:|
| bilinear_ddpgbc | **+2.142**          | **0.787** | 3.18 |
| mlp_ddpgbc      | -0.257              | 0.062  | 3.06 |

=> ATTRACTIVE OFF-REFERENCE PEAK CONFIRMED. The bilinear critic values the DDPG+BC action ABOVE
every behavioral-reference action 79% of the time (median +2.14); the MLP does so only 6%
(median -0.26, i.e. below the best reference). The actor climbs a real critic peak that lies
off the mu support.
=> STEEPER-LANDSCAPE ALTERNATIVE RULED OUT: ||grad_a Q|| is essentially identical (3.18 vs 3.06),
so it is not that bilinear has globally steeper action-gradients shifting the BC/RL tradeoff --
it is specifically an off-support MAXIMUM exceeding the on-manifold max.

CAVEAT: path_rise as coded == the peak endpoint diff (redundant); monotonicity ALONG the path
not separately captured. Peak result is the load-bearing one and is solid.

## Pre-delete checkpoint audit (2026-07-15) — PASSED, safe to delete /dev/shm

Systematic 4-boundary audit of the persistent copies before deleting the /dev/shm source:
1. COUNT/SIZE parity: exact file-count match per tree (gciql 2560, factored-fb 7552, ds 36500).
2. CHECKSUM byte-parity (rsync -c): **0 files differ** on ALL THREE trees -> byte-faithful.
3. LOADABILITY: 24/24 sampled checkpoints load (12 jax pkl under jax venv, 12 torch pt).
   NB the first "12/12 BAD" was an interpreter mismatch (jax pkl under torch venv), not corruption.
4. ATLAS-CRITICAL: all 5 jax critics + FB critic load end-to-end; FB Q works from the MOVED
   persistent datasets (critics.py path updated + committed).

VERDICT: the /dev/shm copies of gciql_outputs, factored-fb, datasets are byte-identical to
persistent and load correctly. SAFE to delete them (reclaims ~165 GB RAM). Keep .venv-jax.
Inventory: analysis/atlas/EXPERIMENT_INVENTORY.csv (85 groups, 483 runs).

## Full per-group audit closed (2026-07-15) — all 85 groups audit=yes

The 42 persistent-only groups (never in /dev/shm, so no checksum baseline) were loadability-
verified per group: loaded ONE checkpoint from each -> 41/41 torch OK + 1/1 jax OK.
Combined with the 43 already-verified (checksum byte-parity or loaded end-to-end), ALL 85
experiment groups are now individually verified. audit=no count: 0.

## Action-grounded mode decode (2026-07-15) — THE ANSWER: direction vs distance

Part A (128 bank states P0-P3, 5 eval goals) + Part B (120 dataset states). Fast (~3 min/part).

**What the dominant modes ENCODE (consistent across all 4 phases):**
- BILINEAR dominant goal-mode = goal DIRECTION (goal_dir_x/y, |r|~0.69-0.85), 28-30/32 states.
- MLP dominant goal-mode = goal DISTANCE (goal_dist, |r|~0.75-0.87), 27-29/32 states.
  (MLP shifts to direction only at P3.) Both action-modes = cube displacement (dx_z lift / dx_x).

=> This IS the content of the 53.7deg basis difference: bilinear factors WHICH-WAY (direction);
MLP encodes HOW-FAR (distance). Direction is what steering needs; distance is a magnitude that
does not tell you the way. => candidate 3 (directional action-goal composition), plan Outcome B.

**Part B — bilinear steers MORE than the goal-agnostic play policy** (T_dir, one-step, tiny
magnitudes but consistent ordering):
- P2 at-contact: bil +0.0002 > data -0.0002, > mlp
- P3 holding:    bil +0.0049 > mlp +0.0040 > data +0.0026
=> bilinear is NOT just replaying the dataset's action distribution (rules out scripted
primitives, candidate 4); it steers toward the specific goal more than the play policy did.

CAVEATS: goal decode is over 5 eval goals (small n per state; the CROSS-STATE consistency
28-30/32 is the evidence) -> confirm on the 64-goal grid. T_dir magnitudes are one-step and
tiny; the ordering is the signal. Descriptive/correlational (per plan), not yet the causal
keep/remove surgery.

## Grid check REFUTES direction-vs-distance (2026-07-15) — retracted

64-goal grid re-run of the mode decode: dominant-goal-mode mean |r| collapses from 0.75/0.80
(5 eval goals, bil/mlp) to 0.37/0.26 (64 goals), and best-field scatters (bil: dist 38/dir 37/
height 24; mlp: mostly goal_dir_y 86/128 -- opposite of the 5-goal story). => the 5-goal
"bilinear=direction, MLP=distance" was small-n inflation. RETRACTED. Finding 2 (real 53.7deg
content difference) stands; its physical NAME is unknown. Doc updated. Next = causal keep/remove
surgery, not another correlational decode.

## Causal value-surgery attempt (2026-07-15) — two design walls, no result

Started plan Step 4 (keep/remove necessity+sufficiency scored by simulator outcome).
- v1 first-action override: overriding only step 1 then continuing with the bilinear flow actor
  does NOT discriminate (all operators = 0.33 success) -- the continuation washes out the single
  action. Dead design.
- v2 induced-policy rollout (each operator drives every step; physical candidates re-scored by
  one-step scratch-env branching to keep the rollout exact): CORRECT but expensive --
  branching is O(H*N*episodes) sequential mujoco steps; first pass >5min for 6 states -> full
  run is multi-hour, not the ~1hr I estimated. Killed the smoke; committed WIP (784d927).
NOTHING concluded. Options: scoped-down induced run (~30-60min, wide bars) | full build | defer.
Findings 1 (off-support peak) + 2 (53.7deg content) stand; Finding 3 (direction/distance)
retracted. Doc: Finding 4 = this status.

## In-hand ranking under a SHARED continuation (2026-07-15) — NULL: it's co-adaptation

Cheap causal test (user design): from in-hand states (P2/P3 bank, n=160/phase), roll ONE shared
continuation policy -> success label Y; score each critic by Q_m(s, a0, g) at the shared first
action; compare AUC(Q_bil,Y) vs AUC(Q_mlp,Y). Both shared policies run (bil, mlp).

| shared | phase | AUC_bil | AUC_mlp |
|--------|-------|---------|---------|
| bil    | ALL   | 0.529   | 0.582   |
| mlp    | ALL   | 0.556   | 0.543   |
(P2/P3 split: bil-policy P3 mlp 0.616>bil; mlp-policy P3 bil 0.602>mlp -- it FLIPS.)

All AUCs near chance (0.37-0.62); differences within ~1 SE (~0.04); the ordering flips with the
shared policy. => the bilinear critic does NOT independently rank in-hand success better than
MLP. Per the test's logic: the ranking advantage does NOT survive removing critic-actor
co-adaptation => the performance win is a property of the CO-ADAPTED critic-actor SYSTEM, not the
critic's value in isolation. Ties to Finding 1 (bilinear critic only usable WITH the flow actor).
Caveat: AUC of first-action-Q vs 150-step rollout is coarse; robust as a null, not a fine measure.
