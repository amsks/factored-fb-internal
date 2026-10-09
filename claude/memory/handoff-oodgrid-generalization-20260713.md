---
name: handoff-oodgrid-generalization-20260713
description: "READ FIRST (bridge track) 2026-07-13: the FB failure is IN-DISTRIBUTION, not generalization; bilinear-vs-MLP is a pure OOD-start interaction. OOD grid sweep + bilinear_sg training RUNNING."
metadata: 
  node_type: memory
  type: project
  originSessionId: 93adbdfc-d736-4213-8901-55257cd9f46b
---

# 2026-07-13 bridge track — the phase/generalization decomposition

## ⚠ ALL THREE CAMPAIGNS ARE DONE AND COLLECTED (2026-07-14). NOTHING IS RUNNING.

Results written up in **`PAPER/fb-gciql-bridge/oodgrid-results-20260714.md`** — read that,
it supersedes finding #2 below. `bilsg` is in the living CSVs (`bilinear-sg-d50` row in
`results/bridge_grafts.csv`); ckpts backed up to `~/bridge_ckpt_backup/GCIQL-State-Flow-Bilinear-SG-d50`.
OOD grid: 133 groups / 347 run-dirs in `analysis/oodgrid/results/`; aggregate with
`.venv/bin/python scripts/oodgrid_report.py`.

**The 2026-07-14 corrections to this note:**
- The grid gained a **C4** cell (OOD start / ID goal), making it a full 2×2. With C4 the two
  shifts separate — and **the discriminating axis is the GOAL shift, not the start shift.**
  All ~20 bilinear cells have `dGOAL ≈ 0` (goal shift is FREE, often helps); both MLP cells
  lose **16–24 pts** to it. dSTART hurts *everyone* (−0.07 to −0.21). **Finding #2's "pure
  OOD-START interaction" framing is WRONG.**
- **`bilinear_sg` (ψ(s,g)) = 0.692 / 0.284 / 0.000.** Not seed noise — a late-training
  **critic divergence** (value_loss 0.15→1–5; `q_mean` crosses *positive*, impossible for a
  ≤0-bounded GC-IQL value). All 3 seeds are healthy at 200k (mean 0.615) and then 2/3 blow up.
  The one stable seed lands at **0.69 ≈ MLP's 0.717**. ⇒ the low-rank **goal bottleneck** (not
  the inner-product head) is what buys the bilinear win, AND it is what keeps the IQL backup
  contractive. Same structure explains both. (Performance claim is n=1; the stability claim is solid.)
- **Never read `dGOAL`/`dSTART` on a torch FB row**: the report takes C1/C4 from `zB` and
  C2/C3 from `zR`, so those deltas are cross-route artifacts (e.g. `fb_bridge_idB_dnorm`
  "dGOAL = −0.897" is a change of z-route, not of goal). Only jax rows decompose.

## The grid (what C1/C2/C3 mean)

All three cells share **byte-identical start states and goals** (`analysis/oodgrid/cells.npz`).

- **C1** dataset start (on-manifold) + hindsight goal — "can it do the task at all?"
- **C2** dataset start (on-manifold) + EVAL task goal — **C1→C2 = the GOAL shift**
- **C3** `env.reset()` start (OOD) + EVAL task goal — **C2→C3 = the START shift** (= the real eval)

Protocol matched to the env's own eval so **C3 reproduces published success**:
H=200; success = `‖cube − target‖ ≤ 0.04 m` (`cube_env.py:715`, read from source).
Validated: MLP 0.747 (pub 0.717), bilinear-d50 0.94 (pub 0.904), FB 0.400 (pub 0.42).

## THE TWO FINDINGS (do not conflate them — this is the session's main lesson)

**1. FB's failure is IN-DISTRIBUTION, not a generalization failure.**
On C1 — dataset starts, hindsight goals, everything on-manifold — FB reaches the goal
**0.43** while both GCIQL agents get **0.86–0.99**. FB **grasps the cube essentially always
(grasped = 1.000)** and then cannot steer it to the goal. With its deployed reward-inference
z it goes C2 **0.450** → C3 **0.400**: only ~5 points lost to the start shift, against a
~45-point deficit against GCIQL *everywhere*. **This keeps the grounding/representation
story alive — it is the story the paper is about.**

**2. bilinear-vs-MLP inside GCIQL is a pure OOD-START interaction, nothing to do with
manipulation.** The ordering FLIPS (3 seeds, replicated):

| | C1 (on-manifold) | C3 (real eval) |
|---|---|---|
| GCIQL-**MLP** | **0.97 / 0.98 / 0.99** | 0.81 / 0.75 / 0.68 |
| GCIQL-**bilinear-d50** | 0.84 / 0.87 | **0.94 / 0.94** |

From on-manifold starts the MLP critic is the BETTER one (also better rank-to-goal 0.905 vs
0.886, better transport 0.997 vs 0.917). It only collapses when the episode *starts* off the
data manifold. `env.reset()` states are **2.4× further from the data** than a held-out dataset
state (median NN distance 2.000 vs 0.818; 13.7% beyond the dataset p99); `initialize_arm()`
samples a random effector pose the play policy never produces.

**I do NOT have a mechanism for #2.** The obvious one (MLP critic extrapolates off-manifold,
expressive flow actor chases it) is **REFUTED**: action-ascent gain is flat IN→OOD
(MLP 0.21→0.18, bilinear 0.33→0.33).

## Phase diagnostic (3 seeds × 100 eps, `analysis/phasecmp/`)

Reproduced each anchor (FB 0.390, bilinear 0.927, MLP 0.710). `scripts/phase_probe_gciql.py`
+ `scripts/probes/phase_probe.py`, aggregated by the scratchpad `agg_phase.py`.

| | success | reach fail | grasp fail | **DROP** | steer fail |
|---|---|---|---|---|---|
| FB | 0.390 | **0.000** | 0.107 | **0.233** | 0.270 |
| bilinear | 0.927 | 0.000 | 0.057 | 0.003 | 0.013 |
| MLP | 0.710 | 0.000 | 0.113 | 0.087 | 0.090 |

`P(reach) = 1.000` for ALL THREE. `P(secure|reach)` differs by ≤5.6 pts. **The entire spread
is in `P(success | secure)`: FB 0.437, MLP 0.801, bilinear 0.982.** Approach is solved by
everyone; the failures are all post-grasp. DROP = secured a lift then the cube ended back on
the table (split via `final_cube_lift`).

## New code (UNCOMMITTED, on gciql-fb-bridge)

- `tools/wandb_mode_shim/gciql_flow.py`: **`GCStatePsiValue`** + `value_arch='bilinear_sg'`
  (Q=φ(s,a)·ψ(s,g)/√d — keeps the inner-product head, destroys the low-rank GOAL bottleneck).
  Separate branch on purpose: the `bilinear` path stays byte-identical so every checkpoint
  still loads. Asserts loudly against `fb_forward_z` / `actor_cond=z` / measure/SF cells
  (a state-dependent ψ has no zero-shot z).
- `tests/test_gciql_psi_sg.py` — 4 tests, green; 18 existing gciql_flow tests still green.
- `configs/gciql/cube_single_state_gciql_flow_bilinear_sg_d50.yaml`, `scripts/launch_bilinear_sg.sh`
- `scripts/oodgrid_{manifest,run_jax,run_torch,report}.py`

## MISTAKES MADE THIS SESSION — do not repeat (all were caught by checks)

1. **Cross-harness comparison.** I claimed MLP collapses 0.97→0.71; matched-harness it is
   0.890→0.815. Never compare numbers from two protocols.
2. **Silent seed drop.** Seed dirs are named inconsistently (`sd000_<ts>` vs
   `..._anneal_sgphi_s0`); keying on `name.split('_')[0]` collapsed all 3 anneal seeds to one
   and hid **108 run-dirs**. Key on the full basename.
3. **Latency-bound rollouts.** Unbatched per-env `sample_actions` = 67 min wall for 4 min CPU.
   Batch: N envs in lockstep, ONE model call per timestep (43 s).
4. **`pkill -f <pattern>` self-kills** when the pattern appears in your own command line.
   Use `pgrep -f 'goal[_]swap' | xargs kill`.
5. **FB z-route is a huge confound**: on C3, `z=project_z(B(g))` → 0.050 vs reward-inference
   → 0.400. Always run/report both.
6. Four of my own mechanisms died on measurement (low-rank spectral bias; the "24% value
   gap"; "B is arm-dominated"; "the low-pass filter kills the arm"). Measure before theorising.

## Facts measured this session (dataset-only, cube-single-play)

- obs is **28-dim**. `obs[18]` = `gripper_contact` = `clip(‖cfrc_ext‖/50,0,1)` — a **continuous
  contact force**, NOT a binary grasp flag (verified `manipspace_env.py:438-440`). De facto
  binary in this data (53.7% ≈0, 44.4% ≈1). Fires at the cube: effector within 1 cm, 99.6%.
- **The contact bit is ~fully redundant with the rest of the state** (AUC 1.000 from the full
  state; 0.928 from effector-cube distance alone; 1.000 from `obs[17]` gripper opening).
  ⇒ the programme's "interaction-variable probe" (FB 0.75 vs scalar-TD 0.91-0.96) is measuring
  **representation fidelity in general**, not a special interaction capability. **This
  challenges the `CLAUDE.md` thesis wording ("under-resolves the interaction variable").**
- Re-grasp is cheap: median **23 steps**, 95.6% recover ⇒ `E[γ^T|free] = 0.760` (24% value cost).
  But **conditional on geometry the held-vs-free value gap is only 2.1%** — the 24% is
  geometry, not contact.
- Occupancy (ψ) variance share: cube task-xyz **7.1%** vs antmaze task-xy **23.5%**; both
  domains' ψ is dominated by a slow task-IRRELEVANT orientation variable (cube 53%, ant 51%).

## Next steps

1. Collect `scripts/oodgrid_report.py` → does any other group show the bilinear crossover
   (`dSTART > 0`)? Which cells are most start-fragile?
2. Collect `bilsg` @500k vs bilinear-d50 **0.904** and MLP **0.717**. If bilinear_sg ≈ MLP,
   the low-rank GOAL bottleneck (not the inner-product head) is what buys OOD robustness.
3. Still open: the **mechanism** for the bilinear-vs-MLP OOD interaction.
4. Still open: `bilinear + DDPGBC = 0.51 < MLP + DDPGBC = 0.70` — unexplained by anything.
5. FB C1 was run with `z=B(g)`; a fully generous FB wants C1 re-run with reward-inference z
   (not defined for hindsight goals — needs a task-shaped goal set).

Related: [[mdose-campaign-20260708]], [[anneal-curriculum-campaign-20260710]],
[[bridge-comparison-both-flow-actors]], [[factored-fb-test-env]], [[always-run-in-tmux]]
