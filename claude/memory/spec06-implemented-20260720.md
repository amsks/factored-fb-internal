---
name: spec06-implemented-20260720
description: "Spec 06: coded+tested+reviewed(5 fixes). PROGRAM RE-CENTERED: success = a SUCCESSOR MEASURE that works on cube, nothing else. Rung1 (4 measure cells, tmux s06gpu) + rung2 FULL-CORNER (fp+expectile+hindsight, tmux s06corner) RUNNING; RG-FB/GPI demoted to anchors"
metadata: 
  node_type: memory
  type: project
  originSessionId: b7116b8e-2489-4eeb-9d03-aeef73b33fdb
  modified: 2026-07-20T23:31:42.413Z
---

2026-07-20 (bridge track): all 8 items of `PAPER/rescuing-fb/specs/06-phenomenon-implementations.md`
are IMPLEMENTED as ready-to-launch code + tests. NO GPU campaign launched yet (user launches per the
spec's ordering + early-stop rule). Knobs are default-off ⇒ v0/anchors bit-identical, param-tree-preserving,
assert-lattice; test suites green + no regression on the 20-test measure set.

**gciql_flow.py knobs** (`tools/wandb_mode_shim/gciql_flow.py`):
- item4 v1 task-prior: `task_prior` {dirac,mix}, `task_prior_probs`, `task_prior_mix_k`; methods
  `_sample_tasks` + `_taskprior_vc_loss`; `actor_loss(..., task_z=)`; `total_loss` mix dispatch.
- item5 `measure_first_passage` (hit into `_measure_terms(hit=)`); item6 `measure_expectile`
  (+`measure_v_phi` tower, `measure_expectile_tau`); item7 `measure_testpoint_dist='hindsight'`.
  All in `measure_loss` critic rho/full branch. Cube xyz = obs[19:22], xyz_scaler=10 ⇒ ε=4cm=0.4 obs units.

**Tests**: `tests/test_gciql_rgfb_v1.py` (4), `tests/test_gciql_measure_axes.py` (5). Run via
`.venv/bin/python -m pytest` (shells to `/dev/shm/.venv-jax`). obs_dim=29 in these tests (cube slice).

**Eval scripts** (CPU, jax venv): item1 `scripts/rescuing_fb/t6_gpi_eval.py` (GPI over goal candidates);
item3 `scripts/rescuing_fb/rgfb_rerank_eval.py` (selection/rerank at inferred z, N-sweep); item8
`scripts/rescuing_fb/measure_readout_head.py` (`--backend jax` RG-FB head / `--backend torch` plain-FB
native F·B / `--mode compare`; asserts checkpoint hash unchanged).

**Configs** (`configs/gciql/`): `cube_single_state_rgfb_d512.yaml` (item2);
`..._rgfb_v1_taskprior_d{50,512}.yaml` (item4); `..._measure_{firstpassage,expectile_fp,expectile,hindsighttp}_d50.yaml`
(items 5/6/7). All build + `update()` clean on real cube dims (obs_dim=28).

**Launcher**: `scripts/rescuing_fb/launch_spec06.sh` (DRY_RUN=1 default; ARMS="d512v0 v1d50 v1d512 measure").
Groups registered in `scripts/results_map.py`. After any campaign: refresh CSVs + apply the pre-registered
read verbatim to DECISIONS.md (reads listed in the DECISIONS 2026-07-20 spec-06 entry). See [[t7-t8-campaign-20260720]].

**REVIEW PASS 2026-07-20 (later session): 5 findings, ALL FIXED, 33/33 adjacent tests pass. LAUNCHABLE.**
1. CRITICAL (item 4): z was fully sg'd ⇒ ψ NEVER trained under task_prior='mix' (= frozen-random-ψ
   0.061). Root cause: a wrong line in spec 06 itself ("no grad into ψ through z") — spec corrected.
   Fix: z carries ψ-grads for Dirac/mixture arms; sg only on z_actor, p3 reward features, mixture
   renorm SCALE, and TD targets (z_t inside `_taskprior_vc_loss`).
2. HIGH (item 4): mixture reward was {0,1} vs Dirac's gc_negative {−1,0} → conflicting targets at
   near-identical z. Fix: r_mix = hit − 1 (mask = −r).
3. HIGH (item 5): first-passage kept FB's LINEAR diagonal attraction → unbounded pull, fixed point
   ≠ E[γ^T]. Fix: diagonal SQUARED when hit is supplied (`_measure_terms`).
4. MED (item 4): Dirac-arm actor consumed value_goals (30% random/20% current) — D8 split broken.
   Fix: `_sample_tasks` returns (z, z_actor, ...); z_actor Dirac rows = ψ(actor_goals).
5. MED (item 7): diagonal positives included 30% random goals. Fix: hindsight-non-fp diagonal from
   ψ(actor_goals); NEW assert refuses unmasked expectile+hindsight (no coherent reward).
Note `_actor_z` default = the VALUE net's ψ (ψ_V is THE task basis under 'mix'; critic's ψ_c is
dead params there — inference/eval consistent since eval_rgfb_zeroshot uses `_actor_z` too).

**RE-CENTER (2026-07-20 late, user decision — binding):** the ONLY success for rescuing-fb =
a successor measure in the GCIQL band on cube. Measure-native def (spec 06): matrix M(s,a,s⁺)
over a test-point field, actor consumes only its readouts, zero-shot z entry. First-passage
QUALIFIES; RG-FB does not. GPI helped but is OUT OF SCOPE. Measure ladder: rung1 = 4 cells
(s06gpu, launched 22:18 with the fixes); rung2 = FULL-CORNER fp+expectile+hindsight
(`cube_single_state_measure_gciqlcorner_d50.yaml`, group ...Measure-GCIQLCorner-d50, tmux
`s06corner`, GPUs 4/7/0 packed); rung3 short-γ two-timescale head (γ≈0.8 + blend, NOT coded);
rung4 mode-indexed φ(obs[18]) (NOT coded); rung5 torch FB-FP port. Bars: ≥0.85 headline /
0.5 real / ≤0.2 dead→next rung. Early rung1 reads @50–200k all 0.0–0.10 (too early to call).
GPU note: 1/5/6 stay fenced (other track) even when idle.

**CELL W (2026-07-21, THE WIN):** existence theorem — `δ_M = (1−γ)·δ_Q` identically under
`M = 1+(1−γ)Q` ⇒ first-passage SM with hindsight-PAIRED columns + expectile backup IS
bilinear GC-IQL (0.904) reparameterized. Corner cell's gap: uniform B×B ⇒ ~99.6%
unpaired-random per-row weighting. Knobs `measure_offdiag_weight` (0=exact equiv / 0.3=
GC-IQL-like mix / 1=corner) + `measure_expectile_source='target_min'`. **cellw0 RUNNING**
(tmux `s06cellw`, GPUs 4/7/0); **cellw03 QUEUED — launch when GPUs free**
(`ARMS="cellw03" bash scripts/rescuing_fb/launch_spec06.sh`). Renewal decomposition
M_occ ≈ M_fp·R(B) (R action-independent) = the occupancy completion + one-line failure
explanation. ⚠ other session is adding ogbench-sota arch cells (latefusion/bounded-ψ) to
the SAME gciql_flow.py — re-run both suites after any merge; my regions: measure branch +
task-prior + spec06 asserts + get_config measure/task knobs.

**COLLECTED 2026-07-21 (@500k, 3 seeds, CSVs refreshed):** FP-only **0.225** (sustained, 3×
ceiling — AXIS C IS REAL, first measure cell above the ceiling); HindsightTP 0.095 (axis A
null); Expectile 0.103 / Expectile-FP 0.039 / Corner 0.035 / **CellW0 0.071 — CONFOUNDED:
≈ ortho-1000 fingerprint (0.072)**. Mechanism: δ_M=(1−γ)δ_Q ⇒ M-units TD ~1e-4 scale vs
ortho λ=1 O(1) ⇒ effective λ~1e3–1e4. Theorem NOT falsified. **RERUNS RUNNING (tmux
`s06fix`): CellW0-NoOrtho + FirstPassage-NoOrtho** (ortho=0, identified via fp reward).
RG-FB anchors (parked direction): d512 zero-shot 0.828 ≈ d50 (d not the gap); rerank
BELOW_ANCHOR (N=1 oneshot 0.867 best; declines with N); GPI d512 mixtures 0.260→0.752
(fixes coverage), d50 0.536 partial. NEXT: expectile-fp/cellw03 at ortho=0 after s06fix.

**GPU rebalance 2026-07-21 (+ fence retired):** the "1/5/6 = other track, never touch" rule
is STALE — sota track demonstrably runs on 0/2/3/4/7; 1/5/6 sat idle for days. New etiquette
(also in launch_spec06.sh header): no fixed ownership, place by `nvidia-smi
--query-compute-apps` emptiest-first, ≤3 procs/GPU for state cells, never touch other
sessions' pids. Moved the 6 noortho reruns to 1/5/6 (2/GPU, ≤16 min training lost — old
copies died via tmux SIGHUP when the s06fix session was killed; lesson: launcher children
are NOT disowned, killing the tmux session kills the jobs). Occupancy-IQL
(GCIQL-State-Flow-Occupancy-d50, 3 seeds — implemented+launched by the other session) now
shares only with 1–2 sota jobs per GPU. Also spotted running: CellW050-d50 (w=0.5 mid-dial,
other session) + the FIXED v1-taskprior d50.
