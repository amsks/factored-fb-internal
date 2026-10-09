---
name: handoff-20260721-campaigns-queue-dashboard
description: "2026-07-21 handoff — spec06 + ogbench-sota code DONE, live campaigns + concurrency queue + project dashboard; config gotchas + calibrations"
metadata: 
  node_type: memory
  type: project
  originSessionId: b7116b8e-2489-4eeb-9d03-aeef73b33fdb
  modified: 2026-07-21T03:41:06.871Z
---

2026-07-21 bridge-track handoff. Two big code deliverables landed + campaigns running + a self-managing GPU queue + a project-wide dashboard.

## Code delivered (all coded + tested, default-off, param-tree-preserving)
- **spec06 (rescuing-fb)** — 8 items, see [[spec06-implemented-20260720]]. gciql_flow.py knobs: task_prior, measure_first_passage/expectile/testpoint_dist. Tests: test_gciql_rgfb_v1.py, test_gciql_measure_axes.py.
- **ogbench-sota** — 7 items (`PAPER/ogbench-sota/implementations.md`). NEW value_archs in gciql_flow.py: `latefusion` (item1), `bilinear_sg_bounded` (item5), `bilinear_split` (item7); knob `phi_norm_cap` (item4b). Tests: `tests/test_gciql_sota_archs.py` (7 pass). CPU probe scripts (fork-written, compile-verified): `scripts/rescuing_fb/t5c_srank.py` (item3), `t9_advantage_rank.py` (item6), extended `t3_offpeak_multienv.py` (item4a, gated by env `T3_PEAK_GEOMETRY`). Configs `configs/gciql/cube_single_state_{latefusion_d50,mlp_pmatched,phinormcap_d50_ddpgbc,phinormcap_d50_flow,bilinear_sg_bounded_d50}.yaml` + `scene_state_bilinear_split_{flow,ddpgbc}.yaml`. Launcher `launch_sota_impl.sh`.

## ⚠ CONFIG GOTCHA (cost us v1d50 silently dying)
`run_gciql.py` emits `--agent.KEY=VALUE`; ml_collections flag-override **rejects `list`** (accepts float/bool/int/str/**tuple**). So a list-valued agent_override (e.g. `task_prior_probs: [..]`, `value_split_dims: [..]`) FAILS at parse. Fixes: rely on the in-code default (task_prior_probs default is 0.6/0.2/0.2 → just omit the override), OR make the knob a **string** (value_split_dims is now a comma-string "19,9,4,4,2,2", parsed in create()), OR tuple-string for genuine tuple fields (`value_hidden_dims: "(737, 737, 737)"` — value_hidden_dims IS a tuple, so this works).

## Calibrations (measured 2026-07-21)
- **bounded** psi_bound_scale = **27.0** (p99 of trained GCIQL-State-Flow-Bilinear-d50 sd000 psi(g) norm; mean 22.9 max 28.6).
- **scene-split** value_split_dims = **"19,9,4,4,2,2"** (scene obs 40 = proprio19+cube9+2×button4+drawer2+window2; num_cubes=1 num_buttons=2 num_button_states=2).

## Live now (tmux)
- `s06gpu` — my spec06 wave (d512v0 best exact-z ~0.84 climbing; measure cells at floor ~0.11). eval_on_cpu=1 @50k caused a CPU-eval contention meltdown (20 s/it, 24 concurrent) — recovered as runs finished.
- `s06corner` + `s06cellw` — OTHER SESSION: GCIQLCorner + CellW0 = the "successor-measure-that-works" arm (measure_offdiag_weight knob they added to my item6 code; at w→0, M=1+(1-γ)Q ≡ bilinear GC-IQL). Don't touch their dirty files.
- `queue` — `scripts/rescuing_fb/queue_runner.sh`: 24 backfill jobs (v1d50×3, latefusion×3, pmatched×3, phinormcap×6, bounded×3, scenesplit×6). Gates on box-wide procs < MAX=12, least-loaded GPU (0/2/3/4/7, never 1/5/6), **eval_interval=100000** (the contention fix), 45s stagger. Output → `~/exp_queue`.

## Dashboard (shared doc)
Artifact **https://claude.ai/code/artifact/7f623adf-fd85-46a8-afd5-c16915daf56f** (📡). Generator `scripts/rescuing_fb/spec06_dashboard.py` (pure stdlib): auto-discovers ALL groups under exp_spec06/exp_sota_impl/exp_queue, reads results/{summary_agents,bridge_grafts}.csv into interactive (sort/filter/search) color-banded tables, + live-run section. Monitoring loop republishes it every ~20 min. Bands: <0.40 weak / 0.40-0.70 mid / ≥0.70 strong (cube: FB floor 0.42, GCIQL-bilinear ceiling 0.90).

## Results so far (partial)
- **item1 GPI**: GPI-over-goal-candidates FIXES two-goal mixtures at d512 (0.752, GPI_FIXES_COVERAGE); d50 partial (0.536). The one method WIN.
- **item3 rerank**: BELOW_ANCHOR (0.76) — selection HURTS (BC-flow→0, one-shot declines with N). Off-support-peak, as predicted.
- **measure cells (5/6/7)**: all at the ~0.11 floor → trending axis-exonerated.
- **item3 t5c smoke**: bilinear φ srank 46 (>28, matches prediction); MLP penultimate 256.

## PENDING / next
- User EXPLICITLY wants to **build a successor measure that WORKS on cube** (brainstorming was started then deferred). The other session's measure_offdiag_weight / M=1+(1-γ)Q direction is the live lead — resume that design with them.
- After campaigns: `scripts/compile_results_csvs.py --refresh` to fold new groups into the CSVs (dashboard benchmark tables read the committed CSVs).
- Nothing committed (convention: code/tests/docs only, artifacts stay untracked).
