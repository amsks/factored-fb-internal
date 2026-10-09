---
name: smb-eval-suite-20260726
description: "SMB benchmark + full measure DONE+gated; campaign v2 (multi-scale + actor-z arms) LAUNCHED 2026-07-27 tmux s10v2"
metadata: 
  node_type: memory
  type: project
  originSessionId: 90c8189d-90e2-4163-a00e-c2b245ce160f
  modified: 2026-07-27T05:53:05.277Z
---

Bridge track, 2026-07-26 (follows [[spec10-full-measure-brainstorm-20260726]]). User directive:
build ONE frozen eval separating GC-IQL < two-phase < spec-10 modification, run it on existing
ckpts, then code the modification to close exactly the exposed gap.

**Spec 11 = `PAPER/rescuing-fb/specs/11-smb-eval-suite.md`** (uncommitted). Two wings:
- Wing A control tiers T0–T7; the new discriminators: **T3 occhoice** (stable on-table vs
  transient elevated region, weights between the reaching ratio and occupancy ratio — a
  reaching value picks the wrong region BY CONSTRUCTION; this is where 2-phase beats GC-IQL),
  T4 union3, **T5 corridor reach-avoid** (avoid ball ON the transport corridor so the penalty
  binds; old avoid-null was task design), T6 linear, T7 dwell metric. Cross-cutting: true-z vs
  inferred-z axis; achieved-discounted-reward primary metric; % of GC-IQL anchor normalization.
- **Wing B measure calibration** (B1 diag / B2 arbitrary-query / B3 ∫r dM vs MC of own
  rollouts) — lit survey (12 papers scraped 2026-07-26) confirmed ZERO prior quantitative
  measure-calibration metrics in the FB/SM literature; only partial precedents: Soft-FB's
  estimate-vs-true correlation + z-search (cite for spec10 route), VC-FB logQ-vs-IQM plot,
  FB-CPR exp-tilted z, TD-JEPA's +1-shift fragility quirk. B2 is inexpressible for diagonal
  ckpts (that's the measurement); calibrated B2/B3 = the spec-10 modification's exact target.

**Tabular playpen DONE**: `scripts/rescuing_fb/tabular_playpen.py` +
`analysis/rescuing-fb/tabular_playpen.json`; tex section `sec:playpen` added to
`two-phase-successor-measure-note.tex` (compiles clean, tectonic). 7×7 grid + pit; column =
cube-position analog, row = mode (conveyor = carried, bowl = at-rest). Numbers: renewal
identity exact 1.4e-9; region-level R-only Spearman 0.96 vs reaching 0.58 (mirrors cube
0.60-vs-0.06); occhoice miniature: arrival ranking picks conveyor, occupancy picks bowl, truth
96.8 vs 47.9; slice-sum 144.7 > best-single 96.8; rank-8 SGD: region regression 0.05–0.09 rel
err vs density-TD 0.38 (ortho makes it WORSE 0.53); density does NOT diverge without actor
loop (honest caveat kept in tex). Gotcha: arrival-mode CV is structurally ~0 in small grids
(re-entry cheap at γ=0.99) — reframed honestly, don't chase it.

**GRID COLLECTED 2026-07-26 ~11:00** (tmux `s11smb`, 36/36 outputs in
`analysis/rescuing-fb/smb/`, all exit 0; DECISIONS.md entry written). Harness =
smb_common/smb_tasks/eval_smb/eval_smb_measure + run_smb_grid.sh, 36 tests green,
artifact byte-identical on --check. Mid-flight fix: Wing-B envs `terminate_at_goal=False`
(stock termination truncated gt occupancy; Wing A deliberately keeps stock env).

**RESULTS (first pass, see DECISIONS for numbers):**
- T3: anchor at chance (discriminator WORKS — reaching value picks wrong region) but
  phase-2 diag also ~chance; ZX rows best 0.62–0.64 chose-correct. R under-resolved at s0
  vs 9× dataset gap ⇒ modification target. ⚠ oracle-z T3 arm is trivial (hands correct
  region) — ignore it.
- T5 RETRACTED after retention pull: ZX "deflection" = task destruction (succA collapses
  zx 0.89→0.24→0.06, visz 0.80→0.03 with λ; z_r corrupted by negative weights). Only
  vis_s0 shows a real trade-off (0.94→0.78 succA, timeB halved), unreplicated. And
  corridor_binds fires on only 1/5 tasks — widen the gate in v1.1. Verdict = pre-registered
  ✗ everywhere.
- Wing B: anchor units-broken + fp err≈1.0 as predicted; **visz = best-calibrated object**
  (B1 err ~0.5 ρ~0.35, B3 ρ 0.65–0.76); boxes ρ 0.07 + union additivity defect 0.96 —
  arbitrary sets fail exactly as pre-registered (spec-10 L7/L8 is the right target).
- B1 audit CLOSED — **REAL FINDING, headline-grade**: readout path verified correct
  (vis_s0 own-successor diag 73.5/96.5 = dwell scale; random-goal off-diag 2.4/−0.4).
  The diagonal ball-visit critic is occupancy-calibrated ONLY near its hindsight
  support; far-field (incl. reset states) collapses to ≈0/negative — a
  proximity-calibrated object, not a measure over (s,g). ZX's better far-field fits
  (random-z spread the support). Audit gotcha: my first pass compared permuted copies
  of the same goal set (readout returns [N,G]; take the diagonal).
- Artifacts: `smb/smb_matrix_v1.csv` (12 rows × 21 cols) + `smb_t5_retention.csv`.

**FULL MEASURE BUILT + GATED (same day, user 8h directive; commits 5f2e74d + 03d64de):**
`full_measure.py` = eval-layer m̂=M̃/N̂ over visz ckpts + SMB routes (--predictor density,
--z-inference cmoment, --z-select measure, --density-cap-q). Gate: **T6 linear RESOLVED**
(beats both controls, both seeds); additivity 0.96→0.19; **B3 slope 25×→0.4** (near the
0.5–2 band); **T5 destruction fixed by measure-select** (succA retained 0.77–0.83;
cmoment-only control still collapses); T3 needed winsorization — 1/N̂ tail overweights
thin-support regions, cap95 no-op, **cap80 → chose 0.58/0.65, G/oracle 0.74/0.93** (beats
no-density baseline both seeds), cap60 point pending at save. zx units-control behaved as
registered. Open: B2 boxes (ρ 0.05/0.31 — needs trained region columns, spec10 L7/L8);
chose-gate 0.7; genuine T5 avoidance. Full numbers in DECISIONS 2026-07-26 final entry.

**CAMPAIGN V2 LAUNCHED 2026-07-27** (tmux `s10v2`, `launch_spec10_v2.sh`, out
`/home/adityamohan/exp_spec10`, logs `/dev/shm/spec10_logs/`): 4 jobs, 300k warm-started
from visz@500k — Arm A `measure_eps_multiscale='0.2,0.4,0.8'` (msA_s0/s1, GPUs 0/3, group
...ZX-MultiEps-d50) targets B2 boxes + T3 R-resolution; Arm B `actor_z_greedy_frac=0.3`
(azB_s0/s1, GPUs 4/7, group ...ZX-ActorZ-d50) targets T6 margin + T5 avoidance. Knobs in
gciql_flow.py are additive, default-off bit-identical (test_gciql_campaign_v2.py 9/9 +
regressions 28/28) but **gciql_flow.py stays UNCOMMITTED** (mixed-track dirty state — user
must reconcile). New `eps_gain` module fresh-inits on warm-start (verified). Gates
pre-registered: ArmA B2 box ρ≥0.4 + T3 chose≥0.7, no T6/B3 regression; ArmB T6 margin ↑ +
T5 deflection WITH succA retention. After ckpts land: SMB gate reruns (Wing B density
fine-ε + Wing A t3/t5/t6, cmoment+msel+cap0.8). Note tex updated with §smb (compiles).
Agent smoke nugget: measure-select prefers transient on 4/5 T3 pairs pre-training —
independent repro of the thin-support finding.

**V2 COLLECTED 2026-07-27**: **azB (actor_z_greedy_frac=0.3) = best full-SM ckpt, ADOPTED**
— native 0.84/0.916, T3 chose 0.79/0.74 ✓, T6 ✓ both seeds, boxes 0.35/0.33 (best; gate
0.4 missed), B3 ρ 0.60 slope 0.4. msA (multi-scale) SEED-SPLIT (s1: T3 0.99-of-oracle,
chose 0.72 ✓; s0: chose 0.54 ✗ + native 0.64) and regressed T6 on s1 — do not adopt; its
box mechanism NEVER exercised (eval readout has no per-scale query — eps_gain only in the
training loss). NEXT: (1) wire per-scale readout + matching-ε N̂ into density predictor,
retest boxes on msA; (2) avoidance needs data support (play data has no detours) — likely
env/data-side, not another knob; (3) reconcile+commit gciql_flow.py (user); (4) tighten B3
slope toward 1 (self-normalized q_r candidate). Ckpts exp_spec10/.../{MultiEps,ActorZ}/.

**PHASE 2B COLLECTED 2026-07-27 — FULL MEASURE IS TRAINED**: nhat_head (regression on
hit-column means) trained in phase 2; nh_s0/s1 (azB+150k, group ...ActorZ-Nhat-d50, gate
at params_100000 — save_interval gotcha) reproduce the KDTree numbers within noise and
BEAT them on B3 slope (0.46/0.53 — s1 crosses the 0.5 band, first ever); T3 chose 0.70/0.79
✓, s1 G/or 1.00; native 0.808/0.928. Blemish: s1 trained-union additivity 0.71 (open).
Test-scaffold gotcha fixed: zx_cfg must set measure_coef=1.0 (get_config default 0.0
zeroes the measure loss in total_loss under measure_only). BEST CKPTS NOW: nh_s0/s1.
Uncommitted: gciql_flow.py + campaign tests + eval hooks — user reconciliation.

**TORCH FB PORT STARTED 2026-07-27** (user directive: build the trainable SM in the torch FB
stack). Recon done: template = agents/fb/isfb/agent.py subclass pattern (override update_fb
only, wire agent=fb_measure_flowbc in train.py + configs/agent yaml); n×n matrix native
(Ms=F@B.T, columns = batch next_obs via _bw_encoder Identity); density loss
agent.py:677-793 + fb_successor_terms :67-92 (linear diag) + ortho_cov :722-731 (launcher
uses ortho_coef=1000!); sample_mixed_z native :390-419; FlowBC actor greedy on <F,z> native
(= actor_z objective); NO freeze-B exists (implement via requires_grad_(False) +
backward_optimizer=None, code None-tolerant); ckpt = torch.save state_dict step_*.pt;
partial restore evals/analysis.py:18-49 strict=False. Cube obs[19:22], eps 0.4. Torch venv
= repo .venv; launch scripts/agents/fb/state.sh / train.py hydra domain=cube_single.
Implementation DELEGATED (agents/fb/measure/agent.py FBMeasureFlowBCAgent, knobs
measure_mode/hit_eps/ortho0/scale_norm/freeze_b/nhat_head/warm_start_ckpt + 7 tests +
smoke). Plan after green: phase-1 first_passage 500k → phase-2 visitation+freeze_b+nhat
warm-started; gate = the torch stack's OWN eval (FB-native z-inference protocol, 10k
relabel) vs torch FB baseline (~0.42-0.48 band, note torch band INVERTED vs jax per
[[handoff-torch-isfb-20260702]]); SMB torch adapter later.
