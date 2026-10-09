---
name: handoff-hiql-bilinear-20260707
description: "2026-07-07 session handoff: HIQL-bilinear mechanism — spec+plan written, all knob code COMMITTED (4 commits), probe results in, antmaze 8s@1M RUNNING (amhiql), 12-run hiqlmech campaign READY TO LAUNCH pending smoke + free GPUs; E2 guardrail PASSED"
metadata: 
  node_type: memory
  type: project
  originSessionId: 061c5b6d-fefa-4082-ada9-d56198028c2d
---

**Session 2026-07-07 (day). Question: why does bilinear buy GCIQL +26 / GCIVL +18–30 on cube
but HIQL only +4 native and −12 flow-both?** Hypotheses + evidence: [[hiql-bilinear-brainstorm]].
Spec: `docs/superpowers/specs/2026-07-07-hiql-bilinear-mechanism-design.md` (M6 addendum to
the ogbench-sota ICLR spec). Plan (EXECUTED): `docs/superpowers/plans/2026-07-07-hiql-bilinear-mechanism.md`.

## Done this session

- **goal_rep probe built + run** (`scripts/probe_hiql_goalrep.py`, results
  `analysis/hiql_goalrep/*.json`, 11 ckpts @500k): NO rank collapse (PR rises 3.4→4.0–4.3);
  bilinear warps GLOBAL geometry (R² goal-xyz random 0.14→0.08–0.10, geomSpearman
  0.115→0.059–0.087, worst in worst cell) while subgoal-scale is spared (~0.30); ALL cells
  under-resolve the cube (best R²=0.31 sub/0.14 rnd) = H3 signature. MLP reps identical
  across actor types (sanity ✓).
- **Published anchors** (OGBench paper, protocol = 1M, 8 seeds, avg last-3 evals, 50 eps —
  resolves the parent spec's open TODO): HIQL cube 15±3 / scene 38±3 / antmaze-medium 96±1;
  GCIQL 68/51/71; GCIVL 53/42/72. Our sg10 MLP ~0.17 reproduces published cube HIQL.
- **Code, all TDD green (9 tests), 4 commits on `gciql-fb-bridge`:**
  - `bde8be2` bilinear_form knob ('rep'|'goal' = option-A|Schaul) in vendored hiql.py +
    hiql_flow value defs; gradient-routing test proves 'goal' cuts value→goal_rep grad.
  - `045e0da` value_split='high_bilinear': separate TD-trained bilinear head (stop_rep_grad)
    read ONLY by the high-actor advantage; MLP V keeps TD/low path + trains goal_rep;
    _ivl_value_loss refactor; target-update wiring.
  - `d7e24a8` same mirrored in hiql_flow.py (flow-both cell).
  - `0f85284` probe → parametric --dataset-name/--goal-slice/--action-dim/--subgoal-horizon
    (antmaze xy = obs[0:2], VERIFIED range [-1.7,21.8]; action_dim 8; sg 25); cube path
    byte-identical regression ✓.
- **E2 guardrail PASSED** (see [[e2-measure-mix-campaign]]): 15/15 q_mean ∈ [−21,−8]
  full-history; flat/coexist reading certified.
- SF-feature-TD campaigns confirmed ALL FINISHED (flow 0.09–0.16, ddpgbc ≤ flow, ortho1k
  dead ~0.00) — nothing running; thread superseded by E1a/E2.

## In flight / next actions (in order)

1. ~~Task-4 smoke + commit~~ **DONE** — smoke killed mid-final-eval on user request (all 4
   configs had built + trained 2k clean, logs error-free); configs+launcher committed
   `a828ee8`.
2. ~~Launch hiqlmech~~ **DONE 2026-07-07 ~12:10** — 12 runs on GPUs 0/1/2/4 (3/GPU, shares
   with the 3 scene runs), tmux `hiqlmech`, logs `/dev/shm/hiqlmech_logs/`. Collect @500k
   (~3–5h packed): decision rules in spec §Experiments (E-B rep50 / E-C schaul vs
   native-bilinear 0.21 & MLP 0.17; E-D flow-both-highbil vs flow-MLP 0.26 & flow-bilinear
   0.14).
3. **Collect amhiql @1M** (antmaze HIQL MLP vs bilinear-d50, 8 seeds, groups
   HIQL-AntMaze-Medium-{MLP,Bilinear-d50}): use avg of last-3 evals (800k/900k/1M) to match
   OGBench. Decision rules in the spec §Experiments (E-A): bilinear ≥ MLP → H3 hid the win;
   bilinear ≤ MLP → H1'/H2 domain-general. Then **E-E**: run the probe on those ckpts
   (--dataset-name antmaze-medium-navigate-v0 --action-dim 8 --subgoal-horizon 25
   --checkpoint-step 1000000).
4. **E2 prize cross-eval** when the last ~8 e2measure runs hit 500k:
   eval_zeroshot_inference.py (v0/v1) per cell ψ ([[e2-measure-mix-campaign]]).
5. Spec decision rules for E-B/E-C/E-D once hiqlmech lands: E-C ≈ GCIVL-size win → H2;
   E-D flow ≥ 0.26 → H1'; E-B moves nothing → 10-dim bottleneck exonerated.

## v4 both-bilinear follow-up (2026-07-07 evening)

All 12 hiqlmech runs DONE @500k (schaul 32.6/highbil 34.3 latest-mix/flow-highbil 41.3@400k/
rep50 24.3@400k — final @500k rows still landing in eval.csv for some). NEW v4 cell launched:
factorize BOTH levels = value_arch=bilinear + bilinear_form=goal + value_split=high_bilinear +
low_actor_rep_grad=true (knobs COMPOSE, no new agent code; 2 composition tests added to
test_hiql_bilinear_knobs.py, 7/7 green; flow low-actor rep-grad routing verified in code).
Committed+pushed f752725. RUNNING: tmux `bothbil`, 6 runs (native + flow-both × 3 seeds,
1/GPU on 0-5), logs /dev/shm/bothbil_logs, groups HIQL-State-{Native,Flow-both}-BothBil-d50-sg10.
Question: does the low level also want bilinear once η's trainer is safe? Compare v2 32.8 /
v3 34.3 / flow-v3 41.3. Also: ogbench-sota.tex CONDENSED to 3-section research note
(math/eval/mechanism, ζ notation, citations, no em-dashes) + PDF force-added, all pushed
(52c0edd, cc26df9).

## Antmaze bc03 reruns + coverage (2026-07-07 late, tmux `ambc03`)

td_jepa HP audit finding: scene HPs were CORRECT (ddpgbc α=1.0 both arms = tuned; flow
bc_coeff=3.0 = exactly td_jepa's scene value → scene negative is REAL). But antmaze flow
cells ran cube's bc_coeff=3.0 where td_jepa's antmaze sweep uses **0.3** (10× weaker BC,
mirrors OGBench α: antmaze 0.3 vs cube/scene 1.0) → "flow collapses on antmaze" (23.9/33.3)
possibly over-imitation artifact. NOTE: bc_coeff exists ONLY in gciql_flow; gcivl_flow/
hiql_flow are AWR-flow-BC (knob = alpha, and antmaze-tuned == defaults: GCIVL 10, HIQL 3/3
sg25). RUNNING 12 runs @500k (tmux ambc03, logs /dev/shm/ambc03_logs, commit 0a275fd):
GCIQL-AntMaze-Medium-Flow-{Bilinear-d50,MLP}-bc03 (the fix + control, vs old 23.9/33.3 and
ddpgbc-bilinear 83.7) + GCIVL-AntMaze-Medium-Flow-Bilinear-d50 (never run; cube 79.2) +
HIQL-AntMaze-Medium-Flow-both-HighBil-d50 (v3 on antmaze; cube 41.3, published HIQL 96).
If flow-bc03 recovers, the paper's actor-flip claim needs rewriting (BC-strength, not
actor family, is the domain knob).

## awr_flow readout-matrix cell (2026-07-08, tmux `awrflow`, DEFER-GATED)

New `actor_kind='awr_flow'` in gciql_flow.py (commit 16c0355, 3 TDD tests green):
IQL's canonical AWR extraction on a GOAL-CONDITIONED flow-BC vf — weight
exp(awr_alpha·(minQ−V)), awr_alpha=10 (OGBench GCIVL temp), Euler-rollout sampling,
NO one-step actor/distillation (asserts state-only, no z-cond). Completes readout
matrix {AWR,Q-max}×{GCIQL,GCIVL,HIQL}. 6 runs (cube+antmaze × bilinear-d50 × 3s
@500k) in tmux `awrflow` — launcher DEFERS until run_gciql procs ≤18 (box was
saturated: bothbil+ambc03+the OTHER session's mdose = 32 procs). Logs
/dev/shm/awrflow_logs. DECISION: antmaze recovery ⇒ Q-max-distillation readout is
the antmaze killer (not GCIQL critic); cube ≪ 90.4 ⇒ cube needs Q-max greediness.
Context: ambc03 partial results = GCIVL-flow antmaze 0.84/0.84 (pub 72), HIQL-v3-flow
antmaze 0.94 (pub 96), GCIQL flow-MLP bc03 0.40/0.31 (bc_coeff NOT the fix); ALL
bilinear-HIQL cube cells peak @100k then decay 10-20pts (v4 flow peaks 0.62-0.64!);
v3 native final @500k = 30.3±8.4 (not 34.3), flow-v3 = 41.0±11.0.

## Gotchas rediscovered

- OGBench runs do a step-0 eval AND a post-training eval (51 rollouts each, ~19s/rollout
  when 4-packed) — a "2k-step smoke" is ~30 min wall-clock, dominated by evals.
- flow configs need `agent_file: tools/wandb_mode_shim/hiql_flow.py` (full path, not bare).
- hiql_flow.create builds its OWN network dict — any hiql.py network-topology knob must be
  mirrored there or flow cells silently run without it (tests catch this now).
- docs/superpowers/ is GITIGNORED in this repo — specs/plans stay local-only.
