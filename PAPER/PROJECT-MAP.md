# Factored-FB — Working Project Map

_Living single-page orientation. Last refreshed **2026-07-07**. Consolidates
`MEMORY.md`, the per-project papers under `PAPER/`, and `docs/superpowers/specs/`.
Update this when a campaign finishes or the plan turns._

The repo is a **faithful port of Forward–Backward (FB) / FlowBC** to **OGBench**,
and the work has split into **three distinct projects**. They share substrate
(OGBench cube/antmaze/scene, GCIQL/FB/PSM/RLDP agents, wandb `amsks/factored-fb`)
but ask different questions. Numbers are cube-single-play **state**, `@500k`,
`success/overall` (FB torch) / `evaluation/overall_success` (GCIQL jax) unless noted.

| # | Project | Question | Verdict so far |
|---|---|---|---|
| ① | **RLBrew** | Do future-reachability reps (GC values, successor measures, contrastive, predictive) fail on manipulation, and where? | Yes — they under-resolve the interaction variable |
| ② | **OGBench-SOTA** | Does bilinear value factorization set a new OGBench SOTA? | Conditional yes (method/actor/domain-dependent) |
| ③ | **FB-GCIQL Bridge** | Why does the successor-**measure** objective break the *same* bilinear form that IQL runs at ~90? | The objective is the binding ingredient; every fix falls short |

---

## ① RLBrew — failure modes of future-reachability methods on manipulation

**Methods.** The *successor-measure* estimators **FB, PSM, RLDP, CRL** plus **GCIQL**
as the goal-conditioned **value** representative. Precise framing: GCIQL is *natively*
IQL value-TD (not a successor measure) — it enters ① as the GC-value baseline in the
phase diagnostic. Its successor-**measure** variant, and why that variant collapses
the same bilinear form, is **project ③** — so "does the measure objective itself
fail?" is answered there, while ① asks "do future-reachability reps fail, and where?"

**Papers:** `rlbrew/rlbrew.tex` (umbrella: *"Is Manipulation Hard for Unsupervised
RL?"*) · `fb-iwr/fb-iwr.tex` (*"Why Successor-Measure Methods Fail at Manipulation,"*
one object → four estimators FB/PSM/CRL) · `fb-cube-failure/fb_cube_failure.tex`
(FB-specific diagnosis, phase probe, figures).

**Thesis.** Future-reachability methods (GC values, contrastive scores, successor
measures, predictive reps) fail on manipulation when the representation
**under-resolves the interaction variable** (cube free / held / slipping / dropped).

**Key findings.**
- **Phase diagnostic** splits rollouts approach → control-acquisition → maintained-
  control → transport. Across 4 state methods, approach failures are negligible;
  **most failures occur *after* control acquisition** — the missing object is
  interaction-conditioned successor structure, not grasping.
- **Task-distribution probe:** PSM resolves the interaction variable (AUC **0.94**)
  far better than FB (**0.75**) — the readout, not the basis, is the gap
  ([[task-distribution-analysis-handoff]], [[psm-readout-ablation]]).

**Run status.** PSM+RLDP tuned 1M×10 + probe suite done ([[psm-rldp-probe-suite]],
[[psm-rldp-sweep-campaign]]); CRL+FlowBC campaign ([[crl-flowbc-campaign]]); pixel
paths (DrQ). Analysis pipeline merged to main (`feat/cube-analysis-pipeline`).

---

## ② OGBench-SOTA — is bilinear factorization a new SOTA?  ← **ACTIVE**

**Paper:** `ogbench-sota/ogbench-sota.tex` (+ `data-500k.json`). Swap each method's
MLP value head for bilinear `Q=φ(s,a)·ψ(s,g)/√d`; measure vs OGBench-published.
Green = local @500k, orange = OGBench paper @1M.

**Verdict = conditional SOTA.** Wins over published: GCIQL cube+flow **90.4 vs 68**;
GCIQL antmaze+DDPG+BC **83.7 vs 71**; GCIVL cube+AWR **67.5 vs 53**. Non-wins: HIQL
(see confound below), scene (only flow ran, 29.2 < 51). **The winning actor flips by
domain** (flow wins cube, DDPG+BC wins antmaze) — no single actor is uniformly SOTA.
Rank cliff at d5 (12.4 vs ~90 for d∈{50,128,512}).

**Run status — gap campaign RUNNING** (tmux `sotagap`,
`scripts/launch_sota_bilinear_gaps.sh`, 15 jax runs @500k, launched 2026-07-06):
- **scene GCIQL bilinear-d50 + DDPG+BC** (`GCIQL-Scene-Bilinear-d50-DDPGBC`) — the
  missing scene-SOTA contender.
- **HIQL debug at correct `subgoal_steps=10`** — native-AWR & flow, MLP vs bilinear.
  *Root cause found:* all prior HIQL runs used the OGBench default `subgoal_steps=25`
  (tuned = 10), so the negative HIQL result is confounded; our MLP baseline 24.6 is
  itself off-protocol (> published 15). Watch: does bilinear-sg10 clear 15?
- NEXT: collect @500k, update the HIQL rows + scene cell, rebuild PDF. Still open:
  HIQL/GCIVL on antmaze/scene, pixel bilinear. See [[sota-bilinear-paper]],
  [[hiql-subgoalsteps-bug-and-comparison]].

---

## ③ FB-GCIQL Bridge — why the measure objective breaks bilinear GCIQL

**Papers:** `fb-gciql-bridge/fb-gciql-bridge.tex` (+ `morph-table-500k.md` full grid,
`untried-design-space.md`) · `fb-modified/{fb-modified,isfb}.tex` (modified-objective
fix attempts). This is the **mechanism-isolation** study under project ①.

**Core finding (both morph directions agree).**
1. **The objective is the binding ingredient.** IQL value-TD on a bilinear critic ≈
   **90**; FB successor-**measure** TD on the *identical* form ≈ **9–34**. Flipping
   the objective costs ~55–80 pts with actor / form / rank / τ held fixed.
2. **Two structural levers, neither closes the gap *under the measure objective*:**
   forward-map task index `fb_forward_z` = the largest measure-side recovery
   (8.8 → **25.5**); the **learned backward map B is FB's bottleneck** — `z=g`
   (identity-B) recovers FB-IQL 18.8 → ~50, but the measure framework still tops out
   near FB's own **~48** ceiling. But the *same* task index under the **IQL** objective
   (**B1** `fb_forward_z`, measure OFF) = **0.848** ([[audit-sf-measure-20260706]]) —
   structure is fine; the objective is what breaks.
3. **Zero-shot GCIQL works (E1a, 2026-07-07).** The ~0.90 IQL critic recovers FULL
   exact-goal success from FB's API (reward samples → z → policy, no fine-tune):
   dataset-based moment `z=E[r·ψ]` = **0.955 = the exact-goal anchor 0.955** (3 seeds
   ×5 tasks). The prior "inferred-z → 0.07" G4 dissociation was an **on-policy
   sampling artifact** (a badly-conditioned policy rarely reaches the goal → degenerate
   z), not the formula or the actor. Normalization matters (~0.29): the actor consumes
   RAW ψ(g) (‖ψ‖≈41≠√d), so rescale z to ‖ψ(g)‖, not √d. Whitening (GLS) is refuted on
   the rank-deficient ψ (cond 7e14). ([[zeroshot-gciql-e1a-works]],
   `scripts/eval_zeroshot_inference.py`.)

**Anchors:** plain FB ≈ 48 @1M · bilinear-GCIQL ≈ 90 (exact) · **zero-shot GCIQL ≈ 95.5
(E[r·ψ] inference)** · B1 fb_forward_z-under-IQL ≈ 84.8 · FB-IQL bridge best ≈ 65 ·
torch reward-TD ceiling @500k×3 ≈ 0.65.

**Detours, both NULL/NEGATIVE.**
- **ISFB** (torch SG-ISFB, measure-improvement operator): coord 0.33 / awr 0.27 /
  hard 0.08 — does not rescue FB ([[handoff-torch-isfb-20260702]]).
- **SF-feature-TD on φ** (train only φ, frozen/identity base): 8-cell flow campaign
  + DDPG+BC control DONE (~0.09–0.16; DDPG+BC ≤ flow in every cell → the flow actor is
  not the bottleneck). **RETRACTED reading (audit 2026-07-06):** this is **NOT** a
  representation collapse. The SF pipeline never uses `batch['rewards']`; with
  identity/frozen ψ the implicit reward is `⟨s,g⟩/√d` — linear, maximized by
  *overshooting* the goal — so a *perfectly fit* SF critic yields the wrong policy and
  the ~0.09 is over-determined (`sf_loss` converged fine, sane scales). Corrected
  reading: **a sparse goal indicator is not linearly expressible in identity/frozen
  features** — you need a *learned* basis (which is exactly FB's learned-B + ortho job).
  This **strengthens the readout-side half** of the bridge thesis and **retracts the
  training-collapse half** ([[audit-sf-measure-20260706]]). Code committed `7b405bf`.

**Now running / done.** **E1a zero-shot = DONE** (0.955, above). **E2 — scale-controlled
IQL+λ·measure interpolation RUNNING** (tmux `e2measure`, 15 runs @500k, launched
2026-07-07): grafts the measure onto the working ~0.90 IQL critic with
`measure_scale_norm` ON (the 2 knobs fix the confound that crashed 18/18 raw mc* runs;
smoke confirms q_mean bounded). Reads whether the measure gradient destroys / coexists /
helps via a λ dose-response + a stopgrad-φ quarantine; the **prize** is cross-evaluating
each cell with the E1a inference script ([[e2-measure-mix-campaign]]). E1b (inference-
consistent actor training) was **dropped** — E1a made it moot.

**Remaining untried queue** (`untried-design-space.md §3`): B1 (0.848) and C2-MeasureTwoTower
(0.076–0.084) are **done**; still open: B2/B3/B4 further FB structure on the 90 IQL critic ·
C3/C4 rebuild real FB inside the measure · two-tower d∈{128,512} + transfer. Emerging
outcome (B1 + zero-shot): structure survives on the IQL base → the gap is **objective, not
structure**.

---

## Code knowledge graph (`graphify-out/`)

A navigable graph of the whole repo lives at **`graphify-out/`** (untracked, ~9 MB,
rebuilt automatically — see below). Built 2026-07-10 over all 674 files:
**4,558 nodes · 8,583 edges · 356 communities** (4,095 nodes from AST extraction,
472 semantic from the `configs/`, `PAPER/`, and figure files).

| File | What it is |
|---|---|
| `graphify-out/graph.html` | interactive graph — open in a browser, no server |
| `graphify-out/GRAPH_REPORT.md` | audit report: god nodes, cohesion, token cost |
| `graphify-out/graph.json` | raw graph, GraphRAG-ready |

**Use it before grepping** for architecture questions — `graphify query "<question>"`,
`graphify path "<A>" "<B>"`, `graphify explain "<node>"`. Community labels are
hand-written and persist in `.graphify_labels.json` across rebuilds.

**Auto-rebuild.** `.githooks/post-commit` (+ `post-checkout`) re-runs AST extraction on
changed code files and rebuilds the graph in the background; log at
`~/.cache/graphify-rebuild.log`. Requires `core.hooksPath=.githooks` (already set).
It is a no-op when no code topology changed.

**Two caveats, both real.**
1. **The hook only tracks code.** Doc, config, and image changes are ignored — a new
   `PAPER/` writeup or `configs/gciql/` cell will not enter the graph until someone runs
   `/graphify . --update` by hand. Given how much of ③'s design space lives in
   `configs/gciql/`, this matters.
2. **The graph under-connects.** The build reported **1,582 dangling-endpoint edges** —
   edges naming a node that does not exist, mostly AST `references` to external symbols
   (`torch.Tensor` &c.). Traversal answers can be confidently incomplete; verify anything
   structural against the source before trusting it.

---

## Operating rules (memory)
Run everything in **tmux** ([[always-run-in-tmux]]); wandb entity **amsks** only
([[wandb-entity-amsks]]); you ARE the GPU box, act locally
([[host-identity-you-are-aws-gpu]]); commits omit the Claude co-author trailer
([[commit-no-claude-coauthor]]); never co-init 2 jax on one GPU
([[jax-gpu-packing-preallocate]]); build PDFs with `~/.local/bin/tectonic`
([[latex-engine-tectonic]]). Two venvs: `.venv` (torch/FB) · `/dev/shm/.venv-jax`
(GCIQL jax, [[jax-venv-location]]). PAPER artifacts stay untracked.
```
PAPER/
  rlbrew/  fb-iwr/  fb-cube-failure/    ← ① RLBrew
  ogbench-sota/                          ← ② OGBench-SOTA
  fb-gciql-bridge/  fb-modified/         ← ③ FB-GCIQL Bridge
  specs/                                 ← shared design docs
```
