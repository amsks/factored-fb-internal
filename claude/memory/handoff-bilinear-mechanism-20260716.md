---
name: handoff-bilinear-mechanism-20260716
description: 2026-07-16 session handoff — bilinear-mechanism program + algorithm ideas; full state in analysis/atlas/HANDOFF-20260716.md
metadata: 
  node_type: memory
  type: project
  originSessionId: 38f4d6b7-66fb-4a86-95f6-252ae7793c47
---

Active program: "why does bilinear GCIQL + flow win?" mechanism study + 3 new algorithm ideas +
supporting the research note `PAPER/atlas/findings.tex`. **Full handoff (live jobs, results, what
remains, gotchas): `analysis/atlas/HANDOFF-20260716.md` — READ IT FIRST.**

Headline results (all confirm: bilinear = compatible policy-learning system, not a better value
estimator): reranking the frozen bilinear critic over behavior samples = 0.91-0.96 (~its trained
actor) vs MLP 0.57-0.59; fresh actors vs frozen critics 0.88 (bil) vs 0.68 (MLP) — co-adaptation
refuted; fresh DDPG+BC vs frozen bilinear = 0.52 — off-support peak intrinsic; **FB reranking on the
proper 1.5M plain-FB checkpoint COLLAPSES with N (representation bottleneck, opposite of bilinear)**.

Live at handoff: dueling_v2 training (⚠ 50k eval floored 0.05 despite b-loss fix — watch 100k),
FB-rerank sweep, exp5 trajectory (5/10), antmaze/scene capped audits. tmux: dueling/audit/exp23/exp5.

Key fixes: `_load_fb` pointed at a 100-step SMOKE STUB → fixed to `fb_state_plain__s0` 1.5M ortho=1000;
dueling b-loss scale-normed. FB 1M checkpoint was NOT lost — inventory CSV had it; a subagent was wrong.

Related: [[handoff-oodgrid-generalization-20260713]] [[results-csv-pipeline]] [[fb-state-run-name-map]]
[[jax-venv-location]] [[wandb-entity-amsks]]
