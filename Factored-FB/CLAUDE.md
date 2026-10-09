# Factored-FB

Faithful port of **Forward–Backward (FB) / FlowBC** representations (from
`td_jepa/metamotivo`) to **OGBench cube manipulation** (state + pixel). Core:
`Q(s,a,z) = F(s,z,a)·B(s')`; task `z` inferred from relabelled offline
transitions at test time, no fine-tuning. The scientific thrust is a **failure
analysis of future-reachability methods on manipulation**.

**Living experiment doc = `PAPER/`.** The current draft is **`PAPER/rlbrew/`**
("Is Manipulation Hard for Unsupervised RL?"): future-reachability methods (GC
values, contrastive scores, successor measures, predictive reps) fail when the
representation **under-resolves the interaction variable** (cube freely-moving /
held / slipping / dropped). A phase diagnostic splits rollouts into approach →
control acquisition → maintained control → transport; across 4 state-based
methods, approach failures are negligible and most failures occur **after**
control acquisition — the missing object is interaction-conditioned successor
structure, not grasping per se. The fuller writeup is `PAPER/fb-cube-failure/`.
For any experiment, the overall picture is whatever `PAPER/` currently argues.

**Conventions:** commit only code/tests/docs (artifacts `PAPER/ analysis/
results/ datasets*/` and this `CLAUDE.md` stay untracked); two venvs — `.venv`
(torch, FB+analysis) and `.venv-jax-cpu` (jax, GCIQL). `origin/main` is the
trunk; local `main` is often stale, fetch first.

**Code knowledge graph.** A graphify graph of the repo lives in `graphify-out/`
(untracked). For architecture questions — what calls what, where a concept lives,
how two subsystems connect — run `graphify query "<question>"` **before** grepping.
`.githooks/post-commit` rebuilds it from changed **code** on every commit; doc,
config, and image changes need a manual `/graphify . --update`. The graph has
~1.6k dangling-endpoint edges, so verify structural claims against the source.
See `PAPER/PROJECT-MAP.md` for the full entry.

**Branches** (against `origin/main` @ `712ea28`):
- **`feat/cube-analysis-pipeline`** — the paper analysis itself (representation
  aliasing, counterfactual-value probes, phase diagnostic). **Merged into main.**
- **`feat/goal-conditioned-fb`** (16 ahead) — does giving FB the interaction/goal
  variable restore the missing structure? **V1** `goal_cond`: feed explicit goal
  obs into forward map + actor. **V2** `fixed_b=cube_xyz`: privileged USFA
  backward map `B=cube xyz` (`z_dim→3`). Both default-OFF (baseline unchanged).
- **`feat/psm-integration`** (21 ahead) — adds **PSM** (proto successor measure)
  + PSM-FlowBC as another future-reachability method in the comparison;
  transcribed bit-exact from reference, with pixel DrQ + RLDP-pixel paths.

Runs are pulled from the AWS box via `scripts/pull_fb_runs.sh` into
`results/factored-fb-*`, then evaluated by the analysis pipeline.

---

_This file is gitignored (local-only). After editing, sync it to the remote box:_
_`rsync -avz CLAUDE.md aws-gpu:<repo-path>/CLAUDE.md` (see `scripts/sync_claude_md.sh`)._
