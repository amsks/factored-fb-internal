---
name: handoff-20260914-levers-ablations-walker
description: "HANDOFF 2026-09-14 10:10Z (Measure program, paper PAPER/ICLR): RLDP/HILP lever results (objective transfers WITHOUT privileged input: FB 46→91, RLDP 28→53, HILP 71→77 cube-single; object-coordinate rows = input effect, plain RLDP/HILP oc reach 88/94), squared-vs-CE ablation running (D-FB 78 vs 81, D-RLDP 40 vs 53), walker: regression-task retrofits LOSE (RLDP 625→277, HILP 797→679 in-training evals; official scoring running), feature-index walker arm relaunched; toy ablation + rare-bottleneck lemma validated and written into 4.1 / failure.tex; reward-family eval done and in 4.3; 3 commits on density-fb (b8b3017, 4dfc2c3, 5dc6095) UNPUSHED (user pushes), more uncommitted."
metadata: 
  node_type: memory
  type: project
  originSessionId: be24e170-3bbf-4591-adc3-bb535ce89254
  modified: 2026-09-14T10:08:23.179Z
---

# Where things stand (2026-09-14 10:10Z) — continue from here

## Live tmux sessions on this box (all launched by me; do not kill others')
- `walker_score`: official TD-JEPA-protocol scoring of exp/walker_20260914 (48 cells, GPU 3, ~3 h). Output exp/paper_eval_walker_20260914.jsonl, log logs/walker/score.log (ends with SCORE_DONE). Driver needs the nested layout exp/<root>/dmc-walker/<critic>/<run_dir> — I created symlinks; per-critic loop: `scripts/exorl_paper_eval.py --root exp/walker_20260914 --critic <c> --actor ddpgbc --domains dmc-walker --seeds 0 1 2 --steps 500000 --episodes 10 --inference-samples 10000 --task-set tdjepa --out ...`.
- `walker_feat` (GPUs 3,4): objective-only (feature-index, NO hindsight regression) Density-RLDP at ortho 1 and Density-HILP (discount .98) on walker, 3 seeds, 500k, exp/walker_feat, logs/walker_feat. Seeds 1/2 logs still show the OLD traceback from the first attempt (missing `FB_MUJOCO_SITE_PACKAGES=$HOME/git/Factored-FB/.venv/lib/python3.11/site-packages` — REQUIRED for every ExoRL launch); they get overwritten when queued runs start. Score with the same driver (create symlink layout first).
- `sqloss` (GPUs 0,1,4,5): Density-FB with `measure_density_loss=sq` (squared error on softmax probs) cube-single DONE (85/76/73 → 78 vs CE 81.3), cube-double s0 at 500k (.21), s1/s2 at 200k. exp/sqloss.
- `sqloss_retro` (GPUs 6,7,2): Density-RLDP reg@ortho1 with `rldp_density_loss=sq` DONE (33/46/41 → 40 vs CE 53); Density-HILP reg with `hilp_density_loss=sq` running (just started). exp/sqloss_retro.
- `walker` (finished training; 12 runs at 500k, exp/walker_20260914): plain RLDP@ortho1, Density-RLDP reg@ortho1, plain HILP (disc .98), Density-HILP reg. In-training evals (mean of 4 tasks): RLDP 625 (540/684/651), D-RLDP-reg 277 (298/253/280), HILP 797 (805/849/737), D-HILP-reg 679 (648/626/764). ⇒ hindsight-regression retrofits LOSE on walker; user's read: don't use hindsight regression on ExoRL, use the objective-only variant there (hence walker_feat).
- `hlevers` DONE (exp/hlevers, cube-single 1M 3 seeds, Route B): D-HILP reg full-state 78/74/83; D-HILP reg + object coords 98/94/93 (=95); plain HILP + object coords 91/93/97 (=94); plain RLDP@ortho1 + object coords 84/92/87 (=88).
- `levers` DONE (exp/levers): D-RLDP@ortho1 trainb 32, reg 53 (52/51/57), oc 88, full 80. Older: ortho1/ortho1000/rldp_ortho01 arms all done (see docs/PAPER_CONFIGS.md).
- `rewardfam`, `failfig`, `toyabl`, `rarebn`: finished analysis sessions (can be killed).

## Headline results now
- Full state (no privileged input), cube-single Route B: FB 46→91 (density_fb_z_state), RLDP@ortho1 28→37 (objective only)→53 (+regression tasks), HILP 71→71→77. Cube-double FB 2→15 (state). Table: PAPER/ICLR/baselines/levers.tex (two blocks: full state / object coords; oc rows = input effect).
- Walker re-score with repaired inference: old D-RLDP 362→495 (plain 574→584 control). docs/notes/2026-09-14-walker-rescore.md. The Sept-3 walker cell was produced with a since-repaired reward-inference bug (mean instead of lstsq) + ortho .01.
- Non-goal reward families (eval-only, 6 methods, cube-single/double): Density-FB solves the two-cube conjunction (0.72 vs FB/RLDP < random floor); region rewards separate; preference flip complete; lift doesn't separate; dense z worse than sparse z everywhere. PAPER/ICLR/figures/reward_families/, baselines/reward_families.tex, docs/notes/2026-09-14-reward-families.md. Written into experiments.tex §4.3 + appendix app:reward-families.
- Toy ablation (didactic 9-state MDP): normalization-alone inert (A=B); softmax parameterization carries the gap under EITHER loss (C≈D); linear readout on softmax model fails (E); soft target caps at .77 (F). ⇒ CE is NOT the source of the gain (proposition holds for any proper loss); CE kept for gradient reason (p−t, occupancy-independent). PAPER/ICLR/figures/toy_ablation/, docs/notes/2026-09-14-toy-ablation.md. Written into §4.1 "Which of the three changes matters".
- Rare-bottleneck lemma validated: FB optimum gap → 0 as bottleneck weight w→0 (1/16: .19, 1/64: 0) or with ≥4 added ordinary states; Density-FB keeps it; bottleneck share of objective → 0; cost of keeping the gap does NOT vanish (~22% of optimum) — only the incentive does; limit = projected gap ⟨dΠ_k^0,r⟩ (0 at rank 2/3, .92 at rank 4). PAPER/ICLR/figures/rare_bottleneck/, docs/notes/2026-09-14-rare-bottleneck.md. Written into §4.1 "Making the bottleneck rare" (opening paragraph) and failure.tex lemma (user-written, I fixed typos; needs eigengap + projected-limit wording, which the user's version has).
- §4.1 CORRECTION: the existing "FB needs rank 6" numbers are TD-trained; FB's squared OPTIMUM keeps the gap at rank 3 (97.5%) at uniform weights, fails only at rank 2 — text now says so. failure.tex's "96% of row difference outside top-2" did NOT reproduce (99.9% with right-singular projection; 88% did).
- Failure figures (§4.2 new subsection sec:exp-failure, two subfigures): FB fails after the grasp (maintain/transport); FB prediction error ~.85 TV at every state type (near dataset prior), ranking at chance; Density-FB .55 TV, ranks carry actions correctly (.48 vs .06). Panel C (error vs success) dropped — no within-method relation.
- Levers: object-coordinate control came back LEVEL for both RLDP (88=88) and HILP (94≈95) ⇒ oc rows are NOT evidence for the objective.

## Paper state (PAPER/ICLR is gitignored; never committed)
- experiments.tex: 4.1 (rarity para + figure, rank para, ablation para + figure), manipulation table (no published cols; ± std; MW/MS spreads in), 4.2 failure subsection, 4.3 transfer + reward-family para/figure. Compiles with wrapper (needs subcaption, cleveref, algorithm, algpseudocode — added to main.tex).
- method.tex: user-rewritten; I fixed labels/typos, added actor/readout/task-index paragraph earlier; algorithm block present.
- appendix/theory.tex: task-index section rewritten (regression, invariances, guarantee wording per audit); appendix/implementation.tex: heads (min of scalar values, NOT elementwise), zero-shot protocol (no 4096 pool; goal route = kernel reward on 256-pool + same regression), outcome-coordinate table (hit specs per env), Figure-1 construction + restore, failure-diagnostic protocol + regret table, reward families, hyperparameter table (batch 1024, lr actor 1e-4, BC 3.0/0.3).
- introduction.tex: contributions paragraph (i–iv). abstract.tex: HILP/RLDP claim removed.
- baselines/: td_jepa_density.tex, reimplementation.tex, retrofits.tex, levers.tex, reward_families.tex (+ standalone PDFs).
- Dangling refs to check: eq:fb-factorization, eq:fb-value-readout, eq:action-gap (defined in failure.tex now?), app:loss-geometry exists; `newt` bib key missing (data-gen appendix); bib symlink broken in seam worktree.

## Code / git
- Commits on density-fb (local, UNPUSHED — user pushes; box has no creds): b8b3017, 4dfc2c3 (levers, ablation knobs, failure probe, docs/PAPER_CONFIGS.md), 5dc6095 (HILP lever, MW/MS pipeline + handoff docs/notes/2026-09-13-mwms-baselines-handoff.md).
- UNCOMMITTED: sq knobs in rldp_density/hilp_density (`rldp_density_loss`, `hilp_density_loss`) + test; walker reg configs rldp_density_tdj_reg.yaml / hilp_density_tdj_reg.yaml + aliases; the [M,1] reward-vector fix in both retrofits' regression infer_cond; launch scripts (hilp_levers, sqloss, sqloss_retro, walker_feat); scripts/reward_families/; 4 notes. Commit these next (one commit), leave push to user.
- Collaborator (MW/MS): HILP mask fix is in 5dc6095 (HILPFutureDataset now applies obs/goal mask); they must pull after push; their datasets ≠ the HF copy (see handoff note); Slack message drafted in transcript.

## Next steps (in order)
1. When walker_score finishes: tabulate official walker numbers (old/re-scored/new) → docs note + baselines table; decide the ExoRL claim (likely: FB no-loss; retrofits with hindsight regression lose; objective-only variants = walker_feat pending).
2. Score walker_feat when done (symlink layout + driver).
3. Collect sqloss cube-double + sqloss_retro HILP → ablation table (CE vs squared for all three backbones) → put into §4.1/appendix as the benchmark loss ablation; wording: parameterization is the mechanism, CE the fit that trains better (D-RLDP 53 vs 40 supports).
4. Commit uncommitted work; tell user to push.
5. Still open from audits: plain RLDP/HILP with regression tasks + squared loss (fairness control for the lever), MW/MS baseline runs (collaborator), lr_b 1e-5 RLDP cell on scene/puzzle (optional), reward-sample-count sweep (eval-only), didactic MDP figure caption check.

## Gotchas learned today
- ExoRL launches need FB_MUJOCO_SITE_PACKAGES (see above) or they crash at env build.
- `pkill -f "save_dir exp/x"` matches the shell running it — kills your own command. Use pgrep on the python process line.
- Subagents yield on background waits/monitors; tell them to use blocking `sleep` loops.
- User rules: no seed/CI caveats (more seeds coming); academic register, no metaphors; keep user's paragraph structure; plain RL language; RLDP ortho higher (1) is the tuned value; antmaze out of scope; cube/scene/puzzle focus.

## 2026-09-14 11:40Z update (paper tables + full-state queue)

- **Tables are manipulation-only now.** Antmaze dropped from `baselines/{td_jepa_density,reimplementation,retrofits}.tex`, `table.tex`, `table_main_1m.tex`, and the antmaze distillation weight in `appendix/implementation.tex`. `baselines/td_jepa.tex` deliberately KEPT intact (it is a verbatim transcript of the published table). Dropping the five navigation rows buys back a real **avg** row over the four manipulation tasks, recomputed per column.
- **`tab:tdjepa_density` is now the general comparison**: per backbone, published / ours / +density. Every column reads the FULL STATE (user ruling: no object coordinates). FB 15.2 / 17.5 pub vs Density-FB 90.9 cube-single, 15.8 cube-double, scene+puzzle EMPTY. HILP 30.7 vs D-HILP 30.9; RLDP 12.7 vs D-RLDP 18.6. `reimplementation.tex` demoted to the fidelity check (pub vs ours only).
- **RLDP baseline is now ortho 1 everywhere** (exp/ortho1, scored here): cube-single 27.6±3.7, cube-double 2.7±1.3, scene 10.2±3.2, puzzle 10.4±1.7 — much harder than TD-JEPA's grid value 0.01 (scene 4.7, puzzle 2.4).
- **Two holes the table now makes visible.** (1) Density-FB at full state exists ONLY on the two cube envs — verified against the HF hub, scene/puzzle were never run. (2) The D-HILP / D-RLDP columns use the regression task index only on cube-single; the other three rows are the host's own task rule (the variant that does nothing), so those two averages are not yet a measurement of the objective. Six runs each would close them.
- **tmux `dfbstate` = the queue for hole (1).** `scripts/launch_dfb_state.sh`: density_fb_z_state on scene_play and puzzle_3x3, 3 seeds each, 1M, into `exp/dfb_state`, logs `logs/dfb_state/`. Box was full, so it polls GPUs 0/1/2/5/6/7 (3 and 4 excluded — walker_feat chains reclaim those between seeds) and starts a run when one frees twice in a row. Both configs smoke-tested 300 steps: clean, both eval routes written, `route_b_valid=1`. Launch line = the campaign's density_fb_z line with the critic swapped; hub configs confirm the two differ in exactly one key, `backward_input=state`.
- Commit `c7e6e5c` on density-fb (sq-loss knob, walker reg configs, [M,1] reward fix, reward-family scripts, 4 notes). **Four commits unpushed**: b8b3017, 4dfc2c3, 5dc6095, c7e6e5c.
- Killable tmux (verified idle): gk, gk_smoke, hilp_rescore, zcov, ortho1, ortho1000, rldp_ortho01, levers, hlevers, failfig, rewardfam, walker.
