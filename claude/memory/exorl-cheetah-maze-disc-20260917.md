---
name: exorl-cheetah-maze-disc-20260917
description: 2026-09-18 RESOLVED maze = actor lr 1e-6 (Density-FB 302 -> 768, above every published method); cheetah = METHOD gap (FB-ref 366 in our pipeline vs Density-FB 232), kernel radius not the lever; round-2 cheetah probes (lr_actor 1e-4, kernel actor) + maze seeds 2-4 running tmux exorl_disc2
metadata:
  type: project
---

Table context: only the "Ours" column is our Density-FB (2M/5 seeds); FB/HILP/RLDP/PSM/Laplacian are
PUBLISHED (RLDP-paper Table 1). Cheetah 232 vs FB 586; maze 302 vs FB 427.

Reference check (td_jepa launch_fb_dmc.py sweep_pointmass, rldp_train_dmc.py:57-85, HILP README, Touati):
- cheetah: no reference changes ANY knob for cheetah => not a missing hyperparameter.
- point-mass: every reference drops actor lr to 1e-6 (td_jepa; RLDP's FB also lr_b 1e-6), discount 0.99.
  Our maze runs used lr_actor 3e-4 and the failure is a saturated/collapsed TD3 actor => hyperparameter.
- kernel is NOT the lever on either: maze bump already cube-sharp; sup86 sharper radius changed neither
  cheetah (212 vs 232) nor maze (273 vs 302).
- Cheetah pipeline is UNVERIFIED: only other cheetah runs are inert-ortho RLDP (14/65/4/17), which is worse
  than Density-FB. FB-reference config has never run on cheetah/maze.

LAUNCHED 2026-09-17 22:52Z (tmux `exorl_disc`, scripts/launch_exorl_disc.sh, logs/exorl_disc/, 1M, 8 GPUs):
(a) fbref cheetah s0-2 -> exp/exorl_fbref/dmc-cheetah_*  (~500 => method gap; ~230 => pipeline gap)
(b) fbref maze s0-2, discount .99, lr_actor 1e-6 -> exp/exorl_fbref/dmc-point-mass-maze_*  (target 427)
(c) density_fb_z maze lr_actor 1e-6 s0-1 -> exp/exorl_dfb_mazelr/  (target >302, unsaturated actor)
Score: scripts/exorl_paper_eval.py on params_1000000.pkl (fb -> 50k samples). Note appended to
docs/notes/2026-09-17-exorl-audit.md. Related: [[exorl-audit-20260917]].

RESULT 2026-09-18 13:10Z (exp/paper_eval_exorl_disc*.jsonl, 1M, paper protocol):
- Density-FB maze lr_actor 1e-6: 942/685/799/646 = **768** (seeds 0,1) vs our 2M@3e-4 302; published best RLDP 671.
  MAZE CLOSED = hyperparameter. Seeds 2,3,4 running -> exp/exorl_dfb_mazelr (five-seed table).
- FB reference maze: 731/329/414/4 = 370 vs RLDP-paper FB 427 (same shape) => pipeline reproduces published FB on maze.
- FB reference cheetah: 136/166/495/668 = 366 at 1M vs published 412-513 at 1M => pipeline fine; Density-FB 232 is a
  METHOD gap on cheetah. Round 2 (tmux exorl_disc2, scripts/launch_exorl_disc2.sh): cheetah lr_actor 1e-4 x3 ->
  exp/exorl_dfb_cheetahlr; cheetah kernel actor x2 -> exp/exorl_dfb_cheetahkernel. Score with
  scripts/exorl_paper_eval.py --task-set paper via symlink layout exp/exorl_disc_score/<domain>/<critic>/<run>.

PAPER 2026-09-18 13:40Z: ExoRL block ADDED to tab:manipulation in PAPER/ICLR/content/experiments.tex (FB/HILP/RLDP
columns = TD-JEPA Table 1 proprio, mean ± s.e.; GCIQL/CRL/QRL = ---; Density-FB walker 633.8±18.2, cheetah
232.0±39.0, quadruped 568.7±50.1, point-mass 768.2±30.8; s.e. = over per-seed domain means). Caption now says
"success rate on manipulation, episode return on ExoRL, mean ± standard error". One prose sentence added after the
GCIQL sentence. REFRESH point-mass when maze seeds 2-4 land, cheetah if a round-2 probe wins. User ruling: do NOT
mention that the published columns are transcribed from TD-JEPA. Full main.tex does not build from PAPER/ICLR
(figure paths are relative to PAPER/); verify tables standalone.

RECIPE 2026-09-18 14:00Z: user says "1M run was better". Confirmed for WALKER only: 1M recipe (tau 0.005, value
[512,512,512], exp/walker_arch, 3 seeds) = 680.8±8.3 vs 2M reference recipe 633.8; the reference recipe stopped at
1M (exp/walker_dfbref, now scored: exp/paper_eval_exorl_disc_walkerdfbref.jsonl) = 639 => it is the RECIPE, not
the budget. Cheetah 1M recipe 197 < 232. Table walker cell -> 680.8±8.3 (mixed recipe for now).
QUEUED tmux `exorl_1mrecipe` (scripts/launch_exorl_1mrecipe.sh, starts after exorl_disc2 ends): 1M recipe on
quadruped x5, maze (lr_actor 1e-6) x5, walker seeds 3,4 -> exp/exorl_dfb_1mrecipe/, logs/exorl_1mrecipe/.
Then refresh all four ExoRL cells on the one recipe.

REQUEUED 2026-09-18 14:05Z: tmux `exorl_1mrecipe` now = ALL four domains x 5 seeds at the 1M recipe (tau .005,
value [512,512,512]) with the REFERENCES' actor lr 1e-4 (maze 1e-6) -- the one maze-style reference deviation
never applied to walker/cheetah/quadruped (ours was 3e-4 everywhere). 20 jobs, starts after exorl_disc2.
Control: walker at 3e-4 = 681 (exp/walker_arch). -> exp/exorl_dfb_1mrecipe/.

OGBENCH 5-SEED 2026-09-18 15:00Z: Density-FB seeds 3,4 existed (Phase-2 top-up) but table cells were 3-seed.
Recomputed (docs/notes/2026-09-18-ogbench-five-seed-aggregate.md): density_fb_z cube-single 82.9, cube-double
31.7, scene 54.0, puzzle 5.9 (all +1.2..1.6); density_fb_z_state scene 49.6->46.3. tab:manipulation UPDATED to
5-seed cells; spread convention = POPULATION STD over seeds (the table's existing convention; RLDP row is sample
std, ortho1 arm) -- caption reverted to "standard deviation"; ExoRL cells now pop-std too (walker 680.8±11.8,
cheetah 232.0±78.1, quadruped 568.7±100.2, maze 768.2±30.8). Baselines fb_tdj/hilp_tdjc/rldp = 3 seeds; 24
top-up commands + message in handoff/baseline_seeds34/ (NOT posted; rldp lines go to exp/ortho1 at ortho 1).
GCIQL/CRL/QRL cells = published OGBench numbers. Luis idle since 09-12, 09-16 handoffs unacknowledged.

ROUND 2 SCORED 2026-09-18 16:30Z: cheetah lr_actor 1e-4 = 292 (3 seeds, +60 vs 232; FB-ref 366); kernel actor
238 (no gain); maze lr 1e-6 five seeds = 692±88 (still > RLDP 671). Table point-mass -> 692.2. NEXT: score
exp/exorl_dfb_1mrecipe (20 jobs, tau .005 [512,512,512], lr_actor 1e-4 / maze 1e-6, all 4 domains x 5 seeds;
launched ~14:00Z, ~10 h) with scripts/exorl_paper_eval.py --task-set tdjepa (maze: paper) via symlink layout
exp/exorl_disc_score_*/<domain>/density_fb_z/<run>; then replace ALL four ExoRL cells (pop std) in
PAPER/ICLR/content/experiments.tex and re-render (standalone tex in scratchpad tab_standalone.tex).

POSTED 2026-09-18 16:28Z: commits 53aa629 (rebalance lever + handoffs) and 47a369e (ExoRL scripts/notes) on density-fb
(unpushed; bundle on hub handoff/rebalance/factored-fb-rebalance.bundle); mailbox posts to luis:
mailbox/evalbox/2026-09-18T16-28-31Z.md (rebalance, 72 cmds, pre-registered reading) and ...16-28-34Z.md (baseline
seeds 3,4, 24 cmds). Onset-window contents note: docs/notes/2026-09-18-onset-window-contents.md (windows = grasp
closure decision; carry phase untouched). From here: ExoRL only (rule-tokens-subagents-exorl-focus-20260918).

IMPL AUDIT 2026-09-18 17:00Z (docs/notes/2026-09-18-density-fb-exorl-impl-audit.md, 168 tests pass): NO math bugs.
A1: our FB-ref runs used lr_actor 3e-4 (actor yaml overrides fb.yaml's 1e-4) => walker 681 cells ARE rate-matched
to fbref; the 20-job batch at 1e-4 is matched to PUBLISHED, not to fbref. B1 (new lever): backward features
collapse under measure_density_b_norm=true (mean pairwise cos 0.95 @1M vs FB 0.03; actor reward std halves,
q_mean decays). PROBE LAUNCHED tmux `exorl_bnorm0` (scripts/launch_exorl_bnorm0.sh): walker x3 + cheetah x2,
walker_arch recipe (tau .005, [512,512,512], lr_actor 3e-4) + b_norm=false -> exp/exorl_dfb_bnorm0/. Readout at
250k: B cosine vs walker_arch; then paper-score at 1M. If collapse persists: next lever = FB ortho term on B (coef 1).

BNORM PROBE 250k 2026-09-18 19:30Z: b_norm=false does NOT fix the B collapse (mean energy 97% of total, proxy std
.35 vs FB .90) and walker/cheetah 250k returns are level-or-below the batch => not a lever; left to 1M for a score
only (exp/exorl_dfb_bnorm0). 1M-recipe+lr1e-4 batch 250k in-run: walker 593-764, cheetah 49-106, quadruped 101-530
(s4 weak), maze 935-944 except s1=44. Batch done ~00:00Z 09-19 -> score + refresh ExoRL cells (monitor armed).

★★★ BATCH SCORED 2026-09-19 00:50Z (exp/paper_eval_exorl_1mrecipe.jsonl; 5 seeds, 1M recipe, lr_actor 1e-4 / maze 1e-6):
walker 724.6±41.2, cheetah 301.4±45.4, quadruped 655.9±15.9 (> published FB 596), maze 751.5±74.1 (> all published).
tab:manipulation ExoRL row UPDATED to these (pop std). All four cells now one recipe / one budget / five seeds.
Still pending: exp/exorl_dfb_bnorm0 1M score (not expected to matter). Gap to published FB: walker (812), cheetah (673).
BNORM0 1M 2026-09-19: b_norm=false HARMFUL (walker 519 vs 681, cheetah 159) -> keep b_norm=true. ExoRL job CLOSED
at the 09-19 cells; remaining walker/cheetah gap to FB is not explained by any tested lever (radius, kernel actor,
b_norm); untested matched knobs = F 1024x1 + tau .001 separately, 256-d left encoder.

LEVER SWEEP LAUNCHED 2026-09-19 11:26Z (tmux `exorl_levers`, scripts/launch_exorl_levers.sh, 16 jobs = 8 levers x
{walker, cheetah} x seed 0, 1M, base = 1M-recipe batch): qnorm (actor.q_normalize=true), sphere
(z_rand_frac=1.0), tower1024 ([1024] @ tau .005), tau001, headmean (NEW knob measure_density_actor_head_reduce=mean,
113 tests pass), temp2 (actor_temp 2), disc099, zdim100. Score: bash scripts/score_exorl_levers.sh ->
exp/paper_eval_exorl_levers.jsonl, deltas vs seed-0 baseline walker 743.6 / cheetah 274.0. Winners (> seed spread,
walker ~40, cheetah ~45) -> 5 seeds. Monitor armed for completion. Published per-task comparison in the 09-19 reply:
walker 7-20% below FB on stand/walk/run; cheetah 40-60% of FB on every task; quadruped/maze fine.

LEVER SWEEP SCORED 2026-09-19 16:40Z (1 seed): sphere z_rand_frac=1.0 (+19 walker, +66 cheetah) and actor.q_normalize=true
(+35 walker -> 778 = FB-ref level, +18 cheetah) help both; tower1024 +106 cheetah / -125 walker; tau001, headmean,
disc099 +62..66 cheetah / -36..-60 walker; temp2 bad; zdim100 neutral. ROUND 2 (tmux `exorl_combo`,
scripts/launch_exorl_combo.sh, exp/exorl_combo/): sq = sphere+qnorm x5 seeds walker+cheetah; 1-seed cheetah stacks
sq_tower1024 / sq_tau001 / sq_headmean / sq_tower1024_tau001 / sq_tower1024_headmean; walker sq_tower1024. Score:
scripts/score_exorl_combo.sh. If sq wins at 5 seeds -> rerun quadruped + maze with sq (10 jobs) for one recipe.

500k LIKE-FOR-LIKE 2026-09-19 19:40Z (5 seeds, paper protocol, exp/paper_eval_exorl_500k_*.jsonl): sphere+qnorm
walker 728.8±20.5 / cheetah 282.0±34.5 vs table batch @500k 692.7 / 249.7 => +36 / +32, consistent across seeds.
1M combo scores pending (tmux exorl_combo; score with scripts/score_exorl_combo.sh). If sq holds at 1M: rerun
quadruped + maze with sq (10 jobs) and replace all four table cells. GAP FIGURE DONE 2026-09-19:
PAPER/ICLR/figures/gap_preservation/ + docs/notes/2026-09-19-gap-preservation-figure.md; magnitude version is
NOT a result (both critics keep a few % of the true gap); the signal = ranking reversal at the largest gaps
(carrying): FB 53% reversed vs Density-FB ~30%; recommend middle panel + reversal fraction.

ROUND 2 @1M 2026-09-19 22:00Z (exp/paper_eval_exorl_combo.jsonl): sq (sphere+qnorm) 5 seeds walker 730.0±26.9 (+5),
cheetah 341.2±13.2 (+40). sq_tower1024_tau001 cheetah seed0 = 472 (+171!); sq_tower1024 walker 730 (tower no longer
hurts under sq). ROUND 3 = arm `sqt` (1M-recipe + z_rand_frac 1.0 + q_normalize true + value [1024] + tau .001) on
ALL four domains x 5 seeds (19 jobs; cheetah s0 reused) -> exp/exorl_sqt/, scripts/launch_exorl_sqt.sh,
score_exorl_sqt.sh. If it holds -> replaces all four table cells (one recipe). GAP FIGURES DONE: fig_decision_by_phase
(FB carry ranking 0.50 vs DFB 0.77) + fig_separation_by_phase (FB keeps 5% of true future separation, DFB 15-19%,
all phases; both far from 1). Notes docs/notes/2026-09-19-{gap-preservation-figure,separation-preservation}.md.

ROUND 3 STATUS 2026-09-20 02:00Z: all 19 `sqt` jobs alive (tmux exorl_sqt), 500k-750k, done ~06-07Z; in-run walker/cheetah
evals above the table batch at matched steps. Monitor armed; score with bash scripts/score_exorl_sqt.sh.
PAPER 2026-09-20 02:15Z: experiments.tex was OVERWRITTEN from elsewhere at 02:13Z (ExoRL rows gone, OGBench cells back to
3-seed 81.3/30.4/52.7 -- NOT re-fixed). USER DECISION: report walker/quadruped/point-mass NOW, cheetah WAITS for the
remaining runs ("we still need to fix cheetah"). Added standalone tab:exorl (FB/HILP/RLDP/ours, 3 rows, 1M-recipe
cells 724.6±41.2 / 655.9±15.9 / 751.5±74.1 pop std, published = TD-JEPA proprio mean±s.e.) after tab:manipulation +
rewrote the ExoRL prose sentence to \cref{tab:exorl}. Compiles standalone. Best 5-seed cheetah so far = sq 341.2±13.2.

★★ ROUND 3 SCORED 2026-09-20 ~06:30Z (exp/paper_eval_exorl_sqt.jsonl, paper protocol, 5 seeds, 1M): sqt arm
(sphere z + q_normalize + value [1024] + tau .001) walker 750.2±29.8 (+26), cheetah 431.0±82.0 (+130; seeds
451/495/517/406/286), quadruped 491.2±206.7 (-165; seed1 collapsed 112), maze 497.3±82.4 (-254). NOT one recipe:
helps walker/cheetah, hurts quadruped/maze. Table cells LEFT at the 1M-recipe batch (walker 724.6 / quad 655.9 /
maze 751.5); cheetah still withheld (best 431 < RLDP 576). Open choice for the user: per-domain recipe (walker 750,
cheetah 431) vs one recipe.

CHEETAH ROUND 4 LAUNCHED 2026-09-20 09:40Z (user: "better performance in 3-4 h"): tmux `exorl_cheetah3`,
scripts/launch_exorl_cheetah3.sh, exp/exorl_cheetah3/{velblock (hit spec 8:17:7.264 = velocity block, seeds 0-2),
speedcol (8:9:0.021 = forward-speed column, support 29/1024, seeds 0-1), tau0025 (round-3 recipe, tau .0025, seeds
0-2)}, 1 job/GPU (~340k steps/h => done ~12:45Z). Cheetah obs = qpos[1:] (8) + velocity (9). Score with
scripts/exorl_paper_eval.py via a symlink layout (no score script written yet). ORACLE Z-SEARCH (test 1):
scripts/exorl_zsearch.py (CEM on the sqrt(d) sphere, eval-only), tmux `zsearch`, exp/zsearch/cheetah.jsonl, round-3
seeds 2 (best) and 4 (worst) x {run, walk}; inferred-z 10-ep returns: s2 run 260 / walk 763, s4 run 122 / walk 512.

★★ ORACLE Z-SEARCH RESULT 2026-09-20 11:10Z (exp/zsearch/cheetah.jsonl, round-3 ckpts, 10-ep returns):
s2 run 260 -> best 266 (= published FB run 247-258: critic at ceiling); s2 walk 763 -> 911 (> FB 827);
s4 run 122 -> 180; s4 walk 512 -> 795. => RUN gap = weak seeds (stability), WALK gap = TASK INFERENCE (150-280
left on the table). Next lever for cheetah = the reward->z path on walk (encoder readout vs ridge z vs #samples),
eval-only on the same ckpts. velblock/speedcol/tau0025 arms (exp/exorl_cheetah3) address the critic side only.
INFERENCE VARIANTS SCORED 2026-09-20 11:35Z (exp/paper_eval_exorl_sqt_infer_{lam0001,lam01,lam1,n10k,n200k}.jsonl,
scorer now has --override passthrough): ridge .001/.1/1 -> 440/450/426, samples 10k/200k -> 442/442 vs base 431;
walk stays 590-640 (oracle 800-900). => NOT the ridge or sample count; the linear reward->z regression itself is
the limit on cheetah's thresholded rewards. Untested zero-shot lever: model-based z search (pick z maximizing the
critic's predicted value of the reward samples, no env). Round-4 training at 500k (11:40Z), 1M ~12:45Z.

★★★ CHEETAH BREAKTHROUGH 2026-09-20 ~18:00Z: SPEED-COLUMN HIT SPEC (measure_hit_spec "8:9:0.021", hit kernel on the
forward-speed obs column only, support 29/1024) on the round-3 recipe: seeds 0/1 = 568 / 528 (mean 548) vs 473 for
the same seeds in round 3; per task run 187, run_back 262, walk 778, walk_back 966 (published FB 258/307/827/981).
Seeds 2-4 launched (tmux exorl_speedcol, GPUs 3-5, exp/exorl_cheetah3/speedcol) -> five-seed cell ~21:00Z. Score
with: EXTRA="--override measure_hit_spec=\"8:9:0.021\"" bash scripts/score_exorl_cheetah3.sh speedcol "0 1 2 3 4" <gpu>
(the scorer forces the default spec otherwise -> ScopeParamShapeError). tau0025 = NOT a lever (407 vs 488).
velblock scoring in progress. ExoRL TABLE: still the one-recipe batch (walker 724.6 / quad 655.9 / maze 751.5);
walker 750.2 (round 3) NOT in the table; cheetah withheld pending the 5-seed speedcol cell.
USER DECISION 2026-09-20 ~18:30Z: ExoRL table = BEST five-seed cell PER DOMAIN (per-domain hyperparameters, caption
says so, \cref{app:exorl-hparams}): walker 750.2±29.8 (round-3 recipe), quadruped 655.9±15.9 (1M recipe), point-mass
751.5±74.1 (1M recipe); cheetah -> speedcol five-seed cell when seeds 2-4 land (~21:00Z; seeds 0/1 = 548).
★★★ VELOCITY-BLOCK ARM 2026-09-20 ~19:30Z (hit spec "8:17:7.264"): seeds 0/1/2 = 506/569/606, mean 560.5 vs 488 round-3
same seeds; run 218 / run_back 279 / walk 810 / walk_back 936 (published FB 258/307/827/981). Seeds 3,4 launched
(tmux exorl_velblock, GPUs 6-7). Speed column 548 (2 seeds), seeds 2-4 running. Both five-seed cells ~21:30Z; score with
EXTRA="--override measure_hit_spec=\"8:17:7.264\"" (velblock) / "8:9:0.021" (speedcol) bash scripts/score_exorl_cheetah3.sh <arm> "0 1 2 3 4" <gpu>.
Lesson: the hit-kernel COLUMNS (velocity block) are the cheetah lever, not the radius (sup86 was radius-only).
MAIN TABLE MEANS 2026-09-20 ~20:00Z (tab:manipulation, computed from the cells AS IN THE FILE -- OGBench cells are
still the 3-seed 81.3/30.4/52.7, not the 5-seed 82.9/31.7/54.0): OGBench mean FB 19.3 / HILP 40.5 / RLDP 13.5 /
GCIQL 53.0 / CRL 16.0 / QRL 3.7 / DFB 54.8; MetaWorld 36.7/28.9/35.8/10.0/16.9/87.7/80.2; ManiSkill
49.7/40.4/46.3/36.5/13.4/65.8/79.5; mean over 11 tasks 37.9/37.3/34.5/33.8/15.1/54.9/72.9. Rows inserted as
\quad\textit{... mean}; best bolded.
SPEED-COLUMN FIVE SEEDS 2026-09-20 ~16:40Z: 568/528/483/511/571 = 532.1 ± 33.7 (pop sd); run 192 / run_back 289 / walk
711 / walk_back 937. Velocity-block (3 seeds 560.5) seeds 3,4 finishing -> five-seed cell decides the table row.
LEVER 1 LAUNCHED 2026-09-20 ~17:00Z (tmux exorl_kernels, scripts/launch_exorl_kernels.sh, exp/exorl_kernels/, 12 jobs, 2/GPU
on 0-5, ~5 h): walker_vel "15:24:15.428", walker_hv "14:15:0.007,15:24:15.428" (base = round-3 recipe); quadruped_vel
"44:47:0.110", quadruped_vu "44:47:0.110,47:48:0.001" (base = 1M recipe); seeds 0-2. Compare vs walker 750 / quad 656.
LEVER 2 (agent): scripts/exorl_value_infer.py = CEM/grad over z scored by the critic's exact-reward value on a pool
(no env), on cheetah sqt seeds 2/4 walk+run vs ridge z and oracle.
★★★ CHEETAH CELL 2026-09-20 ~17:40Z: velocity-block five seeds 506/569/606/526/492 = 539.9 ± 42.0 (run 187 / run_back
266 / walk 767 / walk_back 940) > speed column 532.1 ± 33.7. IN THE TABLE (tab:exorl cheetah row) + app:exorl-hparams line.
vs TD-JEPA FB 672.7 / HILP 618.3 / RLDP 575.6.
LEVER 2 NEGATIVE 2026-09-20 ~18:20Z (scripts/exorl_value_infer.py, note docs/notes/2026-09-20-exorl-value-inference.md):
model-based z inference (CEM / Adam on J(z) = critic's exact-reward value on a pool) does NOT close the walk gap: the
critic ranks the ORACLE z BELOW the ridge z (J .41 vs .45 on s2 walk), so maximising J moves away from it; walk gain
vs ridge = +25 (CEM) / +2 (grad) on 2 seeds. Model error exploited, same as spec-31 z-search on cube. Readout parity
with q_fn verified (9e-8 at f32). => the inference gap is a CRITIC VALUE ACCURACY across-z problem, not the regression.
EXORL TABLE 2026-09-20 ~19:00Z: tab:exorl = FB/HILP/RLDP (TD-JEPA Tab.1) + ours, cells walker 750.2 / cheetah 539.9 /
quadruped 655.9 / point-mass 751.5, mean row 648.2/620.1/610.2/**674.4±22.9** (propagated). Per-task appendix table
tab:exorl-tasks in app:exorl-hparams (walker stand 836/walk 830/run 350/spin 985; cheetah 187/266/767/940; quadruped
stand 904/walk 527/run 460/jump 733; maze 944/801/740/521). Kernel arms (exorl_kernels) still training: quadruped ~90%,
walker ~50% (slow at 2/GPU); in-run evals favour quadruped_vu and walker_hv strongly.
