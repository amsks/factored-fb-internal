---
name: failure-tex-vs-tabular-20260920
description: 2026-09-20 audit of PAPER/ICLR/content/failure.tex vs the tabular bottleneck experiments -- lem:rare-bottleneck is referenced 5x in experiments.tex + proved in appendix/theory.tex but NOT DEFINED (failure.tex only has lem:linear-action-difference, a fixed-features projection lemma); gap identity carries a spurious 1/(1-gamma) vs prelims; family numbers all reproduce
metadata:
  type: project
---

failure.tex (user-modified ~09-19/20) states only lem:linear-action-difference: with backward features
FIXED, fitted gap = projection of the row difference onto span(B); lost gap = the part outside.
That is the "projection identity" of docs/notes/2026-09-14-rare-bottleneck.md and matches arm A
(weighted truncated SVD, readout F z). It does NOT imply the two predictions experiments.tex
attributes to sec:failure: (1) gap lost as the bottleneck rows get rare at fixed rank (needs the
subspace-selection step: Gram C_w -> C_ord, Pi_k^w -> Pi_k^0, now only in appendix/theory.tex
app:bottleneck-proof, whose target label lem:rare-bottleneck is dangling); (2) rank recovery and
Density-FB needing less rank (no lemma covers the softmax fit; the note says (iv) is empirical).
Other mismatches: eq:gap-identity has 1/(1-gamma) but prelims define M with mass 1/(1-gamma) and
Q = int r dM (factor belongs to the normalized measure); eq:discarded-gap has the LHS after the
label (renders wrong); "action action-value" typo; experiments ablation caption unfinished ("adn >");
main text says the rarity panel is on the five generated processes but fig_rare_bottleneck_lemma is
the single hand-drawn MDP at rank 3 (analysis/rare_bottleneck/run.py); process seed 2 has successor
rank 8 not 9 (state 0 unreachable). Numbers verified from analysis/toy_ablation/family_*.csv
restricted to mdps 0-4: FB optimum .088/.216/.491 at d=2/3/4; D optimum min .849 (d=2); rank-90%
table 5,8,6,7,7 / 5,6,5,7,6 / 2,2,3,3,2 / 3,2,4,5,3; retained mass per-arm-rank mean >= .836.
Rarity (single MDP, d=3): FB optimum 1.03 (w=1/2) -> .185 (1/16) -> .004 (1/128) -> -.002 limit;
TD .98 -> .05 (uniform) -> -.007; Density-FB optimum 1.00 at every w, TD .95 -> .885.
Recommended fix: restore a rare-bottleneck lemma statement in failure.tex (proof already in
appendix), keep the projection lemma as its first step, drop the 1/(1-gamma). Not edited (user
is editing the file live). Related: [[exorl-cheetah-maze-disc-20260917]].

LEMMA CHECK DONE 2026-09-20 (analysis/lemma_check/run.py, docs/notes/2026-09-20-lemma-check.md,
PAPER/ICLR/figures/lemma_check/fig_lemma_check.pdf, 3 tests pass): the fixed-features lemma HOLDS on the
five generated processes -- trained-B projection prediction vs FB's trained-F readout corr .994 / sign .998
over 400 runs (mean |diff| .02, worst .40 at rank 5); reward sweep 2950 pairs corr .997; appending the
residual direction restores the gap to 9e-16 for every reward at an objective cost of 4-28% over the
optimal rank d+1. Uncommitted.

EDITS APPLIED 2026-09-20 ~10:00Z (backups *.bak-20260920): failure.tex = new fig_lemma_mdp.pdf
(scripts/didactic/plot_lemma_mdp.py: graph + m+/m-/Delta m bars at state 7), typos fixed, 1/(1-gamma) dropped
(prelims: Q = int r dM), closing paragraph pointing to app:bottleneck-proof; experiments.tex didactic subsection
rewritten around lem:linear-action-difference + new fig:lemma-check (panels a, c), rarity paragraph now cites
prop:rare-bottleneck (NEW proposition statement added in appendix/theory.tex, proof wrapped in \begin{proof}),
ablation "more stable" sentence corrected, caption completed; bottleneck-family.tex + app:lemma-rewards (panel b).
No lem:rare-bottleneck refs remain. Full paper compiles (2 passes, no undefined refs) once method.tex's missing
\softmax macro is stubbed -- that macro is a pre-existing break, not ours.

SINGLE-BOTTLENECK MDP 2026-09-20 ~10:30Z (user wants ONE bottleneck state -> two clusters, measure only there,
no goal state): DISJOINT clusters do NOT show the failure (FB rank-2 optimum keeps .81-1.01 of the gap: the
difference is the top-mass direction). Clusters that MERGE into a shared tail do: analysis/lemma_check/
design_single_bottleneck.py, chosen layout build(entry=4, cluster=2, tail=6, merge=True) = 15 states (0-3 entry
chain, 4 bottleneck, {5,6} A, {7,8} B, 9-14 tail): FB reads 0.00 of the gap for d<=5 (.85 at d=6), softmax CE
1.11/1.02 at d=2/3; share of |dv|^2 in ordinary rows' top-5 span = 0. Figure PAPER/ICLR/figures/didactic/
fig_single_bottleneck_mdp.pdf (scripts/didactic/plot_single_bottleneck_mdp.py). NOT yet wired into failure.tex;
all tabular experiments (rarity, rank, ablation, lemma check, swap/margin) must be RE-RUN on this process.
SIMPLIFIED 2026-09-20 ~11:00Z: entry states nearly free (1 suffices); tail NOT needed -- what is needed is that the two
futures OVERLAP. Minimal process = 6 states (entry 0, bottleneck 1, A {2,3}, B {4,5}, each cluster's 2nd action
leaks into the other cluster's first state): FB rank-2 optimum reads 0.00 of the cluster-A gap (recovers at 3),
softmax 0.92; disjoint clusters never fail (FB 1.01 at rank 2). Tail length = rank at which FB recovers (~1 rank
per 2 tail states); use a tail only if the rank sweep needs a wider failure region. Figures:
figures/didactic/fig_single_bottleneck_mdp_notail.pdf (6 states) and fig_single_bottleneck_mdp.pdf (args: entry
tail; 2 4 = 11 states, fails at ranks 2-3). build(..., mutual=True) in design_single_bottleneck.py.

★★★ SINGLE-BOTTLENECK SUITE DONE 2026-09-20 ~12:00Z (analysis/single_bottleneck/run.py + gap_direction.py, note
docs/notes/2026-09-20-single-bottleneck-suite.md, figs PAPER/ICLR/figures/single_bottleneck/fig_sb_*, fig_gap_direction):
processes P6, P11, R1-R5 (template draws). Rank-90% medians TD: FB 5 / normsq 5 / softmax+sq 3 / Density-FB 3; trained
fraction at d=2/3/4: FB .11/.38/.63, DFB .70/.94/1.00. Lemma check 560 runs corr .997 sign .989; reward sweep corr 1.000;
residual direction restores gap (7-8% cost). READOUT SWAP (closed, d=2/3/4): FB feats+linear .07/.61/1.03, FB feats+
softmax .22/.65/1.00, DFB feats+linear .30/.20/.56, DFB feats+softmax .89/.97/.99 => at rank 2 NEITHER half alone
rescues; mitigation = normalization + features that SEPARATE cluster A (softmax rows keep >=.85 whenever A separable;
FB rank-2 feats separate A in 1/6 processes, DFB 23/30, 30/30 at d>=3; margin-vs-gap Spearman .91 linear / .69 softmax).
Gap direction: P11 mass .75/.11/.04/.03, gap share 0/0/0/.945; projected reward identical on A and B at d<=3.
CAVEATS: (1) TD-trained FB on P6 at rank 2 converges to the SAME span for every w (reads .714), so the reweighting
sweep moves the closed-form optimum but not the TD fixed point on P6 -> use P11 / rank sweep for the trained story;
(2) DFB at rank 2 fragile on P6 (w->0 limit .38; TD seeds .00-.91), solid from rank 3; (3) arm F plateaus .76.
Paper tex NOT yet updated to the new suite (experiments.tex still cites the old family/rarity figures).
PAPER UPDATED 2026-09-20 ~11:30Z to the single-bottleneck suite (backups *.bak-20260920b): failure.tex = 6-state
figure fig_single_bottleneck_mdp_notail + "What the process shows" paragraph (91/95% mass, gap 96% in direction 3,
Pi_B r identical on A and B at rank 2, FB 0.00 / rank-3 full / softmax .88); experiments.tex didactic subsection =
lemma check (fig_sb_lemma_check_a/_c), rank sweep + readout swap (fig_sb_gap_two_arm, fig_sb_readout_swap), ablation
(fig_sb_ablation_trained), separability text; rarity paragraph -> appendix (app:rare-bottleneck-full, with the
TD-fixed-point caveat); bottleneck-family.tex fully rewritten (generator, fig:family-graphs = P11+R1-R5, gap-direction,
optima+per-mdp, tab:family-rank new numbers, lemma-rewards, separability). Compiles clean (softmax stub). Experiments
intro sentence fixed. PENDING: replace "Isolating the Failure of FB" subsection with the cube reward-view probe
(agent running) as the bridge to manipulation.

★★★ CUBE REWARD-VIEW PROBE DONE 2026-09-20 ~12:15Z (scripts/probes/cube_bottleneck/reward_view.py, note
docs/notes/2026-09-20-cube-reward-view.md, figs PAPER/ICLR/figures/gap_preservation/fig_reward_view_{hist,gaps}):
(1) reward as each critic reads it on 20k dataset states: FB AUC .855 (d' 1.6) vs Density-FB AUC .998 (d' 8.9) --
FB's features blur cube-at-goal states with the rest. (2) three-number test (re-roll with full obs, K=10 H=100,
1125 pairs): FB in-hand true .133 / feature-limited .013 (10%) / critic .004 (3%); carrying .455 / .005 (1%) /
.002; corr(ii,iii) .69 carrying => the lemma's mechanism on cube: FB's gap is the gap of its own reward view, which
has erased it. Density-FB: reward view keeps 34% in-hand / 17% carrying (sign 97/72%); critic gap 4% / 12% with
sign 64/77% (vs FB 65/50%). Magnitudes small for BOTH; the difference is the reward view + sign. Caveats: (ii)/(iii)
scaled to reward units by one LS slope per ckpt; re-roll uses a different MC draw than the action probe (corr .93).
NEXT: replace "Isolating the Failure of FB" with the bridge subsection (compact panels requested from the agent).
FIGURES RESTYLED 2026-09-20 ~13:00Z (user): 6.1 = TWO figures only (lemma-check scatter fig_sb_lemma_check_a with a
rank colorbar; ablation fig_sb_ablation_trained with the legend inside, arm E dropped), both as figure+side-caption
minipages; y axes labelled $\widehat G_r/G_r$ (NOT "action gap" -- failure.tex defines the action-value difference);
rank two-arm + readout swap moved to appendix (fig:bottleneck-appendix), reward-sweep figure dropped (numbers kept in
text). Failure figure = two subfigures (graph; discounted next-state probabilities). 6.2 = "The same failure in
manipulation" with fig_failure_where + fig_reward_view_hist_main + fig_reward_view_gaps_main (figure*). Compiles.
6.1 FINAL LAYOUT 2026-09-20 ~13:40Z: fig 3 = fig_sb_reward_through_features (P11, cluster-A reward through rank-3 vs
rank-4 closed-form span: identical on A and B at rank 3) -- the "why"; fig 4 = ablation (rank axis clamped 2-6).
Lemma-curve figure (fig_sb_lemma_curves, analysis/single_bottleneck/lemma_curves.py) moved to appendix
app:lemma-rewards; scatter fig_sb_lemma_check_a unused. User: the lemma check adds nothing visible beyond the ablation.
6.1 FIG 3 FINAL 2026-09-20 ~14:00Z: fig_sb_values_rank3 (analysis/single_bottleneck/reward_through_features.py [rank]):
P11, TD-trained rank 3, Q(a+)/Q(a-) at the bottleneck: exact 1.67/0, FB .54/.54, Density-FB 1.67/.01 (rank 2: FB
.62/.62, DFB 1.31/.18). User rejected rank-axis versions ("shows FB approximates better with rank"); the message
must be "same rank: FB cannot tell the actions apart, Density-FB can". Label fig:values-rank3.
FIG 3 = TWO SUBFIGURES 2026-09-20 ~14:20Z (fig:bottleneck-rank3): (a) fig_sb_values_rank3 (Q(a+)/Q(a-) exact / FB / DFB
at rank 3), (b) fig_sb_residual_rank3 (analysis/single_bottleneck/residual_panel.py): Delta m split into the part
inside FB's trained span (carries 0.00 of the 1.67 difference) and the residual (carries 1.67); FB reads 0.00;
residual direction appended -> 1.67. User asked for this explicit demonstration of the lemma's core claim.
FIG 3(b) REPLACED 2026-09-20 ~14:40Z: residual bar chart rejected (colours clashed with FB/DFB, "makes no sense");
now fig_sb_fitted_rows_rank3 (analysis/single_bottleneck/fitted_rows_panel.py): true futures after a+/a- vs FB's
fitted futures at rank 3 (closed-form top-3 span) -- fitted rows IDENTICAL (max diff 8e-17). Neutral dark/light
grey for a+/a-, cluster shading A green / B blue. Caption + text updated; compiles.
FIG 3(b) NOW fig_sb_spectrum_rank3 (analysis/single_bottleneck/spectrum_panel.py): per singular direction of nM (P11),
mass share vs share of Delta m; first 3 (kept at rank 3) = 90% mass, 0% of Delta m; direction 4 = 3% mass, 95% of
Delta m. User rejected the fitted-rows panel ("still does not answer why the lemma is true").
FIG 3(b) FINAL = SCHEMATIC 2026-09-20 ~15:00Z (fig_sb_lemma_schematic, analysis/single_bottleneck/lemma_schematic.py):
feature span as a plane, m+ and m- as points whose perpendicular feet coincide, Delta m arrow "outside the span".
User rejected all data versions of "why" (residual bars, fitted rows, spectrum) as making no sense; the geometry
is what they wanted. Spectrum numbers (90% mass / 3% direction) stay in caption + text.
★★ 6.1 FIGURES FINAL 2026-09-20 ~15:40Z (user-specified): fig:values-rank3 (values bars, side caption) + fig:features-lose
(figure*, analysis/single_bottleneck/features_lose_panel.py: trained rank-3 FB B frozen, forward vectors REFIT by
rho-weighted least squares vs exact densities; (a) rho*Delta m vs fitted diff vs projection [both 0], (b) rho*e
left out, (c) G-G_hat = sum rho e r = 1.67 = full G; all 10 runs' features give G_hat = 0) + ablation. Schematic and
spectrum panels dropped from main text. User's terminology note: values figure shows equal action VALUES.
★★ DECISION 2026-09-20 ~16:30Z: ELEVEN-STATE PROCESS EVERYWHERE (user: didactic and experiments must share one
process). failure.tex figure = fig_single_bottleneck_{graph,densities} (entry 2, clusters 2+2, tail 4; bottleneck =
state 2, values 5/3 vs 0); caption/paragraph numbers updated (90% mass in top-3, 95% of Delta m on direction 4,
softmax rank 3 reads .97). Family template fixed to entry [2] cluster [2] tail [4] (random policies); suite rerun in
tmux `sb_suite` (logs/suite/sb_rerun_11state.log) -> refresh 6.1 wording ("five further processes drawn from the
template"), appendix tab:family-rank, fig:family-graphs, numbers. Six-state P6 stays only in the design script.
Also produced fig_sb_features_lose_rank3_dfb (Density-FB rank 3 reads 1.66 on the same plot, sits on the bars).
FAMILY RERUN DONE 2026-09-20 ~17:30Z (eleven-state template, R1-R5 random policies, gaps 1.4-2.5, none rejected):
TD rank-90% medians FB 7 / normsq 7 / softmax+sq 3 / DFB 3 (closed 4/4/2/2); trained means d=2/3/4 FB -.02/.36/.71,
DFB .86/.99/1.00; swap d=2 FB lin .04 softmax .07, DFB lin .15 softmax 1.03; separability FB 0/6 at d=2, 4/6 d=3,
6/6 d>=4, DFB 27/30 d=2, 30/30 d>=3; lemma 600 critics corr .999 sign .995; reward sweep 3660 pairs; residual
cost 12%/5%. P11 values unchanged (FB .54/.54, DFB 1.67/.01). Rarity sweep still on the tail-free 6-state variant
(appendix says so). gap_direction figure now P11 only. Paper text + appendix updated to these numbers.
CHEETAH: tau0025 3 seeds = 407 (vs 488 same seeds round 3) -> NOT a lever; speedcol/velblock rescoring with their
own hit specs (scorer forces the default spec -> shape error; pass EXTRA override) in tmux score_c3.
6.2 CUBE FINAL 2026-09-20 ~19:00Z (user spec): figure = fig_failure_where + fig_alignment_by_phase_main
(scripts/probes/cube_bottleneck/fig_alignment_main.py; phases relabelled approach / maintain(in-hand) /
transport(carrying) with the diagram's colours; cosine FB -.02/.07/.03, DFB -.02/.13/.19; no D_true text). Text =
"does the critic capture how changing the action changes future visits (cube positions)"; direction, not
magnitude. Reward-view hist/gaps + separation figures moved to appendix app:failure-figs (fig:app-separation,
fig:app-reward-view). Compiles.
6.2 FIGURE SIZES 2026-09-20 ~19:20Z: fig_failure_where regenerated at exactly 3.2x2.3 in (failure_plots.py --which a
--figsize 3.2 2.3, tight_layout pad .3, no crop) and fig_alignment_by_phase_main saved without tight crop at the same
size -> equal PDFs, minipages 0.48/0.48. Both bars now wide (width .68), env label "cube-single" kept under bars.
★★★ CUBE INTERVENTION DONE 2026-09-20 ~21:00Z (scripts/probes/cube_bottleneck/intervention.py, note
docs/notes/2026-09-20-cube-intervention.md, figs gap_preservation/fig_intervention_by_phase{,_held}{,_density}):
first action at restored states chosen by (a) FB actor / (b) FB critic argmax / (c) rollout argmax over the same 12
candidates, then the same frozen FB policy; fresh K=10 eval rollouts, H=100, 3 seeds x 192 states/phase. FB success:
approach .36/.36/.38, maintain .38/.39/.44, TRANSPORT .59/.52/.72 (paired (c)-(b) = +.195±.022, every seed). FB's own
critic ranking is WORSE than its actor at transport (-.07). Density-FB at ceiling once held (1.00). => CAUSAL claim
licensed: replacing FB's action values with rollout estimates at one step reduces transport failures by a fifth.
Caveat: rollout rule picks `release` at 65/192 transport states (cube over goal) so held-rate there mixes placing.
NOT yet in the paper text.
⚠ PAPER FILES OVERWRITTEN AGAIN 2026-09-20 14:46-14:48Z (user's sync, hybrid snapshot: their own rewrite of 6.1/6.2/
appendix D, lost my intervention + mean rows + 750.2 walker). I now EDIT ON THEIR COPY (bak: experiments.tex.bak-20260920d).
REVIEW-RESPONSE EDITS 2026-09-20 ~15:10Z: (2) Fig 3 recomputed like Fig 4 -- analysis/single_bottleneck/fig3_family.py:
six settings x 10 runs, values / exact difference: FB .56/.20, DFB 1.01/.02; best fit with FB feats keeps .353 of the
difference (= Fig 4 rank-3 FB .36); fig_sb_values_rank3 + fig_sb_features_lose_rank3{,_dfb} regenerated; text/captions
updated (63%->71%, left/right fixed). (4) rarity: user's appendix already free of it; intro sentences + lem:rare ref fixed.
(7) table caption: no "same architectures"; "hyperparameters selected on a grid (app:hparams)". Restored: intervention
paragraph+figure (fig:intervention), aggregate mean rows, walker 750.2. Compiles. STILL OPEN from the review: (1) two-head
eq. in App B.1 (code = min AFTER integration, line density_fb.py:1149; check eval path), (3) lemma attribution wording in
failure.tex, (5) task-inference paragraph in method + Alg.1 vs B.3 pool, (6) abstract causal wording, (7b) MetaWorld
goal coords 36-38 masking?, eval counts, B.3 pool sweep claim, App D arm definitions, (8) Prop 1 scope + terminal convention.
HEADS RESOLVED 2026-09-20 ~15:30Z: impls/critics/density_fb.py:1149 q_fn = per-head normalized softmax -> integrate reward
per head -> q = min(q1,q2) (or mean); this is the ONLY place heads combine; eval executes the frozen actor (ddpgbc
sample_actions never calls q_fn; q_fn only in the actor loss + probes). App B.1 eq:app-double-q rewritten to
min AFTER integration (reward-shift invariant); B.3 pool text: diagnostics only, deployment = frozen actor; the P-sweep
claim removed. The Aug "double_q min-over-heads sub-normalizes w" note referred to the tabular readout, not q_fn.
FIG 3 = RANK 2 (user decision ~15:50Z): fig_sb_values_rank2 (FB .33/.35, DFB .92/.07, values / exact difference, six
settings x 10 runs) + fig_sb_features_lose_rank2_dfb (best fit with FB feats -.02 on every setting; DFB .86). Text:
"as rank grows FB catches up: rank 3 recovers a third, all on R1/R3/R4 whose policies keep the futures apart longer
(share of |dv|^2 in top-3 span .40/.68/.39 vs 0 on P11/R2/R5); rank 4 most". fig3_family.py takes rank as argv.
REVIEW ITEMS CLOSED 2026-09-20 ~16:10Z: lemma attribution reworded (failure.tex, after the lemma); method.tex: task-inference
+ deployment paragraph (fitted reward r~, actor rankings are for r~, deployment = frozen actor, value readout = diagnostics)
and Prop-1 scope sentences after \end{proposition}; abstract causal sentence -> the intervention claim; protocol: OGBench
10 eps x final 3 evals, MW/MS 50 eps final step; data appendix: MetaWorld goal dims 36-38 (and ManiSkill target dims) ARE
MASKED for every method (obs_mask_dims+goal_mask_dims in launch_density_table.sh / mwms handoff) + datasets have no
terminals; Appendix D: objectives of the six arms + why R1/R3/R4 keep part of the difference at rank 3. All compile.
