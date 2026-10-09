---
name: four-state-mdp-61-20260921
description: "2026-09-21 built the 4-state bottleneck MDP + regenerated all 6.1 tabular figures on it (replaces 11-state); FINAL = bare 4-state, rank sweep 1-4, FB fails only at rank 1; ICLR font/size pass done"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0e064614-b179-4c70-8a8c-8837735a38b6
  modified: 2026-09-21T16:23:04.351Z
---

FINAL 2026-09-21 = GROW-WITH-D (converged here after a long exchange). Didactic figure = the BARE
4-state bottleneck; ablation/values/features = grow-with-d (Lemma 1). KEY FACT settled with the user: no
FIXED finite MDP fails at every rank -- a 4-state MDP has 2 futures so the action-difference is the 2nd
singular direction and FB ALWAYS recovers at rank 2 (verified: family of random 4-state MDPs still has FB
rank2=1.00; adding K sinks only delays catch-up to rank K+2). Lemma 1 is "for every d THERE EXISTS an
MDP" => MDP depends on d: at dim d it is base + (d-1) disconnected sink components; FB flat 0, Density-FB 1
at every d. A sink = absorbing state + source(s); proof uses 5-state components (COMPONENT_SIZE=5 in
mdp.py), minimal 2-state sink (1 abs+1 src) gives identical curve (+2 states = +1 failing rank).
run.py: DIMS=[1..6], VALUES_DIM=2. Numbers (closed/TD): FB 0.00 every d (TD noisy ~0); Density-FB 0.69/0.95
at d=1 then 1.00 (d=1 = degenerate bare 4-state, uniform policy symmetric, seed-noisy -- start at d=2 for a
flat fig). Values d=2: FB 0.50/0.50, Density 1.00/0.00. Features-lose d=2: FB proj 0.000, Density 5.000.
Soft-target arm F REMOVED (user). ICLR pass: figs 4.6x3.4in (1380x1020), big fonts, ablation y-label TEXT
"recovered fraction of the action gap", features-lose dots s=95.
TWO ABLATIONS now generated (user hit the "magic states" coherence problem: grow-with-d adds states not in
the 4-state figure). (1) FIXED CONNECTED MDP mdp.make_connected_mdp (entry1/cluster2/tail6, n=12, shared
tail, DRAWN WHOLE in fig_connected_graph): rank sweep 1..8, closed FB fails ranks1-3 catches up rank4
(1.04), Density-FB 0.21/0.99/1.0 by rank2 -> figs fig_fs_ablation_fixed_{closed,trained}; this is the
coherent single-MDP story (= old 11-state). (2) GROW-WITH-D unchanged (fig_fs_ablation_{closed,trained}).
User will decide how to present both. Key impossibility restated to user: can't have all of {show whole
MDP, ablate on it, FB fails every rank} -- pick 2. run.py builds both; FIXED_RANKS=[1..8], DIMS=[1..6].
Everything below is earlier-iteration detail (proof-exact d=2..6, then a bare-4-state rank-sweep detour),
kept for reference.

--- (superseded) User first asked to replace the eleven-state process in 6.1 with the four-state
construction, so figure and proof share one process. INITIALLY chose
proof-exact rank axis (at feature dim d the process = base + (d-1) added components; FB flat 0,
Density-FB flat 1 -- two flat lines, the literal theorem, NOT the old recover-with-rank curve) and
asked that all new figures live in a NEW folder `four-state-mdp`.

Construction (base states 0,1,2,3, gamma 0.8): state 0 -> {1 w.p 1/2, 2 and 3 w.p 1/4}; bottleneck =
state 1, a_+ (action 0) -> rewarded absorbing state 2, a_- -> absorbing state 3; reward = e_2; policy
1/2. Gap = 1/(1-gamma) = 5. States 2,3 are exchange-symmetric (Perron-Frobenius), so the difference
direction e2-e3 is the SECOND base singular direction (unscaled sv^2 95.4 / 75.0 / 0.39); each added
component (10 state-action pairs, one point mass, sv^2 = 10 x scale) outranks it, so a rank-d SVD keeps
d-1 components + base top-symmetric and drops the difference -> FB reads 0 at EVERY d.

VERIFIED numbers (closed form = the optimum the theorems describe): FB 0.000 and FB-normalized-target
0.000 at d=2..6; softmax+sq and Density-FB 1.000; Density-FB soft-target 0.79. Values at bottleneck
(exact pair -> 1,0): FB 0.50/0.50 (indistinguishable), Density-FB 1.00/0.00. Frozen-feature refit
(Lemma): projection onto FB's rank-d features reads 0.000 of the true difference 5 at the rewarded
state, Density-FB reads 5.000. TD-trained (5 seeds x 2 rates, 50k) agrees: FB near 0 (a couple of d=2
seeds escape to ~0.2), Density-FB/softmax+sq 1.0 -- so the paper uses the closed-form ablation.

Code (all NEW, uncommitted): analysis/four_state_mdp/{mdp.py, run.py, plot_mdp.py, NOTES.md} +
four_state_{closed,trained}.csv, four_state_summary.json. Reuses analysis/toy_ablation/family.py arms
A-F and scripts/didactic/bottleneck_family_td.py exact(). Figures in
PAPER/ICLR/figures/four-state-mdp/: fig_four_state_{graph,densities}, fig_fs_{values,features_lose,
ablation_closed,ablation_trained}. Rebuild: `.../python analysis/four_state_mdp/run.py` (~2.5 min);
`--figures-only` from CSVs; `plot_mdp.py` for the didactic figure.

DID NOT edit failure.tex / experiments.tex (user edits them live -- see
[[failure-tex-vs-tabular-20260920]]); NOTES.md carries drop-in LaTeX + the numbers. STORY CHANGE the
user must apply: the ablation no longer says "FB recovers with more features / softmax needs fewer" --
it now says FB fails at every feature dimension, isolating the softmax parameterization (not capacity).
Also change failure.tex wording: bottleneck is state 1, clusters A/B -> single absorbing states 2/3.
