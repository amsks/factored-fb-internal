# Section 6.1 on the four-state bottleneck MDP -- grow-with-d (2026-09-21)

The didactic figure is the bare four-state bottleneck; the ablation is grow-with-`d`, the faithful
illustration of Lemma 1.

## Why grow-with-d (not a fixed MDP)
Four states = two futures (states 2, 3), so the action-difference is the **second** singular direction
of the successor measure and FB recovers it at rank 2 -- on any fixed finite MDP FB is fit exactly once
the rank reaches its successor rank, so no fixed MDP fails at every rank. Lemma 1 states "for every d,
**there exists** an MDP": the MDP depends on d. At feature dimension `d` it is the four-state bottleneck
plus **(d-1) disconnected components**, each a large future-visit direction that outranks the difference
and pushes it to position d+1. FB then reads none of the gap at every d; Density-FB reads it all.

Adding a component does not touch the bottleneck (state 1, `a_+`->rewarded state 2, `a_-`->state 3,
reward only at 2). A component is a sink: an absorbing state fed by source states. The written proof
uses 5-state components (1 absorbing + 4 others); a minimal **2-state** sink (1 absorbing + 1 source)
gives the identical ablation curve and pushes FB's failure one rank further per sink -- `mdp.py` uses the
proof's 5-state components (`COMPONENT_SIZE = 5`); change it to 2 to match a simplified proof.

## Numbers (`four_state_summary.json`, closed / TD-trained)
Recovered fraction of the action gap, `d = 1..6` (MDP has n = 4, 9, 14, 19, 24, 29 states):

| arm | d=1 | d=2 | d=3 | d=4 | d=5 | d=6 |
|-----|----|----|----|----|----|----|
| FB (A) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| FB, normalized target (B) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| softmax + squared error (C) | 0.69 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Density-FB (D) | 0.69 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

`d=1` is the bare four-state (0 components); with a uniform policy states 2 and 3 are exactly
exchangeable, so the rank-1 softmax fit is ambiguous and Density-FB only reads 0.69 there (wide seed
spread). From `d=2` on, FB is flat 0 and Density-FB flat 1. Start the sweep at `d=2` for a perfectly flat
figure; keep `d=1` to show the degenerate bare bottleneck.

- Values at the bottleneck, `d=2` (exact pair mapped to 1, 0): FB **0.50 / 0.50** (indistinguishable),
  Density-FB **1.00 / 0.00**, exact 1.00 / 0.00.
- Frozen-feature refit (Lemma), `d=2`: the projection onto FB's rank-2 features reads **0.000** of the
  true difference of 5 at the rewarded state; Density-FB reads **5.000**.

## Two ablations (2026-09-21, user asked for both -- "we will figure out how to explain this")
The grow-with-d ablation needs states not in the four-state picture (the "magic states" problem). So we
also generate the ablation on ONE fixed connected MDP that is drawn whole, and keep both:

1. **Fixed connected MDP** (`mdp.make_connected_mdp`, entry 1 / cluster 2 / tail 6, n=12, gap 1.67):
   a bottleneck whose two actions enter clusters A/B that then merge into a shared closed tail -- the
   physically faithful "decide, then flow into a shared continuation." Everything is in the figure
   `fig_connected_graph`. Rank sweep 1..8 (closed): FB fails ranks 1-3, catches up at rank 4 (reads
   1.04); Density-FB recovers by rank 2 (0.21 at rank1, 0.99 rank2, 1.00 rank3); softmax+sq by rank 3.
   TD-trained: FB catches up rank 4-5, Density-FB by rank 3. Figures `fig_fs_ablation_fixed_{closed,
   trained}`. This is the coherent single-MDP story (= the old 11-state ablation, drawable).
2. **Grow-with-d** (Lemma 1, MDP = base + (d-1) components): FB flat 0 at every d, Density-FB 1 (0.69 at
   d=1). Figures `fig_fs_ablation_{closed,trained}`. Matches the lemma but the MDP is not fully shown.

## Figures (`PAPER/ICLR/figures/four-state-mdp/`, all 4.6x3.4 in, ICLR fonts)
- `fig_four_state_graph`, `fig_four_state_densities`, `fig_four_state_mdp` (combined) -- the bare
  four-state bottleneck (Section 4 / failure.tex).
- `fig_connected_graph` -- the 12-state connected shared-tail MDP used by the fixed ablation, drawn whole.
- `fig_fs_values` -- action values at the bottleneck at `d=2` (grow-with-d).
- `fig_fs_features_lose` -- the projection identity at `d=2` (FB reads 0, Density-FB reads all 5).
- `fig_fs_ablation_fixed_closed` / `_trained` -- fixed connected MDP, rank sweep 1..8.
- `fig_fs_ablation_closed` / `_trained` -- grow-with-d, sweep over d.

## Rebuild
    JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/mpl /dev/shm/.venv-jax/bin/python analysis/four_state_mdp/run.py       # ~2 min
    JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/mpl /dev/shm/.venv-jax/bin/python analysis/four_state_mdp/plot_mdp.py
    ... run.py --figures-only     # figures from the CSVs

## Drop-in LaTeX
### failure.tex (the process figure -- the bare four-state bottleneck)
```latex
\includegraphics[width=\linewidth]{ICLR/figures/four-state-mdp/fig_four_state_mdp.pdf}   % or _graph / _densities
```
Bottleneck is state 1; `a_+`->rewarded absorbing state 2, `a_-`->absorbing state 3; gap `1/(1-\gamma)=5`.
Text should say: for feature dimension `d`, the lemma's MDP is this bottleneck plus `d-1` disconnected
components.

### experiments.tex 6.1
```latex
\includegraphics[width=\linewidth]{ICLR/figures/four-state-mdp/fig_fs_values.pdf}
\includegraphics[width=\linewidth]{ICLR/figures/four-state-mdp/fig_fs_features_lose.pdf}
\includegraphics[width=\linewidth]{ICLR/figures/four-state-mdp/fig_fs_ablation_closed.pdf}
```
- values / features-lose (`d=2`): "FB assigns both actions the same value 0.50 and its best fit reads 0
  of the difference at the rewarded state, while Density-FB recovers it in full."
- ablation: "FB reads none of the action gap at every feature dimension, and normalizing its target does
  not help; the softmax fits recover it in full. The softmax parameterization -- not more capacity --
  preserves the action-value difference."
