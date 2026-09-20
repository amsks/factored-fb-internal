# 2026-09-20 -- Validating the fixed-features lemma of failure.tex on the generated bottleneck processes

Code `analysis/lemma_check/run.py` (tests `tests/probes/test_lemma_check.py`); data
`analysis/lemma_check/{results.npz, summary.json, run_log.txt}`; figure
`PAPER/ICLR/figures/lemma_check/fig_lemma_check.{pdf,png}` (+ `_a`, `_b`, `_c` panels).
Reuses the family generator and FB trainer of `analysis/toy_ablation/family.py` (five
single-cut processes, discount 0.8, uniform reference 1/n, TD harness: bootstrapped target,
EMA 0.005, Adam, 50k steps, 5 seeds x 2 learning rates, orthogonality penalty at 1).

## The lemma

With the backward features fixed (U = column space of B), the least-squares fit of every row of
the density ratio X = nM is its projection onto U, so at the decision state
gap_hat(r) = dv^T P_U r / n with dv = X[i0] - X[i1], and the discarded gap is
(dv - P_U dv)^T r / n. Three tests, all exact population quantities.

## 1. Trained features (panel a)

For every process, rank 2..9 and run, the gap FB reads with its trained F,
(F[i0] - F[i1]) B^T r / n, against the lemma's prediction computed from the trained B alone.
400 runs: correlation 0.994, sign agreement 0.998, mean |difference| 0.020, worst 0.40 (one
rank-5 run on process 5). Per rank (mean lemma | mean read | worst |diff|):
d=2 .048|.048|.0005, d=3 .121|.120|.17, d=4 .347|.376|.21, d=5 .676|.703|.40,
d=6 .843|.862|.28, d=7 .955|.969|.09, d=8 .991|.998|.06, d=9 1|1|.0002.
The residual is the TD fixed point's distance from the row-wise least-squares optimum given
B; it is largest at the ranks where the gap is half-kept and vanishes at both ends.
Closed-form SVD fits sit on the diagonal to 1e-9 (asserted).

## 2. Reward sweep at rank 3 (panel b)

Same trained B, unit reward on each state and 50 random rewards: 2950 (reward, run) pairs,
correlation 0.997, worst |lemma - read| 0.21. With the same features the kept fraction
ranges over rewards from -0.10 to +3.7 (process 4, reward at state 5, true gap 0.11): the
loss depends on r only through its inner product with the discarded residual, as the lemma
says, and the goal reward is not special.

## 3. Appending the discarded direction (panel c)

Add the unit residual dv - P_U dv to the span and refit. Gap fraction 1.000 at every rank on
every process, for every reward (max error 9e-16 over the 2950 pairs; asserted for trained B
too). Cost, closed-form B: the augmented rank-(d+1) fit's objective is above the optimal
rank-(d+1) fit by 28% (d=2), 11% (3), 11% (4), 11% (5), 5% (6), 4% (7), 0 (8, 9), while FB's
own optimal rank-d fit keeps .09 / .22 / .49 / .93 / .97 / .97 / 1.00 / 1.00 of the gap.

## Reading

The lemma holds as stated: the fitted gap is the projection of the row difference, for
trained as well as closed-form features, and for every reward. What it does NOT say is which
span training selects; the rarity and rank results (rare-bottleneck note, family note) are
about that choice and need the subspace-selection step (Gram convergence) that is currently
only in the appendix proof of the undefined `lem:rare-bottleneck`.

## Reproduce

```bash
cd /home/adityamohan/git/Factored-FB-seam
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/mpl /dev/shm/.venv-jax/bin/python analysis/lemma_check/run.py            # ~70 s CPU
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/mpl /dev/shm/.venv-jax/bin/python analysis/lemma_check/run.py --figures-only
JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python -m pytest tests/probes/test_lemma_check.py -q
```
