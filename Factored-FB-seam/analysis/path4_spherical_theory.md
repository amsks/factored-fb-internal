# Exact certificate for spherical FB on the same four-state path

This extends the exact-target path4 construction to spherical backward features
and the actual soft orthogonality penalty at the stronger setting used by the
modern table configuration. It does not establish a global TD result.

Let `N` be the existing normalized successor matrix, `A=N^2`, and constrain each
row of the 4-by-2 backward matrix `B` to norm `sqrt(2)`. Write

```
C = B^T B / 4,  eigenvalues(C) = 1 +/- delta,
Pi = B (B^T B)^(-1) B^T,
E = ||N - N Pi||_F^2.
```

Every global optimum with full-rank B optimizes its forward factors exactly,
so its normalized prediction is `N Pi`. Two exact-target FB heads at discount
4/5 contribute `25 E`; the actual orthogonality excess is
`lambda/2 ||C-I||_F^2 = lambda delta^2`. Thus at `lambda=1000`, division by 25
gives `J=E+40 delta^2`. One head with `lambda=500` has the same relative objective.

A rational spherical, exactly whitened candidate is `B=sqrt(2) U` with

```
U = [[1,0], [255/257,32/257], [0,1], [-32/257,255/257]].
```

Every row of U has unit norm and `U^T U=2I`. Its exact error is below `.081`.
Consequently every global minimizer has `J<.081`, hence `delta<.05` because
`40*(.05)^2=.1`. In particular B has full rank, and

```
|Pi_ii - 1/2| < 1/38.
```

This follows from `Pi_ii=(1/4) B_i^T C^(-1) B_i`, the row-norm constraint, and
the eigenvalues of C. The rank-deficient boundary costs at least 40, so it
cannot contain a minimizer. The bounded sphere domain, uniformly nonsingular
C on this sublevel, and continuous eliminated-forward objective imply that
a global optimum exists.

Define `d=N[1,:]-N[0,:]` and `D=(d e_1^T + e_1 d^T)/2`. The normalized goal
action gap, right minus left from state 0, is `g=tr(D Pi)`. Use the rational
constants

```
mu = 43/50,
z = (-1/10, -1/20, 17/100, 0),
beta = 13/100,
v = (357,469,573,478)/1000.
```

The matrix inequality

```
H = A + mu D + diag(z) <= beta I + v v^T
```

has an exact rational certificate: all 15 principal minors of the difference
are strictly positive. The verification script computes each with Fraction.
For any rank-two orthogonal projector Pi,

```
tr(H Pi) <= 2 beta + ||v||^2.
```

Therefore on the global-minimum sublevel,

```
E >= tr(A) - 2 beta - ||v||^2 + sum(z)/2 - ||z||_1/38 + mu g
  = 0.105729649790566... + 0.86 g.
```

Since `E <= J < .081`, every global minimizer has

```
g < (.081 - .105729649790566...)/.86 < -.02875 < 0.
```

Every global spherical FB optimum therefore stays at state 0 forever. This is
not a selected numerical minimum or a claim that all TD trajectories find it.

The same spherical class can predict this goal's values exactly: use backward
rows `(1,1),(1,1),(1,-1),(1,-1)` and forward row `(1,4N_i1-1)`. Its normalized
prediction has rows `(N_i1,N_i1,1/2-N_i1,1/2-N_i1)`, which are positive and sum
to one; the backward second moment is I. Its existence separates control
capacity from objective preference.

The existing explicit rank-two Density candidate and KL certificate are
unchanged. They certify optimal return from start 0 for every global
categorical optimum. Density's backward outputs are unrestricted, matching
that method's default; the comparison is not loss-only. Both target classes
remain misspecified for the full successor distribution.

Run `../Factored-FB/.venv/bin/python analysis/path4_spherical_theory.py`.
