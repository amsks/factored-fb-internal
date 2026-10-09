# Four-state deterministic path

The main figure is [mdp.png](mdp.png), with vector exports [mdp.pdf](mdp.pdf)
and [mdp.svg](mdp.svg). The optional comparison is [values.png](values.png),
also exported as [PDF](values.pdf) and [SVG](values.svg).

Each circle is a state. Left and Right are deterministic actions, labeled on
the edges. At an endpoint the outward action stays at that endpoint. The agent
starts at state 0 and receives reward +1 whenever it enters state 1, the green
Goal. Other arrivals pay zero. **Goal is continuing:** its actions lead back to
state 0 or onward to state 2. The graph displays no transition probabilities.

## What the value panel compares

The continuation policy chooses Left and Right uniformly at every state.
The discount is 0.8. Training rows weight all eight state-action pairs equally;
their natural next-state reference is uniform over the four states. Both models
have feature dimension two and unrestricted backward output norms. Whitening
the FB backward second moment is compatible with the reported squared optimum.
Requiring every backward feature to have the same norm is a different constraint.

The critic predicts discounted future-state occupancy under the fixed
continuation policy. All displayed values are **normalized** goal occupancies:
the discounted expected reward multiplied by `1 - gamma = 0.2`. They are not
single-step success probabilities or returns under the subsequently greedy
policy. A greedy decision maximizes this predicted reward integral.

The three labels distinguish an exact target from two approximations:

- **Exact successor:** the analytically computed target.
- **FB: best squared fit:** its unique global rank-two squared-loss approximation.
- **Density: feasible fit:** an explicit softmax model with rank-two logits.
  Its numerical values are not asserted to be the global cross-entropy optimum.

The density factors, target matrix, predictions, and certificate are recorded in
[verification_data.json](verification_data.json). The generator recomputes every
displayed value and checks the certificate before exporting the comparison.

## Exact-target claim and capacity check

The normalized successor rows, indexed by the state entered by the first action,
are

```text
        1 / 85 * [[43, 22, 12,  8],
                  [22, 33, 18, 12],
                  [12, 18, 33, 22],
                  [ 8, 12, 22, 43]].
```

The unique best squared fit retains the constant and first nonconstant path
eigenvectors. At Start it predicts Left = 0.33140491 and Right = 0.28371902,
reversing the true ordering 0.25882353 < 0.38823529. Its entries are all positive
and its rows sum to one. Greedy execution of this fit stays at Start forever
and earns zero.

Dimension two is enough for this reward's control information. Given the target
matrix `N`, define `C[i,1] = N[i,1]` and `C[i,j] = (1 - N[i,1]) / 3` for all
other columns. This is a positive rank-two probability matrix and preserves
every action value for the displayed goal reward. Preservation of all possible
rewards is not claimed.

The explicit categorical candidate uses these rational factors:

```text
B = [[-1,  1], [-1/2, -1], [1/2, -1], [1, 1]]
F = [[-4/5, 1/10], [-2/5, -23/100], [2/5, -23/100], [4/5, 1/10]]
q = softmax(F @ B.T), independently within each row.
```

Its mean forward KL is 0.00330419925761. At every global categorical optimum,
duplicate state-action rows with the same exact target must predict identically:
for fixed backward features their forward rows independently minimize the same
strictly convex loss in identifiable logits. Their weights therefore combine
from 1/8 to 1/4. Aggregating Goal versus non-Goal gives lower bounds of
0.00480689816708 and 0.02019950996663 for any wrong or tied decision at states
0 and 2, respectively. The feasible KL is smaller than both. Consequently every
global categorical optimum enters Goal from these states. Either action at Goal
then returns through one of them, achieving the optimal start-state return
`1 / (1 - gamma**2) = 25/9`.

Both classes misspecify the full successor distribution: the target has rank
four, while its row-centered log-probability matrix has rank three. The example
compares a linear rank-two density and a normalized exponential rank-two score
model. It does not isolate cross-entropy from the change in model class.

## Separate TD evidence

The plotted bars concern fitting exact successor targets. With temporal-difference
training, FB instead regresses against an immediate successor term plus a
discounted frozen successor estimate; the density critic uses cross-entropy
against the corresponding normalized mixture. These are different fitting
procedures, and the figure makes no global TD-optimum claim.

The [TD report](../../../../../docs/notes/2026-09-11-deterministic-path-td.md)
describes separate full-expectation joint-forward/backward updates on this graph.
At discount 0.8, all six FB runs selected the wrong start action and all six
density runs selected the correct action at every state: three seeds crossed
with two learning rates, 100,000 updates per run. FB used unrestricted backward
norms and a second-moment penalty; density used unrestricted backward norms.
There was no learned actor and no reward-inference evaluation. Restoring
spherical FB features recovered the start action in two of six control runs.
This recovery prevents extending the main result to the production spherical
constraint. The displayed density candidate is also distinct from those TD fits.

## Reproduction and export

From the repository root, using Python with NumPy and Matplotlib:

```bash
/home/adityamohan/git/Factored-FB/.venv/bin/python \
  PAPER/ICLR/figures/didactic/deterministic_path/draw.py
```

Use `--graph-only` to render just the MDP. The generator does not train or fit
parameters. PNGs have a white background; PDFs retain vectors and embed fonts;
SVGs retain text. [caption.tex](caption.tex) supplies a caption for the graph.
No paper source file is modified. This directory is inside the ignored
`PAPER/` tree, so a clean Git status does not imply these local assets are tracked.
