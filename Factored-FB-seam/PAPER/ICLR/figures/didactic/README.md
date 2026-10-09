# Three-location placement MDP

`placement_mdp.png` is the preview. `placement_mdp.pdf` and
`placement_mdp.svg` are vector exports for the paper.

There are exactly three states: the object is at Left, Centre, or Right.
Each state has all three actions: place left, place centre, and place right.
Each panel displays the complete transition graph for one action; the panels
are not separate environments. Colored nodes mark the action's intended
destination, not a fixed reward or goal.

The transition first keeps the current state with probability 0.9. Otherwise,
it draws a placement outcome: intended destination with probability 0.8,
or either other destination with probability 0.1 each. Thus the **total**
transition probabilities drawn on the graph are:

- Command the current location: stay with 0.98; move to each other state with 0.01.
- Command another location: stay with 0.91; reach the intended location with
  0.08; reach the third location with 0.01.

All states are continuing, with no terminal state. No reward is fixed in the
diagram; downstream tasks assign rewards to the three locations. The proposed
analysis uses discount 0.98. Uniform random actions give a uniform stationary
state distribution and an expected 15 consecutive steps at a location.

Exact transition matrices are in `transitions.json`, indexed by action,
current state, then next state. `draw_mdp.py` generates all files and checks
the probability rows and the uniform stationary distribution.

```bash
MPLCONFIGDIR=/tmp/didactic-mpl python draw_mdp.py
```

This figure depicts the proposed environment. It contains no trained results
and makes no assertion that TD training has already demonstrated a separation
between FB and Density-FB.
