# Carry-or-drop diagnostic implementation plan

> **For agentic workers:** Use superpowers:executing-plans to execute inline. The user has authorized the experiment; no additional execution approval is needed.

**Goal:** Test whether full-capacity bilinear FB loses control-relevant action differences as resting persistence increases.

**Architecture:** A continuing finite MDP with a chain of held states, a goal, and a resting state. Exact linear algebra supplies the reference; tabular JAX critics learn a fixed uniform continuation policy using full-support TD targets. Results and plots are local analysis artifacts, separate from production agents.

**Tech Stack:** Python, NumPy, JAX, Optax, Matplotlib, unittest.

**Spec:** User-approved design in this conversation; precise contract below.

## Global constraints / experimental contract

- Depth L in {1,3,6}; states H0,...,H(L-1),G,R; actions advance/drop.
- Advance succeeds to the next held state (or G) with probability .9, otherwise R. Drop goes to R. G resets to H0. R resets to H0 with probability 1/k, otherwise stays. Both actions coincide at G and R.
- k in {1,10,50}; gamma=.98; reward is 1 on arrival at G.
- Uniform continuation actions; uniform input state-action weights and uniform candidate reference. The intervention changes the true transition kernel, so compare each run with its own exact solution.
- Dimension equals state count. Explicitly construct exact FB and density parameters and check Bellman residuals and action advantages.
- Main methods: FB with spherical B and orthogonality coefficient 1; same FB without orthogonality; density softmax with unconstrained B; normalized squared bilinear control; direct unnormalized successor-density-ratio table.
- Single head, no actor or task conditioning, complete unique candidate set: this is a critic diagnostic, not a reproduction of the whole production algorithm.
- First use expected transition targets to remove sampling noise; then balanced sampled transitions if useful. Sampled runs share random draws across methods.
- Adam, target EMA .005, 100,000 updates, seeds 0/1/2. Start with learning rates .001/.01 and retain every run. No selection based on a desired result.
- Report action-gap ratios, wrong held-state decisions, state-resolved successor error, mass error, exact evaluation of a single greedy policy improvement, and learning curves. No claim of learned control from critic error alone.
- Preserve all prior files, results and paper content. No commits or remote jobs.

## Task 1: Mathematical model and validation

Files: create `scripts/didactic/carry_drop.py`, `scripts/didactic/test_carry_drop.py`.

Interfaces: `build_mdp(depth, persistence)` returns P[state,action,next]; `exact_successor(P,gamma,policy=None)` returns M[state,action,next]; `exact_parameters(M,method,gamma)` returns a dict with F/B or table; `predict(params,method)` returns raw ratio for FB/direct, probability for density/normalized squared.

- [x] Write independent tests for the one-state transport case: at k=1, immediate goal probability under advance is .9, under drop 0, both reset states go to H0; exact first-action advantage is .9 for every gamma because G and R have identical continuation.
- [x] Run tests before implementation and confirm failure from missing functionality.
- [x] Implement transition kernel and exact successor solve `P_flat @ inv(I-gamma*P_policy)`.
- [x] Test exact parameterizations for every method using `M = P + gamma * P @ mean_action(M)` and probability row mass 1.

## Task 2: Training and diagnostic metrics

Interfaces: `loss_fn(params,target,P,method,gamma)` implements the expected TD semi-gradient; `train_group` batches conditions/seeds/rates of equal state count, saving predictions every 1000 updates. `metrics` consumes learned normalized successors and exact M.

- [x] Test zero gradients at analytically exact parameters, including the FB orthogonality term; test a wrong goal column is detected by gap and greedy-policy metrics.
- [x] Implement the expected FB loss `.5 mean((F B^T - gamma P M_target)^2) - mean(sum(P * F B^T))`; constant target-only terms do not affect its gradient. For normalized squared replace immediate coefficient by `(1-gamma)/S`; density uses cross-entropy against `(1-gamma)P + gamma P p_target`.
- [x] Run a short numerical pilot and check finite values, valid density mass and improving reference errors.
- [x] Run the declared full sweep. If a separation occurs, examine learning-rate dependence and sampled transitions before making a mechanism claim.

## Task 3: Artifacts and interpretation

Files: `analysis/carry_drop/` for configuration, numerical references, checkpoints, curves and summary; `PAPER/ICLR/figures/didactic/carry_drop/` for graph and plots; `docs/notes/2026-09-10-carry-drop-results.md` for durable findings.

- [x] Plot the actual MDP and per-depth persistence sweeps with all seeds, plus training curves.
- [x] Run the mathematical tests, inspect plots and verify reported numbers against saved tables.
- [x] Record null results as null results; distinguish failure of this proposed example from conclusions about cube.

## Execution record

Completed inline in the existing linked worktree on density-fb. Mathematical tests: 5 passed. Main expected and sampled sweeps: 270 runs each, all 100k updates. All saved main-method decisions and gap metrics were cross-checked against NPZ arrays. See docs/notes/2026-09-10-carry-drop-results.md for the null final result, temporary expected-update effect, and control outcomes. No actor stage was launched because the proposed main-method separation did not survive the declared training and sampling controls. No commit or integration action was taken.
