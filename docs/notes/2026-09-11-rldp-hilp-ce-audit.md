# RLDP and HILP categorical-objective audit — September 11, 2026

The implemented categorical Bellman update is consistent with normalized successor
prediction in the **continuing setting used by the inspected runs**. That does not
make every adaptation or comparison correct. This audit reproduced and repaired
configuration defects, an actor division by zero, and a contradictory goal label in
object-only HILP. It also repaired the latent Density-RLDP reward-inference
coefficient solve. Some training-task and goal-evaluation mismatches remain.

No training or checkpoint evaluation was run. The repairs do **not** establish why
the existing results are mixed, and do not repair old trained checkpoints or CSVs.
This is a current-source audit with a separate inventory of local historical runs.

## What the critic and actor compute

Plain RLDP fits a linear successor density ratio, `F(s,a,z)ᵀB(u)`, with a squared
Bellman objective. Its backward features receive self-prediction and orthogonality
gradients; the successor term detaches them. Density-RLDP instead makes those scores
logits, normalizes over the minibatch next-state candidates, and fits each head to
`(1-d) one_hot(own_next_state) + d stop_gradient(target_distribution)`.
The actor integrates the candidate reward `B(u)ᵀz` under this distribution.
See [plain RLDP](../../impls/critics/rldp.py) and
[Density-RLDP](../../impls/critics/rldp_density.py).

Plain HILP first learns a Hilbert feature map, then fits successor **feature
vectors** using detached features. Density-HILP replaces the latter regression
with the same categorical construction and integrates `phi(u)ᵀz` for the actor.
The original Hilbert loss still learns the feature map. See
[HILP](../../impls/critics/hilp.py) and
[Density-HILP](../../impls/critics/hilp_density.py).

Consequently, neither retrofit gives categorical loss a direct gradient into its
backward feature geometry. This is intentional preservation of the backbone,
rather than a missing `stop_gradient` repair. It limits transfer of the
end-to-end feature-selection explanation from Density-FB and the tabular figures.

The categorical actor returns normalized reward occupancy rather than the old
linear forward-vector value. The centered actor normalizer is also deliberate:
only the second `q_fn` return is centered, while the first retains the value whose
gradient the actor ascends. The shared Density-FB implementation centers its
**selected minimum-head value**, not the unselected head mean. Centering changes
the weighting relative to behavior cloning; it is not an accidental zeroing of
the greedy objective. Sources: [Density-FB `q_fn`](../../impls/critics/density_fb.py),
[FlowBC](../../impls/actors/flowbc.py), and
[normalizer contract tests](../../tests/impls/test_density_fb.py).

## Reproduced defects and repairs

| Defect | Repair and verification | Relevance to inspected runs |
|---|---|---|
| Density-HILP ignored the campaign temperature names. | Accept campaign and legacy names; reject conflicting values. Nondefault temperature tests failed before the repair. | Nine saved campaign configs used temperature 1 and actor temperature null, which equal the old defaults. This defect does not explain those runs. |
| Density-HILP ignored the inherited minimum-head actor choice. | Select mean/min after reward integration; center the selected value when requested. | Inspected campaign configs request mean, so their configured behavior is unchanged. |
| Density-RLDP ignored the requested online backward target, diagonal-only orthogonality, and scalar actor pessimism. | Read the requested backward snapshot, use the inherited orthogonality helper, and apply pessimism after reward integration. Four real-network contract tests failed before the repair. | These options occur in plain official/control runs, but in no local resolved density run. They are exposed implementation defects, not demonstrated causes of the main comparison. |
| Latent Density-RLDP inherited a reward moment where its actor needs a reward coefficient. | Solve least squares against the actual backward features, then apply the existing index normalization. Preserve the parent encoding in exact-goal mode. | Reward-inference conditions now change for nonwhite features; white-feature behavior is preserved. This applies to the main latent variant, but its performance effect requires checkpoint re-evaluation. |
| FlowBC divided by zero when centered values had zero spread. | Floor the detached denominator at `1e-6`; preserve the old objective above the floor. Constant-value loss/gradient tests now stay finite. | Reproduced in the actual actor loss. No nonfinite numeric entry was found in 118 local baseline training CSVs containing 38,025 logged rows; periodic logs cannot exclude unlogged transients. |
| Object-only HILP compared the full observation with the goal when defining its Hilbert reward. | Compare the same observed input projection used by the feature map. Full-state behavior remains unchanged. | Active in plain and density object-input variants, including the latest exact-goal object-input HILP run. Training must be repeated to measure its effect. |

The last defect is a direct contradiction: two observations with identical objects
and different arm coordinates have identical object-only features, hence zero
feature distance, but previously received reward `-1` and a nonterminal target.
The real-network regression produced loss `0.9999800001` instead of the reached-goal
floor loss `1e-6`. The fix uses observed projected coordinates, **not equality of
learned features**: collapsed features cannot declare different objects reached.
See [dedicated tests](../../tests/impls/test_hilp_goal_predicate.py).

## RLDP's repaired reward inference and remaining training-task distinction

Let `b(u)` be fixed backward features, `C = E_rho[b bᵀ]`, and
`m_r = E_rho[b r]`. At the population forward fixed point of plain RLDP's
measure-regression objective, with the continuation policy fixed,

\[
 C\bigl(F(s,a,z)-\gamma\mathbb E F(s',a',z)\bigr)
 =\mathbb E[b(s')].
\]

Thus its forward factor includes `C^{-1}`: the linear value `Fᵀz` integrates the
projected reward `bᵀ C^{-1} z`. The inherited reward inference sets
`z proportional to m_r`, so the inverse in the forward factor supplies the
least-squares coefficient. This derivation concerns population measure regression;
it is not an assertion of exact neural convergence or a finite-batch identity.

The categorical critic predicts probabilities, and its actor explicitly integrates
`bᵀz`. Previously, reusing `z proportional to m_r` omitted this inverse unless `C=I`
on the relevant feature subspace. Spherical feature norms do not imply whitening;
the inspected main runs use a soft orthogonality coefficient of `0.01`.
The [RLDP audit counterexample](../../analysis/rldp_ce_audit.json) has spherical
features, `C=diag(1.6,0.4)`, and an exactly linear reward whose action ordering is
reversed by the uncorrected categorical reward proxy.

The repaired latent variant solves `lstsq(B, rewards)` before applying the existing
normalization. This recovers the coefficient required by its actual actor reward,
including a minimum-norm solution for deficient feature rank. The regression tests
recover an exactly representable reward and restore the correct action ordering
through the production `q_fn`; they also cover a real backward network, white
features, zero rewards, and zero features. Parent RLDP is unchanged. Exact-goal
Density-RLDP explicitly delegates to the parent's moment encoding because it trains
on raw normalized goal embeddings and integrates an explicit kernel reward.

This repair aligns reward inference with the **existing density actor's task
family**. It does not make paired pretraining indices denote identical rewards:
off-white, plain RLDP's index `z` represents `bᵀ C^{-1}z`, while density's index
represents `bᵀz`. The task sampler and goal mapping were deliberately preserved.
Any claim that the comparison changes only the successor estimator must retain
the whitening assumption or separately align the training-task meanings.

HILP does not have this particular missing inverse: it subtracts a running feature
mean and solves `lstsq(phi, rewards)` before normalizing the coefficient. Density-FB
also uses centered reward moments and a covariance/ridge solve. Positive index
normalization does not remove an anisotropic covariance distortion. The available
RLDP training logs contain orthogonality scalars, not evaluation-pool covariance
spectra; this audit did not demonstrate a covariance-induced ranking reversal in
a saved policy.

## HILP's exact-goal evaluation uses a different index

The exact-goal Density-HILP variant trains with an **absolute whitened goal
feature**, then inherits HILP goal evaluation's **goal-minus-current feature
direction**. These differ even away from the goal. This remains unresolved:
replacing the direction requires specifying the covariance/reference pool used
for the absolute goal index, rather than merely deleting the current-state term.

This mismatch is active on goal-conditioned OGBench evaluation:
[the evaluator](../../impls/utils/evaluation.py) supplies the goal every step and
[the agent](../../impls/agents/__init__.py) calls `map_eval_cond`.
It is bypassed during reward-inference evaluation:
[the training entry](../../impls/main.py) solves the reward index and wraps the
agent in [InferredCondAgent](../../impls/utils/eval_cond.py), which forwards
`goals=None`. The latest local exact-goal HILP CSVs explicitly record
`inferred_cond=1` and intercepted goals, confirming that those reward-inference
scores do not use the relative-goal hook.

The old missing HILP goal hook is a separate, previously repaired defect. Historical
goal-conditioned CSVs generated before that repair remain invalid. Reward-inference
scores cannot be substituted for goal-conditioned scores, or conversely.

## Terminal handling and finite candidate support

Both categorical targets are valid probability rows and use the same task and
candidate support for prediction and bootstrap. An independent tabular adapter
called the **actual stage methods** on a deterministic four-cycle at discount
`0.8`. At the exact continuing Bellman fixed point, both had maximum logit
gradient `2.78e-17`, zero target gradient, and the correct update direction after
perturbing an overpredicted candidate. Terminal-branch targets were nonnegative,
had unit mass, and collapsed to their observed successor. See
[fixed-point check](../../analysis/ce_baseline_audit_fixedpoint.py) and
[results](../../analysis/ce_baseline_audit_fixedpoint.json).

Unit mass alone does **not** establish episodic-return equivalence. With discount
`0.8`, a single terminating arrival reward of 1 has episodic return 1 and globally
normalized value 0.2. The implemented terminal target is a full one-hot, with
integrated value 1; dividing by `1-gamma` yields 5. That branch corresponds to an
absorbing terminal outcome when its reward is retained. A zero-reward cemetery
outcome or an explicit mass/readout treatment is needed for general episodic
rewards. This can change rankings: a one-time terminal reward 1 is worse than
reward 0.3 forever (return 1.5), but these categorical readouts rank 1 above 0.3.
The paper's continuing normalization and its terminal-dependent target
must state compatible assumptions.

For the inspected runs, [FBDataset and WindowedDataset](../../impls/utils/datasets.py)
emit zero terminals, and HILPFutureDataset preserves them. The density losses prefer
that field over masks, so they use scalar `gamma`, as the plain successor losses do.
Goal-dependent masks from GCDataset do not control these configured runs. The
terminal distinction is therefore a mathematical boundary, not evidence of a
termination bug in the current comparison. The finite candidate set is a sampled
approximation to the reference distribution; this check does not prove unbiased
continuous-state population learning at finite batch size.

## Additional sampler defect outside the main density comparison

The optional no-clamp window shifts only future rows backward to fit an episode,
while retaining the original current state, action, and next state. At anchor 8
in a ten-state trajectory with horizon 5, the actual RLDP rollout starts at state 8,
uses actions `[8,5,6,7,8]`, and targets states `[9,6,7,8,9]`. It does not restart
at the shifted window anchor. This violates the explicit future-rollout contract
in `RLDPCritic._sp_pairs` and `_sp_loss`, rather than merely choosing a different
valid sampling distribution. See [actual sampler and downstream check](../../analysis/ce_baseline_audit_contracts.py).

Eight local plain RLDP orthogonality-control configs enable this option; no main
local density config does. It was documented and left unmodified in this audit.

## Provenance and validation

The [inventory](../../analysis/ce_baseline_audit_inventory.json) contains 118 local
HILP/RLDP configs, including historical and discarded variants. None explicitly
records a source commit or source hash. The current config writer saves the resolved
configuration; checkpoints save agent state without a source manifest. A current
source fix cannot be assigned to a historical run solely from its name.

Resolved configurations also disagree with current preset YAMLs. The latest saved
exact-goal HILP cube-single runs used batch 256, four-layer 512-wide features,
the TD-JEPA successor architecture and left encoder, and shared stage parameters.
Those are the same substrate as the saved main HILP campaign. Current exact-goal
YAMLs instead describe an older batch-1024 substrate. Using today's presets to
explain those saved results would introduce a false architectural confound.

Verification artifacts and commands:

- [Prepatch source hashes and training-log scan](../../analysis/ce_baseline_audit_snapshot.json),
  [RLDP prepatch repro](../../analysis/rldp_ce_audit.json), and
  [HILP prepatch repro](../../analysis/hilp_ce_audit_prepatch.json).
  [Postpatch source hashes](../../analysis/ce_baseline_audit_postpatch.json) separately
  identify the repaired code; they do not identify the historical training source.
- RLDP: `pytest -q tests/impls/test_rldp_density_contracts.py tests/impls/test_rldp_density.py tests/impls/test_rldp_seam.py`
  — 59 passed; the four new contracts failed before repair.
- The later latent-inference repair adds 15 cases in
  [test_rldp_density_inference.py](../../tests/impls/test_rldp_density_inference.py).
  Five failed before that repair. The final command
  `pytest -q tests/impls/test_rldp_density_inference.py tests/impls/test_rldp_density_contracts.py tests/impls/test_rldp_density.py`
  passed all 37 cases; see [its log](../../analysis/rldp_ce_audit_inference_green.txt).
- HILP configuration/head contracts: `pytest -q tests/impls/test_hilp_density_contracts.py`
  — 10 passed after their recorded prepatch failures.
- FlowBC: `pytest -q tests/impls/test_flowbc_q_normalizer.py` — 6 passed, including
  a value-dependent scale that distinguishes detached from differentiated normalization.
- Four existing Density-HILP tests and one existing FlowBC test also passed,
  giving 21 focused HILP-density/FlowBC checks alongside the separate predicate suite.
- HILP goal predicate and parity: `pytest -q tests/impls/test_hilp_goal_predicate.py tests/impls/test_hilp_seam.py tests/impls/test_hilp_oc.py`
  — 28 passed; the two object-goal regressions failed before repair.
- `python analysis/ce_baseline_audit_fixedpoint.py` and
  `python analysis/ce_baseline_audit_contracts.py` — actual-stage and actual-sampler
  checks described above. All numerical checks used CPU; tests enabled float64.

The local virtual environment is `/home/adityamohan/git/Factored-FB/.venv/bin/python`.
For this audit, missing import paths were supplied from the existing uv package
cache, without installing dependencies. Test logs reside under
`analysis/{rldp_ce_audit*,hilp_ce_*,ce_baseline_audit*}`; one existing TensorFlow
Probability/JAX deprecation warning appeared in the relevant runs.

Useful next measurements are fixed-checkpoint reward-projection residuals and
action rankings under the competing RLDP readouts; matched HILP exact-goal index
construction across training and evaluation; and new training after the object-goal
label repair. These are needed before attributing the mixed benchmark scores to
the defects, the categorical objective, or hyperparameters.
