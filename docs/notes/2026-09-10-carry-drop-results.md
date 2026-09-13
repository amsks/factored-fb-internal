# Full-capacity carry-or-drop experiment

Date: 2026-09-10. Local CPU experiment; production agents and paper text unchanged.

## Verdict

**This candidate does not establish a robust FB control failure.** FB and Density
FB both select advance at every held state after 100,000 updates in every tested
environment, seed and learning rate, with both expected and sampled TD updates.
Their greedy policies achieve the exact optimal return. This is a result about
this controlled toy, not a resolution of the failures observed on cube.

There is a temporary failure of full-capacity FB with expected updates and the
smaller learning rate. It is removed by more updates, a larger learning rate, or
(in the longest-rest condition at the inspected checkpoint) transition sampling.
The evidence therefore supports a limited optimization-speed observation, not an
unavoidable representational failure or a categorical-loss-specific cure.

## Protocol

Source and reproduction instructions: [scripts/didactic/README.md](../../scripts/didactic/README.md).
The user-approved design is recorded in
[the implementation plan](../superpowers/plans/2026-09-10-carry-drop.md).

- L={1,3,6} consecutive held-state decisions, followed by goal G; another state R
  represents the dropped object. Advance progresses with probability .9 and
  reaches R with probability .1; deliberate drop always reaches R.
- G resets to H0. R returns to H0 with probability 1/k, k={1,10,50}. All states
  continue; both actions coincide at G and R. Reward is 1 on arrival at G.
- Discount .98; fixed uniform continuation policy; latent dimension L+2 equals
  the number of states. Uniform state-action weights and uniform reference over
  the complete distinct state set eliminate visitation and candidate starvation.
- Single-head tabular FB (spherical B, orthogonality coefficient 1), Density FB,
  FB without orthogonality, normalized squared bilinear TD, and a free squared
  successor table. See the README for exact objective and normalization details.
- Adam rates .001 and .01; target EMA .005; seeds 0,1,2; 100,000 updates.
- Expected updates average over the next state exactly but still bootstrap from
  learned target networks. They are **not** supervised fits to exact successors.
- Sampled updates draw 16 next states independently per state-action per update;
  all methods share the draws. Continuation actions remain averaged exactly.
  This is a generative sampler, not a finite offline dataset.

Each sweep has 54 runs per method: 3 depths × 3 persistence levels × 3 seeds ×
2 rates. Five methods and two sweeps give 540 runs, in addition to a short pilot.
No rate or seed was selected to favor a method.

## Capacity and control validation

For each MDP, compute the exact successor measure
`M=P_flat (I-gamma P_pi)^(-1)`, with next-state reward convention. Exact FB
parameters are `B=sqrt(N) I, F=sqrt(N) M`; exact density parameters are
`B=I, F=log((1-gamma) M)`. All successor entries are positive here.

The largest normalized prediction error of these analytic parameterizations
over all settings and five methods was 3.0e-15. Tests verify the Bellman equation,
row mass, zero training semi-gradients at the exact parameters, and action-gap
metrics. An independent enumeration of every deterministic policy confirms the
always-advance policy is optimal in all nine MDPs. Five mathematical tests pass.

The control diagnostic takes the greedy policy induced by the learned goal
values and evaluates its return by a separate exact linear solve. It does not
train an actor. In this chain, any deterministic drop decision on the path
prevents reaching G from H0; thus a policy with any such error has zero return.
This is why inspecting every held-state action matters, not just the first gap.

## Final main-method results

The action-gap ratio is the learned advance-minus-drop value difference at H0
divided by its exact value under the uniform continuation policy. Ranges below
cover all 54 runs per method, not confidence intervals.

| TD updates | Method | Runs with all decisions correct | First action-gap ratio range | Greedy return / optimal return |
|---|---|---:|---:|---:|
| Expected | FB | 54/54 | 0.9592–1.0457 | 1 in every run |
| Expected | Density FB | 54/54 | 0.9799–1.0102 | 1 in every run |
| Sampled | FB | 54/54 | 0.4357–1.2978 | 1 in every run |
| Sampled | Density FB | 54/54 | 0.7875–1.1781 | 1 in every run |

The sampled FB gap is noisier, but its sign is correct at every held state at
the final checkpoint. Gap calibration and control failure are different claims.

All three additional controls have correct decisions in 54/54 expected-update
runs. With sampled updates, FB without orthogonality and the direct table also
have correct decisions in 54/54 runs; normalized squared has correct decisions
in only 41/54 runs. Its expected-update success therefore does **not** establish
that target normalization alone is a reliable substitute for the categorical
objective. These controls retain their declared rates and regularization rather
than receiving separate tuning after the outcomes were observed.

## The temporary effect and its limits

At L=6, k=50 and learning rate .001:

| Update regime and checkpoint | FB: seeds with optimal greedy policy | Density FB: seeds with optimal greedy policy |
|---|---:|---:|
| Expected TD, 10,000 updates | 0/3 | 3/3 |
| Sampled TD, 10,000 updates | 3/3 | 3/3 |
| Expected TD, 100,000 updates | 3/3 | 3/3 |
| Sampled TD, 100,000 updates | 3/3 | 3/3 |

The 10k comparison was inspected from the saved learning curves after the full
expected sweep; it is a descriptive checkpoint, not a new pre-registered endpoint.
With learning rate .01, expected-update FB already succeeds in 3/3 seeds at 10k.

Under expected updates at rate .001, the first recorded checkpoint after which
all held-state decisions remain correct through 100k ranges across seeds as:

| Method | k=1 | k=10 | k=50 |
|---|---:|---:|---:|
| FB | 3k–8k | 11k–18k | 17k–30k |
| Density FB | 1k | 1k–2k | 1k–2k |
| Normalized squared | 1k–3k | 2k–3k | 2k–4k |

Checkpoint resolution is 1k; this is not an exact hitting-time estimate. The
normalized squared control also succeeds early with expected updates, so that
particular separation cannot uniquely establish a cross-entropy advantage.
The controls' sampled behavior should be read from the complete tables rather
than assuming their expected-update success transfers to noisy training.

At 10k in the sampled sweep and rate .001, FB fails in 2/3 seeds for k=10 but
0/3 for k=50; Density FB succeeds in all three. Thus there is no clean monotonic
persistence-induced control failure at this checkpoint under sampled training.

## Was persistence actually substantial?

Yes. At L=6,k=50, the resting state holds 94.80% of normalized discounted
occupancy averaged over uniformly weighted initial state-action pairs. Its
column accounts for 99.8975% of the exact successor matrix's squared Frobenius
norm. At k=1, these values are 35.40% and 44.13%. Yet both main methods eventually
resolve the much smaller transport action differences. These are properties of
the exact target, not measurements of which entries dominate training gradients.

The true initial action gap increases from .010847 at k=1 to .014893 at k=50.
Changing persistence changes the true MDP: each comparison uses its own exact
reference. Raw prediction errors across settings should not be compared without
accounting for these changing values.

## Implications for the paper

The graph remains useful to illustrate why consequential transport decisions
can coexist with large occupancy in an unproductive state. It is **not** a
validated example of FB failing at full capacity after a reasonable sweep of
training duration and learning rate. A figure showing only the 10k expected
failure would omit decisive recovery and sampling controls.

A defensible narrower description is: "In a full-capacity tabular diagnostic,
persistent resting states can delay the acquisition of correct transport
decisions under a particular optimization setting; normalized objectives reduce
that delay." Any use must show the recovery and state that this setting isolates
one fixed-policy critic, with no actor, task encoder, reward inference, neural
observation encoder, or natural offline state-frequency distribution.

The next mechanism experiment should be motivated by a measured difference in
the cube runs (for example shared-function optimization, task/policy coupling,
or coverage), rather than making this graph harder until FB happens to fail.

## Saved artifacts

- [MDP diagram](../../PAPER/ICLR/figures/didactic/carry_drop/carry_drop_mdp.png).
- [Expected-update final results](../../PAPER/ICLR/figures/didactic/carry_drop/expected_results.png).
- [Expected-update control learning curves](../../PAPER/ICLR/figures/didactic/carry_drop/expected_control_learning.png).
- [Sampled-update final results](../../PAPER/ICLR/figures/didactic/carry_drop/sampled_results.png).
- [Sampled-update control learning curves](../../PAPER/ICLR/figures/didactic/carry_drop/sampled_control_learning.png).
- Complete numeric data: `analysis/carry_drop/{expected,sampled}/`, plus
  `analysis/carry_drop/reference_summary.json` and the pilot directory.
- Each figure also has PDF and SVG exports. Figure and numeric directories are
  ignored local artifacts; no publication or paper-source inclusion was made.
