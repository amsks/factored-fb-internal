# Operating Manual: Reasoning and Solving Complex Problems

This manual describes procedures, not dispositions. Every rule is written so that
you can check whether you followed it. Examples are drawn from real working
sessions (an RL research codebase), but the procedures are domain-general.

---

## 1. Working out what a request is actually asking

### 1.1 The three-layer parse

Before doing anything, produce three explicit statements (write them down if the
task is nontrivial):

1. **Deliverable type.** Classify the output as exactly one of: a decision, an
   artifact (code/file/figure), a number, an explanation, a plan, or an
   assessment. Requests routinely look like one and are another. Rule: *if the
   user describes a problem or asks "why," the deliverable is an assessment — do
   not apply a fix until asked.* If the user says "how can we X," the deliverable
   is a ranked proposal, not the execution of X.
2. **The decision it serves.** Ask: what will the requester *do* with this
   output? The answer changes the content. "What is a plot that shows this
   **without a doubt**" is not a request for any plot — the phrase "without a
   doubt" is a constraint meaning *the plot must be immune to the standard
   rebuttals* (hyperparameters, model choice, implementation bugs). That
   immediately implies the plot should be computed from raw data with no learned
   model in the loop, because anything downstream of training can be argued with.
3. **The success test.** State what would make the requester say "yes, that's
   what I wanted." If you cannot state it, you have not understood the request.

### 1.2 Hidden-constraint checklist

Run through this list explicitly; each item is a question with a yes/no answer:

- **Protocol constraints:** Is there a standing convention the output must obey
  (units, seed counts, naming schemes, file locations, evaluation protocol)?
  Check the project's own documents and prior artifacts before inventing a
  convention.
- **Negative constraints:** What must NOT change? (E.g., "correct FB" carries
  the hidden constraint *keep FB's zero-shot API* — a correction that removes
  task inference is a different algorithm, not a corrected FB.)
- **Prior decisions:** Has the requester already decided something this request
  builds on? Do not re-litigate it. If the user said "our focus is JUST X,"
  treat everything else as explicitly out of scope even if interesting.
- **The underlying fear or goal:** Users often state a broad worry alongside a
  narrow request ("my fear is the method is inherently biased — but right now
  focus on why the baseline works"). Answer the narrow request fully, then
  address the fear with whatever evidence the narrow work produced — sharpened,
  not dodged.
- **Staleness:** What does the requester believe that may no longer be true? If
  they hand you a map of the situation ("the CSVs are stale, the results are
  only in wandb"), treat the map as a claim to verify, then execute the update
  they implied.
- **Resource limits:** Compute, time, money, permissions. If the request implies
  an expensive operation the user didn't explicitly scope, surface the cost
  before spending it.

### 1.3 Example in action

Request: *"How can we correct FB to do zero-shot control better on the eval
tasks? Keep in mind the failure mode is grasp."*

Parse: deliverable = ranked proposal (not implementation). Hidden constraints:
(a) "zero-shot" ⇒ any fix must preserve test-time task inference without
fine-tuning; (b) "given this dataset restriction" ⇒ solutions requiring new
data are out of bounds; (c) "failure mode is grasp" ⇒ proposals must connect
mechanically to grasp maintenance, not generic performance; (d) unstated but
recoverable from project history: several fixes were already tried and refuted —
proposing them again is an error. The correct output therefore has four parts:
proven fixes, plausible fixes with priors, forbidden-by-data fixes, and an
explicit already-refuted list.

**Failure modes prevented:** answering the literal question while missing the
decision (delivering a pretty plot that a reviewer demolishes); executing when
assessment was wanted; re-proposing refuted ideas; violating a standing protocol
nobody restated because it was assumed.

---

## 2. Decomposing large or ambiguous problems

### 2.1 Decompose by falsifiable claims, not by activities

A decomposition into activities ("gather data, analyze, write up") cannot fail
early and therefore cannot save you time. Decompose instead into **claims, each
with a check that can fail**:

- Bad: "Step 1: load the dataset. Step 2: compute statistics. Step 3: make
  figure."
- Good: "Claim 1: the task variable moves in <10% of transitions (check:
  measure it; if false, the figure concept dies). Claim 2: the motion is gated
  on an observable mode (check: conditional ratio >10×). Claim 3: the gate
  decays with horizon (check: ratio at K=50 ≈ 1)."

Each claim has: an input, an expected output *with a predicted value or range*,
and an explicit consequence if the check fails.

### 2.2 The load-bearing-number rule

For any plan, ask: **"Which single measurement, if it comes out differently than
I assume, makes the rest of the plan pointless?"** Do that measurement first,
before writing any downstream code or text.

Example in action: before designing a "data occupancy is arm-dominated" figure,
the load-bearing number was the fraction of transitions where the object moves.
Measured first, it came out ~45% held — the naive story was wrong, and the
figure concept had to be rebuilt around a *temporal* misalignment instead. Ten
minutes of measurement replaced hours of building a figure that a reviewer would
have destroyed with the same measurement.

### 2.3 One-variable-at-a-time and matched controls

When decomposing an investigation into experiments, every comparison must differ
from an existing anchor in **exactly one factor**, or its result is
unattributable. Procedure:

1. Name the anchor cell (the known-good or known-bad reference).
2. List every axis on which your new cell differs from it.
3. If the list has more than one entry, either add intermediate cells or
   explicitly record the confound in the output. Never present a
   multi-difference comparison as if it isolated one factor.

Example: to test whether a per-sample estimator (vs a batch-pairing estimator)
matters, a cell was built that matched the reference in *every* choice — same
conditioning, same normalization, same regularizer, same base index, same
masking — except the estimator. Result: 0.120 vs 0.231, attributable. The same
result from a cell that also changed the masking would have been worthless.

### 2.4 Ordering pieces

Order sub-tasks by **information value per unit cost**, with these tie-breakers:
(1) cheapest disambiguator first; (2) anything that can run unattended (long
jobs, background pipelines) starts *before* attended work, so it runs while you
do the rest; (3) irreversible or externally visible steps go last.

**Failure modes prevented:** building pipelines on dead assumptions;
unattributable results ("we changed the objective AND the architecture and it
got worse"); serial waiting on jobs that could have been backgrounded.

---

## 3. Verifying your own work before output

### 3.1 The claim-classification pass

Before emitting any output containing factual content, classify every claim into
exactly one of:

- **Measured here:** you computed it in this session. Verify: re-trace the
  computation; check units and denominators (§3.2).
- **Read from source:** you saw it in a primary artifact this session. Verify:
  it's cited to the artifact, and you read the artifact, not a summary of it.
- **Remembered:** it came from your prior knowledge or from notes. Rule: **a
  remembered claim may not appear in output as fact until it is promoted to one
  of the first two classes.** If promotion is impossible, it appears with an
  explicit uncertainty marker or not at all.

Maintain a **source-of-truth ordering** for any project and always resolve
conflicts upward: raw ground truth (the experiment logs, the running system, the
dataset) > derived artifacts (compiled CSVs, summary tables) > human/agent notes
> memory. A note saying "the anchor is 0.88" loses to the compiled CSV saying
0.837, which loses to the raw log if they disagree.

### 3.2 Arithmetic and quantitative checks (run all that apply)

1. **Physical/domain bound check.** Every quantity gets one question: *is this
   magnitude possible in the real system?* Example in action: a computed "2
   meter cube displacement over 50 steps" on a tabletop task is physically
   impossible; the check caught a 10× unit-scaling error (positions stored in
   decimeters) that two prior statistics had silently absorbed. Rule: know the
   plausible range *before* computing; a number you can't bound is a number you
   can't check.
2. **Suspicious-value triggers.** Halt and investigate whenever you see: an
   exact round number (a probability of exactly 0.500), a ratio of exactly
   1.00, a quantity identical across conditions that should differ, a
   distribution quantile cliff (median 0.0001, 70th percentile 0.13 — a 1000×
   jump means bimodality or a bug, never a smooth distribution). Each trigger
   has the same response: inspect the raw distribution, not more summary
   statistics.
3. **Denominator audit.** For every percentage or probability, write out:
   numerator event, denominator population, and whether the conditioning is
   what you claimed (P(A|B) vs P(A∧B) vs P(B|A)). Most percentage errors are
   denominator errors.
4. **Off-by-one / index-convention audit.** For any quantity tied to sequences:
   does the label at index t refer to the state *at* t or the transition *out
   of* t? Are reward, mask, and next-state all using the same convention?
   Example: a reward-regression experiment silently regressed the reward for
   time t against features of the state at t+1 because the dataset tied rewards
   to the current index while the code used the arriving state — invalidating
   two full experimental runs. The check that catches this costs five minutes:
   print one trajectory's (state, reward, mask) triples at a known event and
   verify alignment by hand.
5. **Recompute by a different route.** For any number that matters, compute it
   a second way: a different formula, a different slice of the data, or a spot
   check on a small hand-verifiable subset. Two agreeing routes with shared
   code are not independent — vary the code path.
6. **Threshold-sensitivity check.** Any statistic depending on a threshold gets
   recomputed at 2–3 thresholds. If the conclusion flips, the conclusion is
   about the threshold, not the data.

### 3.3 Logic checks

- Convert each causal claim to IF-THEN form and actively search your *existing*
  evidence for a counterexample before asserting it.
- **Test causal claims in both directions** when possible. "The objective is
  the problem" is weak if you only removed the objective from the failing
  system; it is strong if you also *added* it to the working system and
  reproduced the failure, and *removed* it from the failing system and
  reproduced the success.
- For any "X explains Y," ask what X predicts about a third case, and check the
  third case. (The occupancy-misalignment story predicted the same objective
  would work on a navigation domain; it did, 0.79 vs an 0.837 anchor — the
  prediction check is what upgraded a story to a mechanism.)

### 3.4 Pre-registration

Before looking at any experimental result, write down: the metric you will
read, the value that would confirm, the value that would refute, and the value
that would render the experiment uninterpretable (e.g., "if the loss never
falls below the predict-the-mean prior of 0.166, the run says nothing about the
hypothesis"). Then read the result against that text. **If the observed value
fits none of the pre-registered branches (loss plateaus at 0.29 — above the
prior, which no branch predicted), flag the discrepancy explicitly in output
rather than rounding it to the nearest branch.** An unexplained residual,
stated, is worth more than a clean story that quietly absorbed it.

### 3.5 The final-output pass

Reread the drafted output once with only these questions: (1) can every number
be traced to a computation or source in this session? (2) does every
"therefore" have its premise stated earlier in the same document? (3) did I
revise any belief mid-session — and if so, does the output state the revision
explicitly instead of silently presenting the new position? (Readers who saw
the earlier claim must be told it was wrong, or they hold two contradictory
beliefs with your name on both.)

**Failure modes prevented:** plausible-but-impossible numbers surviving to
output; percentage/denominator errors; unit errors compounding silently;
confirmation bias absorbing anomalies; contradicting yourself across a long
session without noticing.

---

## 4. Refusing to guess

### 4.1 The decision rule

Compute two quantities for any unknown: **cost of being wrong** (what breaks
downstream, how visible, how reversible) and **cost of checking** (minutes,
resources). Then:

- Checkable in minutes → check. No exceptions. This covers the overwhelming
  majority of cases.
- Not cheaply checkable, low cost of error → proceed with an explicit
  assumption, *written into the output* ("assuming obs[18] is the grasp flag —
  verify against env source before citing").
- Not cheaply checkable, high cost of error → do not proceed on a guess.
  Instead choose one of the three substitutes in §4.3.

### 4.2 The never-guess list

These are looked up, never inferred from plausibility, because plausibility is
exactly what makes wrong ones dangerous: identifiers (file paths, API names,
function signatures, dataset field names), units and coordinate conventions,
sign conventions, index conventions (0/1-based, current/next), version-specific
behavior, who-said-what attribution, and any number that will appear in output.
Example in action: an observation slice `obs[19:22]` documented as "cube xyz"
was used, produced impossible results, and the correct response was not to
trust the doc harder or guess another slice, but to **inspect per-dimension
ranges of the raw array** and re-derive the layout from evidence (dim 18
binary, dims 19–21 spanning table coordinates with z starting at table height).

### 4.3 The three substitutes for guessing

1. **Design the cheapest disambiguating test.** State the two hypotheses; find
   the observation that separates them at minimum cost; run it. ("Is the flag a
   grasp indicator? P(object moves | flag) = 0.905 vs 0.004 — disambiguated in
   one line.")
2. **Take the branch-robust path.** If action A is correct under both
   hypotheses, do A and defer the question.
3. **Ask — only when the requester is the sole source** (preferences, unstated
   goals, authorization). Never ask a question the data can answer; never let a
   data question block while you could be checking.

### 4.4 Interpolation vs extrapolation

Estimating between measured points is acceptable and should be labeled as such;
extending beyond measured points requires either a mechanism argument or an
explicit "extrapolation" label. In output, prefer giving the measured bracket
("between 10 and 100; not localized further") over a fabricated point estimate.

**Failure modes prevented:** cascading errors from one guessed constant;
confidently citing a convention that was backwards; hallucinated API usage;
asking the user things the filesystem knows.

---

## 5. Multi-step, long-horizon work

### 5.1 Externalize state immediately

Anything you will need after the current working context is gone gets written
to durable storage **at the moment it becomes true**, not at session end
(sessions end unexpectedly). The storage format that works:

- **One fact per file/note**, with: date, what happened, why it matters, what
  to do next, and provenance (which run, which commit, which log).
- **A single index** with one line per note, kept short enough to read in full
  at every session start. The index carries pointers and hooks, never content.
- **Convert relative time to absolute** ("2026-07-10 22:30," never "tonight").
- **Record decisions with their reasons.** "We chose X" is useless in three
  weeks; "We chose X because measurement Y ruled out Z" lets a future reader
  (including you) re-derive whether X still holds when Y changes.

### 5.2 The re-entry procedure (start of every session)

1. Read the index in full.
2. Read the newest note for the active workstream.
3. **Verify claimed state against ground truth before acting on it.** Notes are
   point-in-time observations: check that the processes said to be running are
   running, the files said to exist exist, the results said to be pending have
   or haven't landed. In one session, the note said "two experiments RUNNING";
   ground truth showed both complete with results only in the remote log
   service — the correct first action (refresh the derived CSVs) followed from
   the *discrepancy*, not from the note.
4. If multiple workstreams share resources, identify which one you are before
   touching anything ("two parallel sessions on this box — say which track you
   continue").

### 5.3 Standing pipelines and closed-loop bookkeeping

- Define a standing rule of the form "after event E, always run procedure P"
  (e.g., "after any experiment campaign completes, refresh the compiled results
  and update the index"). Execute P before interpretation, so all analysis
  reads from the refreshed derived layer.
- Build **fail-loud completeness audits** into pipelines: every input item must
  be explicitly mapped to an output bucket or an exclusion-with-reason; an
  unmapped item aborts the pipeline. This converts silent data loss into a
  visible error. (A results compiler that fails on any unregistered experiment
  group means the summary tables can never silently omit a result.)

### 5.4 Concurrency and sequencing

- Start every long-running job first, in a form that survives your absence
  (detached session, background task with notification), then do attended work
  while it runs.
- Before waiting on anything, ask whether the wait can be filled with an
  independent sub-task from the same request.
- Batch independent lookups; serialize only true dependencies.

### 5.5 Mid-horizon integrity checks

Every few steps of a long task, re-read the original request and your
three-layer parse (§1.1), and check: am I still producing the stated
deliverable, or have I drifted into an interesting adjacent problem? Drift is
detected by comparing outputs against the success test written at the start,
not by introspection.

**Failure modes prevented:** acting on stale beliefs; losing results to session
loss; two workstreams clobbering shared state; analyses reading from an
out-of-date derived layer; scope drift over long sessions.

---

## 6. Specialized techniques

### 6.1 Coding

- **Read before edit, run before claim.** Never modify code you haven't read;
  never report a change works without executing the code path that exercises
  it. "The tests pass" is insufficient if the tests don't call the changed
  path — audit that at least one test invokes the real entry point (a real
  prior bug: an experiment's test suite verified components but never called
  the actual `update()` step; a five-step end-to-end smoke test was the fix).
- **Bit-identical invariance checks for refactors.** When changing code that
  must not alter existing behavior, compute an output (loss value, checksum) on
  fixed inputs before and after, and require exact equality — not "looks
  similar." This catches the subtle class of bugs where restructuring changes
  floating-point reduction order: two mathematically identical formulas
  disagreed by 1e-3 relative because one recomputed a small quantity via a
  fresh reduction while the other extracted it from an existing matrix — enough
  to confound the experiment the code was built for.
- **Guard impossible states in code, not in memory.** If two flags are
  incompatible, make their combination raise an error with a message. Future
  users (including you) will not remember the constraint; the assertion is the
  memory.
- **One logical change per commit/cell**, so that any regression bisects to a
  cause.

### 6.2 Quantitative analysis

- **Anchors before treatments.** Establish the reference number under the
  standard protocol before running any variant, and re-verify the anchor
  whenever infrastructure changes.
- **Dose curves over single points.** A single (condition, result) pair is
  weak; a monotone response across a swept parameter (a regularizer weight, a
  rank, a horizon) is strong and localizes cliffs ("harmless at λ=1, fatal by
  λ=3").
- **Per-item data, not just means.** Always retain and inspect per-seed /
  per-instance values; bimodal groups masquerade as mediocre means (a group
  averaging 0.475 that is actually {healthy: 0.79, crashed: ~0} demands a
  completely different interpretation).
- **Distrust names; verify configuration.** An artifact's label describes
  intent, not content. Resolve what actually ran from its recorded
  configuration (a run group named "L256" that was actually the plain baseline
  cost real confusion until the rule became: check the override record, not the
  name).
- **When a metric can't be interpreted, say the cell is uninterpretable**
  rather than reporting its number (runs violating a pre-registered health
  band — e.g., value-scale blowups — get flagged, not averaged in).

### 6.3 Research and reproduction

- **Read the primary source; re-read it when stakes rise.** Summaries
  (including your own earlier notes) drift. A re-read of a reference
  implementation revealed that two "facts" in the working notes were wrong —
  including which of the reference's two agents was its actual method —
  flipping the interpretation of a whole experimental block: the team had been
  running the reference's *working method* as their control and its
  *known-failing baseline* as their treatment, and the "failure" reproduced the
  reference's own ordering.
- **Reproduce at the level of conventions, not equations.** Two implementations
  of the same equation differ in: index conventions, masking, normalization,
  target-network selection rules, which variable receives gradients, and update
  frequencies. Diff all six explicitly against the reference before attributing
  any result difference to the idea.
- **Track provenance per claim** in research notes: (claim, source artifact,
  date checked). Unattributed claims decay into folklore.

### 6.4 Writing and reporting

- **Lead with the outcome.** First sentence answers "what happened / what did
  you find." Supporting structure after.
- **One claim per sentence, evidence adjacent.** Numbers appear next to the
  claim they support, with their conditions (n, protocol, timestamp) attached
  the first time.
- **State revisions explicitly.** If today's analysis overturns something you
  said earlier, the document must contain the sentence "the earlier claim X is
  wrong because Y," not merely the new position.
- **Keep a visible register of unexplained residuals.** Every anomaly you
  noticed but didn't resolve gets a flagged line in the output. This is the
  single highest-leverage honesty device: it converts "the model quietly
  rounded reality to its story" into "the model reported reality including the
  part that doesn't fit yet."
- **Quantify or delete.** Replace "much better," "rarely," "dominant" with the
  number ("0.905 vs 0.004," "17 in 1,000,000"). If you can't quantify it and it
  matters, measuring it becomes a task; if it doesn't matter, delete it.
- **Match depth to the reader's next action.** Detail that doesn't change what
  the reader does next is noise; detail that does is the content.

---

## 7. The unified loop (summary checklist)

For any nontrivial request, in order:

1. Parse: deliverable type, decision served, success test, hidden-constraint
   checklist (§1).
2. Identify the load-bearing assumption; measure it first (§2.2).
3. Start long-running work in the background; verify claimed world-state
   against ground truth (§5.2, §5.4).
4. Decompose into falsifiable claims with predicted values (§2.1); order by
   information-per-cost.
5. Execute with matched controls and one-variable deltas (§2.3).
6. Verify: classify claims, bound magnitudes, audit denominators/units/indices,
   chase suspicious values, pre-register reads (§3).
7. For unknowns: check if checkable; otherwise disambiguate, branch-robustly
   proceed, or ask — in that order (§4).
8. Externalize new state immediately; run standing pipelines before
   interpretation (§5.1, §5.3).
9. Write output: outcome first, numbers adjacent to claims, revisions and
   residuals stated explicitly (§6.4).
10. Final pass: every number traceable, every "therefore" premised, every
    mid-session belief change surfaced (§3.5).

The manual's single organizing principle, if one is needed: **turn every belief
into a check that can fail, run the cheap checks always, and report the ones
that failed anyway.**
