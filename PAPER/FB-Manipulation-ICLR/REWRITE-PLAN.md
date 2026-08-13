# Rewrite plan for main.tex — communicate the idea, kill the circularity

2026-07-30. Diagnosis + new skeleton + ready-to-paste text. The current draft is
rigorous and honest but reads as research archaeology: it narrates the
investigation in the order it happened, names the same object five ways, states
every claim three times at three abstraction levels, and hedges inside every
paragraph. It also predates the §9.14/§9.15b results, which make the story
*simpler and stronger*.

## 1. What's wrong, concretely

**1a. The key idea is buried under estimator jargon.** The reader meets
"the pointwise successor-density Bellman equation contains a singular immediate
term" (abstract, sentence 4) before any intuition. The idea is one sentence:

> FB asks its critic "what is the probability density of being *exactly at
> point x*?" — a question whose true answer is a Dirac spike with no finite
> regression target. We ask instead "how much discounted time will you spend
> *within 4 cm of x*?" — a question whose answer is an ordinary bounded number
> that plain Bellman regression can fit.

Everything else in the paper is a consequence of this substitution. Say it in
plain words in the abstract and in the first 150 words of the intro, then
derive it formally in §3.

**1b. Circular structure.** The same claims recur at multiple altitudes:
- "the estimator, not the representation" — stated ≥6 times (abstract, intro,
  §3 opener, §3.2, finding box, conclusion).
- "success does not certify a measure" — §2.2, §5 opener, §5.1, finding, related
  work.
- "the abstraction is supplied, not discovered" — §2.3, §6.1, two Limitations
  paragraphs, conclusion.
Rule: each claim gets ONE home (a finding box); elsewhere it may be referenced,
never re-argued.

**1c. Five names per object.** reaching value / phase 1 / absorbing backup /
$Q^{\Reach}$ / "bounded reaching regression"; visitation / phase 2 /
nonabsorbing / $M^{\Visit}$ / "the separated object" / "the deployed
composition". Rule: two names, fixed at first definition — **reaching value**
and **visitation measure** — plus $\hat N$-corrected readout. "Phase 1/2"
survives only inside the training-procedure paragraph.

**1d. Forward references force re-reading.** §2.2 explains diagonal-vs-separated
abstractly before the reader has seen either object; "the separated form" is
used pages before it is instantiated. Rule: no distinction is introduced until
the section that needs it, and it is introduced on a concrete example (two
goals, one summed reward).

**1e. No Figure 1.** The first figure is a divergence curve. The paper needs a
schematic: panel (a) point query → Dirac target → unbounded "push up the seen
successor" + ortho crutch; panel (b) region query → 0/1 hit → ordinary Bellman
regression; panel (c) stop-counting vs keep-counting = reaching vs visitation.
Plus a results-at-a-glance strip (0.42 → 0.97 control; 27% → 1% negative mass;
+33% return on unseen rewards).

**1f. Chronology instead of argument.** §6 (transfer) → §7 (scale error →
task vectors) reads as lab-notebook order; the 1/N̂ detective story gets a whole
section for what is ultimately a bookkeeping factor. Compress to one paragraph +
the log-log figure inside the transfer section.

**1g. Hedge density.** Nearly every paragraph carries in-line caveats. Move them
to (i) one caveat sentence per finding box and (ii) the Limitations section,
deduplicated (currently 9 paragraphs, several restating §5/§6 text verbatim).

## 2. The new spine: one promise, three separable claims, three fixes

The best idea in the current draft is its *last* sentence ("three problems that
are often conflated"). Make it the skeleton and the contribution statement:

> The zero-shot promise of a successor measure decomposes into three separately
> testable claims: (1) the object can be **learned** (the estimator converges to
> something that controls), (2) the object is a **measure** (non-negative,
> mass-calibrated, integrates rewards), (3) the policy can **act** on task
> vectors inferred from new rewards. Standard FB fails all three on manipulation
> — for three different reasons — and each has a targeted fix: region targets
> fix learning; a sign hinge anchored by mass conservation fixes validity;
> reward-derived task-vector coverage fixes acting.

Section plan (8pp):
1. **Introduction** — promise; the clean gap (0.42 vs 0.93, same data/actor/
   critic); the one-sentence diagnosis + substitution in plain words; Fig 1;
   the three-claims roadmap; contributions = the three fixes + the diagnostic
   methodology.
2. **Why a point density cannot be regressed** (≤1 page) — Eq. (density
   Bellman), the plain-words version of the linear-term/ortho pathology, and
   the frozen-$B$ experiment as the *proof* (one table). Kill repeated
   restatements; the tabular playpen moves to a two-sentence pointer.
3. **Region-Bellman successor measures** (method) — hit event, the two backups
   with the stop/keep-counting intuition, contraction prop, two-phase training,
   notation box. State once: the region is supplied, not discovered (details →
   §7).
4. **Claim 1: learning** — suite table (reaching beats anchor everywhere;
   visitation costs success off cube-single). Add §9.14's 2×2: bounded loss on
   Dirac support = 0.108, FB loss on region = 0.28, conjunction = 0.92, and
   effector-region 0.823 ⇒ *bounded regression and region structure are jointly
   necessary; the region need not be the evaluation variable*. This REPLACES the
   "converse experiment has not been run" limitation and the current §8's
   hedge.
5. **Claim 2: a valid measure** — action-ordering divergence (one paragraph);
   set calibration table; negativity as MASS (24–29%, worse in dense regions —
   the D1 numbers, stronger than the point-count story); then the fix:
   one-sided hinge + total-mass conservation anchor
   $((1-\gamma)\,\E_g[M/\hat N] - 1)^2$ ⇒ negative mass 27% → **1.0–1.4%**,
   ratios 1.15–1.33, success unchanged; report the one cost honestly (Spearman
   at hw 0.8: 0.68 → 0.50; soft-hinge arm pending). Then dwell semantics:
   1.33× return, oracle-ceiling mechanism.
6. **Claim 3: acting on inferred rewards** — merged current §6+§7: the 48-reward
   harness; transfer degrades with bandwidth; the $\bar N$ scale factor as one
   paragraph + log-log figure; correcting value ≠ correcting behavior ⇒
   task-vector coverage experiment (kernel/rff table); with the §9.15b arms,
   smooth-0.8 rises 0.24/0.43 → 0.46–0.86 — the last unsolved family becomes
   "partially solved by validity + coverage together".
7. **What is still open** (honest, deduplicated) — region *discovery*; small-box
   (hw 0.4) calibration = rank-limited (d128 experiment if landed); visitation's
   control cost off cube-single; iso-budget control; seeds.
8. Related work (tight) + short conclusion (no re-summary of every section —
   the three-claims sentence + one forward-looking line).

## 3. Ready-to-paste: new abstract

> Successor measures promise reinforcement learning's cleanest division of
> labor: learn once where a policy goes, then evaluate any reward by
> integration. On offline robot manipulation this promise fails at the first
> step — forward–backward (FB) representations reach 0.42 success on OGBench
> cube manipulation where goal-conditioned IQL with the same data, actor, and
> bilinear critic reaches 0.93. We show the failure is not the representation
> but the question the critic is asked. FB regresses a pointwise successor
> *density*, whose Bellman target at the observed successor is a Dirac spike:
> no finite number is the right answer, so the standard surrogate rewards
> pushing that value up forever and leans on an orthonormality penalty as an
> indirect brake — remove the brake and the critic's scale diverges 350-fold on
> a frozen, known-good embedding; keep it and control is strangled (0.23).
> Asking a bounded question dissolves the dilemma: "how much discounted time
> will the cube spend within 4 cm of g?" has a 0/1 immediate event and an
> ordinary Bellman regression, no extra terms. The resulting reaching value
> beats the matched anchor on every environment tested (0.97 / 0.55 / 0.48 vs
> 0.93 / 0.27 / 0.28); its visitation extension, made non-negative by a sign
> hinge anchored with a mass-conservation penalty, is calibrated within
> 10–33% on coarse sets with under 1.5% negative mass, and earns 1.33× the
> reaching value's return on 48 never-trained rewards — because it learns to
> dwell, not just arrive. A final experiment separates value inference from
> control: policies fail on broad rewards until reward-derived task vectors are
> added to training, which repairs broad and compositional rewards without
> costing task success. What remains open is discovering the region structure
> we supply by hand.

(Verify every number against the ledger before submission; the 350-fold is
154→5.4e4.)

## 4. Ready-to-paste: intro opening (replaces first two paragraphs)

> A successor measure answers "where will this policy spend its future?" Given
> that answer, any new reward can be evaluated by integration instead of
> retraining — one representation, every task. Robot play data is the setting
> this idea was made for: a million transitions of a cube being picked up,
> carried, and set down, with the reward of interest unknown at collection
> time.
>
> The idea currently does not survive contact with this setting. On OGBench
> cube manipulation, tuned forward–backward (FB) methods reach 0.42 success.
> Goal-conditioned IQL — same dataset, same flow actor, same rank-50 bilinear
> critic — reaches 0.93. Nothing about exploration, capacity, or architecture
> explains the gap. Something about the *objective* does.
>
> Here is the objective's problem in one picture (Figure 1). FB's critic is
> asked for the probability density of the policy being *exactly at point x*.
> The true immediate target at the observed successor is a Dirac spike — there
> is no finite number the critic should output, so FB's surrogate loss simply
> rewards making that output larger, forever, and an orthonormality penalty on
> the representation acts as the brake. We show this brake is the load-bearing
> defect [...]. Then we change the question: [...] "how much discounted time
> within 4 cm of x?" [...]

## 5. Content updates the rewrite must fold in (behind the current draft)

| stale claim in main.tex | replace with (source) |
|---|---|
| Limitations: "converse experiment has not been run" | §9.14 2×2: Dirac 0.108 / FB+ball 0.28 / both 0.92; effector region 0.823 ⇒ interaction, region-ness not privilege |
| §5.3 + Limitations: hinge rejected, "no parameterization preserves nonneg + mass + linearity" | §9.15b: hinge + mass-conservation anchor → neg mass 1.0–1.4%, ratios 1.15–1.33, success 0.93–0.95; honest cost: Spearman hw0.8 0.68→0.50 (soft-hinge mcE pending) |
| negativity quoted as 45% of *points* | D1: 24–29% of *mass*, worse in high-density quartiles (stronger + more intuitive) |
| smooth0.8 "remains unsolved" (0.33 best) | mcA/mcB: 0.40/0.86 and 0.46/0.72 gap_frac; L2-A Spearman 0.82–0.83 |
| transfer means (rzK50-era): mean return 24.4 best | corrected arms 32.4/32.5; twozone 33.0 vs oracle 24.7 |
| quartic/kernel-smoothness unmentioned | one sentence: smoothing the kernel alone does NOT fix negativity (mcC null) — pins the fix on the penalties, not the target shape |

## 6. Language rules for the pass

- One name per object; a notation box after Prop. 1; delete the \BRR macro or
  use it everywhere.
- No forward references: cut §2.2 (diagonal-vs-separated) and introduce the
  distinction in §5 where it is first needed, on the two-zone example.
- Findings boxes: claim + number + one caveat. All other hedges → §7.
- Delete meta-discourse ("The distinction matters for the paper's scope",
  "This leaves a more basic question", "tells a different story").
- Active voice; "we ask / we freeze / we replace".
- Every table gets a one-line "how to read this" sentence, not a paragraph.
