# Spec 19b §0.1 — ORS transcription note

Status: WRITTEN 2026-08-06. Produced by direct extraction from the primary
source, not paraphrase or inference from the method name. Two independent
sources were read first-hand (not summarized by an intermediate model):

1. **The paper PDF**, `arXiv:2604.20627v1 [cs.LG] 22 Apr 2026` ("Occupancy
   Reward Shaping: Improving Credit Assignment in Offline Goal-Conditioned
   Reinforcement Learning", Venugopal, Chen, Wu, Zheng, Eysenbach, Schneider;
   ICLR 2026, OpenReview `EW8DskWQ1K`) — fetched via `arxiv.org/pdf/2604.20627`,
   extracted to text with `pymupdf` (27 pages), read in full (main text +
   Appendix A proofs + Appendix B hyperparameters). Local copy of the extracted
   text: `/tmp/claude-1001/.../scratchpad/ors_paper.txt` (session-scoped;
   not durable — re-fetch if this note needs re-auditing).
2. **The released code**, `github.com/aravindvenu7/occupancy_reward_shaping`
   (linked from the paper's own front page) — fetched with `curl` directly
   (raw file bytes, not an LLM-summarized proxy) for `impls/main.py`,
   `impls/agents/gciql.py`, `impls/agents/rew.py`, `impls/utils/datasets.py`,
   `impls/occupancy.py`, `impls/reward_distillation.py`. Quoted lines below
   are copy-pasted from these fetches; each is marked CODE so it isn't
   confused with paper text. OpenReview's PDF endpoint blocked automated
   fetches (bot-verification page) — not used; the arXiv PDF is identical
   content (same arXiv ID is on the paper's own title page) so this is not a
   gap.

Every item below is graded VERIFIED (direct quote, first-hand) or
**UNVERIFIED** (explicitly flagged, not guessed). Nothing below is inferred
from the method's name or from general priors about occupancy/W2 shaping.

---

## (a) Is r_W used ALONE, or composed with the sparse reward?

**VERIFIED: r_W REPLACES the sparse reward outright** (not added, not
blended) for every transition **except** the exact goal-reached transition,
which is force-set to a fixed positive value regardless of r_W. This is a
three-way convention, not a clean binary — record it as such.

Three independent confirmations:

1. **Paper, Appendix B.3**, the GCIQL critic loss ORS actually trains with:

   > "L^Q_GCIQL(Q) = E_{(s,a,s')~p_D(s,a,s'), g~p_D_mixed(g|s)} [(r_W(s,a,g) +
   > γV(s',g) − Q(s,a,g))²] ... Here, Q̄ is the target Q function ..., κ is the
   > expectile and r_W(s,a,g) is the ORS reward."

   The TD target is `r_W + γV`. There is no `r_sparse` term anywhere in this
   equation or its surrounding text.

2. **Code, `impls/utils/datasets.py`, `GCDataset.sample()`** (fetched
   directly, lines ~317–332 of the fetched file):

   ```
   successes = (idxs == value_goal_idxs).astype(float)
   batch['masks'] = 1.0 - successes

   if self.config['reward_shaping'] and batch_size > 1:
       ends = self.get_observations(value_goal_idxs)
       ...
       batch['rewards'] = compute_shaped_rewards(fwd_model, inits, ends, rew_model=rew_model,
                                                   nexts=batch['next_observations'])
       batch['rewards'] = jnp.where(successes, 1.0, batch['rewards'])
   else:
       batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)
   ```

   `batch['rewards']` is **assigned**, not incremented — a hard replacement
   of the sparse-reward line (`else` branch, same key) — **and then the
   success-index rows are overridden to `1.0`** regardless of what r_W said
   there. So the actual training reward is:
   `r(s,a,g) = 1.0 if s==g else r_W(s,a,g)` — r_W supplies every
   non-terminal transition, the terminal (goal-reached) transition keeps a
   fixed sparse-style value.

3. **Code, `impls/agents/gciql.py`**, the critic TD target consumes
   `batch['rewards']` with no separate sparse term:

   ```
   q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v
   ```

   confirming (2) flows straight into the critic with nothing added.

**Conflict with our spec:** spec 19b §2.3 asks the implementing agent to
determine "whether r_W REPLACES or is ADDED to the sparse reward ... per the
transcription note." **The paper's answer is REPLACE, with a terminal-state
override to a fixed positive value at the exact success index** — not a pure
r_W-everywhere scheme. §2.3's own pseudocode text ("Whether r_W REPLACES or
is ADDED... Stop-gradient everything ORS-side") did not anticipate the
terminal override; the extractor implementer should treat this as a THIRD
option to consider (replace-with-terminal-override), and flag it explicitly
if the primary arm departs from it (e.g. if the repo's own `gc_negative`
{-1,0} convention differs from the paper's `{r_W, 1.0}` convention and the
two are not directly transplantable). **This is the paper's PRIMARY arm's
literal convention — per repo standing pattern (bit-faithful transcription,
PSM precedent), the extractor's primary arm should replicate it, and any
pure-replace-no-override variant becomes an explicitly-labeled ablation.**

---

## (b) Normalization / scaling of r_W

**VERIFIED, and this is the single biggest surprise relative to our spec's
assumed default:** the paper does NOT use raw, unscaled r_W. **Appendix B.7,
Table 10** ("Specific hyperparameters for policy training: ORS and Go-Fresh")
has a column literally titled **"reward scaling for ORS (linear scaling
x : (r/x))"** — i.e. the reward actually fed to the critic is `r_W / x` for a
**per-task, per-dataset** divisor `x`, hand-tuned:

| Dataset | x (divisor) | our target env? |
|---|---|---|
| cube-double-play | **0.25** | yes (cd) |
| cube-triple-play | 0.25 | no |
| scene-play | **0.75** | yes (scene) |
| puzzle-4x4-play | 0.75 | no |
| puzzle-4x5-play | 0.75 | no |
| puzzle-4x6-play | 0.25 | no |
| antmaze-large-navigate | 2.0 | no |
| antmaze-giant-navigate | 2.0 | no |
| puzzle-4x4-noisy | 0.25 | no |
| puzzle-4x6-noisy | 0.25 | no |
| cube-triple-noisy | 1.0 | no |
| scene-noisy | 0.25 | no |
| antmaze-large-explore | 3.0 | no |

Read literally, `r/x` with x=0.25 **amplifies** r_W by 4×; x=0.75 amplifies
by 1.33×; x=2.0–3.0 shrinks it. There is no single scalar; it is swept per
task like every other GCIQL hyperparameter in the same table (α, κ, γ, actor
goal ratio).

No closed-form justification for these values is given anywhere in the main
text — Section 4.2/4.3/4.4 (the method description) contain no mention of a
scale constant; Proposition 2's constant `C` is purely theoretical ("upper
bounded, up to a multiplicative constant C > 0") and is never estimated or
reported numerically. The scale in Table 10 is an empirical hyperparameter,
tuned like any other, not a quantity derived from the theory.

**Code cross-check reveals a real inconsistency in the public artifact**,
worth recording rather than silently resolving: `impls/utils/datasets.py`,
`compute_shaped_rewards()`:

```
else:
    rewards = rew_model.compute_reward(inits, ends)
    return 0.75 * rewards
```

This is a single **hardcoded** `0.75` multiplier (note: `0.75*r`, not
`r/0.75` — a different arithmetic direction than the table's `r/x` notation
for the one row where the numbers happen to coincide, scene-play x=0.75),
applied **unconditionally to every task**, with **no CLI flag or config key
anywhere in `main.py` or `occupancy.py`** for a per-task scale (confirmed by
grep over the fetched `main.py`: the only reward-shaping flags are
`--reward_shaping` (on/off) and `--use_rew_model` (on/off), no scale flag).
So the per-task Table 10 values (0.25 → 3.0) **cannot currently be
reproduced by config alone from the public repo snapshot** — they must have
been obtained by manually editing that one hardcoded line per experiment
before launch, an un-exposed and unautomated step. **This is flagged, not
resolved**: it is real evidence the repo's HEAD does not fully match the
paper's reported experiments, but it does not change what the paper says to
do (scale by a per-task constant) — only that the constant's provenance
outside Table 10 is opaque.

**Decision for our primary arm (paper wins per repo convention):** implement
`ors_reward_scale` as `1/x` from Table 10 for the specific env being run
(cd: `1/0.25 = 4.0`; scene: `1/0.75 ≈ 1.333`), reading the table's own
`r/x` convention literally (divide by x). Do NOT default to `1.0` — that
is now a labeled ablation ("unscaled r_W"), not the primary arm. Spec 19b
§2.1's shipped default `config.ors_reward_scale = 1.0` needs a per-env config
override for the primary arm; leave the *knob* itself default-1.0/off
(bit-identical-off discipline is orthogonal to what the *config files* set).

---

## (c) Discount γ inside the occupancy `(1−γ)` normalization vs. the critic's γ

**VERIFIED, with a genuine exception the paper does not flag as an
exception.** The occupancy measure's own definition (§3.1) and Bellman
recursion (Eq. 2, §4.1) use a single symbol γ throughout — there is no
separate discount for the occupancy model in the *notation*:

> "dπ(s+ | s, a) = (1 − γ) Σ_{Δt=1}^∞ γ^{Δt−1} P(s_{t+Δt} = s+ | s, a, π)"
> (§3.1) — and Theorem 1's Bellman-optimality operator for the downstream
> policy, `Q*(s,a,g) = r_W(s,a,g) + γ max_{a'} Q*(f(s,a),a',g)` (Appendix
> A.6) — uses the **same** γ symbol as the occupancy definition.

But **the hyperparameter tables reveal these are two separately-set numbers
in practice**, and for one task family they actually differ:

- **Appendix B.6, Table 8** ("Occupancy and reward-model hyperparameters"):
  a single row, "Discount factor γ (occupancy model) = **0.99**" — not
  broken out per task.
- **Appendix B.7, Table 10** ("policy training hyperparameters"): γ is listed
  **per task**: `0.99` for every cube/puzzle/scene row, but **`0.995`** for
  `antmaze-large-navigate`, `antmaze-giant-navigate` (n-step-baseline
  Table 6 shows the same antmaze γ=0.995 pattern independently).

So **for our target envs (cube-double, scene) the two γs coincide at 0.99**
— no discrepancy for the primary arm. But **the paper's own antmaze cells
run the occupancy model at γ=0.99 (Table 8, not overridden per-task there)
while the downstream GCIQL critic runs at γ=0.995 (Table 10)** — a genuine
mismatch the paper does not discuss or justify. This matters for us only as
a precedent: **the paper does not treat γ-matching as a hard invariant in
practice**, even though the theorem's proof (Appendix A.6) implicitly assumes
one shared γ (the Bellman operator for `Q*` is written with a single γ, and
`V*`'s layer-monotonicity lemma, Lemma A.1, is derived under that same
single-γ recursion). **Our spec's plan (§2.2: "read γ/τ/hit_spec from the
LOADED [19a] config — refusing mismatches with the extractor's own
hit_spec/discount") is stricter than the paper's own practice and should be
kept as-is for the primary arm** (it is a defensible choice matching the
theorem's assumptions, and the paper's antmaze mismatch is arguably a
paper-side inconsistency, not a licensed pattern to reproduce). Flag: this
means the extractor's primary-arm lattice refusing a γ mismatch is a
**stricter-than-paper** choice, made deliberately for correctness, not
because the paper requires it.

---

## (d) Optimality-preservation condition (do not silently violate)

**VERIFIED — Theorem 1 (§4.3, proof in Appendix A.6)**, quoted in full:

> "Under suitable assumptions on goal reachability, dynamics, and dataset
> coverage, the greedy goal-conditioned policy: π^greedy(a|s,g) =
> argmax_a Q*(s,a,g), where Q*(s,a,g) denotes the optimal action-value
> function induced by the reward r_W(s,a,g), coincides with the optimal
> shortest-path policy π*(a|s,g)."

The assumptions (Appendix A.3), verbatim:

- **A.1 (Deterministic dynamics).** "the environment dynamics are
  deterministic: ... p(·|s,a) = δ_{f(s,a)}(·)."
- **A.2 (One-step reachability).** "∀k≥1, ∀s∈S_k, ∃a*(s) such that
  s'|s,a* ∈ S_{k−1}" — the goal is reachable and a shortest path exists from
  every relevant state.
- **A.3 (Strong monotonicity of Φ).** the potential Φ(s,g)=‖s−g‖² is
  strictly increasing along shortest-path layers **by at least a fixed gap
  ΔΦ>0** between adjacent layers ("there exists a constant ΔΦ > 0 such that
  Φ(s2,g) ≥ Φ(s1,g) + ΔΦ").
- **A.4 (Layer monotonicity of π_D, i.e. dataset-quality condition).** "the
  offline policy π_D is consistent with the shortest-path layer structure":
  successors under the *behavior* policy from adjacent layers must preserve
  relative layer ordering. Remark A.4 calls this "a condition on the quality
  of the offline data, which is a standard requirement for obtaining
  theoretical guarantees in the offline RL setting."

**These are non-trivial and NOT automatically satisfied by an arbitrary
offline dataset.** In particular A.1 (determinism) is false for OGBench cube
manipulation in the strict sense (contact dynamics, gripper slip), and A.4 is
a real, checkable-in-principle condition on the *specific* dataset (play
data is exactly the kind of noisy, meandering, non-monotone-progress data
A.4 worries about). **The paper offers no empirical audit of A.1–A.4 on its
own OGBench datasets** — Theorem 1 is presented as the formal justification
but the experiments do not verify the assumptions hold; the paper's own
"Limitations" paragraph (§6) is the closest acknowledgment, and it is about a
*different* failure mode (combinatorial puzzles / diffuse occupancy), not
about A.1–A.4 directly:

> "Reward shaping methods, by virtue of being sample-based, are less
> effective in long-horizon tasks with highly combinatorial state spaces and
> very few ways to achieve task success... even ORS rewards may provide
> limited signal since the occupancy measure has a high Wasserstein distance
> from goal for almost all states."

**Implication for our extractor:** do not claim ORS "provably" preserves
optimality on cube/scene without noting A.1 (determinism) is not literally
satisfied by manipulation contact dynamics — the theorem is a heuristic
justification transplanted to a domain where its own hypotheses are not
verified, exactly like the paper's own practice. The P1.2 calibration probe
(deliverable 2) is the closest thing we have to an empirical stand-in for
"is r_W ordering actually correct here," which is precisely why it is a hard
gate before spending a training run on the extractor.

---

## (e) Reported OGBench numbers and eval protocol

**VERIFIED — Table 1 exactly reproduces our spec's quoted numbers**, cross-
checked digit-for-digit against the fetched PDF's Table 1 (page 8 of the
extracted text):

| Dataset | GCIQL | ORS (ours) |
|---|---|---|
| cube-double-play | **40 ± 5** | **45 ± 7** |
| cube-triple-play | **7 ± 3** | **37 ± 8** |
| scene-play | **51 ± 4** | **80 ± 4** |

This matches our spec 19's quoted "cd 45±7 / triple 37±8 / scene 80±4 vs
GCIQL 40/7/51" **exactly** — no correction needed.

**Eval protocol, verbatim (§5.1):**

> "Performance on OGBench is measured by average (binary) success rates on 5
> test-time goals of each task. We train algorithms to convergence and
> average the results over 8 seeds."

**Table 1's caption, verbatim:**

> "ORS vs baselines: Overall average (binary) success rate (%) across the 5
> testtime goals over 8 seeds per task per algorithm. We report the 95%
> bootstrapped C.I. size after the ± sign."

**Training length, Appendix B.5, Table 7** ("Training iterations per task.
All algorithms are trained for the same number of iterations."):
cube-double-play **1M**, cube-triple-play **2M**, scene-play **1M** iterations
(everything else in our 4 target-adjacent rows is also 1M except
cube-triple's 2M).

**Puzzle-4x4-play is also reported** (70±5 ORS vs 26±3 GCIQL) — not in our
spec's quoted trio but present in Table 1, noted for completeness in case
puzzle is added to the suite later.

**Not directly transplantable to our harness, flagged as a difference in
kind, not a correction:** our repo's own eval convention (cf.
`eval_density_zeroshot.py`, `eval_zeroshot_rewards.py`) is zero-shot
retrieval-decode + fixed-goal rollout against a frozen critic/measure at a
single checkpoint step; the paper's numbers are a **native, trained
GCIQL-with-ORS-reward policy**, evaluated in-distribution on the task's own
5 canonical test goals with a Gaussian-policy DDPG+BC actor extracted
end-to-end — i.e. exactly the extractor wave spec 19b §4 describes (a plain
trained GCIQL-flow run whose reward is r_W), not a zero-shot readout. This
was already correctly scoped by spec 19 §1 P1.3 ("the honest scene
comparison") — recorded here only so the extractor implementer does not
conflate "matches ORS's protocol" with "matches our zero-shot density-readout
protocol" when writing eval code for the P1.3 wave.

---

## Summary of conflicts with spec 19b's stated defaults

| Item | Spec 19b default/assumption | Paper's actual convention | Verdict |
|---|---|---|---|
| (a) reward composition | ambiguous, "replace vs add... per transcription note" | **REPLACE**, but success-index rows forced to a fixed `1.0` regardless of r_W (not pure replace) | Paper wins; extractor's primary arm should special-case the success index, not just swap `batch['rewards']` wholesale |
| (b) `ors_reward_scale` default `1.0` | implied unscaled | **task-specific `1/x`**, x ∈ {0.25, 0.75, 1.0, 2.0, 3.0} per Table 10; cd→4.0, scene→1.333 | Paper wins for the primary arm's *config value* (per env); knob stays default-1.0/off |
| (c) γ match with the 19a checkpoint's discount | strict refusal on mismatch (spec §2.2) | paper's own antmaze cells actually mismatch (occ 0.99 vs policy 0.995); cd/scene do not mismatch (both 0.99) | Keep the strict refusal — it is *stricter* than the paper, deliberately, not paper-violating for our target envs |
| (d) A.1–A.4 | not previously stated | real, unaudited assumptions; A.1 (determinism) not literally true for cube contact dynamics | Record as a caveat on any "provably optimal" claim we make; P1.2 probe is our stand-in check |
| (e) numbers | quoted cd 45±7/triple 37±8/scene 80±4 vs GCIQL 40/7/51 | **exact match**, Table 1 | No correction; protocol difference (native training vs our zero-shot harness) noted for scoping |

## What remains genuinely unverified

- **The exact provenance of Table 10's per-task scale values** — the public
  code has only a single hardcoded `0.75` with no per-task hook; how the
  authors actually swept/applied 0.25–3.0 per task is not shown in the
  fetched files (`main.py`, `datasets.py`, `rew.py`, `occupancy.py`,
  `reward_distillation.py`, `gciql.py` were all checked; no scale flag
  exists in any of them). Possible the actual experiment scripts differ from
  what's on `main` at fetch time, or the constant was hand-edited per run —
  either way, **not reproducible from the repo alone**, flagged rather than
  guessed.
- **Constant `C` in Proposition 2** (`W₂² ≤ C·E[flow MSE]`) is never
  estimated numerically anywhere in the paper or the fetched code (the code
  uses the raw flow-MSE-based estimate with an implicit `C=1`, i.e.
  `return -1.0 * rewards` in `compute_shaped_rewards`'s no-`rew_model`
  branch) — so the "upper bound up to a multiplicative constant" is used
  uncalibrated in practice; recorded, not resolved.
- Full architecture/optimizer details for the reward-distillation network
  (Eq. 5) beyond Table 8/9 were read but not exhaustively cross-checked
  against every field in `rew.py`'s `get_config()` — spot-checked (MLP
  {512,512,512,512}, GELU, layer norm, discount 0.99, lr 3e-4 all agree
  between the paper's Table 8/9 and the fetched `rew.py`/`occupancy.py`
  defaults) but a field-by-field diff was not done since architecture
  hyperparameters are not load-bearing for this note's five questions.
- OpenReview's own PDF (`EW8DskWQ1K`) could not be fetched directly (bot
  wall); relied on the arXiv PDF instead, which is the same paper (title
  page cross-references both). Not expected to differ, but marking the
  substitution honestly.
