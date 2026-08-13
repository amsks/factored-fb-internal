# Phase-2 Ledger — everything tried, every result, one page

**Purpose:** single point of reference for brainstorming. Consolidates phase-2
(visitation / non-absorbing measure) work that is currently scattered across
specs 07–11, `DECISIONS.md`, `two-phase-successor-measure-note.tex`,
`analysis/rescuing-fb/**`, run-dir `eval.csv`, and `/dev/shm` logs.

**Last updated:** 2026-07-28. **Every number below was re-read from its source
file for this document**, not copied from prior prose — several published tables
were stale (see §6).

> ⚠ **Volatility:** the torch campaign's logs live in `/dev/shm/spec10_logs/`,
> which gets wiped. Checkpoints under `~/exp_spec10/torchfb/` and JSONs under
> `results/spec10_torchfb/` survive; the logs do not. Copy before relying on them.

---

## 1. What phase 2 is

Phase 1 learns the **absorbing** reaching value: a hit terminates the recursion,
fixed point `E[γ^T] ∈ [0,1]`.

    phase 1:  diff = M − h − γ(1−h)·sg[M⁻]

Phase 2 drops the absorption mask so revisits are counted, giving the
**visitation measure**, fixed point `≤ 1/(1−γ)`:

    phase 2:  diff = M − h − γ·sg[M⁻]

with `h(s',g) = 1[‖c(s') − c(g)‖ ≤ ε]`, ε = 4 cm on the privileged cube slice.
Standard protocol: warm-start every network from the phase-1 checkpoint, **freeze
ψ** (the backward map), keep the flow actor greedy on `⟨F,z⟩`.

**The point of phase 2 is the semantics, not the score.** Phase 1 already gives
the control quality; phase 2 exists to turn a reaching value into something that
integrates rewards and composes over sets. So it must be judged on **two axes**,
and they do not agree (§4).

---

## 2. Variants tried — control quality

Native GC overall success, `cube-single` unless stated, final step unless noted.

| # | variant | group / run dir | result | status |
|---|---|---|---|---|
| 1 | **canonical**: ball-visit, frozen ψ | `BallVisit-FrozenPsi-d50` | **0.955** (0.952/0.928/0.984) | reference |
| 2 | + **frozen actor** (E2) | `BallVisit-FrozenActor-d50` | **0.972** (0.952/0.992) | see §5.1 |
| 3 | **separated (z,x)** (E1) | `BallVisit-ZX-d50` | **0.778** (0.844/0.712) | best-calibrated, worst control |
| 4 | ZX + **actor-z** (L6, azB) | `BallVisit-ZX-ActorZ-d50` | 0.878 @**300k** (0.840/0.916) | adopted |
| 5 | ZX + actor-z + **trained N̂** (2b) | `BallVisit-ZX-ActorZ-Nhat-d50` | 0.868 @**150k** (0.808/0.928) | incomplete |
| 6 | ZX + **multi-ε heads** (L5, msA) | `BallVisit-ZX-MultiEps-d50` | 0.716 @**300k** (0.640/0.792) | seed-split, **not adopted** |
| 7 | **torch port** (FB codebase) | `exp_spec10/torchfb/p2_s4{2,3}` | **0.887** (0.867/0.907)† | replication |
| 8 | torch + **non-negativity λ=1** | `p2_nn1.0_s4{2,3}` | ~0.910 (in flight) | §7 |
| 9 | torch + **non-negativity λ=10** | `p2_nn10.0_s4{2,3}` | ~0.904 (in flight) | §7 |

† torch rows are FB's **zero-shot relabel** protocol, not native GC — not
interchangeable with rows 1–6. Mean over evals ≥250k.

**Other environments** (canonical phase 2, native GC, 500k):

| env | anchor | phase 1 | phase 2 | Δ |
|---|---|---|---|---|
| cube-single | 0.928 | 0.967 | 0.955 | −0.01 |
| cube-double | 0.265 | 0.549 | **0.358** | **−0.19** |
| scene | 0.281 | 0.484 | **0.342** | **−0.14** |
| puzzle-3x3 | 0.117 | 0.200 | 0.198 | flat (degenerate cell*) |

\* every seed scores exactly task1=1.0, tasks2–5=0.0; anchor task1=0.74. All
methods fail 4/5 tasks; the "2×" is one task's margin.

**⇒ Phase 2 costs success everywhere except cube-single.** The "visitation phase
is free" reading was drawn when cube-single was the only completed column.

---

## 3. Variants tried — measure quality (SMB Wing B)

From `analysis/rescuing-fb/smb/smb_matrix_v1.csv`. `fp` = phase 1, `vis` =
diagonal phase 2, `zx` = separated phase 1, `visz` = separated phase 2.

| row | B1 Spearman | B1 rel-err | B2 union defect | B3 Spearman | B3 slope |
|---|---|---|---|---|---|
| anchor (GC-IQL) | 0.26–0.31 | 2.05–2.11 | 3.76–5.79 | −0.04…−0.14 | 4.1–5.5 |
| `fp` (phase 1) | −0.01…0.39 | ~1.00 | **0.048–0.056** | 0.23–0.42 | 652–669 |
| `vis` (diag p2) | **−0.57…−0.06** | 1.07–1.10 | 2.17–3.85 | 0.33–0.34 | 8.2–9.3 |
| `zx` (sep p1) | 0.11–0.14 | 0.98 | **0.029–0.043** | **0.70–0.76** | 1336–1505 |
| `visz` (sep p2) | **0.33–0.37** | **0.48–0.52** | 0.96–1.48 | 0.65–0.75 | **25–28** |

**Reading:** `visz` (separated phase 2) is the **best-calibrated** row — the only
one with positive B1 Spearman *and* rel-err below 1 *and* a slope within ~25× of
sane. The **diagonal** phase 2 (`vis`) is the worst: B1 Spearman goes **negative**
(−0.569 on s1). Slope 652–1505 on `fp`/`zx` means those objects have no
meaningful scale at all.

### 3.1 Direct calibration test (spec 10 L1, torch phase 2)

Model set-mass vs MC occupancy, 32 starts × 3 rollouts, H=200, both seeds:

| half-width | MC | model/MC (exact N̂) | Spearman |
|---|---|---|---|
| 0.4 | 0.009 | **215× / 171×** | +0.09 |
| 0.8 | 0.16 | **72× / 66×** | +0.17 |
| 1.6 | 0.64 | **100× / 92×** | +0.30…0.34 |

- **`Pr[M̃ < 0]` = 45.0% / 45.9%** (phase 2); 33.8% / 35.5% on phase 1 ⇒ **phase 2
  makes negativity worse.** A measure is non-negative by definition.
- Total mass `M̂(everything)` = 190 vs target `1/(1−γ)` = 100 ⇒ ratio layer is
  ~right; the **allocation** is wrong, not the scale.
- Trained N̂ head vs exact kernel counts: corr **0.994**, median rel-err **23.7%**,
  **+29% bias**. The *easy* half (pure counting) is only this good.
- Gate was ≤15% rel-err. Result: 6,500–21,500%. **Gate fails; L2/L3 not licensed.**

Consistent with §3: SMB's B3 slope of 25–28 for `visz` and this L1's 66–215× are
the same statement in different units — best-in-class is still ~25–100× off.

---

## 4. The two axes disagree, and why

Task success scores the **actor**, which consumes only `argmax_a⟨F,z⟩`. A critic
can order actions perfectly while being an arbitrary function of `g`.

Measured (spec 08 E5, 24 states × 150 goals × 16 actions):

    Pr[argmax_a Q = argmax_a M] = 0.295 / 0.299     (bar was >0.9)
    ICC(R | g)                  = 0.242 / 0.292
    within-goal CV of R         = 1.35 / 1.75

**⇒ The actor disagrees with its own occupancy measure on ~7 of 10 pairs.** A
0.89 success rate certifies `Q`, not `M`. This also refutes the pointwise
action-independence of `R` that the `M = Q·R` factorization assumes.

**Never quote phase-2 success as evidence the measure works.**

---

## 5. Results that contradict their own pre-registration

| # | experiment | pre-registered read | measured | verdict |
|---|---|---|---|---|
| 5.1 | **E2** frozen ψ+actor | ≥0.90 **and ≤** frozen/updated ⇒ loop inert | 0.972 vs 0.940 | ≥0.90 ✓, **"≤" FAILS** ⇒ real actor drift in phase 2 |
| 5.2 | **E3** ridge dense inference | ridge closes the seed split | s0 **0.896→0.472/0.512/0.648**; s1 0.128→0.432/0.464/0.432 | **REFUTED** — rescues bad seed partly, destroys good one |
| 5.3 | **E5** action-independence | argmax agreement >0.9 | **0.295/0.299** | **REFUTED** |
| 5.4 | **T3** occupancy choice (SMB's designed discriminator) | anchor ✗, phase-2 diag ✓ | anchor 0.352/0.608/0.432, `vis` **0.440/0.512** — both ≈chance | **did NOT separate**; spec 11 says this triggers suite redesign — not invoked |
| 5.5 | **T5** corridor reach-avoid | deflection ⇒ avoidance works | retracted: succA `zx 0.888→0.240→0.064` | **task destruction, not avoidance**; no method does reach-while-avoid |
| 5.6 | **L1** calibration | ≤15% rel-err | 6,500–21,500% | **REFUTED** (§3.1) |
| 5.7 | sparsity hypothesis | denser off-diagonal ⇒ better | cube-double conjunctive 0.549 vs 1-cube control **0.150**; scene 0.484 vs cube-only 0.228 | **sparsity does NOT predict success** (30× sparser wins 3.7×) |

---

## 6. Where the record is stale or missing

- `results/summary_agents.csv`, `results/bridge_grafts.csv` — last written
  **2026-07-22**. Missing: all spec-07 phase-2, spec-08, spec-10, spec-11, torch.
  Groups *are* registered in `scripts/results_map.py`; only `--refresh` is owed.
- `DECISIONS.md` — last entry 2026-07-27; torch campaign absent.
- `two-phase-successor-measure-note.tex` — §`tab:suite` was stale (phase 2 shown
  as "training" on 3 rows); **fixed 2026-07-28**, along with the phase-2
  regression, puzzle degeneracy, ridge refutation, E5, and the L1 section.
- Note line ~128 still claims the argmax diagnostic is "not yet run" — it **was**
  (E5, §4).
- Unwritten anywhere: **E8 DDPGBC actor-isolation** — phase1+DDPGBC
  0.884/0.876/0.784 at α=0.3/1.0/3.0 vs anchor+DDPGBC 0.604/0.864/0.868. The
  advantage **reverses at α=3.0**. 1 seed only; stage 2 never ran. This is the arm
  that decides the standing caveat "all comparisons share one actor class."

---

## 7. In flight (2026-07-28, tmux `tfbnn`)

Non-negativity penalty `nonneg_coef`: one-sided squared hinge `mean(relu(−M)²)`.
Penalty **not** softplus/square, because `Q_r = ⟨F, z_r⟩` needs `M` linear in ψ.
Default 0.0 bit-identical (test-pinned, `tests/test_fb_measure_nonneg.py`).

Phase 1 complete:

| arm | success (≥250k) | `frac(M<0)` |
|---|---|---|
| control λ=0 | 0.922 | 33.8% / 35.5% |
| **λ=1** | **0.965** | 2.4% / 2.4% |
| λ=10 | 0.947 | 0.43% / 0.46% |

Phase 2 in flight: λ=1 → ~3.1% negative, λ=10 → ~0.7%, success ~0.90–0.91.
L1 auto-runs on completion → `results/spec10_torchfb/L1_nn{λ}_s{seed}.json`.

**Fork:** ratios → ~1 means sign was the cause and L2 unblocks. Ratios still
~100× with `frac(M<0)≈0` means the mass is in the wrong *place*, and no loss term
reaches that. Note §3.1's Spearman was already only +0.09…+0.35 — weak *ordering*
is independent of sign, so the second outcome is the more likely one.

---

## 8. Open threads for brainstorming

**Never run:** E4 kernel normalization · E6 region/linear eval on `zf0`/`zfan` ·
spec-07 item 5 (position-only-B retrain, d∈{3,5,10,20,50} sweep) · spec-10 **L3**
compromise, **L7** random-region columns, **L8** additive B(X), **L9** phase-3
renewal head, **L10** distillation to `(F_dens, b)` · 5 of 9 E8 envs
(cube-triple/quadruple, puzzle-4×4/4×5/4×6 — configs exist, never launched) ·
zero-shot harness off cube-single (blocked on `env_geometry.py`).

**Live questions the evidence raises:**

> ⚠ **See §9 (2026-07-28 audit) before acting on §§2–5** — several entries above
> are corrected there, including the L1 numbers (harness bug) and the
> first-passage-necessity claim (mislabeled control).

1. **The best-calibrated object is not the one being fixed.** `visz` (separated
   phase 2) beats the diagonal form on every Wing-B metric, but the
   non-negativity campaign (§7) runs on the torch stack. The torch agent samples
   `z` 50% random via `sample_mixed_z(train_goal_ratio=0.5)`, so it is *closer to*
   separated than diagonal — but this has not been verified as equivalent to
   `visz`. Worth pinning before generalizing §7's result.
2. **Ordering, not sign, may be the real defect** — Spearman +0.09…+0.35 and B1
   Spearman negative on `vis`. What would fix *ordering*? L7 (trained region
   columns) and L8 (additive B(X)) are the designed answers and are both unrun.
3. **Phase 2 costs success off cube-single and buys no measure.** Is the visitation
   phase worth keeping at all, versus phase 1 + a separately-trained occupancy
   head (the L9 renewal route)?
4. ~~**Actor drift is real** (§5.1)~~ — DOWNGRADED by §9 audit: matched-seed tie
   (0.952 vs 0.952) on s0; the gap is one seed within checkpoint wobble.
5. **`M = Q·R` is refuted pointwise** (§4) — the curriculum's motivating identity
   holds only in aggregate. What survives of the derivation?

---

## 9. 2026-07-28 five-agent protocol audit — corrections to the record

Five independent auditors re-verified §§2–5 against configs, run dirs, driver
code, and raw logs. Full transcripts summarized here; every correction below was
verified against source.

### 9.1 FATAL: the L1 harness was broken — §3.1 numbers are void

`torchfb_full_measure.py` never called `env.reset()` between MC rollouts, so the
gymnasium TimeLimit counter carried over: **31 of 32 starts truncated after ONE
step**, making the "MC ground truth" a single-step occupancy (verified in
`L1_p2_s4{2,3}.json`: start #0 mc_sum≈162–182, all other starts mc_max=exactly
1.000). The 65–215× ratios, Spearman +0.09…+0.35, and the "6,500–21,500% vs 15%
gate" verdict are ~90% this bug. On the one clean start, model/MC ≈ **2–4×**,
consistent with the model-side total-mass self-check (190 vs ~87 measurable).
The self-check "exonerating the harness" was model-side only — it could never
exonerate the MC side; in hindsight it *condemned* the per-box numbers (implied
MC total ~1–3 vs the ≥86.6 the cube must occupy somewhere).

**What survives of §3.1:** negativity (frac(M̃<0)≈30–46%, model-side, MC-free)
and the N̂-head result (corr 0.994, +29% bias). **Void:** all set-mass ratios and
their Spearman; the L1 gate verdict; the §7 fork as pre-registered. The four
`L1_nn*.json` written 2026-07-28 ~21:10 ran the buggy harness — **do not read
them**. Fixed (env.reset per rollout, audit comment at the site); re-runs on all
6 phase-2 checkpoints → `results/spec10_torchfb/L1fix_*.json`.

### 9.2 "First-passage is required" was never tested

The "occupancy" comparator `CellW0-NoOrtho` has `measure_first_passage: true` in
its recorded flags plus 5 other knobs flipped (expectile on, target_min source,
hindsight test points, offdiag_weight=0, no scale_norm). **No plain
occupancy+no-ortho cell was ever run.** The ortho ON/OFF contrast is a genuine
single-knob diff and stands — but at λ∈{0,1} where the no-ortho config's own
header says λ=1 is ~100× oversized in M-units. Claim to keep: "this ortho term
at this scale is toxic"; claim to drop: "first-passage necessary vs occupancy".
Also: "CellW0 floored" hides a bimodal seed (0.516).

### 9.3 The 3-arm deconfound: strong reading broken, useful reading stands

The hitdirac control used `ortho_coef=1.0`; the 10-seed stock-FB anchor runs
used **`ortho_coef=1000`** (wandb-confirmed). And 0.284 < 0.420 is not
significant (Welch p≈0.12; p≈0.3 vs the window-matched anchor 0.363). **Drop:**
"the privileged ball makes FB WORSE". **Keep:** the control gets nowhere near
0.93 (best seed 0.367; gap >5 anchor-sd) ⇒ the objective *bundle*
{bounded residual, no-ortho, scale_norm} — not the reward support — explains the
jump. The bundle is not internally split by this experiment. Mechanism
(ms_abs_mean 3.70/8.63 matched inits; control→~1000, phase1→0.19) verified
exactly.

### 9.4 Grid/table corrections

- **Row 1 (canonical 0.955) is inflated by cross-attribution**: its third seed
  (0.984) has no run dir; only `vis_s0/s1` exist (0.952/0.928 → **0.940**). The
  0.984 matches a phase-1 FP-NoOrtho re-eval.
- **Row 4 azB "adopted" is unestablished**: run configured to end at 300k (no
  500k data exists); seeds trend oppositely (0.912→0.840 vs 0.712→0.916); and
  ~0.034 of the margin over ZX is the eval-time `actor_z_greedy_frac` knob
  (step-1 eval already 0.812 vs ZX 0.778). Rows 4–6 are also continuations of
  the 500k ZX ckpt ⇒ not iso-budget with rows 1–3.
- **§5.1 E2 "real actor drift" not licensed**: matched-seed s0 is an exact tie
  0.952 vs 0.952; the "≤ fails" rests on one seed's 0.064 gap, within that run's
  own 450k→500k wobble. Relabel OPEN, not contradicted.
- **§5.7 sparsity "wins 3.7×" overstated**: the 1-cube control is scored on
  both-cube success, which spec 08 E7 pre-declared unreadable for controls; and
  ledger's 0.549 doesn't match run finals (0.504/0.540 → 0.522). The
  qualitative refutation (conjunctive does NOT floor) stands.
- **Puzzle footnote nits**: p2 s2 task1=0.98 (not exactly 1.0); anchor task1
  per-seed 0.28/0.74/0.74. Substance unchanged. Note the p2 puzzle seed set
  {1,2} omits exactly the anchor's outlier seed 0.
- **Multi-env table, warm-start/freeze, sparsity matched-ness, anchor
  protocol-match: all verified clean.** Numbers reproduce exactly at 500k.

### 9.5 SMB Wing-B: harness sound, labels need fixing

Wing B does NOT share the L1 bug (proper resets). But: (i) B3 slope for
`fp`/`zx` (652–1505) is a **category error** — first-passage objects (≤1) vs
dwell mass (≤~30), slope≫1 is required, not a defect; (ii) the B3 surrogate
omits the 1/N̂ weight, so the calibrated slope target is ≈1/N̂̄≈**73**, not 1 —
`visz`'s 25–28 is *within ~3× of correct*, which **strengthens** "visz is the
calibrated object"; (iii) "vis worst" holds on both seeds but the −0.57 headline
is one seed (other −0.06); "visz best" holds only on the joint criterion (fp_s0
B1 Spearman 0.392 > visz).

### 9.6 E5: qualitative refutation stands, quantitative bar is proxy-bound

Actions are 16 uniform-random draws (chance 1/16), not actor proposals — the
0.295-vs-0.9 bar is proxy-dependent, and neither argmax is the actor's. The
pointwise R(g) action-independence refutation (ICC 0.24–0.29) does not hinge on
the proxy. Airtight version = re-run with actor-proposed candidates.

### 9.7 Silent omissions now on the record

- E6 arms finished, never reported: `FP-ZX-zfrac0` zf0 0.868/0.828@500k;
  `FP-ZX-zfracAnneal` zfan **0.940**/0.772@500k (zfan_s0 would rank near the
  top of §2).
- The diverged occupancy-TD phase-2 (`Occ-Phase2-{FrozenPsi,TrainPsi}`, C1/C2,
  DECISIONS 2026-07-23) is absent from §2's "everything tried".
- torch hitdirac control arms absent from §2.
- T3's pre-registered suite-redesign trigger was never invoked; instead cap-q
  winsorization was tuned against the failed discriminator (governance smell).
- E8 DDPGBC α-reversal is 1 seed — do not cite until seeded.

### 9.8 Non-negativity campaign outcome (2026-07-28 ~21:10, logs backed up)

Phase 2 with `nonneg_coef`: success (mean last 6 evals) λ=1: 0.920/0.957,
λ=10: 0.897/0.917; final train neg_frac λ=1: 3.4%/2.9%, λ=10: 0.71%/0.60%
(vs control ~45%). Success preserved, negativity ~killed.

### 9.9 CORRECTED L1 (`L1fix_*`, fixed harness, 32 starts × 3 rollouts, all live)

Model set-mass vs MC occupancy, mean over 12 boxes/hw; "head" = trained N̂
end-to-end (the actual phase-2 object); MC tail γ^H=0.134 unmeasured ⇒ true
MC is ~1.10–1.15× larger than shown, i.e. divide ratios by up to 1.155:

| arm | hw0.4 exact/head | hw0.8 exact/head | hw1.6 exact/head | Spearman (0.8/1.6) |
|---|---|---|---|---|
| **λ=0 s42** | 3.22 / 1.94 | 1.92 / **1.10** | 2.33 / **1.23** | +0.73 / +0.79 |
| **λ=0 s43** | 2.03 / 1.22 | 2.08 / **1.16** | 2.18 / **1.14** | +0.72 / +0.75 |
| λ=1 s42/s43 | 4.6–4.9 / 2.5–2.6 | 5.2–5.9 / 2.5–2.9 | 5.9–6.6 / 2.8–3.1 | +0.58–0.66 |
| λ=10 s42/s43 | 5.5–6.7 / 3.0–3.6 | 5.7–6.0 / 2.8–2.9 | 6.1–7.1 / 2.9–3.3 | +0.60–0.72 |

**Readings (upend §3.1 and the §7 fork):**
1. **The λ=0 phase-2 object + trained N̂ head is approximately calibrated at
   the set level**: 1.10–1.23× at hw 0.8/1.6, ≈1.0× after tail correction —
   the spec-10 15% gate is ~met there (small boxes hw0.4 remain 1.2–1.9×).
   Ordering 0.72–0.79. "The object is not a measure" is RETRACTED at set
   level; the true defects are pointwise (45% negative values that
   sign-cancel inside boxes), small-box regime, and ordering precision.
2. **The non-negativity penalty BREAKS calibration** (2.5–3.6× head-ratio
   inflation, monotone in λ) while fixing sign and preserving success. The
   §7 fork resolved a third way: sign was never the calibration problem —
   sign and scale trade off under the hinge because nothing conserves total
   mass. Do not adopt `nonneg_coef` as-is.
3. **L2 (reward-integration readout) is now plausibly licensed** on the λ=0
   object for rewards supported on ≳0.8-halfwidth scales.

### 9.10 L2 zero-shot reward test (spec 12, 2026-07-28/29, `L2_{A,B}_p2_s4{2,3}.json`)

48 never-trained rewards (balls ε∈{0.4,0.8,1.6}, gaussians σ∈{0.4,0.8},
two-zone), naive `reward_inference` readout, λ=0 checkpoints, eval-only.

**Stage A (prediction, 96 pts/family/seed).** Spearman(pred_raw, MC):
ball0.4 **0.85/0.90** ✓ · ball0.8 **0.80/0.81** ✓ · ball1.6 0.72/0.81 ± ·
smooth0.4 0.75/0.86 ± · smooth0.8 0.62/0.75 ✗ · twozone 0.73/0.69 ✗.
Scale gate fails EVERYWHERE at 0.014–0.03 — **which is exactly N̄≈0.0137
(L1's mean kernel mass): the naive readout's missing 1/N̂ density division,
quantitatively confirmed.** The measure orders returns; the readout mis-scales
them by precisely the predicted factor.

**Stage B (control, gap_frac = (zr−floor)/(ceiling−floor), s42/s43).**
ball0.4 **0.87/0.74** ✓ · ball0.8 **0.70/0.61** ✓ · twozone **0.84/0.89** ✓
(near the better single-zone drive from a two-zone reward — composition
works) · smooth0.4 0.54/0.46 partial · smooth0.8 0.23/0.08 ✗ ·
ball1.6 −0.10/0.24 ✗ (zr≈zrand: no steering; note floor is high — a 16cm
ball rewards random behavior, so the gap itself is small).

**Reading: zero-shot transfer WORKS for spatially concentrated novel rewards
(incl. composition) and degrades with reward bandwidth.** Suspected mechanism:
broad rewards average many B's ⇒ z_r collapses toward the dataset-mean
direction (density bias again) and/or leaves the actor's trained z-manifold.
**Next cell (cheap, eval-only): re-run with `reward_inference(weight=1/N̂)`**
— the same density correction that fixed L1's scale; it may rescue both the
scale gate and the broad-reward steering. If broad rewards still fail with
the corrected readout, the defect is the actor's z-manifold, not inference.

**RESULT (`L2w_*`, 2026-07-29): the fork resolved — actor z-manifold is the
remaining defect.** Prediction scale FIXED: 0.79–2.2 (was 0.014–0.03), gate
~met. Prediction ordering mixed (s42 broad families up, ball1.6 0.72→0.83,
smooth0.8 0.62→0.81; s43 mostly down — the 1/N̂ weights reach 5e4 at the floor
and inject variance). But **control did NOT improve anywhere and regressed
where it worked**: twozone 0.84/0.89→0.46/0.46, ball0.8 0.70/0.61→0.50/0.46;
ball1.6/smooth0.8 still ≈floor. ⇒ Split readouts: **naive z for acting,
1/N̂-weighted for return prediction.** Steering broad rewards needs the
*actor* trained on such z's (z law today = 50% random gaussian + 50% B(single
goal); corrected broad-reward z's live off that manifold). That is a
training-side experiment (extend `sample_mixed_z` with synthetic-reward
inference z's), not an eval fix.

### 9.11 Reward-z law campaign (2026-07-29, LAUNCHED, tmux `rzlaw`) — PRE-REGISTERED

The training-side cell 9.10 called for. `agents/fb/agent.py` gains a third
component in `sample_mixed_z`: z's drawn exactly as the zero-shot readout draws
them, `z = project_z(Σ_i r_i B(s'_i))` for a random synthetic reward `r` over the
batch's next-states (`_draw_reward_z`). `reward_z_ratio=0.0` is byte-identical to
the old law (test-pinned + empirically reproduced: `fb_loss=0.1662` at step 100).

Phase 1 is UNCHANGED and reused from `p1_s4{2,3}`, so the ONLY difference from the
control `p2_s4{2,3}` is phase 2's z law.

| arm | family | ratio | run dir |
|---|---|---|---|
| control | — | 0.0 | `p2_s4{2,3}` (already run) |
| rzK25 | kernel (balls/gaussians/two-zone, random centers+radii) | 0.25 | `p2_rzK25_s4{2,3}` |
| rzK50 | kernel | 0.50 | `p2_rzK50_s4{2,3}` |
| rzR25 | rff — `relu(cos(w·c+φ))` random nonneg fields | 0.25 | `p2_rzR25_s4{2,3}` |

`kernel` shares the eval families' FORM (random params) and is the upper bound:
can the actor learn broad-reward z at all? `rff` is a DIFFERENT functional form,
so L2's ball/gaussian/two-zone families stay held out and a win there is not
family-circular. Both arms matter; report them separately.

**Pre-registered reads (write the verdict here either way):**

- **PRIMARY** — stage-B `gap_frac` on the BROAD families rises: smooth0.8 from
  0.23/0.08 and ball1.6 from ≈0 to **>0.4**, while the concentrated ones hold
  (ball0.4 ≥0.7, twozone ≥0.7). Partial credit is a partial result, not a win.
- **GUARD 1** — task success (eval.csv, zero-shot relabel protocol) stays within
  one seed-sd of the control's 0.887. Buying breadth with control is not a win;
  it is §2's "phase 2 costs success" trade in a new place.
- **GUARD 2** — L1 set-mass ratio stays ~1.1–1.2× at hw 0.8/1.6 and Spearman
  ≥0.7. The measure must not decalibrate to fit the wider z set.
- **NULL** — if rzK (matched form) ALSO fails, the actor's z-manifold is NOT the
  defect and the next suspect is F's capacity to be accurate at averaged z (the
  bilinear rank bound), which no sampling law can fix.
- If rzK wins and rzR does not, the claim is **family-conditional** and must be
  reported as "trained on the eval family's form", not as general zero-shot
  transfer.

**Early training signal (100-step smoke, seed 42, not a result):** `neg_frac`
falls monotonically with reward-z mass — 0.374 (control) → 0.354 (K25) → 0.301
(R50) — i.e. broader z coverage eases the negativity defect §9.9 flagged, without
the nonneg hinge's 2.5–3.6× mass inflation. Worth checking at 500k.

**Wiring bug found and fixed en route:** `train.py`'s `shared` dict is an explicit
whitelist, so a knob present in `configs/agent/fb.yaml` but absent there is
accepted by hydra and **silently dropped**. Caught because two runs with
different `reward_z_family` produced byte-identical losses. Now pinned by
`tests/test_reward_z_law.py::test_train_py_propagates_the_knobs`. Any past cell
whose knob was added to a yaml without touching `shared` is suspect.

### 9.12 Behavior baseline for the L2 rewards — "broad ≈ floor" was partly a FLOOR artifact

§9.10 reported the broad families at ≈floor (smooth0.8 0.23/0.08, ball1.6 −0.10/0.24)
and concluded "no steering". That used `z_rand` as the floor: a trained actor under
**one** random task vector (`sample_z(1)`, drawn ONCE outside the reward loop —
n=1, shared by all 48 rewards). Adding the missing **dataset-behavior baseline**
(pure numpy over 1000 episodes × H=200, γ=0.99, same reward instances, seed 0;
`results/spec10_torchfb/L2_behavior_baseline.json`) changes the reading.

| family | behavior | `z_rand` | `z_r` | `z_goal` | gap vs `z_rand` | gap vs behavior |
|---|---|---|---|---|---|---|
| twozone | 2.04 | 2.09 | 15.83 | 18.11 | 0.86 | **0.86** |
| ball0.4 | 1.21 | 0.91 | 15.57 | 19.13 | 0.80 | **0.80** |
| ball0.8 | 5.66 | 5.94 | 21.84 | 30.10 | 0.66 | **0.66** |
| smooth0.4 | 2.75 | 3.67 | 11.38 | 19.07 | 0.50 | **0.53** |
| smooth0.8 | 11.99 | 16.81 | 19.43 | 34.19 | 0.15 | **0.34** |
| ball1.6 | 25.07 | 35.58 | 37.53 | 53.06 | 0.07 | **0.45** |

The two floors agree on the four concentrated families (so the reported numbers
there stand unchanged) and **diverge on exactly the two broad ones**: on ball1.6 a
random z earns 35.58 while the offline behavior earns 25.07 — a 16 cm ball rewards
an actor that wanders. **`z_r` beats the offline behavior on EVERY family**,
including the broad ones; what it fails to beat is an arbitrary task direction.

**Amend §9.10:** keep "transfer degrades with reward bandwidth" (0.66–0.86
concentrated vs 0.34–0.45 broad). **Retract "≈floor / no steering"** — that phrasing
overstated it. Caveat: dataset episodes start from the data distribution, policy
rollouts from `env.reset()`; the behavior column is ±, not exact.

**⚠ The §9.11 pre-registration is NOT amended.** Its thresholds are stated in
`gap_frac` vs `z_rand` and stay that way — rebasing a pre-registered metric after
seeing the baseline is precisely what this ledger exists to prevent. Report the
rzlaw verdict on the pre-registered metric, and this one alongside it.

**Baselines still missing for these rewards:** (1) competing methods — GC-IQL
anchor / stock FB have NEVER been run through this harness, so every L2 number is
the SM against its own oracle with no cross-method comparison; (2) a random-ACTION
policy (distinct from random-z); (3) `z_rand` at n>1; (4) a true per-reward optimum
— `z_goal` is "go to the center", which is not optimal for a broad gaussian, so the
ceiling is understated for exactly the families that score worst.

### 9.11b RESULT — reward-z campaign (2026-07-29, complete, all 24 stages exit 0)

Verdict against §9.11's pre-registration, no thresholds moved.

**Zero-shot control, `gap_frac` vs `z_rand` (the pre-registered metric), mean of 2 seeds:**

| family | control | rzK25 | rzK50 | rzR25 |
|---|---|---|---|---|
| ball0.4 | 0.80 | 0.87 | 0.87 | 0.79 |
| ball0.8 | 0.66 | 0.71 | **0.95** | 0.73 |
| twozone | 0.86 | 0.92 | **1.35** | 1.11 |
| smooth0.4 | 0.50 | 0.49 | **0.78** | 0.55 |
| **smooth0.8** | 0.15 | 0.27 | 0.33 | 0.12 |
| **ball1.6** | **0.07** | **0.53** | **0.49** | **0.55** |
| *concentrated (3)* | 0.773 | 0.831 | **1.056** | 0.874 |
| *broad (2)* | 0.112 | 0.400 | 0.411 | 0.331 |
| *all 6* | 0.507 | 0.630 | **0.795** | 0.640 |

- **PRIMARY: HALF MET.** `ball1.6` clears decisively — −0.10/0.24 → 0.47–0.55 on
  every arm and both seeds, all above the pre-registered 0.4. `smooth0.8` does
  **not** clear: best arm rzK50 is 0.239/0.428 (mean 0.33) against a >0.4 bar.
  Concentrated families were not merely preserved but improved.
- **GUARD 1 PASS, with a gain.** Success (≥250k, relabel protocol) control
  0.887 (0.867/0.907, seed-sd 0.028) → rzK25 **0.948**, rzK50 **0.950**, rzR25
  **0.930**. The z law does not cost control; it appears to help by ~2 seed-sd.
  2 seeds — suggestive, not established.
- **GUARD 2 PASS.** L1 `ratio_head` at hw 0.8/1.6: control 1.10/1.16, 1.23/1.14 →
  rzK25 1.10/1.19, 1.18/1.17 (identical); rzK50 and rzR25 1.12–1.31 (marginally
  higher). Spearman unchanged: hw0.8 0.67–0.71 vs control 0.68/0.72; hw1.6
  0.82–0.89 vs control 0.85/0.82. No decalibration.
- **NULL NOT TRIGGERED.** The matched-form arm worked ⇒ **the actor's z-manifold
  WAS a real defect**, as 9.10 inferred. This is the first training-side
  confirmation of that mechanism.
- **NOT family-circular.** `rzR25`, whose training rewards are `relu(cos(w·c+φ))`
  fields — a different functional form from every eval family — matches or beats
  the matched-form arms on ball1.6 (0.55 vs 0.53/0.49) and is the best arm on it
  under the behavior-normalized view (0.74). The fix is populating the broad-z
  region at all, not memorizing the eval family's shape. Report as general.
- **Dose-response is real:** rzK50 > rzK25 on all-6 (0.795 vs 0.630), driven by
  the concentrated families and smooth0.4.
- **twozone 1.35 (rzK50) beats the single-goal oracle ceiling by 35%** — the
  ceiling is the better of the two zones, so exceeding it is exactly what
  composition should do. Strongest composition evidence to date.

**Behavior-normalized (§9.12 denominator), mean of 2 seeds:** control → rzK50 —
ball1.6 0.44→0.59, smooth0.8 0.32→0.39, twozone 0.86→1.39, ball0.8 0.65→0.95.

**Prediction (L2-A) roughly unchanged**: smooth0.8 improves (0.68 → 0.74/0.78 on
rzK50/rzR25), twozone degrades on rzK50 (0.71→0.53), rest flat.

**⚠ The negativity signal did NOT survive.** §9.11's 100-step smoke showed
`neg_frac` 0.374→0.301; at 500k all arms sit at **0.43–0.47** against the control's
~0.45 (measured post hoc). **No improvement in negativity.** The smoke reading was
a transient of the warm-start, correctly labelled "not a result" at the time —
treat it as refuted, and do not cite it.

**Standing:** adopt the reward-z law (rzK50 as the default arm); broad-bandwidth
smooth rewards remain the one unsolved family, and negativity remains untouched by
every intervention tried so far (hinge inflates scale, z law does nothing).

### 9.13 Cross-method L2 arms (2026-07-29) — and a METRIC DEFECT that invalidates cross-method gap_frac

First run of non-measure checkpoints through the identical L2 stage-B harness
(`p1` = first-passage reaching value; `hitdirac` = FB's own loss at matched ball
support). Two findings, the first methodological and load-bearing.

**⚠ `gap_frac` IS A WITHIN-METHOD METRIC AND CANNOT RANK METHODS.** It normalizes
`z_r` against *that checkpoint's own* oracle ceiling and random-z floor. Those
denominators differ enormously across methods: p1's oracle ceiling is
**1.13–2.00× lower** than p2's on the *same* rewards. So p1 scores HIGHER in
`gap_frac` (mean 0.686 vs p2's 0.507) while earning **less absolute reward**. Any
cross-method conclusion drawn from `gap_frac` is invalid. Within a method (all of
§9.10–§9.12) it remains sound.

**On absolute discounted return, the measure beats the reaching value** (2-seed
means, the honest cross-method statistic):

| family | p1 reach | p2 measure | p2/p1 | p2+rzK50 | hitdirac |
|---|---|---|---|---|---|
| ball0.4 | 10.01 | 15.57 | **1.56×** | 16.09 | 1.52 |
| ball0.8 | 17.22 | 21.83 | 1.27× | 28.56 | 11.70 |
| twozone | 8.45 | 15.82 | **1.87×** | 23.04 | 4.96 |
| smooth0.4 | 9.31 | 11.38 | 1.22× | 15.98 | 6.01 |
| smooth0.8 | 19.71 | 19.43 | 0.99× | 20.80 | 16.43 |
| ball1.6 | 35.79 | 37.53 | 1.05× | 42.23 | 37.90 |
| **mean** | **16.75** | **20.26** | **1.33×** | **24.45** | **13.09** |

**The mechanism is the oracle gap, and it is the dwell semantics.** p2's *ceiling*
— the goal-conditioned oracle drive, handed the answer — earns 1.13–2.00× more
than p1's ceiling on identical rewards. A reaching policy arrives and stops; a
visitation policy stays and keeps accumulating discounted reward. **This is the
first direct evidence that phase 2 buys the thing it was introduced for**, and it
answers the standing objection that phase 2's justification "rests on semantics
not yet demonstrated". Note the gain is largest on the concentrated and
compositional families (1.56×, 1.87×) and vanishes on the broad ones
(0.99×, 1.05×), which is consistent: broad rewards are earned by wandering, so
dwelling buys nothing.

**⚠ NOT ISO-BUDGET.** p2 is warm-started from p1 and trained 500k *more* steps, so
p2 has 1M total against p1's 500k. Part of the absolute gap may be budget. The
*ceiling* gap is less vulnerable — it measures what the policy class accumulates
when handed the goal — but a p1-at-1M control is owed before this is quoted as
settled.

**hitdirac (FB's loss, matched ball support) fails transfer outright.** Lowest
absolute return of every arm (13.09), and on ball0.4 and ball1.6 its oracle drive
earns *less* than a random z (ceiling−floor = −0.61, −2.64), so `gap_frac` there is
undefined rather than merely bad — the source of the nonsensical −5.6e8 values in
the first pass. Reading: FB's loss at matched support does not yield a usable
reward readout at all. This strengthens §9.3's "the objective bundle, not the
support, explains the jump" on a second, independent axis.

### 9.14 THE 2x2 IS AN INTERACTION — the bounded loss FAILS without a region (2026-07-29)

The converse experiment the ICLR draft calls "the main unresolved generalization
question". New knob `hit_identity` (agents/fb/measure/agent.py, 6 tests) makes the
hit matrix exactly I — FB's own Dirac support — under the bounded loss. Phase 1,
500k, 2 seeds, matched to `p1_s4{2,3}` in every other respect.

**cube-single success, mean over evals >=250k:**

| | FB loss (linear diag + ortho) | bounded loss (squared, no ortho) |
|---|---|---|
| **Dirac support (hit = I)** | 0.420 † | **0.108** (0.107/0.110) |
| **ball on cube xyz** | 0.284 (0.367/0.313/0.173) | **0.922** (0.940/0.903) |
| **ball on EFFECTOR slice** | — | **0.823** (0.803/0.843) |

† stock-FB anchor from a different run set (`summary_agents.csv`, ortho_coef=1000
per §9.3) — the one cell not matched to this stack. The other three are torch,
same code, same budget.

**Neither factor is sufficient, and each alone is WORSE than stock FB.** The
bounded objective on a Dirac support collapses to **0.108** — less than a third of
stock FB. FB's loss on the ball gets 0.284. Only the conjunction reaches 0.922.
The 0.92 is therefore a genuine *interaction*, not the additive effect of "a better
loss" that §9.3's "the objective bundle explains the jump" reading implied. §9.3
established the bundle is load-bearing *given the ball*; this establishes the ball
is equally load-bearing given the bundle.

**But the privileged VARIABLE is not what matters — region-ness is.** Rebuilding
the same ball on the **effector** slice (obs 12:15) instead of the cube (19:22) —
an abstraction that is *not* the evaluation variable — still reaches **0.823**,
within 0.10 of the privileged version and 2x stock FB. Caveat recorded before the
run: the effector lies within 0.4 of the cube in 64.5% of dataset states, so it is
a correlated abstraction, not an independent one; read 0.823 as "a related region
structure suffices", not "any region suffices".

**Consequence for the paper.** The current framing ("finite regions turn density
matching into Bellman regression"; "the bounded objective no longer needs the
geometric constraint") is incomplete and, read alone, misleading. The honest claim
is narrower and more interesting: *bounded regression and a finite region
structure are jointly necessary; neither works without the other, and the region
need not be the evaluation variable.* The Limitations paragraph "the converse
experiment has not been run" must be replaced with this result.

L2 transfer evals for all four arms are chained and pending.

**Transfer axis (2026-07-29, same 48 rewards, absolute discounted return — the
only cross-method-valid statistic; 2-seed means over 6 families):**

| arm | return | oracle ceiling |
|---|---|---|
| visitation + ball (cube) | **20.26** | 28.94 |
| bounded reaching + ball (cube) | 16.75 | 21.37 |
| bounded reaching + ball (**effector**) | 16.16 | 19.98 |
| FB loss + ball | 13.09 | 12.93 |
| bounded reaching + **Dirac** | **9.49** | 9.91 |

**The interaction replicates on transfer, not just control.** Dirac is the worst
arm on both axes (0.108 success, 9.49 return); the effector region nearly matches
the privileged cube region on both (0.823 vs 0.922 success; 16.16 vs 16.75
return). The mechanism is visible in the ceilings: the Dirac arm's oracle drive
earns only 9.91, i.e. its policy class cannot earn reward anywhere, so the low
transfer number is not an inference failure but a control failure inherited from
training.

⚠ **Behavior- and z_rand-normalized gaps are DEGENERATE for the Dirac and FB-loss
arms** (ceiling ~= floor ~= behavior ⇒ division by ~0, producing values like 1e8).
Report those two arms in absolute return only. This is the third time the
normalized metric has broken on a weak arm; treat §9.13's warning as a standing
rule, not a one-off.

### 9.15 Negative-MASS diagnostic + valid-measure campaign (2026-07-29 ~23:00, PRE-REGISTERED)

**D1 diagnostic (new, eval-only, 32 starts × 20k dataset goals, L1 query
convention `M = F(s,z,a)·B(g)`, z = project_z(B(g_rand))):** the ~45–50%
negativity is **not cosmetic**. Share of *absolute mass* that is negative:

| ckpt | point neg_frac | **neg mass share** | mean\|M\| neg vs pos | neg_frac by ρ-quartile (low→high) |
|---|---|---|---|---|
| p2_rzK50_s42 | 0.529 | **0.287** | 2.20 vs 6.09 | 0.464 0.486 0.575 **0.589** |
| p2_rzK50_s43 | 0.504 | **0.269** | 2.00 vs 5.52 | 0.448 0.475 0.538 **0.555** |
| p2_s42 | 0.483 | **0.244** | 1.57 vs 4.57 | 0.422 0.461 0.519 **0.531** |

Negative entries carry ~24–29% of total absolute mass at ~36% of the positive
entries' magnitude, and negativity is WORSE in high-density regions — the
opposite of "sign noise where data is thin". The set-level 1.1× head calibration
(§9.9) is therefore achieved by large-scale sign cancellation.

**Mechanism hypothesis (the missing algorithmic ingredient):** `M(s,a,·)`
restricted to `span{B_1..B_50}` is a rank-50 approximation of a **sharp boxcar
kernel** (0/1 ball, ε=0.4) over a 3-D workspace with ≫50 distinguishable balls.
The L2-optimal truncated-basis fit of a discontinuous indicator *must* ring
negative (Gibbs / eigen-truncation), and rings hardest where the fit is tightest
(high-density regions) — matching the quartile gradient. Under this reading
{d=50 bilinear, boxcar kernel, plain L2 Bellman regression, non-negativity} is an
over-constrained set: pick three. The hinge (§9.8/9.9) relaxed L2-optimality
without an anchor and paid 2.5–3.6× mass inflation because *nothing conserves
total mass*.

**Arms (all = adopted stack: phase-2 visitation, freeze_b, nhat_head,
reward_z_ratio=0.5 kernel; warm-start `p1_s4{2,3}/checkpoints/final.pt`; 2 seeds;
control = `p2_rzK50_s4{2,3}`, already run):**

| arm | delta vs control | attacks |
|---|---|---|
| mcA | `nonneg_coef=1` + `mass_coef=1` (NEW: per-row 1/N̂-weighted total-mass penalty toward 1/(1−γ)) | the hinge's missing anchor |
| mcB | mcA + `hit_kernel=quartic, hit_eps=0.6` (NEW: smooth compact kernel `(1−(d/h)²)²`) | ringing at the source + anchor |
| mcC | `hit_kernel=quartic, hit_eps=0.6` only | ringing alone (deconfound) |

`mass_coef` penalizes `((1−γ)·mean_j[M_ij/N̂_j] − 1)²` with in-batch
`N̂_j = hit.mean(col j)` clamped at 2/B; identity: `E_{g~ρ}[M(s,g)/ρ(B_ε(g))] =
total occupancy = 1/(1−γ)`. Quartic h=0.6 has ~0.77× the ball-0.4 kernel mass
(comparable hit_rate); **for quartic arms quote head ratios only** — the L1
"exact" N̂ is ball-based and mismatched by construction.

**Pre-registered gates (a "fully-working measure" must pass ALL):**
- **G1 validity**: point neg_frac ≤ 0.10 AND neg mass share ≤ 0.05 (D1 harness).
- **G2 calibration**: L1fix ratio_head ∈ [0.85, 1.30] at hw 0.8/1.6, Spearman ≥ 0.65.
- **G3 control**: success (relabel protocol, ≥250k) ≥ 0.92 (control 0.950, sd ~0.03).
- **G4 transfer**: L2-B gap_frac ball0.4 ≥ 0.7, twozone ≥ 0.7, ball1.6 ≥ 0.4.
- **NULL**: if no arm passes G1 without breaking G2 (hinge-style trade-off
  reappears), the signed-bilinear parameterization itself is the obstruction and
  the next move is structural non-negativity (softplus'd F and B = nonneg
  factorization, needs d≫50) or d128 retrain — not more penalties.
- Verdict to be written as §9.15b either way. smooth0.8 and multi-env are OUT of
  scope for this campaign.

### 9.15b RESULT — valid-measure wave 1 (2026-07-30 02:15, complete, all 6 chains exit 0)

Verdict against §9.15's gates, no thresholds moved. Control columns = `p2_rzK50`
re-parsed with the same scripts.

| gate | control | mcA (hinge1+anchor) | mcB (+quartic) | mcC (quartic only) |
|---|---|---|---|---|
| G1 neg_frac / neg-mass share | 0.50 / 0.27–0.29 | **0.090, 0.096 / 0.013, 0.014 ✓** | **0.078, 0.089 / 0.010, 0.012 ✓** | 0.52, 0.50 / 0.27 ✗ |
| G2 ratio_head hw0.8/1.6 | 1.23/1.19, 1.31/1.24 | 1.15/1.26, 1.29/1.24 ✓ | 1.25/1.33, 1.28/1.31 (1.6 marginal ✗) | 1.22/1.22, 1.06/1.11 ✓ |
| G2 Spearman hw0.8 / 1.6 | 0.68/0.85, 0.69/0.82 | **0.51/0.83, 0.47/0.83 ✗** | **0.50/0.86, 0.48/0.84 ✗** | 0.67/0.85, 0.69/0.85 ✓ |
| G3 success ≥250k | 0.937/0.963 | 0.940/0.953 ✓ | 0.933/0.950 ✓ | 0.957/0.930 ✓ |
| G4 (b0.4 / tz / b1.6) | ✓ | s42 ✗ (b1.6 0.36), s43 ✓ | **✓ both** (b1.6 0.56/0.81) | ✗/✓ |

**No arm passes all four gates strictly.** But the outcome is NOT the §9.15 NULL:
the hinge-style trade-off (2.5–3.6× mass inflation) did NOT reappear — the anchor
pins total mass at 1.002 throughout training and set-ratios stay 1.15–1.33. The
residual defect is ONE number: mid-scale ordering (Spearman hw0.8 0.68→~0.50 on
all four penalty runs; hw1.6 untouched at 0.83–0.86).

**Findings:**
1. **mass_coef works as designed** — train `mass_ratio` 1.002 on every penalty
   arm; sign and scale are solved JOINTLY for the first time (G1 passes with
   calibrated ratios). The §9.9 "sign and scale trade off" is closed by the anchor.
2. **The quartic kernel alone does NOTHING for negativity** (0.50–0.52 = control)
   — refutes the pure-Gibbs mechanism of §9.15; the negativity is regression
   slack that only the explicit hinge removes. Quartic also does not fix the
   small-box ratio (hw0.4 1.9–2.4). Kernel smoothness is NOT a lever here.
3. **⚠ MECHANISM SUPERSEDED by §9.15c RESULT — the rescue is the ANCHOR's, not
   non-negativity's (anchor-only `mcD`, neg_frac 0.27, rescues it just as well).
   The effect below is real; the explanation is wrong.**
   **UNEXPECTED: the penalties rescue smooth0.8** — the family §9.11b left
   unsolved. gap_frac control 0.24/0.43 → mcA 0.40/**0.86**, mcB 0.46/**0.72**.
   With mcB, EVERY L2 family clears its bar on seed 43 and all but none on s42.
   Hypothesis: a non-negative M yields cleaner broad-reward z integration.
4. **⚠ REFUTED by §9.15c RESULT — do not cite.** Anchor-only `mcD` gives
   Spearman hw0.8 0.483/0.468, i.e. MORE ordering damage than hinge-only
   (0.58–0.66). The anchor is the primary cause; dose reduction (`mcE`) duly
   failed its PRIMARY gate. Original (wrong) claim retained for the record:
   ~~Ordering damage is attributable to the hinge, not the anchor (prior
   evidence: hinge-only `L1fix_p2_nn1.0` Spearman hw0.8 0.58–0.66, already
   degraded without any anchor). Dose reduction is the designed fix.~~
5. hw0.4 stays 1.4–1.9× on all arms — rank-limited as hypothesized; d128's target.

### 9.15c PRE-REGISTERED — wave 2 (tmux `mcons2`, launched 02:20) + d128 chains

Arms (same stack/protocol/evals as §9.15): `mcD` = mass_coef=1 alone (deconfound:
if its Spearman hw0.8 ≈0.68 and neg_frac ≈0.5, the hinge is convicted of the
ordering damage and the anchor is exonerated); `mcE` = nonneg_coef=**0.3** +
mass_coef=1 (dose reduction; PRIMARY: G1 pass AND Spearman hw0.8 ≥0.65; if G1
fails at λ=0.3, try λ interpolation once more, else accept mcB and report the
ordering cost). d128 phase-2 chains (GPUs 5/6, after p1_d128* finish ~03:25):
mcB-config on `p1_d128q_s42`, mcE-config on `p1_d128_s42`; d128's pre-registered
target is the hw0.4 ratio (≤1.3 would confirm the rank story) and it may also
lift Spearman. Success bar unchanged (≥0.92).

**Addendum (2026-07-30 00:45, before any campaign result was read):** two
**d128 phase-1 hedges** launched on GPUs 5/6 (tmux `mcd128`): `p1_d128_s42`
(ball) and `p1_d128q_s42` (quartic h=0.6), 1 seed each. Rationale: rank is the
fourth member of the over-constrained set; Gibbs overshoot from a *sharp*
kernel does not vanish with rank (it narrows), so d128 alone is NOT expected to
pass G1 — but d128 is the designed fix for the **small-box (hw0.4) 1.2–1.9×**
residual (a resolution defect), and the structural-nonneg NULL arm needs d≫50
anyway (nonneg rank > rank). Phase-2 d128 will be chained after §9.15b in the
best gate-passing configuration; this addendum changes no gate.

### 9.15c RESULT — wave 2 deconfound: the ANCHOR is the active ingredient, in both the benefit and the cost (runs 2026-07-30, collected 2026-08-05)

**Collection note.** All six chains (mcD/mcE ×2 seeds, d128mcB/d128mcE ×1) ran
to `final.pt` on 07-30 and their L1fix / negmass / L2_B JSONs have been sitting
in `results/spec10_torchfb/` since. No verdict was ever written; §9.15c's
questions stood open for six days while §9.16 consumed the campaign. Numbers
below are read straight from those JSONs, gates as pre-registered, nothing moved.

| gate | control `rzK50` | `mcD` anchor ALONE | `mcE` hinge0.3+anchor | `mcA` hinge1+anchor | d128+`mcB` | d128+`mcE` |
|---|---|---|---|---|---|---|
| G1 neg_frac | — | 0.269 / 0.279 ✗ | 0.148 / 0.157 ✗ | **0.090 / 0.096 ✓** | **0.099 ✓** | 0.172 ✗ |
| G1 neg-mass share | — | 0.117 / 0.115 ✗ | 0.029 / 0.032 ✓ | **0.013 / 0.014 ✓** | **0.014 ✓** | 0.032 ✓ |
| G2 ratio hw0.4 | 1.44 / 1.88 | 1.13 / 1.56 | 1.40 / 1.36 | 1.52 / 1.38 | 1.55 | **1.30** |
| G2 ratio hw0.8 / 1.6 | 1.23 / 1.19 | 1.15 / 1.24 | 1.15 / 1.23 | 1.15 / 1.26 | 1.16 / 1.33 | 1.18 / 1.21 |
| G2 Spearman hw0.8 | **0.679 / 0.693** | **0.483 / 0.468** ✗ | 0.545 / 0.480 ✗ | 0.509 / 0.470 ✗ | 0.568 ✗ | 0.507 ✗ |
| G2 Spearman hw1.6 | 0.85 / 0.82 | 0.79 / 0.80 | 0.86 / 0.84 | 0.83 / 0.83 | 0.83 | 0.82 |
| G3 success | 0.937 / 0.963 | 0.94 / 0.96 ✓ | 0.92 / 1.00 ✓ | 0.94 / 0.95 ✓ | 0.94 ✓ | 0.96 ✓ |
| G4 ball1.6 | 0.47 / 0.51 | **0.62 / 0.85** | **0.62 / 0.85** | 0.36 / 0.82 | 0.67 | 0.85 |
| G4 smooth0.8 | 0.24 / 0.43 | **0.63 / 0.57** | 0.53 / 0.65 | 0.40 / 0.86 | 0.49 | 0.51 |
| G4 twozone | 1.41 / 1.28 | 1.34 / 1.37 | 1.40 / 1.45 | 1.12 / 1.69 | 1.93 | 1.47 |

**mcD verdict — the pre-registration is REFUTED IN BOTH HALVES.** §9.15c said:
"if mcD's Spearman hw0.8 ≈0.68 and neg_frac ≈0.5, the hinge is convicted of the
ordering damage and the anchor is exonerated." mcD gives Spearman hw0.8
**0.483 / 0.468** — the full penalty arms' damage (0.51/0.47), not the control's
0.68 — and neg_frac **0.27**, not ≈0.5. The anchor alone does essentially all
the ordering damage AND by itself already halves the negativity.

**This overturns two of §9.15b's findings:**

- **§9.15b #4 is WRONG** ("ordering damage is attributable to the hinge, not the
  anchor"). It was inferred from hinge-only evidence (`L1fix_p2_nn1.0`, Spearman
  0.58–0.66) without ever running anchor-only. Anchor-only is *worse* than
  hinge-only on ordering. Correct statement: **the mass-conservation anchor is
  the primary cause of the mid-scale ordering loss; the hinge adds little on top.**
- **§9.15b #3's mechanism is WRONG** (the smooth0.8 rescue attributed to "a
  non-negative M yields cleaner broad-reward z integration"). mcD is NOT
  non-negative (neg_frac 0.27) and rescues smooth0.8 just as well
  (0.24/0.43 → 0.63/0.57), and ball1.6 better than any hinge arm
  (0.47/0.51 → 0.62/0.85). The rescue is the **anchor's** doing, not
  non-negativity's.

Both corrections point one way: **the anchor is the active ingredient in the
benefit and in the cost.** The hinge buys sign-cleanliness (G1) on top of it and
nothing else. That makes the validity/ordering trade-off structural rather than
an implementation artifact of the hinge — the identity
`E_g[M/N̂] = 1/(1−γ)` is what a measure *is*, and enforcing it is what costs the
mid-scale ordering. A weaker hinge cannot buy the ordering back, because the
hinge was never what spent it.

**mcE verdict — PRIMARY FAILS on both halves.** Pre-registered PRIMARY was "G1
pass AND Spearman hw0.8 ≥0.65". Got neg_frac 0.148/0.157 (G1 needs ≤0.10) and
Spearman 0.545/0.480. Per §9.15c's own instruction ("if G1 fails at λ=0.3 ...
else accept mcB and report the ordering cost"), and given the mcD result kills
the premise of further λ interpolation, **the λ-interpolation branch is closed
and mcB stands as the adopted validity arm, with the ordering cost reported.**

**d128 verdict — the rank story gets WEAK support only.** Pre-registered target
was hw0.4 ratio ≤1.3. d128+mcE reaches **1.30** (exactly at the bar, one seed)
against d50+mcE 1.40/1.36; d128+mcB reaches 1.55 against d50+mcB 1.92/1.48. So
doubling the rank moves the small-box ratio by ~0.1–0.4 and lands at best on the
line. d128 does NOT lift Spearman hw0.8 (0.568 / 0.507, still ≪0.65), which
§9.15c had hoped for. hw0.4 remains the one defect no arm has closed.

**Standing after wave 2: still no arm passes all four gates.** Best validity =
mcB (G1 ✓, ordering ✗). Best transfer = mcD/mcE (G4 strongest, G1 ✗). The
G1↔G2-ordering trade is now understood to be a property of mass conservation
itself, so "one object that is both a certified measure and a well-ordered
critic" has no remaining pre-registered route on cube-single. This is the
single-object analogue of the multi-env dissociation in §9.16 — and it
strengthens the case for fork (B) there (decoupled actor head): if validity
costs ordering even with one object, asking one object to be both may be the
wrong target.

### 9.16 WHY phase 2 costs success off cube-single — mechanism probed (2026-07-30)

Question: §2's multi-env table shows phase 1 beats every anchor (cd 0.549 vs
0.265, sc 0.484 vs 0.281) but phase 2 gives a chunk back (−0.19 / −0.14). Why,
and is it an HP problem? Probes: eval/train traces of the E7/E8 runs +
checkpoint Q-grid probe (`p2_ranking_probe.py`, scratchpad; p1-500k vs p2-500k,
same seed, 192 states × 64 dataset goals × 16 dataset actions) + dataset dwell /
hit-rate measurement.

**Finding 1 — the damage is an early actor drift, not a warm-start or eval bug.**
Step-1 eval of every phase-2 run equals its phase-1 parent (cd 0.552/0.532, sc
0.496/0.452); success then decays in the FIRST 50–100k steps (cd 0.552→0.368 by
100k) and plateaus ~0.35–0.40 — between the anchor (0.27) and phase 1 (0.55).
cube-single phase 2 never decays (0.976→0.984). Puzzle is flat at its degenerate
0.200 (nothing to lose).

**Finding 2 — on sparse-region envs the visitation fixed point collapses to the
hindsight dwell constant, quantitatively.** Median dwell inside one's own hit
region is ~identical across envs (cs 56, cd 56, sc 37, p3 32 steps), so the
hindsight-diagonal grounding says "M ≈ (1−γ^dwell)/(1−γ)" everywhere. What
differs 28× is the OFF-diagonal grounding rate (random s vs random g at the
training ε): **cs 1.30%, sc 0.29%, p3 0.20%, cd 0.046%**. Predicted dwell mass
vs measured phase-2 `q_abs_mean` (train.csv, final):

| env | offdiag hit rate | dwell-implied mass | measured p2 q_abs | Δ success |
|---|---|---|---|---|
| cube-double | 0.046% | **43.0** | **43.5** (all-positive, batch variation ~1.4%) | −0.19 |
| scene | 0.29% | **31.0** | **32.1** | −0.14 |
| cube-single | 1.30% | 43.0 (ceiling) | **23.4**, 11–18% signed variation | −0.01 |
| puzzle-3x3 | 0.20% | 27.5 | 10.7 | flat (degenerate) |

cd/sc phase-2 Q equals the dwell constant to 3%: F·ψ learned "every (s,g) pair
carries the dwell mass" — no goal- or action-discriminative structure survives
at batch level (the train.csv `1−q_mean/q_abs` statistic is a *negative-mass*
fraction: cd ~1.4% vs cs 11–18%). Note the inversion: the 28×-sparser region
ends with the *largest* Q — pure bootstrap misallocation, the same
mass-in-the-wrong-place defect as §9.9/§9.15, but taken to the constant-function
limit. Phase 1 is protected by the absorbing bound (M≤1, h=1 pins the diagonal
at exactly 1 ≠ dwell mass).

**Finding 3 — the actor drifts because `actor_q_center` renormalizes the
residual to full strength.** The q-term is divided by sg[mean|q−mean(q)|], so a
~flat Q does not silence Q-guidance — it amplifies whatever residual variation
exists (here: misallocated bootstrap noise) to unit scale against bc_coeff=3.
Hence fast drift on cd/sc, none on cs where the residual is real signal.

**Checkpoint-probe caveats:** p1↔p2 action-argmax agreement is low on ALL envs
incl. cube-single (0.19–0.29 on random dataset actions) — consistent with §4/E5;
action-ranking disagreement does not discriminate. The discriminator is the mass
structure above. Per-s variance over the (g,a) grid is ~93–97% goal-main-effect
for every checkpoint — a fixed batch's Q spread is goal-popularity, not action
signal, on all envs.

**Consequences / arms (not launched):**
1. **The HP that fails to transfer is the training-region width ε
   (`measure_hit_spec`), not lr/batch**: ε=0.4 was calibrated on cube-single's
   density; conjunctions collapse the grounding rate 28×. Density-matched ε for
   phase 2 (target ≈1.3%): cube-double ≈ ×2.4 (ε≈0.95/cube), scene ≈ ×1.9.
   Pre-registered read: cd p2 q_abs comes DOWN off 43 toward sub-dwell values
   with restored signed variation, and the 0→100k decay shrinks.
2. **`freeze_actor=true` phase 2** (E2 recipe, cs 0.972) erases the success
   regression by construction and still trains F's visitation semantics —
   correct default for multi-env phase 2 given §4 (success never certified the
   measure anyway).
3. `measure_testpoint_dist='hindsight'` — grounded test columns at a guaranteed
   per-batch rate, independent of region sparsity (asserts: incompatible with
   nhat_head/multiscale).
4. The §9.15b mass anchor is the principled fix for the inflation but its
   in-batch N̂ clamps at 2/B when the hit rate is 0.05% — port to jax only
   together with (1) or (3).
5. Protocol notes for the paper table: p2 is not iso-budget (p1+500k), and the
   multi-env phase-2 claim should be scored on the measure axes once
   `env_geometry.py` unblocks the zero-shot harness — native GC success is the
   wrong axis to pay for.

### 9.16b PRE-REGISTERED — p2fix arms (2026-07-30, tmux `p2fix`, launch_p2fix_arms.sh)

Three single-knob arms against the §9.16 mechanism, on cube-double + scene,
2 seeds each (warm-start = the same E7 phase-1 ckpts as `p2_cd_*`/`p2_sc_*`,
500k, canonical phase-2 recipe otherwise). All six configs smoke-tested 300/300
(freeze mask + warm-splice messages verified in logs). Controls = the existing
`p2_cd_s{0,1}` / `p2_sc_s{0,1}` (cd 0.358, sc 0.342) and phase 1 (cd 0.549,
sc 0.484).

| arm | tag | knob delta | groups |
|---|---|---|---|
| 1 freeze | `p2f_*` | `freeze_actor=true` (E2) | `*-BallVisit-FrzActor-d50` |
| 2 eps | `p2e_*` | `measure_hit_spec` widened to match cube-single's grounding: cd `19:22:0.96,28:31:0.96` (1.21%), scene ×1.9 `19:22:0.76,28:30:0.95,32:34:0.95,36:37:1.37,38:39:1.14` (1.29%); eval unchanged | `*-BallVisit-EpsDM-d50` |
| 3 hindsight | `p2h_*` | `measure_testpoint_dist='hindsight'` | `*-BallVisit-Hindsight-d50` |

**Pre-registered reads (write §9.16c either way):**
- **Arm 1 (sanity/prediction):** native success == phase 1 within eval noise BY
  CONSTRUCTION (actor untouched); prediction: train `q_abs_mean` STILL collapses
  to ≈ the dwell constant (43/31) — the measure defect is not actor-caused.
  If q_abs does NOT collapse under a frozen actor, the collapse is
  actor-feedback-driven and §9.16's data-sparsity story is incomplete.
- **Arms 2+3 (the mechanism test):** PRIMARY = final success within 0.05 of
  phase 1 (cd ≥0.50, sc ≥0.43) AND the 0→100k decay < half the control's
  (cd control drops 0.18 by 100k). SECONDARY (mechanism signature, read even if
  PRIMARY fails): `q_abs_mean` final well below the dwell constant (cd <35,
  sc <27) and negative-mass fraction `1−q_mean/q_abs` > 5% (control: cd ~1.4%).
- **NULL:** if arms 2 AND 3 both keep the collapsed Q (q_abs ≈ dwell constant,
  neg-mass <2%) despite matched grounding, sparsity is NOT the mechanism —
  falsifies §9.16's HP claim; next suspect is the conjunctive region's
  bilinear representability (rank), pointing at d128 instead.
- Caveat pinned in advance: arm 2 changes the trained object (wider region), so
  its measure-axis numbers are not comparable to the ε=0.4 cells; its success
  IS comparable (native eval unchanged).

### 9.16c RESULT — p2fix arms + hindsight scale-out (2026-08-02, wave A complete)

All §9.16b arms finished, plus the wave-A scale-out (tmux `hsscale`, launched
08-01 13:53): hindsight on a third env (puzzle-3x3), on a control env
(cube-single, ZX form), and seed-2 replicates of the two cells that won.
`neg-frac` below is the §9.16b proxy `1 − q_mean/q_abs` off `train.csv`;
`q_abs` is final `training/actor/q_abs_mean`.

| env | cell | ev@0 | ev@100k | final | q_abs | neg-frac |
|---|---|---|---|---|---|---|
| cube-double | phase 1 ×3 | 0.000 | .03–.16 | .504/.540/.604 | 0.50 | 1.4–1.9% |
| cube-double | p2 control ×2 | .552/.532 | .368/.344 | .368/.348 | 43.5 | 1.4/1.9% |
| cube-double | arm 1 freeze ×2 | — | — | .516/.592 | — | — |
| cube-double | arm 2 eps ×2 | — | — | .340/.380 | — | — |
| cube-double | **arm 3 hindsight ×3** | .540/.520/.592 | .576/.612/.684 | **.500/.544/.548** | 54.4–56.5 | 0.0–0.1% |
| scene | phase 1 ×3 | 0.000 | .27–.32 | .468/.480/.504 | 0.44 | 4.2–4.8% |
| scene | p2 control ×2 | .496/.452 | .372/.228 | .332/.352 | 32.1 | 6.8/7.4% |
| scene | arm 1 freeze ×2 | — | — | .504/.524 | — | — |
| scene | arm 2 eps ×2 | — | — | .288/.392 | — | — |
| scene | **arm 3 hindsight ×3** | .516/.524/.544 | .824/.820/.832 | **.916/.904/.960** | 58.1–59.8 | 0.2–0.3% |
| puzzle-3x3 | phase 1 ×3 | 0.000 | .15–.17 | .200 ×3 | 0.25 | 13–19% |
| puzzle-3x3 | **arm 3 hindsight ×2** | .200/.200 | .200/.120 | **.204/.188** | 98–103 | 0.0% |
| cube-single | parent FP-ZX ×2 | — | — | .904/.960 | — | — |
| cube-single | **arm 3 hindsight ×2** | .904/.980 | .744/.780 | **.784/.744** | 72.9/75.8 | 0.0% |

**Verdicts against the pre-registration.**

1. **Arm 1 (freeze) — sanity holds.** Success sits at phase-1 level by
   construction (cd .516/.592 vs .549; sc .504/.524 vs .484). Not re-run here.
2. **Arm 2 (eps) — NULL, as recorded at launch of wave B.** cd .340/.380 vs
   control .368/.348; sc .288/.392 vs .332/.352. Density-matched grounding
   changes nothing. This fired §9.16b's NULL clause and is what put rank (d128)
   next in the queue.
3. **Arm 3 (hindsight) — PRIMARY PASSES, SECONDARY REFUTED.** The primary bar
   (final within 0.05 of phase 1, decay < half the control's) is met on both
   regression envs and then some: cd .531 mean vs phase-1 .549 with the 0→100k
   decay *reversed* (+.04 vs control −.18); scene .927 mean vs phase-1 .484,
   nearly double, decay reversed (+.30 vs control −.12). Seed-2 replicates hold
   on both. **But the pre-registered mechanism signature fails in the opposite
   direction from the prediction.** §9.16b predicted the working arm would drive
   `q_abs` *below* the dwell constant (cd <35, sc <27) with neg-mass *above* 5%.
   Observed: `q_abs` rises far *above* the dwell constant on every env
   (cd 43.5→~55, sc 32.1→~59, puzzle 0.25→~100, cube-single →~74) and negative
   mass *collapses* to ~0 (≤0.3% everywhere, from a 1.4–7.4% baseline).

**⇒ The arm works and §9.16's explanation of why does not.** Hindsight
test-points do not rescue phase 2 by relieving the collapse-to-dwell-constant
described in §9.16 Finding 2 — they leave that fixed point entirely and land
somewhere with much larger, almost purely positive Q. The success gain is real
and replicated; the mechanism is OPEN. Do not cite §9.16's sparsity/collapse
story as the explanation for the hindsight win.

**Scale-out reads (both negative, both pre-registered).**

- **puzzle-3x3 = NULL.** Starts phase 2 at .200 (the degenerate phase-1 value)
  and ends at .204/.188. Hindsight *protects* existing structure — the decay
  the control shows on cd/sc never appears — but it *adds* none. On the env
  where phase 1 never found structure, there is nothing to protect.
- **cube-single control PAYS.** .904/.980 → .784/.744, a loss of ~0.17–0.24 on
  the one env where batch test-points already work. Hindsight is not free and
  must not be made the unconditional default.

**A caution this campaign buys for §9.15.** puzzle-3x3 hindsight reaches
neg-frac 0.0% with `q_abs` ≈ 100 while success stays pinned at the degenerate
.200. Near-zero negative mass together with large Q is therefore NOT sufficient
for a useful measure — it is reachable by a fully degenerate solution. The
valid-measure campaign treats low `neg_frac` as the target; on this evidence
that gate needs a companion that rules out the inflate-everything-positive
escape. (Axis caveat: this `neg-frac` is the train-time actor-Q proxy, not the
`negmass_*.json` set-mass measurement — related but not the same instrument.)

**Open after wave A.** Why hindsight works, given the mechanism is refuted.
Whether the scene result survives on the measure axis (native GC success is the
wrong axis to pay for — §9.16 note 5); the zero-shot harness is still blocked on
`env_geometry.py`. Whether the cube-single cost is recoverable by mixing batch
and hindsight test-points rather than switching outright.

**Wave B (d128 rank) — COLLECTED, verdict in §9.16d below.** Stage 1 complete
(cd .484/.360, sc .444/.480 at 500k). All four stage-2 cells launched 08-02
21:15–21:24 (tmux `d128sc` scene GPUs 2–3, `d128cd` cube-double GPUs 4–5), ETA
~08-03 07:00. Pre-registered reads in the config headers.

**Harness defect found and fixed (08-02).** `launch_d128_2stage.sh` decides its
splice pre-check by grepping for `step 300`, or done/finished/total_time in the
last 5 log lines. `run_gciql` emits a tqdm bar and then echoes the resolved
command, so that marker is never in the tail and the check CANNOT pass. All four
stage-2 cells completed their 300-step pre-check off a complete splice and all
four were reported FAIL and skipped — the campaign would have silently lost every
cell. Replacement `scripts/rescuing_fb/launch_d128_stage2_fixed.sh` gates on the
substance (`[warm_start] spliced …; fresh-init []`, `300/300`, no traceback) and
caches passing logs. Separately confirmed by that pre-check: the d128 warm-start
splice path — never exercised before — works, splicing all five module trees with
`fresh-init []` on all four parents.

### 9.16d RESULT — wave B, d128 rank (2026-08-03, collected)

All four stage-2 cells ran the full 500k
(`exp_spec08/factored-fb/GCIQL-{CubeDouble,Scene}-Flow-Measure-BallVisit-FrozenPsi-d128/`).
Judged against the pre-registered reads in the two `*_ballvisit_d128.yaml`
config headers. `neg-frac` = the train.csv proxy `1 − q_mean/q_abs`, as in
§9.16b/c.

| cell | p1 parent | 50k | 100k | 500k final | q_abs (dwell const) | neg-frac |
|---|---|---|---|---|---|---|
| p2_cd_d128_s0 | .484 | .464 | .452 | **.480** | 27.0 (43.0) | 5.0% |
| p2_cd_d128_s1 | .360 | .400 | .344 | **.260** | 27.7 (43.0) | 3.8% |
| p2_sc_d128_s0 | .444 | .412 | .400 | **.312** | 22.2 (31.0) | 10.4% |
| p2_sc_d128_s1 | .480 | .332 | .328 | **.336** | 21.7 (31.0) | 10.0% |

**PRIMARY (success recovery) — FAIL on both envs.** cd needed ≥0.45 on both
seeds: s0 .480 ✓ / s1 .260 ✗ (mean .370 ≈ d50 control .358). sc needed ≥0.42 on
both: .312/.336 ✗✗, statistically the d50 control (.332/.352), with the same
decay-from-parent shape (s1 has already lost .15 by the first eval at 50k).
Rank does not rescue success. Note also that d128 does not improve *phase 1*
(cd .484/.360 vs d50 .549; sc .444/.480 vs .484), so rank buys nothing on
either axis of the success ladder.

**SECONDARY (mechanism signature) — LARGELY CONFIRMED.** This is the
informative half. At d128 the fixed point *leaves* the dwell constant on both
envs: cd q_abs 27.0/27.7 vs the 43.5-pinned d50 control (threshold <35 ✓✓),
sc 22.2/21.7 vs 32.1 (threshold <27 ✓✓). Signed variation returns on cd
(neg-frac ~1.4% → 3.8–5.0%, straddling the pre-registered 5% line); on sc
neg-frac 10.0–10.4% sits inside the control's existing 6.8–11.0% band, so only
the q_abs half of the sc signature moved. The NULL clause ("q_abs stays pinned
despite 2.6× rank") did NOT fire: **rank d50 was a real capacity constraint
behind the dwell-constant collapse.**

**⇒ Verdict: the mass collapse and the success regression are now DOUBLY
dissociated.** Wave A showed hindsight fixes success while driving q_abs far
*above* the dwell constant (mass not fixed); wave B shows d128 fixes the mass
pinning (q_abs well below the constant, structure returns) while success stays
at control. Neither direction of the §9.16 Finding-2 story survives as a causal
account of the phase-2 success cost. The success-side damage is caused by
something the batch-test-point objective does to the *actor signal* that
hindsight test-points avoid — not by the measure's fixed point degenerating to
a constant. Do not cite the dwell-constant collapse as the reason phase 2
regresses; cite it only as a (real, now rank-attributed) defect of the d50
measure itself.

**Standing recipe after §9.16c+d:** multi-env phase 2 = hindsight test-points
(cd/sc), keep batch test-points on cube-single, and rank stays at d50 for
success-axis work (d128 only if the measure axis needs the extra structure —
measure-axis numbers are not d50-comparable per the pinned caveat).

**Open.** WHY hindsight works is still unexplained (both §9.16b mechanisms now
dead). Wave-B checkpoints exist at 100k intervals if a measure-axis autopsy
(L1fix-style, jax) is ever built; the zero-shot harness is still blocked on
`env_geometry.py`. Cube-single mixing arm (batch+hindsight) remains unlaunched.

### 9.16e PRE-REGISTERED — train/query goal-support mismatch (2026-08-03, band probe)

After §9.16c+d killed both capacity-side stories, the surviving candidate for
BOTH open questions (why batch-tpd phase 2 drifts on cd/sc, why hindsight fixes
it) is a **support mismatch on the goal argument**:

- With `measure_testpoint_dist='batch'`, TD constrains M(s,·) at exactly two
  column families: the paired diagonal (own next-state — hit≈1, the dwell mass)
  and marginal columns, whose grounding rate is the §9.16 number (cs 1.30%,
  sc 0.29%, cd 0.046%).
- The actor ALWAYS queries Q(s,a,g) at `actor_goals` = trajectory-future states
  (`gciql_flow.py` actor_loss) — the near-diagonal band, where the TRUE measure
  is O(γ^Δ)=O(1). Band pairs enter the batch-tpd objective only via the rare
  marginal collisions; on cd that is 28× rarer than cs.
- `tpd='hindsight'` trains the columns on the OGBench value_goals mix (50%
  traj-future) — i.e. **on the actor's query distribution** (at the documented
  price: no importance correction, no longer the ρ-referenced measure; the
  gciql_flow.py knob comment already flags that offdiag_weight=0 + hindsight +
  fp degenerates to exactly bilinear GC-IQL). Hindsight would then work not by
  fixing the measure but by making the actor read on-support values — and
  cube-single pays because its band was already covered by 1.3% grounding while
  hindsight gives up ρ-referenced semantics it didn't need to.
- This also predicts §9.16d's dissociation: d128 adds rank, not band data, so
  mass structure improves while the actor band stays uninformative.

**Probe (eval-only, existing ckpts): scratchpad `band_probe.py`.** Q[s,g,a] on a
fixed (160 states × goals × 16 actions) grid; goal sets = band (same-trajectory
future at Δ∈{10,30,100,300}) vs marg (random states). Primary instrument =
**seed_agree**: mean Spearman of the 16-action profile between independent
seeds of the same arm at identical inputs — trained signal ⇒ seeds correlate;
off-support interpolation ⇒ shuffle floor. Secondary: step_agree (400k vs 500k;
the ONLY instrument on cs batch, 1 seed), contrast, goal_collapse.

**Pre-registered reads (thresholds fixed before running):**
- P1 (mismatch): cd/sc **batch_d50 band seed_agree < 0.15** while **hind_d50
  band seed_agree ≥ 0.35**, gap ≥ 0.25, on both envs.
- P2 (cs covered): cs batch band step_agree ≥ 0.5 (and ≥ its own marg value).
- P3 (capacity can't fix it): cd/sc **batch_d128 band seed_agree < 0.15** —
  within noise of batch_d50 despite §9.16d's mass recovery.
- NULL: batch band seed_agree comparable to hindsight's (gap < 0.10) ⇒ the
  actor-band values ARE trained signal and the mismatch story dies too ⇒
  redirect at actor-loss mechanics (actor_q_center × bc_coeff interaction,
  §9.16 Finding 3, which the freeze arm left standing as an amplifier).
- Caveat pinned: seed_agree measures reproducibility, not correctness — a
  shared bias (e.g. both seeds inheriting the same warm-start parent's band
  values) could inflate it. If P1 passes, the cs-batch comparison and the
  d128 arm (fresh stage-1 parents) are the controls against this.

### 9.16e RESULT — support-mismatch NULL; §9.16f PRE-REGISTERED (2026-08-03)

**§9.16e verdict: the NULL fired — the support-mismatch story is dead.** Band
(trajectory-future-goal) action rankings in the batch-tpd arms are exactly as
seed-reproducible as hindsight's (band seed_agree: cd batch .48 vs hind
.42–.43; sc batch .48 vs hind .48–.51; pre-registered P1 needed batch <.15).
d128 batch matches (.47/.41). Everything downstream of P1 is therefore moot;
data: scratchpad `band_probe.{py,json,log}` + raw grids `band_probe_raw_*.npz`.

**Post-hoc exploration (NOT pre-registered — decisions from it need §9.16f):**
the batch arm's converged Q is indistinguishable from the working arm's on
EVERY static axis probed at data-like actions:
- Cross-arm band ranking agreement batch~hind .35/.34 (cd/sc) — no inversion.
- The certified goal-reaching action (the band goal's own trajectory action a0)
  is ranked equally well by all arms: band a0-rank .67–.69 vs marginal control
  .55–.57, identical across batch_d50 / hind_d50 / batch_d128.
- Band-vs-far separation exists in the batch arm too (cd band q_mean 52 vs
  marg ≈0), and action-local dispersion ratios are arm-equal (.12–.16).
So batch-tpd phase 2 produces a Q whose readable structure is FINE at data-like
actions, yet training against it destroys the actor (freeze arm = no decay).

### 9.16f PRE-REGISTERED — uncorrected exploitation at band goals (2026-08-03)

Surviving asymmetry between the arms, after 9.16c/d/e: with tpd='hindsight' the
measure TD evaluates the ACTOR'S OWN bootstrap action a_next at band columns
(vgoals) every step — any actor-exploited overestimate at a band pair gets
pulled back by the grounded hit indicator. With tpd='batch' no TD constraint is
ever applied at (s, band-goal) under the actor's own actions: the actor can
climb interpolated maxima of Q there, and no corrective signal ever reaches the
exploited points. Classic DDPG over-optimization, localized to the goal
argument. This predicts everything §9.16e saw: Q looks healthy at DATA-like
actions (it is), and the damage lives only at the policy's own actions.

**Probe (eval-only): scratchpad `exploit_probe.py`.** Same band grid; for each
arm's policy π and each referee critic R: adv[π|R] = median (R(s,π(s,g),g) −
R(s,a0,g))/cmag_R, plus ‖π−a0‖. a0 = the certified action (the trajectory
that reaches g). Referees = the other arm's critics.

**Pre-registered reads:**
- P1 (exploitation signature = SIGN FLIP): on cd/sc, batch_d50 policies get
  adv_own > +0.05 while adv under the hind_d50 referee < −0.05; hindsight
  policies show NO flip under the batch referee (≥ −0.05).
- P2 (drift in action space): ‖π_batch − a0‖ > ‖π_hind − a0‖ on cd/sc.
- P3 (rank doesn't close the loop either): batch_d128 policies flip like
  batch_d50 (the TD loop, not capacity, is what closes it).
- NULL: no sign flip ⇒ exploitation story dead too ⇒ the remaining suspects
  are pure training-dynamics (co-drift transients invisible in converged
  ckpts); next step would be dense-checkpoint instrumentation or the
  interventional arm (batch + actor_q_center=False, testing Finding 3's
  amplifier as necessary).
- Caveat pinned: hindsight policies are better a priori (success .90 vs .35),
  so adv comparisons use the SIGN structure (own-endorsed vs referee-rejected),
  not magnitudes; and a referee is itself a learned critic, not ground truth —
  P1 requires the ASYMMETRY (batch flips under hind, hind does not flip under
  batch), which critic disagreement alone would not produce.

### 9.16f RESULT — exploitation NULL; the failure is not in the converged artifacts on the data distribution (2026-08-03)

**The NULL fired.** No sign flip anywhere: adv_own = +0.003…+0.013 ≈ adv under
every cross referee (data: scratchpad `exploit_probe.{py,json,log}`). The
policies' actions at dataset states are BC-pinned and arm-identical:
‖π−a0‖ = .27/.27 (batch cd) vs .27/.27 (hind cd), .28/.29 vs .26/.27 (sc) —
the .35-success batch policy and the .90-success hindsight policy are the SAME
DISTANCE from the certified action at dataset states. P1/P2/P3 all fail.

**Standing conclusion after 9.16c–f (four refuted causal stories: ε-sparsity,
rank-for-success, goal-support mismatch, converged-Q exploitation):** every
static probe of the converged batch-tpd artifacts ON THE DATASET DISTRIBUTION
is indistinguishable from the working arm — values, seed-reproducibility,
cross-arm consistency, certified-action ranking, action-local dispersion,
policy actions themselves. Yet eval success differs .35 vs .90 and the freeze
arm proves actor UPDATES cause the damage. Therefore the failure expresses
only (a) on the policy's own ROLLOUT distribution (states the dataset probe
never touches — consistent with the §4 phase-diagnostic finding that failures
happen post-control-acquisition, i.e. exactly where rollouts leave the data
manifold), and/or (b) in the 0→100k training TRANSIENT (co-drift that the
500k checkpoints have papered over).

**Next discriminators (pre-register before running):**
1. ROLLOUT-distribution probe: roll each arm's policy in the env, collect
   visited states (split by phase: pre/post control acquisition), re-run the
   9.16e/f instruments THERE. If batch-Q's certified-action ranking collapses
   on rollout states while hindsight's holds, the mechanism is localized:
   hindsight's TD loop grounds exactly the columns the policy's own rollouts
   generate (vgoals ~ the behavior distribution), batch's never does.
2. Dense-ckpt transient: rerun p2 batch cd s0 with save_interval=10000 for the
   first 100k; find WHERE (state region, timestep) the actor first leaves the
   parent. Pairs with the freeze arm as its own control.
3. Interventional: batch + actor_q_center=False (Finding-3 amplifier
   necessity; predicted ≈ freeze arm by dilution, so informative mainly if it
   DOES decay).

### 9.16g PRE-REGISTERED — rollout-distribution probe (2026-08-03)

§9.16f option 1, launched same day. Roll each arm's seed-0 policy in the real
env (canonical eval semantics: reset(task_id 1–5), goal=info['goal'], temp 0,
2 episodes/task), split visited states at FIRST OBJECT MOTION (max hit-spec
block deviation from t=0 > 0.1 obs units) into approach (pre) vs
post-control-acquisition (post), pool ≤200 states per (roller, phase), and
re-run the §9.16e/f instruments there with all four critics (batch s0/s1,
hind s0/s1; cs = batch s2 with a 400k step-agree instrument). Candidates per
state: the loaded policies' own actions + marginal dataset + uniform (16).
Script: scratchpad `rollout_probe.py`.

**Pre-registered reads (thresholds mirror §9.16e):**
- P1 (the mechanism): on cd/sc POST-acquisition pools, batch seed_agree
  < 0.15 while hind seed_agree ≥ 0.30 — on BOTH rollers' pools (it is the
  state distribution, not whose policy generated it). PRE pools: both arms
  ≥ 0.30.
- P2 (healthy control): cs batch step_agree ≥ 0.5 on both phases.
- P3 (preference flip appears where it was absent): on post pools,
  pref_batch* > +0.05 AND pref_hind* > +0.05 simultaneously (each critic
  endorses its own arm's action over the other's — the §9.16f flip that
  dataset states did NOT show).
- NULL: batch seed_agree ≥ 0.30 on post pools too ⇒ the rollout-distribution
  story is ALSO dead; the only suspect left is the 0→100k training transient
  (next: dense-ckpt rerun, §9.16f option 2).
- Caveats pinned: (i) pool sizes may be lopsided if a roller rarely acquires
  control — report pool sizes; (ii) first-object-motion is a coarse proxy for
  the §4 phase diagnostic's control-acquisition event; (iii) success rates of
  the rollers double as a sanity gate (batch ≈ .35, hind ≈ .53/.90 expected —
  if not, the rollout harness is wrong, stop).

### 9.16g RESULT — post-phase reads FAIL; a replicated PRE-phase collapse instead (2026-08-03)

Data: scratchpad `rollout_probe.{py,json,log}`. Sanity gate: cs roller 1.00
(10 eps); cd batch .40 / hind .30 (10-episode noise around .36/.53); sc batch
.30 / hind .80 — harness credible.

**Strict pre-registered verdicts:** P1 **FAIL** — no post-acquisition collapse
(batch post seed_agree cd .32/.38, sc .17/.26; hind post .28/.32, .22/.29 —
neither side clears its threshold cleanly). P2 **PASS** — cs step_agree .57
pre / .90 post. P3 **FAIL** — preference advs ≈ ±.005 on all post pools (one
pre pool, sc/hind-roller, shows a real flip: pref_batch_s0 −.17 vs
pref_hind_s1 +.17; single pool, not corroborated by the paired seeds). NULL
clause: ambiguous (cd batch post ≥.30, sc batch post below it).

**POST-HOC finding (not pre-registered; needs its own confirmation before any
design decision):** the arm separation exists but sits in the APPROACH phase,
replicated on 4/4 pre pools across both envs and both rollers: batch
seed_agree .047/.171 (cd), .069/.089 (sc) vs hind .315/.427, .210/.486.
q_mean is NEGATIVE there (−3.6…−8.2): on (approach-state, task-goal) pairs —
far-field — the batch measure is signed noise that even its own seeds cannot
agree on, while hindsight (whose TD columns include 30% random + 20% current
goals evaluated under the actor's own bootstrap actions) retains reproducible
structure.

**Interpretation constraint discovered while reading this:** the recorded
config has `actor_p_trajgoal=1.0, actor_p_randomgoal=0.0` — the actor NEVER
trains on far goals, in either arm, and eval never consults Q. So the
pre-phase collapse has no direct causal path to the success regression; it is
a MARKER that batch-Q degenerates to seed-noise off its grounded support. The
causal question narrows to the one quantity the actor actually consumes and
no probe has measured yet: the local action-gradient ∂Q/∂a at band pairs,
applied 500k times. Coarse rankings over scattered candidates (§9.16e) can
agree while the fine gradient field is noise.

### 9.16h PRE-REGISTERED — action-gradient seed-coherence at band pairs (2026-08-03)

**Probe (eval-only): scratchpad `grad_probe.py`.** On the §9.16e band grid
(dataset states × traj-future goals at Δ∈{10,30,100,300}): per critic, compute
G = ∇_a min(Q1,Q2)(s, g, a) at the certified action a0 and at 4 uniform
actions. Instruments: cross-seed cosine of G within arm (per env, per goal
set: band + marg); gradient magnitude relative to per-critic scale.

**Pre-registered reads:**
- P1: on cd/sc band pairs, batch grad seed-cos is BELOW hind grad seed-cos by
  ≥ 0.2 on both envs (the actor's actual input is where the arms separate).
- P2: cs batch grad step-cos (400k vs 500k) ≥ the cd/sc batch band level +0.2
  (healthy reference).
- NULL: no gap (< 0.1) ⇒ the fine gradient field is ALSO arm-equivalent at
  band pairs ⇒ every static axis of the converged artifacts is exhausted; the
  mechanism is then pinned to the 0→100k TRAINING TRANSIENT alone, and the
  next step is the dense-ckpt rerun (save_interval=10k, §9.16f option 2) —
  no further converged-ckpt probes.
- Caveat: gradients at a0 only sample the field at one point; agreement there
  does not preclude disagreement along the ascent path — but disagreement
  there suffices for the mechanism.

### 9.16h RESULT — gradient-field NULL: the converged artifacts are exhausted (2026-08-03)

**The NULL fired** (data: scratchpad `grad_probe.{py,json,log}`). Band
∂Q/∂a cross-seed cosine: cd batch .394 vs hind .333 (batch HIGHER), sc batch
.323 vs hind .373 (gap .05 < the .1 NULL line); a0-only variant identical
(.30–.40 all arms). cs batch step-cos .755 ≈ cd/sc batch step-cos .70–.78 —
no healthy-reference separation either. The one systematic difference is
gradient MAGNITUDE (batch ~10.2–10.5 vs hind ~5.9–7.0), consistent with the
arms' different mass scales, and q_center normalizes scale away by design.

**Standing conclusion after §9.16c–h (SIX pre-registered probes, six NULLs on
converged-artifact stories):** rank, ε-density, goal-support mismatch,
exploitation, rollout-distribution collapse (as pre-registered), and now the
local gradient field — every axis on which the actor interacts with the
converged batch-tpd artifacts is statistically indistinguishable from the
working hindsight arm. Per the §9.16h pre-registration, the mechanism is
hereby pinned to the **0→100k TRAINING TRANSIENT**: the damage is done during
the phase-1→phase-2 objective migration (absorbing first-passage fixed point →
ball-visit fixed point) while the actor co-adapts to a moving Q, and the
converged artifacts retain no readable scar. The §9.16g pre-phase
irreproducibility (batch far-field = seed noise) stands as a marker of WHERE
batch-tpd leaves Q unconstrained, not as a causal path (actor_p_trajgoal=1.0,
eval never consults Q).

### 9.16i PRE-REGISTERED — dense-transient wave (2026-08-03, tmux `p2dense`)

5 cells, 150k steps, save_interval=eval_interval=10000, log_interval=1000,
same seeds/parents as wave A so the known curves replicate: cd batch+hs (s0,
warm cd_fp_s0), sc batch+hs (s0, warm sc_fp_s0), cs batch (s2, warm
exp_spec06 m_firstpassage_noortho_s2) — cs is the no-decay control.
Launcher: scratchpad `launch_p2dense.sh`; groups `*-Dense`.

**Instruments (run on the 10k ckpt series when it lands, §9.16e band grid):**
(a) the harness eval curve; (b) actor divergence mean‖π_t − π_0‖ (π_0 = the
spliced parent actor); (c) Q ranking churn spearman(profile_t, profile_{t−10k});
(d) train.csv q_abs / neg-frac trajectories at 1k resolution.

**Pre-registered reads:**
- R1 (co-adaptation turbulence): on cd/sc BATCH, (c) shows an early churn dip
  materially deeper than the SAME env's hindsight cell, the success drop
  coincides with or lags (b)'s rise, and cs batch's dip is shallow.
- R2 (objective-migration): the churn-dip window coincides with q_abs
  migrating from the phase-1 absorbing level toward the visitation constant;
  hindsight reaches its fixed point with less ranking churn in the window.
- NULL: batch and hindsight transient dynamics indistinguishable on (b)+(c)
  ⇒ the co-adaptation story falsified as well; next is INTERVENTIONAL only —
  implement `actor_freeze_steps` (freeze actor through the migration window,
  then unfreeze) and run it on cd/sc batch: if decay vanishes, migration
  turbulence is confirmed causally even though observation couldn't see it;
  if not, the q_center×bc interaction arm follows.
- Caveat: 1 seed per cell (transient shapes, not means); any adopted
  conclusion needs the second seed replicated.

### 9.16j PRE-REGISTERED — actor_freeze_steps interventional arm (2026-08-03, tmux `p2frz`)

**Knob shipped (TDD green):** `actor_freeze_steps=N` in gciql_flow.py —
optax.multi_transform gates the actor modules' GRADIENTS to zero (before adam,
so the update is exactly 0 and adam state is effectively fresh at unfreeze)
for the first N optimizer steps; the critic trains from step 0. Default 0
bit-identical; refuses freeze_actor / freeze_critic_ckpt. 3 tests in
tests/test_gciql_phase2_occupancy.py; phase2/warmstart/measure-axes suites
green (20 passed).

**Window choice:** dense cd_batch train.csv shows q_abs migrating 2.7 → ~43
(the visitation fixed point) over ~60k steps, ~93% done by 40k; the dense eval
curves put the success damage inside 0→10k (mid-migration). N=50000 covers
the migration with margin.

**Cells (GPUs 4–7, 500k, standard intervals):** p2frz_{cd,sc}_{s0,s1} =
ballvisit_d50 batch columns + actor_freeze_steps=50000, warm from the same
phase-1 parents as wave A. Control = the existing p2_{cd,sc}_{s0,s1} (decay)
and p2h_* (hindsight recovery).

**Pre-registered reads:**
- P1 (transient story SUFFICIENT — the fix): no decay while frozen (evals at
  10k–50k ≈ phase-1 parent, cd ~.55 sc ~.48), AND after unfreezing success
  stays ≥ parent−0.05 through 500k on both envs. Then batch columns + delayed
  actor = the standing multi-env recipe, ρ-referenced semantics preserved, and
  hindsight's §9.16c win is reinterpreted as shock-recovery.
- P2 (transient necessary but not sufficient): parent level holds during the
  freeze, then decays after 50k. If the post-unfreeze decay has the same
  fast (≤10k) shape as the phase-2-start collapse, the migration story is
  WRONG (the batch critic's actor-gradient is damaging even at its fixed
  point — in tension with §9.16h's gradient-equivalence, revisit); if it is
  slow/partial, the transient is real but a standing defect remains ⇒ next
  arm = batch+hindsight COLUMN MIXING (grounded columns at the actor's pairs
  with mostly-batch semantics).
- NULL/sanity: success during the freeze ≠ parent ⇒ harness defect (the
  §9.16b freeze arm guarantees parent-level under a frozen actor), stop and
  debug.
- Caveat: warm-start optimizer state is fresh in ALL phase-2 arms, so the
  delayed cells differ from control ONLY in the gate; 2 seeds/env.

### 9.16k PRE-REGISTERED — SPECULATIVE composition cell: d128 + delayed actor (2026-08-03, tmux `p2frzd128`)

Launched OPTIMISTICALLY before §9.16j resolves (idle-GPU pipelining; queued
behind the dense wave on GPUs 0–3). The eventual multi-env recipe, if the
campaign's pieces compose, is d128 (fixes the mass structure, §9.16d) +
actor_freeze_steps (fixes the success transient, §9.16j-pending) + ported
validity penalties (§9.15, separate code work). This cell tests the first
composition: ballvisit_d128 + actor_freeze_steps=50000, warm from the WAVE-B
stage-1 d128 parents (p1_{cd,sc}_d128_{s0,s1}), 4 cells, 500k.

**KILL RULE (decided now, before data):** kill the wave iff §9.16j shows the
FAST-P2 signature on both d50 envs (post-unfreeze collapse within ≤10k of
unfreezing — the read that falsifies the migration story entirely; if the
freeze premise is dead at d50 it is dead at d128). Keep it under P1 or
slow-P2 (still informative on the mass axis either way).

**Pre-registered reads:**
- Sanity: 50k evals (frozen window) ≈ the d128 stage-1 parents (cd .484/.360,
  sc .444/.480); deviation ⇒ harness defect, stop.
- C1 (composition works): post-unfreeze success ≥ parent−0.05 through 500k
  AND q_abs settles OFF the dwell constant (cd <35, sc <27, the §9.16d d128
  signature) — success and mass structure healthy in the SAME artifact for
  the first time on these envs.
- C2 (interference): success holds but q_abs re-pins at the constant (the
  delayed actor changes the measure's basin — would be surprising given the
  freeze arm left the collapse intact at d50) or success decays where d50+frz
  did not (d128's weaker/different parents or transient shape) — either way
  the composition needs its own investigation; do not average over it.
- Caveat: d128 parents are 1-seed-weaker on cd (s1 parent .360); read s0/s1
  separately.

### 9.16i RESULT — transient instruments do NOT separate the arms; NULL fires (2026-08-03)

Data: scratchpad `dense_transient_analysis.{py,json,log}` over the 75 dense
ckpts. Success/q_abs context: damage lands 0→10k in batch cells, hindsight
dips then recovers (sc .39→.91), cs flat; q_abs migrates over ~60k.

- **Actor divergence (b): uniform, uninformative.** ‖π_t − π_parent‖ on the
  band grid jumps to ~0.10 by the FIRST ckpt (10k) and stays exactly there
  through 150k in EVERY cell. The healthy cs control has the LARGEST
  divergence (0.13–0.155) with zero success loss. The actor always jumps at
  the phase switch (fresh adam + new objective scale); the jump's magnitude
  carries no information about damage.
- **Q ranking churn (c): INVERTED from R1.** First-interval churn
  (parent→10k, lower = more churn): cs batch **0.314** (deepest, healthiest),
  cd hs 0.506, sc hs 0.553, sc batch 0.590, cd batch 0.714 (shallowest churn,
  decays anyway). All cells re-stabilize to 0.87–0.93 by 30k. R1 predicted
  batch churns deeper — the working cells churn MORE. R2 is moot given (c)
  does not discriminate.

**⇒ R1 FAIL, R2 moot, the §9.16i NULL fires.** Seven observational probe
families (§9.16c–i) now agree: no instrument we have constructed — converged
or transient, dataset-distribution or rollout-distribution, values, rankings,
gradients, policy actions, divergence, churn — separates the damaged arm from
the working arm, except task success itself. Whatever distinguishes where the
actor's phase-switch jump LANDS is not visible in these observables. Per the
pre-registration, the campaign's weight now rests entirely on the
INTERVENTIONAL arms, both already running: §9.16j (d50 + actor_freeze_steps,
tmux `p2frz`) and §9.16k (d128 composition, tmux `p2frzd128`). Timing note
for §9.16j: all transient signatures (jump, churn, q_abs) settle well inside
the 50k freeze window — the window choice is safe.

### 9.16j PARTIAL RESULT — P2: the freeze holds, the unfreeze decays anyway (2026-08-03)

Curves at 250–350k of 500k (final table owed at completion):

| cell | step-1 | 50k (frozen) | 100k | latest |
|---|---|---|---|---|
| p2frz_cd_s0 | .520 | .528 | .392 | .340 @300k |
| p2frz_cd_s1 | .584 | .572 | .332 | .352 @350k |
| p2frz_sc_s0 | .516 | .508 | .404 | .348 @250k |
| p2frz_sc_s1 | .508 | .440 | .264 | .304 @250k |

- **Sanity gate PASSES:** success sits at the phase-1 parent level through the
  full 50k freeze (the knob works; the §9.16b freeze-arm guarantee replicates).
- **P1 FAILS / P2 CONFIRMED:** within 50k of unfreezing (eval resolution),
  both envs decay to the never-frozen control's plateau (~.34–.45 cd,
  ~.28–.35 sc). Shielding the actor through the objective migration does NOT
  prevent the damage: the actor decays when it starts training against the
  batch-column critic AT ITS SETTLED FIXED POINT.
- **⇒ the migration/transient story is falsified as the CAUSE.** The damage
  is a property of the (d50, batch-column) critic's actor-training signal at
  its fixed point — despite that signal being seed-coherent, arm-equivalent
  in ranking/gradient statics (§9.16e/h), and correct on certified actions
  (§9.16f). Reconcile later: static equivalence of the observables we chose
  evidently does not imply equivalence of the CLOSED-LOOP actor-training
  outcome; success remains the only separating observable (§9.16i).
- **Kill rule for §9.16k:** the pre-registered kill condition (collapse ≤10k
  post-unfreeze) is UNRESOLVABLE at 50k eval granularity; the keep-branch
  (slow-P2) applies. §9.16k is moreover now the CRITICAL cell rather than a
  luxury: with §9.16j = P2 and §9.16d = d128-alone decays, the two-factor
  hypothesis (damage needs BOTH an unshielded transient AND a degenerate
  fixed point to prevent recovery) is tested exactly by d128+freeze — if it
  holds parent success post-unfreeze, the recipe is d128+freeze and BOTH
  §9.16d and §9.16j get reinterpreted as single-factor failures; if it
  decays too, batch columns at d50/d128 are dead for actor training on these
  envs and column MIXING is the last batch-semantics option.

### 9.16k PARTIAL RESULT — the composition decays too; batch columns are dead for actor training here (2026-08-03)

Curves at 100–200k of 500k (final table owed):

| cell | step-1 (=parent) | 50k (frozen) | 100k | latest |
|---|---|---|---|---|
| p2frzd128_cd_s0 | .444 | .468 | .376 | .412 @200k |
| p2frzd128_cd_s1 | .356 | .340 | .300 | .308 @150k |
| p2frzd128_sc_s0 | .448 | .460 | .332 | .332 @100k |
| p2frzd128_sc_s1 | .476 | .444 | .304 | .320 @150k |

Sanity passes (frozen window = parent). Post-unfreeze, every cell decays with
the same shape as d50+freeze (§9.16j) — **C1 FAILS**. The two-factor
hypothesis is dead: even a critic with HEALTHY mass structure (d128, §9.16d
signature) at its settled fixed point, reached with the actor fully shielded,
damages the policy as soon as DDPG+BC training against it begins.

**Campaign conclusion (c–k, pending final tables):** on conjunctive envs, the
single load-bearing ingredient for actor training is whether the measure TD
targets are GROUNDED at the actor's goal distribution — i.e. whether the
hit-indicator term actually fires at (state, trajectory-future-goal) columns.
Hindsight columns fire there by construction and the actor thrives; batch
columns fire at 0.046–0.29% and the actor decays, regardless of rank,
ε-density, or transient shielding. Every static equivalence of the converged
critics (e–h) notwithstanding, the closed-loop actor-training outcome tracks
grounding-at-the-actor's-pairs and nothing else we have measured.

**Fork for the next arm (pre-register ONE before launching):**
- (A) Column MIXING: measure_testpoint_dist mixes batch/hindsight columns
  per-column with prob p (~0.25). One object, partially compromised
  ρ-reference (mixture reference measure, known ratio).
- (B) DECOUPLED ACTOR HEAD: keep the measure's TD columns pure batch
  (ρ-referenced, certifiable) and give the ACTOR its own hindsight-grounded
  scalar GC head (the line-3080 note already observes hindsight+fp+
  offdiag_weight=0 degenerates to exactly bilinear GC-IQL) — full semantics
  for the measure AND full grounding for the actor, at the cost of a second
  head and the question of whether the certified measure is then doing any
  control work at all (it would not be — the paper claim changes shape).

### 9.16j FINAL — all four cells at 500k (2026-08-03)

p2frz_cd .328/.272 (control .358 mean, phase 1 .549); p2frz_sc .384/.352
(control .342, phase 1 .484). The §9.16j partial verdict stands unchanged:
freeze holds the parent exactly, unfreeze decays to the control plateau on
both envs. P2 CONFIRMED at completion; no late recovery.

### Spec-14 §7 DELIVERED — jax zero-shot harness multi-env + first table (2026-08-03)

Port of `eval_zeroshot_rewards.py` to env_geometry (delegated agent; cube-single
code paths bit-identical, both documented silent bugs fixed, hindsight
visited-state region centers, checkpoint hit-spec wins). **Regression gate
BIT-EXACT** vs `analysis/rescuing-fb/zeroshot_infonly_s2_RET_k1.json` (dirac
mean .968 float-exact, region mean exact). 6 new unit tests
(tests/test_spec14_jax_port.py), 42 total green. Runner:
scripts/rescuing_fb/run_spec14_jax_table.sh; table:
`python3 scripts/rescuing_fb/spec14_jax_table.py`; JSONs in
results/spec14_jax/. NOTHING COMMITTED yet (incl. the harness + tests).

**First multi-env zero-shot table (500k, retrieval-k1, 25 eps/task):**

| env | ckpt | zero-shot dirac | native GC | region fin/any |
|---|---|---|---|---|
| cd | cd_fp_s0 (p1) | .480 | .504 | .150/.230 |
| cd | p2_cd_s0 (batch) | .312 | .368 | .040/.040 |
| cd | p2h_cd_s0 (hind) | .488 | .500 | .080/.180 |
| sc | sc_fp_s0 (p1) | .520 | .468 | .260/.420 |
| sc | p2_sc_s0 (batch) | .376 | .332 | .210/.350 |
| sc | p2h_sc_s0 (hind) | **.872** | .916 | **.450/.740** |

**Reads:** (1) zero-shot tracks native GC within ~.06 on all six ckpts ⇒
task-inference (z retrieval + goal decode) adds NO failure of its own on
conjunctive envs — the §9.16 batch-p2 regression is the policy, not the
inference stack. (2) The §9.16 pattern replicates verbatim on the zero-shot
axis (batch p2 below its parent on both envs; hindsight ≥ parent; scene
hindsight .872 zero-shot — against the FB-in-our-stack scene anchor of .047,
this is the strongest zero-shot manipulation number the project has). (3)
Caveat: this is the success axis; the G2/G4 calibration gates (L1 ratios / L2
gap fractions) remain torch-side and multi-env-pending. (4) Multi-env region
cells are NOT comparable to cube-single R1–R3 (different protocol, per the
port notes). CSV refresh deferred: results_map.py is dirty from the parallel
session.

### 9.16k FINAL — all four cells at 500k (2026-08-03)

p2frzd128_cd .412/.232 (d128 controls .480/.260); p2frzd128_sc .312/.276
(d128 controls .312/.336). Identical to the unfrozen d128 controls within
seed noise — the partial verdict stands at completion: freeze buys nothing at
d128 either. C1 dead, campaign conclusion (§9.16k partial) unchanged.
