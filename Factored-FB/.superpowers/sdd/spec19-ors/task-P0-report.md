# Spec 19 P0.1 + P0.3 — task report

Scope: `PAPER/Multistep-FB/specs/19-ors-program.md` §1, items **P0.1** (G4
Spearman calibration read) and **P0.3** (CSV bookkeeping for the d18 run
groups). P0.2 (BC-only control) is out of scope for this task.

---

## ⚠ UPDATE 2026-08-06: the original P0.1 FAIL verdict below is WITHDRAWN

An independent adversarial verification
(`.superpowers/sdd/spec19-ors/g4-verification.md`) found a real bug in
`scripts/rescuing_fb/eval_measure_g4_spearman.py`'s model readout, plus a
second-order defect in the Spearman helper. **The original "all 6
checkpoints FAIL, pooled Spearman 0.13–0.18" reading is WRONG and
withdrawn.** Corrected verdict: **all 6 checkpoints PASS** the pooled ≥0.65
gate, with a caveat on fine-scale ordering (below). See "Corrected P0.1
results" further down for the full table and the honest per-multiplier
read — that section supersedes the original "Results" / "Interpretation"
subsections of P0.1 (left in place, unedited, for the record).

**Root cause (readout bug).** Spec 18 §1's own formula, `Q̂_r = (r(x₀) + (w
− κ(x₀))ᵀr) / (1−γ)`, defines `r(x₀)` and `−κ(x₀)ᵀr` as a **de-biasing pair
that cancels** ("exact-r at x₀ replaces the smoothed estimate"), leaving
`wᵀr/(1−γ)` as the quantity that matches the MC ground truth (both count
mass from the state *after* the transition, never `x₀` itself). The
original script zeroed `r(x₀)` but kept the paired `−κ(x₀)ᵀr` /
`−φ(x₀)ᵀr` subtraction — injecting `−1[x₀∈box]/(1−γ) = −100·1[x₀∈box]`
into a 0–100-scale quantity. Boxes containing `x₀` are exactly the
high-MC boxes, so the injected term was **anti-correlated** with the
ground truth. This is what the original report mis-attributed to
"numerical noise" at `mult=4.0` (`model_mass_mean=−0.21` vs `mc_mean=24.9`)
— it was the subtracted indicator, not noise; corrected, the magnitude is
32.9 vs 24.9 (calibrated in scale, not just rank).

**Root cause (Spearman bug, second-order).** `measure_readout_head.spearman`
uses ordinal ranks (`np.argsort(np.argsort(x))`) with no tie averaging.
~60% of the probe's MC values are exactly 0.0 (box never entered); ordinal
ranking assigns that tie block an arbitrary rank permutation by array
order instead of the tied average rank, measurably suppressing the
statistic on this data.

**Fix applied** (`scripts/rescuing_fb/eval_measure_g4_spearman.py`):
1. Readout now computed as `wᵀ1_box(U_P)/(1−γ)` (density) /
   `pᵀ1_box(protos)/(1−γ)` (partition) — the same `compute_density_primary`
   / `compute_Qr_from_dots` calls, but with the second operand
   (`kappa_x0`/`phi(x0)` term) zeroed instead of `r_x0`. One line changed
   per branch, per the verifier's own recommended fix; the `−κ(x₀)`/`−φ(x₀)`
   term was **removed**, not "also zeroed alongside `r(x₀)`" (that would
   still be the broken pair-splitting, just relabeled).
2. Added `spearman_tie` (average-rank, via `scipy.stats.rankdata`) as the
   **gating** statistic. The old ordinal `spearman` is kept alongside,
   reported as `spearman_ordinal_uncorrected`, for reference/comparability
   with the pre-fix numbers only — it no longer decides PASS/FAIL.
3. Added `tests/test_eval_measure_g4_spearman.py` (8 tests, all green under
   `.venv/bin/python -m pytest`): `spearman_tie` sanity floor/ceiling
   (MC-vs-itself ≈1.0, monotone transform ≈1.0, pure noise ≈0, nan-guards),
   a direct demonstration that tie-correction measurably beats ordinal
   ranking on ~60%-tied data (the exact shape of this probe), and two
   regression pins on the readout formula itself (`compute_density_primary`
   / `compute_Qr_from_dots` called with the buggy vs. fixed operand
   zeroed must disagree whenever the box overlaps `x₀`) — so this class of
   bug cannot recur silently.
4. Re-ran all 6 checkpoints (`--starts 24 --mc 2 --horizon 200 --boxes 12`,
   density `--pool-size 16384`, same seed/flags as the original run).
   Re-verified bit-identical on repeat (`diff` empty, both before and after
   the fix).

Everything below this point in P0.1 is the ORIGINAL (now-superseded) run;
see "Corrected P0.1 results" for the current numbers.

## P0.1 — G4 Spearman calibration read

### Protocol used, and its provenance

The gate is stated identically in three places: spec 17 §6 ("G4 (ordering,
both): Spearman of MC region/cell masses ≥ 0.65 on the post-hoc calibration
read"), spec 18 §7 ("G4 (ordering, carried from spec 17): ... Run it for
part K256 too — it is still owed."), spec 19 §1 P0.1 (restates the debt).
**This gate had never been run on the jax gciql_flow stack before this
session** — grepping `scripts/` and `tests/` for `spearman` found the
machinery, not a prior result file for these checkpoints.

Reference implementation identified and reused (not reimplemented):
`scripts/rescuing_fb/torchfb_full_measure.py` part B ("NUMERATOR /
calibration") — the exact G2/G4 Spearman protocol the valid-measure campaign
built and reported in `PAPER/rescuing-fb/PHASE2-LEDGER.md` §9.15 (its own
`spearman_exact`/`spearman_head` per half-width, e.g. the §9.15c table's "G2
Spearman hw0.8/1.6" rows). That script is TORCH-only (imports `torch`,
`agent.model.backward_map/forward_map`) and cannot be imported into the jax
venv, so its protocol was **translated, not ported**, into a new script:

**`scripts/rescuing_fb/eval_measure_g4_spearman.py`** (new file; full
rationale in its module docstring). It reuses, live, without reimplementing:
- `env_geometry.py`: `geometry_for`, `episode_bounds`, `resolve_dataset`,
  `HitSpec.box_contains` (the same L-inf box-membership test
  `full_measure.FullMeasure.member_indices`'s `'box'` query type uses), and
  `MULTS = (1.0, 2.0, 4.0)` (the archived cube-single half-width multipliers).
- `eval_density_zeroshot.py`: `build_pool`, `density_query`,
  `compute_density_primary` (spec 18's own PRIMARY readout algebra).
- `eval_partition_zeroshot.py`: `compute_Qr_from_dots`,
  `load_protos_verified` (spec 17b's own readout algebra + the sha256
  prototype-file check).
- `measure_readout_head.py`: `spearman` — the exact rank-Pearson function the
  valid-measure campaign's own G2 table quotes throughout (confirmed
  jax/torch-free at module top level, matching how `smb_common.py` already
  imports it).

**MC ground truth** (torch protocol, translated): for `--starts` (24)
uniformly-random dataset rows, teleport the env to that row's exact
`(qpos, qvel)`, execute one action `a0 = actor(s0, g)` (g = a second
uniformly-random dataset row's observation, exactly mirroring
`torchfb_full_measure.py`'s own `goal_rows = rng.choice(...)`), then MC-roll
the SAME goal-conditioned policy for `--horizon` (200) steps, `--mc` (2)
rounds, accumulating `sum_k gamma^k 1[x_{1+k} in box]` for `--boxes` (12)
query boxes per multiplier, drawn ONCE per multiplier before any
start/goal row is touched (cube-single's `probe_centers == "random"` — the
archived protocol; the rng draw order is part of the protocol, reproduced
byte-for-byte from `torchfb_full_measure.py`'s own `fixed_queries` comment).

**Model read-out mass** for the SAME box: the model's own read-out object is
a **normalized** probability measure (spec 18 §1: `ν = (1-γ)M`, `w`/`κ(x0)`
are softmax vectors summing to 1), whereas the torch FB measure is raw/
unnormalized (`FullMeasure.set_mass` divides by nothing beyond dataset
count). So the comparable "successor mass of box X" is each script's own
`Q_r` formula with the `r(x0)` offset **zeroed** and `r_pool`/`z_r` set to
the box indicator — this is the raw successor-mass term Bellman-derived from
spec 18 §1's own recursion (`ν` is defined over the state **after** the
transition, matching `torchfb_full_measure.py`'s MC sum starting at `x_1`,
not `x_0`):
- density: `compute_density_primary(0.0, w, kappa_x0, box_indicator(U_P), gamma)`
- partition: `compute_Qr_from_dots(0.0, p·box_indicator(protos), phi(x0)·box_indicator(protos), gamma)`

Both `(s0, a0, g)` are scored **once** per start (mirroring
`torchfb_full_measure.py`'s single `F0` per start), then dotted against
every query box — same reduction the reference script uses.

**Determinism check** (this session's own stated finding, verified
directly): re-ran `d18_cs_dens_s0` twice with identical flags, both at
smoke scale and at the full protocol scale (`--starts 24 --mc 2 --horizon
200 --boxes 12 --pool-size 16384`) — output JSON **byte-identical** both
times (`diff` empty).

**Pool size**: density readout run at `--pool-size 16384` (this session's
own stated convergence finding: under-converged at 4096, converged at
≥16384).

**Ambiguity flagged**: the gate statement ("Spearman ... ≥ 0.65") is a
single number, not broken out by box multiplier. I compute and report a
**pooled** Spearman (all 3 multipliers' probes combined, `n`≈864/checkpoint)
as the gated statistic, and a per-multiplier breakdown (mirroring the torch
precedent's own per-multiplier table) for diagnosis. The threshold is
applied to the pooled number.

### Results

All 6 checkpoints, `--checkpoint-step 500000`, `--starts 24 --mc 2
--horizon 200 --boxes 12` (density: `--pool-size 16384`), `--seed 0`.
Raw JSONs in `results/spec19_g4/g4_<run>.json`.

| checkpoint | object | pooled Spearman (n) | mult=1.0 | mult=2.0 | mult=4.0 | degenerate_frac (pooled) | gate (≥0.65) |
|---|---|---|---|---|---|---|---|
| d18_cs_dens_s0  | density   | **+0.160** (864) | 0.163 | 0.338 | 0.262 | 0.61 | **FAIL** |
| d18_cs_dens_s1  | density   | **+0.166** (864) | 0.115 | 0.321 | 0.261 | 0.61 | **FAIL** |
| d18_cs_dens_s2  | density   | **+0.174** (864) | 0.117 | 0.279 | 0.268 | 0.61 | **FAIL** |
| d17_cs_part_s0  | partition | **+0.132** (864) | 0.105 | 0.315 | 0.233 | 0.61 | **FAIL** |
| d17_cs_part_s1  | partition | **+0.144** (864) | 0.122 | 0.288 | 0.255 | 0.60 | **FAIL** |
| d17_cs_part_s2  | partition | **+0.177** (864) | 0.107 | 0.272 | 0.276 | 0.60 | **FAIL** |

**PASS/FAIL against 0.65: FAIL for all 6 checkpoints, both methods.** No
threshold was adjusted. Both objects land in the same narrow band
(0.13–0.18), well below the 0.65 bar, at every individual multiplier too
(best single-mult reading is 0.34, still far short).

Diagnostic note (not a script bug, checked): at `mult=4.0` (the largest
box, half-width 1.6 in obs units ≈ most of the reachable workspace),
`model_mass_mean` is small/slightly negative while `mc_mean≈25` — the box
covers most of both the current AND future support, so `w`'s and `κ(x0)`'s
box-masses are both near 1 and their difference is numerically noisy after
the `/(1-γ)=×100` amplification. This affects the mult=4.0 column's
precision but not the overall verdict: mult=1.0 and mult=2.0 alone are also
far below 0.65.

### Interpretation

Per spec 19 §1: "For ORS this is not a paper gate but load-bearing
engineering — the shaping signal is only as good as the occupancy
ordering." **The occupancy ordering is not calibrated on either the
density or the K=256 partition object at 500k, on any of the 3 seeds
each.** Per spec 17 §6's own NULL clause language and spec 19 §3's
pre-registered read (P1.2: "gate fails ⇒ the behavior-density occupancy is
miscalibrated in ordering ⇒ fix the measure before touching shaping"), this
is a genuine negative finding that should gate any P1 (ORS ORS extractor)
launch on these specific objects until re-examined — P1.2 will need its own
run of this same gate on the retrained BEHAVIOR-mode density checkpoints
(19a), which are a different training target and may calibrate
differently.

**⚠ WITHDRAWN — see the corrected results immediately below. Both the
FAIL verdict and this interpretation are wrong; they were an artifact of
the readout bug described in the UPDATE section at the top of this file.**

### Corrected P0.1 results (supersedes "Results"/"Interpretation" above)

Same protocol, same 6 checkpoints, same flags
(`--starts 24 --mc 2 --horizon 200 --boxes 12`, density `--pool-size
16384`, `--seed 0`), readout + Spearman bugs fixed as described in the
UPDATE section. Raw JSONs re-written in place at
`results/spec19_g4/g4_<run>.json`; re-verified bit-identical on repeat.

**Pooled (all mults combined) — the literal gate statistic:**

| checkpoint | object | pooled Spearman (tie-corr) | pooled (ordinal, for reference) | non-degenerate subset (mc>0) | gate (≥0.65) |
|---|---|---|---|---|---|
| d18_cs_dens_s0 | density   | **0.778** | 0.684 | 0.801 (n=340) | **PASS** |
| d18_cs_dens_s1 | density   | **0.789** | 0.702 | 0.796 (n=337) | **PASS** |
| d18_cs_dens_s2 | density   | **0.767** | 0.676 | 0.802 (n=334) | **PASS** |
| d17_cs_part_s0 | partition | **0.790** | 0.692 | 0.791 (n=338) | **PASS** |
| d17_cs_part_s1 | partition | **0.784** | 0.687 | 0.805 (n=348) | **PASS** |
| d17_cs_part_s2 | partition | **0.777** | 0.713 | 0.783 (n=342) | **PASS** |

**PASS/FAIL against 0.65: PASS for all 6 checkpoints, both methods** —
opposite of the withdrawn verdict. Threshold not adjusted.

**Per-multiplier (report this prominently — pooling is NOT innocent):**
box size drives both `mc` and `model_mass`, so the pooled figure sits
above every individual multiplier column except mult=4.0. This is the row
that should actually be quoted for anything scale-sensitive (e.g. ORS,
which builds its shaping reward from ε-scaled distances at the FINE
scale):

| mult (half-width) | dens s0 | dens s1 | dens s2 | part s0 | part s1 | part s2 | range |
|---|---|---|---|---|---|---|---|
| 1.0 (0.4) | 0.456 | 0.462 | 0.407 | 0.477 | 0.455 | 0.423 | **0.41–0.48** |
| 2.0 (0.8) | 0.596 | 0.641 | 0.590 | 0.639 | 0.612 | 0.608 | **0.59–0.64** |
| 4.0 (1.6) | 0.848 | 0.836 | 0.831 | 0.837 | 0.839 | 0.830 | **0.83–0.85** |

### Corrected interpretation

**The "both measures order occupancy no better than chance" claim from the
original run is withdrawn**, and it does **not** license spec 19 §3's "fix
the measure before touching shaping" as a blanket verdict. The defensible,
scale-aware statement is: **coarse-scale occupancy ordering is strong
(0.83–0.85 at half-width 1.6); fine-scale ordering is only moderate
(0.41–0.48 at half-width 0.4, the native hit-spec ε)**. The pooled PASS is
real but partly a resolution artifact (box size is itself informative and
inflates the pooled correlation above the fine-scale column). Whether ORS
shaping specifically needs fine-scale ordering — since its reward is built
from ε-scaled distances at that same fine scale — is the open question
this raises for spec 19 §1 P1.2, not something this gate alone resolves.
Density and partition (K=256) are statistically indistinguishable at every
grain (both objects land in the same band at every multiplier, across all
3 seeds each) — no evidence either object orders occupancy better than the
other.

## P0.3 — CSV bookkeeping

Added 5 new `run_group` entries to `scripts/results_map.py` (only appended
at the end of `MAP`; nothing else in the file touched — verified with `git
diff scripts/results_map.py`, which shows the pre-existing other-track
edits untouched and my addition confined to the final block):

- `GCIQL-CubeSingle-Flow-Measure-Density-d50` (`d18_cs_dens_s{0,1,2}`)
- `GCIQL-CubeSingle-Flow-Measure-DensityHard-d50` (`d18_cs_hard_s0`)
- `GCIQL-CubeSingle-Flow-Measure-DeltaPartition-K64-d50` (`d18_cs_partK64_s0`)
- `GCIQL-CubeSingle-Flow-Measure-DeltaPartition-K512-d50` (`d18_cs_partK512_s0`)
- `GCIQL-CubeSingle-Flow-Measure-DeltaPartition-K1024-d50` (`d18_cs_partK1024_s0`)

Ran `.venv/bin/python scripts/compile_results_csvs.py --refresh`. Result:
`results/summary_agents.csv` 98 rows, `results/bridge_grafts.csv` 180 rows,
`results/compile_audit.txt` 36 UNMAPPED groups remaining — **none of them
are the 5 groups just added** (confirmed by grep; they no longer appear in
the UNMAPPED list and each has 0 partial-seed flags in the audit).

Compiled rows (`results/bridge_grafts.csv`, @500k column = `mean, ci95,
n_seeds, per_seed`):

| variant | mean | n_seeds | per_seed @500k | ground truth given | match |
|---|---|---|---|---|---|
| measure-density | 0.9293 | 3 | 0.896 0.928 0.964 | s0 0.896, s1 0.928, s2 0.964 | **exact** |
| measure-density-hard | 0.876 | 1 | 0.876 | 0.876 | **exact** |
| measure-delta-partition-k64 | 0.556 | 1 | 0.556 | 0.556 | **exact** |
| measure-delta-partition-k512 | 0.872 | 1 | 0.872 | 0.872 | **exact** |
| measure-delta-partition-k1024 | 0.868 | 1 | 0.868 | 0.868 | **exact** |

All 5 compiled numbers agree exactly with the ground truth supplied in the
task brief. No discrepancy to investigate.

## Files touched / added

- **New**: `scripts/rescuing_fb/eval_measure_g4_spearman.py` (the G4 harness;
  readout + Spearman bugs fixed in place 2026-08-06, see UPDATE above).
- **New**: `tests/test_eval_measure_g4_spearman.py` (8 tests, added post-fix:
  `spearman_tie` sanity floor/ceiling, tie-vs-ordinal demonstration,
  readout-formula regression pins, parser smoke — all green).
- **New/overwritten**: `results/spec19_g4/g4_d18_cs_dens_s{0,1,2}.json`,
  `results/spec19_g4/g4_d17_cs_part_s{0,1,2}.json` (raw per-checkpoint
  reports; overwritten in place with the corrected re-run — the pre-fix
  JSONs are not separately retained on disk, but their pooled numbers are
  preserved verbatim in the "Results" table above for the record).
- **Edited**: `scripts/results_map.py` (append-only, 5 new entries).
- **Regenerated** (via `--refresh`, not hand-edited):
  `results/summary_agents.csv`, `results/bridge_grafts.csv`,
  `results/compile_audit.txt`.

No `git add`/`commit`/`stash`/`checkout` performed, per the hard rules.

## Concerns / things worth a second look

1. **Ambiguity in the G4 statistic** (flagged above, still live post-fix):
   the spec never specifies pooled-vs-per-multiplier for the threshold
   comparison, and now that ambiguity is load-bearing — pooled PASSES
   (0.77–0.79) while the finest column (mult=1.0, the native hit-spec ε)
   only reaches 0.41–0.48. I gate on pooled (matching the spec's literal,
   unqualified wording) but report per-multiplier prominently per the
   coordinator's instruction; any downstream use of this gate (P1.2 in
   particular) should decide explicitly which column it needs and re-read
   this table rather than quoting "PASS" bare.
2. The `degenerate_frac` (fraction of MC probes with truth exactly 0) is
   ~0.60 pooled — cube-single's "random" probe-center policy (vs.
   "hindsight" used on multi-object envs) is known from the torch precedent
   to produce a fair number of zero-truth boxes. This is now correctly
   *tie-handled* (item 3 fix) rather than a residual concern; it does still
   mean the non-degenerate-subset numbers (n≈340/checkpoint, ~0.80) are the
   more information-dense read where sample size allows.
3. **This class of bug** (a de-biasing/cancelling pair split across two
   zeroed-vs-kept terms) is exactly what the two new regression-pin tests
   (`test_density_readout_is_w_dot_rpool_not_the_kappa_x0_variant`,
   `test_partition_readout_is_p_dot_zr_not_the_phi_x0_variant`) exist to
   catch going forward — they assert the buggy and fixed call shapes
   numerically disagree whenever the box overlaps `x₀`, so a future edit
   that reintroduces the split would fail loudly rather than silently.
4. I did not get independent adversarial re-verification of THIS fix (the
   fix itself was produced in response to one). The re-run numbers match
   the verifier's own reported figures closely (pooled ordinal 0.676–0.713
   here vs. their 0.676–0.713; tie-corrected 0.767–0.790 here vs. their
   0.767–0.789; per-mult bands match their stated ranges at every
   multiplier) — self-consistent, but a second independent check would be
   healthy before this gate is treated as fully closed.
