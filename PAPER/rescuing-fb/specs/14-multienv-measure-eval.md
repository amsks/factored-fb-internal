# Spec 14 — the measure axis off cube-single

**Date:** 2026-07-31 · **Track:** bridge · **Status:** design approved, unimplemented

## 1. Why

Every measure-side claim we have is a cube-single claim. `results/spec10_torchfb/`
holds 89 JSONs; all 89 are cube-single. The ledger says so in its own limitations
(§9.16 consequence 5), the paper says so at `main.tex:818–820`, and
`MULTIENV-RESULTS.md` §6 lists the JAX zero-shot harness as *blocked* off
cube-single pending an `env_geometry.py` that nobody has written.

Meanwhile four torch checkpoints finished on 2026-07-30 and sit **unscored**:

| run | path | protocol |
|---|---|---|
| `p1_scn_s42` | `~/exp_spec10/torchfb/p1_scn_s42` | first-passage, scene |
| `p2_scn_s42` | `~/exp_spec10/torchfb/p2_scn_s42` | visitation, scene, rzK50 stack |
| `p1_cdbl_s42` | `~/exp_spec10/torchfb/p1_cdbl_s42` | first-passage, cube-double |
| `p2_cdbl_s42` | `~/exp_spec10/torchfb/p2_cdbl_s42` | visitation, cube-double, rzK50 stack |

Each has `final.pt` plus `step_{100..500}k.pt`. They use FB's own zero-shot
relabel protocol — exactly the protocol we lack off cube-single. **The training
is already paid for; this spec buys the numbers with eval time only.**

## 2. What actually blocks it

Not much, and the same thing in both stacks. Every eval script redeclares

    CUBE_OBS = slice(19, 22)      # cube xyz
    EPS      = 0.4                # kernel radius

whereas scene and cube-double are trained against a **multi-block conjunctive**
region, carried in the checkpoints' own `hit_spec`:

    scene       19:22:0.4, 28:30:0.5, 32:34:0.5, 36:37:0.72, 38:39:0.6
    cube-double 19:22:0.4, 28:31:0.4

`FullMeasure`'s KDTree denominator N̂ assumes the same single L2 ball. Two of the
JAX scripts additionally contain *silent wrong-answer* bugs on multi-object envs
(spec 08 §"Not ported"): `goal_xyzs.reshape(-1)[:3]` truncates cube-double to
cube 0, and scene has no `goal_xyzs` key at all.

### 2.1 The trap that would waste the campaign

L1's ground truth is a Monte-Carlo occupancy: teleport to a state, roll out, count
discounted visits to the region. On cube-single the region is a 4 cm ball and MC
hits are plentiful. **On cube-double the conjunctive region has an off-diagonal
hit rate of 0.046%** (§9.16). Probe centers drawn from random dataset rows would
essentially never be reached, MC truth would be 0, and every ratio would be 0/0.
A naive port returns an empty table and teaches us nothing. Probe-center
selection is therefore part of the design, not an afterthought (§3.3).

## 3. Design

### 3.1 `scripts/rescuing_fb/env_geometry.py` — one registry

Keyed by env name (`cube-single-play-v0`, `cube-double-play-v0`,
`scene-play-v0`), giving per env: the canonical trained hit spec, the obs-block
slices, dataset-path resolution, probe-center policy, and — for phase 2 — the
native `EVAL_TASKS`. It parses the same `"a:b:eps,..."` string the agents parse,
with a test asserting block-for-block agreement against
`agents/fb/measure/agent.py`, so the repo holds exactly one definition of "the
measure coordinate".

### 3.2 `FullMeasure` generalized from (cube[n,3], scalar eps) to (coords[n,D], HitSpec)

Membership becomes the **conjunction** of per-block L2 balls, matching the
agent's training-time indicator exactly.

Exactness argument (this is where a subtle wrong-answer bug would live): divide
each block's columns by its own `eps`; a conjunctive hit implies every block's
scaled L2 norm ≤ 1, which implies scaled **L∞** ≤ 1. A Chebyshev KDTree query at
radius 1 therefore returns a strict **superset**; an exact per-block filter over
those candidates gives exact conjunctive membership. No approximation, and it
stays a tree query rather than an O(n) scan. With one block it reduces to today's
L2 ball exactly — which is the regression test (§5).

`nhat`, `density`, `member_indices` (`ball` → conjunctive ball at a radius
multiplier, `box`, `union`) and `set_mass` follow unchanged in signature.

### 3.3 Probe centers

A registry-level choice, not a hardcoded `rng.choice` over dataset rows.

- **cube-single: random dataset rows** — preserves the archived protocol so the
  regression test is meaningful and old numbers stay comparable.
- **cube-double / scene: hindsight centers** — regions the rolled-out policy's own
  episode actually enters later, so the MC denominator is non-degenerate.

The policy governs **every** center draw in the chain, not just L1's: the L1
query boxes, `negmass_diag.py`'s goal sample, and L2's reward-family centers all
resolve their centers through the registry, so a single env switch cannot leave
one script sampling on a different distribution than the others.

Every JSON reports `degenerate_frac`, the fraction of probes whose MC truth was
zero. A degenerate cell is then a visible number rather than something hiding
inside a mean.

⚠ This makes cube-double's L1 protocol **differ** from cube-single's. Wherever
the two appear in one table, the footnote is mandatory.

### 3.4 The three torch scripts

`negmass_diag.py`, `torchfb_full_measure.py`, `torchfb_zeroshot_l2.py` drop their
`CUBE_OBS`/`EPS` constants and take the hit spec from **the checkpoint's own
recorded `cfg.hit_spec`**, overridable by `--hit-spec`. Defaulting to the
checkpoint kills a class of silent mismatch: you cannot score a scene checkpoint
with cube-single geometry, because the geometry travels with the weights.

Two fixes fall out of the port:

- **Dataset path is broken for these envs.** The scripts do
  `np.load(f"{cfg.data_path}/{cfg.domain}.npz")` and the s42 runs recorded
  `data_path=/dev/shm/factored-fb/datasets`, which holds
  `cube-single-play-v0.npz` but for scene/cube-double only
  `<env>/buffer/episode_*.npz` — no top-level npz. The real files are in
  `~/.ogbench/data`. Resolution: recorded path first, fall back to the ogbench
  data dir, error loudly if neither.
- **L2 reward families** are defined over cube xyz with literal radii. They
  generalize to the block coordinate with radii as **multipliers on the trained
  per-block radii**, so `ball1.6` means the same thing on every env instead of
  meaning "16 cm" everywhere.

### 3.5 Runner and outputs

`scripts/rescuing_fb/run_multienv_measure.sh` chains negmass → L1fix → L2-B →
L2-A over the four checkpoints, writing

    results/spec10_torchfb/{negmass,L1fix,L2_B,L2_A}_{scn,cdbl}{p1,p2}_s4{2,3}.json

the existing naming scheme, so `results_map.py` and
`compile_results_csvs.py --refresh` pick them up without special-casing.

### 3.6 Phase 1 is scored too — but NOT on L1

⚠ **Correction (2026-07-31, found during implementation).**
`torchfb_full_measure.py` already asserts `measure_mode == "visitation"`, with a
correct rationale: L1 calibration is defined against the visitation fixed point
(discounted occupancy), while phase 1's object is the absorbing `E[γ^T]`.
Scoring phase 1 against an MC *occupancy* would be a category error, so the
assert stays and **phase-1 cells are scored on negmass + L2 only**. The runner
branches on the recorded `measure_mode` rather than deleting the guard.

The paragraph below still holds for the axes phase 1 *is* scored on.


Phase 1 is the absorbing reaching value, so its object is `E[γ^T]`, not an
occupancy; L1 calibration against MC occupancy is arguably the wrong question for
it. It is scored anyway because §3's SMB table shows phase 1 (`fp`) beating
diagonal phase 2 on several axes, and that comparison is load-bearing for the
paper's argument. **Phase-1 rows carry a footnote stating what they do and do not
mean.**

## 4. Seeds

Every torch multi-env checkpoint is **seed 42 only**;
`MULTIENV-RESULTS.md` open action #2 says add a second seed before quoting
anything. GPUs 3–7 are idle. Therefore, launched immediately and in parallel with
the build: `p1_scn_s43 → p2_scn_s43` and `p1_cdbl_s43 → p2_cdbl_s43`, chained,
configs byte-matching the s42 `.hydra/overrides.yaml` except seed and warm-start
path.

**Launcher gotcha (found while smoke-testing 2026-07-31):** hydra rejects a bare
`hit_spec=19:22:0.4,28:30:0.5,...` as an ambiguous list. It must be passed
shell-quoted *inside* the argument — `hit_spec="'$HS'"` — which is why the s42
`overrides.yaml` records it with inner quotes. A bare pass is a hard parse error,
not a silent misread, so it cannot corrupt a run.

## 5. Tests (written first)

| test | asserts |
|---|---|
| **cube-single regression** | generalized code re-scoring `p2_rzK50_s42` reproduces the archived `L1fix_rzK50_s42.json` numerically |
| conjunctive membership | superset-plus-filter equals brute-force scan on random specs/points |
| spec parser agreement | `env_geometry` blocks == `agents/fb/measure/agent.py` blocks, per env |
| hit-rate cross-check | registry scene/cube-double hit rates match `measure_region_density.py` |
| dataset resolution | recorded path → ogbench fallback → loud error |

## 6. Pre-registered reads

Gates mirror §9.15's G1–G4 so multi-env rows are comparable to cube-single.
**The multi-env phase-2 runs have the validity penalties OFF** (the mass anchor's
in-batch N̂ clamps at 2/B at conjunctive hit rates — §9.15 multi-env note), so the
honest comparison for G1 is the cube-single **no-penalty** control (neg-mass
share ≈ 0.27), not the mcA/mcB arms.

- **G0 (harness validity, checked first):** `degenerate_frac` < 0.5. Above that
  the cell is uninformative and must be re-run at wider radius multipliers before
  any other gate is read.
- **G1 non-negativity:** neg-mass share vs the cube-single no-penalty control.
  Pre-registered expectation: **worse** on conjunctive envs, since sparser
  grounding gave more bootstrap slack everywhere else we have looked.
- **G2 calibration:** L1 head ratio in 0.85–1.30 with Spearman ≥ 0.65.
- **G4 transfer:** L2 gap fraction on held-out families, against cube-single's
  0.84–0.89 on concentrated rewards.
- **NULL:** if scene and cube-double both pass G2/G4 at cube-single levels, then
  the measure axis transfers and the native-GC regression (§9.16) is purely an
  actor-side artifact — which would be a substantially stronger paper claim than
  we currently make, and must be replicated on s43 before it is written.

## 7. Phase 2 (deferred)

`env_geometry.py` consumed by the five JAX scripts (`eval_zeroshot_rewards.py`,
`t6_gpi_eval.py`, `t6_generality_probe.py`, `renewal_head.py`,
`measure_readout_head.py`), including the two silent wrong-answer fixes. That is
what scores the headline E7/E8 runs and the `p2fix` hindsight arm on the measure
axis. Starts once the abstraction has proven itself on the torch side.

## 8. Deliverable

A measure-axis table for scene and cube-double alongside the existing native-GC
table, written into `MULTIENV-RESULTS.md` §5 and a new `PHASE2-LEDGER.md`
section, plus the standing CSV refresh (`results_map.py` +
`compile_results_csvs.py --refresh`) and a `DECISIONS.md` entry.
