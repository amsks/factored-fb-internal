# Density-FB HPO: one uniform config that wins the BC / GCIQL / Density-FB table

**Date:** 2026-08-14 · **Status:** SPEC — ready to hand to an implementation agent
**Scope directive (user, 2026-08-14):** headline comparison is **BC / GCIQL / Density-FB
only** (biGCIQL dropped), **default-HP discount cells only** — see the two hard rules:
`default-discount-cells-only`, `only-1m-datasets-citable`.

## 1. Goal

Find **one** Density-FB configuration — a single set of overrides applied identically to
every environment — that makes the Density-FB column lead (or tie within noise) every
citable row of `PAPER/decoding-bilinear/table.tex` against BC and monolithic GCIQL, and
then re-run the full column with it at 3 seeds.

Non-goals: per-environment hyperparameters (indefensible), γ-sweep arms as table cells
(rule above; a discount change is allowed ONLY by changing the single default in
`impls/configs/critic/density_fb.yaml` and re-running EVERY cell with it), any MS work
(both MS cells are certified non-discriminating — push is a degenerate control, pick is a
contact-precision floor; do not spend jobs there).

## 2. Integrity protocol (read before launching anything)

Tuning our method while baselines run defaults is exactly what reviewers kill tables for.
The defensible version, which this spec follows:

- **Dev/holdout split.** All screening and selection happens on the DEV cells only:
  `cube-single` (anchor, must not regress), `antmaze-medium`, and `mw-pick-place` on
  play-resample v3. Every other row (`cube-single-OOD`, `cube-double`, `scene`, both
  pnh-1m cells, both resample-hard cells, `mw-push` v3) is HOLDOUT: the frozen config is
  run there once, 3 seeds, and whatever comes out goes in the table. No peeking, no
  re-selection after seeing holdout numbers. If the holdout result is bad, the honest
  outcomes are (a) report it, or (b) declare a NEW spec revision with a fresh holdout
  — never silently iterate.
- **Disclosure.** The paper gets one sentence: Density-FB's config was selected on
  {cube-single, antmaze, mw-pick-place-v3} with N configurations; baselines use their
  published defaults (GCIQL's defaults are upstream-OGBench-tuned, which is a fair
  statement to make).
- **Fairness floor.** The one axis where our stack is KNOWN to diverge from the reference
  implementation — `lr_actor` (ours 1.0e-4, bridge single lr 3.0e-4, see §4/S0) — is a
  fidelity repair, not tuning; if it moves density, offer GCIQL the same `lr_actor=3.0e-4`
  arm on the two MW dev-adjacent tasks (6 cheap jobs) so the column comparison can't be
  attributed to actor-lr asymmetry.

## 3. Where density stands today (2026-08-14 landings, all γ=0.99, 3 seeds)

| Cell | BC | GCIQL | Density-FB | status for us |
|---|---|---|---|---|
| cube-single | 0.086 | 0.732 | **0.916** | anchor — protect |
| cube-single OOD | 0.073 | 0.492 | **0.913** | won |
| cube-double | 0.023 | 0.377 | **0.623** | won |
| scene | 0.048 | 0.589 | **0.637** | thin margin |
| antmaze-medium | 0.160 | 0.253 | ~0.34 interim (0.388/0.364@400k/0.268@400k) | **port gap**: bridge reference got 0.676±0.188 (n=2) on the same data |
| mw-pick-place, resample v3 | 0.440 | **0.993** | 0.133 | **worst cell** |
| mw-push, resample v3 | 0.540 | 0.733 | **0.820** | won |
| mw-pick-place, pnh-1m | 0.340 | 0.520 | 0.460 | tie-ish, need +0.06 |
| mw-push, pnh-1m | 0.327 | 0.527 | 0.500 | tie |
| mw-pick-place, resample-hard v3 | 0.367 | **0.967** | 0.573 (seeds 0.70/0.20/0.82) | losing + seed collapse |
| mw-push, resample-hard v3 | 0.613 | 0.813 | **0.827** | won (barely) |

Two distinct problems, needing different levers:
- **P-A (fidelity):** antmaze at half the reference number on identical data. Something in
  the port or its launch config, not a fundamental limit.
- **P-B (task family):** mw-pick-place fails on both resample datasets while mw-push wins
  on the same data with the same config — a pick-place-specific failure. It already
  SURVIVED the `p_randomgoal=0.3` ablation (0.140±0.170), so goal-mixture is ruled out.

## 4. Failure hypotheses → sweep axes

Current defaults (`impls/configs/critic/density_fb.yaml` + `actor/flowbc.yaml`), all cells
run with the `measure_density_hard=true` override (the "densityhard" tag):
`lr=3e-4, lr_actor=1e-4, batch=1024, hidden=[512,512,512], layer_norm, discount=0.99,
tau=0.005, measure_density_dim=50, measure_density_temp=1.0, actor_q_center=true,
bc_coeff=3.0, q_coeff=1.0, flow_steps=10`. Per-domain `measure_hit_spec` (provenance in
`scripts/launch_table_fill_ogb_ms.sh` §Hit-spec): cube-single "19:22:0.4",
cube-double "19:22:0.4,28:31:0.4", scene 5-block, antmaze "0:2:0.5",
mw-push "4:7:0.05", mw-pick-place "4:7:0.07".

Axes, in priority order (override strings are exact; `measure_hit_spec` values MUST keep
their escaped quotes — use the `tr` pattern from `launch_measure_tier1.sh:68`, xargs
strips them):

- **H1 `lr_actor=3.0e-4`** — the known bridge-parity delta (memory: "cheapest follow-up
  arm"). Prime suspect for P-A and the pnh-1m ties; the bridge numbers we're chasing were
  produced at this value. Zero-risk to try everywhere.
- **H2 hit-radius for pick-place** — `measure_hit_spec` is the SUCCESS-threshold radius
  (0.07) but a 3-D pick goal has most of its density mass on the table plane; the measure
  may be starved of positive hits in the lift phase. Sweep eps: `"4:7:0.05"`, `"4:7:0.07"`
  (current), `"4:7:0.10"`, `"4:7:0.14"`, and the split-block variant `"4:6:0.07,6:7:0.14"`
  (xy tight, z loose — grammar is comma-joined conjunction of a:b:eps blocks).
- **H3 actor exploitation** — prh pick-place's 0.20 seed and v3's 0.13 look like Q-ascent
  into a bad measure (same signature as the bon N=64 pathology). Sweep `q_coeff` ∈
  {0.3, 1.0, 3.0} and `bc_coeff` ∈ {1.0, 3.0} (crossed only at the extremes: 4 combos,
  not 6).
- **H4 measure sharpness/capacity** — `measure_density_temp` ∈ {0.5, 1.0, 2.0},
  `measure_density_dim` ∈ {50, 128}. Dim matters if pick-place's lift+place measure needs
  more rank than push's planar one.
- **H5 target stability** — `tau` ∈ {0.005, 0.001} for the prh seed-collapse.
- **H6 discount** — {0.98, 0.99, 0.995}. Screen it, but adoption is all-or-nothing
  (§Non-goals): a non-0.99 winner must win the DEV set jointly, become the yaml default,
  and trigger a full-column re-run. Expected payoff: cube-double/antmaze (long horizon)
  might like 0.995; MW might not. If the sign splits across dev envs, discount stays 0.99.
  γ=0.999 is out of scope entirely.
- **P-A specific, before any sweeping:** diff our antmaze effective config against the
  bridge's antmaze run config dump (bridge repo `origin/gciql-fb-bridge` @ 542c058,
  `tools/wandb_mode_shim/gciql_flow.py` runs). Check especially: lr_actor (H1),
  `gc_negative`/goal-sampling probabilities, EVAL_EPISODES (bridge may have used more —
  n=2 with ±0.188 spread means their cells are noisy too; it is possible 0.676 is partly
  luck and the honest target is ~0.5), and that `boot["next_action_vgoal"]` is on the
  antmaze path. Also note both incomplete seeds were at 0.364/0.268 @400k and rising —
  re-read final numbers before declaring the gap size.

## 5. Stages, budget, gates

All jobs: `scripts/vista_train.slurm`, `EVAL_INTERVAL=100000 SAVE_INTERVAL=100000
EVAL_EPISODES=50 NUM_VIDEO=2`, per-TAG `SAVE_DIR` subdirs (run-dir names do NOT encode
overrides — the dirclash lesson), `VENV=$SCRATCH/.venv-jax-fb` for OGB/MW (`.venv-ms` only
for MS, which this spec doesn't touch), frozen `CODE_DIR` snapshot per stage, manifest in
`logs/`. Steps: 500k OGB/MW. Data roots per regime as in `launch_measure_tier1.sh`.
Never `--dependency`-chain (slurm purges completed jobs); drip-feed with retry.

- **S0 — fidelity repair (9 jobs).** `lr_actor=3.0e-4`, 3 seeds × {antmaze,
  mw-pick-place-v3, mw-pick-place-pnh1m}. Gate: any cell moves >2σ → H1 enters the
  frozen config; antmaze reaching ≥0.5 closes P-A.
- **S1 — screening (1 seed each, DEV cells only, ~26 jobs).**
  mw-pick-place-v3: H2 ×5, H3 ×4, H4 ×5 (temps×2 extra, dim×1), H5 ×2 — 16 jobs.
  antmaze (only if S0 didn't close P-A): H4 ×3, H6 ×3 — 6 jobs.
  cube-single anchor guard: the current best combo re-run 1 seed — must stay ≥0.85 —
  plus H6 screens ×3 (cube-single is the discount canary). ~4 jobs.
  Single-seed screens are for ORDERING only; nothing graduates on one seed.
- **S2 — confirmation on DEV (≤18 jobs).** Top-2 combos per dev cell × 3 seeds. Gate to
  freeze: mw-pick-place-v3 ≥0.75 (from 0.13; if no combo clears 0.5, pick-place is a
  structural failure — write that up instead of grinding), antmaze ≥0.45, cube-single
  ≥0.88.
- **S3 — freeze + holdout (≤27 jobs).** ONE override set, identical everywhere (hit-spec
  stays per-domain — it's task geometry, not tuning, but if H2's winner changed pick-place
  eps, disclose it as part of the protocol). Run the 8 holdout cells (＝ every table row
  minus the 3 dev rows, minus both MS rows) + re-run the 3 dev rows at final config,
  3 seeds. Table gets ONLY S3 numbers.

Total ≤80 jobs, ~4–12h each. Fits the queue in ~3 drip-fed days.

## 6. Landmines (all previously hit — do not rediscover)

- 1M rule: every dataset above is 1M — do not add any non-1M regime (`play-noise` 100k is
  banned from tables).
- `make_config` takes a dotlist LIST; passing one string char-iterates and silently drops
  overrides (fix pattern: `action="append"` / the launcher's `ov_flags` array).
- `measure_hit_spec` quoting: `tr`, not `xargs` (strips escaped quotes).
- Per-TAG SAVE_DIR subdirs, always (dirclash quarantine of 2026-08-05).
- Eval reseeds the global RNG → seeds share one batch stream after first eval; error bars
  understate. Report seed triples, not just mean±σ, for any cell near a tie.
- eval.csv can be null-byte-corrupted by duplicate passes; job `.o` logs are ground truth;
  take highest job ID per (tag, seed) within a generation.
- `set -u` + unset var in echo killed a slurm script once; smoke the launcher with
  `DRY_RUN` first.

## 7. Deliverables

1. `logs/density-hpo-<stamp>.txt` manifests per stage + one results CSV
   (cell, config, seed, success) under `results/`.
2. Updated `impls/configs/critic/density_fb.yaml` (+ `actor/flowbc.yaml` if H1 lands)
   encoding the frozen config as the new default, with a comment naming this spec.
3. table.tex Density-FB column refresh from S3 numbers only, bridge-provenance `d` cells
   replaced by this repo's runs, plus the one-sentence tuning disclosure in the caption.
4. A short verdict note in this spec's Status line per stage (S0..S3), including negative
   results — especially if pick-place refuses to move (that bounds the method and belongs
   in the paper's limitations, not in a drawer).
