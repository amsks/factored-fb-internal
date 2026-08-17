# Why Density-FB collapses on mw-pick-place: investigation spec

**Date:** 2026-08-16 · **Status:** SPEC — ready to execute
**Parent specs:** `docs/specs/2026-08-14-density-fb-hpo.md` (P-B), `docs/specs/2026-08-10-measure-objectives-on-mw-play.md`
**Code under investigation:** `impls/critics/density_fb.py`, `impls/configs/critic/density_fb.yaml`, `impls/actors/flowbc.py`
**Rules in force:** 1M-only, default-discount-cells-only, dev/holdout split of the HPO spec (everything below runs on the DEV cell `mw-pick-place` play-resample v3 unless marked mechanism-side).

## 1. The failure, pinned to numbers

From `PAPER/decoding-bilinear/table.tex` and run directories under `$SCRATCH/factored-fb-tier1/` (`.o`/eval.csv ground truth, final-step eval, 50 episodes, 3 seeds, repaired protocol `obs/goal_mask_dims=[36:39]`, `value_p_randomgoal=0`, `eval_goal_mode=synth`, `measure_density_hard=true`, hit-spec `"4:7:0.07"`):

| Dataset (all 1M) | BC | GCIQL | Density-FB | gap to GCIQL |
|---|---|---|---|---|
| pr (play-resample v3) pick-place | 0.440 | **0.993±0.014** | 0.133±0.057 (0.18/0.08/0.14) | **−0.86, below BC floor** |
| prh (resample-hard v3) pick-place | 0.367 | **0.967** | 0.573 (0.70/0.20/0.82) | −0.39, seed collapse |
| pnh-1m pick-place | 0.340 | 0.520 | 0.460 (0.46/0.50/0.42) | −0.06, stable tie |
| pr push (same data, same config) | 0.540 | 0.733 | **0.820±0.045** | density WINS |

Already-run ablations: `p_randomgoal=0.3` (prg03) → 0.140±0.170, failure survives; S0 `lr_actor=3.0e-4` → pr pick-place 0.367 (0.14/0.42/0.54) — helps but nowhere near repair; pnh-1m unchanged (0.42 all seeds).

## 2. New evidence (this investigation, from eval.csv / train.csv / eval_episodes.jsonl / raw npz)

- **E1 — the failure is a mid-training COLLAPSE, not a capacity floor.** pr pick-place density reaches 0.50/0.26/0.52 at 100–400k, then decays to 0.18/0.08/0.14 at 500k. prh peaks at **0.72/0.84/0.82** before oscillating down. pnh-1m never collapses (flat 0.42–0.58). GCIQL on the same data/actor is 0.98 from 100k and never moves. The table's 0.133 is the final point of a decaying curve whose peak was ~0.5; prh's "0.20 seed" is a step-500k draw from an oscillation whose other evals were 0.36–0.84.
- **E2 — failure-mode shift, not near-miss.** Per-episode `goal_dist_final` (pr seed 0): frac > 0.25 m (object never delivered) grows 0.24 → 0.54 from 100k → 500k while frac < 0.07 shrinks 0.34 → 0.14. The policy increasingly does the wrong thing outright as training continues.
- **E3 — pick-place's measure is ~5× more concentrated than push's from the start, but concentration is NOT the collapse trigger.** `density_ess_median` at 100k: pick-place ~13 (pr), ~14–16 (prh), ~19–23 (pnh-1m) vs push ~60–106. All decay ~2× by 500k. Crucially pnh-1m pick-place has the same low/decaying ESS yet stable success, and pr push halves its ESS (70→32) with no decay. Low ESS is a pick-place geometry signature (peaked occupancy near a lifted goal); it can amplify a failure but doesn't cause one alone.
- **E4 — collapse severity tracks behavior-support width, inversely.** pr (`noise_max=0.02`, cleanest actions) collapses hardest; prh (`noise_max=0.1`) intermediate; pnh-1m (`eps_max=0.5` random-mix + `noise_max=0.3`) not at all. Classic offline-RL OOD-action exploitation scaling.
- **E5 — no Q blow-up.** Logged actor `q` stays ~0.003 flat for 500k (unlike the td_fb vgoal pathology's 4-orders inflation). Whatever degrades is in the *shape* of Q, not its scale.
- **E6 — data is not lift-starved and goals are covered.** pr pick-place obj z: median 0.127 m, p90 0.233, ~30% of transitions have obj z > 0.10; goal z spans 0.05–0.30. `tab:coverage`: in-support distance median 0.03 radii, p95 0.36.
- **E7 — structural difference from GCIQL.** Same flowbc actor, same data, same protocol. Two differences: (a) GCIQL's expectile backup is **in-sample** (no actor action in the critic target) while density_fb's target `w_bar` is evaluated at `a+ = π(·|s′, g_value)` — the actor is inside the critic's fixed point; (b) GCIQL's Q is a smooth MLP readout, density's Q is `Σ_j w_j·kg_j`, a softmax-kernel readout that E3 shows is spiky (~6–13 effective queries out of 1024) on pick-place.

## 3. Ranked hypotheses

### R1 — Actor–critic bootstrap feedback loop (deadly triad through `a+`), gated by behavior support. **Prime suspect.**
The critic target bootstraps at actor actions (`boot["next_action_vgoal"]`); the actor ascends the critic's spiky readout; on narrow-support data the ascent leaves the region where the measure is trained; `w_bar` at the drifted `a+` corrupts the target; repeat. Predicts E1 (good early, decays after the actor has moved), E2, E4 (support width is the brake), E7a, and push's immunity (E3: smooth landscape, 5–10× higher ESS → gentler ascent).
**Against:** nothing observed; cube-single (also clean scripted data) doesn't collapse — but cube's ESS is ~89 (healthy) so the spike-ascent precondition is absent there, consistent.
**Kill-tests:** KT-1, KT-2, KT-3.

### R2 — Peaked-occupancy readout (pick-place geometry → low-ESS spiky Q) as the amplifier.
Not a standalone cause (E3: pnh-1m has low ESS and no collapse) but the reason pick-place is the vulnerable cell. Temp/dim reshape the readout smoothness directly.
**Kill-test:** KT-4. If a smoother measure removes the collapse at unchanged q_coeff, R2 is load-bearing; if not, R1 must be attacked at the loop itself.

### R3 — Hit-spec z-geometry (`"4:7:0.07"` isotropic) starves/mis-shapes the goal kernel (HPO spec H2).
**Against (now stronger):** prh peaks at 0.84 with the current spec — the kernel demonstrably supports near-perfect behavior; E6 shows rich lift support; coverage table fine.
**For:** kg at eval conditions on a synth frame whose tcp is collapsed exactly onto the target (MW layout has no `grasp_offset` exemption, `env/unified.py::synth_goal_frame`), and the last-cm precision regime is where pick-place differs from push.
**Kill-test:** KT-5. Demoted to third; expected mostly-negative.

### R4 — Final-step reporting convention overstates the failure.
Not a repair, but a paper-truth issue: 0.133 is the endpoint of an oscillating curve (E1). Any fix that merely *stabilizes* at the peak (~0.7–0.85 on prh, ~0.5 on pr) changes the table qualitatively.
**Kill-test:** KT-0 (free, do first).

### R5 — Value-goal mix contamination (`value_p_curgoal=0.2` trivial rows).
Low prior: same mix on push wins; prg03 already varied the mix with no effect.
**Kill-test:** KT-6, piggyback only.

### Ruled out (do not re-spend jobs)
Goal observability (masked protocol on); eval goal channel (synth, per-episode); goal coverage (tab:coverage, p95 ≤ 0.36 radii); random-goal mixture (prg03 survived); dataset missing lift states (E6); Q-scale blow-up / td_fb-style vgoal bug (E5, and density_fb correctly reads `next_action_vgoal`); `lr_actor` alone (S0 ran: 0.133→0.367, insufficient).

## 4. Kill-tests

All training jobs: `scripts/vista_train.slurm`, `DOMAIN=mw-pick-place CRITIC=density_fb ACTOR=flowbc`, `FB_DATA_ROOT=$SCRATCH/fb-data/play-resample`, base override `measure_density_hard=true measure_hit_spec=\"4:7:0.07\"` + E3P protocol string, 500k steps, **`EVAL_INTERVAL=50000`** (the collapse is the observable — resolve it), per-TAG SAVE_DIR subdirs, `tr`-not-`xargs` quoting, DRYRUN smoke first. Screens are 1 seed, pr pick-place; ordering only.

- **KT-0 (free, no GPU): checkpoint forensics.** `scripts/reeval_checkpoint.py` on the existing pr/prh `params_100000–500000.pkl`, 50 episodes: certify the E1 peaks are real behavior, not 50-episode eval noise, and produce the peak-vs-final curve for the writeup. Under R4: peak re-evals reproduce ~0.5 (pr) / ~0.8 (prh). Also watch two videos each from a 100k and a 500k checkpoint and name the behavioral difference — E2 says episodes end far out; the video says whether the actor stopped *attempting* the pick.
- **KT-1 (R1, actor side): `actor.q_coeff` ∈ {0.3, 0.1}** (2 jobs; verify nested-dotlist takes `actor.q_coeff` with a DRYRUN — HPO spec H3 already planned this sweep). Under R1: collapse delayed/removed, final ≥ 0.45 (above BC 0.44) with peak roughly preserved; under R3: no change in shape, only level.
- **KT-2 (R1, loop-speed side): `tau=0.001`** and **`actor.bc_coeff=10.0`** (2 jobs). Both slow the loop from opposite ends (target lag; actor anchored to support). Under R1: either one flattens the post-peak decay. If tau=0.001 alone fixes it, that's also HPO-spec H5's prh seed-collapse answer.
- **KT-3 (R1 vs R2 discriminator): γ=0.98 mechanism-side screen** (1 job, NOT a table cell — default-discount rule). Doubles the (1−γ) grounding mass against the self-referential `w_bar` term. Under R1: collapse weakens. Under pure R2: no effect.
- **KT-4 (R2): `measure_density_temp=2.0`; `measure_density_dim=128`** (2 jobs). Track `density_ess_median`: temp 2.0 must move it up from ~13; if ESS rises and the collapse persists, R2 is dead and R1 stands alone; if collapse vanishes, adopt into the HPO S1 candidate set.
- **KT-5 (R3): split-block hit spec `"4:6:0.07,6:7:0.14"`** (1 job; grammar per `parse_hit_spec`). Under R3: pr peak rises toward prh's 0.84 *and* final follows; under R1: same peak, same collapse.
- **KT-6 (R5, piggyback): `value_p_curgoal=0.0 value_p_trajgoal=1.0`** on whichever KT-1 arm wins (1 job). Only if something else has already stabilized the curve.
- **Optional instrumentation (code, no new runs):** log `Q(s, a_data) − Q(s, a_actor)` and `‖a_actor − a_data‖` in `flowbc.loss` info. Under R1 the action gap grows past the 100–200k peak on pr and stays flat on pnh-1m. One-file change, mechanism-side; converts R1 from inference to measurement.

**Gates.** Any single-seed arm whose 500k final ≥ 0.55 with no >0.25 drop from its own peak graduates to 3 seeds, then enters the HPO spec's S1/S2 machinery — selection stays on the DEV cell, holdout untouched. If NO arm clears 0.5 (HPO spec's structural-failure clause), the writeup is: *the one-stage actor-bootstrapped softmax density is unstable on precision tasks over narrow-support data — its critic target contains the actor, and unlike GCIQL it has no in-sample backup; the instability, not the ceiling, is the deficit (peaks 0.72–0.84 on prh)* — with E1–E5 as evidence and R4 disclosed in the table caption.

**Budget:** KT-0 free; 8–9 training jobs × 1 seed ≈ one drip-fed day, well inside the HPO spec's ≤80-job envelope (these subsume its planned H3/H4/H5 screens on the same dev cell).

## 5. Landmines (inherited, verbatim rules)

1M-only; γ table cells stay 0.99 (KT-3 is mechanism-side only); `make_config` dotlist is a LIST; hit-spec escaped quotes via `tr`; per-TAG SAVE_DIR; `.o` logs over eval.csv (null bytes); eval reseeds the global RNG — report seed triples near ties; never `--dependency`-chain; no ortho/mass-anchor/scale-norm on the density path (bridge-measured harmful); prg03 and lra3e4 arms already exist — reuse, don't rerun.

**Key files:** `impls/critics/density_fb.py` (bootstrap at `boot["next_action_vgoal"]`, `q_fn` kernel readout), `impls/actors/flowbc.py` (q-term normalization, `sample_bootstrap`), `scripts/launch_density_hpo.sh` (launcher pattern to extend), `env/unified.py:187` (`synth_goal_frame`), run data under `$SCRATCH/factored-fb-tier1/{pr,prh,pnh1m}/mw-{pick-place,push}/densityhard*/`.
