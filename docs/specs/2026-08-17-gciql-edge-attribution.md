# GCIQL-edge attribution: can Density-FB reach 0.99 on precision cells? (for the implementation agent)

**Date:** 2026-08-17 · **Status:** ALL ARMS READ OUT (2026-08-18). M3 confirmed for push; M1 shown NECESSARY-BUT-NOT-SUFFICIENT; best density arm is ktC-atemp2 at 0.94/0.90.

| item | state |
|---|---|
| W1 SARSA bootstrap | done — `measure_density_bootstrap: actor\|sarsa`; `GCDataset` now emits `next_actions` (trajectory-boundary clamped); `density_sarsa_subst` logged in both modes |
| W2 actor readout temp | done — `measure_density_actor_temp: null` used only in `q_fn`; `stage_loss` keeps `measure_density_temp` |
| W3 expectile | verified, no code — `expectile=0.7` composes on `gciql` (default 0.9); no inert nested alias exists |
| W4 forensics | **DONE — verdict: REAL per-checkpoint instability** (job 917542). See below. |
| arms | **DONE** — 7 jobs (917535–41) via `scripts/launch_edge_attribution.sh`; results in §Results below |

**W4 substitution (blocking correction).** The spec names
`kt1-qc03/…seed_2/params_350000.pkl`; it does not exist. That run used
`EVAL_INTERVAL=50000` with `SAVE_INTERVAL=100000`, so the 350k *eval* (0.02) has no saved
params. The bracketing checkpoints do exist and the seed-2 curve reads
**0.36 / 0.02 / 0.80 at 300k / 350k / 400k**, so the forensics job re-scores 300k and 400k
instead — if 300k's 0.36 moves under a fresh episode draw, a 50-episode eval is not a reliable
point estimate here and the 0.02 needs no separate explanation. New arms launched from
`launch_edge_attribution.sh` use `SAVE_INTERVAL=50000` so the gap cannot recur.

**W4 RESULT (job 917542, `$SCRATCH/factored-fb-edge-attr/kt1-qc03-s2-forensics.jsonl`).**

| checkpoint | training-time eval | offset 0 (control) | offset 1000 (fresh draw) |
|---|---|---|---|
| 300k | 0.36 | **0.36 — exact** | 0.42 (+0.06) |
| 400k | 0.80 | **0.80 — exact** | 0.84 (+0.04) |

The control reproduces bit-for-bit at both checkpoints, so the harness is faithful and the
fresh draw is interpretable. It moves by +0.06 / +0.04, i.e. **50-episode eval variance is
only ~±0.05** (1 sd of a 50-episode draw is 0.057–0.068 here; worst case, at p = 0.5, 0.071).

Therefore the curve's swings are **real per-checkpoint instability, not measurement noise**.
The 350k dip from 0.36 to 0.02 is 4.8 worst-case sd — it cannot be an episode-draw artefact
even though that checkpoint itself is unavailable. Consequences:

- **R4's peak/final disclosure is mandatory**, not optional: the training trajectory genuinely
  visits near-zero between two good checkpoints.
- Single-seed screen numbers are trustworthy to ~±0.05, so the arm ordering in the collapse
  and edge-attribution screens is meaningful and the gate's 0.25 drop threshold sits ~3.5 sd
  above noise.
- A final-step number on an oscillating arm is a draw from the *trajectory*, not from the
  eval; more eval episodes would not fix it, and only stabilising training can.

**New knob required by W4:** `eval_seed_offset` in `env/unified.py::evaluate`. Episode seeds
are the loop index, so re-evaluating a checkpoint reproduces its number *exactly* and proves
nothing; the offset shifts the block onto disjoint initial states. Default 0 — leave it there
for anything compared against existing numbers.

**Original status:** SPEC — ready to implement
**Question:** GCIQL's remaining wins are pr/prh `mw-pick-place` (0.993/0.967 vs repaired density 0.76–0.90) and the untested OOD-goal split. The 2026-08-17 debug attributed the *collapse* to M1 (actor inside density's critic target; fixed by `actor.q_coeff=0.3` / soft) and left a residual gap whose prime suspect is **M2: the spiky kernel readout** (ESS ~13 under qc03 while its action-gap is flat; soft runs at ESS ~400 and is steadier). This spec's experiments decide whether ~0.85 is structural or whether density closes to ~0.99.
**Companions:** `2026-08-16-density-fb-pickplace-collapse.md` (R1/E-facts, gate definition), `2026-08-16-density-fb-bridge-port.md` (Part B machinery, uniform-config rule).

## Mechanisms under test
- **M1** in-sample backup (GCIQL: `impls/critics/gciql.py:72-80` — V has no action input, no `boot` read; density: `w_bar` at `boot["next_action_vgoal"]`, `impls/critics/density_fb.py:186-198`). Confirmed collapse cause; unknown whether fully sufficient.
- **M2** readout smoothness (GCIQL monolithic MLP vs density softmax-kernel `q_fn`, `density_fb.py:227-243`). Prime residual suspect.
- **M3** expectile pessimism (0.9 in-sample max ≈ copy-the-expert on narrow support; suspected cause of GCIQL's *push* mediocrity).
- **M4** actor q-normalizer: density divides by MAD(Q) (`density_fb.py:235-242` returns `q−q̄` as normalizer input), GCIQL by mean|Q| (`flowbc.py:98`). Noise-amplifying gain at Q≈0.003.

## Implementation work items

### W1 (KT-B, P0): SARSA bootstrap knob on density_fb
New config knob `measure_density_bootstrap: actor | sarsa` (default `actor`, bit-identical).
Under `sarsa`, `stage_loss` evaluates the target `w_bar` at the **dataset's** next action instead of the actor's:
- Data: `batch["next_actions"]` must reach the critic. GCDataset already serves density (dataset_class GCDataset); add `next_actions` to its sample output (shift-by-1 like `next_observations`, trajectory-boundary-safe via `valids`/terminal_locs — mirror bridge spec19a: `origin/gciql-fb-bridge` `specs/19a-behavior-density-impl.md` and its `next_actions` plumbing in `tools/wandb_mode_shim/gciql_flow.py`; keep our g-conditioned F tower, do NOT port the rest of behavior-policy mode).
- Critic: in `density_fb.stage_loss`, replace `boot["next_action_vgoal"]` with `batch["next_actions"]` when the knob is `sarsa`. The actor then trains against a critic whose fixed point it cannot influence — GCIQL's M1 property with the kernel readout intact.
- Log `density_sarsa_subst` (fraction/mean action distance between the two bootstrap choices) for the writeup.
- Static tests (`tests/impls/test_density_fb.py`): default path bit-identical; sarsa path reads `next_actions` and never calls the actor in the critic target; boundary transition uses a valid action.

### W2 (KT-C, P0): actor-readout-only temperature + normalizer flip
New knob `measure_density_actor_temp: null` (default → use `measure_density_temp`, bit-identical). When set, `q_fn` uses this temperature in its softmax **only for the actor readout** — the CE training target keeps `measure_density_temp`. One divisor in `q_fn`; must not touch `stage_loss`.
Also verify `actor_q_center=false` selects the mean|Q|-style normalizer cleanly (M4) — note the bridge asserts center=true on this path, so mark the arm mechanism-side.
- Static test: `actor_temp` changes `q_fn` output, leaves `stage_loss` loss/info bit-identical.

### W3 (KT-D, config only): GCIQL expectile sweep
No code. Confirm `expectile` is overridable via dotlist on `critic/gciql.yaml` (add to `_INERT_ALIASES`-style guard only if a nested alias exists that is silently inert).

### W4 (KT-A, no code): checkpoint forensics
`scripts/reeval_checkpoint.py` on `$SCRATCH/factored-fb-density-port/kt1-qc03/mw-pick-place_density_fb_flowbc_seed_2/params_350000.pkl` (scored 0.02) and `params_400000.pkl` (0.80), 50 episodes each, plus the same pair re-evaled a second time with a different eval seed offset. Outcome decides the paper's word: reproducible 0.02 → real per-checkpoint instability; ~0.5+ on re-eval → 50-episode eval variance. Runs as one small SLURM job (login nodes cannot run JAX — pthread limits).

## Experiment arms (all 1-seed screens on the DEV cell pr `mw-pick-place` unless stated; same launcher pattern/protocol as `scripts/launch_density_port_screens.sh` — BASE + E3P override strings, 500k steps, `EVAL_INTERVAL=50000`, per-TAG SAVE_DIR under `$SCRATCH/factored-fb-density-port/`)

| arm | override (on top of BASE+E3P) | reads on |
|---|---|---|
| ktB-sarsa | `measure_density_bootstrap=sarsa` (q_coeff stays 1.0!) | M1 sufficiency + ceiling: stable-at-full-strength ⇒ M1 sufficient; its LEVEL is the cleanest 0.99-reachability read |
| ktC-atemp2 | `actor.q_coeff=0.3 measure_density_actor_temp=2.0` | M2 residual: qc03+smooth ≥0.95 stable ⇒ ~0.85 not structural |
| ktC-nocenter | `actor.q_coeff=0.3 actor_q_center=false` | M4 (mechanism-side; bridge asserts center=true) |
| ktD-e07 / ktD-e095 | GCIQL: `expectile=0.7` / `0.95`, run on pr pick-place AND pr push (4 jobs) | M3: pick-place invariant ⇒ M3 dead for the edge; push moving ⇒ explains GCIQL's push mediocrity |
| ktB-sarsa-prh (only if ktB-sarsa ≥0.9 stable) | same, `FB_DATA_ROOT=play-resample-hard` | precision generalization |

Budget: ≤8 jobs + 1 reeval job. Landmines (verbatim from prior specs): dotlist is a LIST; hit-spec quotes via `tr`; never `--dependency`; `.o` logs over eval.csv (`\r\n` + null bytes); per-TAG SAVE_DIR; eval reseeds the global RNG; login nodes cannot run JAX.

## Gates & interpretation
- Screen gate (unchanged): 500k final ≥ 0.55 AND drop-from-own-peak ≤ 0.25; graduation → 3 seeds.
- **Freeze interaction:** Part B's config freeze does NOT wait for this spec — it proceeds with the best of {qc03, soft, qc03+soft} when their 3-seed results land. If ktB-sarsa or ktC-atemp2 beats the frozen config by ≥0.1 on the dev cell at 3 seeds, that is a *second-generation* freeze: rerun the column again or ship as an ablation row, per the uniform-config rule — never mix.
- **KT-F** (OOD-goal-split density cell) is already mandated by Part B (`launch_density_table.sh ogb`); run it with the frozen config. Bridge prediction: density ~0.65 takes the cell from GCIQL's 0.492.
- Paper sentences this spec buys: (i) whether "GCIQL's remaining edge is the in-sample backup" or "…the smooth readout"; (ii) whether density's precision deficit is structural or an instability artifact; (iii) whether GCIQL's push weakness is expectile pessimism.

## Curve facts to reuse (do not remeasure)
GCIQL pr pick-place: 0.88–1.00 from 100k, sd-last3 ≤0.034. qc03: peaks 0.98/0.90/0.82, mean-last3 0.927/0.727/0.813, seed-2 350k eval = 0.02; action-gap flat 0.13–0.17 (R1 arrested), ESS ~13. soft: peak 0.90/final 0.80, ESS ~400. GCIQL pr push: oscillates 0.42–0.88, mean-last3 0.65–0.76 vs density 0.79–0.83 stable. Bridge doc: density wins iff a manipulated object with dwell-heavy diffuse occupancy exists; two-sided null on antmaze.

## Results (2026-08-18, 1 seed each, DEV cell pr `mw-pick-place` unless stated)

| arm | peak | final | drop | gate |
|---|---|---|---|---|
| ktB-sarsa | 0.180 @100k | 0.100 | 0.08 | fail |
| ktC-atemp2 | 0.940 @350k | **0.900** | 0.04 | **GRADUATES** |
| ktC-nocenter | 0.740 @400k | 0.520 | 0.22 | fail (final) |

### M1 is necessary but NOT sufficient — and severing it is worse than weakening it

`ktB-sarsa` was the spec's cleanest 0.99-reachability read. It fails, and *how* it fails is the
finding. Taking the actor out of the critic's fixed point does exactly what it was supposed to
mechanically — the collapse disappears (the curve is FLAT at 0.02–0.18, never decaying) and the
measure is the healthiest of any arm (`density_ess_median` 50 → 106 → 69, against ~13 for qc03
and 20 → 10 for atemp2). But the policy signal goes with it: `action_gap_l2` grows to 0.74 and
stays, `density_sarsa_subst` sits at 0.61–0.64, and success never leaves the floor.

Under SARSA the target is the *behaviour* policy's occupancy, which is diffuse and not an
improvable objective — ascending it does not correspond to reaching the goal. So the actor
being inside the target is not merely the collapse's cause, it is also where the learning
signal comes from. The two are the same coupling. That is why `q_coeff=0.3` (weaken the
ascent) graduates while `sarsa` (sever it) does not, and it means M1 cannot be "fixed" by
removal — only damped.

### M2 is real and worth ~0.1 over qc03 alone

`ktC-atemp2` (`q_coeff=0.3` + actor-readout temperature 2.0) is the best density arm to date:
0.94 peak, 0.90 final, drop 0.04, and the last five evals are 0.64/0.86/0.94/0.88/0.86/0.90.
qc03 alone is 0.900/0.560/0.820 across three seeds — seed 1 decays; atemp2 removes that at
seed 0. Smoothing ONLY what the actor ascends (the CE target keeps `measure_density_temp`)
buys stability, so the residual gap was at least partly readout spikiness, not a structural
ceiling. It falls just short of the spec's ≥0.95 bar, so "≈0.85 is not structural" is
supported but the 0.99 question stays open. **Next: 3 seeds on ktC-atemp2.**

`ktC-nocenter` (M4) reaches 0.74 but ends at 0.52 with a 0.22 drop — the uncentered normalizer
does not help, consistent with the bridge asserting `center=true` on this path.

### M3 confirmed — for push, not for the pick-place edge

GCIQL expectile sweep on pr (baseline expectile 0.9 is the 3-seed table value):

| expectile | mw-pick-place | mw-push |
|---|---|---|
| 0.70 | 0.860 | **0.840** |
| 0.90 (baseline) | 0.993 | 0.733 |
| 0.95 | 0.980 | 0.640 |

Push is **monotone decreasing in expectile** (0.840 → 0.733 → 0.640, a 0.20 span against the
±0.05 eval noise W4 measured), while pick-place stays high and roughly flat. Both halves of
the pre-registered discriminator fire: expectile pessimism is what costs GCIQL on push, and it
is *not* what its pick-place edge is made of.

Consequence worth stating in the paper: at `expectile=0.7` GCIQL reaches 0.840 on push, which
erases density's push win (0.820). Density's advantage on that cell is against a
*mis-tuned* GCIQL, not against GCIQL's best. These are 1-seed screens — confirm at 3 seeds
before the claim goes in the table.

### Not run
`ktB-sarsa-prh` was gated on ktB-sarsa ≥0.9 stable; it fails, so the job was never spent.
