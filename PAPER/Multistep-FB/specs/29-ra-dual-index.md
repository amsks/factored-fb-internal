# Spec 29 — R-A: the dual-corrected index, plus the supervised topline

Status: WRITTEN 2026-08-14, user-approved (R-A build + spec27-Q4 topline
green-lit; antmaze plain-FB cell resolved by decision, §4). House rules as
specs 27/28: TDD, default-off bit-identical knobs, thin wrappers over the
committed spec-28 eval code with control gates against archived artifacts,
tests `.venv/bin/python -m pytest`, evals CPU jax, training in tmux, wandb
amsks, no co-author lines.

Evidence base (spec 28 P0, commit 7af7c07): Σ_B is rank-1 dominated (eff
rank 1.0002/50, cond 1e6–1e7) ⇒ z(g)=√d·norm(B̄(g)) is collapsed (pairwise
cos .998–1.000) and the primal moment is near-orthogonal to the dual
coefficient (cos .010–.025). Ridge readout fixes reward reconstruction
(AUC .99–1.00) and reward→index assignment (chance → .000–.008). All of
that is READOUT-ONLY; conditioning on a dual index needs retraining —
that is this spec.

## 1. R-A: what it is

One new encode knob, composed with the already-built (never-run) P1
readout knob:

- `measure_density_z_encode ∈ {sphere (default), dual}`. `dual`:
  ẑ = √d · normalize( (Σ̂_B + λI)⁻¹ m̂ ), where m̂ = (1/M) Σ_j B̄(u_j)·r_j
  over the row's query set — for hindsight rows r_j = kg_j =
  `_delta_phi(goals, u_raw)` (the query-set-normalized kernel; match it
  exactly, spec 28 rule). ‖ẑ‖=√d is re-imposed after the ridge solve so
  the conditioning distribution keeps the standing invariant.
- Arms (cube-single 500k, s0 screen):
  - **R-A/kernel** = `z_encode=dual` + `measure_density_z_actor_reward=
    kernel` (the spec-28 cell-ii knob, now fundable): actor integrates the
    ACTUAL reward samples, q = Σ_j w_j(s,a,ẑ)·r(u_j). Latent = policy
    index only; reward semantics external — what made the g path work.
  - **R-A/latent** = `z_encode=dual` + latent readout `B̃ᵀẑ/√d`. With the
    dual index this readout IS the ridge-reconstructed reward (P0b AUC
    .99–1.00), so this arm tests whether the closed-form interface (no
    reward samples at actor time) is recovered. Secondary but cheap.

### Implementation contract

- Σ̂_B (and μ if centered — pin UNCENTERED second moment, matching
  `spec28_encodings.py`; assert equivalence in a test against its pool
  computation) is EMA-tracked over query-batch B̄(u) outer products and
  stored as ORDINARY PARAM LEAVES (the `density_b_zenc` pattern) so it
  serializes and `sample_actions` reads the identical object. Batch stats
  are forbidden — they reintroduce the train/eval mismatch that broke the
  first z build.
- B̄ for the encode = the slow-EMA encoder tower (`enc_tau>0`, C3
  machinery) — REQUIRED, not optional: the ridge solve through a fast-
  drifting B is the known non-stationarity risk.
- λ = 0.01·tr(Σ̂)/d default; knob over {.001,.1} (the three P0c-winning
  scales). Log eff_rank(Σ̂) and cond(Σ̂) every eval interval.
- Eval-time z*: stored (serialized) Σ̂ + m̂ from the standing P=16384 pool
  — same object train and eval.
- Task mixture unchanged (`z_rand_frac=.5`, fresh √d-sphere rows).
  Random-sphere rows have no reward samples ⇒ in R-A/kernel the actor-q
  term is masked/renormalized via the existing `z_hindsight` machinery
  (already built and tested for cell ii). Behaviorally-grounded random
  tasks (random dataset-goal kernels instead of spheres) = follow-up
  knob, default OFF — no multi-variable changes in the first wave.
- Default-off bit-identity: with both knobs at default, update() and
  sample_actions bit-exact over 3 real steps; mutation-tested (mirror the
  P1 knob test suite).
- ⚠ This is a NEW ARM: archived z24/z25 numbers stop being comparable to
  it. Never mix in one table without saying so.

### Gates, in order

- **G29-0 (pre-flight, eval-only, BEFORE any training — minutes):** on
  existing z24/z25/g ckpts, compute dual indices ẑ(g) over the 4096-goal
  set with the P0 machinery and measure mean pairwise cos (P0a
  instrument). PASS = de-collapsed (≪ .99; sphere ref .002). If dual
  indices are also collapsed, the arm is dead — do not launch.
- **G29-1 (mid-run, from first eval):** `eval_z_action_ranking.py` on the
  training ckpt — actor goal-sensitivity must lift off 0.000 (noise floor
  .19); pairwise cos of train-time ẑ; `density_w_ztv`; never-extracts
  curve. An arm flat at sensitivity 0 by 250k is killed (don't wait for
  500k — but never quote mid-run SUCCESS numbers, spec-18 rule).
- **G29-2 (screen):** cube-single 500k s0 vs anchors g .93 / BC .111.
  Success = clearly off the BC floor; promote to cube-double (+ s1)
  vs g .736 / BC .019.
- **G29-3 (the payoff, binding user requirement):** zero-shot reward-
  family eval (region/dense/hard, the standing G3-style harness) through
  the dual interface: reward samples → ẑ → policy, no fine-tuning. This
  is the general-reward-interface claim; nothing below it licenses that
  sentence in the paper.

## 2. Q4 — offline supervised topline (green-lit)

As contracted in spec 27 Q4, now approved: offline TD3 — or the repo's
stock single-task baseline if one exists, check first — trained with the
NATIVE task reward on the same data; cube-single, cube-double, scene;
3 seeds each; standing recipe budget. Deliverable = a topline row so the
paper quotes both absolute success and %-of-supervised, as the FB
literature does. Run as FILL-IN compute behind the R-A screen — R-A owns
the GPUs when they conflict.

## 3. Priority

G29-0 today (eval-only) → R-A/kernel + R-A/latent cs screen → Q4 topline
fill-in → promote R-A winner to cd → G29-3 reward families.

## 4. Resolved by decision (do not run)

- **Antmaze plain-FB ddpgbc cell:** NOT rerun. The value is already known
  and will be sourced from TD-JEPA (`td_jepa` reference results). Action
  is bookkeeping only: replace the tex `pending` cell with the TD-JEPA
  number, cite provenance in the caption, and drop the broken launch from
  the queue.

## 5. Bookkeeping

- New groups (suggest `ra_cs_*`, `ra_cd_*`, `q4_top_*`) →
  `scripts/results_map.py` + `compile_results_csvs.py --refresh`.
- Results backup to HuggingFace is being handled separately (2026-08-14);
  new R-A artifacts join the same backup prefix when collected.
