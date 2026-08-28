# ExoRL official-parity validation: vanilla RLDP + HILP, then their density variants

Status: WRITTEN 2026-08-28, design user-approved in session (RLDP+HILP first;
HILP port in parallel; vanilla replication GATES the density arms; PSM
decision deferred until the user sees this campaign's report). Intended
executor: a fresh subagent session working in `~/git/Factored-FB-seam`
(branch `density-fb`, fast-forwarded to 3bf818a on 2026-08-28 with WIP
preserved — the WIP off-by-one fix and the untracked `*_density` drafts are
part of the baseline, do NOT discard them).

## 0. Evidence base (read these first)

- `docs/notes/2026-08-28-rldp-audit.md` — line-by-line audit, seam vs
  official `~/git/baselines/rldp` vs td_jepa. Verdict: seam = faithful
  td_jepa-variant transcription (61/61 tests in `/dev/shm/.venv-jax`);
  official variant differs structurally (items §1 below). WIP SP-alignment
  fix is CORRECT vs both references. `rldp_density` invariants all hold.
- `docs/notes/2026-08-28-rldp-paper-eval.md` — official RLDP eval recipe +
  Table 1 (which includes FB/HILP/PSM columns = this campaign's target
  table) + 10 unresolved ambiguities.
- `docs/notes/2026-08-28-hilp-paper-eval.md` — HILP zero-shot recipe, φ
  objective, repo map. ⚠ its Table-3 numeric sketch is UNVERIFIED
  (WebFetch transcription disagreements) — see §6 item 1.
- `docs/2026-08-25-exorl-humenv-compendium.md` — prior arms; T3/T4 caveat.

Motivating fact: our compendium RLDP numbers (walker 198±238 under T3) are
three confounds away from official (td_jepa variant × cube HPs × our eval).
This campaign removes all three at once and measures each method under ITS
OWN paper's recipe.

## 1. Work item A — official-parity RLDP knobs (default-OFF, parent bit-identical)

Add to `impls/critics/rldp.py` (house rule: every knob default-off; parent
td_jepa arm must remain bit-identical — extend `tests/impls/test_rldp_seam.py`
with a default-off equivalence test):

1. `rldp_two_phase` (bool) + `rldp_phase1_steps` (int): phase 1 = SP + ortho
   only (no FB loss, F untouched); phase 2 = FB loss with B **frozen**
   (stop-grad + optimizer masking, not just detach — B params must not move)
   . Official reference: `baselines/rldp rldp/agent.py:167-243`.
2. `rldp_pred_norm` (bool): √d-normalize the predictor output at every
   unroll step (`nn_models.py:324-332` in the reference).
3. `rldp_actor_pessimism` (float, default 0.0): actor maximizes
   mean − coef·unc; official example uses 0.5 (`agent.py:392`).
4. `rldp_hard_b_target` (int, default 0=off): B target = hard copy every N
   steps (official N=250); F target polyak τ=0.01 in phase 2 only.
5. `rldp_no_left_encoder` (bool): identity left encoder (official F consumes
   raw obs; the seam's learned 4-layer PhiMap is a td_jepa-ism).
6. `rldp_sp_official_scale` (bool): SP loss per-step mean over (B,z) summed
   over H (official) instead of sum-over-z/mean ÷h (td_jepa) — ~z_dim/h
   relative reweighting vs ortho.

New config `impls/configs/critic/rldp_official.yaml`: all six knobs ON,
z_dim 512, batch 1024, discount 0.98 (0.99 pointmass), F 1024×1, B 256×2,
predictor as reference, τ_F 0.01, ortho λ=1 in the PAPER's Eq-7 form (note:
official CODE's ortho term is inert under b_norm — we follow the paper,
footnote the code discrepancy; keep a `rldp_ortho_inert` ablation knob so
the code-exact variant is one flag away if the gate fails).

`rldp_density_official.yaml` = same knobs on `rldp_density.py` (the CE
estimator already trains F+encoder only, so the phase-2 B-freeze composes
with it unchanged — the audit verified gradient isolation).

## 2. Work item B — paper-exact eval harness

New `scripts/exorl_paper_eval.py` (adapt, don't replace, the branch's eval
path; the T3/T4 routes stay for continuity):

- **RLDP protocol**: z inferred from 50,000 dataset transitions with
  rewards obtained by RE-SIMULATING MuJoCo physics for the target task
  (reference `rldp/model.py:128-136`, Algorithm 1); z = project to
  hypersphere of E[φ(s′)·r]. Final eval only, 50 episodes/task (paper text;
  code default 10 recorded as an ablation flag), 4 tasks per domain, raw
  episode return, mean±std over 4 seeds.
- **HILP protocol**: z = least-squares regression of re-simulated rewards
  on φ over 10,000 transitions, then √d·normalize (their convention ==
  ours). 50 final-eval episodes, 4 seeds, report both mean±std per cell and
  the IQM aggregate.
- **Task sets** (RLDP Table 1 row names): walker {stand, run, walk, flip};
  cheetah {run, run_backward, walk, walk_backward}; pointmass {reach_top_left,
  reach_top_right, reach_bottom_left, reach_bottom_right}; quadruped
  {stand, run, walk, jump}; (HILP additionally jaco 4 reach tasks, phase-1b).
  The branch already added cheetah walk/walk_backward (commit 6962f86).
- **Buffers**: RLDP arms = local RND 10k-episode buffers as-is (paper does
  not state a subsample). HILP arms = first 5,000 episodes (their stated
  buffer size; jaco 20,000). Record buffer fingerprints in every run dir.
- Record the walker/cheetah RANDOM-POLICY floors once (missing per the
  compendium) into the eval report.

## 3. Work item C — HILP seam port

New `impls/critics/hilp.py` transcribed from the OFFICIAL
github.com/seohongpark/HILP (`sf.py` + `ddpg.py`), same discipline as the
rldp transcription (audit doc = template):

- φ objective (from the extraction, verify against source when transcribing):
  r(s,g) = 1[s≈g] − 1, mask = 1 − 1[s≈g]; twin V_i(s,g) =
  −‖φ_i(s) − φ_i(g)‖; Q_i = r + hilp_discount·mask·V̄_target;
  expectile (IQL) loss on advantage, twin heads, hindsight-relabeled goals.
- Skill/SF layer and DDPG actor exactly as the repo (z on the sphere,
  successor features on frozen-or-joint φ per their code — transcribe, do
  not redesign), official HPs from repo configs (1M state steps).
- Tests: `tests/impls/test_hilp_seam.py` — numpy reconstruction of each
  loss at 1e-10 on a pinned batch; a reference fixture (generate with the
  official repo in a throwaway venv IF torch installable offline; else pin
  against hand-computed values and mark the fixture TODO); default-config
  round-trip through `impls/main.py`.
- `hilp_density` is OUT OF SCOPE until the vanilla gate passes (user's
  process, step 4).

## 4. Work item D — run matrix, gates, budget

Phase 1 (replication gate, local buffers): {walker, cheetah, pointmass} ×
{rldp_official, hilp} × 4 seeds = 24 pretrainings. RLDP budget: paper text
2M SP + 3M FB (code default 3M+3M recorded as ablation); HILP 1M. One run
per GPU, 8 GPUs, tmux (no slurm on this box — adapt the .slurm scripts'
bodies into `scripts/launch_exorl_official.sh`; env vars
`EXORL_DATA_ROOT=$HOME/exorl_data`, `EXORL_CACHE_ROOT` set explicitly).

Phase 1b (parallel, non-blocking): download quadruped (+jaco) RND buffers
(denisyarats/exorl GCS bucket), convert with the branch's converter; join
the matrix when ready.

**GATE (per method, evaluated against the official Table 1 in
`2026-08-28-rldp-paper-eval.md`):** each domain's per-task mean within 1σ
(their std) of the corresponding official cell for ≥ 3 of 4 tasks, no task
catastrophically off (< 50% of official mean where official > 100). RLDP
gates against the RLDP column; HILP against the HILP column (RLDP-authors'
rerun) AND, once re-extracted, HILP's own Table 3 — report both, flag
disagreements between the two published sources rather than picking one.

Phase 2 (only after a method passes): {rldp_density_official, hilp_density
(to be designed)} on the identical recipe, seeds, and eval. Report format:
one table, official | our-vanilla | our-density per cell.

If a vanilla gate FAILS: do NOT tune toward the number. Flip the recorded
ambiguity flags one at a time (steps 3M+3M, eval-10, ortho-inert,
pointmass γ) as labeled ablations, re-audit the relevant code path, and
report the failure with the ablation ladder — the user decides what ships.

## 5. Ops and discipline

- Tests: `/dev/shm/.venv-jax/bin/python -m pytest tests/impls/ -x -q`
  (repo has no .venv; Makefile names this interpreter; it is VOLATILE —
  recreate per the Makefile if missing).
- Every training run in tmux, one GPU each, `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- Commits on `density-fb`: conventional style, NO co-author line. The
  pre-existing WIP (rldp.py fix, datasets.py, *_density drafts) should be
  committed FIRST as its own commit(s) (it is validated by the audit)
  before new work stacks on it.
- Results: `exp/` run dirs + `scripts/exorl_report.py` (pass `--steps`
  explicitly — it silently defaults to 500k). Artifacts stay untracked.
- Do not touch `~/git/Factored-FB` (gciql-fb-bridge) — different track.

## 6. Loose ends folded in

1. Re-extract HILP Table 3 from the downloaded PDF with a PDF-capable
   reader (the Claude Read tool handles PDFs; the earlier WebFetch numbers
   are UNTRUSTED) → update `2026-08-28-hilp-paper-eval.md`.
2. Resolve RLDP ambiguity items 1–5 (step budget, eval episodes, baseline
   dims, per-domain dim, pointmass task set) by the conventions fixed in
   §2/§4; record each choice + the alternative in the final report.
3. Random-policy floors for walker/cheetah (one cheap rollout job).

## 7. Deliverable

One report to the user: the replication table (official vs ours, all gated
cells), gate verdicts per method, the ablation ladder for any failure, and
— only where gates passed — the density-variant columns. The user then
decides whether PSM joins the matrix.
