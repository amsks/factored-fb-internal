# Design: Clean up `feat/jaxify` into a standalone JAX (OGBench-shaped) FB↔GCIQL bridge repo

**Date:** 2026-07-22
**Branch:** `feat/jaxify` (worked in worktree `/home/adityamohan/git/Factored-FB-jaxify`)
**Status:** approved design, pending spec review

## Goal

`feat/jaxify` becomes a **single-stack JAX repo**. The `impls/` package is the only
implementation; all torch code and the older JAX shim/campaign path are removed; the
top-level is OGBench-shaped; the README describes the new world. No behaviour change to
`impls/` beyond porting a live gap surfaced by the audit.

## Context / findings

- `feat/jaxify` branched off `gciql-fb-bridge` and added a **clean-room JAX package
  `impls/`** (OGBench-idiomatic: `impls/agents`, `impls/critics`, `impls/actors`,
  `impls/utils`, `impls/configs`, `impls/main.py`). It decomposes the bridge into a
  **critic × actor seam**:
  - critics: `fb, gciql, isfb, iql_bilinear, simple_sf, sf_feature, centered_bilinear,
    fb_forward_bilinear`
  - actors: `flowbc, ddpgbc, awr, fql`
  - each with 1e-10 torch-parity fixtures under `tests/impls/fixtures/`.
- **`impls/` is fully self-contained.** It imports only from `impls/` plus the `ogbench`
  pip package — **zero** imports from the torch top-level (`agents/`, `envs/`, `buffers/`,
  `evals/`, `nn_models.py`, …). Verified by grep. => deleting torch cannot break `impls/`.
- The branch still carries, in parallel: the entire torch stack (`agents/{fb,psm,rldp,
  tdmpc2}`, `train.py`, `base_model.py`, `nn_models.py`, `buffers/`, `envs/`, `evals/`)
  **and** the older JAX path the research campaigns actually ran on
  (`tools/wandb_mode_shim/gciql_flow.py` [~1950 LOC], `run_gciql.py`, ~60
  `configs/gciql/*.yaml`).

## Decisions (locked)

1. **Port scope = bridge family only.** Only the FB↔GCIQL bridge agents must exist in JAX
   (they already do, in `impls/`). **PSM, RLDP, TD-MPC2 are out of scope** and are removed
   from this branch entirely (they live on their own branches:
   `feat/psm-integration`, `rldp-*`, etc.).
2. **`impls/` is the only survivor.** The old JAX shim/campaign path
   (`gciql_flow.py`, `run_gciql.py`, `configs/gciql/*`) is deleted, **gated on a coverage
   audit** (Phase 1) so no research capability is silently lost.
3. **Execution = audit-gated in-place cleanup** (Approach A). Stay on `feat/jaxify`
   (in an isolated worktree). Preserve git history/blame and the parity fixtures.
4. **Dead-path knobs are dropped, not ported.** A shim knob whose only use was a campaign
   that concluded negatively (recorded in project memory) is dropped with a one-line note
   in the audit citing the campaign — not reimplemented in `impls/`.
5. **Pixel/DrQ is deferred.** `impls/` stays state-only. The pixel path
   (`encoder`, `p_aug`, `frame_stack`, `*_visual_*_drq.yaml`) is tracked as an explicit
   follow-up, not part of this cleanup. If the Phase-1 audit finds a *live state-path*
   dependency on pixel code, that is flagged and re-scoped before any deletion.

## Plan

### Phase 1 — Coverage audit + gap closure (the safety gate)

The `impls/` critic configs were deliberately authored to mirror `gciql_flow.py`'s knobs
(the config comments say so). The audit makes that claim explicit and checked.

1. Produce `docs/jaxify-coverage-audit.md`: a table with one row per research knob found
   in `gciql_flow.py` / `configs/gciql/*.yaml`. Columns: knob, what it does, `impls/`
   status ∈ {**covered** (name the critic/config key), **dead-path-drop** (cite the
   campaign memory that killed it), **live-gap-port**}, note.
   - Enumerated knob surface (from a scan of the configs + `gciql_flow.py`): `value_arch,
     value_latent_dim, bc_coeff, measure_*, sf_*, fb_forward_z, project_z, fwd_two_tower,
     centered_*, iql_ortho_coef, actor_cond, actor_z_*, expectile, double_q, isfb, alpha,
     anneal_*, freeze_critic_ckpt, measure_z_random_frac, subgoal_steps, low_actor_rep_grad,
     encoder/p_aug/frame_stack (pixel)`.
   - Prior candidate classifications (to confirm during audit, not assume):
     - Most `measure_*`, `isfb`, conservatism/`centered_*` arms → **dead-path-drop** per
       T7/T8 + measure-cell campaigns (measure fixed point is bad on cube).
     - `anneal_*` → likely **dead-path-drop** (anneal campaign: IQL was propping the basin).
     - Core knobs (`value_arch, value_latent_dim, bc_coeff, project_z, fb_forward_z,
       fwd_two_tower, sf_feature_td, actor_cond, expectile, double_q`) → expected **covered**
       by existing `impls/` critic configs.
     - Pixel (`encoder/p_aug/frame_stack`) → **deferred** (decision 5).
2. For each **live-gap-port** row: implement the knob in the relevant `impls/` critic and
   add a parity or smoke test. (Expected: few or none.)
3. **Gate:** full `tests/impls/` suite green under `/dev/shm/.venv-jax`
   (CPU, `XLA_PYTHON_CLIENT_PREALLOCATE=false`). Record the pass count. **Nothing in Phase 2
   is deleted until this gate is green and the audit table has no unresolved live gap.**

### Phase 2 — Deletion + restructure (mechanical, gated on Phase 1)

Delete (torch):
`agents/`, `base_model.py`, `base_config.py`, `nn_models.py`, `normalizers.py`,
`train.py`, `resume.py`, `run_gciql.py`, `utils.py`, `buffers/`, `envs/`, `evals/`.

Delete (old JAX shim/campaign path):
`tools/wandb_mode_shim/*`, `configs/gciql/*`, and torch-only entries under `scripts/`.

Delete (orphaned tests): `tests/agents/tdmpc2/`, `tests/atlas/`, and any torch test
(e.g. `tests/test_torch_isfb_*`). Keep `tests/impls/` intact.

Keep: `impls/`, `tests/impls/` (+ fixtures), `third_party/ogbench` (parity reference),
the `ogbench` pip dependency, `pyproject.toml`, `Makefile`, `LICENSE`.

Restructure to OGBench shape (OGBench itself keeps a top-level `impls/` dir, so `impls/`
does **not** move):
- Repo root after cleanup = `impls/` + `tests/` + `third_party/` + packaging files
  (`pyproject.toml`, `Makefile`, requirements) + `README.md` + `docs/`.
- Prune `requirements*.txt` / `pyproject.toml` / `Makefile` to JAX-only deps and targets
  (drop torch, `exca`, DMC, torch smoke targets).
- **Research-artifact trees are removed from this branch.** On `feat/jaxify` these are
  *tracked* (`PAPER/` 145 files, `analysis/` 12, `results/` 3, `data/` 2), unlike the
  CLAUDE.md convention. `feat/jaxify` becomes a clean **code-only** branch:
  `git rm -r PAPER analysis results data`, then add them to `.gitignore` so the branch
  stays clean. This is branch-scoped: the removal is a commit on `feat/jaxify` only —
  other branches (e.g. `gciql-fb-bridge`) keep them tracked, and the **main worktree's**
  on-disk copies (a separate working directory) are untouched. The user's requirement is
  "keep PAPER on disk and on other branches, but not on `feat/jaxify`."

Each deletion batch is its own commit with a message naming what went and why it was safe
(self-contained `impls/`, out-of-scope agent, dead-path config).

### Phase 3 — README rewrite

Replace the current torch-era README (it still describes building `FBAgent`/`FBFlowBCAgent`,
`device=cpu`, a torch `.venv`, `make smoke` over Box obs) with a JAX-first one:
- Method recap (`Q(s,a,z)=F(s,z,a)·B(s')`, z-inference at test time).
- Install: JAX venv + `ogbench` (+ how the parity fixtures / `third_party/ogbench` relate).
- The **critic × actor seam** and the **valid combo matrix** (from `impls/agents/__init__`
  `CRITICS`/`ACTORS` + `check_combo`).
- Running `impls/main.py` with Hydra/OmegaConf config composition (critic/actor/domain
  groups).
- Getting OGBench data.
- Running the test/parity suite.

## Verification

- `tests/impls/` green **before** Phase 2 (baseline) and **after** Phase 2 (identical pass
  set) — deletion must not change any pass/fail.
- `grep -rE '(^|[^.])(agents\.|from agents|import train|base_model|nn_models|gciql_flow)'`
  over the post-cleanup tree returns no surviving reference to a deleted torch/shim module
  (excluding `third_party/` and `docs/`).
- One end-to-end smoke: at least one critic × actor pair runs a train step via
  `impls/main.py` on cube-single.

## Out of scope

- PSM, RLDP, TD-MPC2 (remain on their own branches).
- Any new agent or behaviour change to `impls/` beyond a live-gap port.
- Pixel/DrQ port (deferred, decision 5).
- Scene domain + `value_split` factored value (deferred, like pixel; `impls/` stays
  cube-single).
- Merging `feat/jaxify` anywhere — this cleanup does not touch `main`/`gciql-fb-bridge`
  or the main worktree's on-disk artifacts.

## Baseline recorded

Worktree `/home/adityamohan/git/Factored-FB-jaxify`, venv `/dev/shm/.venv-jax`
(jax 0.10.0, flax 0.12.7, ogbench importable). Quick parity baseline green:
`tests/impls/test_fb_math.py tests/impls/test_fb_critic_static.py` → 5 passed.
