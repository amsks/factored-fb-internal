# PSM & RLDP hyperparameter sweep (state) — tuning parity with FB

**Date:** 2026-06-08 · **Branch:** `feat/psm-integration` (or a fresh sweep branch)

## Motivation

There is a tuning asymmetry across the future-reachability comparison. **FB-state**
was hyperparameter-swept (`scripts/run_sweep.sh`: `ortho_coef ∈ {100,1000} × lr_b ∈
{1e-4,1e-5}`; `fb_state_plain` = `ortho_coef=1000, lr_b=1e-4` is that grid's winner).
**PSM and RLDP were not swept** — PSM runs reference defaults, RLDP inherits FB's HPs
with a hand-set `ortho_coef=0.01` that was never validated. Conclusions that "method X
fails on manipulation" are only fair if every method got comparable tuning effort.

This sweep closes that gap on the **state** modality (the only modality FB was actually
tuned in). Pixel is untuned for all three methods and is deferred.

## Faithfulness audit (done 2026-06-08; prerequisite for the PSM change)

Audited paper (`~/PSM-PAPER/src/*.tex`, `psuedocode/psm.py`) ↔ reference
(`~/git/baselines/PSM/agent/psm.py`) ↔ ours (`agents/psm/`):

- **Ours is a faithful, bit-exact replication of the reference.** `_update_psm`,
  `_update_sf`, `_update_td3_actor` are byte-for-byte equivalent, pinned by
  `tests/test_psm_reference_equivalence.py` (CPU/float64/`torch.equal` on losses, grads,
  one Adam step). The **only** substantive deviation is the FlowBC actor (isolated to
  `agents/psm/flow_bc/`; base TD3 path untouched).
- **The "dead `ortho_coef`" is the reference's own behavior, and matches the paper.**
  Reference `psm.py:464` `psm_loss += 1 * orth_loss` (φ basis regularized at strength
  **1**); `psm.py:519` `sf_loss += 0*ortho_coef * orth_loss` (off in the SF branch, where
  φ is detached → no-op anyway). The paper states orthonormality weight **= 1** (fixed,
  no equation, no measure-vs-SF distinction). So orthonormality is genuinely active at the
  paper's λ=1; the `ortho_coef` knob just isn't *wired* to the live term.

## Components

### 1. Make PSM `ortho_coef` live (1-line, guarded, bit-exact at default)

In `agents/psm/agent.py` `_update_psm`, change the orthonormality combine:

```python
# before (hardcoded, reference-exact):
psm_loss = psm_loss + 1 * orth_loss
# after (configurable; default ortho_coef=1.0 reproduces the reference bit-for-bit):
psm_loss = psm_loss + self.ortho_coef * orth_loss
```

`ortho_coef` already exists as a constructor kwarg / config field (default `1.0`). At the
default this is numerically identical to `1 * orth_loss`, so
`tests/test_psm_reference_equivalence.py` still passes (it asserts `orth_loss` and the
total `psm_loss` bit-exactly). Leave the SF-branch `0 * self.ortho_coef * orth_loss`
untouched (faithful no-op). Document in the agent header that `ortho_coef` is now an
**intentional ablation knob beyond the paper** (paper uses fixed 1).

### 2. Generic grid-sweep launcher

`scripts/run_sweep.sh` hardcodes FB's `ortho_coef`/`lr_b` hydra keys and the FB agent.
Add a method-parameterized grid runner (reuse its GPU-slot scheduler) — e.g.
`scripts/run_grid_sweep.sh` with env knobs:

```
AGENT=psm_flowbc | rldp           # hydra agent
DOMAIN=cube_single
KNOB_A_NAME=ortho_coef            KNOB_A_VALS="0.1 1 10"
KNOB_B_NAME=lr_phi | lr_b         KNOB_B_VALS="1e-4 1e-5"
SEEDS="0 1 2"   TRAIN_STEPS=500000
N_GPUS=8  JOBS_PER_GPU=<fit>      RUN_GROUP_BASE=<method>_state_sweep
DATA_PATH=/dev/shm/factored-fb/datasets   wandb_entity=amsks
```

Run-group / run-name encode both knob values (e.g.
`psm_state_sweep__ortho10__lrphi1e-5__s0`). Round-robin GPUs; several jobs/GPU (state
models are small). Launch inside **tmux** ([[always-run-in-tmux]]).

### 3. The two grids (state, 3 seeds each, 500k steps for selection)

- **PSM** (`agent=psm_flowbc`): `ortho_coef ∈ {0.1, 1, 10}` × `lr_phi ∈ {1e-4, 1e-5}`
  = 6 combos × 3 seeds = **18 runs**. (`ortho_coef=1, lr_phi=1e-4` = current default,
  serves as the in-grid control.)
- **RLDP** (`agent=rldp`): `ortho_coef ∈ {0.001, 0.01, 0.1}` × `lr_b ∈ {1e-4, 1e-5}`
  = 6 combos × 3 seeds = **18 runs**. (`ortho_coef=0.01, lr_b=1e-4` = current default.)

Total **36 runs @ 500k** (~6–7h each; ~one overnight parallelized over 8 GPUs).

### 4. Selection + winner runs

Selection metric: mean `eval/success` over the 3 seeds at the final eval (≈500k),
tie-break by mean over the last K evals (robustness to a single noisy eval). Pick the
winning (knob_a, knob_b) per method. Then launch the winner at **1M × 10 seeds**
(mirroring `fb_state_plain`), run groups `psm_state_tuned` / `rldp_state_tuned`.

## Selection / analysis

`scripts/summarize_sweep.py` (or extend `scripts/plot_sweep.py`): read each run's
`wandb/.../output.log` (or the per-run json) → table of (method, knob_a, knob_b) →
mean±std `eval/success` over seeds + the per-task breakdown; emit
`analysis/sweep/<method>_state.{md,csv}` and a heatmap per method
(`ortho_coef × lr` grid colored by success). Plotting/IO is TDD-exempt; any pure
aggregation helper gets a unit test.

## Tests / verification

- `tests/test_psm_reference_equivalence.py` still **passes** after the `ortho_coef`
  change (default 1.0 ⇒ bit-exact). Run it explicitly as the guard.
- New: `ortho_coef ≠ 1` actually changes the PSM loss — a small test that `_update_psm`
  with `ortho_coef=10` differs from `ortho_coef=1` (the knob is now live).
- `--dry-run` on `run_grid_sweep.sh` prints the 18+18 commands with correct hydra keys
  before any GPU launch.

## Scope / caveats

- **State only.** Pixel tuning for PSM/RLDP/FB is a separate (larger) campaign.
- Sweeping `ortho_coef` is a deliberate **deviation from the paper's fixed λ=1** (an
  ablation), enabled by component 1; the default remains paper-faithful.
- 500k is a **selection** budget (FB success plateaus by ~500k); the reported numbers
  come from the 1M × 10-seed winner runs, not the 500k grid.
- wandb entity **`amsks`** ([[wandb-entity-amsks]]); data in `/dev/shm/factored-fb/datasets`.
