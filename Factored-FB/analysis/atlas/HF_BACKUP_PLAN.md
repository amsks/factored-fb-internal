# HuggingFace checkpoint backup — sequenced plan

Decisions (2026-07-15):
- Scope: ALL real checkpoints, all domains/seeds (~179 GB). Drop 25 smokes + 3 single-file
  cruft roots (gciql_outputs2/3/vis).
- Ordering: canonical taxonomy from results_map -> `<framework>/<env>/<agent>/<actor>/<variant>/seed<N>/`.
- Reorg = **non-destructive COPY** of whole run-dirs (preserves each run's `.hydra/` +
  `checkpoints/` internally), so NO existing script breaks. Originals untouched.
- Script fix: the ~15 scripts hardcoding fragile (mostly volatile /dev/shm) paths are a
  SEPARATE follow-up (a single identity->path resolver module), NOT part of the backup.

## Sequence

1. [RUNNING] Primary A C2/C3 (atlas experiment) — waiter armed. Do the heavy copies AFTER
   it finishes to avoid contention (CPU/IO on a GPU-less box).
2. [QUEUED] /dev/shm -> persistent copy (~90 GB delta: gciql_outputs 80G + factored-fb 7.1G
   + datasets 3.4G). COPY ONLY, no delete (user choice). Skip .venv-jax (live + rebuildable).
3. [DONE] MANIFEST.csv generator (scripts/hf_manifest.py). jax 180/180 mapped; torch 303
   runs -> 60 experiments, self-describing via .hydra override hints.
4. [NEXT, safe now] Torch 60-entry hint->variant map -> completes MANIFEST to 100% mapped.
5. [QUEUED] Reorg: rsync each real run-dir (whole, intact) into the clean tree + write
   README.md and the final MANIFEST.csv into it.
6. [QUEUED] `huggingface-cli upload` from the clean tree.

## Follow-up (separate task, own review)

- checkpoint-resolver module: identity (env/agent/actor/variant/seed) -> persistent path.
- migrate the ~15 hardcoded-path scripts to it (they also gain volatile-loss resilience):
  reproduce_eval, task_distribution_probe, phase_probe_gciql, z_inference_ladder,
  morph_readout, grad_scale_audit, mlp_readout_control, probes/*, figures/outcome_clips, etc.

## Manifest coverage (final, after overrides-group fix)

508 runs = 483 real + 25 smoke(drop).
- 359/483 real MAPPED to a paper cell (env/agent/actor/variant + success ± CI).
- 124 real IDENTIFIED but not-in-paper (24 wandb_groups: HP sweeps psm/rldp_state_sweep_*,
  extra baselines fb_state_v1/v2/L256/encmatch, tdmpc2/rldp/psm 10seeds, pixel runs).
- Only 7 runs have NO group ([nogroup] fb_flowbc x6, tdmpc2 x1) -> hint-based id.

Bug fixed: torch wandb_group lives in .hydra/OVERRIDES, not config.yaml (which is null).

## HF tree buckets
- mapped (359):     <fw>/<env>/<agent>/<actor>/<variant>/seed<N>/   (canonical)
- grouped-only(124): <fw>/misc/<wandb_group>/seed<N>/
- nogroup (7):       <fw>/unidentified/<hint>/seed<N>/
