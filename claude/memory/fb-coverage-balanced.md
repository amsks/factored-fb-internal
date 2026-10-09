---
name: fb-coverage-balanced
description: "Coverage-balanced FB reweight experiment — impl, data staging, and the after-CRL auto-launch"
metadata: 
  node_type: memory
  type: project
  originSessionId: 52c7de9a-dc0d-44e3-9c0a-6711de243819
---

Coverage-balanced (goal-agnostic) FB ρ′ reweight — spec
`docs/superpowers/specs/2026-05-25-coverage-balanced-fb.md` (supersedes the diag-only
Experiment B of `2026-05-25-fb-Bg-cause-isolation.md`). Implemented + committed on
`gciql-pixel-integration` (impl `c952ab2`, launchers `50d7935`).

PyTorch FB code (`agents/fb/agent.py`, `train.py`) — SEPARATE from the JAX/OGBench CRL
path. Reweights EVERY goal-side term by w(g) (inverse-density over offline cube xyz,
mean-norm, clipped): `coverage_weights`, `fb_successor_terms` (off-diag+diag),
`ortho_cov` (B~=√w·B — the centerpiece the diag-only version missed), q-cov. Flag-gated
`reweight_alpha` (0=byte-identical). Goal-agnostic (no eval goals). Tests
`tests/test_fb_reweight.py` (10, green).

Dataset: FB needs `datasets/<env>/buffer/*.npz` (per-episode, `physics`=qpos, cube
xyz=physics[:,14:17]); IT IS the same OGBench data the CRL uses, just extracted.
`scripts/extract_ogbench.py --env cube-single-play-v0 --dataset_path ~/.ogbench/data
--output_folder /dev/shm/factored-fb/datasets` → staged 1000 eps (176M) in /dev/shm
(durable / is 100% full). Density: `scripts/cube_density.py` → `analysis/cube_density.npz`
(16×16×8, 1790 nonzero cells). CRL data lives at `~/.ogbench/data/*.npz` (→/dev/shm,
OGBench monolithic format, different from FB's).

Launch: `scripts/run_fb_reweight.sh` (10 seeds, α default 0.5, ortho_coef=1000, all
outputs→/dev/shm, wandb online). 2026-05-25: GPUs (8×23GB) were full with the CRL
state campaign, so `scripts/fb_after_crl.sh` watcher (detached, ALPHA=0.5) polls GPU
free mem and auto-`exec`s the sweep once all 8 free; log `/dev/shm/fb_after_crl.log`.
Measurement probes (factorization_probe, privileged_z_probe) live on
`feat/cube-analysis-pipeline`, NOT here — needed to evaluate the sweep. See
[[crl-actor-loss-critic-gradient]], [[crl-flowbc-campaign]].
