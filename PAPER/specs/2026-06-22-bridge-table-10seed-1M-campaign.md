# Plan: promote the FB↔GCIQL bridge ablation grid to 10 seeds × 1M steps

**Date:** 2026-06-22  **Status:** plan (no launch / no backup yet — user executes)
**Target:** every row of the `fb-gciql-bridge` full ablation grid at **n=10 seeds, 1M steps**.
**Decision on record:** step budget = **1M** (not 500k / not peak).

> ⚠️ Caveat the user accepted: the bridge analysis sections were *deliberately* the 500k
> snapshot because FB-family methods peak early and **degrade after** (stock GCIQL
> 70.3@500k → 67.4@1M). Moving every row to 1M will likely pull several FB-side rows
> *down* and is a change of measurement, not just more precision. Headline table stays
> "best ≥500k"; this campaign makes the **grid** uniformly 10×1M.

---

## 1. What is reused vs. what is new

| Row | current | action |
|---|---|---|
| stock GCIQL (MLP, Gaussian) | 12 seeds @1M | ✅ reuse (already >10×1M); just recompute @1M |
| GCIQL-FlowBC (MLP, flow) | 9 seeds @1M | ✅ reuse; optional +1 fresh for clean 10 |
| CRL, flow | 10 seeds @1M | ✅ reuse |
| plain FB (anchor) | 10 seeds @≥1M | ✅ reuse |
| bilinear **d=512** | 5 seeds @1M | **+5 fresh @1M** |
| bilinear d=50 / d=128 / d=5 | 3 @500k (+ckpt) | resume 3→1M, **+7 fresh @1M** each |
| τ=0.5 / τ=0.95 / single-Q / Gaussian-actor | 3 @500k (+ckpt) | resume 3→1M, **+7 fresh @1M** each |
| G4 z-cond | 3 @500k (+ckpt) | resume 3→1M, **+7 fresh @1M** |
| FB zHER / FB-IQL critic / FB-IQL q-loss | 3 @500k (+ckpt) | resume 3→1M, **+7 fresh @1M** each |
| **CRL, Gaussian (ddpgbc)** | 3, **no ckpt** | **10 fresh @1M** (cannot resume) |

**New training totals:** ~92 fresh 1M runs + ~33 resume-extensions (each +500k).
≈ **108 1M-run-units**.

---

## 2. Commands

### 2a. GCIQL/JAX (run in TORCH `.venv` which has hydra; it dispatches the child to the jax venv)

Config dir: `configs/gciql/`. Resume uses `restore_path`+`restore_epoch` → shim sets
`GCIQL_STEP_OFFSET`/`GCIQL_CSV_APPEND` automatically.

**Resume an existing seed 500k→1M** (per ablation `<CFG>`, seed `s`, its run dir `<RUNDIR>/sdNNN_*`):
```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=<g> \
.venv/bin/python run_gciql.py --config-name <CFG> \
  seed=<s> train_steps=1000000 save_interval=1000000 \
  restore_path=<RUNDIR>/sd00<s>_<ts> restore_epoch=500000
```

**Fresh seed (3..9) @1M:**
```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=<g> \
.venv/bin/python run_gciql.py --config-name <CFG> \
  seed=<s> train_steps=1000000 save_interval=1000000
```

Config / override per row (verified from each run's `flags.json`):
| Row | `--config-name` | extra overrides |
|---|---|---|
| d=50 | `cube_single_state_gciql_flow` | `agent.value_arch=bilinear agent.value_latent_dim=50` |
| d=128 | `cube_single_state_gciql_flow` | `agent.value_arch=bilinear agent.value_latent_dim=128` |
| d=512 | `cube_single_state_gciql_flow` | `agent.value_arch=bilinear agent.value_latent_dim=512` |
| d=5 | `cube_single_state_gciql_flow_bilinear_d5` | — |
| τ=0.5 | `cube_single_state_gciql_flow_bilinear_d50_tau05` | — |
| τ=0.95 | `cube_single_state_gciql_flow_bilinear_d50_tau095` | — |
| single-Q | `cube_single_state_gciql_flow_bilinear_d50_singleq` | — |
| Gaussian actor | `cube_single_state_gciql_flow_bilinear_d50_ddpgbc` | — |
| G4 z-cond | `cube_single_state_gciql_flow_g4` | (d=512 value, `actor_cond=z`) |
| CRL Gaussian | `cube_single_state_crl` | (set `save_interval=1000000` so a ckpt is actually written!) |

> NB the CRL-Gaussian row had **no checkpoint** because its `save_interval` exceeded
> `train_steps`. Set `save_interval≤train_steps` on every launch or you get no ckpt.

### 2b. FB/PyTorch (`.venv`, `train.py`)

Resume landed (`resume.py`): `resume_from=<final.pt>` + `num_train_steps=1000000` + a
**stable** `wandb_run_name`/output dir.

**Resume seed s 500k→1M:**
```bash
CUDA_VISIBLE_DEVICES=<g> .venv/bin/python train.py \
  domain=cube_single agent=fb <ABLATION_FLAGS> seed=<s> \
  num_train_steps=1000000 resume=true \
  resume_from=/dev/shm/factored-fb_outputs/<RUN>_s<s>/checkpoints/final.pt
```
**Fresh seed (3..9) @1M:** same without `resume*`.

FB ablation flags:
| Row | `<ABLATION_FLAGS>` |
|---|---|
| zHER | `agent.train_goal_mode=traj` |
| FB-IQL critic | `agent.train_goal_mode=traj agent.critic_mode=iql` |
| FB-IQL q-loss | `agent.train_goal_mode=traj agent.q_loss_coef=1.0` (measure kept) |

---

## 3. GPU packing & orchestration (8× 23 GB, idle)

- JAX state runs are small; with `XLA_PYTHON_CLIENT_PREALLOCATE=false` pack **~4/GPU** → ~32 slots.
- FB torch ~3/GPU → ~24 slots. Run mixed; budget **~24–28 parallel slots**.
- Launch every run in **detached tmux** (per the always-run-in-tmux rule), one session per
  wave or a queue script (`scripts/run_grid_sweep.sh` already exists as a launcher).
- Suggested layout: GPUs 0–3 = GCIQL fresh seeds (queue), GPUs 4–5 = GCIQL resumes,
  GPUs 6–7 = FB torch (resumes + fresh). Rebalance as the JAX queue drains first.

## 4. Cost estimate

- GCIQL: ~83 1M-units × ~2.5 h = ~208 slot-hours.
- FB: ~25.5 1M-units × ~5.5 h (calibrated: ~50 steps/s) = ~140 slot-hours.
- Total ≈ **350 slot-hours** / ~26 parallel slots ≈ **~13 h wallclock**, realistically
  **half a day to a day** with eval overhead + imperfect packing.

## 5. Pre-flight & risks

1. **/dev/shm is volatile.** All resume checkpoints live there — a reboot wipes them and
   then nothing resumes. **Back up** `*/checkpoints/final.pt` (FB) and `params_500000.pkl`
   (GCIQL) to persistent disk *before* launching. (User will handle.)
2. **save_interval ≤ train_steps** on every launch (the CRL-Gaussian no-ckpt lesson).
3. **wandb entity = amsks**, groups per existing names (`GCIQL-State-Flow-Bilinear-*`,
   `fb_state_*`); fresh seeds must land in the SAME group as their 3 originals.
4. **Resume step bookkeeping:** GCIQL `GCIQL_STEP_OFFSET`/`GCIQL_CSV_APPEND` continue the
   csv; FB resume writes to the stable dir. Verify the first resumed eval logs at step
   >500000, not restart-from-0.
5. **Degradation:** expect FB-side rows to move (often down) at 1M vs 500k — that's
   the measurement change, surface it in the table as @1M.

## 6. After training

1. Re-extract @1M overall_success per seed (same scripts used for verification:
   `eval.csv` for GCIQL, `eval/reward/eval/success` for FB) across all 10 seeds.
2. Recompute mean ± **95% CI** for the whole grid (locked convention). Compute as
   Student-$t$: `ci = t(0.975, n-1) * stdev/sqrt(n)` from the *unrounded* per-seed
   success (NOT the rounded display integers). At the final n=10, t(9)=2.262 ≈ the
   main table's 1.96×SE, so the grid and headline table are consistent. (The interim
   @500k grid already uses this convention; n=3 rows have very wide CIs by design.)
3. Update the grid in `PAPER/fb-gciql-bridge/fb-gciql-bridge.tex`; relabel "@500k" → "@1M";
   re-examine prose claims that depend on specific values (43→86 jumps, etc.).
4. Rebuild PDF (`/tmp/tectonic -X compile`).
