---
name: tdjtab-campaign-prep-20260907
description: "2026-09-07 \"tdjtab\" campaign PREPARED — TD-JEPA-protocol OGBench proprio table (FB/HILP/RLDP reimpl + Density-FB hard-only) across 9 envs; 167-line manifest, launcher, non-cube relabel rewards + per-env hit-specs implemented and smoke-tested; all UNCOMMITTED on density-fb seam worktree"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a72e21f-f79c-41ef-ac95-9dc284d1779e
  modified: 2026-09-07T22:52:29.323Z
---

**Decision 2026-09-07: hard-target Density-FB only; soft arms out of scope.** (Audit: every density yaml already defaults hard; hilp_density/rldp_density are hard-only with no switch; the 12 resumable 500k ckpts are all hard.)

**Deliverables (Factored-FB-seam, branch density-fb, UNCOMMITTED):**
- `docs/notes/2026-09-07-tdjepa-table-needed-runs.md` — the campaign doc: status matrix, run list, tier-2 grid (243), blockers, smoke matrices.
- `scripts/tdjtab_manifest.sh` — **167 tier-1 runs** (fb_tdj 27, hilp_tdjc 21, rldp 24 [3 resumes], density_fb 27 [6 resumes], density_fb_z 23, hilp_density_tdjc 21, rldp_density 24 [3 resumes]); fresh runs `--save_dir exp/tdjtab`, antmaze `actor.bc_coeff=0.3`, density runs `measure_density_hard=true` (+`latent_centered` for z) + per-env hit spec. ~870 GPU-h on L4s.
- `scripts/launch_tdjtab.sh '<regex>' '<gpus>'` — round-robin in tmux `tdjtab`, logs `logs/tdjtab/<domain>/<critic>/s<seed>.log`, skips finished, `DRY=1`.
- `impls/configs/domain/{scene_play,puzzle_3x3,antmaze_{medium,large}_{navigate,stitch},antmaze_medium_explore}.yaml`.
- `env/ogbench.py` — relabel rewards for antmaze/scene/puzzle transcribed from td_jepa (physics row = qpos ++ button_states, NOT raw qpos); `tests/impls/test_ogbench_relabel_families.py` 20 tests. Positive rate @10k samples: cube ~2%, antmaze ~0.9%, scene ~0.3%, puzzle ~0.2%; scene/puzzle shifted reward is graded (#satisfied − k + 1), not 0/1.

**Per-env hit-specs (density critics; MUST be quoted `--override 'measure_hit_spec="..."'` or OmegaConf parses `0:2:0.5` as 120.5):** antmaze `0:2:0.5`; scene `19:22:0.4,28:30:0.5,32:34:0.5,36:37:0.72,38:39:0.6`; puzzle nine one-hot blocks `19:21:0.5,23:25:0.5,…,51:53:0.5` (= all buttons match). ⚠ hilp_density_tdjc / rldp_density only read hit_spec under `goal_kernel=true` → inert there.

**Gotchas:** `/dev/shm/.venv-jax` volatile; fb_tdj cube-single old dirs contaminated (never resume); `exp/partb-tier1/*density_fb_z*` are pre-fix latent reward; cluster 1M density_fb_z ckpts must show `measure_density_hard: true` + `latent_centered` before quoting. `rldp_tdj.yaml` is DMC-shaped — don't use for OGBench.

**Stop/rethink gates:** judge only at 500k/1M ckpts; if clean FB/HILP/RLDP cube reimpl land >10 pts under published, fix harness before scaling; cube-double z stuck near .2 = conjunction ceiling (not a tuning problem); Density-FB not beating fb_tdj on scene/antmaze at 500k ⇒ mechanism cube-specific.

See [[tdjepa-sweep-audit-20260907]], [[partb-campaign-handoff-20260906]], [[z-index-cube-double-conjunction-20260907]].
