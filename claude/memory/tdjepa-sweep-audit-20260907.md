---
name: tdjepa-sweep-audit-20260907
description: 2026-09-07 audit of td_jepa OGBench sweep vs our FB/HILP/RLDP configs — privcoord FB row ran ortho 1.0 (never run by reference); fb_tdj run dirs contaminated by two lr_b arms; HILP best lr = phi 1e-4; td_jepa.tex baseline file was EMPTY
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a72e21f-f79c-41ef-ac95-9dc284d1779e
  modified: 2026-09-07T21:27:49.206Z
---

**td_jepa reference OGBench sweep** (`~/git/baselines/td_jepa/scripts/train/proprio/launch_*_ogbench.py`, 1M steps, 10 seeds, paper reports best-of-grid per domain, winning cell NOT published):
- FB: ortho_coef {100, 1000}, lr_b {1e-4, 1e-5}; bc_coeff 3.0 manip / 0.3 nav; ortho 1.0 (BASE_CFG) is always overridden ⇒ no reference number ever used 1.0.
- RLDP: ortho {0.01, 0.1} manip / {0.001, 0.01} antmaze, lr_b {1e-4, 1e-5}; td_jepa's RLDP regularizer is LIVE (the dead one is the official RLDP repo).
- HILP: lr_features (phi) {1e-4, 1e-5}, prob_random_goal {0.375, 0.5}; lr_sf/lr_actor fixed 1e-4. **Most likely HILP cell = phi 1e-4 / p_rand .375** — our hilp_tdjc at exactly that cell = .83 > published .742.
- No humanoidmaze launcher; manip sweep = cube-single, cube-double, scene, puzzle-3x3; nav = 5 antmazes.

**Our side:**
- FB ortho 1000 used by: torch "FB reference reimpl" .532, 10-seed 1M bridge campaign, jax fb_ref (.20 @500k), fb_tdj. FB ortho **1.0** used by: torch "Tuned FB" .42, the whole `table_privcoord.tex` FB row (.11/.34/.02/.22) and every `fb_oc` cell. Ortho 100 never run.
- ⚠ **fb_tdj cube-single dirs CONTAMINATED**: `scripts/launch_fb_tdj_parity.sh` ran lr_b 1e-4 and 1e-5 into the same `exp/cube-single-play-v0_fb_tdj_flowbc_seed_N`; clean lr_b=1e-4 cell from `/dev/shm/fbtdj_logs/` = .40/.38/.48 (.42), lr_b 1e-5 = .37; memory's ".38±.07" mixes windows. Must re-run with distinct save_dirs.
- privcoord HILP row = `hilp_cube.yaml` (phi lr 5e-4, batch 1024, ExoRL arch, goal-Dirac eval) — not comparable to reference.
- RLDP config already matches a swept cell (ortho .01, lr_b 1e-4); 1M relabel cube-single = .25.

**Paper files:** `Factored-FB-seam/PAPER/ICLR/baselines/td_jepa.tex` was 0 bytes on 2026-09-07; published numbers reconstructed from arXiv HTML parse (`scratchpad/tdjepa.html` of session b6e49391). Tables written: `td_jepa.tex` (full), `td_jepa_density.tex` (FB/HILP/RLDP + Density-FB), `reimplementation.tex`; colours defined in `colors.tex` (\paperc yellow = from paper, \ckptM green = our 1M, \ckptK blue = our 500k — internal code, never write a legend).

See [[hilp-tdjc-cube-not-inert-20260905]], [[density-fb-privileged-coords-20260902]], [[tdjepa-published-ogbench-table-20260907]], [[partb-campaign-handoff-20260906]].
