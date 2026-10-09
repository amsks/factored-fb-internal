---
name: spec29-ra-dual-index-20260814
description: "2026-08-14 spec 29 WRITTEN + user-approved — R-A dual-corrected index build (two arms kernel/latent, gates G29-0..3), Q4 supervised topline green-lit as fill-in, antmaze plain-FB cell resolved by decision (source from TD-JEPA, do not run), HF results backup launched"
metadata: 
  node_type: memory
  type: project
  originSessionId: 973481bf-5a6b-4ead-8080-c40f063e1e9d
  modified: 2026-08-14T21:35:35.196Z
---

Spec: `PAPER/Multistep-FB/specs/29-ra-dual-index.md`. Follows [[spec28-p0-z-canonicalization-20260814]].

**R-A build (user-approved):** new knob `measure_density_z_encode ∈ {sphere, dual}`; dual: ẑ = √d·norm((Σ̂_B+λI)⁻¹ m̂), m̂ from query-set reward samples (hindsight rows: kg = `_delta_phi`, matched exactly). Two arms on cube-single 500k s0: **R-A/kernel** (dual encode + the already-built spec-28 cell-ii knob `measure_density_z_actor_reward=kernel` — actor integrates ACTUAL reward samples; random-sphere rows masked via existing `z_hindsight` machinery) and **R-A/latent** (dual encode + B̃ᵀẑ readout, which under the dual index IS the ridge-reconstructed reward, P0b AUC .99–1.00 — tests the closed-form interface).

**Load-bearing implementation pins:** Σ̂_B = UNCENTERED second moment matching `spec28_encodings.py` (test-asserted), EMA-tracked as serialized param leaves (`density_b_zenc` pattern) — batch stats forbidden (train/eval mismatch broke the first z build); encode B̄ must be the slow-EMA tower (`enc_tau>0`) — ridge through a fast-drifting B is the non-stationarity risk; λ=.01·tr(Σ̂)/d default, knob {.001,.1}; ‖ẑ‖=√d re-imposed post-solve; z_rand_frac=.5 unchanged (behaviorally-grounded random tasks = default-off follow-up knob); NEW ARM ⇒ not comparable to archived z24/z25.

**Gates:** G29-0 pre-flight eval-only (dual ẑ(g) pairwise cos over 4096 goals must be ≪ .99 on existing ckpts — if dual is also collapsed, arm dead, don't launch); G29-1 mid-run kill at 250k if actor sensitivity stays 0.000 (never quote mid-run success); G29-2 cs screen vs g .93 / BC .111 → promote to cd; G29-3 = zero-shot reward families through reward-samples→ẑ→policy — the ONLY gate that licenses the general-reward-interface paper claim (binding user requirement).

**Other decisions (2026-08-14):** Q4 supervised topline GREEN-LIT (spec 27 Q4 contract, fill-in compute behind R-A). **Antmaze plain-FB ddpgbc cell: DO NOT RUN — value sourced from TD-JEPA reference results; tex `pending` cell becomes a cited number.** HF backup DONE: private dataset repo `amsks/factored-fb-results-backup`, prefix `backup-2026-08-14/results/`, all 492 files (84M) verified 492/492 on hub — no prior backup existed (user's recollection referred to other amsks dataset repos); restore via `.venv/bin/hf download amsks/factored-fb-results-backup --repo-type dataset --include "backup-2026-08-14/results/*"`. Note: `hf` CLI is not on PATH, use `.venv/bin/hf`.

**Doc debt:** DENSITY-FB-EXPLAINED §6.5 sits AFTER §6.6 and still says "z(g) works" + offers R-B — contradicts §6.6 (collapse, R-B not indicated); mark superseded or fold in.
