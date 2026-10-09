---
name: psm-rldp-probe-suite
description: Full paper probe suite (phase/factorization/counterfactual/aliasing) on tuned PSM+RLDP 1Mx10; PSM support added to analysis scripts (UNCOMMITTED)
metadata: 
  node_type: memory
  type: project
  originSessionId: b798f8b4-b3ec-4145-b887-98d31d022aab
---

Launched 2026-06-10 22:20 (tmux `probe_suite`) — folds the two NEW tuned 1M×10 state runs into the paper's cross-method analysis:
- `psm_state_tuned__ortho_coef1__lr_phi1e-5` (10 seeds, agent psm_flowbc)
- `rldp_state_tuned__ortho_coef0p001__lr_b1e-5` (10 seeds, agent rldp_flowbc)
Run dirs under `/dev/shm/factored-fb_outputs/<name>__sN/` (`.hydra/config.yaml` + `checkpoints/final.pt`). Driver: `scripts/run_psm_rldp_probe_suite.sh` (CPU-only via `CUDA_VISIBLE_DEVICES=` + `--device cpu`; `--mujoco-gl egl`; N_PAR=24 on the 192-core box). 70 phase-1 jobs + 20 aliasing value passes + 1 aggregate.

**Four probes & output layout:**
- `phase_probe.py` → `analysis/profiles/<m>_phase/sN/` (per_episode/summary parquet, funnel). Method-agnostic (rollouts only); PSM uses `_infer_z`→`PSMModel.reward_inference`.
- `factorization_probe.py` → `analysis/factorization/<m>/sN/` (readout_ceiling, verdict.json). NOTE min `--n-states ≥ ~1500` (min_per_group=80) or it crashes on empty table.
- `counterfactual_value_probe.py` → FIXED path `analysis/value/repsep/cf_{value,records}_<m><out-tag>.{json,parquet}` (no `--out`; seed via `--out-tag _<m>_sN`). Needs ≥5 successes/task or AUC=NaN (use --n-states 200+).
- aliasing chain: `training_value_profile.py` writes shared `training_states.npz` per seed dir then `<m>_values.parquet`; `aliasing_probe.py --root analysis/value/aliasing_psm_rldp --out .../aggregate`. npz bootstrapped via `--method fb` on the RLDP net (FBModel) then spurious `fb_values.parquet` deleted → PSM-vs-RLDP aliasing.

**UNCOMMITTED code edits adding PSM support (remote push blocked — local only, see [[host-identity-you-are-aws-gpu]]):**
`evals/representation_profile.py` (+psm_q_values/psm_v_values), `scripts/training_value_profile.py` (+_psm), `scripts/counterfactual_value_probe.py`, `scripts/factorization_probe.py`, `scripts/aliasing_probe.py` (psm in METHOD dicts). PSM Q math VERIFIED vs `agents/psm/flow_bc/agent.py`: `Q=mean_heads(sf_psi(_fw_encoder(_normalize(obs)),z,a)·z)` — matches the agent's actor Q but OMITS the `actor_pessimism_penalty·Q_unc` term (deliberate: reports representation value not policy-conservative value; note in writeup). Also installed scikit-learn into `.venv` (was missing; needed by counterfactual ROC-AUC).

Probe-suite mapping/smoke-test done by subagent (agentId aeee311cacd83fa0f).

**COMPLETED 2026-06-10 22:37** — all 81 jobs ok, 0 fails, ~17 min on 192 cores. Aggregated (more UNCOMMITTED fixes: factorization_probe_aggregate needs `-m`/PYTHONPATH; aggregate_cf_multiseed METHODS += psm; cf files had to be renamed `cf_*_<m>_s<N>`→`cf_*_<m>_ms<N>` because driver `--out-tag` double-encoded method vs the aggregator's `_ms*` glob; phase aggregator globs `s*_final/` so symlink `sN→sN_final`). **Results (PSM / RLDP):** funnel reach 100%/100%, secure 88%/85%, success 28.6%/26.2%, fail@transport 59%/59% (NOT grasp); counterfactual rank-AUC 0.516/0.489 (≈chance, vs GCIQL 0.89); factorization verdict **B2** R²(B→d) 0.91/0.69 (basis HAS transport info); aliasing ~0.61/0.59 (value can't separate controllable vs failing). => corroborates "readout not basis" ([[psm-readout-ablation]]). Cross-method synthesis written to `analysis/cross_method_comparison_psm_rldp.md`.

**Paper integration DONE 2026-06-11** (PSM promoted to a headline 5th method): regenerated `PAPER/rlbrew/figures/fig_funnel.png` + `fig_outcomes.png` via `scripts/plot_rlbrew_figures.py` (added PSM to METHODS/MC/OUTCOMES; **installed seaborn into `.venv`** — was missing). PSM IQM success = **30.7±5.7** (computed from `logs/grid_20260609_072020/psm_state_tuned__*` final eval, ranks FB>PSM>CRL>RLDP). results.tex cites PSM in 5 spots (IQM list, failure profile 45% maintain-fail, geometry R²(B→d)=0.94 best, cf-value AUC 0.52 flat, "sharpest readout failure" synthesis). "four→five" fixed in abstract/introduction/diagnostics/conclusion; PSM rows added to appendix per-task, terminal-outcome, and geometry(R²) tables. **2026-06-11: PAPER'S PSM SWITCHED to the ortho1000 config** (per [[psm-readout-ablation]] diagnostic, the new best). All PSM numbers updated: IQM success 36.7±4.5, maintain 35%, R²(B→d) 0.90, cf-AUC 0.49, post-contact-share range 67–87%, 5-way outcome (38.4/1.0/19.2/35.4/6.0); figures + all 3 appendix tables regenerated. NB ortho1000 grasps less reliably (secured 79.8 vs ortho1 87.9) so grasp-fail rose to 19% — softened "acquire control reliably"→"acquire control". Working doc `analysis/cross_method_comparison_psm_rldp.md` still has OLD ortho1 PSM numbers (not yet refreshed). NOT done: PSM row in the support-delta appendix table (needs a support-gap probe not in the suite) — minor, illustrative. No TeX toolchain on box (couldn't compile-check). All edits UNCOMMITTED.
