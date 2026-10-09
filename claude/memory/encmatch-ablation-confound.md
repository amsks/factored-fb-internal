---
name: encmatch-ablation-confound
description: "fb_state_encmatch evals 0% — not a bug; critic-only MLP encoder is a lossy bottleneck that degrades the actor's state input (design confound, not a capacity test)"
metadata: 
  node_type: memory
  type: project
  originSessionId: b298f176-c051-48e6-9758-67157a6aa211
---

`fb_state_encmatch` (param-matched state encoder ablation, see PAPER/specs/2026-06-07-encmatch-state-encoder-ablation.md) trained cleanly to 1M steps (no NaN, losses ≈ plain FB) but evals **0%** across all 5 cube tasks / 5 seeds, vs `fb_state_plain` ~0.5.

**Root cause (diagnosed 2026-06-08, reproducible):** NOT a code bug — the implementation does exactly what the spec designed. It is a **design confound**. The ablation inserts MLP `_fw_encoder`/`_bw_encoder` trained ONLY by the FB critic loss (`forward_optimizer` owns `_fw_encoder`; actor consumes it **detached**, `agents/fb/agent.py:233,356`). For *state* FB the raw 28-d obs is already a perfect representation, so a critic-only encoder can only LOSE info. Plain FB's actor sees the raw state (Identity encoder, linear-probe R²=1.0 for every state var); encmatch's actor sees a lossy 256-d critic-shaped feature: linear-probe R² = **arm joints 0.35**, effector ~0.71, cube xyz ~0.80. The actor can't read its own arm config → 0% manipulation.

**Why:** mirroring the *pixel* DrQ design (critic-only encoder — works for pixels because conv features preserve spatial info) is ill-posed for state, where raw obs is already optimal. So encmatch's 0% does NOT support "capacity doesn't help state FB" — it conflates matched-param-count with degrading the actor's input. Uninterpretable as a capacity test.

**Ruled out (all reproducible):** logging artifact (reproduced 0% from checkpoint), NaN/collapse, degenerate actor (diverse actions, z-sensitivity HIGHER than plain), goal-collapse in backward map (|cos|≈0.07), actor-optimization bug (actor beats random under its own Q). The learned Q is self-consistent but misaligned with reward because its input feature is lossy.

**How to apply:** Diagnostics live in `scripts/probe_encmatch.py` (feature/action/z-sensitivity/Q-rank/cube-recovery probes) and `scripts/reproduce_eval.py` (re-runs training-time eval on a checkpoint — pattern: `load_cfg`→`build_env_and_agent`→`load_checkpoint`, `OGBenchEvaluator`). NB dataset domain string is the resolved env id `cube-single-play-v0`, not `cube_single`; run analysis with `PYTHONPATH=. .venv/bin/python scripts/…`. Fix options if revived: skip-connect raw obs to actor (keeps param-match), or train encoder with actor loss too (diverges from pixel mirror). See [[fb-state-run-name-map]].
