---
name: hiql-gcivl-bilinear
description: "DONE (2026-06-29): added value_arch=bilinear to HIQL + GCIVL (OGBench jax); both smoke-pass. HIQL uses a custom rep-feeding module (option A); state path only."
metadata: 
  node_type: memory
  type: project
  originSessionId: 29d24434-2cf7-47b2-89cc-f7b4407c6f44
---

**Added `value_arch: mlp|bilinear` to OGBench HIQL + GCIVL** (jax, `third_party/ogbench/impls/agents/`),
backward-compatible (default `mlp` unchanged). Goal: run HIQL/GCIVL as bilinear-value rows alongside
the FB↔GCIQL bridge. Both **jax-smoke-pass** (mlp + bilinear construct & train, finite losses;
`/dev/shm/.venv-jax`, JAX_PLATFORMS=cpu). Relates [[fb-to-gciql-bridge-knobs]], [[gciql-flow-agent]].

- **GCIVL** (`gcivl.py`): clean `GCValue -> GCBilinearValue` swap (flat, no subgoal rep), so
  V=⟨φ(s),ψ(g)⟩/√d directly — same as the gciql_flow precedent. + config `value_arch`, `value_latent_dim`.
- **HIQL** (`hiql.py`): NOT a drop-in — `GCBilinearValue.psi` only sees raw goals, but HIQL's goal
  embedding is the **state-dependent** subgoal rep φ([s;g]). Added a custom module `_BilinearRepValue`
  (option A): V(s,g)=⟨φ(s), ψ(goal_rep([s;g]))⟩/√d, sharing the existing `goal_rep` net so it's still
  trained end-to-end; keeps the rep the high-actor predicts. value+target_value both use it.

**Design confirmed:** HIQL has ONE shared value V(s,g); high vs low differ only by the GOAL fed
(low=subgoals, high=final goals) + the two actors — so "decompose high vs low independently" is N/A
without splitting the critic (not done). One `value_arch` flag bilinearizes both levels at once.

**CAVEATS / TODO:**
- **State path only.** `_BilinearRepValue` does φ on raw obs (no visual encoder); pixel HIQL would skip
  the encoder. GCIVL bilinear passes `state_encoder=encoders.get('value')` (None for state) — pixel untested.
- **No unit tests** (jax side is subprocess-harnessed) — verified by smoke only. Run = needs a cube_single
  state config for hiql/gcivl + `value_arch=bilinear value_latent_dim=50` overrides (mirror the gciql d50 configs).
- Not yet launched. The FB-side ladder (`scripts/launch_bridge_knobs.sh`, 15 runs) is separate and ready.

**CONFIGS CREATED + VALIDATED (2026-06-29):**
- `configs/gciql/cube_single_state_hiql_bilinear_d50.yaml` (agent_file agents/hiql.py, value_arch=bilinear,
  value_latent_dim=50, eval_on_cpu=0 since AWR actors). **Validated end-to-end** via real dispatcher
  (`run_gciql.py --config-name cube_single_state_hiql_bilinear_d50 train_steps=1500`): agent + HGC dataset +
  bilinear value all built, ran step-0 eval rollouts, **0 errors**. (Killed after construction confirmed.)
- `configs/gciql/cube_single_state_gcivl_bilinear_d50.yaml` (agents/gcivl.py, bilinear d50). GCIVL bilinear
  already jax-unit-smoked.
- mlp-baseline configs NOT created (user declined — use existing / not needed).
- **LAUNCHED 2026-06-29 (tmux `hiqlbil`, `scripts/launch_hiql_gcivl_bilinear.sh` DRY_RUN=0 STEPS=500000):**
  6 jax runs (hiql_bilinear_d50 x3 + gcivl_bilinear_d50 x3 @500k), GPUs 2-7, XLA_PREALLOCATE=false,
  logs `/dev/shm/hiql_gcivl_bil_logs/`. Training confirmed (GPUs 100%, 0 tracebacks). run_groups
  HIQL-State-Bilinear-d50 / GCIVL-State-Bilinear-d50.

**RESULTS @500k (3 seeds, NATIVE AWR actors — NOT flow):** GCIVL-bilinear **67.5** (80/60/63,
near stock GCIQL+flow 71) · HIQL-bilinear **~16** (—/17/15; hierarchy doesn't help short-horizon cube,
untuned). Caveat: AWR actors, not the flow-actor ladder; no MLP baseline so HIQL low may be HIQL-on-cube.

**NEW: `gcivl_flow` agent (2026-06-29)** — GCIVL IVL value + **FlowBC actor**. GCIVL is V-only (no Q) so
the flow actor can't do FQL Q-max; instead **advantage-weighted flow-matching BC** (weight exp(alpha*(V(s',g)
-V(s,g)))). Shim `tools/wandb_mode_shim/gcivl_flow.py` (subclasses GCIVLAgent, custom actor_loss + create
with no critic), registered in sitecustomize. jax-smoke-pass (mlp+bilinear). Config
`cube_single_state_gcivl_flow_bilinear_d50.yaml` (eval_on_cpu=1, flow actor). **LAUNCHED 3 seeds @500k**
tmux `gcivlflow` GPUs 3-5, logs `/dev/shm/gcivl_flow_logs/`, run_group GCIVL-State-Flow-Bilinear-d50.
Compare vs GCIVL-bilinear-AWR 67.5 — does flow improve it?

⚠️ **gcivl_flow v1 was BUGGY (collapsed 0%)** — `GCVectorField` velocity field is goal-AGNOSTIC
(takes obs,action,t only); FQL gets goal-conditioning from Q-max which GCIVL lacks; v1 trained only
the goal-agnostic velocity field AND sampled from an untrained GCNoiseActor → 0%. **FIXED v2:**
goal-conditioned velocity field (concat [obs;goal] into its obs input), sample via Euler flow rollout
(no GCNoiseActor), AWR-weighted flow matching. jax-smoke confirms goal-conditioning (diff goals→diff
actions) + unbatched eval-path shape. RELAUNCHED 2026-06-29 tmux `gcivlflow2` GPUs 3-5, 3 seeds @500k.
NOTE earlier "2/3 done" gcivl_flow numbers were the 0% BUG, not real; cleared those output dirs.

**Identity-B ladder runs (2026-06-29, tmux `idbdnorm`, GPUs 0/1/2/6/7, FB train.py):** r2_dnorm config
(IQL goal_match, decoupled GCIQL mix, HER, d_norm, lr3e-4, flow) + fixed_b=identity (z=g), 3 seeds @500k:
- `fb_bridge_idB_dnorm` (identity B, F task-conditioned) — Q=<F(s,a,g),g>/sqrt(d)
- `fb_bridge_idB_decond_dnorm` (identity B + decondition_F) — Q=<phi(s,a),g>/sqrt(d) = bilinear-GCIQL w/ psi=identity
Tests if learned psi is necessary (vs identity readout). Compare vs r2 learned-B 65, bilinear-GCIQL 90.
Logs /dev/shm/idB_dnorm_logs/. At step 46k, 0 tracebacks.
- Launch each: `.venv/bin/python run_gciql.py --config-name <cfg> seed=<s> train_steps=500000` (mirror the
  bridge campaign; eval_on_cpu=0 → GPU eval, fast for AWR/gaussian actors). NB HIQL eval rollout ~12s/ep.
