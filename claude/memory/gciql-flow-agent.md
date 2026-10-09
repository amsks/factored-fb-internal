---
name: gciql-flow-agent
description: "gciql_flow JAX agent = GCIQL IQL critic + FQL flow actor (actor-only ablation); files, how it works, how to run"
metadata: 
  node_type: memory
  type: project
  originSessionId: 86447647-071e-49fb-ae40-21a8ab6b35cd
---

`gciql_flow` is a custom JAX/OGBench agent added 2026-06-09: **GCIQL with a flow actor** — the IQL expectile value + double critic are inherited UNCHANGED from vendored `GCIQLAgent`; only the Gaussian DDPG+BC actor is swapped for FQL's flow actor (flow-matching BC velocity field + one-step noise actor distilled from it + Q-max on the FROZEN IQL `min(q1,q2)`). It's the actor-only ablation of GCIQL, directly comparable to the GCIQL DDPG+BC runs. Differs from canonical FQL only in the critic (IQL vs FQL's plain TD).

Built by cloning the proven `crl_flowbc.py` pattern (CRL + same flow actor). Reuses `GCNoiseActor`/`GCVectorField` from `tools/wandb_mode_shim/ogbench_flow.py`. The frozen-critic Q-term is essential — see [[crl-actor-loss-critic-gradient]].

**Files:**
- `tools/wandb_mode_shim/gciql_flow.py` — `GCIQLFlowAgent(GCIQLAgent)` + `get_config` (adds flow_steps=10, bc_coeff=3.0 [key knob], actor_vf/actor dims).
- `tools/wandb_mode_shim/sitecustomize.py` — registers `agents.agents['gciql_flow']` (setdefault block, parallel to crl_flowbc).
- `configs/gciql/cube_single_state_gciql_flow.yaml` — state launcher; `agent_file: tools/wandb_mode_shim/gciql_flow.py`, `run_group: GCIQL-State-Flow` (per [[wandb-group-naming-convention]]).
- `tests/test_gciql_flow_injection.py` — 5 smoke tests (build/update/finite/registration/unbatched state+pixel), run via the jax-venv subprocess ([[jax-venv-location]]).

**Pixel launcher/config added + LAUNCHED 2026-06-09:** `configs/gciql/cube_single_visual_gciql_flow_drq.yaml` (run_group `GCIQL-Pixel-DrQ-Flow`), launchers `scripts/agents/gciql_flow/{state,pixel}.sh`. Running: **state 10 seeds @1M** (tmux `gciql_flow_state`, group `GCIQL-State-Flow`) + **pixel 5 seeds @500k** (tmux `gciql_flow_pixel`, group `GCIQL-Pixel-DrQ-Flow`), all on GPUs 0-4, logging to amsks ([[wandb-entity-amsks]]). bc_coeff=3 (constant across all flow methods). Required `XLA_PYTHON_CLIENT_PREALLOCATE=false` to pack 3 runs/GPU — see [[jax-gpu-packing-preallocate]].
