---
name: jax-gpu-packing-preallocate
description: JAX runs preallocate ~75% of a GPU; to pack >1 jax run/GPU set XLA_PYTHON_CLIENT_PREALLOCATE=false
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 86447647-071e-49fb-ae40-21a8ab6b35cd
---

JAX (the `run_gciql.py`/OGBench training path) **preallocates ~75% of the GPU per process by default**, so only ONE jax run fits per GPU unless you change it. The repo has NO mem-fraction wiring, so the launchers (`scripts/agents/*/{state,pixel}.sh` via `_run_jax_sweep`) assume 1 run/GPU.

**Why:** On 2026-06-09 I packed 3 gciql_flow runs/GPU (10 state + 5 pixel on 5 free GPUs); 14 seed logs hit RESOURCE_EXHAUSTED and GPUs showed 22GB used at 0% util (first process grabbed the whole card, the rest OOM-died).

**How to apply:** To run multiple jax seeds per GPU, export `XLA_PYTHON_CLIENT_PREALLOCATE=false` in the tmux launch command BEFORE the launcher — it propagates through `run_gciql.build_child_env` (which copies the full parent env) to the child `main.py`. After the fix, 3 cube runs/GPU sat at ~3-5GB each, 0 OOM. Alternative: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.3` (deterministic per-process cap) but preallocate=false is more robust for heterogeneous (state+pixel) packing. `run_gciql` already pins `MUJOCO_EGL_DEVICE_ID` per-GPU so concurrent eval renders don't collide. Always verify after launch: `nvidia-smi` memory should be SHARED/low, not 22GB/GPU. See [[gciql-flow-agent]], [[always-run-in-tmux]].