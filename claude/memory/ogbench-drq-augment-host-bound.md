---
name: ogbench-drq-augment-host-bound
description: Pixel RL throughput gotcha — data-pipeline numpy<->jax round-trips starve the GPU
metadata: 
  node_type: memory
  type: reference
  originSessionId: 538b0dc6-3753-482b-a260-1fb03bbdd2b0
---

Diagnostic lesson from the CRL+FlowBC pixel slowdown (1.24 it/s, ~15x slower than
GCIQL pixel). Symptom: GPU at ~13% util in bursts (`nvidia-smi dmon -s u`) — the
loop is **host-bound**, not compute-bound.

Root cause: the DrQ augment (`tools/wandb_mode_shim/ogbench_drq.py`) did, per image
key, `np.asarray(jitted_shift(jnp.asarray(arr)))` — a blocking device->host sync
every train step (×4 image keys for CRL: obs, next_obs, value_goals, actor_goals)
= ~800ms/step vs a 63ms device update. Fix: pure-NumPy vectorized shift on CPU
(~28ms), which overlaps the jitted update → ~10 it/s.

General rule for this repo's pixel pipeline: keep per-step data-pipeline work off
the device (pure numpy) OR fully inside the jitted step; never round-trip
numpy<->jax per batch. To split host vs device cost, time `dataset.sample()` vs
`agent.update()` separately with `jax.block_until_ready`. See [[crl-flowbc-campaign]].
