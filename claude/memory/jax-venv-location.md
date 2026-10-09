---
name: jax-venv-location
description: "JAX/GCIQL venv on the aws-gpu box is /dev/shm/.venv-jax (GPU, volatile); not the repo .venv-jax-cpu"
metadata: 
  node_type: memory
  type: project
  originSessionId: 050d377f-6bdf-4360-9c4b-c777b32e72e9
---

On the aws-gpu box the JAX/GCIQL venv is **`/dev/shm/.venv-jax`** — GPU JAX (jax 0.10.0, sees all 8 CUDA devices), NOT cpu-only. The repo's CLAUDE.md and most scripts say `.venv-jax-cpu` (repo-relative), which is a stale name from a CPU-only machine and **does not exist here**.

- Only `scripts/smoke_test_gciql.sh` defaults to the correct `/dev/shm/.venv-jax/bin/python`. The others (`run_pixel_profile.sh`, `run_gciql_phase_funnel.sh`, `run_phase_probe_pixel_v2.sh`, `smoke_gciql_pixel_profile.sh`) default `JAX_PY=.venv-jax-cpu/bin/python` → override with `JAX_PY=/dev/shm/.venv-jax/bin/python`.
- `run_gciql.py` is pure orchestration (must not import jax); it shells `third_party/ogbench/impls/main.py` in the jax venv.
- **Volatile:** it lives on tmpfs (`/dev/shm`), so it disappears on reboot and must be rebuilt. That's why it isn't a stable repo path.
- The torch/FB venv is the repo-root `.venv` (no jax). See [[factored-fb-test-env]].
