#!/usr/bin/env bash
# campaign/run_jax_after_native.sh — wait for the native queue to finish, then launch
# the JAX-baseline queue on the (now free) GPUs. GPUs are compute-saturated by native,
# so running JAX sequentially wastes nothing. Detached-friendly.
set -uo pipefail
trap '' HUP

FB=/raid/tongle/rl/Factored-FB
NATIVE_STATUS=/raid/tongle/rl/fb-data/campaign/logs_native/status.json
JAX_LOGDIR=/raid/tongle/rl/fb-data/campaign/logs_jax
mkdir -p "$JAX_LOGDIR"

echo "[run_jax_after_native] waiting for native queue to finish... $(date)"
while true; do
    fin=$(python3 -c "import json;print(json.load(open('$NATIVE_STATUS')).get('finished',False))" 2>/dev/null || echo False)
    [ "$fin" = "True" ] && break
    sleep 120
done

echo "[run_jax_after_native] native done -> launching JAX queue $(date)"
exec "$FB/.venv/bin/python" "$FB/campaign/run_queue.py" \
    --jobs "$FB/campaign/jobs_jax.txt" --gpus 0,1,2,3,4,5,6,7 --pack 4 \
    --logdir "$JAX_LOGDIR"
