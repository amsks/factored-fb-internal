#!/bin/bash
# Parquet-only pulls (no video). DROID full = 95k eps; cap shards for a first pass.
cd "$(dirname "$0")/.."
PY=.venv-dl/bin/python
$PY scripts/fetch_lerobot_states.py lerobot/berkeley_autolab_ur5 data/ur5.npz > logs/fetch_ur5.log 2>&1
$PY scripts/fetch_lerobot_states.py IPEC-COMMUNITY/libero_90_no_noops_lerobot data/libero90.npz > logs/fetch_libero90.log 2>&1
$PY scripts/fetch_lerobot_states.py IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot data/libero10.npz > logs/fetch_libero10.log 2>&1
$PY scripts/fetch_lerobot_states.py lerobot/droid_100 data/droid100.npz > logs/fetch_droid100.log 2>&1
$PY scripts/fetch_lerobot_states.py IPEC-COMMUNITY/bridge_orig_lerobot data/bridge.npz > logs/fetch_bridge.log 2>&1
echo ALL_DONE >> logs/fetch_bridge.log
