"""Download ONLY the state/action parquet shards of a LeRobot-format HF dataset (no videos)
and convert them to a single npz {observations, actions, terminals, episode_index}.

Usage: .venv-dl/bin/python scripts/fetch_lerobot_states.py REPO_ID OUT.npz [--max_files N] [--state_key K]
"""
import argparse, glob, json, os, sys
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument("repo"); ap.add_argument("out")
ap.add_argument("--max_files", type=int, default=0, help="cap on parquet shards (0 = all)")
ap.add_argument("--patterns", default="", help="comma list of allow_patterns (default meta/*,data/**)")
ap.add_argument("--state_key", default="observation.state")
ap.add_argument("--cache", default=os.path.join(os.path.dirname(__file__), "..", "data", "hf"))
a = ap.parse_args()

local = snapshot_download(a.repo, repo_type="dataset", allow_patterns=(a.patterns.split(",") if a.patterns else ["meta/*", "data/**"]),
                          cache_dir=a.cache, max_workers=16)
files = sorted(glob.glob(os.path.join(local, "data", "**", "*.parquet"), recursive=True))
if a.max_files: files = files[: a.max_files]
print(f"{a.repo}: {len(files)} parquet shards", flush=True)
obs, act, epi = [], [], []
for i, f in enumerate(files):
    t = pq.read_table(f, columns=[a.state_key, "action", "episode_index", "frame_index"])
    obs.append(np.stack(t.column(a.state_key).to_numpy(zero_copy_only=False)).astype(np.float32))
    act.append(np.stack(t.column("action").to_numpy(zero_copy_only=False)).astype(np.float32))
    epi.append(t.column("episode_index").to_numpy().astype(np.int64))
    if i % 500 == 0: print(f"  {i}/{len(files)}", flush=True)
obs = np.concatenate(obs); act = np.concatenate(act); epi = np.concatenate(epi)
# terminals: last frame of each episode (episodes contiguous within shard order)
term = np.zeros(len(epi), bool); term[:-1] = epi[1:] != epi[:-1]; term[-1] = True
# reorder so each episode is contiguous (stable sort by episode)
order = np.argsort(epi, kind="stable"); obs, act, epi = obs[order], act[order], epi[order]
term = np.zeros(len(epi), bool); term[:-1] = epi[1:] != epi[:-1]; term[-1] = True
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
np.savez_compressed(a.out, observations=obs, actions=act, terminals=term, episode_index=epi)
print(f"wrote {a.out}: obs {obs.shape}, episodes {term.sum()}, mean len {len(epi)/term.sum():.1f}")
