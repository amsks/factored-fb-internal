#!/usr/bin/env python3
"""hf_dataset/pack_and_manifest.py — pack the per-episode *unified* datasets under a
COLLECTED root into per-task stacked ``.npz`` files (efficient for HuggingFace
hosting: one LFS object per task instead of thousands of tiny episode files) and
emit ``manifest.json`` with global + per-task statistics.

    <fb>/.venv/bin/python hf_dataset/pack_and_manifest.py \
        --collected /raid/tongle/rl/fb-data/collected \
        --out       /raid/tongle/rl/fb-data/hf_export

Input  layout : <collected>/<Family>__<task>/buffer/episode_*.npz  (+ meta.json)
Output layout : <out>/data/<family>/<task>.npz   (stacked [N,T,·])  (+ meta.json)
                <out>/manifest.json               (index the loader / card read)

Each per-task ``.npz`` holds the 8 unified fields stacked over episodes:
    observation next_observation      -> [N, T, obs_dim]
    action achieved_goal              -> [N, T, act_dim] / [N, T, 3]
    reward terminal timeout success   -> [N, T]
Fixed-horizon per task (MW=100, MS=25, Box2D=200), so a dense stack is exact.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

FIELDS = ["observation", "action", "next_observation", "achieved_goal",
          "reward", "terminal", "timeout", "success"]


def _ep_index(p: Path) -> int:
    return int(re.findall(r"\d+", p.stem)[-1])


def pack_task(task_dir: Path, out_root: Path) -> dict | None:
    meta = json.loads((task_dir / "meta.json").read_text())
    # Collectors disagree on the key ("env_family" vs "family") and on case. The output path
    # data/<family>/<task>.npz is what env/domains/*.yaml resolves against, and those expect
    # the lowercase form (data/maniskill/..., data/metaworld/...), so normalise here.
    family = (meta.get("env_family") or meta.get("family") or "unknown").lower()
    task = meta.get("task", task_dir.name)
    eps = sorted((task_dir / "buffer").glob("episode_*.npz"), key=_ep_index)
    if not eps:
        return None

    # Not every collector writes all 8 fields, and some write extras. The upstream
    # play-resample sets, for instance, carry `goal_switch` (the target-relocation boundary)
    # but no `achieved_goal`. Pack whatever the episodes actually contain rather than
    # KeyError-ing: load_packed only reads observation/action/next_observation/reward, and
    # dropping a recorded field like goal_switch would throw away the one marker of segment
    # structure that survives collection.
    present = [k for k in FIELDS if k in np.load(eps[0]).files]
    extra = [k for k in np.load(eps[0]).files if k not in FIELDS]
    keys = present + extra
    lengths, cols = [], {k: [] for k in keys}
    for ep in eps:
        z = np.load(ep)
        lengths.append(int(len(z["action"])))
        for k in keys:
            cols[k].append(z[k])
    fixed = len(set(lengths)) == 1
    packed = {k: (np.stack(cols[k]) if fixed else np.array(cols[k], dtype=object))
              for k in keys}

    out_dir = out_root / "data" / family
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / f"{task}.npz"
    np.savez_compressed(out_fp, **packed)
    (out_dir / f"{task}.meta.json").write_text(json.dumps(meta, indent=2))

    succ = packed.get("success", np.zeros((len(eps), lengths[0]), bool))   # [N,T] bool
    ep_success = np.stack([s.any() for s in succ]) if fixed else \
        np.array([np.asarray(s).any() for s in succ])
    return {
        "family": family,
        "task": task,
        "num_episodes": len(eps),
        "episode_length": lengths[0] if fixed else None,
        "transitions": int(sum(lengths)),
        "obs_dim": int(meta.get("obs_dim", packed["observation"][0].shape[-1])),
        "act_dim": int(meta.get("act_dim", packed["action"][0].shape[-1])),
        "success_rate": round(float(ep_success.mean()), 4),
        "policy": meta.get("policy", "tdmpc2-expert"),
        "success_only": bool(meta.get("success_only", False)),
        "file": str(out_fp.relative_to(out_root)),
        "size_mb": round(out_fp.stat().st_size / 1e6, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collected", default="/raid/tongle/rl/fb-data/collected")
    ap.add_argument("--out", default="/raid/tongle/rl/fb-data/hf_export")
    args = ap.parse_args()

    collected, out = Path(args.collected), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dirs = [d for d in sorted(collected.iterdir())
            if (d / "buffer").is_dir() and (d / "meta.json").exists()]

    stats = []
    for d in dirs:
        s = pack_task(d, out)
        if s:
            stats.append(s)
            print(f"[pack] {s['family']:9s} {s['task']:26s} "
                  f"N={s['num_episodes']:4d} T={s['episode_length']} "
                  f"succ={s['success_rate']:.2f} -> {s['file']} ({s['size_mb']} MB)")

    by_family = {}
    for s in stats:
        f = by_family.setdefault(s["family"], {"tasks": 0, "episodes": 0, "transitions": 0})
        f["tasks"] += 1
        f["episodes"] += s["num_episodes"]
        f["transitions"] += s["transitions"]

    manifest = {
        "name": "factoredfb-testbed-demos",
        "format": "unified (per-task stacked npz); fields: " + ", ".join(FIELDS),
        "totals": {
            "tasks": len(stats),
            "episodes": sum(s["num_episodes"] for s in stats),
            "transitions": sum(s["transitions"] for s in stats),
        },
        "by_family": by_family,
        "tasks": stats,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[pack] wrote {out/'manifest.json'}")
    print(f"[pack] TOTAL: {manifest['totals']['tasks']} tasks, "
          f"{manifest['totals']['episodes']} episodes, "
          f"{manifest['totals']['transitions']} transitions")
    for fam, v in by_family.items():
        print(f"       {fam:9s} {v['tasks']} tasks  {v['episodes']} eps  {v['transitions']} trans")


if __name__ == "__main__":
    main()
