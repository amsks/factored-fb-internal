#!/usr/bin/env python3
"""hf_dataset/manifest_from_packed.py — build the HuggingFace export for datasets that are
ALREADY packed, i.e. written straight to <root>/data/<family>/<task>.npz by a collector.

pack_and_manifest.py exists for the older collectors, which write one npz per episode under
<collected>/<Family>__<task>/buffer/ and need stacking. The play regimes
(collect_mw_play_resample.py, collect_mw_standalone.py) emit the packed layout directly, so
there is nothing to stack -- only manifest.json is missing, and push_to_hub.py asserts on it.

    $SCRATCH/.venv-jax-fb/bin/python hf_dataset/manifest_from_packed.py \
        --root play-resample=$SCRATCH/fb-data/play-resample \
        --root play-resample-hard=$SCRATCH/fb-data/play-resample-hard \
        --out $SCRATCH/fb-data/hf_export_play

Each --root becomes a REGIME directory in the export (data/<regime>/<family>/<task>.npz), so
one repo can hold several regimes without task-name collisions. Statistics in the manifest are
recomputed from the arrays, not copied from the collector's meta sidecar.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

FIELDS = ["observation", "action", "next_observation", "achieved_goal",
          "reward", "terminal", "timeout", "success"]


def stat_task(npz: Path, meta: dict) -> dict:
    z = np.load(npz)
    missing = [f for f in FIELDS if f not in z.files]
    obs, succ = z["observation"], z["success"]
    n, t = obs.shape[0], obs.shape[1]
    ep_succ = float(succ.reshape(n, t).any(axis=1).mean())
    # Flag placeholder fields explicitly rather than letting a consumer discover them.
    degenerate = [f for f in ("achieved_goal", "terminal", "timeout")
                  if f in z.files and not np.any(z[f])]
    return {
        "num_episodes": int(n),
        "episode_length": int(t),
        "transitions": int(n * t),
        "obs_dim": int(obs.shape[-1]),
        "act_dim": int(z["action"].shape[-1]),
        "episode_success_rate": round(ep_succ, 4),
        "step_success_rate": round(float(succ.mean()), 4),
        "missing_fields": missing,
        "all_zero_fields": degenerate,
        "size_mb": round(npz.stat().st_size / 1e6, 2),
        "collector_meta": meta,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True, metavar="REGIME=DIR")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="factoredfb-testbed-play")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = []
    for spec in args.root:
        regime, _, d = spec.partition("=")
        src = Path(d) / "data"
        assert src.is_dir(), f"no data/ under {d}"
        for npz in sorted(src.glob("*/*.npz")):
            family = npz.parent.name
            meta_fp = npz.with_suffix("").with_suffix(".meta.json")
            meta = json.loads(meta_fp.read_text()) if meta_fp.exists() else {}
            dst_dir = out / "data" / regime / family
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(npz, dst_dir / npz.name)
            if meta_fp.exists():
                shutil.copy2(meta_fp, dst_dir / meta_fp.name)
            rec = {"regime": regime, "family": family, "task": npz.stem,
                   "file": str((dst_dir / npz.name).relative_to(out))}
            rec.update(stat_task(npz, meta))
            tasks.append(rec)
            print(f"[pack] {regime:22s} {family:10s} {npz.stem:16s} "
                  f"N={rec['num_episodes']} T={rec['episode_length']} "
                  f"ep-succ={rec['episode_success_rate']:.3f} ({rec['size_mb']} MB)")

    manifest = {
        "name": args.name,
        "format": "unified (per-task stacked npz [N,T,.]); fields: " + ", ".join(FIELDS),
        "loader": "env/unified.py::load_packed",
        "totals": {
            "regimes": len({t["regime"] for t in tasks}),
            "tasks": len(tasks),
            "episodes": sum(t["num_episodes"] for t in tasks),
            "transitions": sum(t["transitions"] for t in tasks),
        },
        "tasks": tasks,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[pack] wrote {out/'manifest.json'}: {manifest['totals']}")


if __name__ == "__main__":
    main()
