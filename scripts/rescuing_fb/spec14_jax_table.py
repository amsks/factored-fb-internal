#!/usr/bin/env python3
"""Spec 14 §7 — compact measure-axis table from results/spec14_jax/zsr_*.json.

Prints env x checkpoint x family -> success plus z-inference diagnostics.
Pure stdlib; run with any python.
"""
import glob
import json
import os
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "results/spec14_jax"

ROWS = []
for p in sorted(glob.glob(os.path.join(OUT, "zsr_*.json"))):
    d = json.load(open(p))
    m = d["meta"]
    fam = d["families"]
    dirac = fam.get("dirac", {})
    region = fam.get("region", {})
    dvals = {k: v.get("success") for k, v in dirac.items()
             if k.startswith("t") and isinstance(v, dict)}
    npos = [v["diag"]["n_pos"] for k, v in dirac.items()
            if k.startswith("t") and isinstance(v, dict)]
    n_deg = sum(1 for v in dvals.values() if v is None)
    ROWS.append(dict(
        env=m["env"].replace("-play-v0", ""),
        ckpt=os.path.basename(m["run_dir"]),
        zinf=f"{m.get('z_inference', '?')}k{m.get('retrieval_k', '?')}",
        dirac=dirac.get("mean"),
        dirac_per_task=" ".join("--" if dvals[k] is None else f"{dvals[k]:.2f}"
                                for k in sorted(dvals)),
        dirac_npos="/".join(str(x) for x in npos),
        dirac_degen=n_deg,
        region_final=region.get("mean_final"),
        region_any=region.get("mean_any"),
        region_rej=region.get("centers_rejected"),
        file=os.path.basename(p),
    ))

if not ROWS:
    print(f"no zsr_*.json under {OUT}")
    sys.exit(0)

hdr = ("env", "ckpt", "zinf", "dirac", "per-task", "n_pos", "deg",
       "region_fin", "region_any", "rej")
rows = [(r["env"], r["ckpt"], r["zinf"],
         "--" if r["dirac"] is None else f"{r['dirac']:.3f}",
         r["dirac_per_task"], r["dirac_npos"], str(r["dirac_degen"]),
         "--" if r["region_final"] is None else f"{r['region_final']:.3f}",
         "--" if r["region_any"] is None else f"{r['region_any']:.3f}",
         str(r["region_rej"])) for r in ROWS]
w = [max(len(hdr[c]), max(len(x[c]) for x in rows)) for c in range(len(hdr))]
print("  ".join(h.ljust(wc) for h, wc in zip(hdr, w)))
print("-" * (sum(w) + 2 * len(w)))
for x in rows:
    print("  ".join(v.ljust(wc) for v, wc in zip(x, w)))
