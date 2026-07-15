#!/usr/bin/env python
"""scripts/value/cf_campaign.py — match-by-reproduction provenance + K-continuation
relabeling for the outcome-separation (aliasing) experiment. TORCH methods (fb|rldp|psm).

Pipeline for method M:
  1. Fingerprint each stored seed record cf_records_M_msNN.parquet by its per-task
     grasp value vector (the 200 grasps are fixed: sample_grasps seed=0).
  2. Run every candidate checkpoint in --values-only mode and correlate its per-task
     value vectors to every stored seed; greedily assign seed<->checkpoint by max mean
     per-task correlation (state-dominated, so the true checkpoint correlates ~1 while
     HP-variant sweeps correlate less). Warn below --corr-thr.
  3. Emit, per matched (seedNN -> ckpt), a shell line that runs the probe with
     --k-rollouts K (-> cf_records_M_kNN.parquet with p_succ) && cf_repfeat_torch
     (-> cf_repfeat_M_sNN.parquet). A launcher runs the lines with xargs -P for
     box-wide parallelism.

Writes cf_provenance_M.json (mapping + corr) and cf_campaign_M.sh (command lines).
Run under .venv on the box (Linux: MUJOCO_GL=egl).
"""
from __future__ import annotations
import argparse, glob, json, os, re, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
REPSEP = REPO / "analysis/value/repsep"
PY = sys.executable


def latest_ckpt(d):
    ck = Path(d) / "checkpoints"
    if (ck / "final.pt").exists():
        return ck / "final.pt"
    steps = sorted(ck.glob("step_*.pt"),
                   key=lambda p: int(re.search(r"step_(\d+)", p.name).group(1)))
    return steps[-1] if steps else None


def stored_fps(m):
    fps = {}
    for f in sorted(REPSEP.glob(f"cf_records_{m}_ms*.parquet")):
        nn = int(re.search(r"_ms(\d+)", f.name).group(1))
        d = pd.read_parquet(f)
        fps[nn] = {int(t): d[d.task == t]["value"].to_numpy(float)
                   for t in sorted(d.task.unique())}
    return fps


def cand_fp(m, cand, data_path, mujoco_gl):
    cfg = Path(cand) / ".hydra" / "config.yaml"
    ck = latest_ckpt(cand)
    if not cfg.exists() or ck is None:
        return None
    out = Path(tempfile.mkdtemp()) / "fp.json"
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    r = subprocess.run(
        [PY, "scripts/value/counterfactual_value_probe.py", "--config", str(cfg),
         "--checkpoint", str(ck), "--method", m, "--data-path", data_path,
         "--mujoco-gl", mujoco_gl, "--values-only", str(out)],
        cwd=str(REPO), env=env, capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        sys.stderr.write(f"  [fp FAIL] {Path(cand).name}: {r.stderr[-300:]}\n")
        return None
    fp = {int(t): np.asarray(v, float) for t, v in json.loads(out.read_text()).items()}
    return (cand, cfg, ck, fp)


def _corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def match(stored, cands):
    seeds = sorted(stored)
    C = np.full((len(seeds), len(cands)), -1.0)
    for i, nn in enumerate(seeds):
        for j, (_, _, _, fp) in enumerate(cands):
            cs = [_corr(stored[nn][t], fp[t]) for t in stored[nn] if t in fp]
            if cs:
                C[i, j] = float(np.mean(cs))
    assign = {}
    used = set()
    for i in sorted(range(len(seeds)), key=lambda i: -C[i].max()):   # confident seeds first
        for j in np.argsort(-C[i]):
            if j not in used:
                assign[seeds[i]] = (int(j), float(C[i, j])); used.add(int(j)); break
    return assign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["fb", "rldp", "psm"])
    ap.add_argument("--candidates", required=True, help="glob for candidate seed dirs")
    ap.add_argument("--k-rollouts", type=int, default=10)
    ap.add_argument("--data-path", default="datasets")
    ap.add_argument("--mujoco-gl", default="egl")
    ap.add_argument("--corr-thr", type=float, default=0.85)
    ap.add_argument("--fp-workers", type=int, default=12)
    a = ap.parse_args()
    m = a.method

    stored = stored_fps(m)
    print(f"[campaign:{m}] {len(stored)} stored seed fingerprints", flush=True)
    cand_dirs = sorted(glob.glob(a.candidates))
    print(f"[campaign:{m}] fingerprinting {len(cand_dirs)} candidate dirs "
          f"({a.fp_workers} workers)", flush=True)
    with ThreadPoolExecutor(max_workers=a.fp_workers) as ex:
        cands = [c for c in ex.map(
            lambda d: cand_fp(m, d, a.data_path, a.mujoco_gl), cand_dirs) if c]
    print(f"[campaign:{m}] {len(cands)} candidates fingerprinted", flush=True)
    if not cands:
        raise SystemExit(f"no candidates fingerprinted for {m}")

    assign = match(stored, cands)
    mapping, lines, low = {}, [], 0
    for nn, (j, cc) in sorted(assign.items()):
        cand, cfg, ck, _ = cands[j]
        mapping[f"{nn:02d}"] = dict(dir=str(cand), ckpt=str(ck), corr=round(cc, 4))
        flag = "" if cc >= a.corr_thr else "  << LOW CORR"
        low += cc < a.corr_thr
        print(f"  seed {nn:02d} <- {Path(cand).name}  corr={cc:.3f}{flag}", flush=True)
        tag = f"{nn:02d}"
        probe = (f'MUJOCO_GL={a.mujoco_gl} OMP_NUM_THREADS=1 {PY} '
                 f'scripts/value/counterfactual_value_probe.py --config {cfg} '
                 f'--checkpoint {ck} --method {m} --data-path {a.data_path} '
                 f'--mujoco-gl {a.mujoco_gl} --k-rollouts {a.k_rollouts} --out-tag _k{tag}')
        feat = (f'MUJOCO_GL={a.mujoco_gl} OMP_NUM_THREADS=1 {PY} '
                f'scripts/probes/cf_repfeat_torch.py --config {cfg} --checkpoint {ck} '
                f'--method {m}_s{tag} --records {m}_k{tag} --data-path {a.data_path} '
                f'--mujoco-gl {a.mujoco_gl}')
        lines.append(f"{probe} && {feat}")
    (REPSEP / f"cf_provenance_{m}.json").write_text(json.dumps(mapping, indent=2))
    (REPSEP / f"cf_campaign_{m}.sh").write_text("\n".join(lines) + "\n")
    print(f"[campaign:{m}] wrote provenance ({len(mapping)} seeds, {low} low-corr) "
          f"+ cf_campaign_{m}.sh", flush=True)


if __name__ == "__main__":
    main()
