"""Consolidate trained checkpoints on the Hugging Face Hub.

Uploads, for every selected run directory: config.json, eval*.csv, and the FINAL params
pickle only (the largest params_<step>.pkl), under the same relative path as on disk, plus a
manifest.json (run, critic, env, seed, final step, file sizes, sha of config). Nothing is
deleted locally. Re-running is incremental: files already on the hub with the same size are
skipped by the hub's dedup, and the manifest is rewritten.

    python scripts/hf_upload_checkpoints.py --dry            # print manifest + total size
    python scripts/hf_upload_checkpoints.py                  # upload
    python scripts/hf_upload_checkpoints.py --repo amsks/factored-fb-checkpoints --private
"""
import argparse, glob, hashlib, json, os, re, sys
from pathlib import Path

DEFAULT_GLOBS = [
    "exp/cube-*_flowbc_seed_*",                    # every cube campaign run (all critics)
    "exp/exorl-tdj/dmc-walker/*/*_seed_*",         # td_jepa-protocol walker arms
    "exp/exorl-official-r2/*/*_seed_*",            # official-RLDP replication (run 2)
    "exp/exorl-ortho/*/*_seed_*",                  # live-ortho RLDP arms
]
EXTRA_FILES = ["exp/paper_eval*.jsonl", "exp/random_floors.json",
               "analysis/cube_bottleneck/cube-single-play-v0/*.csv",
               "analysis/cube_bottleneck/cube-single-play-v0/RESULTS.md",
               "analysis/cube_bottleneck/cube-single-play-v0/figures/fig_lead_combined.pdf",
               "analysis/relabel_rescore*.jsonl"]


def final_params(run: Path):
    ps = sorted(run.glob("params_*.pkl"), key=lambda p: int(re.search(r"params_(\d+)", p.name).group(1)))
    return ps[-1] if ps else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="amsks/factored-fb-checkpoints")
    ap.add_argument("--repo-type", default="model")
    ap.add_argument("--private", action="store_true", default=True)
    ap.add_argument("--globs", nargs="+", default=DEFAULT_GLOBS)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--manifest", default="hf_manifest.json")
    args = ap.parse_args()

    runs = sorted(set(p for g in args.globs for p in glob.glob(g) if os.path.isdir(p)))
    files, manifest, total = [], [], 0
    for r in runs:
        run = Path(r); fp = final_params(run)
        if fp is None:
            continue
        cfg = run / "config.json"
        entry = dict(run=str(run), final_step=int(re.search(r"params_(\d+)", fp.name).group(1)),
                     files=[], size_bytes=0)
        if cfg.exists():
            meta = json.load(open(cfg)); entry.update(critic=meta.get("critic"), seed=meta.get("seed"),
                                                     domain=meta.get("domain"), actor=meta.get("actor"))
            entry["config_sha256"] = hashlib.sha256(cfg.read_bytes()).hexdigest()[:16]
        for f in [fp, cfg, *run.glob("eval*.csv"), *run.glob("eval_episodes.jsonl")]:
            if f.exists():
                files.append(str(f)); entry["files"].append(f.name); entry["size_bytes"] += f.stat().st_size
        total += entry["size_bytes"]; manifest.append(entry)
    extras = [f for g in EXTRA_FILES for f in glob.glob(g) if os.path.isfile(f)]
    files += extras; total += sum(os.path.getsize(f) for f in extras)
    json.dump(dict(repo=args.repo, n_runs=len(manifest), total_bytes=total, runs=manifest, extras=extras),
              open(args.manifest, "w"), indent=1)
    print(f"{len(manifest)} runs, {len(files)} files, {total/1e9:.2f} GB -> {args.repo}  (manifest: {args.manifest})")
    if args.dry:
        for e in manifest[:8]: print("  ", e["run"], e["final_step"], f"{e['size_bytes']/1e6:.0f} MB")
        print("   ...")
        return
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi()
    api.create_repo(args.repo, repo_type=args.repo_type, private=args.private, exist_ok=True)
    ops = [CommitOperationAdd(path_in_repo=f, path_or_fileobj=f) for f in files]
    ops.append(CommitOperationAdd(path_in_repo="manifest.json", path_or_fileobj=args.manifest))
    # chunk the commit so a transient failure does not lose everything
    CH = 40
    for i in range(0, len(ops), CH):
        api.create_commit(args.repo, repo_type=args.repo_type, operations=ops[i:i + CH],
                          commit_message=f"checkpoint consolidation part {i//CH + 1}/{(len(ops)+CH-1)//CH}")
        print(f"  committed {min(i+CH, len(ops))}/{len(ops)} files", flush=True)
    print("DONE_HF_UPLOAD")


if __name__ == "__main__":
    main()
