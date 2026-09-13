#!/usr/bin/env python
"""Campaign dashboard: which tdjtab run is where, and how far along.

Joins four sources into one table:
  scripts/tdjtab_manifest.sh   the full run-id list (the denominator)
  assignments/<site>.json      who OWNS each id            (hub)
  status/<run-dir>.json        what a site reports finished (hub)
  exp/<run-dir>/params_*.pkl   what is actually uploaded    (hub)
  logs/tdjtab/<id>.log         live step + eval state       (local, evalbox only)

    python scripts/tdjtab_status.py                 # summary + anything not idle
    python scripts/tdjtab_status.py --all           # every run id
    python scripts/tdjtab_status.py --site luis     # one site
    python scripts/tdjtab_status.py --no-hub        # local logs only (fast, offline)
"""
import argparse, json, os, re, subprocess, sys, datetime as dt
from collections import defaultdict

REPO = "amsks/factored-fb-checkpoints"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "scripts", "tdjtab_manifest.sh")
LOGROOT = os.path.join(ROOT, "logs", "tdjtab")
DROPPED = re.compile(r"^(antmaze_medium_stitch|antmaze_large_stitch|antmaze_medium_explore)/")
ID_RE = re.compile(r"^# ([a-z0-9_]+/[a-z0-9_]+/s[0-9]) :: (.*)$")
# The cube-single fb_tdj dirs once mixed two lr_b arms (1e-5 and 1e-4) appended into one dir.
# Do NOT hardcode them as contaminated: vista and evalbox re-ran all three into exp/tdjtab and
# their uploads replaced the hub dirs, so the same paths now hold the CLEAN arm. Decide from the
# evidence instead -- a cell is contaminated iff its hub config.json records the wrong lr_b.
SUSPECT = {f"cube-single-play-v0_fb_tdj_flowbc_seed_{n}": ("lr_b", 1e-4) for n in (0, 1, 2)}

ENV_OF = {"cube_single": "cube-single-play-v0", "cube_double": "cube-double-play-v0",
          "scene_play": "scene-play-v0", "puzzle_3x3": "puzzle-3x3-play-v0",
          "antmaze_medium_navigate": "antmaze-medium-navigate-v0",
          "antmaze_large_navigate": "antmaze-large-navigate-v0"}


def run_dir_name(rid):
    """<env>_<critic>_flowbc_seed_<n> — how the id appears in status/ and exp/."""
    dom, critic, seed = rid.split("/")
    return f"{ENV_OF[dom]}_{critic}_flowbc_seed_{seed[1:]}"


def manifest_ids():
    ids = []
    for line in open(MANIFEST):
        m = ID_RE.match(line.rstrip("\n"))
        if m and not DROPPED.match(m.group(1)):
            ids.append(m.group(1))
    return ids


def local_state(rid):
    """(step, note) from this box's log, or (None, None) if we are not running it."""
    log = os.path.join(LOGROOT, rid + ".log")
    if not os.path.exists(log):
        return None, None, False
    try:
        tail = subprocess.run(["tail", "-c", "200000", log], capture_output=True, text=True,
                              timeout=20).stdout
    except Exception:
        return None, None, True
    steps = re.findall(r"^(\d+) \{", tail, re.M)
    step = int(steps[-1]) if steps else None
    started = True                       # the log exists, so this box launched it
    note = ""
    if re.search(r"Traceback|Error|FAILED|Killed|OOM", tail):
        note = "ERROR"
    elif re.search(r"\d+/20 \[", tail.split("\n")[-1] if tail else ""):
        note = "in eval"
    elif "[signal] caught" in tail:
        note = "signalled"
    age = (dt.datetime.now() - dt.datetime.fromtimestamp(os.path.getmtime(log))).total_seconds()
    if age > 1800 and note != "ERROR":
        note = (note + " " if note else "") + f"stale {age/3600:.1f}h"
    return step, note, started


def hub_state():
    """(owner_by_id, finished_dirs, uploaded_steps_by_dir) from the hub."""
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    files = api.list_repo_files(REPO, repo_type="model")
    owner, finished, uploaded = {}, {}, defaultdict(list)
    for f in files:
        if f.startswith("exp/") and f.endswith(".pkl"):
            parts = f.split("/")
            m = re.match(r"params_(\d+)\.pkl", parts[-1])
            if m and len(parts) >= 3:
                uploaded[parts[1]].append(int(m.group(1)))
    for f in files:
        if f.startswith("assignments/") and f.endswith(".json"):
            site = f.split("/")[-1][:-5]
            try:
                d = json.load(open(hf_hub_download(REPO, f, repo_type="model")))
            except Exception:
                continue
            for rid in d.get("run_ids", []):
                owner[rid] = site
        elif f.startswith("status/") and f.endswith(".json"):
            try:
                finished[f.split("/")[-1][:-5]] = json.load(
                    open(hf_hub_download(REPO, f, repo_type="model")))
            except Exception:
                pass
    return owner, finished, uploaded


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="list every run id, not just active ones")
    ap.add_argument("--site", help="only this site")
    ap.add_argument("--no-hub", action="store_true", help="skip the hub (local logs only)")
    a = ap.parse_args()

    ids = manifest_ids()
    owner, finished, uploaded = ({}, {}, {}) if a.no_hub else hub_state()

    # Resolve the suspect dirs from their uploaded config, not from a hardcoded list.
    contaminated = set()
    if not a.no_hub:
        from huggingface_hub import hf_hub_download
        for d, (key, want) in SUSPECT.items():
            if not uploaded.get(d):
                continue
            try:
                cfg = json.load(open(hf_hub_download(REPO, f"exp/{d}/config.json",
                                                     repo_type="model")))["config"]
                if cfg.get(key) != want:
                    contaminated.add(d)
            except Exception:
                contaminated.add(d)          # cannot verify -> do not count it as done

    rows, tally = [], defaultdict(lambda: defaultdict(int))
    for rid in ids:
        d = run_dir_name(rid)
        site = owner.get(rid, "unassigned")
        step, note, started = local_state(rid)
        up = sorted(uploaded.get(d, []))
        if d in contaminated:
            if step is not None:
                state, detail = "RUNNING", f"step {step:>7,}"
            elif started:
                state, detail = "RUNNING", "pre-step-1 eval"
            else:
                state, detail = "REDO", "hub config records the WRONG lr_b - must re-run"
        elif d in finished or 1_000_000 in up:
            state, detail = "DONE", (f"1M on hub" if 1_000_000 in up else "status/ only")
            fin = finished.get(d, {})
            if fin.get("route_a") is not None:
                detail += f"  A={fin['route_a']} B={fin.get('route_b')}"
        elif note == "ERROR":
            state, detail = "ERROR", f"step {step}" if step else "see log"
        elif step is not None:
            state, detail = "RUNNING", f"step {step:>7,}" + (f"  {note}" if note else "")
        elif started:
            # launched here but no training row yet -- almost always the unconditional
            # step-1 eval block, which runs before the loop prints anything.
            state, detail = "RUNNING", "pre-step-1 eval" + (f"  {note}" if note else "")
        elif up:
            state, detail = "PARTIAL", f"ckpt {max(up):,} on hub"
        else:
            state, detail = "ASSIGNED" if site != "unassigned" else "UNASSIGNED", ""
        tally[site][state] += 1
        rows.append((site, rid, state, detail))

    print(f"TDJTAB CAMPAIGN — {len(ids)} runs — {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M UTC}"
          + ("   [local only]" if a.no_hub else ""))
    print()
    order = ["DONE", "RUNNING", "PARTIAL", "REDO", "ERROR", "ASSIGNED", "UNASSIGNED"]
    print(f"  {'site':<12}" + "".join(f"{s:>11}" for s in order) + f"{'total':>8}")
    for site in sorted(tally):
        t = tally[site]
        print(f"  {site:<12}" + "".join(f"{t.get(s,0) or '.':>11}" for s in order)
              + f"{sum(t.values()):>8}")
    tot = defaultdict(int)
    for t in tally.values():
        for s, n in t.items():
            tot[s] += n
    print(f"  {'ALL':<12}" + "".join(f"{tot.get(s,0) or '.':>11}" for s in order)
          + f"{len(ids):>8}")

    show = [r for r in rows if a.all or r[2] in ("RUNNING", "ERROR", "PARTIAL", "DONE", "REDO")]
    if a.site:
        show = [r for r in show if r[0] == a.site]
    if show:
        print()
        last = None
        for site, rid, state, detail in sorted(show):
            if site != last:
                print(f"\n{site}")
                last = site
            print(f"  {rid:<42} {state:<10} {detail}")
    if not a.no_hub and not owner:
        print("\n(no assignments/ read from the hub — check auth)", file=sys.stderr)


if __name__ == "__main__":
    main()
