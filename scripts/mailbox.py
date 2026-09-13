"""Persistent mailbox between the three compute sites, on the Hugging Face repo they all
write to (amsks/factored-fb-checkpoints). Survives any side shutting off; nothing is lost
if a message lands while the receiver is offline.

Sites (2026-09-07): `evalbox` = the AWS box, and the COORDINATOR — it reads resources and
writes the assignments; `vista` = the TACC cluster (formerly `cluster`; that folder holds
the pre-rename history); `luis` = the LUIS cluster, shared with another project.

Layout in the repo:
    mailbox/<sender>/<UTC-timestamp>.md     one message per file, never overwritten
    status/<run-dir-name>.json              per-run status written by the training side
                                            (final step, eval routes, wall-clock)
    resources/<site>.json                   what a site currently has; overwritten on each
                                            report, so it is always "now", not a log
    assignments/<site>.json                 written by the coordinator only: which run ids
                                            that site owns. A site launches ONLY these.

    python scripts/mailbox.py post --from vista --text "Tier 1 done: 24/24 at 1M"
    python scripts/mailbox.py post --from luis  --file note.md
    python scripts/mailbox.py read --since 2026-09-07T00:00   # everything newer, all senders
    python scripts/mailbox.py read --sender vista --last 5
    python scripts/mailbox.py status                          # summarize status/*.json

    # a site reports what it has (overwrites its own file):
    python scripts/mailbox.py resources post --site vista --file res.json
    python scripts/mailbox.py resources show                  # coordinator: all sites, one table

    # the coordinator hands out work; a site reads only its own:
    python scripts/mailbox.py assign --site luis --file assign_luis.json
    python scripts/mailbox.py assignment --site luis

Needs an HF token for the amsks account (`hf auth login`). Read side needs read access only.
"""
import argparse, datetime as dt, io, json, sys
from huggingface_hub import HfApi, hf_hub_download

REPO = "amsks/factored-fb-checkpoints"


def _api(): return HfApi()


def post(args):
    text = open(args.file).read() if args.file else args.text
    assert text, "give --text or --file"
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    path = f"mailbox/{args.sender}/{ts}.md"
    body = f"---\nfrom: {args.sender}\nutc: {ts}\n---\n{text.rstrip()}\n"
    _api().upload_file(path_or_fileobj=io.BytesIO(body.encode()), path_in_repo=path,
                       repo_id=REPO, repo_type="model", commit_message=f"mailbox: {args.sender} {ts}")
    print(f"posted {path}")


def read(args):
    files = [f for f in _api().list_repo_files(REPO, repo_type="model") if f.startswith("mailbox/")]
    if args.sender: files = [f for f in files if f.startswith(f"mailbox/{args.sender}/")]
    if args.since: files = [f for f in files if f.split("/")[-1][:16].replace("-", ":", 0) >= args.since.replace(":", "-")]
    files = sorted(files)[-args.last:] if args.last else sorted(files)
    if not files: print("(no messages)"); return
    for f in files:
        p = hf_hub_download(REPO, f, repo_type="model"); print(f"===== {f}"); print(open(p).read())


def status(args):
    files = sorted(f for f in _api().list_repo_files(REPO, repo_type="model") if f.startswith("status/") and f.endswith(".json"))
    if not files: print("(no status files)"); return
    for f in files:
        d = json.load(open(hf_hub_download(REPO, f, repo_type="model")))
        print(f"{f.split('/')[-1][:-5]:60s} step={d.get('final_step')} routeA={d.get('route_a')} routeB={d.get('route_b')} {d.get('note','')}")


SITES = ("evalbox", "vista", "luis")


def _put_json(path, obj, msg):
    body = json.dumps(obj, indent=2, sort_keys=True) + "\n"
    _api().upload_file(path_or_fileobj=io.BytesIO(body.encode()), path_in_repo=path,
                       repo_id=REPO, repo_type="model", commit_message=msg)
    print(f"wrote {path}")


def _get_json(path):
    return json.load(open(hf_hub_download(REPO, path, repo_type="model")))


def resources(args):
    """post: a site overwrites resources/<site>.json. show: the coordinator reads all of them."""
    if args.action == "post":
        obj = json.load(open(args.file)) if args.file else json.loads(args.json)
        obj["site"] = args.site
        obj["utc"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        _put_json(f"resources/{args.site}.json", obj, f"resources: {args.site}")
        return
    files = sorted(f for f in _api().list_repo_files(REPO, repo_type="model")
                   if f.startswith("resources/") and f.endswith(".json"))
    if not files:
        print("(no resource reports)"); return
    for f in files:
        d = _get_json(f)
        print(f"===== {f}  (reported {d.get('utc', '?')})")
        print(json.dumps({k: v for k, v in d.items() if k not in ("site", "utc")}, indent=2, sort_keys=True))


def assign(args):
    """Coordinator writes assignments/<site>.json. Overwrites: it is the current assignment."""
    obj = json.load(open(args.file)) if args.file else json.loads(args.json)
    obj["site"] = args.site
    obj["utc"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    _put_json(f"assignments/{args.site}.json", obj, f"assign: {args.site}")


def assignment(args):
    """A site reads its own assignment (or the coordinator reads them all to check disjointness)."""
    files = sorted(f for f in _api().list_repo_files(REPO, repo_type="model")
                   if f.startswith("assignments/") and f.endswith(".json"))
    if args.site:
        files = [f for f in files if f == f"assignments/{args.site}.json"]
    if not files:
        print("(no assignment)"); return
    seen, dupes = {}, []
    for f in files:
        d = _get_json(f)
        print(f"===== {f}  (assigned {d.get('utc', '?')})")
        print(json.dumps({k: v for k, v in d.items() if k not in ("site", "utc")}, indent=2, sort_keys=True))
        for rid in d.get("run_ids", []):
            if rid in seen:
                dupes.append((rid, seen[rid], d["site"]))
            seen[rid] = d["site"]
    if dupes:
        print("\n!! OVERLAP -- the same run id is assigned to two sites:")
        for rid, a, b in dupes:
            print(f"   {rid}: {a} and {b}")
    elif len(files) > 1:
        print(f"\nOK: {len(seen)} run ids, no site overlap.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("post"); p.add_argument("--from", dest="sender", required=True); p.add_argument("--text"); p.add_argument("--file")
    r = sub.add_parser("read"); r.add_argument("--sender"); r.add_argument("--since"); r.add_argument("--last", type=int, default=0)
    sub.add_parser("status")
    rs = sub.add_parser("resources"); rs.add_argument("action", choices=["post", "show"], nargs="?", default="show")
    rs.add_argument("--site", choices=SITES); rs.add_argument("--file"); rs.add_argument("--json")
    asg = sub.add_parser("assign"); asg.add_argument("--site", choices=SITES, required=True)
    asg.add_argument("--file"); asg.add_argument("--json")
    an = sub.add_parser("assignment"); an.add_argument("--site", choices=SITES)
    a = ap.parse_args()
    {"post": post, "read": read, "status": status, "resources": resources,
     "assign": assign, "assignment": assignment}[a.cmd](a)
