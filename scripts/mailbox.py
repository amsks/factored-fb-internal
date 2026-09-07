"""Persistent mailbox between the eval box and the cluster agent, on the Hugging Face repo
both sides already write to (amsks/factored-fb-checkpoints). Survives either side shutting
off; nothing is lost if a message lands while the receiver is offline.

Layout in the repo:
    mailbox/<sender>/<UTC-timestamp>.md     one message per file, never overwritten
    status/<run-dir-name>.json              per-run status written by the training side
                                            (final step, eval routes, wall-clock)

    python scripts/mailbox.py post --from cluster  --text "Tier 1 done: 24/24 at 1M"
    python scripts/mailbox.py post --from evalbox  --file note.md
    python scripts/mailbox.py read --since 2026-09-07T00:00   # everything newer, all senders
    python scripts/mailbox.py read --sender cluster --last 5
    python scripts/mailbox.py status                            # summarize status/*.json
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("post"); p.add_argument("--from", dest="sender", required=True); p.add_argument("--text"); p.add_argument("--file")
    r = sub.add_parser("read"); r.add_argument("--sender"); r.add_argument("--since"); r.add_argument("--last", type=int, default=0)
    sub.add_parser("status")
    a = ap.parse_args(); {"post": post, "read": read, "status": status}[a.cmd](a)
