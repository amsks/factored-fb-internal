# hf_dataset/ — offline dataset for HuggingFace (setup & maintenance)

Turns the collected per-episode datasets (`data_collection/`) into a single,
HuggingFace-ready dataset and keeps it maintainable. Kept in its own folder so it
does not touch training code.

```
hf_dataset/
  pack_and_manifest.py   # per-episode npz  ->  per-task stacked npz + manifest.json
  push_to_hub.py         # upload the packed export to a HF dataset repo (re-runnable)
  load_hf_dataset.py     # load a task back from the hub or a local export
  README.md              # the HF dataset card (shipped as the repo README)
  SETUP.md               # this file
```

Data roots (durable, on `/raid`):

| what | path |
|---|---|
| collected per-episode (source of truth) | `/raid/tongle/rl/fb-data/collected/<Family>__<task>/buffer/*.npz` |
| packed HF export (generated) | `/raid/tongle/rl/fb-data/hf_export/` |

## One-time setup

```bash
FB=/raid/tongle/rl/Factored-FB
$FB/.venv/bin/pip install -U huggingface_hub          # once
huggingface-cli login                                 # paste a HF token (write scope)
```

## Build / refresh the packed export

```bash
$FB/.venv/bin/python hf_dataset/pack_and_manifest.py \
    --collected /raid/tongle/rl/fb-data/collected \
    --out       /raid/tongle/rl/fb-data/hf_export
# -> data/<family>/<task>.npz (+ .meta.json), manifest.json
```

## Push to the hub (create or update)

```bash
$FB/.venv/bin/python hf_dataset/push_to_hub.py \
    --export  /raid/tongle/rl/fb-data/hf_export \
    --repo-id <username>/factoredfb-testbed-demos \
    --private                                          # start private; make public later
# -> https://huggingface.co/datasets/<username>/factoredfb-testbed-demos
```

`push_to_hub.py` overwrites only changed files, so it is also the **update** step:
collect more tasks -> re-pack -> push again.

## Load it back

```bash
# list everything in a local export or on the hub
$FB/.venv/bin/python hf_dataset/load_hf_dataset.py --export /raid/tongle/rl/fb-data/hf_export

# in code:
from hf_dataset.load_hf_dataset import load_task
eps = load_task("ms-pick-cube", repo_id="<username>/factoredfb-testbed-demos")
```

## Extend the dataset

1. **New MW/ManiSkill tasks** — add to `data_collection/run_collection.sh`, run it.
2. **New Box2D/Pusher2D tasks** — train the policy (newt `train_rlmini.py`), then add
   to `data_collection/run_collection_box2d.sh` and run it.
3. Re-run `pack_and_manifest.py` then `push_to_hub.py`.

## For Factored-FB training

Train directly on the **collected** root (not the packed export) — the packed export is
for hosting/inspection:

```bash
$FB/.venv/bin/python train.py domain=<Family>/<task> \
    data_path=/raid/tongle/rl/fb-data/collected data_format=unified \
    build_env=false eval_every=0
```
