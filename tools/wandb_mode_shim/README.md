# wandb_mode_shim

This directory is **not a package** — it is a `sitecustomize.py` deliberately
placed on `PYTHONPATH` by `run_gciql.py` for the OGBench child process only.

OGBench's vendored `setup_wandb` hardcodes `wandb.init(mode='online')`, and
wandb (>= 0.27) lets that explicit kwarg override `WANDB_MODE` /
`WANDB_DISABLED`. Since `third_party/ogbench/` must stay byte-for-byte upstream
(see its `PROVENANCE.md`), `sitecustomize.py` monkeypatches `wandb.init` at
interpreter startup to honor `wandb_mode=offline|disabled`.

It is a complete no-op unless `GCIQL_WANDB_FORCE_MODE` is set, so it never
affects normal online runs or any other Python process.
