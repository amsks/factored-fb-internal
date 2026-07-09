# Installation

Factored-FB uses [`uv`](https://github.com/astral-sh/uv) with a project venv
(`.venv/`). Python **3.11–3.13** is supported; **3.14 is not** (the `Box2D`
package has no wheels for it yet).

The default `uv sync` / `uv run` setup installs the core FB stack **plus**
`rlmini-envs` (Air Hockey, Box2D, etc.) and **does not** pull in OGBench /
`labmaze` / `bazel`. Use the `ogbench` extra when you need the OGBench training
track.

## Quick start (x86_64 or aarch64 with prebuilt Box2D wheels)

```bash
cd Factored-FB
uv venv --python 3.11
source .venv/bin/activate
make install          # uv sync (core + rlmini-envs)
make smoke            # optional: synthetic pipeline check (~20s)
```

Run scripts with either `python …` or `uv run …` (both use the same `.venv`):

```bash
uv run python scripts/data/run_crl_iws_train.py --help
```

## Sibling repo: rlmini-envs

Air Hockey and Box2D tasks require a sibling checkout of
[`rlmini-envs`](https://github.com/CalCharles/rlmini-envs):

```text
workspace/
├── Factored-FB/
└── rlmini-envs/          # cloned next to this repo (symlink OK)
```

If you use a `~/work/Factored-FB` symlink without a sibling `~/work/rlmini-envs`,
either clone/symlink rlmini-envs there or set `RLMINI_ENVS_PATH` before
`scripts/dev/install_vista.sh` (the script can auto-link from
`/work/.../vista/rlmini-envs` on TACC).

`pyproject.toml` points at `../rlmini-envs` as an editable path dependency.
Initialize submodules after cloning:

```bash
cd ../rlmini-envs
git submodule update --init --recursive
cd ../Factored-FB
make install
```

## TACC Vista / Linux aarch64 (Box2D from source)

On **linux aarch64** (including TACC Vista), PyPI does not ship prebuilt `Box2D`
wheels. Load GCC + SWIG, force GCC as the C++ compiler (avoid `nvc++` on
`PATH` after GPU modules), then sync:

```bash
cd Factored-FB
module load gcc/14.2.0 swig/4.2.1   # adjust module names if needed
export CC=gcc CXX=g++

uv venv --python 3.11
source .venv/bin/activate
make install-vista
```

`make install-vista` runs `uv sync` with the git-sourced Box2D override defined
in `pyproject.toml` (`tool.uv.sources.box2d` →
`https://github.com/pybox2d/pybox2d.git`). You do **not** need a separate
`uv pip install git+…` step (unlike the manual flow in
[`Unsupervised_Manipulation`](../Unsupervised_Manipulation/README.md)).

**Runtime:** load the same GCC module before importing Box2D environments (the
extension links against `GLIBCXX_3.4.32+`):

```bash
module load gcc/14.2.0
python -c "import Box2D; print('Box2D OK')"
```

If import still fails, try:

```bash
export LD_LIBRARY_PATH=/opt/apps/gcc/14.2.0/lib64:$LD_LIBRARY_PATH
```

This matches the procedure documented in the sibling
[`Unsupervised_Manipulation`](../Unsupervised_Manipulation/README.md#rlmini-envs-air-hockey-and-box2d)
repo.

## OGBench track (optional)

OGBench pulls in `dm-control` → `labmaze`, which requires **bazel** to build on
some platforms. Only install this extra when running the OGBench FB/GCIQL
pipelines:

```bash
make install-ogbench
make data
python train.py domain=antmaze_medium
```

## Dependency layout

| Install command | What you get |
|---|---|
| `make install` / `uv sync` | Core FB deps + `rlmini-envs` + Box2D (git on aarch64) |
| `make install-ogbench` | Above + `ogbench` (needs bazel for labmaze on some hosts) |
| `uv sync --group dev` | Above + black, ruff, pytest, pre-commit |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `uv run` tries to build `labmaze` | Use `make install` (default sync omits ogbench). Do not `uv sync --extra ogbench` unless you need OGBench. |
| `ModuleNotFoundError: Box2D` | Run `make install-vista` on aarch64; ensure `module load gcc swig` and `CC=gcc CXX=g++`. |
| `command 'bazel' failed` | OGBench extra only; skip unless needed, or install bazel. |
| `Environment AirHockeyPuckGoalPosition doesn't exist` | Use `python`/`uv run` with rlmini installed; the script requires `rlmini_envs`, not plain gymnasium. |
| `VIRTUAL_ENV` warning from parent `.venv` | `deactivate` or `unset VIRTUAL_ENV` before `uv run` in Factored-FB. |
