# exorl_dmc_tasks — vendored ExoRL task definitions (point_mass_maze / walker / cheetah)

- **Upstream:** https://github.com/denisyarats/exorl
- **Pinned commit:** a3fb07a420939280aa0918150923dcca7e82bf2a (2022-02-07)
- **License:** MIT (exorl repo); see `LICENSE` in this directory.
- **Scope:** only the task modules the three T3 ExoRL domains need —
  `custom_dmc_tasks/{point_mass_maze,walker,cheetah}.py` plus their XML models
  (`walker.xml`, `cheetah.xml`, `point_mass_maze_reach_{top,bottom}_{left,right}.xml`).
  Not `hopper`/`quadruped`/`jaco`, not the agents, not the collection loop.

These modules exist because ExoRL's benchmark tasks are **not all** dm_control suite
tasks: `cheetah run_backward / flip / flip_backward`, `walker flip` and the four
`point_mass_maze reach_*` targets are ExoRL additions. The suite tasks that ExoRL also
scores (`walker stand/walk/run`, `cheetah run`) are still loaded from
`dm_control.suite`, exactly as ExoRL's own `dmc.py` resolves them (suite first, custom
tasks second) — see `env/exorl_env.py::_load_dmc`.

`<include file="./common/*.xml">` directives in the task XMLs resolve against
`dm_control.suite.common.ASSETS`, which ships identical copies under those exact keys, so
exorl's own `custom_dmc_tasks/common/` directory is not needed and was not vendored.

## Deliberate patches (documented, not upstream-silent)

1. **XML path resolution** in all three modules' `get_model_and_assets()`. Upstream walked
   `dirname(dirname(__file__))` and rejoined the hardcoded package name `custom_dmc_tasks`,
   i.e. it assumed `<root>/custom_dmc_tasks/<name>.py` sits next to
   `<root>/custom_dmc_tasks/<name>.xml`. Vendored under `env/exorl_dmc_tasks/`
   (no `custom_dmc_tasks` directory), that path does not exist. Patched to a plain
   `__file__`-relative join. This is the only line changed in each module; the
   `Physics` / task classes, `TASKS` tables and reward functions are byte-identical to
   upstream.
2. **`__init__.py`** is a rewrite, not upstream's. Upstream's did
   `from custom_dmc_tasks import cheetah` (absolute, package-name-dependent) at import
   time, which both breaks under the vendored name and drags dm_control/mujoco into every
   importer. Ours resolves sub-modules lazily by relative name and keeps upstream's
   `make(domain, task, ...)` dispatch signature.

## Reward relabelling — why these files are load-bearing

The reward column stored in ExoRL's own episode buffers is the reward of the task the data
was **collected** under, not of the task you want: for `point_mass_maze/rnd` it is shape
`(T+1, 4)` holding four identical copies of the `reach_top_left` reward (verified on the
downloaded data, 2026-08-22 — corr 0.9994 against relabelling for that target, max 0.87).
ExoRL's offline protocol never reads it — `exorl/utils.py::relabel_episode` replays each
stored `physics` state into the *target* task's env and calls `task.get_reward(physics)`.
`env/exorl_env.py::relabel_rewards` reproduces that, which is why the reward functions in
these modules have to be present and faithful: a loader that passed the stored column
through would be silently correct on one task per domain and silently wrong on all the
others.
