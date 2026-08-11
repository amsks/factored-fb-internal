# exorl_dmc_tasks — vendored task definition (point_mass_maze only)

- **Upstream:** https://github.com/denisyarats/exorl
- **Pinned commit:** a3fb07a420939280aa0918150923dcca7e82bf2a (2022-02-07)
- **Vendored on:** 2026-08-11 (spec 23 P0)
- **License:** MIT (exorl repo); see `LICENSE` in this directory.
- **Scope:** ONLY `custom_dmc_tasks/point_mass_maze.py` +
  `custom_dmc_tasks/point_mass_maze_reach_{top,bottom}_{left,right}.xml`
  (the 4-goal point-mass domain used for ExoRL's `point_mass_maze` dataset
  and the P0 cell of spec 23). Not the full exorl repo, not the other
  domains (walker/cheetah/quadruped) — those are out of scope for P0 and
  should be vendored separately if/when spec 23a opens them.
- The `<include file="./common/*.xml">` directives in the task XMLs resolve
  against `dm_control.suite.common.ASSETS`, which already ships identical
  copies of `skybox.xml` / `visual.xml` / `materials.xml` under those exact
  keys (verified empirically 2026-08-11: `dm_control.suite.common.ASSETS`
  has keys `./common/{skybox,visual,materials}.xml`). exorl's own
  `custom_dmc_tasks/common/` dir is therefore NOT needed and was not
  vendored here.

## One deliberate patch (documented, not upstream-silent)

`get_model_and_assets()` in `point_mass_maze.py` originally resolved the
task XML by walking `dirname(dirname(__file__))` and rejoining
`'custom_dmc_tasks'` — i.e. it hardcoded the upstream repo's package name
and layout (`<root>/custom_dmc_tasks/point_mass_maze.py` next to
`<root>/custom_dmc_tasks/point_mass_maze_{task}.xml`). Vendored directly
under `third_party/exorl_dmc_tasks/` (no `custom_dmc_tasks` package name),
that path breaks. Patched to a plain `__file__`-relative join instead.
This is the only line changed from upstream; everything else in
`point_mass_maze.py` (the `Physics`/`MultiTaskPointMassMaze`
classes, `TASKS` list, reward function) is byte-identical to upstream.

## Verification (2026-08-11)

Instantiated `reach_top_left()` from a cwd OUTSIDE this directory
(`/home/adityamohan/git/Factored-FB`, `sys.path` pointing at `third_party/`)
and called `env.reset()`: obs keys `['position', 'velocity']`, both
2-dim, matching the flat-observation layout used by the ExoRL collector
(`dmc.py`'s `flat_observation=True`, which concatenates `OrderedDict`
values in insertion order -> `[x, y, vx, vy]`). `action_spec` bounds are
already `[-1, 1]` (the XML's own `ctrlrange`), so ExoRL's
`action_scale.Wrapper(minimum=-1, maximum=1)` is an identity map for this
domain — not vendored here since it is not load-bearing (any rescale
wrapper is inert; the eval adapter steps the raw `[-1, 1]`-bounded action
spec directly).

See `PAPER/Multistep-FB/specs/23-humanoid-eval.md` §4 and
`.superpowers/sdd/spec23-p0/task-report.md` for how this is used.
