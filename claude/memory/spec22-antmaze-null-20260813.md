---
name: spec22-antmaze-null-20260813
description: "2026-08-13 spec22 antmaze-medium RESOLVED at 3 seeds: two-sided NULL, not a demonstrated non-improvement. hard+ddpgbc .877±.097 vs matched GCIQL control .837 vs soft+ddpgbc .829±.058 — both intervals CONTAIN the control. Also corrected a best-of-3 number the paper was quoting as '0.772 (1 seed)' for hard+flow; full 3 seeds give .669±.240"
metadata:
  node_type: memory
  type: project
  originSessionId: 576b53a0-d404-4a89-9380-6fece089b102
  modified: 2026-08-13T01:11:03.998Z
---

Measure track. Spec 22 (antmaze-medium + actor ablation), pre-registered NULL
= "no dwell asymmetry in locomotion". Follows [[spec18-density-measure-20260806]].

**All four antmaze cells now at 3 seeds** (config-verified via `flags.json`,
NOT run-group names, which lie here: the `gap_am_hard_ddpg_*` runs sit in a
group string containing "Flow"):

| arm | seeds | mean ± 95% t-CI |
|---|---|---|
| soft + flow | .504/.636/.548 | .563 ± .167 |
| hard + flow | .772/.580/.656 | **.669 ± .240** |
| soft + ddpgbc | .820/.812/.856 | .829 ± .058 |
| hard + ddpgbc | .920/.868/.844 | **.877 ± .097** |

matched GCIQL control = **.837**; HIQL published = .96.

**The verdict is a TWO-SIDED null.** Both ddpgbc intervals contain .837, so
antmaze separates neither variant from the control in either direction. ⚠ Do
NOT write "DensityFB does not improve AntMaze" — that phrasing was licensed
only by the soft number (.829 < .837) and the hard arm's point estimate is
ABOVE the control (.877). Equally, do not claim hard wins: .837 is inside its
CI. Resolving a .04 difference needs more seeds than 3.

**Correction shipped to `density-fb.tex`.** The table had quoted hard+flow as
"$0.772$ (1 seed)". That .772 is the BEST of the three seeds (.772/.580/.656);
the honest cell is .669 ± .240. Same class of error as the mid-run-eval trap in
[[spec18-density-measure-20260806]]: a single number read off an incomplete
wave flatters the method. Fill in every seed before quoting a cell.

Also filled: scene hard+ddpgbc "pending" → .664, .648 (2 seeds, `dbx_sc_hard_s*`;
s2 was at 450k and was NOT quoted). Still training: cube-double s2 both variants,
scene hard s2, all plain-FB ddpgbc cells.

**Aggregation trap.** Sweeping runs by (env, actor_kind, measure_density_hard)
merges distinct waves and produces nonsense: cube-single flow soft came back as
6 seeds `[.896 .928 .964 .132 .124 .076]`, two clearly different populations,
and plain-FB cube-single as n=129 mean .475. Key on the wave's run TAGS
(`a22_*`, `gap_*`, `dbx_*`, `hx_*`, `d18x_*`), never on a broad config sweep.

Related: [[spec24-z-index-20260813]], [[density-audit-20260810]],
[[fb-state-run-name-map]], [[primary-paper-dir-multistep-fb]].
