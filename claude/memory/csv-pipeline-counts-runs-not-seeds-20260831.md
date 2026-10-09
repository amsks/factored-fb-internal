---
name: csv-pipeline-counts-runs-not-seeds-20260831
description: "2026-08-31 AUDIT — compile_results_csvs.py's n_seeds is n_RUNS; it never reads seed identity, so relaunched seeds AND distinct arms inflate n and shrink ci95"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0c552c33-cc1f-4376-8470-a0e87d93b4eb
  modified: 2026-08-31T17:49:45.654Z
---

`scripts/compile_results_csvs.py` (the pipeline behind `summary_agents.csv` /
`bridge_grafts.csv`) groups by **wandb group**, iterates runs, appends one value per run,
and reports `'n_seeds': len(vals)` with `ci95 = 1.96*sd/sqrt(n)`. **It never parses a seed.**
The column named `n_seeds` is `n_runs`. Its only defense is the hand-maintained
`exclude_runs` list in `results_map.py` — retrospective and per-incident (that is what
[[hx-cd-hard-collected-20260816]] added after the restore-continuation contamination).

Audited the cached dump on 2026-08-31. Most duplicate-name runs are crashed twins the
step-window correctly drops. **Five groups actively double-count**, in two distinct flavors:

**(A) same seed, two finished runs** — inflates n, shrinks ci95, flattering direction:
- `GCIQL-CubeDouble-Flow-Measure-Density-DDPGBC-d50` (`dbx_cd_soft_s0` + `_s0_re`):
  .6460±.0113 n=4 → **.6440±.0120 n=3**
- `GCIQL-CubeDouble-Flow-Measure-DensityHard-DDPGBC-d50` (`dbx_cd_hard_s2` + `_s2_re`):
  .5990±.0474 n=4 → **.6040±.0642 n=3** (ci95 understated ~35%)
- `GCIVL-State-Flow-Bilinear-d50` (`gcivl_flow_bil_s0`/`_s2` each twice, same name):
  .4752±.3802 n=5 → **.5287±.2620 n=3**

**(B) DISTINCT ARMS merged into one row — worse, and not an error-bar issue but a
meaningless number.** `ISFB-Torch-Exp1-fixed` reports .2267 "n=9" which is the average of
THREE different methods, with their between-method spread reported as seed noise:
coord .3333±.0346 n=3 / scalar_gate_awr .2667±.0795 n=3 / scalar_gate_hard .0800±.0226 n=3.
(HF manifest confirms these shipped as three separate methods.) `GCIQL-State-Flow-Bilinear-d50-ddpgbc`
mixes `db_a_a03/a10/a30_s0` alpha arms and needs the same look — not yet confirmed.

**Why:** found because the other track's session hit the identical shape in
`exorl_paper_eval.report()` (seed parsed from the row tag then discarded) and in
`campaign/wandb_agg_daemon.py::read_group` (jsonl rows counted as seeds). Three components,
one failure mode: per-row records counted as seeds, silent, always flattering.

**⚠ TWO OF THESE ARE IN THE PRIMARY PAPER'S MAIN TABLE** (`PAPER/Multistep-FB/density-fb.tex`,
`tab:main`, lines 508-512 + caption 546/554-555). The caption *explicitly* claims
"`cube-double` with DDPG + BC ... uses four [seeds]" and prints four per-seed values that
reproduce the contaminated set bit-for-bit. Both are really 3 seeds with one `_re` re-run
double-counted (soft: `dbx_cd_soft_s0` .644 + `_s0_re` .660; hard: `dbx_cd_hard_s2` .596 +
`_s2_re` .572). Paper uses a t-based 95% CI (df=n-1), reproduced exactly.
- soft DDPG+BC (**bolded cell**) $0.646\pm0.019$ → **$0.644\pm0.026$** (n=3, interval 1.4x)
- hard DDPG+BC $0.599\pm0.077$ → **$0.604\pm0.141$** (n=3, interval 1.8x)
Means barely move, so the surrounding claims survive (the "0.646 vs published 0.40 GCIQL"
comparison at line 562 still clears comfortably); what is wrong is n, the widths, and the
caption's seed count and per-seed lists.

**How to apply:** the structural fix is to key by seed (`{seed: value}`, last/finished wins)
and set `n_seeds = len(distinct seeds)`, so a relaunch cannot inflate n — not another
`exclude_runs` entry. Flavor (B) is a `results_map.py` mapping defect, not a dedup one: those
groups need splitting into separate rows. **Not yet fixed and not yet applied to the paper —
report was delivered 2026-08-31, awaiting the user's call.** Any table quoting these rows is
suspect until then. See [[results-csv-pipeline]].
