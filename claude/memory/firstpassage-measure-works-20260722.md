---
name: firstpassage-measure-works-20260722
description: BREAKTHROUGH — first-passage measure-only critic WITHOUT ortho scores ~0.967 on cube (near-SOTA); the successor-measure-that-works we were hunting for
metadata: 
  node_type: memory
  type: project
  originSessionId: 991bc365-f4fa-4a35-9217-fd621fb9903d
  modified: 2026-07-22T07:52:23.482Z
---

2026-07-22 (bridge track, spec06). A **pure measure-only** GCIQL critic finally WORKS on cube,
overturning the prior "the measure's own fixed point is BAD on cube" conclusion
([anneal-curriculum-campaign-20260710], [mdose-campaign-20260708]).

**The two ingredients that matter (both required):**
1. `measure_first_passage: true` — first-passage target, NOT discounted occupancy.
2. `measure_ortho_coef: 0.0` — orthonormalization regularizer OFF.

**Clean 2×2 evidence (all `measure_only=true`, `value_arch=bilinear`, d50, 3 seeds, 500k, flow actor):**
- FirstPassage + **no-ortho**: s0=0.956 s1=0.964 s2=0.980 → **mean ≈ 0.967** (stable 10-ckpt curve, 0.94 by 50k)
- FirstPassage + with-ortho: 0.188 / 0.272 / 0.216 → mean ≈ 0.225  ⇒ **ortho was killing it**
- CellW0 (occupancy) + no-ortho: 0.004 / 0.516 / 0.004 → floored/unstable ⇒ **no-ortho alone insufficient; needs first-passage**

So: first-passage successor structure is learnable AND controllable on cube where discounted
occupancy floors at ~0.10; the ortho constraint was the toxin for it. Contrast with
[t7-t8-campaign-20260720] (conservatism/one-step FB both dead) and
[handoff-20260721-campaigns-queue-dashboard] ("measure cells at floor", "user wants a
successor-measure-that-works").

**Configs:** `configs/gciql/cube_single_state_measure_firstpassage_noortho_d50.yaml` (+ the
with-ortho `..._firstpassage_d50.yaml` as the control). Runs in `/home/adityamohan/exp_spec06`.
Agent knobs live in `tools/wandb_mode_shim/gciql_flow.py` (`measure_first_passage`,
`measure_ortho_coef`). wandb groups: `GCIQL-State-Flow-Measure-FirstPassage-NoOrtho-d50` /
`...-FirstPassage-d50`.

**Next:** verify it's not an eval artifact (independent eval / more seeds); add both groups to
`scripts/results_map.py` + refresh CSVs ([results-csv-pipeline]); write up in
`PAPER/rescuing-fb/phenomenon.tex`. **How to apply:** this is likely the paper's positive
result — a measure grounding that WORKS — so treat it as high-priority, not a one-off cell.
