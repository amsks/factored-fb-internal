---
name: rule-rldp-ortho-higher-cube-focus-exorl-locomotion
description: "USER RULE (2026-09-12, emphatic, repeated): for RLDP a HIGHER ortho coefficient is better — never propose lowering it or running a lower-ortho cell; antmaze is OUT of scope (do not analyse or propose antmaze runs); the benchmark focus is CUBE MANIPULATION (cube-single, cube-double, scene, puzzle); locomotion validation of our RLDP/HILP reimplementations comes from ExoRL, where we already matched published, and ExoRL numbers belong in the paper's baselines tables."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: be24e170-3bbf-4591-adc3-bb535ce89254
  modified: 2026-09-12T00:46:22.990Z
---

**Rule.** (1) RLDP's orthonormality regularizer: higher coefficient = better control. On ExoRL walker the live regularizer at coef 1.0 gave 62-94% of published, weakening it (1e-4) was worse than inert ([[rldp-ortho-dead-regularizer-20260831]]). Never suggest a lower-ortho cell again. (2) Do not spend analysis or compute on antmaze; the user does not care about it. (3) The paper's benchmark focus is cube manipulation (cube-single, cube-double, scene, puzzle). (4) Our RLDP and HILP reimplementations are validated on ExoRL locomotion (docs/notes/2026-09-04-exorl-rldp-hilp-results-table.md), and those numbers go into PAPER/ICLR/baselines as the locomotion-validation table.

**Why:** the user has said the ortho point several times and got angry when I proposed the low-ortho grid cell as the reason for the RLDP gap on 2026-09-12.

**How to apply:** when explaining an RLDP gap, the only ortho direction to consider is UP (e.g. the td_jepa manipulation grid's 0.1 cell vs our 0.01). Frame reimplementation fidelity as: ExoRL matches published ⇒ the implementation is right; OGBench manipulation is where the method comparison lives. Related: [[tdjepa-published-ogbench-table-20260907]], [[exorl-validation-campaign-20260828]].
