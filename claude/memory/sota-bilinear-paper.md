---
name: sota-bilinear-paper
description: The SOTA-Bilinear paper — REWRITTEN 2026-07-07 with HIQL mechanism section + audit corrections; @400k cells (highbil 34.7/41.3, rep50 23.5) need @500k refresh
metadata: 
  node_type: memory
  type: project
  originSessionId: 00be5f9a-855e-4726-bc21-154ca4e4bf29
---

New paper **`PAPER/ogbench-sota/ogbench-sota.tex`** (created 2026-07-06; dir renamed
from `sota-bilinear/` on 2026-07-06). This is project ② "OGBench-SOTA" in the
3-project split (see `PAPER/PROJECT-MAP.md`). Question: does swapping a
method's MLP value head for bilinear `Q=φ(s,a)·ψ(s,g)/√d` (rank d) set a new OGBench
SOTA? Factorized across **GCIQL, HIQL, GCIVL** on cube-single / antmaze-medium /
scene (state, 3-seed @500k, `evaluation/overall_success` from `amsks/factored-fb`).
Flow AND native/DDPG+BC actors shown as separate rows. Colour: green=local,
orange=OGBench-paper (Table 2, arXiv:2410.20092, @1M/8-seed).

**Verdict = conditional SOTA.** Wins over published: GCIQL cube+flow 90.4 vs 68;
GCIQL antmaze+DDPG+BC 83.7 vs 71; GCIVL cube+AWR 67.5 vs 53. Non-wins: HIQL nowhere
(≤ MLP 24.6), scene (only ran GCIQL+flow 29.2 < 51). **Key confound: the winning
actor flips by domain** — flow wins cube, DDPG+BC wins antmaze. Rank cliff at d5
(12.4 vs ~90 for d∈{50,128,512}). GCIVL flow-bilinear bimodal (47±39, unreliable).

Data snapshot: `PAPER/ogbench-sota/data-500k.json`. Build PDF via
[[latex-engine-tectonic]]. Companion to [[fb-to-gciql-bridge-knobs]] (the bilinear
runs originate there).

**HIQL debug (2026-07-06):** the negative HIQL result is CONFOUNDED. OGBench
`hiql.py` defaults `subgoal_steps=25` (line 370) but the tuned cube/scene value is
**10** (`hyperparameters.sh`); ALL prior HIQL runs (Stock 24.6, Bilinear 15.9,
Flow-both 22.8) used 25 (verified via wandb config), and our MLP baseline 24.6 is
itself off-protocol (> published 15). Ties to [[hiql-subgoalsteps-bug-and-comparison]].
Fix = re-run bilinear-vs-MLP at sg=10.

**MAJOR REWRITE 2026-07-07 (tex + PDF rebuilt):** full incorporation of the sg10
correction, the audit-20260707 fixes, and the HIQL mechanism campaign. New content:
- Main table HIQL rows now sg10: MLP 16.7±5.8 / bilinear-via-rep 18.5±0.2 (old
  "bilinear hurts HIQL" conclusion RETIRED — parity at correct HPs) / Schaul 32.8±0.4
  (done @500k) / high-head split 34.7±6.1° / flow high-head 41.3±2.3° (°=@400k
  in-flight, marked with \inflight). Antmaze HIQL matched-protocol column: MLP 95.3±0.6
  vs bilinear 95.9±0.5 (8s@1M avg-last-3) = warp harmless at ceiling.
- New §4 "Why bilinear fails HIQL — and how to fix it": shared-rep trainer mechanism +
  probe evidence (PR no collapse, R² 0.14→0.08, Spearman 0.115→0.06) + mechanism-ladder
  table + sg25 history paragraph.
- GCIVL flow-bilinear corrected 47.5±38.8 → healthy-batch **79.2±0.4** (bad-batch
  artifact footnoted); now the largest GCIVL effect.
- Scene bilinear DDPG+BC landed: **25.4±0.3** (n=2@500k, s0@300k=20.8) — bilinear loses
  scene with BOTH actors; actor-flip table gained scene row + stock column.
- α=0.3 confound footnoted on cube DDPG+BC 51.3; antmaze rephrased method-relative;
  rank table gained @1M row (93.1/94.2/94.6, d5=14.7); param confound (2.06×) in caveats.
TODO when hiqlmech finishes: refresh the three ° cells @500k, rebuild PDF. Then: 8-seed
1M confirmation of E-C/E-D winners for external claims; scene Schaul-form probe.

**Gap campaign (tmux `sotagap`, launched 2026-07-06) — COLLECTED into the tex above:**
`scripts/launch_sota_bilinear_gaps.sh`. Cells: (1) scene GCIQL bilinear-d50+DDPGBC
(new config `scene_state_gciql_bilinear_d50_ddpgbc`, group `GCIQL-Scene-Bilinear-d50-DDPGBC`);
(2) HIQL sg=10 matrix — native-AWR {MLP,bilinear} (new configs
`cube_single_state_hiql_native_{mlp,bilinear_d50}_sg10`) + flow {MLP,bilinear}
(staged `..._hiql_flow_both_{mlp,bilinear}_d50_sg10`). Groups: `HIQL-State-Native-{MLP,Bilinear-d50}-sg10`,
`HIQL-State-Flow-both-{MLP,Bilinear-d50}-sg10`. NEXT: collect @500k, update Table 1
(HIQL rows + scene cell), rebuild PDF. Watch: does bilinear-sg10 beat published 15?
Still open after this: HIQL/GCIVL on antmaze/scene, pixel bilinear. Code UNCOMMITTED.
