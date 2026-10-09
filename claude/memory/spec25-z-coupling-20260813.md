---
name: spec25-z-coupling-20260813
description: "2026-08-13 spec25 z-coupling ports (C1 tilt / C2 rdecode / C3 actor) IMPLEMENTED (17/17 tests) + stage-1 cube-single screens RUNNING. P0 closed: B-geometry fine on dense (B .85 g / .68-.70 z, beats a dim-matched RBF control .466) so C2 de-prioritized; sparse/region UNINFORMATIVE (RBF control also ~.02). Early density_w_ztv says C3 >> C1: actor .587 stable vs tilt .028 and dec .012 both DECAYING"
metadata:
  node_type: memory
  type: project
  originSessionId: 576b53a0-d404-4a89-9380-6fece089b102
  modified: 2026-08-13T19:38:47.627Z
---

Measure track. Spec `PAPER/Multistep-FB/specs/25-z-coupling-port.md`. Follows
[[spec24-z-index-20260813]] (z-extraction fails; measure + index both healthy).

**The idea.** FB couples index to behavior three ways spec 24 never ported.
C1 structural logit tilt `β·zᵀB(ũ)/√d` (β=1.0); C2 ortho-free reward-decodability
ridge auxiliary; C3 actor-side (`bc_hindsight` mask + slow-EMA index encoder
`enc_tau=5e-4`). All default-off, four knobs, plus a NEW gate instrument
`density_w_ztv` = E[TV(w(s,a,z), w(s,a,z'))], the direct index-sensitivity meter
the spec-24 wave lacked.

**Implementation VERIFIED** (17/17: 8 new + 9 spec-24 regression, run
independently). One spec deviation is CORRECT and worth remembering: spec says
mask the *flow-BC* term for C3, but under this lattice `bc_flow_loss` reads
`vf_obs`, which only carries z when `actor_vf_cond='z'`, which requires
`actor_cond='z'` (line ~3358), which `measure_density` refuses (line ~3381). So
flow-BC provably cannot see z and **distillation is the only z-carrying imitation
term** — that is what got masked. λ=1e-4 is a module constant, not a 5th knob.

**P0 CLOSED (eval-only, cube-double ckpts).** Ridge-reconstruction R²:

| family | B (g) | B (z) | raw-linear | RBF 50-dim |
|---|---|---|---|---|
| dense | .853 | .68-.70 | .247 | .466 |
| sparse | .030 | .025 | .002 | .027 |
| region | .019 | .015 | .002 | .016 |

Dense ⇒ spec's branch (b): **B-geometry is fine, C1/C3 carry the failure, C2
droppable if its solo arm is flat.** ⚠ sparse/region are **UNINFORMATIVE, not
negative** — a dim-matched multiscale RBF bank also reads ~.02 (bandwidth sweep
.25-8.0 confirms), so NO generic coordinate feature map decodes them at P=16384
and the near-zero B numbers there say nothing about B. **The spec's own control
spec was wrong**: it asked for a raw-coordinate baseline "known high on dense",
but a Gaussian bump is not linear in 6 coords, so that arm could never clear its
gate. Enriching the control was required to tell (a) from (c).

**★ STAGE-1 VERDICT AT 500k (tilt/dec/actor final; full at 400k tracking identically).
GATE S1 FAILS IN EVERY ARM, and the way it fails is the result:**

| arm | succ@500k | density_w_ztv | S1 ztv | S1 E1≥.5 |
|---|---|---|---|---|
| tilt C1 | .096 | .054 | fail | FAIL |
| dec C2 | .108 | .009 | fail | FAIL |
| **actor C3** | .108 | **.602** | **PASS** | **FAIL** |
| **full** | **.108** | **.588** | **PASS** | **FAIL** |

(all four FINAL at 500k)

cube-single BC floor = **.111**, g-indexed anchor = **.93** (which reaches .792 by
50k). Every arm sits AT the BC floor for all 500k. **C3 fully restored index
sensitivity (ztv .602, ~65x the C1/C2 level) and bought EXACTLY ZERO success.**
⇒ **Index-coupling is not the bottleneck; making the measure use its index is
necessary-at-best and demonstrably not sufficient.** The pre-registered NULL
("all four flat on ztv") does NOT fire literally — C3 is emphatically not flat —
but its program consequence holds by a stronger route: coupling SUCCEEDED and
extraction still failed, so stop iterating on index coupling. Spec's own fallback
is now the live route: **reward-embedding actor (w_r conditioning, spec-19
extension)**, which also directly serves the user's binding general-reward-
interface requirement.

**⚠ THE EARLIER FORMING READ (through 300k, all 4 arms + comparator): a clean
DISSOCIATION — index-coupling succeeded, extraction did NOT follow.** C3 has the
HIGHEST index-sensitivity (ztv .59-.63) and among the LOWEST success (.044-.096).
Every z arm is FLAT at the BC floor (.06-.11) through 300k while the g-indexed
comparator on the SAME env hits **.792 by 50k** and .91+ by 150k — so these arms
are dead, not slow (same signature as spec-24 z on cube-double .017). ⇒ **making
the measure index-sensitive is NOT sufficient; index-coupling is not the
bottleneck.** Spec 24 could not have seen this: `density_w_ztv` is what
distinguishes "measure ignores z" from "measure uses z and extraction still fails".
⚠ The pre-registered NULL ("all four screens FLAT on ztv") will NOT fire
literally — C3 is emphatically not flat — but the program conclusion it guarded
(index-coupling is not the route) holds via a different path. Spec's own fallback
then applies: the **reward-embedding actor (w_r conditioning, spec-19 extension)**
becomes the live route to the user's binding general-reward-interface requirement.
Let all 4 run to 500k anyway; the gate is defined there and mid-run reads have
burned this program before.

**Stage-1 early read (50k, cube-single, 1 seed each):**
`density_w_ztv` actor **.587 and stable** vs tilt **.028** and dec **.012**, BOTH
DECAYING from ~.09/.07 at 40k. ⇒ **C3 carries index-sensitivity; C1's structural
guarantee is NOT holding at β=1.0** (spec claimed F "can reshape around the tilt
but not delete it"; it appears to be deleting it, or β is too small vs the base
logit scale). Success has NOT separated (.084/.064/.112 vs BC floor .111).
⚠ 10% of training — direction only, not a result; gate S1 needs ztv away from 0
AND E1 ≥ .5 at 500k. z_cos_spread .34-.35 everywhere (index never collapses).

**Ops.** All 4 cells RUNNING on GPUs 1-4 as of 17:38 (`z25_cs_full` launched when
a slot briefly freed, so the block cleared itself). Three redundant jobs
(`hx_cd_hard_s0`, `dbx_cd_soft_s0`, `dbx_cd_hard_s2`, all stuck at ckpt 100k,
all stacked on GPU 0) hold slots while their `_re` twins already finished 500k
(.652/.676/.572 — the values now in tab:main). Killing them was **blocked by the
permission classifier**; needs the user. Don't kill the dispatcher to raise MAX —
the signal propagates to its live children.

Related: [[spec24-z-index-20260813]], [[spec18-density-measure-20260806]],
[[spec22-antmaze-null-20260813]], [[primary-paper-dir-multistep-fb]].
