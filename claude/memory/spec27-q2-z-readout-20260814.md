---
name: spec27-q2-z-readout-20260814
description: "2026-08-14 spec27 Q2 ANSWERED: the C3 (index-sensitive) z-measure's READOUT IS CALIBRATED — G4 spearman_tie .728/.713 vs g-indexed controls .739 (cd) / .778 (cs), with LOWER degenerate_frac. So index-sensitive-but-unextractable does NOT mean unreadable. New decisive diagnostic: e2b's reward→index argmax lands at CHANCE (percentile .35/.47 vs .50 null) despite exact hits in the candidate set ⇒ the break is REWARD→INDEX ASSIGNMENT, not the measure"
metadata:
  node_type: memory
  type: project
  originSessionId: 576b53a0-d404-4a89-9380-6fece089b102
  modified: 2026-08-14T06:49:27.230Z
---

Measure track. Spec 27 Q2 (`PAPER/Multistep-FB/specs/27-implementation-queue.md`).
Follows [[spec25-z-coupling-20260813]] (C3 revived index-sensitivity, ztv .602,
success still at the BC floor) and [[spec24-z-index-20260813]].

**Q2 asked:** does an index-sensitive-but-unextractable measure still support
reward readout, or is it unreadable too?

**⚠ E2 CANNOT ANSWER IT — third time this trap has appeared.** E2a/E2b score
`success_final`, a pipeline ending in an actor rollout, and the z25 actor is dead
(.108 vs .93 anchor). E2 read .040-.088 sparse/dense and ~0 region against
cs anchors .816/.725/.293 (soft) and .888/.912/.647 (hard) — which looks like
"unreadable" and is actually the dead actor showing through. Floor context: 125
episodes, granularity .008, so those cells are 5-11 successes, at/below the BC
floor .111. **The control gate passing does NOT make a pipeline metric able to
isolate a component** — the gate was bit-exact (diff 0.000 vs archived, on BOTH
a cd and an env-matched cs control) and the metric still could not answer.

**The actor-independent instrument settles it** (`eval_measure_g4_spearman.py`,
gate `spearman_tie` ≥ .65, verified from the JSONs myself):

| ckpt | pooled | degenerate_frac | mc>0 |
|---|---|---|---|
| cs control (g) | .778 | .606 | .801 (n=340) |
| cd control (g) | .739 | .635 | .666 (n=315) |
| **z25_cs_actor** | **.728** | **.463** | .718 (n=464) |
| **z25_cs_full** | **.713** | **.503** | .704 (n=429) |

Both z arms pass, sit BETWEEN the two g controls, and have LOWER degenerate_frac
than either ⇒ not the frozen-trivially-predictable-policy artifact (checked
explicitly, as on the spec-24 round). **The readout is calibrated. Extraction is
the sole remaining z blocker.**

**★ THE SHARPEST NEW FINDING — reward→index assignment is at CHANCE.** A
secondary diagnostic on e2b's argmax (coordinate percentile of the selected
z-candidate vs the reward center, 14 cells): **.35 (actor) / .47 (full) against a
.50 null**, *despite exact hits existing in the candidate set*. Combine with what
was already known and the localization is tight:
- index ENCODING z(g) works (spec-24 zid, top1 .74-.79 vs ceiling .818, chance .0005)
- measure is index-SENSITIVE (spec-25 ztv .602)
- measure is CALIBRATED (this note)
- reward→index ASSIGNMENT is at chance ← **the break**

z(g) (goal→z) and z_r (reward→z) are DIFFERENT maps; the first works, the second
does not. z_r ∝ E[r·B] is exactly FB's headline interface and exactly what the
user's binding general-reward-interface requirement needs.

⚠ Standing caveat: G4 rolls the agent's OWN policy, so "calibrated" = faithfully
predicts its own policy's occupancy (M^π models π). It rules out a broken
measure; it is not proof a better policy is extractable. The chance-level
assignment result is the load-bearing evidence, not the calibration alone.

**Fork:** fix reward→index assignment, or bypass z with the reward-embedding
actor (w_r conditioning, spec-19 extension). The latter serves the binding
requirement directly.

Related: [[spec25-z-coupling-20260813]], [[spec24-z-index-20260813]],
[[spec18-density-measure-20260806]], [[primary-paper-dir-multistep-fb]].
