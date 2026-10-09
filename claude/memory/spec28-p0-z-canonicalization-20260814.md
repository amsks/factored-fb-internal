---
name: spec28-p0-z-canonicalization-20260814
description: "2026-08-14 spec28 P0 COLLECTED — the z-interface diagnosis is SOLVED and REPAIRABLE. P0a: z(g) is COLLAPSED (pairwise cos .998-1.000 vs sphere ref .002) so the actor is exactly goal-invariant (sens 0.000 vs noise .19) while a random z moves it 1.9-2.0; also overturns spec24's 'no collapse' (z_cos_spread .35 = FULLY-COLLAPSED prediction .326 after 50% sphere dilution, spread would be .113). P0b: ridge (dual) ≫ point/moment on reward reconstruction (AUC .99-1.00 vs .43-.87). P0c (gate PASS 14/14 bit-exact): ridge moves reward→index assignment from CHANCE (.35-.51) to g-anchor level (.000-.008). G28-1 PASSES ⇒ R-A licensed"
metadata:
  node_type: memory
  type: project
  originSessionId: 576b53a0-d404-4a89-9380-6fece089b102
  modified: 2026-08-14T20:18:38.789Z
---

Measure track. Spec `PAPER/Multistep-FB/specs/28-z-interface-canonicalization.md`.
Closes the diagnosis opened by [[spec24-z-index-20260813]] /
[[spec25-z-coupling-20260813]] / [[spec27-q2-z-readout-20260814]].

**The mechanism (external feedback, audited CORRECT and NEW).** Density-FB's
softmax kills FB's linear identity `Q_r = Fᵀ E[rB]`, so `z_r = E[r·B]` is no
valid reward→index inverse; and CE pins only the PRODUCT `FᵀB`, leaving B's
Euclidean geometry gauge-arbitrary. `z(g) = √d·normalize(B̄(g))` is a **PRIMAL**
feature vector while the readout `B̃(u)ᵀz/√d` needs a **DUAL** coefficient
`c = (Σ_B+λI)⁻¹m`. They coincide only under the whitened geometry FB's ortho
loss enforces — which this project dropped (and keeps dropped: ortho is a
convicted toxin, first-passage .967 no-ortho vs .225 with).

**★ P0a — z(g) IS COLLAPSED (`eval_z_action_ranking.py`).** Over 4096 dataset
goals: mean pairwise **cos .998-1.000** vs a sphere reference of **.002**;
median distance .05-.48 where √(2d)=10. Consequences measured:
- z-actor is EXACTLY goal-invariant: `E‖π(s,z)−π(s,z′)‖` = **0.000** (4e-6…5e-4)
  vs its own noise floor **.19** — ~1000× below;
- a RANDOM-sphere z′ moves the C3/Full actors **1.9-2.0** (ceiling 1.83) ⇒ the
  actor is z-CAPABLE, it is simply never handed two distinguishable z's;
- latent readout span across actions `relspan_z` **.000-.002** vs kernel
  `relspan_g` **.13-1.15**; ρ(q_g,q_z) at the null (.01-.31).
⚠ **Both effects appear on the g-INDEXED anchors too** (cos .998/.999,
relspan_z .001) ⇒ property of the encode map and of `B̃ᵀz`, NOT of z-training.
Verdict = branch 1, READOUT indicted; cell (iii) convicted at eval time.
Control gate: cs/cd PASS, **pmm FAIL** (AUC_g .586<.65 — RND data has no
goal-directed actions, so the 'good action' label is invalid there; pmm cells
recorded UNLICENSED and not read).

**⚠ OVERTURNS spec 24's "no index collapse".** `z_cos_spread` is computed on the
MIXED index (50% random spheres), so a fully-collapsed hindsight branch dilutes
to **.326** and a fully-spread one to **.113**. Observed .346-.350 matches the
COLLAPSED prediction. The instrument could not see what it was built to detect.
Related: the spec-24 zid probe's `nearest_wrong_cosine ≈ 1.0000` (dismissed then
as "saturated, uninformative") WAS this finding. Retrieval top-1 .74-.79 stays
literally true — it is RANK-based and a tight cone preserves ordering — but it
does NOT mean the index is usable for control.

**P0b — reward reconstruction from the pool (means over cells, target tower):**

| ckpt | metric | point | moment | ridge .001/.01/.1 | oracle | random |
|---|---|---|---|---|---|---|
| g_cd | AUC | .871 | .867 | **1.000/.999/.999** | 1.000 | .501 |
| g_cs | AUC | .484 | .480 | **.994/.991/.990** | 1.000 | .516 |
| z24_cd s0/1/2 | AUC | .81-.86 | .81-.86 | **1.000** | 1.000 | .50-.57 |
| z25_cs (4 arms) | AUC | **.43-.57** | .42-.55 | **.99-1.00** | 1.000 | .48-.55 |
| z25_cs (4 arms) | spearman | **−.05-.28** | same | **.58-.72** | 1.000 | −.07-−.01 |

`moment ≈ moment_raw ≈ point` everywhere ⇒ the issue is PRIMAL-vs-DUAL, not
scaling. On cube-single the point encoding is **at or below chance** for reward
reconstruction while ridge is near-perfect. **GATE G28-1 PASSES ⇒ R-A licensed.**

**★★ THE MECHANISM, QUANTIFIED (verified from the P0b JSONs).** `Σ_B` has
**effective rank 1.0002 out of d=50** on every checkpoint, cond **1e6-1e7**
(z25_cs_actor: eig_max 1.18e6, eig_min 0.115). B is rank-1-DOMINATED, nowhere
near whitened. So the primal moment `E[rB]` is almost entirely the dominant
direction and carries nearly no reward-specific information, while the dual
`(Σ+λI)⁻¹m` divides that direction out and recovers the reward component.
Measured: **cos(primal moment, dual ridge) = .010-.025** — the two indices are
very nearly ORTHOGONAL. (The agent's own report said .03-.15; the artifacts say
.010-.025, i.e. even more extreme.) This is why dropping ortho was safe for the
g path and fatal for the z path: the g path never reads reward off `Bᵀz`.

**★ P0c — assignment repaired (control gate PASS, 14/14 cells BIT-EXACT vs the
archived spec-24 e2b JSON).** Metric = coordinate percentile, **0 best, .50
chance, LOWER BETTER**, convention pinned by reproducing spec-27's .35/.47:

| ckpt | current readout | moment | ridge .001/.01/.1 |
|---|---|---|---|
| z24_cd s0/1/2 | .506/.431/.489 (chance) | .000-.001 | **.000-.001** |
| z25_cs_actor | .347 | .020 | **.007-.008** |
| z25_cs_full | .473 | .062 | **.002-.004** |
| g anchors cs/cd | .008/.000 | .013/.000 | .011-.014/.000-.002 |

**The dual correction moves reward→index assignment from chance to g-anchor
level.** ⚠ This is READOUT-ONLY. `‖z‖=√d` wherever z enters F/actor and a ridge
coefficient is NOT norm-preserving, so CONDITIONING on it is OOD on existing
checkpoints and needs retraining. Licenses BUILDING R-A, not an end-to-end claim.

**P1 2×2 status: knobs BUILT, tested (8/8 + 31/31 neighbours, bit-identity over
3 real update() steps + sample_actions, mutation-tested), NOT LAUNCHED** —
`measure_density_z_actor_reward ∈ {latent,kernel}` (cell ii),
`measure_density_actor_reward ∈ {kernel,latent}` (cell iii),
`scripts/rescuing_fb/launch_spec28_p1.sh` carries a DO-NOT-RUN gate header.
**Neither cell is fundable as written**: ii conditions on the collapsed index so
no readout can make it goal-specific; iii is convicted by P0a on g-anchors.

**Whitened-index variant (estimated, not implemented):**
`measure_density_z_encode ∈ {sphere, whitened}`, ~60-80 lines + ~120 test lines.
The cost is the STATISTICS, not the transform: μ_B and Σ_B must be EMA-tracked
as ordinary param leaves (the `density_b_zenc` pattern) so they serialize and
`sample_actions` reads the identical object — batch stats would reintroduce the
train/eval mismatch that broke the first z build. Risks: ‖z‖=√d must be
re-imposed after whitening, so whitened CONDITIONING and whitened READOUT are
two different cells needing two knobs; Σ^{-1/2} through a drifting B tower is
non-stationary (needs λ and probably `enc_tau>0`); whitening amplifies whatever
shares the goal-varying subspace including noise; and it is a NEW ARM, so
archived z24/z25 numbers stop being comparable.

Related: [[spec27-q2-z-readout-20260814]], [[spec25-z-coupling-20260813]],
[[spec24-z-index-20260813]], [[primary-paper-dir-multistep-fb]].
