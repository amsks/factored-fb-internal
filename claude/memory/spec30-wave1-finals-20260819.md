---
name: spec30-wave1-finals-20260819
description: "2026-08-19 spec30 wave-1 FINALS (all 500k, 1 seed): centered point .824 / centered DUAL through the real reward-sample API .816 (≈ uncentered .812, vs BC .111, g .93); paper-exact TD InfoNCE cd .576 and same-φ LSIF cd .300 both well below soft density .736 ⇒ decision-tree strongest-outcome branch holds pending G29-3"
metadata: 
  node_type: memory
  type: project
  originSessionId: 973481bf-5a6b-4ead-8080-c40f063e1e9d
  modified: 2026-08-23T20:31:16.382Z
---

Measure track. Runs: ~/exp_spec30_p1 (ra2_cs_centpoint .824, ra2_cs_centdual
.816 — the centdual eval goes through the FROZEN M=256 reward-sample pool per
the no-silent-shortcut rule, so this is the FIRST rollout of the general
reward-sample encoder path, not the point shortcut) and ~/exp_spec30_i23
(tdi_cd_exact_s0 .576, lsif_cd_phi_s0 .300).

**Readings (all 1 seed, 500k finals — quote with mean≥350k when writing up):**
- Centered encoder ≈ uncentered (.824/.816 vs .812) ⇒ per spec-30 gate, SHIP
  CENTERED (principled: exact B+b / r+β / αr invariance); .812 becomes an
  ablation row.
- Sample-API parity with point prompting on goal kernels ⇒ the reward-sample
  interface WORKS for goal rewards; what remains for the general-reward claim
  is exactly G29-3: unseen families (mixtures, target−λ·avoid, conjunctions,
  negatives, SHIFTED r+β), pool-size curves M∈{256,1024,16384}, search-over-z
  oracle — ALL EVAL-ONLY on the ra2_cs_centdual ckpt now on disk.
- Paper-exact TD InfoNCE cd .576 < hard .680 < soft .736 ⇒ the estimator alone
  does NOT reproduce the result (1 seed).
- Same-φ LSIF cd .300: privileged-φ confound QUANTIFIED — plain FB .005 → .300
  from φ+external readout alone, .300 → .736 from the categorical estimator.
  Both the "φ matters" concession and the "normalization/CE still carries the
  bulk" claim survive.
- Decision-tree (spec 30 §9) branch 1 ("strongest outcome") holds at 1 seed
  pending G29-3 + promotion seeds.

**⚠ G29-3 attempt 2 (discovered 2026-08-23): CONTROL GATE FAILED, script
correctly hard-aborted, NO family cells read** (results/spec30_g293/gate.json
only): native re-roll .804 (noise around .816, ok) but the STANDALONE
harness's reward-sample path .704, task 3 collapsing .06 vs .48 native
(z_norm √d correct). Diagnosis candidate: the arm's native eval encodes ẑ
through its FROZEN internal M=256 pool; the harness built its OWN pool ⇒ the
sample encoder is POOL-COMPOSITION SENSITIVE (kernel×pool-density base
measure, reviewer §2.4). Next: rerun the gate encoding through the ckpt's
own `ZEvalPoolState` pool (or quantify ẑ(pool_A) vs ẑ(pool_B) cos per goal —
task 3's goal is where composition matters most). The watcher failed to
notify AGAIN — both G29-3 attempts died silently; treat g293 watchers as
unreliable, poll results/spec30_g293/ directly.

**Next:** fix the gate's pool provenance → rerun G29-3; then 3-seed
promotion of centdual + cd; results_map entries for ra2_*/tdi_*/lsif_*.
