---
name: zeroshot-gciql-e1a-works
description: E1a DONE — zero-shot GCIQL WORKS; dataset-based moment inference v0=0.955=exact; the G4 93→7 gap was an on-policy sampling artifact; whitening refuted; E1b not needed
metadata: 
  node_type: memory
  type: project
  originSessionId: 0c042022-8040-4720-98dd-c66e745e506e
---

**E1a (zero-shot GCIQL) resolved 2026-07-07 — the "inferred z→0.07" G4 dissociation
was a SAMPLE-DISTRIBUTION artifact, not the formula or the actor.** Zero-shot GCIQL
works: the ~0.90 IQL critic recovers full exact-goal success from FB's API
(reward samples → z → policy, no fine-tune).

Script: `scripts/eval_zeroshot_inference.py` (unifies+extends eval_zcond_gciql.py /
infer_ablation_gciql.py). Results: `analysis/eval_zeroshot_inference/s{3,4,9}.json`.
G4 zcond @1M checkpoints: `~/bridge_ckpt_backup/gciql_outputs/factored-fb/gciql_g4_zcond/sd00{3,4,9}_*`
(d=512, actor_cond=z). 3 seeds × 5 tasks × 25 ep.

| rule (mean over 3 seeds) | success | cos to psi(g) |
|---|---|---|
| v2 exact z=psi(g) (anchor) | **0.955** | 1.0 |
| v0 moment z=E[r·psi], **match_g norm** | **0.955** | 0.882 |
| v0 moment, sqrtd norm (spec's choice) | 0.669 | 0.882 |
| v1 whitened (GLS), ridge=1.0 | 0.008 | 0.479 |

Three load-bearing facts:
1. **v0 dataset-based moment = exact (0.955 both).** Key change vs the old 0.07: reward
   samples come from the OFFLINE singletask dataset (`train_ds['rewards']+1` = relabeled
   indicator over dataset states, no privileged qpos needed) instead of on-policy rollout.
   On-policy states came from a badly-z-conditioned policy → few goal-reaching positives →
   degenerate z. Dataset has proper goal-reaching coverage → z=E[r·psi] concentrates on
   psi(g) (cos 0.88), and the actor is robust to that 0.88-cos vector.
2. **Normalization matters ~0.29:** actor was trained on RAW psi(g) (value net does NOT
   sphere-project unless project_z=True, which G4-zcond isn't), so ||psi(g)||≈41 ≠ √512≈22.6.
   Rescaling z to ||psi(g)|| (match_g) = 0.955; the spec's √d = 0.669. Testing BOTH norms
   caught this; the spec would have under-reported zero-shot at 0.67.
3. **v1 whitening REFUTED:** G=E[psi psi^T] is rank-deficient (cond ≈ 7e14, min eig 1.8e-13),
   so (G+εI)^-1 injects noise; v1 monotone-approaches v0 from below as ridge→∞, never beats
   it (cos ≤0.51 even at ridge=3). The spec's "correct dual vector" hypothesis fails here.

**Consequence for the spec (`docs/superpowers/specs/2026-07-06-zeroshot-and-interpolation-design.md`):**
E1b (inference-consistent actor training) is NOT load-bearing — v0 already hits the exact
anchor. E2 (scale-controlled IQL+measure interpolation) remains the other open question.
Supersedes audit-20260706 §4 open item "inference-consistent conditioning untested" and the
untried-doc's G4 gap. See [[audit-sf-measure-20260706]], [[sota-bilinear-paper]].

Ops note: green SF checkpoint (fb_forward_z + DDPGBC-SF + 8-cell configs, 11/11 tests) was
committed 2026-07-07 as `7b405bf` on branch gciql-fb-bridge before this E1a work.
