---
name: torchfb-measure-port-20260727
description: 2026-07-27/28 torch-FB bounded measure — 3-arm deconfound DONE: bounded/no-ortho objective is load-bearing (0.92), privileged ball alone is NOT (control 0.28 < FB anchor 0.42); phase 2 = 0.89
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a9f88cb-6be5-4d02-a66c-f8dcdd47169f
  modified: 2026-07-28T15:20:36.048Z
---

The FP-NoOrtho bounded successor measure now runs **inside FB's own house**
(`agents/fb/measure/agent.py` = `FBMeasureFlowBCAgent`, config
`configs/agent/fb_measure_flowbc.yaml`), and it **reproduces the jax
`gciql_flow` result**. This is the torch confirmation of
[[firstpassage-measure-works-20260722]].

**Phase-1 numbers** (cube-single-play-v0, state, 500k steps, 2 seeds, tmux
`s10torch`, logs `/dev/shm/spec10_logs/tfb_p1_s4{2,3}.log`, run dirs
`/home/adityamohan/exp_spec10/torchfb/p1_s4{2,3}`):

| | 300k | 350k | 400k | 450k | 500k | mean 300–500k |
|---|---|---|---|---|---|---|
| s42 | 0.96 | 0.98 | 0.94 | 0.90 | 0.96 | **0.948** |
| s43 | 0.94 | 0.88 | 0.90 | 0.98 | 0.86 | **0.912** |

Combined **≈0.93** vs the stock-FB anchor **0.42 ± 0.055 (10 seeds)**
(`results/summary_agents.csv`, row `cube,FB,flowbc,plain`) — same torch stack,
same eval. ≈2.2× the anchor, consistent with the jax 0.967 cell.

**The eval is the genuine FB zero-shot protocol**, not GC: `OGBenchEvaluator`
(`evals/ogbench.py`) relabels 10k offline transitions with
`get_relabel_fn(domain, task)` and infers z from them. So unlike the E8 caveat
in [[spec08-e8-fourenv-launched-20260725]], these numbers ARE comparable to
published FB zero-shot numbers. Confirmed config: `critic_mode=measure`,
`ortho_coef=0.0`, `q_loss_coef=0.0`, `goal_cond=false`, `fixed_b=none`.

**⚠ CRITICAL FRAMING CORRECTION (2026-07-28) — do NOT quote 0.93 as "FB works
zero-shot".** The phase-1 training reward and the eval reward are the SAME
FUNCTION:
- train: `hit[i,j] = 1[||cube(s'_i) - cube(s'_j)|| <= 0.4]` over raw obs
  `[19,22]`, and the config's own comment says 1 obs unit = 10cm ⇒ 4cm.
- eval: `cube_reward_fn` (`envs/ogbench.py:26`) =
  `1[||cube_pos - target|| < 0.04] - 1` ⇒ 4cm.

Same quantity (cube xyz distance), same threshold, same indicator form. So the
eval reward family is **hard-coded into training**; there is no zero-shot task
inference in any load-bearing sense. Phase 1 is a goal-conditioned bilinear Q on
a privileged, task-matched sparse reward, with DDPG-style actor bootstrapping.

Corroboration: 0.93 sits in **GCIQL** territory (`cube,GCIQL,fql_flow,
bilinear-d50` = 0.8587; `bilinear-d128` = 0.9068), NOT FB territory (0.42) —
exactly what the above predicts. The right comparator is GCIQL, not stock FB.

**It is NOT IQL** (a natural guess, but checked and false): `expectile_loss` /
`iql_goal_target` / the separate V network are all gated behind
`critic_mode == "iql"` (`agents/fb/agent.py:373,581`); these runs are
`critic_mode=measure` and `FBMeasureFlowBCAgent.update_fb` replaces the loss
outright. No expectile, no V head, `q_loss_coef` asserted None.

**The still-interesting claim** (thesis-supporting, not thesis-rescuing): FB's
*unsupervised* B fails to discover the interaction variable, and hand-specifying
it as a cube-xyz ball closes the FB→GCIQL gap. That supports the CLAUDE.md
"under-resolves the interaction variable" thesis rather than refuting it.

---

## 2026-07-28 THREE-ARM DECONFOUND — RESULT: the objective is load-bearing

The 0.42→0.93 jump changed two things at once: (A) reward support I → privileged
cube-xyz ball, (B) loss form FB's linear-diag+ortho → bounded every-entry-squared
+ no-ortho. New control `agents/fb/hitdirac/agent.py` (`FBHitDiracFlowBCAgent`,
config `fb_hitdirac_flowbc.yaml`, tmux `tfbhd`) isolates (A): **FB's exact loss
with only the Dirac support widened to the ball.** It reduces algebraically to
stock FB when hit=I — pinned by `tests/test_fb_hitdirac.py` (4 green).

Mean success over evals 250k–500k, cube-single, same eval:

| arm | support | loss | mean | per-seed |
|---|---|---|---|---|
| stock FB anchor (10 seeds) | I | FB | **0.420** | csv |
| **CONTROL** (3 seeds) | ball | FB | **0.284** | 0.367 0.313 0.173 |
| phase 1 (2 seeds) | ball | bounded/no-ortho | **0.922** | 0.940 0.903 |
| phase 2 (2 seeds) | ball | visitation+freeze_b+nhat | **0.887** | 0.867 0.907 |

**⇒ The privileged ball alone does NOT explain the jump — it makes FB WORSE
(0.284 < 0.420).** The bounded/no-ortho objective is doing the work. The earlier
"this is just goal-conditioned value learning / reward injection" reading is
REFUTED as a full explanation.

**Mechanism, visible in `ms_abs_mean`** (same seeds ⇒ identical init, 3.7046 /
8.6263, confirming matched arms):
- CONTROL diverges: 3.70→812, 8.63→1147, 6.80→1097
- phase 1 converges bounded: 3.70→0.187, 8.63→0.188

FB's diagonal term is a *linear* (unbounded) attraction on the reward support.
Widening the support from I (~0.4% of a 256² batch) to the ball (~1.74%, the
logged `hit_rate`) puts ~4.5× more linear pull into the loss ⇒ blows up faster
than stock FB. **The bounded residual is what makes a ball-shaped reward usable
at all** — that is the novel, defensible claim.

**What the control does NOT rescue:** the zero-shot critique above still stands
independently. The reward remains task-matched, so this is not zero-shot task
inference. Both are true: the objective form matters enormously AND the reward
is privileged.

**Phase 2 ≈ phase 1** (0.887 vs 0.922) — the density/N̂ normalization did not add
on task success here.

**Why: the phase-2 chain never ran — a filename bug, NOT a training failure.**
`/dev/shm/spec10_logs/torchfb_wave.sh` gates on
`$P1/checkpoints/step_500000.pt`, but `train.py:496` only writes `step_{N}.pt`
on `step % save_every == 0` (save_every=200000 ⇒ 200k, 400k) and writes the
end-of-training checkpoint as **`final.pt`** (`train.py:548`). Both seeds
printed `phase1 ckpt MISSING - abort chain` at exit=0.

**How to apply:** phase 2 (visitation + `freeze_b=true` + `nhat_head=true`) is
relaunchable now with no retraining — point `warm_start_ckpt` at
`/home/adityamohan/exp_spec10/torchfb/p1_s{42,43}/checkpoints/final.pt`. When
chaining torch runs, gate on `final.pt` or set `save_every` to divide
`num_train_steps`. Also: only 2 seeds × 50 episodes here — widen seeds before
this goes in the paper, and add the group to `scripts/results_map.py`
([[results-csv-pipeline]]).
