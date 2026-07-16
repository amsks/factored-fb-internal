# Why does bilinear GCIQL + flow win? — the mechanism program

*Spec, 2026-07-16. Follows `oracle-free-findings.md` (Findings 1–5) and the value-vs-advantage
decomposition. Code: `scripts/atlas/`; results: `analysis/atlas/`.*

## Where we are

The decomposition established a **value-geometry vs advantage-geometry split**: the MLP critic's
in-hand state ranking lives entirely in its return-magnitude baseline `U=E_μQ` (AUC 0.82);
bilinear's `U` is anti-calibrated (0.40) and uncorrelated with the MLP's (Spearman −0.04), while
the two critics' **advantage** orderings partially agree (+0.22). "MLP fits return magnitude" is
confirmed. The strong clause — "**bilinear organizes policy improvement**" — is **open**: the
causal action-ranking test (advantage vs realized discounted return under a shared frozen
continuation) returned a low-power null because a competent continuation washes out the first
action (Finding 5, causal subsection).

**Working conclusion to be tested:** bilinear GCIQL + flow is a *compatible policy-learning
system*, not a better value estimator. The bilinear bottleneck supplies a low-dimensional
goal-conditioned energy over actions; the flow actor projects it onto the competent behavior
manifold, shaping a strong policy despite poor absolute value calibration. Direct DDPG+BC leaves
that manifold and chases unsupported bilinear peaks (Finding 1). Cube is favorable because its
behavior data already contains competent pick-and-place and the required goal-conditioned
improvements are low-dimensional and reusable; the bias should become inconsistent when the
behavior prior lacks the needed actions, when improvement is entangled/hierarchical, or when
calibrated value magnitude matters.

This program tests that conclusion. Experiments are ordered by value/cost; **Exp 1 is the lead
(pure inference, no training, no branching)** and is running now.

---

## Exp 1 — Behavior-sample reranking  *(DONE — bilinear wins decisively)*

**Result (25 eps/task, N=32, C3 protocol):** bilinear-rerank **0.960** vs MLP-rerank **0.592**
(μ=bilinear candidates); **0.912** vs **0.568** (μ=MLP candidates — robust to candidate source);
μ-floor 0.06. Bilinear-rerank **≈ its trained flow actor (0.947) with no actor training**;
MLP-rerank (0.59) < MLP's trained actor (0.747). ⇒ **the final bilinear critic's action ordering
is intrinsically better** (not actor-training dynamics); the flow actor does real work for MLP,
almost none for bilinear. This is the powered confirmation of "bilinear organizes policy
improvement" that the causal test's wash-out denied. See `oracle-free-findings.md` Finding 6.



**Method.** Deploy, per critic `m`, the reranking policy: at every env step sample
`a_1..a_N ∼ μ(·|s)` (frozen BC flow) and execute `a* = argmax_i Q_m(s, a_i, g)`. Roll out on the
5 eval tasks from `reset()` (OOD start, H=200, success = cube within 4 cm) — the C3 protocol —
and measure success. Compare **bilinear-rerank vs MLP-rerank**, with anchors: the trained flow
actors (bilinear-flow C3 0.947, MLP-flow 0.747) and a μ-only (no reranking) floor.

**Interpretation.**
- *Bilinear rerank wins* ⇒ the **final** bilinear critic ranks supported (μ-sampled) actions
  better — the advantage is in the critic's action ordering, not only actor-training dynamics.
- *MLP rerank wins or ties* ⇒ the bilinear advantage comes from actor-training dynamics, not the
  final action ordering.

Files: `scripts/atlas/rerank_eval.py`, `analysis/atlas/rerank_eval.json`.

## Exp 1b — Reranking audit across environments (N-sweep)  *(cube DONE; antmaze/scene running)*

Deploy `argmax_{i≤N} Q_m(s,a_i,g)` over `N` μ-candidates as the policy, sweep
`N∈{1,2,4,8,16,32,64,128}` per environment, vs μ-floor / bilinear-rerank / MLP-rerank / trained
actors (`scripts/atlas/rerank_audit_multienv.py`, `analysis/atlas/rerank_audit_<env>.json`).
Separates candidate coverage vs ranking vs planning.

**Cube** (floor 0.067; trained bilinear 0.933, MLP 0.747):

| N          | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|------------|---|---|---|---|----|----|----|-----|
| bilinear   | .067 | .467 | .800 | .853 | .920 | **.947** | .907 | .853 |
| MLP        | .067 | .240 | .453 | .560 | .573 | .587 | .640 | .560 |

- **Bilinear: ranking is excellent, saturates early + HIGH** — reaches 0.95 by N=32 ≈ its trained
  actor (0.933), so argmax-over-behavior alone recovers the policy. Not coverage-limited.
- **MLP: ranking is the bottleneck** — saturates ~0.6 even at N=128, *below* its trained actor
  (0.747): for MLP, continuous action modification / training history matters.
- **Off-support signature in the sweep:** bilinear rerank *peaks at N=32 then declines* to 0.85 at
  N=128 — more candidates → more chances to hit the over-valued off-support action (Finding 1).

(antmaze/scene sweeps running; antmaze `trained_mlp` anchor N/A — a `load_agent` actor-shape
mismatch, reranking unaffected.)

## Arch 11 / idea #1 — Behavior-centered dueling GCIQL  *(TRAINING)*

`Q=U_MLP(s,g)+A_bil`, `A_bil=(φ(s,a)−b(s))·ψ(g)/√d`; `b(s)` a learned head trained by
`||b(s)−sg(φ(s,a_data))||²` (batch actions = behavior samples) so `E_μ[A_bil]≈0`; the expectile V
stays distinct; **the flow actor consumes only `A_bil`**. Implemented as `value_arch='dueling'`
(`GCDuelingValue` in `gciql_flow.py`, config `dueling_b_coef`). Smoke verified: `Q==U+A_bil`,
`dueling_b_loss` ↓ (b tracks φ), losses finite. Full 500k run training now (cube, GPU1,
`exp_name=dueling_d50`). Tests the decomposition constructively: if U (calibration) + A_bil
(policy geometry) + flow (support) ≥ stock bilinear, the split is confirmed by construction.

## Exp 2 — Fresh actor extraction from frozen critics  *(RESULT — bilinear field is intrinsically better)*

Two flow actors from fresh init; train one against the frozen bilinear critic, one against the
frozen MLP critic, no critic updates (masked optimizer). Cube, 1 seed, 500k, `eval.csv`.

| fresh flow actor vs | success | co-trained ref | Exp-1 rerank |
|---------------------|--------:|---------------:|-------------:|
| **frozen bilinear** (exp2a, @150k) | **0.88–0.89** | 0.947 | 0.91 |
| **frozen MLP** (exp2b, done, final) | **0.68** (curve 0.55–0.81) | 0.747 | 0.59 |

**Co-adaptation refuted.** A fresh flow actor against the frozen bilinear critic reaches ~0.88 vs
~0.68 against the frozen MLP — same ~0.2 gap and direction as the co-trained cells and the Exp-1
reranking. Neither actor shared training history with its critic, so the advantage is a property
of the **bilinear critic's learned field**, not actor–critic co-adaptation. (exp2a still training
to 500k; curve flat, number should hold.)

## Exp 3 — Fresh DDPG+BC vs frozen bilinear critic  *(RESULT — off-support peak is intrinsic)*

Fresh DDPG+BC (Gaussian) actor trained against the frozen bilinear critic. Cube, 1 seed.

| fresh actor vs frozen bilinear | success | co-trained ref | flow-actor ref |
|--------------------------------|--------:|---------------:|---------------:|
| **DDPG+BC** (exp3, @200k)       | **~0.52** | 0.557 | 0.88 (exp2a) |

**The off-support peak is intrinsic to the bilinear critic geometry.** A fresh DDPG+BC actor
against the frozen bilinear critic reaches only ~0.52 — matching the co-trained bilinear-DDPG+BC
(0.557) and far below the flow actor's 0.88. Even without co-training, a Q-ascending Gaussian
actor is pulled off-manifold to the bilinear critic's off-support peak (Finding 1) and
underperforms; the flow actor's on-manifold anchoring is what makes bilinear usable. The peak
lives in the frozen critic, not the training history. (exp3 still training; curve flat ~0.52.)

## Exp 4 — Tangent-projected DDPG+BC  *(training; medium)*

Estimate the behavior tangent `T_μ(s)` from local flow samples or the flow Jacobian; replace the
actor's Q-gradient by `P_{T_μ(s)} ∇_a Q`. If this recovers bilinear performance **and** eliminates
the 17.9 support-departure (Finding 1), the flow-projection account becomes very strong.

## Exp 5 — Checkpoint trajectories  *(TRAINING RUNNING — retrain w/ checkpointing)*

Across training checkpoints track: policy success, baseline AUC, advantage geometry, actor update
norm, off-reference peak gap, actor support distance. Distinguishes stable advantage learning
from a teacher-then-drift dynamic.

**Blocked then unblocked:** the saved runs used `save_interval=500000` (only 500k+1M on disk).
Partial 2-point read (`ckpt_static_metrics.py`, 500k vs 1M): rerank success 0.960→0.944 (stable),
eff-rank 4.8→4.9 (stable), **off-ref peak gap +4.5→+5.2 (sharpens)** — the policy-relevant
ordering is stable post-convergence while the off-support peak keeps growing; but this cannot see
the 0→500k learning phase. So retraining **1 seed each** (bilinear + MLP,
`cube_single_state_gciql_flow{_bilinear_d50,}`, `train_steps=500000 save_interval=50000`) →
`/home/adityamohan/exp5_retrain/`, tmux `exp5`, wandb amsks. **Harness ready:**
`scripts/atlas/exp5_trajectory.py` (rerank success + static metrics per checkpoint); run it when
the 10 checkpoints land (~2h).

## Implementation status of the training experiments (2/3/4)

**Frozen-critic infra: BUILT + VERIFIED** (`gciql_flow.py`, config `freeze_critic_ckpt` +
`freeze_critic_epoch`). On `create()`: splices the checkpoint's `modules_value/critic/
target_critic` into a fresh-init tree (actor random) and wraps the optimizer in
`optax.multi_transform` (`set_to_zero` on critic/value, adam on `modules_actor*`). Smoke @300
steps: `modules_value` and `modules_critic` **max|Δ|=0.0 (bit-frozen)**, `modules_actor*` changed
(trained from fresh init). `modules_target_critic` drifts 0.017 via Polyak — benign (feeds only
the zeroed critic-loss; the actor's Q-guidance reads the *online* frozen critic).

**Ready-to-run launches** (each ~2h, GPUs 2–7 free; `output_root=/home/adityamohan/exp5_retrain`
or a new dir; add `+agent_overrides.freeze_critic_ckpt=<dir> +agent_overrides.freeze_critic_epoch=<step>`):
- **Exp 2a** flow vs frozen **bilinear**: `--config-name cube_single_state_gciql_flow_bilinear_d50`,
  freeze from `.../GCIQL-State-Flow-Bilinear-d50/sd000_...` @500000.
- **Exp 2b** flow vs frozen **MLP**: `--config-name cube_single_state_gciql_flow`, freeze from
  `.../GCIQL-State-Flow/sd000_...` @1000000.
- **Exp 3** **DDPG+BC** actor vs frozen bilinear: `--config-name
  cube_single_state_gciql_flow_bilinear_d50_ddpgbc`, freeze from the bilinear ckpt @500000.
- **Exp 4** = Exp 2a + a tangent projection on the actor Q-gradient (still to implement).

Interpretation: Exp 2 — if the flow actor trained against the frozen bilinear critic beats the one
against the frozen MLP critic (matching Exp-1 reranking 0.91 vs 0.59), the critic field is
intrinsically better, independent of co-adaptation. Exp 3 — if the fresh DDPG+BC actor finds the
off-support peak (large support-departure, Finding 1), the peak is intrinsic to bilinearity.

## Arch 11 — The likely improved architecture (dueling)

The decomposition says one head is doing two incompatible jobs. Make it explicit:

    Q(s,a,g) = V_MLP(s,g) + A_bil(s,a,g),
    A_bil(s,a,g) = φ(s,a)·ψ(g) − E_{ã∼μ} φ(s,ã)·ψ(g).

MLP value head = absolute return calibration; bilinear head = behavior-centered action
preference; the **flow actor consumes only `A_bil`**. Prediction: recovers/beats bilinear on cube
and is more consistent across environments (no longer trades value calibration for a structured
action field). Build only if Exps 1–2 support the "advantage field is the load-bearing piece."

---

## Dependencies / cost

| exp | needs | cost | gate |
|-----|-------|------|------|
| 1 | frozen critics + μ + sim rollout | minutes | none — lead |
| 2 | actor training vs frozen critic | ~hours (GPU) | after 1 |
| 3 | eval swap + 1 DDPG+BC train | ~hours | after 1 |
| 4 | tangent projection + train | ~hours | after 1/2 |
| 5 | per-checkpoint analysis | ~1 hr | any time |
| 11 | new agent variant + train | days | after 1–2 |
