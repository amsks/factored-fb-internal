# Why does bilinear GCIQL + flow win on cube? — one experimental stream

*Collated 2026-07-16. Single chronological stream of the bilinear-mechanism investigation,
merging the three prior atlas docs (now in `archive/`): the pre-registered plan
(`bilinear_dominant_mechanism_plan.md`), the oracle-free findings (`oracle-free-findings.md`,
Findings 1–6), and the mechanism program (`bilinear-mechanism-program.md`, Exp 1–5 + dueling).
Code: `scripts/atlas/`, `scripts/value/`; results: `analysis/atlas/`, `analysis/value/repsep/`;
full run log: `analysis/atlas/DECISIONS.md`.*

Status tags used throughout: **[SOLID]** oracle-free / reproduced, **[RETRACTED]**, **[NULL]**
tested but under-powered, **[IN-FLIGHT]** running or partial.

---

## 0. The question and the fact that has to be explained

Bilinear GCIQL with a flow actor is the best cube-manipulation cell in the whole bridge
programme (91.6 @500k vs MLP-critic 71.3). Two numbers define the puzzle:

|                | flow actor | DDPG+BC actor |
|----------------|-----------:|--------------:|
| bilinear d50   | **0.947**  | 0.557         |
| MLP critic     | 0.747      | 0.747         |

*(OOD-grid C3 success.)* Bilinear+flow **wins** (> MLP+flow); but bilinear+flow ≫ bilinear+DDPG+BC
while MLP is actor-invariant — an **actor reversal**: bilinear is better than MLP with a flow
actor and worse with a Gaussian one. Any explanation must produce that sign flip.

We attack it by rendering the part of the value function the actor actually consumes and comparing
it across critics on byte-identical inputs. Most of this needs **no simulator** — forward passes
on checkpoints we already have.

---

## 1. Candidate mechanisms (the pre-registered plan)

Before measuring, four physically-grounded candidate explanations were pre-registered, each with a
simulator-counterfactual candidate operator `T_h(s,a,g)`:

1. **Grasp acquisition** — better identifies actions that establish control (`p_grasp`).
2. **Retained control / viability** — better preserves a stable, steerable grasp (`p_hold`,
   `T_viable` = fraction of goals still reachable after the action).
3. **Directional transport** — better matches local cube motion to goal direction
   (`T_dir = Δx_cube · d_goal`, plus per-axis `T_x/T_y/T_z`).
4. **Scripted primitives** — merely compresses the play-policy's approach→close→lift→translate→
   release phases rather than a general action-effect basis.

A candidate is **dominant** only if it passes **necessity + sufficiency + environmental
intervention** (keep-only preserves ≥80% of the bilinear-over-MLP gain; remove-only destroys ≥80%;
the matching environment edit — secure-in-hand start, weld-cube, rotated workspace, alternate
play-policy — closes the gap; competitors fail at least one). The expected answer was
phase-dependent: *acquire control (P0/P1) → preserve control (P2) → use control directionally (P3)*.

Reality diverged from this plan: the correlational label was **retracted** (Finding 3), the causal
surgery **hit design walls** (Finding 4), and the mechanism was ultimately pinned by a
non-surgical instrument — **reranking** (Finding 6). The plan's environmental interventions
(weld, rotate, alternate-dataset) remain **un-run** and are the main open lever (§9).

---

## 2. The object we measure, and how

**Centered action–goal preference operator.** For critic *m*, fixed physical state *s*, action
bank `{aᵢ}`, goal bank `{gⱼ}`:

    A^m_s(a, g) = Q_m(s, a, g) − E_{ã ~ μ(·|s)} [ Q_m(s, ã, g) ]

Subtracting the μ-mean removes the per-(s,g) offset that does not affect action selection, leaving
exactly the part of Q that ranks actions. μ is a **frozen, goal-independent BC flow** (`actor_vf`,
verified Q-independent). We standardise each `A^m_s` by its Frobenius norm (argmax/SVD/rankings
invariant) and SVD the *evaluated* operator (bilinear latent dims are non-identifiable). **Held
disclaimer:** `rank(A) ≤ d` for a bilinear critic *by construction* — low rank is architecture, not
a finding; the empirical questions are the **MLP's** effective rank and whether the critics span
the *same* modes.

**6 critics** through one uniform adapter `Q(s,A,G)→[n_a,n_g]` (`scripts/atlas/critics.py`),
identity confirmed from each checkpoint's own config: `bilinear_flow` (500k), `bilinear_ddpgbc`
(500k), `mlp_flow` (1M¹), `mlp_ddpgbc` (500k), `simplesf` (500k), `fb` (torch, plain FB 1.5M
ortho=1000). ¹mlp_flow's 1M is a *conservative* confound — the MLP side gets 2× training and still
loses.

**Frozen banks** (`banks.py`, `banks.npz`), byte-identical across critics: state bank (4 phase
anchors × 32 states — P0 `reset()` OOD start, P1 pre-contact, P2 at-contact, P3 holding/transport,
phases labelled from cube–effector geometry + contact channel `obs[18]`, never from a policy;
each state carries its action prefix for exact reconstruction); action bank (`A_mu` 256 μ-samples +
perturbations + Sobol space-filler; no policy's own actions are members); goal bank (5 eval goals +
per-state hindsight + 64-point cube-target grid). Every reset passes `task_id` explicitly — bare
`reset(seed=)` samples a random task from a global RNG and is **not** reproducible (verified
footgun). Compute: pure forward passes, ~30–45 s/critic on CPU.

---

## 3. Finding 1 — the actor reversal is off-reference critic exploitation  [SOLID]

*(`scripts/atlas/gates.py::phase_0a`, `off_reference_test.py`. Whitened distance to the μ cloud;
no simulator.)*

Whitened nearest-neighbour distance from each **deployed actor's** chosen action to the μ-reference
cloud (median P0+P2, 8 states × 5 goals):

| actor / critic | bilinear | MLP  |
|----------------|---------:|-----:|
| **DDPG+BC**    | **17.9** | 1.76 |
| flow           | 2.37     | 2.00 |

The DDPG+BC actor on the **bilinear** critic departs the manifold ~**10×** further than on MLP;
both flow actors stay anchored. **Direct critic-side confirmation it is a peak, not a slope**
(same co-trained agent supplies action + critic; `B_μ` = 256-action reference set):

| critic          | Q(a_DDPG) − maxₐ∈B_μ Q(a) | fraction > 0 | ‖∇ₐ Q‖ |
|-----------------|--------------------------:|-------------:|-------:|
| bilinear+DDPG+BC | **+2.14**                | **0.787**    | 3.18   |
| MLP+DDPG+BC       | −0.26                    | 0.062        | 3.06   |

The bilinear critic values the DDPG+BC action *above every reference action* **79%** of the time
(median +2.14); MLP only 6% (−0.26, below the best reference). Action-gradient norms are
**equal** (3.18 vs 3.06) — a genuine off-support *maximum that exceeds the on-manifold maximum*,
not a globally steeper landscape. **Mechanism for the sign flip:** bilinear *needs* the flow actor
because the flow actor cannot cash in the exploit the bilinear critic exposes; a Q-ascending
Gaussian climbs straight to it. *Pending:* the behavioural half (Phase 0B — that those distant
actions are actually worse in sim) and the full value profile along the segment (endpoints only so
far).

---

## 4. Finding 2 — multi-modality refuted; the difference is *content*, not *shape*  [SOLID]

*(`mode_analysis.py`, `run_oracle_free.py`, `calibrate_angles.py`. 32 states, 64-goal grid.)*

The Gaussian-averages-across-modes story is dead:

| critic        | effective rank (of 64) | top-sv share | basins |
|---------------|-----------------------:|-------------:|-------:|
| bilinear_flow | 6.65 ± 1.68            | 0.500        | **1.0** |
| mlp_flow      | 6.81 ± 1.41            | 0.486        | **1.0** |
| simplesf      | 6.99 ± 2.02            | 0.479        | **1.0** |

Effective rank and mode concentration are the **same** across critics; every critic's advantage
over actions is **unimodal** (basins = 1.0 in the top-10% of |A(·,g)| on a shared k-NN graph). An
earlier single-state teaser (bilinear top-sv 0.969 < MLP 0.980) **does not survive** at n=32.

**But the critics are different operators** — principal angles between top-3 goal-mode subspaces
(degrees; seed-calibrated over 3 seeds each):

| pair                        | mode 1   | mode 2 | mode 3 |
|-----------------------------|---------:|-------:|-------:|
| bilinear vs bilinear (seed) | 17.5     | 38.1   | 70.4   |
| MLP vs MLP (seed)           | 22.2     | 41.6   | 62.7   |
| **bilinear vs MLP**         | **53.7** | 76.2   | 85.8   |
| random vs random (ceiling)  | 71.0     | 79.9   | 86.9   |

On the **dominant** goal-preference mode the critics disagree (53.7°) far more than two seeds of
the same critic (~20°), below the random ceiling (71°); modes 2–3 are at random. **Same spectral
shape, genuinely different content.** *(FB skipped here — torch critic, jax run; needs a separate
pass.)*

---

## 5. Finding 3 — naming the content (direction vs distance)  [RETRACTED]

*(`action_grounded_decode.py`, `physical_fields.py`.)*

Over the **5 eval goals** it looked clean — bilinear's dominant goal mode correlated with goal
*direction* (|r|≈0.75), MLP's with *distance* (|r|≈0.80). **It does not survive the 64-goal grid:**

| goal set | bilinear dominant |r| | MLP dominant |r| | best-field pattern |
|----------|---------------------:|----------------:|--------------------|
| 5 eval   | 0.75 | 0.80 | clean: bil=direction, MLP=distance |
| 64 grid  | **0.37** | **0.26** | scattered — no field wins |

The high 5-goal correlations were small-n inflation; on 64 goals they collapse and the MLP's
dominant mode flips to mostly `goal_dir_y` (opposite the 5-goal story). **No coarse physical field
(direction/distance/height) cleanly explains the dominant goal mode.** The content difference of
Finding 2 is real; its physical identity is **still open**. *(Suggestive-only survivor: scoring the
policy's own action by one-step goal-directed transport `T_dir = Δx_cube·d_goal` put bilinear above
MLP above play at P3 — 0.0049 > 0.0040 > 0.0026 — arguing against a pure scripted-primitive reading;
5 eval goals, one-step, not established.)*

---

## 6. Finding 4 — the causal keep/remove surgery  [two design walls, no result]

*(`value_surgery.py`, `induced_surgery.py`.)* The pre-registered necessity/sufficiency test hit
two walls:

1. **First-action override** (fast): override only the first action, continue with the bilinear
   flow actor → **every operator gave identical 0.33** (the strong continuation washes out one
   first action). Not a causal test.
2. **Induced-policy rollout** (correct): each operator *drives* every step, physical candidates
   re-scored by one-step sim branching on a scratch env → correct but `O(H·N·episodes)` sequential
   MuJoCo steps, multi-hour. Not run.

The causal identity of the basis remains open; this is the same wall that later re-appeared
quantitatively in the causal action-ranking null (§7).

---

## 7. Finding 5 — the winning critic's value does NOT rank in-hand reachability  [SOLID, actor-matched]

*(RLBrew counterfactual value probe: `scripts/value/cf_value_probe_batched.py`.)* Reset 200 in-hand
grasp states, roll out **the agent's own policy** to success/failure, report rank-AUC of the learned
value vs realised success. **Actor held fixed** (both cells share the flow actor) so the comparison
isolates the **critic**:

| critic — flow actor, n=200 | read | in-hand rank-AUC |
|----------------------------|------|-----------------:|
| **MLP** (9-seed campaign)  | V(s,g) | 0.68–0.93, **mean 0.81** |
| MLP, batched cross-check   | V(s,g) | 0.875 |
| **bilinear** (3 seeds @500k) | V(s,g) | 0.363/0.389/0.418, **mean 0.39** |
| bilinear (sd000)           | Q(s,π,g) | 0.458 |

The paper's 0.89 is the **flow-actor MLP** GCIQL (our harness reproduces it: 0.875). The **bilinear
value ranks in-hand reachability below chance** (0.39), no overlap with MLP's band. **Dissociation:
with the same actor, the critic that wins the eval (0.947 vs 0.747) is the *worse* reachability-
ranker.** The bilinear win is **not** value fidelity.

### 7a. Decomposition — value-geometry vs advantage-geometry  [SOLID]

*(`scripts/value/value_decomp_probe.py`.)* Split `Q_m = U_m(s,g) + A_m(s,a,g)`, `U_m = E_{a~μ} Q_m`:

| readout, n=200 | MLP AUC | bilinear AUC |
|----------------|--------:|-------------:|
| **U = E_μ Q** (return magnitude) | **0.817** | **0.396** |
| V (value head)                   | 0.821   | 0.403        |
| Qπ (value at own action)         | 0.817   | 0.455        |
| Adv@π = Qπ − U                    | 0.664   | 0.584        |

MLP's entire in-hand ranking lives in the action-independent baseline **U (0.817)**; bilinear's U
is **anti-calibrated (0.396)**. Across states the two baselines are **uncorrelated**
(`Spearman(U_bil,U_mlp) = −0.04`) while the two **advantage** orderings **partially agree**
(`Spearman(A_bil,A_mlp) = +0.22`, per-task 0.13–0.36). Orthogonal value calibration,
partially-shared policy-improvement geometry. For bilinear, Adv@π (0.584) *out-ranks* its own U
(0.396) — its signal lives in the **action-dependent** part. **"MLP fits return magnitude" is
confirmed; "bilinear organizes policy improvement" is the strong clause, settled below.**

### 7b. Causal action-ranking test  [NULL — continuation wash-out]

*(`action_ranking_causal.py`.)* For each `(s,g)`, score both centered advantages over N=16
`aᵢ∼μ`, execute each first action (bit-exact prefix-replay), continue every branch under one shared
frozen policy (MLP flow — conservative for a bilinear-wins prediction), outcome `Y_i = γ^{t_hit}`,
`ρ_m = Spearman(A_m, Y_i)`:

| cell | prereg | ρ_bil | ρ_mlp | gap (b−m) | cf success |
|------|--------|------:|------:|----------:|-----------:|
| C1 ID/ID | ρ_MLP≥ρ_bil | −0.042 | −0.170 | +0.13 | 0.66 |
| C2 ID/OOD | ρ_bil>ρ_MLP | −0.086 | −0.002 | −0.08 | 0.74 |
| C4 OOD/ID | ρ_MLP≈ρ_bil | −0.049 | −0.017 | −0.03 | 0.75 |
| C3 OOD/OOD | ρ_bil>ρ_MLP | −0.050 | +0.023 | −0.07 | 0.83 |

Every ρ ≈ 0. **Low-power null, not a refutation:** with cf-success 0.66–0.83 the competent 150-step
continuation recovers from almost any first action, so both advantages correlate ~0 — including the
MLP's own advantage on its home turf. The shared-continuation design **cannot** adjudicate the
strong clause; power would need an *unrecoverable* first action (P3 grasped states). Superseded by
Finding 6.

---

## 8. Finding 6 — behavior-sample reranking: the frozen bilinear critic IS the policy  [SOLID, powered]

*(`rerank_eval.py`, `rerank_audit_multienv.py`. This is **Exp 1** of the program — the lead, pure
inference, no training.)* Deploy the critic's ordering as the policy: at every step sample N=32
`aᵢ∼μ`, execute `argmax_i Q_m(s,aᵢ,g)`. OOD start, H=200, C3 protocol.

| policy | μ = bilinear candidates | μ = MLP candidates |
|--------|------------------------:|-------------------:|
| μ-floor (no rerank)   | 0.064 | 0.056 |
| **bilinear-rerank**   | **0.960** | **0.912** |
| MLP-rerank            | 0.592 | 0.568 |
| *(trained flow actor)* | *bilinear 0.947 / MLP 0.747* | |

**N-sweep (cube):**

| N          | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|------------|---|---|---|---|----|----|----|-----|
| bilinear   | .067 | .467 | .800 | .853 | .920 | **.947** | .907 | .853 |
| MLP        | .067 | .240 | .453 | .560 | .573 | .587 | .640 | .560 |

- **Bilinear-rerank ≈ its trained flow actor (0.96 vs 0.947) with no actor training** — the final
  critic's **ordering of supported actions** is intrinsically better. Saturates HIGH by N=32.
- **MLP-rerank (0.59) < MLP's trained actor (0.747)** — the flow actor does real work for MLP,
  almost none for bilinear. Robust to swapping the candidate BC flow.
- **Off-support signature:** bilinear peaks at N=32 then *declines* to 0.85 at N=128 — more
  candidates → more chances to hit the over-valued off-support action (Finding 1).

**This resolves the central question.** Bilinear + flow is a **compatible policy-learning system**:
the bilinear bottleneck supplies a well-ordered goal-conditioned energy over *supported* actions
(extractable by argmax-over-μ) which the flow actor projects onto the behavior manifold; the poor
value *calibration* (Finding 5) is irrelevant because reranking uses only the *ordering*.

---

## 9. Program status — constructive follow-ups (2026-07-16)

### Exp 2 — fresh actor vs FROZEN critic → **co-adaptation REFUTED**  [SOLID]

*(Frozen-critic infra in `gciql_flow.py`: `freeze_critic_ckpt`/`freeze_critic_epoch`; splices ckpt
value/critic/target into a fresh-actor tree, `optax.multi_transform` zeroes critic/value grads.
Smoke-verified bit-frozen.)* Cube, 1 seed, 500k:

| fresh actor vs | success | co-trained | Exp-1 rerank |
|----------------|--------:|-----------:|-------------:|
| **frozen bilinear** (exp2a) | **0.88–0.89** | 0.947 | 0.91 |
| **frozen MLP** (exp2b)      | **0.68**      | 0.747 | 0.59 |

Same ~0.2 gap and direction as co-trained cells and reranking, with **no shared training history**
→ the advantage is a property of the **bilinear critic's learned field**, not actor–critic
co-adaptation.

### Exp 3 — fresh DDPG+BC vs frozen bilinear → **off-support peak INTRINSIC**  [SOLID]

| fresh DDPG+BC vs frozen bilinear | success | co-trained | flow ref |
|----------------------------------|--------:|-----------:|---------:|
| exp3 | **~0.52** | 0.557 | 0.88 |

A fresh Q-ascending Gaussian reaches only ~0.52 (≈ co-trained 0.557 ≪ flow 0.88) — even without
co-training it is pulled off-manifold to the bilinear peak (Finding 1). The flow actor's on-manifold
anchoring is what makes bilinear usable; the peak lives in the frozen critic geometry.

### Exp 1b — Multi-env reranking N-sweep → **cube is special; bilinear collapses off-cube**  [SOLID]

*(`rerank_audit_multienv.py`, `analysis/atlas/rerank_audit_{antmaze,scene}.json`. N-sweep of
`argmax_{i≤N} Q_m(s,aᵢ,g)` as the policy, per env. antmaze/scene capped N≤32.)*

|          | floor | trained bil / mlp | rerank sweep (N = 1,2,4,8,16,32[,64,128]) |
|----------|------:|-------------------|-------------------------------------------|
| **cube**    | .067 | 0.933 / 0.747 | bil .07 .47 .80 .85 .92 **.95** .91 .85 · mlp .07 .24 .45 .56 .57 .59 .64 .56 |
| **antmaze** | .04  | NA / NA¹      | bil .09 .43 **.47** .17 .05 .07 · mlp .17 **.52** .47 .33 .24 .25 |
| **scene**   | .027 | 0.20 / 0.507  | bil .03 .19 **.32** .24 .19 .09 · mlp .03 .51 .57 **.64** .59 .60 |

- **Cube is the exception, not the rule.** Bilinear-rerank wins decisively and saturates HIGH only
  on cube. On **antmaze and scene** bilinear-rerank **peaks early (N=4) then COLLAPSES** (antmaze
  .47→.07, scene .32→.09) while **MLP-rerank wins** (antmaze .25, scene .60 vs bilinear .07/.09).
- **The off-support peak (Finding 1) is catastrophic off-cube.** The same "more candidates → hit the
  over-valued off-support action" signature that only nicked cube at N=128 dominates from N=8 on
  antmaze/scene — the bilinear critic's argmax lies far off μ support where the behavior prior does
  not contain the needed actions / improvement is hierarchical.
- **Scene even reverses the trained cells:** trained bilinear 0.20 < trained MLP 0.507. Bilinear is
  not a general win — it is favorable **specifically** on cube (competent pick-and-place already in
  the data, low-dimensional reusable goal-conditioned improvements), exactly the program's
  pre-registered prediction for where the bias breaks.

¹antmaze trained anchors N/A — a `load_agent` actor-shape mismatch; reranking unaffected.

### Exp 4 — tangent-projected DDPG+BC  [IN-FLIGHT / to implement]

Estimate the behavior tangent `T_μ(s)` (flow Jacobian / local samples), replace the actor's
Q-gradient by `P_{T_μ(s)} ∇_a Q`. If this recovers bilinear performance **and** eliminates the 17.9
support-departure, the flow-projection account becomes very strong. Projection not yet implemented.

### Exp 5 — checkpoint trajectory  [IN-FLIGHT, partial]

Retrain 1 seed each (bilinear+MLP, `save_interval=50000`) → track success, baseline AUC, advantage
geometry, off-ref peak gap over 0→500k. Partial 2-point read (500k vs 1M, `ckpt_static_metrics.py`):
rerank 0.960→0.944 **stable**, eff-rank 4.8→4.9 stable, **off-ref peak gap +4.5→+5.2 sharpens** —
policy-relevant ordering stable post-convergence while the off-support peak keeps growing. The
0→500k learning phase is what the retrain will show.

### Arch 11 — behavior-centered dueling GCIQL  [IN-FLIGHT — ⚠ FLOORED]

`Q = U_MLP(s,g) + A_bil`, `A_bil = (φ(s,a)−b(s))·ψ(g)/√d`, `b(s)` learned to track `sg φ(s,a_data)`
so `E_μ[A_bil]≈0`; **the flow actor consumes only `A_bil`**. Constructive test of the decomposition.
Implemented (`value_arch='dueling'`, `GCDuelingValue`, `dueling_b_coef`), b-loss scale-normed after
a raw-scale explosion. **v2 still floored: 50k = 0.053, 100k = 0.000.** The A_bil-only-actor arch
likely needs a rethink, not just a loss rescale — flagged.

### FB reranking (plain FB 1.5M, ortho=1000)  [SOLID — opposite of bilinear]

*(`rerank_audit_fb.py`, proper FB-eval reward-relabel z.)* trained_fb 0.413, floor 0.52, then
**collapses with N**: N=2 0.33, N=4 0.35, N=8 0.24, N=16 0.21, N=32 0.17. FB's Q is
**anti-informative** about action differences — a **representation** bottleneck, the *opposite* of
bilinear (whose ordering is excellent). Clean at the proper operating point.

### FB action-allocation ("action shift")  [SOLID — fixed 2026-07-16]

*(`action_allocation.py --critic fb`.)* Previously all-0.0 (sparse 4cm hit-indicator z → Q≡0 on OOD
eval goals). **Fixed** in `critics.py` to use the exact FB-eval reward-relabel z (`_infer_z`, same as
the rerank harness; |z|=7.07, relabel_nonzero 148–276/task). Q-variance decomposed on the atlas bank
(64 μ-actions/state):

| critic        | action_fraction | centered_adv_energy |
|---------------|----------------:|--------------------:|
| bilinear_flow | 0.0046          | 1.1e-4              |
| **fb**        | **0.0021**      | **9.7e-5**          |
| mlp_flow      | 0.0010          | 2.0e-5              |

FB Q is ~99.8% state-driven / ~0.2% action-driven — **confirms the note's premise** ("almost none of
the trained FB critic's Q-variation is action-driven"); FB's action fraction sits between bilinear
and MLP. Blunt instrument (baseline dominates for all three) — reranking is the sharper isolation
test; this just makes the FB row honest.

### FB advantage decomposition — subtract the state baseline, focus on actions  [SOLID — 2026-07-16]

*(`scripts/value/value_decomp_probe_fb.py`. FB row of Finding 5's decomposition, which had
bilinear/MLP only. Labels reused from `cf_records_fb_k0.parquet` — no rollout; only forward passes
to add `U_FB`.)* Idea (user): subtract a state baseline from FB's measure so the readout focuses on
the action-dependent part — `Q_FB = U_FB(s,z) + AdvPi`, `U_FB = E_{a~μ}Q_FB`,
`AdvPi = Qπ − U_FB`. AUC each against realized in-hand grasp→goal success (n=200, actor-matched):

| readout | MLP | bilinear | **FB** |
|---------|----:|---------:|-------:|
| **U = E_μ Q** (state baseline)   | 0.817 | 0.396 | **0.690** |
| **AdvPi = Qπ − U** (action part) | 0.664 | 0.584 | **0.448** |

Per-task FB AdvPi: 0.361 / 0.455 / 0.466 / 0.469 / 0.489 — **consistently ≤ 0.5**. FB's in-hand
reachability signal lives **entirely in the state baseline** `U` (0.690 ≈ Qπ 0.690, MLP-like
pattern); the isolated **action part is below chance** (0.448, mildly *anti*-informative — worse
than bilinear's 0.584). So there is **no useful action-advantage in the trained FB representation to
isolate** — consistent with FB reranking collapsing and the ~0.2% action-fraction.

**Why this is the right inference-time test and reranking is NOT.** Within a fixed state the
baseline `U_FB(s,z)` is action-independent, so `argmax_a AdvPi = argmax_a Q_FB` and
`Spearman_a(AdvPi, ·) = Spearman_a(Q_FB, ·)` — subtracting a state-only baseline is a **no-op** for
any within-state action ranking (rerank / per-state transport alignment). It changes only
**cross-state** pooled metrics like this AUC, where removing the state offset isolates the action
part. **Consequence:** the centering is confirmed futile at inference *because the action signal is
absent, not merely drowned* — the empirical case **for** the training arch (Idea #2, §11): only
changing what the measure *learns* can put action-advantage information into `ΔF_μ`.

---

## 10. Where it stands, in one paragraph

Bilinear and MLP critics are, spectrally, the **same kind of object** — a single ~6.7-effective-dim
preference over actions per state, not a multi-modal landscape. What separates them is **content**
(dominant goal-mode angle 53.7° vs a ~20° seed floor) and **off-reference behaviour** (bilinear
places its action-maximum *off* the μ manifold, valuing the DDPG+BC action above every reference
action 79% of the time, +2.14; MLP does not; equal action-gradients → a peak, not a slope). The
flow actor is compatible with bilinear precisely because it never leaves μ support to reach that
peak, while a Q-ascending Gaussian climbs straight to it — which is why bilinear *needs* the flow
actor and MLP does not. Multi-modality is off the table. The physical *name* for the content
difference (direction vs distance) did **not** survive a 64-goal check and is retracted; the
difference is real but its identity is still open. Actor-matched, the winning critic's *value* is
**not** a better reachability-predictor (bilinear 0.39 vs MLP 0.81 on the RLBrew in-hand probe) —
the split is value-geometry (MLP, in the return-magnitude baseline) vs advantage-geometry
(bilinear, in the action-dependent part). The powered resolution is **reranking**: the frozen
bilinear critic's action ordering *is* a strong policy (0.91–0.96 ≈ its trained actor), the MLP's is
not (0.57–0.59) — confirmed intrinsic by fresh-actor-vs-frozen-critic (0.88 vs 0.68, co-adaptation
refuted) and shown off-support-peak-intrinsic by fresh DDPG+BC (0.52). FB, by contrast, *collapses*
under reranking — a representation bottleneck, the mirror image of bilinear. And the win is
**cube-specific**: on antmaze and scene bilinear-rerank collapses with N (the off-support peak turns
catastrophic where the behavior prior lacks the needed actions) and MLP-rerank wins — the bilinear
bias is favorable on cube, not a general property.

---

## 11. Open threads

- **[MOTIVATED, not started] FB successor-advantage training arch (Idea #2)** — the FB analogue of
  dueling: subtract a learned state baseline `b(s,z)` from the *measure* during training so `ΔF_μ`
  learns the action part. Now justified: the inference decomposition showed FB's action-advantage is
  **absent** (AdvPi 0.448), not merely drowned, so only a training change can create it. ~1–2 days +
  a 500k run; heed the dueling floor as a risk.
- **[IN-FLIGHT] Dueling (Arch 11)** floored at 100k — decide arch rethink vs kill.
- **[IN-FLIGHT] Exp 4** tangent-projected DDPG+BC — projection not yet implemented.
- **[IN-FLIGHT] Exp 5** learning-phase trajectory — awaiting the 50k-interval checkpoints.
- **[UN-RUN] The plan's environmental interventions** — secure-in-hand start, **weld-cube**,
  rotated workspace, alternate play-policy. These are the only *causal* discriminators among the
  §1 candidates (hold vs direction vs grasp vs script) and remain the strongest untaken lever.
- **[OPEN] Physical identity of the bilinear content** (Finding 3 retracted) — the ψ(g) goal-factor
  difference is the remaining lead; needs a causal test, not another correlational label.
- **[PENDING] Finding 1 Phase 0B** (behavioural half of the off-peak) and the full value profile
  along a_ref* → a_DDPG.
- **[DONE, §Exp 1b] Multi-env reranking** N-sweeps (antmaze/scene) — bilinear collapses off-cube,
  MLP-rerank wins. Remaining: the SimpleSF rerank cell, and antmaze trained anchors (load_agent
  shape fix).
- **Write-up:** fold these numbers into `findings.tex`; refresh the living CSVs
  (`scripts/results_map.py` + `compile_results_csvs.py --refresh`).
