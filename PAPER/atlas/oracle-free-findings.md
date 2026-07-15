# What is bilinear GCIQL doing? — oracle-free findings

*Working notes, 2026-07-15. Code: `scripts/atlas/`; results: `analysis/atlas/`; full run log:
`analysis/atlas/DECISIONS.md`.*

## The question

Bilinear GCIQL with a flow actor is the best cube-manipulation cell in the whole bridge
programme (91.6 @500k, vs MLP-critic 71.3). Two facts define what we are trying to explain:

|                | flow actor | DDPG+BC actor |
|----------------|-----------:|--------------:|
| bilinear d50   | **0.947**  | 0.557         |
| MLP critic     | 0.747      | 0.747         |

*(OOD-grid C3 success.)* Bilinear+flow **wins** (> MLP+flow); bilinear+flow ≫ bilinear+DDPG+BC
is an **actor reversal** — bilinear is better than MLP with a flow actor and worse with a
Gaussian one. Any explanation has to produce that sign flip.

We attack this by rendering the part of the value function the actor actually consumes, and
comparing it across critics on byte-identical inputs. Most of this needs **no simulator** —
just forward passes on checkpoints we already have — so those results are reported here. The
one experiment that needs a simulator oracle is described at the end, including why it did not
(yet) yield signal.

---

## The object we measure

For critic *m*, a fixed physical state *s*, an action bank `{aᵢ}` and a goal bank `{gⱼ}`, the
**centered action–goal preference operator**:

    A^m_s(a, g) = Q_m(s, a, g) − E_{ã ~ μ(·|s)} [ Q_m(s, ã, g) ]

Subtracting the μ-mean removes the per-(s,g) offset that does not affect action selection,
leaving exactly the part of Q that ranks actions. μ is a **frozen, goal-independent
behaviour-cloning flow** (the BC velocity field `actor_vf`, verified to receive gradient only
from the BC loss — Q-independent). We render `A^m_s` as a matrix over a shared bank and analyse
its structure.

Why the *centered* operator and not raw Q: the raw value is dominated by a state–goal offset
(how far the goal is), which every critic represents and which tells you nothing about action
choice. The centered operator is invariant to that offset by construction.

Two invariances make this the right object:
- **Standardisation.** Each `A^m_s` is divided by one positive scalar (its Frobenius norm), so
  cross-critic magnitudes are comparable. Argmax, singular vectors, rankings and modal
  topology are all invariant to this.
- **Representation non-identifiability.** For a bilinear critic `Q = φ(s,a)·ψ(g)`, the map
  `φ→Rφ, ψ→R⁻ᵀψ` leaves Q unchanged, so individual latent dimensions are meaningless. We SVD
  the **evaluated operator** `A = UΣVᵀ`, whose singular modes are invariant up to sign and
  rotation within degenerate subspaces.

**Disclaimer we hold throughout:** for a bilinear critic `rank(A) ≤ d` by construction, so low
rank is *architecture, not a finding*. The empirical question is the **MLP's** effective rank
and whether the two critics span the *same* modes.

---

## How we ran it

**Critics (6).** Loaded through one uniform adapter `Q(s, A, G) → [n_a, n_g]` spanning both
stacks (`scripts/atlas/critics.py`). Identity confirmed from each checkpoint's own config
(`flags.json` / `.hydra`), not the group label:

| critic          | arch     | actor          | step | stack |
|-----------------|----------|----------------|------|-------|
| bilinear_flow   | bilinear | flow           | 500k | jax   |
| bilinear_ddpgbc | bilinear | DDPG+BC        | 500k | jax   |
| mlp_flow        | MLP      | flow           | 1M¹  | jax   |
| mlp_ddpgbc      | MLP      | Gaussian DDPG+BC | 500k | jax |
| simplesf        | SF       | flow           | 500k | jax   |
| fb              | FB       | flowbc         | final| torch |

¹ mlp_flow saved no 500k intermediate; using its 1M is a **conservative** confound — the MLP
side of "the win" gets 2× the training and still loses.

**Frozen banks** (`scripts/atlas/banks.py`, `analysis/atlas/banks.npz`), byte-identical across
all critics:
- **State bank** — 4 phase anchors × 32 states: `P0` = `env.reset()` (OOD start), `P1`
  pre-contact, `P2` at-contact (cube on table), `P3` holding/transport. Phases labelled from
  cube–effector geometry and the contact channel `obs[18]`, never from a policy. States are
  reached by rolling μ forward from a pinned reset, and each carries its generating action
  prefix so it can be reconstructed exactly.
- **Action bank** — `A_mu` (256 samples from μ) is the behavioural-reference set; plus
  perturbations and a Sobol space-filling set for off-reference probes. Critic-independent by
  construction (no policy's own actions are bank members — that would make an argmax result
  self-fulfilling).
- **Goal bank** — 5 evaluation-task goals + per-state reachable hindsight goals + a 64-point
  grid of cube targets.

**Environment reproducibility.** Every reset passes `task_id` explicitly. A bare
`reset(seed=)` samples a random task from a *global* RNG (`manipspace_env.py:304`) and is
**not** reproducible — a footgun we verified and route around everywhere.

**Compute.** Pure critic forward passes; runs in ~30–45 s per critic on CPU. `A^m_s` computed
over the 64-goal grid (so effective rank is not capped by the number of goals), 32 states per
critic, 128 μ-reference actions.

---

## Finding 1 — the actor reversal is off-reference critic exploitation (Phase 0A)

*(`scripts/atlas/gates.py::phase_0a`. Uses whitened distance to the μ cloud — no simulator.)*

For each critic we take the action its **deployed actor** actually selects at a matched
(state, eval-goal) pair, and measure its **whitened nearest-neighbour distance to the
μ-reference action cloud** — how far off the behavioural manifold the actor goes. Median over
P0+P2, 8 states × 5 goals:

| actor / critic | bilinear | MLP  |
|----------------|---------:|-----:|
| **DDPG+BC**    | **17.9** | 1.76 |
| flow           | 2.37     | 2.00 |

The DDPG+BC actor on the **bilinear** critic departs the behavioural reference ~**10×** further
than on the MLP; both **flow** actors stay anchored regardless of critic. Reading: the bilinear
critic contains **attractive off-reference extrema**. A Q-ascending Gaussian actor (DDPG+BC)
finds and chases them off the data manifold; the flow actor is distillation-anchored to μ and
is structurally protected. The MLP critic has no such extrema, so its DDPG+BC actor also stays
put. This is a mechanism for the sign flip: **bilinear needs the flow actor because the flow
actor cannot cash in the exploit that the bilinear critic exposes.**

*Status:* this is a **departure** result (distance). The critic-side confirmation below shows
that departure lands on a real value peak; the *behavioural* confirmation — that those distant
actions are actually worse in the simulator — is Phase 0B, not yet run.

### Direct critic-side confirmation — it is a peak, not a steeper landscape

*(`scripts/atlas/off_reference_test.py`. Same co-trained agent supplies both the DDPG+BC action
and the critic; Q = min(q₁,q₂); B_μ = the 256-action behavioural-reference set.)*

Phase 0A shows the actor *leaves* the manifold; it does not show *why*. Two hypotheses: (a) an
**attractive off-reference peak** — the critic values some off-support action above everything
on-manifold; or (b) a merely **steeper landscape** — larger action-gradients everywhere, which
would tilt the DDPG+BC BC-vs-Q tradeoff without any separate peak. We separate them directly on
the critic (no simulator), over P0+P2 × 8 states × 5 eval goals:

| critic          | Q(a_DDPG) − maxₐ∈B_μ Q(a) | fraction > 0 | ‖∇ₐ Q‖ |
|-----------------|--------------------------:|-------------:|-------:|
| bilinear+DDPG+BC | **+2.14**                | **0.787**    | 3.18   |
| MLP+DDPG+BC       | −0.26                    | 0.062        | 3.06   |

- **The peak is real (a).** The bilinear critic values the DDPG+BC actor's chosen action
  *above every action in the behavioural-reference set* **79% of the time** (median gap +2.14).
  The MLP does so only 6% of the time (median −0.26 — its actor's action is *below* the best
  reference). The bilinear critic literally places its maximum off the μ support, and the
  Q-ascending actor climbs to it.
- **The steeper-landscape alternative (b) is ruled out.** The action-gradient norm ‖∇ₐ Q‖ is
  essentially **identical** (3.18 vs 3.06). It is not that bilinear is globally steeper; it is
  specifically an off-support *maximum that exceeds the on-manifold maximum* — which only the
  bilinear critic has.

This upgrades Phase 0A from "the actor departs" to "the actor departs *because the critic's
optimum lies off the behavioural manifold*," confirmed by value and not merely distance.

*Method caveat:* the value profile along the segment a_ref\* → a_DDPG was recorded only at its
endpoints (numerically identical to the peak gap above), so monotonicity *along* the path is
not separately established — the peak result stands on the endpoint gap. Capturing the full
profile (to confirm a smooth ridge rather than a distant peak past a valley) is a one-line
follow-up.

---

## Finding 2 — multi-modality is refuted; the difference is in *content*, not *shape*

*(`scripts/atlas/mode_analysis.py`, `run_oracle_free.py`. 32 states, 64-goal grid.
`analysis/atlas/oracle_free_results.json`.)*

A candidate explanation for the actor reversal was **multi-modality**: if `A(·,g)` over actions
were multi-modal for bilinear and unimodal for MLP, a unimodal Gaussian actor would average
across the modes and land between them, while a flow actor could sit on one. We tested it, and
it is **dead**:

| critic        | effective rank (of 64) | top-sv share | basins |
|---------------|-----------------------:|-------------:|-------:|
| bilinear_flow | 6.65 ± 1.68            | 0.500        | **1.0** |
| mlp_flow      | 6.81 ± 1.41            | 0.486        | **1.0** |
| simplesf      | 6.99 ± 2.02            | 0.479        | **1.0** |

- **Effective rank** (entropy of the singular spectrum) and **mode concentration** (top
  singular value share) are the **same** across bilinear / MLP / simplesf, within their SDs.
  An earlier single-state teaser (bilinear top-sv 0.969 < MLP 0.980) that hinted bilinear was
  "more mode-distributed" **does not survive** at n=32 (0.500 vs 0.486). No spectral-shape
  difference.
- **Basins = 1.0** for every critic: each critic's advantage over actions is **unimodal**
  (connected components in the top-10% of |A(·,g)| on a shared k-NN action graph). There is no
  second mode for a Gaussian actor to average across. The multi-modality hypothesis is refuted.
  This is *consistent* with Finding 1: the actor reversal is about **off-reference extrema**,
  not multiple in-support modes.

**But the two critics are not the same operator** — and this is now calibrated against a null
baseline (`scripts/atlas/calibrate_angles.py`, 3 seeds each of bilinear and MLP). Principal
angles between the top-3 goal-mode subspaces (degrees; 0 = identical, 90 = orthogonal):

| pair                        | mode 1   | mode 2 | mode 3 |
|-----------------------------|---------:|-------:|-------:|
| bilinear vs bilinear (seed) | 17.5     | 38.1   | 70.4   |
| MLP vs MLP (seed)           | 22.2     | 41.6   | 62.7   |
| **bilinear vs MLP**         | **53.7** | 76.2   | 85.8   |
| random vs random (ceiling)  | 71.0     | 79.9   | 86.9   |

On the **dominant** goal-preference mode, bilinear-vs-MLP (53.7°) sits **far above the
same-critic seed floor (~20°)** and below the random ceiling (71°): the two critics disagree on
their leading goal-preference direction *much more than two independently trained seeds of the
same critic do*, with only partial shared structure. On modes 2–3 the divergence is at the
random level — no shared structure beyond the first mode. **So the content difference is real,
not seed noise: same spectral shape, genuinely different content.**

Caveat: **FB was skipped** here (it is a torch critic and this run was in the jax venv) — it
needs a separate torch-venv pass.

---

## Finding 3 — attempt to name the content (direction vs distance): REFUTED by the goal-grid

*(`scripts/atlas/action_grounded_decode.py`, `physical_fields.py`. Oracle-free.)*

Finding 2 established that bilinear and MLP span *different* goal-preference content but did not
say **what** the content is. We tried to name it by physically labelling the modes: SVD each
critic's operator, then correlate each leading **goal mode** `v_k(g)` against physical goal
fields (direction x/y/z, cube–goal distance, height, goal-common) and each **action mode**
`u_k(s,a)` against one-step cube displacement (Δx via a single simulator step) and contact.

Over the **5 evaluation goals** this looked clean and tempting — bilinear's dominant goal mode
correlated with goal *direction* (\|r\|≈0.75) and the MLP's with goal *distance* (\|r\|≈0.80),
consistent across 28–30/32 states. **But it does not survive the goal-grid check and we retract
it.** Re-running the decode over a representative **64-goal grid**:

| goal set | bilinear dominant-mode \|r\| | MLP dominant-mode \|r\| | best-field pattern |
|----------|-----------------------------:|------------------------:|--------------------|
| 5 eval   | 0.75 | 0.80 | clean: bil=direction, MLP=distance |
| 64 grid  | **0.37** | **0.26** | scattered — no field wins |

The high correlations on 5 goals were **small-n inflation** (with 5 points almost any field fits
by chance); on 64 goals they collapse to 0.26–0.37, and the best-fitting field splits across
distance / direction / height with no winner — and the MLP's dominant mode becomes mostly
`goal_dir_y`, the *opposite* of the 5-goal story. **Conclusion: no coarse physical field
(direction, distance, height) cleanly explains the dominant goal mode.** The content difference
of Finding 2 is real, but its physical interpretation is *not* "direction vs distance," and
remains unnamed.

**What may still hold (separate, more direct measurement).** Scoring the action each policy
actually takes by its goal-directed one-step transport `T_dir = Δx_cube · d_goal` over the eval
goals (a behavioural measurement, not a small-n correlation) put the bilinear policy above both
the MLP and the goal-agnostic play policy at contact/transport (P3: bilinear +0.0049 > MLP
+0.0040 > play +0.0026), which would argue against a pure "scripted-primitive" reading. This is
on the 5 eval goals only and one-step; treat as suggestive, not established, pending a broader
goal set.

*Lesson:* the goal-grid check was run specifically to harden the direction/distance claim and
instead falsified it — a small-n artifact caught before it entered the paper. The physical
identity of the bilinear basis is still open; the honest next step is a causal keep/remove
value surgery against simulator outcomes, not another correlational label.

## Finding 4 — the causal surgery: attempted, two design walls, no result yet

*(`scripts/atlas/value_surgery.py`, `induced_surgery.py`, `physical_fields.py`. Status, not a
result.)* We began the causal keep/remove test — select actions by a candidate operator
(`dir` = goal-directed transport, `hold` = control retention) vs the full bilinear value vs the
MLP, and score by **simulator outcome**. Two designs, both instructive:

1. **First-action override** (fast): override only the first action, then continue with the
   bilinear flow actor. **Does not discriminate** — every operator gave identical 0.33 success,
   because the strong continuation policy washes out a single first-action choice. Not a causal
   test.
2. **Induced-policy rollout** (correct): each operator *drives* the policy at every step; the
   physical candidates are re-scored by one-step simulator branching on a scratch env (keeping
   the rollout exact). **Correct but expensive** — the branching is `O(H × N × episodes)` of
   sequential MuJoCo steps, and a first-pass implementation ran >5 min for only 6 states. A real
   run is multi-hour, not the ~1 hr first estimated.

**Nothing is concluded from this yet.** The causal identity of the bilinear basis remains open.
Options on the table: a scoped-down induced run (fewer states / shorter horizon, wide error
bars, ~30–60 min) for a first directional read, or the full multi-hour build, or defer.

---

## What we are seeing, in one paragraph

The bilinear and MLP critics are, spectrally, the same kind of object: both express a single,
~6.7-dimensional-effective preference over actions per state, not a multi-modal landscape. What
separates them is **content** (the goal-mode subspaces diverge by far more than seed noise —
53.7° vs a ~20° floor on the dominant mode) and **off-reference behaviour** (the bilinear
critic places its action-maximum *off* the behavioural manifold, valuing the DDPG+BC action
above every reference action 79% of the time, +2.14; the MLP does not, and the two critics'
action-gradients are equally steep, so it is a peak and not a slope). The flow actor is
compatible with bilinear precisely because it never leaves the μ support to reach that peak,
whereas a Q-ascending Gaussian actor climbs straight to it — which is why bilinear *needs* the
flow actor and the MLP does not. The multi-modality story is off the table. We tried to give the
goal-preference difference a physical name (direction vs distance) but that did **not** survive a
64-goal check and is retracted (Finding 3); the difference is real and calibrated (53.7° vs a
~20° seed floor) but its physical identity remains open. In one line: an off-support critic peak
explains why bilinear needs the flow actor; a real-but-not-yet-named difference in the goal
factor `ψ(g)` is the remaining lead on why it outperforms the MLP.

---

## How the oracle experiment was constructed (and why it is inconclusive so far)

The analyses above establish *structure*. To validate that bilinear's structure is actually
*better* — i.e. that it ranks behaviour-supported actions in agreement with ground truth — we
built a privileged simulator oracle. It is worth documenting in full, including its current
failure.

### The construction

**Counterfactual branching by reset + prefix-replay** (`scripts/atlas/prefix_replay.py`). The
cube env holds an IK-target integrator *outside* MuJoCo's `mjData` that is not in the dataset,
so snapshot/restore of `qpos/qvel` is **not** bit-exact (we tried; it drifts). Instead we
reconstruct any state by replaying its recorded action prefix from a pinned
`reset(seed, task_id)`. Verified **bit-exact (0.0 divergence), including counterfactual
branching** — replay a common prefix, then take different first actions, and every rollout is
reproducible. This lets us evaluate many actions from one physical state.

**The oracle** `J^MPC_μ(s, a, g)` (`scripts/atlas/oracle.py`). From the state reached by the
prefix, execute the candidate first action *a*, then estimate the achievable return by rolling
the frozen μ forward and scoring by a discounted goal-hitting reward (cube within 4 cm of the
target = success; `gc_negative` reward so `Q` is a negative discounted pre-hit cost).
Continuation actions are **proposed from μ** (stay on-manifold) and the estimate is **selected
by realised return, critic-free** — the intent being an *in-support optimal* the IQL critic is
meant to approximate, not the behaviour value. A pool of envs steps in lockstep so the cost is
O(horizon) model calls per estimate, not O(N·horizon).

**Primary A** (`scripts/atlas/primaryA.py`) then measures, over the μ-reference action set, the
Spearman correlation between the critic's atlas `A^m_s(a,g)` and the oracle `J^MPC_μ(s,a,g)`,
per factorial cell (C1 ID/ID, C2 ID-start/OOD-goal, C4 OOD-start/ID-goal, C3 = the benchmark).

### Why it is inconclusive

The 2-hour run returned **ρ ≈ 0 for every critic in every cell** (C2: bilinear 0.011, MLP
−0.029; C3: bilinear −0.005, MLP 0.022). This is **not** "bilinear and MLP rank equally" — it
is a broken discriminator, and we verified the mechanism directly. Over 8 actions at C2 the
oracle returns:

    J = [ −33.1, −33.1, −33.1, −33.1, −33.1, −29.35, −29.35, −28.77 ]

5 of 8 actions sit at the **floor** (−33.1); only 3 distinct values. Spearman over data that is
60%+ tied at one value is ~0 regardless of the critic being tested.

**Root cause.** The oracle continuation is **goal-agnostic** μ. It does not steer toward *g*,
so whether the cube reaches *g* within the horizon depends on luck rather than the first
action — most actions never reach the goal and land at the floor. The sparse discounted-hitting
reward gives **no gradient** for the majority of actions.

**The fix (next).** Replace the sparse reward with a **graded** `J = −(min cube-to-goal
distance over the rollout)`. Every action then gets a continuous score reflecting how close
behaviour-continuation got to the goal — no floor, and Spearman becomes meaningful. This also
permits a shorter horizon (the graded signal needs less of it), which would cut the runtime
that made the first attempt a 2-hour crawl.

Crucially, **Findings 1 and 2 do not depend on the oracle J ranking** and are unaffected by
this. The oracle's job was to *validate* the supported-ranking claim; that specific validation
is pending the graded-reward fix. The mechanism story — off-reference exploitation plus a
content (not shape) difference — already stands on the oracle-free evidence.

---

## Summary

- **Solid (oracle-free):** the actor reversal is **off-reference critic exploitation** — the
  bilinear critic's action-maximum lies *off* the μ support (Q(a_DDPG) beats every reference
  action 79% of the time, +2.14; MLP 6%, −0.26), it is a genuine peak and not a steeper
  landscape (equal action-gradients, 3.18 vs 3.06), the DDPG+BC actor climbs to it (17.9 vs
  1.76 whitened departure) and the flow actor is anchored away from it.
- **Solid (oracle-free):** **multi-modality is refuted** — all critics unimodal, identical
  spectral shape (effective rank ~6.7–7.0, top-sv ~0.49).
- **Solid, seed-calibrated:** bilinear and MLP span **different goal-preference content** —
  dominant-mode principal angle 53.7° vs a ~20° same-seed floor and a 71° random ceiling.
- **Retracted:** the physical *name* for that content — "bilinear=direction, MLP=distance" —
  was a 5-eval-goal small-n artifact (\|r\| 0.75→0.37 on a 64-goal grid, best-field scatters).
  The content difference is real; its physical identity is **still open**.
- **Attempted, no result yet:** the **causal keep/remove value surgery** (Finding 4) — first-
  action override doesn't discriminate; induced-policy rollout is correct but multi-hour. The
  physical identity of the basis difference is still open; a scoped run or full build is pending
  a decision.
- **Also pending:** the simulator oracle's **graded-reward** fix for Primary A; **FB (torch)**;
  **Phase 0B**.
