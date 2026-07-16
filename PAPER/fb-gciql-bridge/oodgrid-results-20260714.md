# OOD grid + the ψ(s,g) expressivity ablation — results (2026-07-14)

**What we did:** we re-evaluated every trained agent in the programme on a 2×2 grid that
crosses *where the episode starts* (a dataset state vs. the environment's own `reset()`)
with *what goal it is given* (a state drawn from the data vs. the benchmark's evaluation
task), so that the benchmark score — which shifts both at once — can be split into the part
lost to each shift separately.

Two campaigns launched 2026-07-13, both collected. Together they relocate the
bilinear-vs-MLP effect from the **start** axis (where 2026-07-13 put it) to the
**goal** axis, and identify the low-rank goal bottleneck as the load-bearing piece.

Full per-cell table: **`analysis/oodgrid/oodgrid_bridge_table.csv`** (113 rows — 26 canonical
agents + 87 bridge experiments), regenerate with
`.venv/bin/python scripts/oodgrid_bridge_table.py [--md|--csv]`.

---

## 1. The grid

All cells share byte-identical start states and goals (`analysis/oodgrid/cells.npz`).
Protocol matched to the env's own eval so **C3 reproduces published success**: H=200,
success = `‖cube − target‖ ≤ 0.04 m` (`cube_env.py:715`).

|                       | ID goal (hindsight) | OOD goal (eval task) |
|-----------------------|---------------------|----------------------|
| **ID start** (data)   | **C1**              | **C2**               |
| **OOD start** (reset) | **C4**              | **C3** = the real eval |

    dGOAL  = C2 − C1      goal shift alone
    dSTART = C4 − C1      start shift alone
    INTER  = (C3 − C1) − (dGOAL + dSTART)

Anchors validate the harness: MLP C3 0.747 (pub 0.717), bilinear-d50 C3 0.947
(pub 0.904), FB 0.323 (pub 0.420).

Coverage: **133 groups / 347 run-dirs**; 14 errored (out-of-scope agents, recorded not
dropped). Aggregate with `scripts/oodgrid_report.py`.

---

## 2. The discriminating axis is the GOAL shift, not the start shift

This **corrects the 2026-07-13 reading**, which had only C1/C2/C3 and therefore could not
separate the two shifts. With C4 in hand they separate cleanly:

| group                          | C1    | C2    | C4    | C3 (eval) | dGOAL      | dSTART |
|--------------------------------|-------|-------|-------|-----------|------------|--------|
| GCIQL-**bilinear** (d50)       | 0.867 | 0.930 | 0.773 | **0.947** | **+0.063** | −0.093 |
| GCIQL-**bilinear** (d128)      | 0.890 | 0.937 | 0.823 | 0.933     | **+0.047** | −0.067 |
| GCIQL-**MLP** (`-State-Flow`)  | 0.980 | 0.823 | 0.787 | 0.747     | **−0.157** | −0.193 |
| GCIQL-**MLP** (`factored-fb-`) | 0.983 | 0.747 | 0.770 | 0.677     | **−0.237** | −0.213 |

- **dGOAL is the separator.** Every one of the ~20 bilinear cells in the table sits at
  `dGOAL ∈ [−0.08, +0.08]` — the goal shift is essentially **free** for them, and often
  *helps* (the 5 canonical cube tasks are more structured than random hindsight goals).
  Both MLP cells lose **16–24 points** to the same shift. Replicated across two
  independently-named MLP groups.
- **dSTART hurts everyone** (−0.07 to −0.21) and is *not* where the two families part:
  bilinear loses ~half what MLP does, but neither is immune. The 2026-07-13 claim of a
  "pure OOD-START interaction" does not survive the factorial.
- **Caveat: INTER is large and positive** (+0.11 to +0.14) for both families, so the
  additive decomposition is leaky — the two shifts are not independent and the C3 value is
  consistently *better* than dGOAL+dSTART predicts. Read the dGOAL contrast as a strong
  ordinal signal, not a variance decomposition.
- Nothing shows a genuine crossover (`dSTART > 0`) except two ISFB cells pinned at the
  floor (~0.18), where it is meaningless.

### ⚠ Do not read dGOAL/dSTART on any torch FB row

For FB, `zR` (reward-inference z) is undefined for hindsight goals, so the report takes
**C1/C4 from the `zB` route and C2/C3 from `zR`** — the two are not interchangeable (0.05
vs 0.40 on C3). Every torch-FB `dGOAL`/`dSTART` is therefore **cross-route and
uninterpretable**. This manufactures spectacular artifacts: `fb_bridge_idB_dnorm` shows
C1 0.917 → C2 0.020 (`dGOAL = −0.897`), which is a change of z-route, not a change of goal.
Only the jax rows (single route) support the decomposition.

---

## 3. ψ(s,g): the goal bottleneck is a *stabilizer* (`GCIQL-State-Flow-Bilinear-SG-d50`)

Pre-registered test — keep the inner-product head, destroy the low-rank **goal**
bottleneck: `Q = φ(s,a)·ψ(s,g)/√d`. Same rank d=50, same flow actor, same IQL objective.
Prediction: *if it lands ≈ MLP (0.717), the bottleneck — not the inner-product head — is
what buys the bilinear win.*

3 seeds × 500k. Final: **0.692 / 0.284 / 0.000** → mean **0.325 ± 0.394**
(vs bilinear-d50 **0.904**, MLP **0.717**).

The spread is not seed noise — it is a **late-training critic divergence**:

| seed | 100k | 200k | 300k | 400k | 500k | fate |
|------|------|------|------|------|------|------|
| 0    | 0.51 | 0.60 | 0.75 | 0.68 | **0.69** | stable (`q_mean` ≈ −34 throughout) |
| 1    | 0.56 | 0.65 | 0.05 | 0.00 | 0.28 | diverges ~300k |
| 2    | 0.57 | 0.59 | 0.65 | 0.18 | 0.00 | diverges ~400k |

All three train normally to 0.56–0.65 by 200k (group mean at 200k = **0.615**). Then on
seeds 1–2 `value_loss` jumps **0.15 → 1–5** and `q_mean` drifts from −34 toward zero and
**crosses positive** (+11 on seed 2) — impossible for a discounted negative-reward GC-IQL
value, whose V is bounded ≤ 0. Success collapses in lockstep. `bc_flow_loss` is flat
throughout (~0.15), so the actor is not the source.

**Reading.** Removing the goal bottleneck does not merely revert the critic to MLP quality —
it makes the IQL fixed point **unstable**, and 2/3 seeds blow up. The one seed that never
diverged landed at **0.69 ≈ MLP's 0.717**, exactly as predicted. So:

- The **bottleneck**, not the inner-product head, is what buys the bilinear win. *(n=1
  stable seed — suggestive, not established. The mean is divergence-driven and should not be
  quoted as a policy-quality number.)*
- The bottleneck is **also doing regularisation work**: ψ(g) constrains the value function
  enough to keep the IQL backup contractive; ψ(s,g) does not. **This is the new claim, and
  it is well-supported** (2/3 seeds, 10–30× value-loss blowup, sign violation).

This dovetails with §2: the structure that buys goal-shift robustness (`dGOAL ≈ 0`) is the
same structure whose removal destabilises the critic.

---

## 4. Open

1. **Mechanism still unknown** for *why* the low-rank goal bottleneck confers both
   goal-shift robustness and stability. The obvious "MLP extrapolates off-manifold, flow
   actor chases it" story was **refuted** on 2026-07-13 (action-ascent gain flat IN→OOD).
2. Seed-0 bilinear_sg is the only non-diverged ψ(s,g) model. Push it through the OOD grid:
   does it inherit the **MLP's dGOAL fragility**? If yes, that closes the loop — same
   performance *and* same failure profile ⇒ the bottleneck is the whole story.
   ⚠ Its checkpoint is in **`/dev/shm/gciql_outputs/`** (volatile) — back up before reboot.
3. Unexplained: `bilinear + DDPGBC = 0.51 < MLP + DDPGBC = 0.70`, which the goal-bottleneck
   story does not predict.
4. FB's C1 was run with `z = B(g)`; a fully generous FB wants C1 re-run with reward-inference
   z (needs a task-shaped hindsight goal set).
