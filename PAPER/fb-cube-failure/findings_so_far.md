# Findings so far

## Observation 1 — Where FB fails on cube manipulation

We train FB and GCIQL on OGBench `cube-single` and evaluate zero-shot success on the 5 held-out eval tasks (task1–task5). GCIQL is the strongest goal-conditioned baseline in this regime, so we use it as our reference. We run both algorithms from states and from pixels.

### 1. Overall success — FB vs GCIQL

IQM ± half-width of the 95% stratified-bootstrap CI across seeds; 10 evaluation rollouts per task per seed.

**State (n=10 seeds, final checkpoint at 1M steps):**

| Method | task1 | task2 | task3 | task4 | task5 | overall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| FB    | 16.7 ± 10.0 | 68.3 ± 10.9 | 75.0 ± 10.0 | 40.0 ± 13.3 | 43.3 ± 18.4 | **50.0 ± 7.0** |
| GCIQL | 73.7 ± 5.9  | 70.0 ± 7.0  | 73.0 ± 7.0  | 58.0 ± 5.2  | 64.3 ± 7.0  | **68.3 ± 3.7** |

FB trails GCIQL by ~18 points overall, with the largest per-task gap on task1.

**Pixels, matched DrQ encoder (n=5 seeds, 500k steps):**

| Method | task1 | task2 | task3 | task4 | task5 | overall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| FB    | 36.7 ± 28.4 | 43.3 ± 18.4 | 53.3 ± 21.7 | 23.3 ± 15.0 | 40.0 ± 23.3 | **40.7 ± 13.0** |
| GCIQL | 51.3 ± 6.0  | 36.7 ± 15.0 | 25.3 ± 16.4 | 49.3 ± 5.4  | 38.0 ± 6.7  | **40.9 ± 5.1**  |

With matched encoders, the pixel gap closes to ≈0. So under fair comparison FB ≈ GCIQL at pixels and FB < GCIQL by ~18 points at states. The remainder of this section drills into *where in an eval rollout* those state-side points are lost.

### 2. The four ways a cube episode can fail

A cube-single episode unfolds in four phases. Failing in any one of them gives a distinct, mechanistically different failure mode:

1. **Failure to approach** — the gripper never gets close enough to the cube.
2. **Failure to grasp** — the gripper reaches the cube but never closes around it and lifts it off the table.
3. **Failure to maintain grasp** — the cube *is* picked up, but it ends up back on the table before the episode ends (the agent drops it or it slips out).
4. **Failure to transport** — the cube is picked up and held to the end of the episode, but it is not delivered to the goal pose.

We measure each from the per-step gripper, cube, and goal positions of a deterministic rollout:

| Phase | Operational criterion |
| :--- | :--- |
| approach | minimum gripper → cube distance < 6 cm anywhere in the episode |
| grasp | cube lifted > 3 cm above the table with the gripper closed for ≥ 5 consecutive steps |
| maintain grasp | at the final step, cube height above the table ≥ 3 cm |
| transport (to goal) | at the final step, cube within 5 cm of the goal pose (one OGBench success-tolerance) |

These four phases partition every episode into exactly one outcome: success, or failure at the *earliest* phase that breaks down.

### 3. How often does each failure happen — FB state

Across 7 state seeds × 5 tasks × 10 episodes = 350 S0 evaluation rollouts:

| Outcome | mean (% of all episodes) | std |
| :--- | ---: | ---: |
| success                          | 51.1 | 10.8 |
| failure to approach              | **0.0**  | 0.0 |
| failure to grasp                 | **7.7**  | 3.4 |
| failure to maintain grasp        | **23.7** | 9.9 |
| failure to transport to the goal | **17.4** | 6.6 |

As share of failures only:

| Failure mode | mean (% of failures) | std |
| :--- | ---: | ---: |
| approach        | 0.0  | 0.0  |
| grasp           | 15.6 | 5.1  |
| maintain grasp  | **48.2** | 13.2 |
| transport       | **36.2** | 12.2 |

Per-seed (% of all S0 episodes for that seed):

| seed | success | approach | grasp | maintain | transport |
| :--- | ---: | ---: | ---: | ---: | ---: |
| s3  | 64 | 0 | 4  | 18 | 14 |
| s4  | 36 | 0 | 10 | **42** | 12 |
| s5  | 44 | 0 | 14 | 16 | **26** |
| s6  | 58 | 0 | 8  | 24 | 10 |
| s7  | 42 | 0 | 6  | 32 | 20 |
| s8  | 62 | 0 | 6  | 18 | 14 |
| s10 | 52 | 0 | 6  | 16 | **26** |

### 4. What this says

- **Approach is solved.** Every one of 350 rollouts, the gripper makes contact with the cube. The closest gripper-to-cube distance is < 5 mm in 96% of rollouts; the worst single episode across all seeds and tasks is 4.3 cm (still inside the 6 cm approach threshold).
- **Grasp is mostly solved.** Only ~8% of episodes fail to lift the cube.
- **The two real failure modes are downstream of pick-up.** ~24% of all episodes (≈half of all failures) pick the cube up and then drop it before reaching the goal. Another ~17% (≈a third of failures) carry the cube to the end of the episode but end far from the goal.
- **FB never gets the held cube near the goal.** Across the 350 rollouts there is *not a single episode* where the final cube position is within one success-radius (5 cm) of the goal pose while the cube is still in the gripper. The transport bucket is entirely "wrong place," not "close miss." So the deficit is not a fine-resolution placement problem at the goal — FB does not reach the goal *at all* when carrying the cube.
- **Cross-seed.** Which downstream mode dominates varies by seed. In 5/7 seeds (s3, s4, s6, s7, s8) "maintain grasp" is the largest failure; in 2/7 (s5, s10) "transport" is larger. The "approach" and "grasp" buckets stay small everywhere.
- **Per-task.** task1 (FB's worst task at 16.7% IQM) is dominated by maintain-grasp failures — the policy picks the cube up and then drops it. task2 and task3 (FB's best tasks) lose few episodes to either downstream mode.

We also run two control scenarios on the same FB checkpoints:

- **S1** — identical to S0 (sanity control). Success ≈ S0.
- **S2** — the cube is initialised pre-grasped in the gripper. If transport were easy once the cube is held, S2 success should jump.

Cross-seed aggregate (mean ± std, %):

| Scenario | reached | secured | success |
| :--- | :---: | :---: | :---: |
| S0 baseline    | 100.0 ± 0.0 | 92.3 ± 3.4 | 51.1 ± 10.8 |
| S1 control     | 100.0 ± 0.0 | 95.7 ± 1.7 | 50.0 ± 8.3 |
| S2 pre-grasped | 100.0 ± 0.0 | 85.1 ± 8.7 | 35.7 ± 8.2 |

S2 does **not** boost success above the S0 baseline. Handing FB a held cube does not rescue it. So the downstream failure is not "the handoff is fragile" — it is the policy/value failing to steer the held cube toward the goal.

### 4b. How often does each failure happen — GCIQL state

Same 4-phase breakdown, run on the GCIQL state checkpoints (10 seeds × 5 tasks × 10 episodes = 500 S0 evaluation rollouts):

| Outcome | mean (% of all episodes) | std |
| :--- | ---: | ---: |
| success                          | 69.2 | 8.1 |
| failure to approach              | 0.2  | 0.6 |
| failure to grasp                 | 12.2 | 6.2 |
| failure to maintain grasp        | 9.2  | 4.1 |
| failure to transport to the goal | 9.2  | 3.4 |

Per-seed (% of all S0 episodes for that seed):

| seed | success | approach | grasp | maintain | transport |
| :--- | ---: | ---: | ---: | ---: | ---: |
| s1  | 64 | 2 | 12 | 10 | 12 |
| s2  | 80 | 0 | 4  | 6  | 10 |
| s3  | 62 | 0 | 16 | 14 | 8  |
| s4  | 80 | 0 | 12 | 4  | 4  |
| s5  | 68 | 0 | 6  | 12 | 14 |
| s6  | 78 | 0 | 10 | 6  | 6  |
| s7  | 70 | 0 | 8  | 14 | 8  |
| s8  | 70 | 0 | 12 | 4  | 14 |
| s9  | 64 | 0 | 16 | 14 | 6  |
| s10 | 56 | 0 | 26 | 8  | 10 |

Like FB, GCIQL has **zero "near miss" episodes**: across 500 rollouts there is not a single case where the cube ends within 5 cm of the goal pose while still in the gripper. So "no held-cube ever reaches the goal pose neighborhood" holds for both methods.

### 4c. Side-by-side and what closes the 18-point gap (state)

| Outcome | FB (% eps) | GCIQL (% eps) | FB − GCIQL (pp) |
| :--- | ---: | ---: | ---: |
| success                          | 51.1 | 69.2 | **−18.1** |
| failure to approach              | 0.0  | 0.2  | −0.2 |
| failure to grasp                 | 7.7  | 12.2 | **+4.5** |
| failure to maintain grasp        | 23.7 | 9.2  | **−14.5** |
| failure to transport to the goal | 17.4 | 9.2  | **−8.2** |

Reading the third column: a negative number on a failure row means FB loses *more* episodes to that mode than GCIQL does. The 18.1 pp overall success gap decomposes as

```
−18.1  =  −0.2 (approach)  +4.5 (grasp)  −14.5 (maintain)  −8.2 (transport)
```

i.e. **the entire FB-vs-GCIQL state gap comes from the two downstream failures.** FB is actually *better* at the grasp phase than GCIQL by ~4.5 pp; approach is a tie at zero for both. All 18 (and more, given the grasp-side advantage) points are spent on:

- FB dropping a held cube ~2.6× more often than GCIQL (23.7% vs 9.2%) — the largest single contributor.
- FB carrying a held cube to the wrong place ~1.9× more often than GCIQL (17.4% vs 9.2%).

### 4d. How often does each failure happen — FB and GCIQL pixel (DrQ)

Same 4-phase breakdown on the pixel checkpoints, both with matched DrQ encoders (n=5 seeds × 5 tasks × 10 episodes = 250 S0 evaluation rollouts per method):

| Outcome | FB pixel (mean, std) | GCIQL pixel DrQ (mean, std) |
| :--- | ---: | ---: |
| success                          | 36.8 ± 8.1   | 49.2 ± 5.9 |
| failure to approach              | 0.4 ± 0.9    | 0.4 ± 0.9  |
| **failure to grasp**             | **42.8 ± 14.6** | **42.8 ± 4.1** |
| failure to maintain grasp        | 13.2 ± 5.2   | 6.8 ± 4.1  |
| failure to transport to the goal | 6.8 ± 7.0    | 0.8 ± 1.1  |

(The per-seed success rates here, computed over 50 episodes/seed in this probe, are slightly higher than the headline §1 IQM, which uses a larger 50-episodes-per-task eval budget; the phase proportions are what matters for this section.)

Per-seed (% of all S0 episodes for that seed):

| seed | FB:success | FB:grasp | FB:maintain | FB:transport | GCIQL:success | GCIQL:grasp | GCIQL:maintain | GCIQL:transport |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s0 | 46 | 30 | 18 | 6  | 50 | 42 | 8  | 0 |
| s1 | 30 | 48 | 18 | 2  | 48 | 40 | 10 | 2 |
| s2 | 44 | 34 | 14 | 8  | 56 | 42 | 0  | 0 |
| s3 | 36 | 36 | 10 | 18 | 40 | 50 | 10 | 0 |
| s4 | 28 | **66** | 6  | 0  | 52 | 40 | 6  | 2 |

The picture at pixels is qualitatively different from the state picture, in a way that matches what's in the agent's observation:

- **State** observations contain explicit `proprio/gripper_opening` and `proprio/gripper_contact` channels alongside the cube xyz. Knowing "am I currently holding the cube" is a direct lookup. State FB grasps fine (~8% fail) and the bottleneck is downstream.
- **Pixel** observations are just the rendered RGB frame with frame-stack 3 — no explicit gripper signal at all. *Both* methods now fail to grasp on ~43% of episodes — grasp is the dominant failure mode at pixels for FB and GCIQL alike.

Two things still survive in the pixel regime:

1. **"Near miss" remains 0** across all 250 pixel episodes for both methods (and across all 500 + 350 state episodes earlier). In 1100 total evaluation rollouts spanning state and pixels, neither method ever finishes a held cube within 5 cm of the goal. The "no fine placement at the goal" finding is regime- and method-independent.
2. **The downstream gap pattern survives**, scaled down by the smaller pool of held cubes:

| Outcome | FB pixel | GCIQL pixel DrQ | FB − GCIQL (pp) |
| :--- | ---: | ---: | ---: |
| success                          | 36.8 | 49.2 | **−12.4** |
| failure to approach              | 0.4  | 0.4  | 0    |
| failure to grasp                 | 42.8 | 42.8 | 0    |
| failure to maintain grasp        | 13.2 | 6.8  | **+6.4** |
| failure to transport to the goal | 6.8  | 0.8  | **+6.0** |

```
−12.4  =  0 (approach)  + 0 (grasp)  − 6.4 (maintain)  − 6.0 (transport)
```

i.e. **at pixels too, 100% of the FB-vs-GCIQL gap lives in the two downstream phases.** Grasp is exactly matched (43% for both — both bottlenecked by the same DrQ encoder); the entire 12.4 pp gap is FB dropping the cube more (×1.9) and carrying it to the wrong place more (×8.5) than GCIQL once a cube is in hand. The state and pixel gaps decompose into the same two failures; only the absolute numbers differ.

### 4e. How many pick-and-place cycles per eval episode?

A natural question once we know that training data is full of repeated pick-and-place cycles (~11 placements per 1000-step training trajectory, ~9 distinct workspace spots) is whether the eval-time failures look like the policy is *reproducing* that pattern — picking the cube up, placing it somewhere on the table, picking it up again, and so on — instead of completing one transport to the goal.

We count "pick-and-place cycles" per eval episode the same way as in the training-data analysis: a held-segment of ≥ 5 consecutive eval steps with cube lifted > 3 cm above the table AND gripper closed.

**State, FB (7 seeds × 50 episodes):**

| outcome | n eps | mean cycles | median | % with 1 cycle | % with ≥ 2 | % with ≥ 3 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| success                          | 179 | 1.60 | **1** | 48 | 36 | 22 |
| fail @ grasp                     | 27  | 1.56 | 1     | 19 | 48 | 26 |
| **fail @ maintain (dropped)**    | 83  | **1.86** | **2** | 30 | **52** | 31 |
| **fail @ transport (held, misplaced)** | 61 | **2.11** | **2** | 18 | **61** | **39** |

**State, GCIQL (10 seeds × 50 episodes):**

| outcome | n eps | mean cycles | median | % with 1 cycle | % with ≥ 2 | % with ≥ 3 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| success                          | 346 | 1.09 | **1** | **65** | 20 | 4 |
| fail @ approach                  | 1   | 0.00 | 0     | 0  | 0  | 0  |
| fail @ grasp                     | 61  | 0.03 | 0     | 3  | 0  | 0  |
| fail @ maintain (dropped)        | 46  | 1.65 | 1     | 57 | 43 | 17 |
| **fail @ transport (held, misplaced)** | 46  | **2.35** | **2** | 30 | **70** | 33 |

**Pixel, FB DrQ (5 seeds × 50 episodes):**

| outcome | n eps | mean cycles | median | % with 1 cycle | % with ≥ 2 | % with ≥ 3 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| success                          | 92  | 1.03 | 1 | 52 | 20 | 8  |
| fail @ approach                  | 1   | 0.00 | 0 | 0  | 0  | 0  |
| fail @ grasp                     | 107 | 0.42 | 0 | 22 | 8  | 3  |
| fail @ maintain (dropped)        | 33  | 1.24 | 1 | 36 | 30 | 18 |
| fail @ transport (held, misplaced) | 17 | 1.00 | 1 | 24 | 29 | 12 |

**Pixel, GCIQL DrQ (5 seeds × 50 episodes):**

| outcome | n eps | mean cycles | median | % with 1 cycle | % with ≥ 2 | % with ≥ 3 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| success                          | 123 | 0.90 | 1 | **89** | 1  | 0  |
| fail @ approach                  | 1   | 0.00 | 0 | 0  | 0  | 0  |
| fail @ grasp                     | 107 | 0.00 | 0 | 0  | 0  | 0  |
| fail @ maintain (dropped)        | 17  | 1.18 | 1 | 88 | 12 | 6  |
| fail @ transport (held, misplaced) | 2  | 4.50 | 4 | 0  | 100| 50 |

What this says:

1. **At state, eval failures are markedly more multi-cycle than successes.** A successful eval episode uses 1 pick-and-place for both methods (FB 48% / GCIQL 65% with exactly one cycle). A failed-transport episode uses ≥ 2 cycles in **61% of FB cases and 70% of GCIQL cases**, and ≥ 3 cycles in ~33–39%. The policy isn't failing once — it is performing the *training-data behaviour* of "pick, release somewhere, pick again, release again" inside the 200-step eval budget, never converging on the goal pose. This is consistent with what we saw in the training-data analysis: 100% of training trajectories chain ~11 placements at different workspace spots, and the eval policy is reproducing that chain rather than terminating after one transport.

2. **Multi-cycle attempting is not a FB-only behavior at state.** GCIQL transport-fails are *even more* multi-cycle (70% ≥ 2 vs FB's 61%). The difference between methods at the state regime is *not* "FB tries multiple times and GCIQL doesn't"; it's that GCIQL has so many fewer transport-fails to begin with (46 vs 61 episodes, on 1.4× more total eval rollouts).

3. **At pixels, grasp-fails dominate and they have ~0 cycles by construction** (the cube never gets lifted for ≥ 5 steps with a closed gripper). The successful pixel episodes are very clean — GCIQL pixel completes 89% of successes in exactly one cycle (vs FB pixel's 52%), echoing the pattern that once GCIQL gets a hold on the cube it carries it more directly.

4. **The maintain-fail bucket is mostly single-cycle for everyone except FB state.** FB state's maintain-fail median is 2 cycles — the agent drops the cube, picks it up again, drops again. This is FB's signature failure pattern at state, and matches the training-data-like "pick, release, pick, release" chain.

The picture lines up cleanly with the training-data analysis: training trajectories chain ~11 placements at random workspace positions; eval expects "one placement at the goal, then stop". FB state-side failures look exactly like training behavior (median 2 cycles for fail-transport; up to 5). The eval-time deficit is the policy reproducing a *behavior the training data shows abundantly* — repeated picking and releasing — rather than failing at a behavior the training data never demonstrates.

### 5. Bottom line

Under matched encoders, the FB-vs-GCIQL gap on cube manipulation collapses to the same **two** downstream failure modes in *both* state and pixel regimes:

> **Once FB has the cube in hand, it either drops it (state: ~24% vs GCIQL ~9%; pixel: ~13% vs ~7%) or carries it nowhere near the goal (state: ~17% vs ~9%; pixel: ~7% vs ~1%). Neither method ever finishes a held cube within one success-radius of the goal — across 1100 evaluation rollouts spanning state and pixels, the "near miss" bucket is empty.**

Gap decomposition by phase:

| | State (FB − GCIQL) | Pixel (FB − GCIQL DrQ) |
| :--- | ---: | ---: |
| Overall success deficit | −18.1 pp | −12.4 pp |
| ...due to approach | −0.2 | 0 |
| ...due to grasp | +4.5 (FB better) | 0 (matched) |
| ...due to maintain grasp | **−14.5** | **−6.4** |
| ...due to transport | **−8.2** | **−6.0** |

The state and pixel pictures differ in *where the upstream failures land* (state: grasp is cheap because gripper signals are in obs ⇒ FB and GCIQL fail mostly downstream; pixel: grasp is hard for both methods at 43% ⇒ fewer held cubes to lose downstream), but the *FB-vs-GCIQL gap itself* is the same shape in both regimes: 100% of it lives in maintain-grasp and transport.

This is what the rest of this document sets out to explain.
