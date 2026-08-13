# Play Data in Robot Learning: A Literature Survey

_Compiled 2026-08-05 (web survey, agent-assisted). Companion to specs/15._

## (a) Definition of "play data" as used in the field

The founding definition is from Lynch et al. (2019), *Learning Latent Plans
from Play*: play data is **"an unbounded sequence of states and actions
corresponding to voluntary, repeated, non-stereotyped object interaction
between an agent and its environment."** Operationally, in nearly every paper
in this line, it means:

- **Human-teleoperated, undirected interaction**: an operator drives the robot
  in a scene "interacting with all the objects available in as many ways that
  they can think of," with no predefined task distribution.
- **One long, non-episodic stream**: "no scene staging, no task segmenting,
  and no resetting the scene to an initial state." Cut into windows and
  relabeled in hindsight (reached states become goals).
- **Task-agnostic and label-free**: no rewards, no task IDs, no success
  annotations; "contains both functional and non-functional behavior."
- **Multimodal and suboptimal**: many ways of achieving the same outcome,
  plus retries, accidental drops, exploratory dithering.

A second usage (OGBench and successors) applies the word to **scripted
synthetic data designed to mimic these statistics**: a non-Markovian scripted
controller performs a random sequence of atomic manipulations with temporally
correlated noise, long trajectories, no task structure. A third, emergent
usage (2025-2026) is **autonomous robot play / self-play** (PlayWorld).

## (b) Annotated references

### Founding line (Lynch et al. / Google Brain)

1. **Lynch et al. — "Learning Latent Plans from Play" (Play-LMP, Play-GCBC),
   CoRL 2019.** arXiv:1903.01973. Origin paper: defines play data, ~3h
   teleoperated play in a simulated "Playground" tabletop; Play-GCBC
   (hindsight-relabeled GC-BC) + Play-LMP (latent-plan VAE for
   multimodality). One play policy beats 18 per-task expert-BC models (85.5%
   vs 70.3%). Claims: play is "cheap ... without task segmenting, labeling,
   or resetting"; "covers ~4x more interaction space than task demonstrations
   for the same collection time"; play models "more robust to perturbations
   and exhibit retrying-till-success."
2. **Dinyari, Sermanet, Lynch — "Learning to Play by Imitating Humans,"
   2020.** arXiv:2006.06874. Cloned play augments human play.
3. **Lynch & Sermanet — MCIL / LangLfP, RSS 2021.** arXiv:2005.07648.
   Language over relabeled play windows; "3D Playroom" environment.
   "Teleoperated play collection directly addresses the state space coverage
   problem."
4. **"Interactive Language" / Language-Table, 2022.** arXiv:2210.06407.
   ~600k language-labeled tabletop trajectories; unstructured-data +
   relabeling recipe at scale.

### Play benchmarks and datasets

5. **Gupta et al. — Relay Policy Learning (Franka relay-kitchen), CoRL
   2019.** arXiv:1910.11956. "Unstructured and unsegmented demonstrations";
   became D4RL FrankaKitchen, the de facto play-style offline testbed.
6. **Mees et al. — CALVIN, RA-L 2022.** arXiv:2112.03227. 24h VR-teleop play,
   ~2.4M steps, 4 scenes, ~1% language-annotated. "Unscripted playful
   interactions ... task-agnostic, diverse, and relatively cheap."
7. **Rosete-Beas et al. — TACO-RL, CoRL 2022.** arXiv:2209.08959. First
   offline RL (not BC) on play; 9h real Franka play ("TACO play dataset").
8. **Mees et al. — HULC, RA-L 2022.** arXiv:2204.06252. What matters when
   learning from play (ablations on CALVIN).
9. **Young, Pinto et al. — "Playful Interactions for Representation
   Learning," IROS 2022.** arXiv:2107.09046. Human (hand-tool) play for
   representation pretraining.
10. **BeT (NeurIPS 2022, arXiv:2206.11251) / C-BeT (ICLR 2023,
    arXiv:2210.10047).** Play as a multimodal generative-modeling problem;
    C-BeT: first real-robot task behavior purely from play.
11. **MimicPlay, CoRL 2023.** arXiv:2302.12422. Human hand-video play trains
    the planner; few robot demos train the controller.
11b. **PLATO — object-centric play (added from external review 2026-08-05,
    verify citation before use).** Decomposes play into pre-interaction /
    interaction / post-interaction phases, where post-interaction includes
    the object settling to rest and becomes the next behavior's
    pre-interaction. The published decomposition closest to our
    dwell/persistent-occupancy structure — cite in the phenomenon section.
12. **LUMOS, ICRA 2025.** arXiv:2503.10370. World model on CALVIN play.
13. **PlayWorld — "Learning Robot World Models from Autonomous Play,"
    2026.** arXiv:2603.09030, robot-playworld.github.io. Fully autonomous
    self-play (VLM-proposed instructions + generalist policy + auto-resets)
    for video world models; contrast class = "success-biased human
    demonstrations"; play captures "long-tailed physical interactions."

### OGBench and successors

14. **Park, Frans, Eysenbach, Levine — OGBench, ICLR 2025.**
    arXiv:2410.20092. **Explicitly uses "play" and cites Lynch et al. 2019:**
    "'play'-style datasets (play) collected by non-Markovian expert policies
    with temporally correlated noise." Cube: "a scripted policy that
    repeatedly picks a random block and places it in other random locations
    or on another block." `noisy` variants = Markovian policies +
    uncorrelated Gaussian noise. Design axes: goal stitching, stochastic
    control, long-horizon reasoning.
15. **Horizon Reduction / SHARSA, NeurIPS 2025.** arXiv:2506.04168. 1B-step
    OGBench-style play datasets; play-style trajectories "to ensure
    sufficient coverage and diversity"; even 1000x data doesn't rescue
    standard offline RL — horizon is the bottleneck.
16. **2025-2026 consumers of cube/scene-play** (adoption evidence): Horizon
    Adaptive Policy Learning (2606.21136), Latent Representation Alignment
    (2605.25740), Dual Goal Representations (2510.06714), QHyer
    (2605.01862), Dual Advantage Fields (2606.04188), Adaptive/Decoupled
    Q-Chunking (2605.05544 / 2512.10926). They inherit OGBench's framing;
    none re-analyze the data.

### Dataset-structure analyses

17. **ExORL (arXiv:2201.13425):** reward-free exploration data relabeled post
    hoc flips which offline algorithms work — "data generation is as
    important as algorithmic advances."
18. **Schweighofer et al., CoLLAs 2022 (arXiv:2111.04714):** Trajectory
    Quality × State-Action Coverage coordinates; algorithm families
    partition along them. Play = high SACo, low-medium TQ.
19. **Belkhale, Cui, Sadigh, NeurIPS 2023 (arXiv:2306.02437):** action
    divergence + transition diversity as data-quality axes for imitation.
20. **Ghugare et al., ICLR 2024 (arXiv:2401.11237):** stitching =
    combinatorial generalization; play streams require TD because (state,
    goal) pairs never co-occur on one trajectory.
21. **Park et al., NeurIPS 2024 (arXiv:2406.09329):** the offline-RL
    bottleneck is policy extraction + test-time generalization;
    "suboptimal but high-coverage data" mitigates it.
22. **robomimic, CoRL 2021 (arXiv:2108.03298):** human-data properties vs
    BC/offline-RL (adjacent, not play).

## (c) Standard justifications for play data

1. **Cheap/scalable** (no staging, labels, resets) — Lynch 2019, CALVIN,
   TACO-RL, RPL, MimicPlay.
2. **Coverage** — Lynch 2019 ("~4x interaction space"), MCIL, Horizon
   Reduction, Park 2024, PlayWorld.
3. **Task-agnostic** (no task list at collection time) — Lynch 2019,
   TACO-RL, ExORL.
4. **Natural fit for hindsight/goal-conditioned learning** — Play-GCBC, RPL,
   MCIL, OGBench; Ghugare 2024 adds: play *requires* stitching, so TD.
5. **Multimodality as signal** — Lynch 2019, CALVIN, BeT/C-BeT.
6. **Suboptimality as robustness fuel** (retries, recoveries) — Young 2021,
   CALVIN, Lynch 2019, PlayWorld.
7. **Developmental-psychology framing** — Lynch 2019, MimicPlay (rhetorical).

## (d) "Playworld" disambiguation

1. The Lynch-line simulated environment is the **"Playground" (2019) / "3D
   Playroom" (2021)** — not "playworld". CALVIN's desk scene and OGBench's
   `scene` domain are recognizable descendants.
2. **PlayWorld proper = Yin et al. 2026** (arXiv:2603.09030): autonomous
   robot self-play at scale for video world models.

## (e) Statistical structure of play data — the gap

- Published vocabulary ends at: coverage/quality coordinates (TQ×SACo),
  action divergence, stitching demand, noise correlation (OGBench's
  play-vs-noisy contrast), scaling behavior (Horizon Reduction).
- **No paper we found publishes visitation/dwell/revisit statistics of any
  play dataset (CALVIN, TACO, OGBench play) or connects such statistics to
  downstream GCRL performance.** The dwell/renewal analysis in specs/15 §4
  appears to be unclaimed territory.
