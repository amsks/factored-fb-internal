Results of the six arms (1M steps, seed 0, 10 episodes per task; `results/puzzle_arms.md`):

| arm | 500k mean | 1M mean | 1M per task (1–5) |
|---|---|---|---|
| Density-FB γ=0.99 (actor bootstrap) | 0.12 | 0.22 | 0.6 / 0.3 / 0.1 / 0.0 / 0.1 |
| Density-FB γ=0.995 | 0.18 | 0.12 | 0.3 / 0.0 / 0.1 / 0.0 / 0.2 |
| Density-FB γ=0.998 | 0.24 | 0.24 | 0.5 / 0.2 / 0.3 / 0.0 / 0.2 |
| Density-FB γ=0.95 | 0.08 | – | 0.3 / 0.0 / … (eval in progress) |
| **Density-FB γ=0.99, bootstrap at the dataset's next action** | **0.70** | **0.64** | **0.9 / 0.7 / 0.5 / 0.7 / 0.4** |
| GCIVL (our implementation, AWR alpha 10) | 0.12 | 0.12 | 0.5 / 0.1 / 0.0 / 0.0 / 0.0 |

The horizon dial does nothing, as the probe predicted. Taking the bootstrap action from the
dataset instead of the actor solves every task: 0.64–0.70 against a published ceiling of 18.8
for zero-shot methods and 0.12 for our GCIVL reference on the same data. Mechanism: the critic
now models the play policy's future, which is honest for goals the actor cannot reach; under
random pressing the chance of landing on the goal within the horizon falls with the number of
presses still needed, so the actor's objective is monotone in progress and the flat-J failure
disappears. Caveat from the record: the same bootstrap scored 0.10 on MetaWorld pick-place in
August (`docs/specs/2026-08-17-gciql-edge-attribution.md`), where the play policy's future is
diffuse away from the goal and the data-policy measure is not an improvable objective. So the
two bootstraps fail in complementary ways: the actor bootstrap is optimistic where the actor
cannot reach, the dataset bootstrap is uninformative where the data does not reach.

Confirmation running (tmux `softmax4`, 1M): puzzle seeds 1 and 2 with the dataset bootstrap,
and the dataset bootstrap on cube-single, scene and cube-double to measure what it costs
where the actor bootstrap works. The principled fix remains a target that anchors the
bootstrap with the realised future of the trajectory that produced the hindsight goal.

