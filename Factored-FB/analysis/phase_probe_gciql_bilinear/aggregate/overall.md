# Phase-probe — cross-seed aggregate

Seeds: s0, s1, s2, s4  (n=10 episodes per task x scenario)

- **S0**: reached 100.0±0.0%  secured 96.5±5.0%  success 95.0±5.4%
- **S1**: reached 100.0±0.0%  secured 95.0±1.0%  success 94.5±0.9%
- **S2**: reached 100.0±0.0%  secured 82.0±17.2%  success 79.5±18.9%

## S0 baseline failure attribution

- success: 95.0% of episodes
- fail@transport: 1.5% of episodes
- fail@grasp: 3.5% of episodes

## Readout

Hypothesis was: reach hard? no; grasp/pick-up is the bottleneck; transport easy once held. Evidence: reach is ~100%, secure (grasp+lift) is high, and S0 failures concentrate at **transport**, not grasp. S2 (pre-grasped, off-distribution clamp) does not exceed S0. => the dominant failure is delivering/placing the lifted cube at the goal, not acquiring it.