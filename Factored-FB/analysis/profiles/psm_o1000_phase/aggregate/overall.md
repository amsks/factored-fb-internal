# Phase-probe — cross-seed aggregate

Seeds: s0, s1, s2, s3, s4, s5, s6, s7, s8, s9  (n=20 episodes per task x scenario)

- **S0**: reached 99.0±1.2%  secured 79.8±3.8%  success 38.4±5.3%
- **S1**: reached 100.0±0.0%  secured 77.1±5.0%  success 30.9±5.5%
- **S2**: reached 100.0±0.0%  secured 74.6±7.2%  success 27.3±3.7%

## S0 baseline failure attribution

- success: 38.4% of episodes
- fail@transport: 41.4% of episodes
- fail@grasp: 19.2% of episodes
- fail@reach: 1.0% of episodes

## Readout

Hypothesis was: reach hard? no; grasp/pick-up is the bottleneck; transport easy once held. Evidence: reach is ~100%, secure (grasp+lift) is high, and S0 failures concentrate at **transport**, not grasp. S2 (pre-grasped, off-distribution clamp) does not exceed S0. => the dominant failure is delivering/placing the lifted cube at the goal, not acquiring it.