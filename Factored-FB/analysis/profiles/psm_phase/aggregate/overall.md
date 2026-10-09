# Phase-probe — cross-seed aggregate

Seeds: s0, s1, s2, s3, s4, s5, s6, s7, s8, s9  (n=20 episodes per task x scenario)

- **S0**: reached 100.0±0.0%  secured 87.9±1.4%  success 28.6±4.9%
- **S1**: reached 100.0±0.0%  secured 85.1±5.2%  success 28.8±5.0%
- **S2**: reached 100.0±0.0%  secured 78.9±5.2%  success 20.7±3.9%

## S0 baseline failure attribution

- success: 28.6% of episodes
- fail@transport: 59.3% of episodes
- fail@grasp: 12.1% of episodes

## Readout

Hypothesis was: reach hard? no; grasp/pick-up is the bottleneck; transport easy once held. Evidence: reach is ~100%, secure (grasp+lift) is high, and S0 failures concentrate at **transport**, not grasp. S2 (pre-grasped, off-distribution clamp) does not exceed S0. => the dominant failure is delivering/placing the lifted cube at the goal, not acquiring it.