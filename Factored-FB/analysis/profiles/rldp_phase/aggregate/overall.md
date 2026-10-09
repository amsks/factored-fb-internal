# Phase-probe — cross-seed aggregate

Seeds: s0, s1, s2, s3, s4, s5, s6, s7, s8, s9  (n=20 episodes per task x scenario)

- **S0**: reached 100.0±0.0%  secured 85.1±4.8%  success 26.2±6.9%
- **S1**: reached 100.0±0.0%  secured 87.0±4.9%  success 23.7±5.5%
- **S2**: reached 100.0±0.0%  secured 82.7±8.9%  success 20.3±3.9%

## S0 baseline failure attribution

- success: 26.2% of episodes
- fail@transport: 58.9% of episodes
- fail@grasp: 14.9% of episodes

## Readout

Hypothesis was: reach hard? no; grasp/pick-up is the bottleneck; transport easy once held. Evidence: reach is ~100%, secure (grasp+lift) is high, and S0 failures concentrate at **transport**, not grasp. S2 (pre-grasped, off-distribution clamp) does not exceed S0. => the dominant failure is delivering/placing the lifted cube at the goal, not acquiring it.