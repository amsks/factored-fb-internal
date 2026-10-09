# Cube phase-failure probe — summary

- **cube-single-play-singletask-task1-v0 / S0**: reached 100.0%  secured 50.0%  success 40.0%  (n=20)
- **cube-single-play-singletask-task1-v0 / S1**: reached 100.0%  secured 50.0%  success 35.0%  (n=20)
- **cube-single-play-singletask-task1-v0 / S2**: reached 100.0%  secured 75.0%  success 55.0%  (n=20)
- **cube-single-play-singletask-task2-v0 / S0**: reached 100.0%  secured 75.0%  success 35.0%  (n=20)
- **cube-single-play-singletask-task2-v0 / S1**: reached 100.0%  secured 55.0%  success 20.0%  (n=20)
- **cube-single-play-singletask-task2-v0 / S2**: reached 100.0%  secured 55.0%  success 5.0%  (n=20)
- **cube-single-play-singletask-task3-v0 / S0**: reached 100.0%  secured 100.0%  success 65.0%  (n=20)
- **cube-single-play-singletask-task3-v0 / S1**: reached 100.0%  secured 90.0%  success 40.0%  (n=20)
- **cube-single-play-singletask-task3-v0 / S2**: reached 100.0%  secured 70.0%  success 35.0%  (n=20)
- **cube-single-play-singletask-task4-v0 / S0**: reached 100.0%  secured 70.0%  success 0.0%  (n=20)
- **cube-single-play-singletask-task4-v0 / S1**: reached 100.0%  secured 90.0%  success 0.0%  (n=20)
- **cube-single-play-singletask-task4-v0 / S2**: reached 100.0%  secured 70.0%  success 10.0%  (n=20)
- **cube-single-play-singletask-task5-v0 / S0**: reached 95.0%  secured 60.0%  success 5.0%  (n=20)
- **cube-single-play-singletask-task5-v0 / S1**: reached 100.0%  secured 90.0%  success 25.0%  (n=20)
- **cube-single-play-singletask-task5-v0 / S2**: reached 100.0%  secured 75.0%  success 10.0%  (n=20)

## Hypothesis readout

Prediction: success(S0) ≈ success(S1) (grasp still required) and success(S2) ≫ success(S0) (transport easy once held); baseline reached-rate ≫ secured-rate ⇒ grasp/pick-up is the bottleneck.
- cube-single-play-singletask-task1-v0: S0 reached=100% secured=50% success S0=40% S1=35% S2=55%
- cube-single-play-singletask-task2-v0: S0 reached=100% secured=75% success S0=35% S1=20% S2=5%
- cube-single-play-singletask-task3-v0: S0 reached=100% secured=100% success S0=65% S1=40% S2=35%
- cube-single-play-singletask-task4-v0: S0 reached=100% secured=70% success S0=0% S1=0% S2=10%
- cube-single-play-singletask-task5-v0: S0 reached=95% secured=60% success S0=5% S1=25% S2=10%