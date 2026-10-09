# FB vs PSM — training task-distribution probe

## Probe 1 — goal pool by interaction bucket (shared)

- no_contact: 0.346
- reaching: 0.202
- grasped: 0.105
- lifted_or_transport: 0.346
- **cube-involved (grasped+lifted): 0.452**

## Probe 2 — task-embedding geometry (mean±std over seeds)

| method | z_dim | participation ratio | z-norm | separability AUC (involved vs free) |
|---|---|---|---|---|
| fb_state_plain | 50 | 48.95±0.01 | 7.07 | 0.741±0.006 |
| psm_flowbc_state | 128 | 68.49±0.91 | 11.31 | 0.939±0.014 |
| psm_o1000 | 128 | 126.61±0.01 | 11.31 | 0.965±0.005 |
| fb_z128 | 128 | 126.51±0.02 | 11.31 | 0.905±0.015 |
| fb_state_iwr | 50 | 31.19±0.06 | 7.07 | 0.886±0.002 |
| fb_state_iwr_wz | 50 | 30.94±0.06 | 7.07 | 0.893±0.004 |

## Probe 3 — eval-task z coverage vs training cloud (mean over seeds)

### fb_state_plain
| task | cos-to-nearest | Mahalanobis |
|---|---|---|
| cube-single-play-singletask-task1-v0 | 0.444 | 33.50 |
| cube-single-play-singletask-task2-v0 | 0.653 | 24.25 |
| cube-single-play-singletask-task3-v0 | 0.589 | 27.63 |
| cube-single-play-singletask-task4-v0 | 0.576 | 27.52 |
| cube-single-play-singletask-task5-v0 | 0.588 | 25.45 |

### psm_flowbc_state
| task | cos-to-nearest | Mahalanobis |
|---|---|---|
| cube-single-play-singletask-task1-v0 | 0.505 | 19.39 |
| cube-single-play-singletask-task2-v0 | 0.518 | 19.18 |
| cube-single-play-singletask-task3-v0 | 0.486 | 19.50 |
| cube-single-play-singletask-task4-v0 | 0.506 | 19.75 |
| cube-single-play-singletask-task5-v0 | 0.509 | 20.45 |

### psm_o1000
| task | cos-to-nearest | Mahalanobis |
|---|---|---|
| cube-single-play-singletask-task1-v0 | 0.416 | 38.36 |
| cube-single-play-singletask-task2-v0 | 0.480 | 38.10 |
| cube-single-play-singletask-task3-v0 | 0.406 | 38.87 |
| cube-single-play-singletask-task4-v0 | 0.479 | 33.30 |
| cube-single-play-singletask-task5-v0 | 0.475 | 35.22 |

### fb_z128
| task | cos-to-nearest | Mahalanobis |
|---|---|---|
| cube-single-play-singletask-task1-v0 | 0.541 | 43.63 |
| cube-single-play-singletask-task2-v0 | 0.481 | 40.50 |
| cube-single-play-singletask-task3-v0 | 0.489 | 41.72 |
| cube-single-play-singletask-task4-v0 | 0.531 | 40.22 |
| cube-single-play-singletask-task5-v0 | 0.511 | 39.02 |

### fb_state_iwr
| task | cos-to-nearest | Mahalanobis |
|---|---|---|
| cube-single-play-singletask-task1-v0 | 0.583 | 7.00 |
| cube-single-play-singletask-task2-v0 | 0.545 | 7.60 |
| cube-single-play-singletask-task3-v0 | 0.541 | 7.45 |
| cube-single-play-singletask-task4-v0 | 0.582 | 7.26 |
| cube-single-play-singletask-task5-v0 | 0.629 | 6.88 |

### fb_state_iwr_wz
| task | cos-to-nearest | Mahalanobis |
|---|---|---|
| cube-single-play-singletask-task1-v0 | 0.586 | 7.38 |
| cube-single-play-singletask-task2-v0 | 0.580 | 7.51 |
| cube-single-play-singletask-task3-v0 | 0.543 | 7.74 |
| cube-single-play-singletask-task4-v0 | 0.566 | 7.56 |
| cube-single-play-singletask-task5-v0 | 0.618 | 7.34 |
