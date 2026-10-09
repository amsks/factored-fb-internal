# Factorization probe (state) — R^2(feature -> cube-to-goal d)

**Verdict: B2** — d is linearly recoverable from B (R^2 0.68); failure is downstream (ρ-sampling / z=B(g) / actor)

Per-task verdicts: {'task1': 'B2', 'task2': 'B3-form', 'task3': 'B3-form', 'task4': 'B2', 'task5': 'B2'}

## Transport-phase readout ceilings (mean R^2 over tasks)

| feature | kind | mean R^2 |
| :-- | :-- | :-- |
| B | linear | 0.684 |
| B | mlp | 0.798 |
| left_enc | linear | 0.540 |
| left_enc | mlp | 0.666 |
| raw | linear | 0.964 |
| raw | mlp | 0.986 |
