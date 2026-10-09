# Factorization probe (state) — R^2(feature -> cube-to-goal d)

**Verdict: B2** — d is linearly recoverable from B (R^2 0.89); failure is downstream (ρ-sampling / z=B(g) / actor)

Per-task verdicts: {'task1': 'B2', 'task2': 'B2', 'task3': 'B2', 'task4': 'B2', 'task5': 'B2'}

## Transport-phase readout ceilings (mean R^2 over tasks)

| feature | kind | mean R^2 |
| :-- | :-- | :-- |
| B | linear | 0.890 |
| B | mlp | 0.892 |
| left_enc | linear | 0.965 |
| left_enc | mlp | 0.989 |
| raw | linear | 0.963 |
| raw | mlp | 0.987 |
