# Factorization probe (state) — R^2(feature -> cube-to-goal d)

**Verdict: B3-form** — B-MLP R^2 0.84 but B-linear R^2 0.67: geometry is in B but non-linear; the bilinear FᵀB value (linear in B) cannot read it

Per-task verdicts: {'task1': 'B3-form', 'task2': 'B3-form', 'task3': 'B3-form', 'task4': 'B2', 'task5': 'B2'}

## Transport-phase readout ceilings (mean R^2 over tasks)

| feature | kind | mean R^2 |
| :-- | :-- | :-- |
| B | linear | 0.672 |
| B | mlp | 0.839 |
| left_enc | linear | 0.538 |
| left_enc | mlp | 0.655 |
| raw | linear | 0.966 |
| raw | mlp | 0.981 |
