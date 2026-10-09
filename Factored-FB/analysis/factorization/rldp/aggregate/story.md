# Factorization probe — cross-seed verdict

**B2** — d is linearly recoverable from B (R^2 0.69); failure is downstream (ρ-sampling / z=B(g) / actor)
(n_seeds 10, R^2 pooled over tasks+seeds)

## Transport-phase R^2(feature -> cube-to-goal d)

| feature | kind | mean R^2 |
| :-- | :-- | :-- |
| B | linear | 0.695 |
| B | mlp | 0.821 |
| left_enc | linear | 0.515 |
| left_enc | mlp | 0.656 |
| raw | linear | 0.964 |
| raw | mlp | 0.982 |
