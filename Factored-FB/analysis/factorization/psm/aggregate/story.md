# Factorization probe — cross-seed verdict

**B2** — d is linearly recoverable from B (R^2 0.91); failure is downstream (ρ-sampling / z=B(g) / actor)
(n_seeds 10, R^2 pooled over tasks+seeds)

## Transport-phase R^2(feature -> cube-to-goal d)

| feature | kind | mean R^2 |
| :-- | :-- | :-- |
| B | linear | 0.910 |
| B | mlp | 0.940 |
| left_enc | linear | 0.964 |
| left_enc | mlp | 0.980 |
| raw | linear | 0.964 |
| raw | mlp | 0.981 |
