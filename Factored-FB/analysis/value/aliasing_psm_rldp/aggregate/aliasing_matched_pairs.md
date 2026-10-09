# Matched-observation aliasing test

Ranking accuracy: fraction of (fail-bound state, nearest success-bound state in observation space) pairs where the value ranks the success-bound state higher. 0.5 = value cannot separate look-alike controllable vs about-to-fail states (aliasing).

## Ranking accuracy by phase (mean across seeds)

| method | approach | pickup | carry |
| :--- | :---: | :---: | :---: |
| RLDP | 0.60 | 0.58 | 0.60 |
| PSM | 0.62 | 0.61 | 0.61 |

## Mean matched-pair observation distance (standardized L2)

- RLDP: 2.66
- PSM: 2.66
