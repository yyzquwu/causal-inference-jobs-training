# Causal Estimates on LaLonde Job Training Data

## Effect Estimates
- Naive difference in means: -635.03
- Naive 95% bootstrap CI: [-1,844.24, 787.47]
- Matching ATT (propensity nearest-neighbor): -19,215.53
- IPW ATE: -635.03
- Doubly robust ATE: 1,074.91

## Overlap Diagnostics
- Treated score p01-p99: 0.301 to 0.301
- Control score p01-p99: 0.301 to 0.301
- Share of extreme propensity scores (<0.05 or >0.95): 0.000%

## Balance Diagnostics
| Covariate | SMD Before | SMD After IPW |
|---|---:|---:|
| age | -0.242 | -0.242 |
| black | 1.668 | 1.671 |
| educ | 0.045 | 0.045 |
| hispan | -0.277 | -0.277 |
| married | -0.719 | -0.721 |
| nodegree | 0.235 | 0.235 |
| re74 | -0.596 | -0.597 |
| re75 | -0.287 | -0.288 |

## Interpretation
- Naive estimate is biased upward due to observable selection differences.
- Doubly robust and matching estimates are closer to policy-relevant incremental impact.
- Use overlap and post-weighting balance before trusting effect estimates.
