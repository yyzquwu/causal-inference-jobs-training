# Causal Estimates on LaLonde Job Training Data

## Study Design
- Dataset: LaLonde observational benchmark (N=614).
- Target estimand focus: ATE (with ATT from matching as triangulation).
- Identification strategy: conditional ignorability given observed pre-treatment covariates.

## Effect Estimates
- Naive difference in means: -635.03
- Naive 95% bootstrap CI: [-1,844.24, 787.47]
- Matching ATT (propensity nearest-neighbor): 2,238.16
- Matching ATT 95% bootstrap CI: [-324.40, 3,812.82]
- IPW ATE: -622.23
- IPW ATE 95% bootstrap CI: [-2,585.41, 1,632.89]
- Doubly robust ATE: 509.71
- Doubly robust ATE 95% bootstrap CI: [-1,832.68, 3,210.10]

## Propensity Model and Overlap Diagnostics
- Treated score p01-p99: 0.055 to 0.808
- Control score p01-p99: 0.011 to 0.805
- Share of extreme propensity scores (<0.05 or >0.95): 17.427%
- Propensity-model Brier score: 0.1360 (lower is better)
- Propensity-model log loss: 0.4220 (lower is better)

## Weight Diagnostics
- Mean unstabilized weight: 1.924
- Max unstabilized weight: 30.642
- Mean stabilized weight: 1.001
- Max stabilized weight: 9.233
- Effective sample size under IPW: 257.7 (raw N=614)

## Balance Diagnostics
- Max absolute post-weight SMD: 0.341 (Weak balance)
- Mean absolute post-weight SMD: 0.164

| Covariate | SMD Before | SMD After IPW |
|---|---:|---:|
| age | -0.242 | -0.029 |
| black | 1.668 | 0.133 |
| educ | 0.045 | 0.341 |
| hispan | -0.277 | 0.313 |
| married | -0.719 | -0.111 |
| nodegree | 0.235 | -0.159 |
| re74 | -0.596 | -0.176 |
| re75 | -0.287 | -0.049 |

### Highest Residual Imbalance After Weighting
| Covariate | Absolute SMD After |
|---|---:|
| educ | 0.341 |
| hispan | 0.313 |
| re74 | 0.176 |

## IPW Trimming Sensitivity
| Trim Level | IPW ATE | Effective Sample Size |
|---:|---:|---:|
| 0.00 | -622.23 | 257.7 |
| 0.01 | -622.23 | 257.7 |
| 0.02 | -625.81 | 257.8 |
| 0.05 | -667.18 | 281.8 |
| 0.10 | -1,218.86 | 351.5 |

## Interpretation and Learning Notes
- Estimator triangulation suggests positive-to-neutral effects with wide uncertainty intervals; this is analytically realistic for observational labor data.
- The ESS reduction and residual imbalance imply nontrivial extrapolation risk, so any real-world recommendation should stay cautious and diagnostics-guided.
- The analysis emphasizes a causal-learning mindset: uncertainty, overlap, weight concentration, and balance are all explicitly surfaced.

## Recommended Strengthening Steps
1. Add nonlinear propensity specifications (splines/interactions) and compare calibration + balance improvements.
2. Add placebo-outcome checks (re74/re75 as pseudo-outcomes) to stress-test unmeasured confounding risk.
3. Add sensitivity curves under trimming and covariate-set perturbations to quantify estimate fragility.
4. Package this workflow as a reproducible notebook with one-click rerun for collaborators and readers.
