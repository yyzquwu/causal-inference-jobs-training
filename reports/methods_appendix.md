# Methods Appendix

## Why multiple estimators
- Naive difference is a baseline for how misleading raw selection can be.
- Stabilized IPW and overlap weighting reduce sensitivity to extreme weights.
- Cross-fitted DR is the most defensible primary estimator in this repo because it reduces overfitting and remains consistent if either nuisance model is well specified.

## Decision rules I would use before trusting an estimate
- Post-weight max |SMD| should be close to or below 0.10.
- Effective sample size should not collapse relative to the raw sample.
- Placebo outcomes should remain near zero within uncertainty.
- High-support estimates should be directionally consistent with the full-sample estimate.

## Why this still is not a final answer
- The assignment is still observational and could suffer from unmeasured confounding.
- Functional-form choices matter; the nonlinear propensity specification helps, but it does not remove identification risk.
- The point of the workflow is to make fragility visible, not to eliminate it.
