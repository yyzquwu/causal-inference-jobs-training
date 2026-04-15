# causal-inference-jobs-training

Policy evaluation project using the LaLonde job training dataset. It walks through an observational causal analysis from raw data to estimates, checks, and writeup.

## What's in here

- Naive mean-difference baseline.
- Matching with and without a caliper.
- Raw IPW, stabilized IPW, and overlap weighting.
- Doubly robust estimation, including cross-fitted DR as the main estimator.
- Bootstrap intervals for the main estimators.
- Diagnostics for overlap, balance, calibration, effective sample size, and weight concentration.
- Weight-stability checks, placebo outcomes, subgroup heterogeneity, and trimming sensitivity analysis.

## Dataset

- LaLonde data from `Rdatasets/MatchIt/lalonde`.
- Downloaded by `scripts/download_data.py` into `data/raw/lalonde.csv`.
- The downloader uses multiple mirrors to make setup more reliable.

## Quickstart

```bash
make setup
make test
make demo
```

Generated artifacts:
- `reports/causal_results.md`
- `reports/executive_memo.md`
- `reports/methods_appendix.md`
- `reports/figures/`
- `notebooks/causal_walkthrough.ipynb`

## Repo Layout

- `src/causal_eval/estimators.py`: core causal estimators and bootstrap utilities.
- `src/causal_eval/diagnostics.py`: overlap and covariate-balance diagnostics.
- `scripts/run_demo.py`: end-to-end workflow from data to reports.
- `notebooks/causal_walkthrough.ipynb`: walkthrough of the analysis on top of the package code.
- `tests/`: estimator and diagnostics tests.
- `reports/`: generated outputs, appendix, and figures.

