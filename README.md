# causal-inference-jobs-training

Policy-evaluation portfolio project using the LaLonde job training dataset.  
This repository reflects my genuine interest in causal inference: moving from raw observational data to transparent evidence, then pressure-testing the answer until the fragile parts are visible.

## Project Motivation

I built this project to practice end-to-end causal inference on observational data and document the full workflow clearly:
- Framing estimands and assumptions clearly.
- Estimating treatment effects with multiple complementary estimators.
- Stress-testing identification with overlap, balance, and weight diagnostics.
- Adding falsification checks and subgroup analysis instead of stopping at one full-sample estimate.
- Communicating where I would trust the result, and where I would not.

## Methods Implemented

- Naive mean-difference baseline.
- Matching with and without a caliper.
- Raw IPW, stabilized IPW, and overlap weighting.
- Doubly robust estimation plus cross-fitted DR as the primary estimator.
- Bootstrap uncertainty intervals for the main estimators.
- Diagnostics for overlap, balance, calibration, effective sample size, and weight concentration.
- Weight-stability checks, placebo-outcome checks, subgroup heterogeneity, and trimming sensitivity analysis.

## Dataset

- LaLonde data from `Rdatasets/MatchIt/lalonde`.
- Downloaded by `scripts/download_data.py` into `data/raw/lalonde.csv`.
- Downloader includes multiple mirrors for more reliable reproducibility across environments.

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

## Repository Structure

- `src/causal_eval/estimators.py`: core causal estimators and bootstrap utilities.
- `src/causal_eval/diagnostics.py`: overlap and covariate-balance diagnostics.
- `scripts/run_demo.py`: end-to-end workflow from data to reports.
- `tests/`: estimator and diagnostics tests.
- `reports/`: generated outputs, appendix, and figures.

## What I Focused On While Building This

Core principles I tried to follow in this repository:
1. You do **not** report one model result blindly; you triangulate and diagnose assumptions.
2. You quantify uncertainty and identify extrapolation risk before recommending action.
3. You add falsification pressure where you can: placebo outcomes, subgroup checks, and weight fragility all matter.
4. You keep the writeup honest. A good causal project should make its weak spots easier to see, not harder.
