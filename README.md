# causal-inference-jobs-training

Policy-evaluation portfolio project using the LaLonde job training dataset.  
This repository reflects my genuine interest in causal inference: moving from raw observational data to transparent evidence with explicit diagnostics and uncertainty.

## Project Motivation

I built this project to practice end-to-end causal inference on observational data and document the full workflow clearly:
- Framing estimands and assumptions clearly.
- Estimating treatment effects with multiple complementary estimators.
- Stress-testing identification with overlap, balance, and weight diagnostics.
- Communicating risk-aware recommendations to non-technical stakeholders.

## Methods Implemented

- Naive mean-difference baseline.
- Propensity score matching (nearest-neighbor ATT).
- Inverse probability weighting (ATE).
- Doubly robust estimation (AIPW-style form).
- Bootstrap uncertainty intervals for all estimators.
- Diagnostics for overlap, balance, calibration, and effective sample size.
- Weight-stability checks and trimming sensitivity analysis.

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

## Repository Structure

- `src/causal_eval/estimators.py`: core causal estimators and bootstrap utilities.
- `src/causal_eval/diagnostics.py`: overlap and covariate-balance diagnostics.
- `scripts/run_demo.py`: end-to-end workflow from data to reports.
- `tests/`: estimator and diagnostics tests.
- `reports/`: generated outputs suitable for portfolio presentation.

## What I Focused On While Building This

Core principles I tried to follow in this repository:
1. You do **not** report one model result blindly; you triangulate and diagnose assumptions.
2. You quantify uncertainty and identify extrapolation risk before recommending action.
3. You approach causal work with curiosity and rigor: you test assumptions, look for failure modes, and iterate.
