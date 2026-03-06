# causal-inference-jobs-training

Policy-evaluation portfolio project using the LaLonde job training dataset.  
This repository reflects what keeps pulling me toward causal inference: starting with messy observational data, getting to an estimate, and then spending just as much time testing how much trust that estimate deserves.

## Project Motivation

I built this project because I wanted to work through an observational causal problem in a way that felt honest from beginning to end:
- starting with the estimand and assumptions instead of jumping straight to a model,
- comparing several effect estimators rather than clinging to one result,
- checking overlap, balance, and weight behavior before getting too comfortable,
- adding placebo checks and subgroup slices to see where the story holds up and where it starts to wobble,
- and writing down where the analysis feels credible, and where it still feels exposed.

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
- `notebooks/causal_walkthrough.ipynb`

## Repository Structure

- `src/causal_eval/estimators.py`: core causal estimators and bootstrap utilities.
- `src/causal_eval/diagnostics.py`: overlap and covariate-balance diagnostics.
- `scripts/run_demo.py`: end-to-end workflow from data to reports.
- `notebooks/causal_walkthrough.ipynb`: a more natural walkthrough of the analysis on top of the package code.
- `tests/`: estimator and diagnostics tests.
- `reports/`: generated outputs, appendix, and figures.

## What I Focused On While Building This

Core principles I tried to follow in this repository:
1. One estimate on its own is never enough; the more important part is how the answer changes across estimators and diagnostics.
2. Uncertainty, overlap, and extrapolation risk matter just as much as the headline effect size.
3. Placebo checks, subgroup variation, and fragile weights are part of the story, not side notes.
4. The writeup should make the weak spots easier to see, not smoother to ignore.
