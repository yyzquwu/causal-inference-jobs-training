# causal-inference-jobs-training

Policy-evaluation portfolio project using the LaLonde job training dataset.
This repo demonstrates quasi-experimental thinking and defensible treatment-effect estimation.

## What This Repo Covers

- Naive mean-difference baseline.
- Propensity score matching (nearest-neighbor ATT).
- Inverse probability weighting (ATE).
- Doubly robust estimation.
- Diagnostics for overlap and covariate balance.
- Executive memo for non-technical stakeholders.

## Dataset

- LaLonde data from `Rdatasets/MatchIt/lalonde`.
- Downloaded by `scripts/download_data.py` into `data/raw/lalonde.csv`.

## Quickstart

```bash
make setup
make test
make demo
```

Generated artifacts:

- `reports/causal_results.md`
- `reports/executive_memo.md`

## Why This Is Portfolio-Ready

- Shows method comparison rather than a single-point estimate.
- Includes diagnostics required for estimator trust.
- Provides business-facing interpretation and limitations.
