"""Diagnostics for overlap and covariate balance."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _weighted_mean(values: np.ndarray, weights: np.ndarray | None) -> float:
    if weights is None:
        return float(np.mean(values))
    return float(np.sum(values * weights) / np.sum(weights))


def _weighted_var(values: np.ndarray, weights: np.ndarray | None) -> float:
    if weights is None:
        return float(np.var(values, ddof=1))
    mean = _weighted_mean(values, weights)
    return float(np.sum(weights * (values - mean) ** 2) / np.sum(weights))


def standardized_mean_diff(values: np.ndarray, treatment: np.ndarray, weights: np.ndarray | None = None) -> float:
    treated = treatment == 1
    control = treatment == 0

    wt_treated = weights[treated] if weights is not None else None
    wt_control = weights[control] if weights is not None else None

    mean_treated = _weighted_mean(values[treated], wt_treated)
    mean_control = _weighted_mean(values[control], wt_control)

    var_treated = _weighted_var(values[treated], wt_treated)
    var_control = _weighted_var(values[control], wt_control)
    pooled_sd = np.sqrt(max((var_treated + var_control) / 2.0, 1e-12))

    return float((mean_treated - mean_control) / pooled_sd)


def covariate_balance_table(
    frame: pd.DataFrame,
    covariates: list[str],
    treatment_col: str,
    weights: np.ndarray | None = None,
) -> pd.DataFrame:
    treatment = frame[treatment_col].to_numpy()
    rows: list[dict[str, float | str]] = []
    for cov in covariates:
        smd = standardized_mean_diff(frame[cov].to_numpy(), treatment, weights=weights)
        rows.append({"covariate": cov, "standardized_mean_diff": smd})
    return pd.DataFrame(rows).sort_values("covariate").reset_index(drop=True)


def overlap_summary(scores: np.ndarray, treatment: np.ndarray) -> dict[str, float]:
    treated_scores = scores[treatment == 1]
    control_scores = scores[treatment == 0]

    return {
        "treated_p01": float(np.quantile(treated_scores, 0.01)),
        "treated_p99": float(np.quantile(treated_scores, 0.99)),
        "control_p01": float(np.quantile(control_scores, 0.01)),
        "control_p99": float(np.quantile(control_scores, 0.99)),
        "share_extreme": float(np.mean((scores < 0.05) | (scores > 0.95))),
    }
