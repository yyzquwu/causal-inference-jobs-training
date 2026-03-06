"""Core treatment effect estimators for observational data."""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import minimize


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _add_intercept(features: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(features.shape[0]), features])


def _ols_predict(train_x: np.ndarray, train_y: np.ndarray, score_x: np.ndarray) -> np.ndarray:
    design_train = _add_intercept(train_x)
    design_score = _add_intercept(score_x)
    beta, *_ = np.linalg.lstsq(design_train, train_y, rcond=None)
    return design_score @ beta


def propensity_scores(features: np.ndarray, treatment: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    design = _add_intercept(features)

    def objective(beta: np.ndarray) -> float:
        logits = design @ beta
        probs = np.clip(_sigmoid(logits), 1e-9, 1.0 - 1e-9)
        neg_log_likelihood = -np.sum(treatment * np.log(probs) + (1.0 - treatment) * np.log(1.0 - probs))
        ridge = 1e-4 * np.sum(beta[1:] ** 2)
        return float(neg_log_likelihood + ridge)

    def gradient(beta: np.ndarray) -> np.ndarray:
        logits = design @ beta
        probs = _sigmoid(logits)
        grad = design.T @ (probs - treatment)
        ridge_grad = np.concatenate(([0.0], 2e-4 * beta[1:]))
        return grad + ridge_grad

    result = minimize(
        objective,
        x0=np.zeros(design.shape[1]),
        jac=gradient,
        method="L-BFGS-B",
    )
    if not result.success:
        mean_prob = np.mean(treatment)
        return np.full_like(treatment, fill_value=np.clip(mean_prob, 0.01, 0.99), dtype=float)

    scores = _sigmoid(design @ result.x)
    return np.clip(scores, 0.01, 0.99)


def naive_difference(outcome: np.ndarray, treatment: np.ndarray) -> float:
    return float(outcome[treatment == 1].mean() - outcome[treatment == 0].mean())


def matching_att(outcome: np.ndarray, treatment: np.ndarray, scores: np.ndarray) -> float:
    treated_idx = np.where(treatment == 1)[0]
    control_idx = np.where(treatment == 0)[0]

    sorted_control_order = np.argsort(scores[control_idx])
    sorted_control_scores = scores[control_idx][sorted_control_order]
    sorted_control_indices = control_idx[sorted_control_order]

    matched_controls = []
    for idx in treated_idx:
        score = scores[idx]
        pos = np.searchsorted(sorted_control_scores, score)
        candidates = []
        if pos > 0:
            candidates.append(sorted_control_indices[pos - 1])
        if pos < len(sorted_control_scores):
            candidates.append(sorted_control_indices[pos])
        if not candidates:
            continue
        best = min(candidates, key=lambda candidate: abs(scores[candidate] - score))
        matched_controls.append(best)

    matched_controls = np.asarray(matched_controls, dtype=int)
    if matched_controls.size == 0:
        return float("nan")
    att = outcome[treated_idx].mean() - outcome[matched_controls].mean()
    return float(att)


def ipw_ate(outcome: np.ndarray, treatment: np.ndarray, scores: np.ndarray) -> float:
    influence = treatment * outcome / scores - (1.0 - treatment) * outcome / (1.0 - scores)
    return float(np.mean(influence))


def doubly_robust_ate(outcome: np.ndarray, treatment: np.ndarray, features: np.ndarray, scores: np.ndarray) -> float:
    mu1 = _ols_predict(features[treatment == 1], outcome[treatment == 1], features)
    mu0 = _ols_predict(features[treatment == 0], outcome[treatment == 0], features)

    dr_score = (
        mu1
        - mu0
        + treatment * (outcome - mu1) / scores
        - (1.0 - treatment) * (outcome - mu0) / (1.0 - scores)
    )
    return float(np.mean(dr_score))


def bootstrap_confidence_interval(
    estimate_fn: Callable[[np.ndarray, np.ndarray], float],
    outcome: np.ndarray,
    treatment: np.ndarray,
    num_bootstrap: int = 300,
    alpha: float = 0.05,
    random_seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(random_seed)
    n = len(outcome)
    estimates = []
    for _ in range(num_bootstrap):
        sample_idx = rng.integers(0, n, n)
        y = outcome[sample_idx]
        t = treatment[sample_idx]
        if np.all(t == 0) or np.all(t == 1):
            continue
        estimates.append(estimate_fn(y, t))

    if not estimates:
        return (float("nan"), float("nan"))

    lower = np.quantile(estimates, alpha / 2.0)
    upper = np.quantile(estimates, 1.0 - alpha / 2.0)
    return float(lower), float(upper)
