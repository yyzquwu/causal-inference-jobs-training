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


def _feature_basis(features: np.ndarray, nonlinear: bool = False) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    if not nonlinear:
        return features

    parts = [features, features**2]
    interactions = []
    for left in range(features.shape[1]):
        for right in range(left + 1, features.shape[1]):
            interactions.append((features[:, left] * features[:, right])[:, None])
    if interactions:
        parts.append(np.hstack(interactions))
    return np.hstack(parts)


def _fit_propensity_coefficients(
    features: np.ndarray,
    treatment: np.ndarray,
    nonlinear: bool = False,
) -> np.ndarray | None:
    treatment = np.asarray(treatment, dtype=float)
    design = _add_intercept(_feature_basis(features, nonlinear=nonlinear))

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

    result = minimize(objective, x0=np.zeros(design.shape[1]), jac=gradient, method="L-BFGS-B")
    if not result.success:
        return None
    return result.x


def predict_propensity_scores(
    train_features: np.ndarray,
    train_treatment: np.ndarray,
    score_features: np.ndarray,
    nonlinear: bool = False,
) -> np.ndarray:
    coefficients = _fit_propensity_coefficients(train_features, train_treatment, nonlinear=nonlinear)
    if coefficients is None:
        mean_prob = float(np.mean(train_treatment))
        return np.full(score_features.shape[0], fill_value=np.clip(mean_prob, 0.01, 0.99), dtype=float)
    score_design = _add_intercept(_feature_basis(score_features, nonlinear=nonlinear))
    scores = _sigmoid(score_design @ coefficients)
    return np.clip(scores, 0.01, 0.99)


def propensity_scores(features: np.ndarray, treatment: np.ndarray, nonlinear: bool = False) -> np.ndarray:
    return predict_propensity_scores(features, treatment, features, nonlinear=nonlinear)


def _ols_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    score_x: np.ndarray,
    nonlinear: bool = False,
) -> np.ndarray:
    train_y = np.asarray(train_y, dtype=float)
    if train_x.shape[0] == 0:
        return np.full(score_x.shape[0], fill_value=float(np.mean(train_y)) if train_y.size else 0.0)
    design_train = _add_intercept(_feature_basis(train_x, nonlinear=nonlinear))
    design_score = _add_intercept(_feature_basis(score_x, nonlinear=nonlinear))
    beta, *_ = np.linalg.lstsq(design_train, train_y, rcond=None)
    return design_score @ beta


def inverse_probability_weights(
    treatment: np.ndarray,
    scores: np.ndarray,
    stabilized: bool = False,
    trim: float | None = None,
) -> np.ndarray:
    treatment = np.asarray(treatment, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if trim is not None:
        scores = np.clip(scores, trim, 1.0 - trim)

    if stabilized:
        treated_share = float(np.mean(treatment))
        return treatment * treated_share / scores + (1.0 - treatment) * (1.0 - treated_share) / (1.0 - scores)
    return treatment / scores + (1.0 - treatment) / (1.0 - scores)


def naive_difference(outcome: np.ndarray, treatment: np.ndarray) -> float:
    return float(np.mean(outcome[treatment == 1]) - np.mean(outcome[treatment == 0]))


def matching_att(outcome: np.ndarray, treatment: np.ndarray, scores: np.ndarray) -> float:
    return caliper_matching_att(outcome, treatment, scores, caliper=None)


def caliper_matching_att(
    outcome: np.ndarray,
    treatment: np.ndarray,
    scores: np.ndarray,
    caliper: float | None = 0.2,
) -> float:
    treated_idx = np.where(treatment == 1)[0]
    control_idx = np.where(treatment == 0)[0]
    if treated_idx.size == 0 or control_idx.size == 0:
        return float("nan")

    logit_scores = np.log(np.clip(scores, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - scores, 1e-6, 1.0))
    treated_logit = logit_scores[treated_idx]
    control_logit = logit_scores[control_idx]
    sorted_order = np.argsort(control_logit)
    sorted_control_logit = control_logit[sorted_order]
    sorted_control_indices = control_idx[sorted_order]

    width = float(caliper * np.std(logit_scores)) if caliper is not None else np.inf
    matched_controls = []
    kept_treated = []
    for local_idx, idx in enumerate(treated_idx):
        value = treated_logit[local_idx]
        pos = np.searchsorted(sorted_control_logit, value)
        candidates = []
        if pos > 0:
            candidates.append(sorted_control_indices[pos - 1])
        if pos < len(sorted_control_logit):
            candidates.append(sorted_control_indices[pos])
        if not candidates:
            continue
        best = min(candidates, key=lambda candidate: abs(logit_scores[candidate] - value))
        if abs(logit_scores[best] - value) <= width:
            matched_controls.append(best)
            kept_treated.append(idx)

    if not matched_controls:
        return float("nan")
    matched_controls = np.asarray(matched_controls, dtype=int)
    kept_treated = np.asarray(kept_treated, dtype=int)
    return float(np.mean(outcome[kept_treated]) - np.mean(outcome[matched_controls]))


def ipw_ate(outcome: np.ndarray, treatment: np.ndarray, scores: np.ndarray) -> float:
    influence = treatment * outcome / scores - (1.0 - treatment) * outcome / (1.0 - scores)
    return float(np.mean(influence))


def stabilized_ipw_ate(
    outcome: np.ndarray,
    treatment: np.ndarray,
    scores: np.ndarray,
    trim: float | None = 0.01,
) -> float:
    weights = inverse_probability_weights(treatment, scores, stabilized=True, trim=trim)
    treated = treatment == 1
    control = treatment == 0
    treated_mean = np.average(outcome[treated], weights=weights[treated])
    control_mean = np.average(outcome[control], weights=weights[control])
    return float(treated_mean - control_mean)


def overlap_weighted_ate(outcome: np.ndarray, treatment: np.ndarray, scores: np.ndarray) -> float:
    treated = treatment == 1
    control = treatment == 0
    weights = np.where(treated, 1.0 - scores, scores)
    treated_mean = np.average(outcome[treated], weights=weights[treated])
    control_mean = np.average(outcome[control], weights=weights[control])
    return float(treated_mean - control_mean)


def doubly_robust_ate(
    outcome: np.ndarray,
    treatment: np.ndarray,
    features: np.ndarray,
    scores: np.ndarray,
    nonlinear_outcome: bool = False,
) -> float:
    mu1 = _ols_predict(features[treatment == 1], outcome[treatment == 1], features, nonlinear=nonlinear_outcome)
    mu0 = _ols_predict(features[treatment == 0], outcome[treatment == 0], features, nonlinear=nonlinear_outcome)
    dr_score = mu1 - mu0 + treatment * (outcome - mu1) / scores - (1.0 - treatment) * (outcome - mu0) / (1.0 - scores)
    return float(np.mean(dr_score))


def cross_fitted_dr_ate(
    outcome: np.ndarray,
    treatment: np.ndarray,
    features: np.ndarray,
    num_folds: int = 2,
    random_seed: int = 42,
    nonlinear_propensity: bool = True,
    nonlinear_outcome: bool = False,
) -> float:
    outcome = np.asarray(outcome, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    features = np.asarray(features, dtype=float)

    indices = np.arange(len(outcome))
    rng = np.random.default_rng(random_seed)
    rng.shuffle(indices)
    folds = np.array_split(indices, num_folds)

    dr_scores = np.zeros(len(outcome), dtype=float)
    for holdout in folds:
        if holdout.size == 0:
            continue
        train_mask = np.ones(len(outcome), dtype=bool)
        train_mask[holdout] = False

        train_x = features[train_mask]
        train_t = treatment[train_mask]
        train_y = outcome[train_mask]
        holdout_x = features[holdout]

        holdout_scores = predict_propensity_scores(
            train_x,
            train_t,
            holdout_x,
            nonlinear=nonlinear_propensity,
        )
        mu1 = _ols_predict(train_x[train_t == 1], train_y[train_t == 1], holdout_x, nonlinear=nonlinear_outcome)
        mu0 = _ols_predict(train_x[train_t == 0], train_y[train_t == 0], holdout_x, nonlinear=nonlinear_outcome)

        holdout_t = treatment[holdout]
        holdout_y = outcome[holdout]
        dr_scores[holdout] = (
            mu1
            - mu0
            + holdout_t * (holdout_y - mu1) / holdout_scores
            - (1.0 - holdout_t) * (holdout_y - mu0) / (1.0 - holdout_scores)
        )
    return float(np.mean(dr_scores))


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
