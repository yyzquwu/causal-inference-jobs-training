import numpy as np

from causal_eval.estimators import (
    caliper_matching_att,
    cross_fitted_dr_ate,
    doubly_robust_ate,
    ipw_ate,
    naive_difference,
    overlap_weighted_ate,
    propensity_scores,
    stabilized_ipw_ate,
)


def test_estimators_close_to_true_effect_in_randomized_setting() -> None:
    rng = np.random.default_rng(42)
    n = 5000

    x1 = rng.normal(0.0, 1.0, size=n)
    x2 = rng.normal(0.0, 1.0, size=n)
    treatment = rng.binomial(1, 0.5, size=n)

    true_effect = 2.0
    outcome = 5.0 + true_effect * treatment + 0.7 * x1 - 0.3 * x2 + rng.normal(0.0, 1.0, size=n)

    features = np.column_stack([x1, x2])
    ps = propensity_scores(features, treatment)

    naive = naive_difference(outcome, treatment)
    ipw = ipw_ate(outcome, treatment, ps)
    dr = doubly_robust_ate(outcome, treatment, features, ps)
    sipw = stabilized_ipw_ate(outcome, treatment, ps)
    overlap = overlap_weighted_ate(outcome, treatment, ps)
    cross_fit = cross_fitted_dr_ate(outcome, treatment, features, nonlinear_propensity=False, nonlinear_outcome=False)

    assert abs(naive - true_effect) < 0.2
    assert abs(ipw - true_effect) < 0.2
    assert abs(dr - true_effect) < 0.2
    assert abs(sipw - true_effect) < 0.2
    assert abs(overlap - true_effect) < 0.2
    assert abs(cross_fit - true_effect) < 0.2


def test_propensity_scores_are_bounded_and_non_degenerate_with_signal() -> None:
    rng = np.random.default_rng(7)
    n = 4000
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)

    logits = -0.5 + 1.2 * x1 - 0.8 * x2
    probs = 1.0 / (1.0 + np.exp(-logits))
    treatment = rng.binomial(1, probs)

    ps = propensity_scores(np.column_stack([x1, x2]), treatment)

    assert np.all(ps >= 0.01)
    assert np.all(ps <= 0.99)
    assert float(np.std(ps)) > 0.05


def test_caliper_matching_returns_finite_estimate() -> None:
    rng = np.random.default_rng(11)
    n = 3000
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    logits = -0.2 + 0.8 * x1 - 0.6 * x2
    probs = 1.0 / (1.0 + np.exp(-logits))
    treatment = rng.binomial(1, probs)
    outcome = 3.0 + 1.5 * treatment + x1 + rng.normal(scale=0.5, size=n)

    features = np.column_stack([x1, x2])
    ps = propensity_scores(features, treatment)
    estimate = caliper_matching_att(outcome, treatment, ps, caliper=0.2)

    assert np.isfinite(estimate)
