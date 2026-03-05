import numpy as np

from causal_eval.estimators import doubly_robust_ate, ipw_ate, naive_difference, propensity_scores


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

    assert abs(naive - true_effect) < 0.2
    assert abs(ipw - true_effect) < 0.2
    assert abs(dr - true_effect) < 0.2
