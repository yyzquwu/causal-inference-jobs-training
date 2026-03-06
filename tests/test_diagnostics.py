import numpy as np
import pandas as pd

from causal_eval.diagnostics import covariate_balance_table, effective_sample_size, overlap_summary, weight_summary


def test_overlap_summary_fields_exist() -> None:
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    treatment = np.array([0, 0, 1, 1])
    summary = overlap_summary(scores, treatment)
    assert set(summary.keys()) == {
        "treated_p01",
        "treated_p99",
        "control_p01",
        "control_p99",
        "share_extreme",
    }


def test_covariate_balance_table_shape() -> None:
    frame = pd.DataFrame({"x": [1, 2, 3, 4], "z": [4, 3, 2, 1], "t": [0, 0, 1, 1]})
    table = covariate_balance_table(frame, ["x", "z"], "t")
    assert list(table.columns) == ["covariate", "standardized_mean_diff"]
    assert len(table) == 2


def test_weight_summary_and_ess() -> None:
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    summary = weight_summary(weights)
    assert summary["max"] == 4.0
    assert summary["ess"] == effective_sample_size(weights)
    assert summary["ess"] > 0
