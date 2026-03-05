from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from causal_eval.diagnostics import covariate_balance_table, overlap_summary
from causal_eval.estimators import (
    bootstrap_confidence_interval,
    doubly_robust_ate,
    ipw_ate,
    matching_att,
    naive_difference,
    propensity_scores,
)


def main() -> None:
    data_path = Path("data/raw/lalonde.csv")
    if not data_path.exists():
        raise FileNotFoundError("Run scripts/download_data.py first")

    df = pd.read_csv(data_path)
    df = df.rename(columns={"treat": "treatment", "re78": "outcome"})
    df["black"] = (df["race"] == "black").astype(int)
    df["hispan"] = (df["race"] == "hispan").astype(int)

    feature_cols = ["age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75"]
    treatment = df["treatment"].to_numpy().astype(int)
    outcome = df["outcome"].to_numpy().astype(float)
    features = df[feature_cols].to_numpy().astype(float)

    scores = propensity_scores(features, treatment)

    naive = naive_difference(outcome, treatment)
    match = matching_att(outcome, treatment, scores)
    ipw = ipw_ate(outcome, treatment, scores)
    dr = doubly_robust_ate(outcome, treatment, features, scores)

    naive_ci = bootstrap_confidence_interval(lambda y, t: naive_difference(y, t), outcome, treatment)

    weights = treatment / scores + (1.0 - treatment) / (1.0 - scores)
    balance_before = covariate_balance_table(df, feature_cols, "treatment")
    balance_after = covariate_balance_table(df, feature_cols, "treatment", weights=weights)
    overlap = overlap_summary(scores, treatment)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    result_lines = [
        "# Causal Estimates on LaLonde Job Training Data",
        "",
        "## Effect Estimates",
        f"- Naive difference in means: {naive:,.2f}",
        f"- Naive 95% bootstrap CI: [{naive_ci[0]:,.2f}, {naive_ci[1]:,.2f}]",
        f"- Matching ATT (propensity nearest-neighbor): {match:,.2f}",
        f"- IPW ATE: {ipw:,.2f}",
        f"- Doubly robust ATE: {dr:,.2f}",
        "",
        "## Overlap Diagnostics",
        f"- Treated score p01-p99: {overlap['treated_p01']:.3f} to {overlap['treated_p99']:.3f}",
        f"- Control score p01-p99: {overlap['control_p01']:.3f} to {overlap['control_p99']:.3f}",
        f"- Share of extreme propensity scores (<0.05 or >0.95): {overlap['share_extreme']:.3%}",
        "",
        "## Balance Diagnostics",
        "| Covariate | SMD Before | SMD After IPW |",
        "|---|---:|---:|",
    ]

    merged = balance_before.merge(balance_after, on="covariate", suffixes=("_before", "_after"))
    for row in merged.itertuples(index=False):
        result_lines.append(
            f"| {row.covariate} | {row.standardized_mean_diff_before:.3f} | {row.standardized_mean_diff_after:.3f} |"
        )

    result_lines.extend(
        [
            "",
            "## Interpretation",
            "- Naive estimate is biased upward due to observable selection differences.",
            "- Doubly robust and matching estimates are closer to policy-relevant incremental impact.",
            "- Use overlap and post-weighting balance before trusting effect estimates.",
        ]
    )

    (reports_dir / "causal_results.md").write_text("\n".join(result_lines) + "\n", encoding="utf-8")

    memo_lines = [
        "# Executive Memo",
        "",
        "**Question:** Does job training increase 1978 earnings?",
        "",
        "**What we ran:**",
        "- Naive difference (baseline)",
        "- Propensity-score matching",
        "- Inverse probability weighting",
        "- Doubly robust estimator",
        "",
        "**Decision framing:**",
        "- Treat the doubly robust estimate as the primary signal because it is consistent if either propensity or outcome model is correct.",
        "- Require acceptable overlap and post-weighting balance before production decisions.",
        "- If overlap is poor, target rollout to regions/populations with support and collect more data.",
        "",
        "**Assumptions and limitations:**",
        "- No unmeasured confounding after observed covariates.",
        "- Stable treatment effects and no spillovers.",
        "- Historical labor market context may limit external validity to current programs.",
    ]

    (reports_dir / "executive_memo.md").write_text("\n".join(memo_lines) + "\n", encoding="utf-8")

    print("Wrote reports/causal_results.md and reports/executive_memo.md")


if __name__ == "__main__":
    main()
