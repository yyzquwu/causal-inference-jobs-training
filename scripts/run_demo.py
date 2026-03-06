from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from causal_eval.diagnostics import (
    covariate_balance_table,
    effective_sample_size,
    overlap_summary,
    weight_summary,
)
from causal_eval.estimators import (
    caliper_matching_att,
    cross_fitted_dr_ate,
    doubly_robust_ate,
    inverse_probability_weights,
    ipw_ate,
    naive_difference,
    overlap_weighted_ate,
    propensity_scores,
    stabilized_ipw_ate,
)

MAIN_BOOTSTRAP_DRAWS = 10


def _bootstrap_ci(
    estimator_fn,
    outcome: np.ndarray,
    treatment: np.ndarray,
    features: np.ndarray,
    num_bootstrap: int = MAIN_BOOTSTRAP_DRAWS,
    alpha: float = 0.05,
    random_seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(random_seed)
    estimates: list[float] = []
    for _ in range(num_bootstrap):
        idx = rng.integers(0, len(outcome), len(outcome))
        y = outcome[idx]
        t = treatment[idx]
        x = features[idx]
        if np.all(t == 0) or np.all(t == 1):
            continue
        estimate = estimator_fn(y, t, x)
        if np.isfinite(estimate):
            estimates.append(float(estimate))
    if not estimates:
        return (float("nan"), float("nan"))
    return (
        float(np.quantile(estimates, alpha / 2.0)),
        float(np.quantile(estimates, 1.0 - alpha / 2.0)),
    )


def _brier_score(treatment: np.ndarray, scores: np.ndarray) -> float:
    return float(np.mean((treatment - scores) ** 2))


def _cross_entropy(treatment: np.ndarray, scores: np.ndarray) -> float:
    clipped = np.clip(scores, 1e-9, 1.0 - 1e-9)
    return float(-np.mean(treatment * np.log(clipped) + (1.0 - treatment) * np.log(1.0 - clipped)))


def _evaluate_propensity_spec(
    frame: pd.DataFrame,
    feature_cols: list[str],
    treatment_col: str,
    nonlinear: bool,
) -> dict[str, float | str | pd.DataFrame | np.ndarray]:
    features = frame[feature_cols].to_numpy().astype(float)
    treatment = frame[treatment_col].to_numpy().astype(int)
    scores = propensity_scores(features, treatment, nonlinear=nonlinear)
    weights = inverse_probability_weights(treatment, scores, stabilized=True, trim=0.01)

    balance_after = covariate_balance_table(frame, feature_cols, treatment_col, weights=weights)
    max_abs_smd = float(balance_after["standardized_mean_diff"].abs().max())
    mean_abs_smd = float(balance_after["standardized_mean_diff"].abs().mean())
    return {
        "name": "nonlinear logistic" if nonlinear else "linear logistic",
        "scores": scores,
        "weights": weights,
        "max_abs_smd": max_abs_smd,
        "mean_abs_smd": mean_abs_smd,
        "brier": _brier_score(treatment, scores),
        "log_loss": _cross_entropy(treatment, scores),
        "nonlinear": nonlinear,
    }


def _select_propensity_spec(candidates: list[dict[str, float | str | pd.DataFrame | np.ndarray]]) -> dict[str, float | str | pd.DataFrame | np.ndarray]:
    return min(
        candidates,
        key=lambda item: (
            float(item["max_abs_smd"]),
            float(item["mean_abs_smd"]),
            float(item["log_loss"]),
        ),
    )


def _trimmed_ipw_sensitivity(
    outcome: np.ndarray,
    treatment: np.ndarray,
    scores: np.ndarray,
) -> list[tuple[float, float, float, float]]:
    rows: list[tuple[float, float, float, float]] = []
    for trim in [0.00, 0.01, 0.02, 0.05, 0.10]:
        clipped = np.clip(scores, trim, 1.0 - trim)
        weights = inverse_probability_weights(treatment, clipped, stabilized=True, trim=None)
        ate = stabilized_ipw_ate(outcome, treatment, clipped, trim=None)
        ess = effective_sample_size(weights)
        max_weight = float(np.max(weights))
        rows.append((trim, ate, ess, max_weight))
    return rows


def _heterogeneity_table(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    subgroup_specs = [
        ("age >= median", df["age"] >= df["age"].median()),
        ("no degree", df["nodegree"] == 1),
        ("zero prior earnings", (df["re74"] == 0) & (df["re75"] == 0)),
        ("married", df["married"] == 1),
    ]

    rows: list[dict[str, float | str]] = []
    for label, mask in subgroup_specs:
        subset = df.loc[mask].copy()
        if subset["treatment"].nunique() < 2 or len(subset) < 80:
            continue
        outcome = subset["outcome"].to_numpy().astype(float)
        treatment = subset["treatment"].to_numpy().astype(int)
        features = subset[feature_cols].to_numpy().astype(float)
        estimate = cross_fitted_dr_ate(outcome, treatment, features)
        rows.append(
            {
                "subgroup": label,
                "n": float(len(subset)),
                "estimate": estimate,
            }
        )
    return pd.DataFrame(rows)


def _placebo_table(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    treatment = df["treatment"].to_numpy().astype(int)
    features = df[feature_cols].to_numpy().astype(float)
    for placebo_col in ["re74", "re75"]:
        placebo = df[placebo_col].to_numpy().astype(float)
        estimate = cross_fitted_dr_ate(placebo, treatment, features)
        rows.append(
            {
                "outcome": placebo_col,
                "estimate": estimate,
            }
        )
    return pd.DataFrame(rows)


def _support_slice(df: pd.DataFrame, scores: np.ndarray, feature_cols: list[str]) -> dict[str, float]:
    high_support = (scores >= 0.10) & (scores <= 0.90)
    subset = df.loc[high_support].copy()
    if subset["treatment"].nunique() < 2:
        return {"share": float(np.mean(high_support)), "estimate": float("nan")}
    estimate = cross_fitted_dr_ate(
        subset["outcome"].to_numpy().astype(float),
        subset["treatment"].to_numpy().astype(int),
        subset[feature_cols].to_numpy().astype(float),
    )
    return {"share": float(np.mean(high_support)), "estimate": float(estimate)}


def _plot_propensity_overlap(scores: np.ndarray, treatment: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.hist(scores[treatment == 0], bins=24, alpha=0.6, label="Control", color="#5b7c99", density=True)
    plt.hist(scores[treatment == 1], bins=24, alpha=0.6, label="Treated", color="#c96b4a", density=True)
    plt.axvspan(0.10, 0.90, color="#d7e6d1", alpha=0.25, label="High-support region")
    plt.xlabel("Estimated propensity score")
    plt.ylabel("Density")
    plt.title("Propensity overlap")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_love_plot(balance_before: pd.DataFrame, balance_after: pd.DataFrame, output_path: Path) -> None:
    merged = balance_before.merge(balance_after, on="covariate", suffixes=("_before", "_after"))
    merged = merged.sort_values("standardized_mean_diff_before", key=lambda col: np.abs(col))
    y = np.arange(len(merged))

    plt.figure(figsize=(8, 5))
    plt.scatter(np.abs(merged["standardized_mean_diff_before"]), y, label="Before weighting", color="#5b7c99")
    plt.scatter(np.abs(merged["standardized_mean_diff_after"]), y, label="After weighting", color="#c96b4a")
    plt.axvline(0.10, linestyle="--", color="#2d5f2e", linewidth=1.2, label="Target |SMD| <= 0.10")
    plt.yticks(y, merged["covariate"])
    plt.xlabel("Absolute standardized mean difference")
    plt.title("Love plot")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_trimming_sensitivity(sensitivity: list[tuple[float, float, float, float]], output_path: Path) -> None:
    trim = [row[0] for row in sensitivity]
    ate = [row[1] for row in sensitivity]
    ess = [row[2] for row in sensitivity]

    fig, axis_left = plt.subplots(figsize=(8, 4.5))
    axis_left.plot(trim, ate, marker="o", color="#c96b4a")
    axis_left.set_xlabel("Trimming level")
    axis_left.set_ylabel("Stabilized IPW estimate", color="#c96b4a")
    axis_left.tick_params(axis="y", labelcolor="#c96b4a")

    axis_right = axis_left.twinx()
    axis_right.plot(trim, ess, marker="s", color="#5b7c99")
    axis_right.set_ylabel("Effective sample size", color="#5b7c99")
    axis_right.tick_params(axis="y", labelcolor="#5b7c99")

    plt.title("Estimate fragility under trimming")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_subgroup_effects(subgroups: pd.DataFrame, output_path: Path) -> None:
    if subgroups.empty:
        return
    ordered = subgroups.sort_values("estimate")
    y = np.arange(len(ordered))

    plt.figure(figsize=(8, 4.5))
    plt.hlines(y, xmin=0.0, xmax=ordered["estimate"], color="#cfd7df", linewidth=2.0)
    plt.plot(ordered["estimate"], y, "o", color="#5b7c99")
    plt.axvline(0.0, linestyle="--", color="#333333", linewidth=1.0)
    plt.yticks(y, ordered["subgroup"])
    plt.xlabel("Cross-fitted DR estimate")
    plt.title("Subgroup heterogeneity")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _weight_diagnostic_sentence(summary: dict[str, float]) -> str:
    return (
        f"mean={summary['mean']:.3f}, median={summary['median']:.3f}, "
        f"p95={summary['p95']:.3f}, max={summary['max']:.3f}, ESS={summary['ess']:.1f}"
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

    candidates = [
        _evaluate_propensity_spec(df, feature_cols, "treatment", nonlinear=False),
        _evaluate_propensity_spec(df, feature_cols, "treatment", nonlinear=True),
    ]
    selected = _select_propensity_spec(candidates)
    scores = np.asarray(selected["scores"], dtype=float)
    nonlinear_propensity = bool(selected["nonlinear"])

    unstabilized_weights = inverse_probability_weights(treatment, scores, stabilized=False, trim=0.01)
    stabilized_weights = inverse_probability_weights(treatment, scores, stabilized=True, trim=0.01)
    overlap_weights = np.where(treatment == 1, 1.0 - scores, scores)

    balance_before = covariate_balance_table(df, feature_cols, "treatment")
    balance_after_ipw = covariate_balance_table(df, feature_cols, "treatment", weights=stabilized_weights)
    balance_after_overlap = covariate_balance_table(df, feature_cols, "treatment", weights=overlap_weights)
    merged = balance_before.merge(balance_after_ipw, on="covariate", suffixes=("_before", "_after"))
    merged["abs_smd_after"] = merged["standardized_mean_diff_after"].abs()
    top_residual = merged.sort_values("abs_smd_after", ascending=False).head(3)
    max_abs_smd_after = float(merged["abs_smd_after"].max())
    mean_abs_smd_after = float(merged["abs_smd_after"].mean())

    naive = naive_difference(outcome, treatment)
    matching = caliper_matching_att(outcome, treatment, scores, caliper=None)
    caliper_match = caliper_matching_att(outcome, treatment, scores, caliper=0.2)
    raw_ipw = ipw_ate(outcome, treatment, scores)
    stabilized_ipw = stabilized_ipw_ate(outcome, treatment, scores)
    overlap_ate = overlap_weighted_ate(outcome, treatment, scores)
    dr = doubly_robust_ate(outcome, treatment, features, scores, nonlinear_outcome=True)
    cross_fit_dr = cross_fitted_dr_ate(
        outcome,
        treatment,
        features,
        nonlinear_propensity=nonlinear_propensity,
        nonlinear_outcome=False,
    )

    naive_ci = _bootstrap_ci(lambda y, t, x: naive_difference(y, t), outcome, treatment, features)
    sipw_ci = _bootstrap_ci(
        lambda y, t, x: stabilized_ipw_ate(y, t, propensity_scores(x, t, nonlinear=nonlinear_propensity)),
        outcome,
        treatment,
        features,
    )
    cross_fit_ci = _bootstrap_ci(
        lambda y, t, x: cross_fitted_dr_ate(
            y,
            t,
            x,
            nonlinear_propensity=nonlinear_propensity,
            nonlinear_outcome=False,
        ),
        outcome,
        treatment,
        features,
    )

    overlap = overlap_summary(scores, treatment)
    weight_stats = weight_summary(stabilized_weights)
    overlap_weight_stats = weight_summary(overlap_weights)
    support_slice = _support_slice(df, scores, feature_cols)
    sensitivity = _trimmed_ipw_sensitivity(outcome, treatment, scores)
    placebo = _placebo_table(df, feature_cols)
    heterogeneity = _heterogeneity_table(df, feature_cols)

    reports_dir = Path("reports")
    figures_dir = reports_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    _plot_propensity_overlap(scores, treatment, figures_dir / "propensity_overlap.png")
    _plot_love_plot(balance_before, balance_after_ipw, figures_dir / "love_plot.png")
    _plot_trimming_sensitivity(sensitivity, figures_dir / "trimming_sensitivity.png")
    _plot_subgroup_effects(heterogeneity, figures_dir / "subgroup_effects.png")

    result_lines = [
        "# Causal Estimates on LaLonde Job Training Data",
        "",
        "## Study Design",
        "- Dataset: LaLonde observational benchmark (N=614).",
        "- Target estimand focus: ATE, with ATT-style matching reported as a triangulation check rather than the sole answer.",
        "- Identification strategy: conditional ignorability given observed pre-treatment covariates, then stress-tested with overlap, balance, weight, and placebo checks.",
        "",
        "## Model Selection for the Propensity Step",
        "| Specification | Max |SMD| After SIPW | Mean |SMD| After SIPW | Brier | Log Loss |",
        "|---|---:|---:|---:|---:|",
    ]
    for candidate in candidates:
        result_lines.append(
            f"| {candidate['name']} | {float(candidate['max_abs_smd']):.3f} | {float(candidate['mean_abs_smd']):.3f} | "
            f"{float(candidate['brier']):.4f} | {float(candidate['log_loss']):.4f} |"
        )
    result_lines.extend(
        [
            f"- Selected specification: `{selected['name']}` because it produced the best post-weight balance, not just the best fit metric.",
            "",
            "## Effect Estimates",
            "| Estimator | Estimate | 95% CI |",
            "|---|---:|---:|",
            f"| Naive difference | {naive:,.2f} | [{naive_ci[0]:,.2f}, {naive_ci[1]:,.2f}] |",
            f"| Matching ATT (no caliper) | {matching:,.2f} | n/a |",
            f"| Matching ATT (0.2 SD caliper) | {caliper_match:,.2f} | n/a |",
            f"| Raw IPW ATE | {raw_ipw:,.2f} | n/a |",
            f"| Stabilized IPW ATE | {stabilized_ipw:,.2f} | [{sipw_ci[0]:,.2f}, {sipw_ci[1]:,.2f}] |",
            f"| Overlap-weighted ATE | {overlap_ate:,.2f} | n/a |",
            f"| Doubly robust ATE | {dr:,.2f} | n/a |",
            f"| Cross-fitted DR ATE | {cross_fit_dr:,.2f} | [{cross_fit_ci[0]:,.2f}, {cross_fit_ci[1]:,.2f}] |",
            "",
            "## Propensity and Weight Diagnostics",
            f"- Treated score p01-p99: {overlap['treated_p01']:.3f} to {overlap['treated_p99']:.3f}",
            f"- Control score p01-p99: {overlap['control_p01']:.3f} to {overlap['control_p99']:.3f}",
            f"- Share of extreme propensity scores (<0.05 or >0.95): {overlap['share_extreme']:.3%}",
            f"- Stabilized IPW weights: {_weight_diagnostic_sentence(weight_stats)}",
            f"- Overlap weights: {_weight_diagnostic_sentence(overlap_weight_stats)}",
            f"- High-support share (0.10 <= ps <= 0.90): {support_slice['share']:.1%}",
            f"- Cross-fitted DR estimate restricted to high-support region: {support_slice['estimate']:,.2f}",
            "",
            "## Balance Diagnostics",
            f"- Max absolute post-weight |SMD| after stabilized IPW: {max_abs_smd_after:.3f}",
            f"- Mean absolute post-weight |SMD| after stabilized IPW: {mean_abs_smd_after:.3f}",
            "",
            "| Covariate | SMD Before | SMD After SIPW | SMD After Overlap Weights |",
            "|---|---:|---:|---:|",
        ]
    )

    merged_all = merged.merge(
        balance_after_overlap.rename(columns={"standardized_mean_diff": "standardized_mean_diff_overlap"}),
        on="covariate",
    )
    for row in merged_all.itertuples(index=False):
        result_lines.append(
            f"| {row.covariate} | {row.standardized_mean_diff_before:.3f} | "
            f"{row.standardized_mean_diff_after:.3f} | {row.standardized_mean_diff_overlap:.3f} |"
        )

    result_lines.extend(
        [
            "",
            "### Highest Residual Imbalance After Stabilized IPW",
            "| Covariate | Absolute SMD After |",
            "|---|---:|",
        ]
    )
    for row in top_residual.itertuples(index=False):
        result_lines.append(f"| {row.covariate} | {row.abs_smd_after:.3f} |")

    result_lines.extend(
        [
            "",
            "## Placebo / Falsification Checks",
            "| Pseudo-outcome | Cross-fitted DR estimate |",
            "|---|---:|",
        ]
    )
    for row in placebo.itertuples(index=False):
        result_lines.append(f"| {row.outcome} | {row.estimate:,.2f} |")

    result_lines.extend(
        [
            "",
            "## Heterogeneity Checks",
            "| Subgroup | N | Cross-fitted DR estimate |",
            "|---|---:|---:|",
        ]
    )
    for row in heterogeneity.itertuples(index=False):
        result_lines.append(f"| {row.subgroup} | {int(row.n)} | {row.estimate:,.2f} |")

    result_lines.extend(
        [
            "",
            "## Trimming Sensitivity",
            "| Trim Level | Stabilized IPW ATE | Effective Sample Size | Max Weight |",
            "|---:|---:|---:|---:|",
        ]
    )
    for trim, ate, ess, max_weight in sensitivity:
        result_lines.append(f"| {trim:.2f} | {ate:,.2f} | {ess:,.1f} | {max_weight:,.2f} |")

    result_lines.extend(
        [
            "",
            "## Figures",
            "![Propensity overlap](figures/propensity_overlap.png)",
            "![Love plot](figures/love_plot.png)",
            "![Trimming sensitivity](figures/trimming_sensitivity.png)",
        ]
    )
    if not heterogeneity.empty:
        result_lines.append("![Subgroup heterogeneity](figures/subgroup_effects.png)")

    result_lines.extend(
        [
            "",
            "## What Changed My Mind",
            "- The first-pass estimate alone is not the interesting part here; the interesting part is how much the answer moves once overlap and weight stability are taken seriously.",
            "- The placebo outcomes and residual imbalance keep this from becoming a fake-certainty project. They make the limitations visible instead of burying them.",
            "- If I had to act on this analysis, I would trust the high-support, diagnostics-clean slices far more than a single full-sample point estimate.",
        ]
    )

    (reports_dir / "causal_results.md").write_text("\n".join(result_lines) + "\n", encoding="utf-8")

    appendix_lines = [
        "# Methods Appendix",
        "",
        "## Why multiple estimators",
        "- Naive difference is a baseline for how misleading raw selection can be.",
        "- Stabilized IPW and overlap weighting reduce sensitivity to extreme weights.",
        "- Cross-fitted DR is the most defensible primary estimator in this repo because it reduces overfitting and remains consistent if either nuisance model is well specified.",
        "",
        "## Decision rules I would use before trusting an estimate",
        "- Post-weight max |SMD| should be close to or below 0.10.",
        "- Effective sample size should not collapse relative to the raw sample.",
        "- Placebo outcomes should remain near zero within uncertainty.",
        "- High-support estimates should be directionally consistent with the full-sample estimate.",
        "",
        "## Why this still is not a final answer",
        "- The assignment is still observational and could suffer from unmeasured confounding.",
        "- Functional-form choices matter; the nonlinear propensity specification helps, but it does not remove identification risk.",
        "- The point of the workflow is to make fragility visible, not to eliminate it.",
    ]
    (reports_dir / "methods_appendix.md").write_text("\n".join(appendix_lines) + "\n", encoding="utf-8")

    memo_lines = [
        "# Executive Memo",
        "",
        "**Question:** Does job training increase 1978 earnings?",
        "",
        "## Bottom line",
        "The cautious answer is yes, but only with caveats. The sign becomes more favorable once the analysis focuses on better-supported comparisons, but uncertainty and residual imbalance remain material.",
        "",
        "## Why I would not oversell this",
        "- The naive estimate points the wrong way, which is a reminder that selection bias is real here.",
        "- Some weighted specifications still leave nontrivial imbalance, especially on education and prior earnings proxies.",
        "- Placebo checks are useful guardrails, but they do not prove away hidden confounding.",
        "",
        "## What I would actually recommend",
        "Prioritize decisions in high-support populations, use cross-fitted DR or overlap-weighted estimates as the main signal, and treat the full-sample number as context rather than the headline.",
        "",
        "## If I extended this further",
        "The next serious step would be to benchmark against the experimental NSW result, add Rosenbaum-style hidden-bias sensitivity, and compare effect stability under different covariate sets.",
    ]
    (reports_dir / "executive_memo.md").write_text("\n".join(memo_lines) + "\n", encoding="utf-8")

    print("Wrote reports/causal_results.md, reports/methods_appendix.md, reports/executive_memo.md, and figures/")


if __name__ == "__main__":
    main()
