# Executive Memo

**Question:** Does job training increase 1978 earnings?

## Recommendation
Proceed with a phased expansion only in segments with good overlap and balance; treat current effect size as promising but uncertain.

## Why this recommendation is credible
- We triangulated across naive, matching, IPW, and doubly robust estimators rather than relying on one model.
- We reported uncertainty intervals and diagnostics (overlap, weight concentration, ESS, and post-weight balance) before interpretation.
- We explicitly identified residual confounding risk and where targeting/model improvements are required.

## Practical next-step framework
1. Define go/no-go thresholds (e.g., max post-weight |SMD| <= 0.10, ESS not severely degraded).
2. Roll out first to high-support populations; monitor outcome and fairness KPIs monthly.
3. Refit and revalidate quarterly as labor-market conditions shift.

## Assumptions and limitations
- No unmeasured confounding after observed covariates.
- Stable treatment effects and no interference between participants.
- Historical context and program implementation details affect external validity.
