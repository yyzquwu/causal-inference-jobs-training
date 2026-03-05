# Executive Memo

**Question:** Does job training increase 1978 earnings?

**What we ran:**
- Naive difference (baseline)
- Propensity-score matching
- Inverse probability weighting
- Doubly robust estimator

**Decision framing:**
- Treat the doubly robust estimate as the primary signal because it is consistent if either propensity or outcome model is correct.
- Require acceptable overlap and post-weighting balance before production decisions.
- If overlap is poor, target rollout to regions/populations with support and collect more data.

**Assumptions and limitations:**
- No unmeasured confounding after observed covariates.
- Stable treatment effects and no spillovers.
- Historical labor market context may limit external validity to current programs.
