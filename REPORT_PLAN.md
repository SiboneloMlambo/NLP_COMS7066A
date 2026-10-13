# Proposed research focus

**Question:** How does recursive synthetic-label training affect behavioural
in-context learning and the attention patterns associated with induction heads?

**Primary hypothesis:** The extended task will exhibit faster loss of output
diversity and exact accuracy than the base task because errors in both symbol
selection and value retrieval are fed into subsequent generations.

**Null hypothesis:** Once training budget and sampling noise are controlled,
performance and attention-pattern scores do not change systematically across
generations.

## Minimum defensible experiment matrix

| Factor | Levels |
|---|---|
| Task | base, extended |
| Generation | 0 through 8 |
| Synthetic fraction | 0.5, 1.0 |
| Seed | at least 3 seeds |

Report means and standard deviations across seeds. Select one configuration
before examining the final test results.

## Claims that require evidence

- “Collapse occurred”: falling clean-test performance and/or shrinking output
  diversity over generations, consistently across seeds.
- “Rare modes were lost”: class-conditional recall or generation frequency for
  rare symbols falls more quickly than for common symbols.
- “The induction circuit degraded”: a predefined induction-head score weakens,
  and ideally ablating the candidate head changes task performance.
- “No collapse occurred”: confidence intervals and sufficient generations show
  no practically important trend under the tested conditions.

## Important limitations

- Synthetic targets are generated from fixed real contexts; this studies label
  recursion, not fully recursive generation of both inputs and outputs.
- Attention weights alone do not establish causality.
- Small Transformers and artificial tokens may not generalise to production LLMs.
- Sampling temperature and synthetic fraction materially change the process.

