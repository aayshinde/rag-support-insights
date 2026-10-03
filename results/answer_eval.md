# Answer-quality evaluation (objective metrics, gold labels, no judge)

Grounded pipeline on 200 held-out queries.

| Metric | Ungated (answer everything) | Gate: answered | Gate: escalated |
|---|---|---|---|
| n | 200 | 172 | 28 |
| Intent accuracy | 95.5% | 95.9% | 92.9% |
| Category accuracy | 87.5% | 90.7% | 67.9% |
| Similarity to the gold human resolution | 0.805 | 0.804 | 0.809 |
| (baseline) top-1 retrieved resolution | 0.914 | 0.912 | 0.927 |
| Stays inside its sources (support sim) | 0.861 | 0.866 | 0.832 |
| Answers with unsupported numbers/timeframes | 2.0% | 1.7% | 3.6% |
| Cites at least one source | 44.5% | 44.2% | 46.4% |

Intent accuracy 95% CI (ungated): [0.925, 0.98], (gate answered): [0.9244, 0.9884]

