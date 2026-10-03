# Stress tests on real, out-of-distribution text

Two-stage gate: escalate if top-1 similarity < 0.703 (relevance guard) or calibrated confidence < 0.271 (correctness gate). In-domain held-out tickets are escalated 14.0% of the time (13.0% by the correctness gate alone).

## Does the gate escalate what the knowledge base should never answer?

| Query set | n | escalated (two-stage) | escalated (correctness gate only) | AUROC: gate | AUROC: raw top-1 | AUROC: vote share |
|---|---|---|---|---|---|---|
| out-of-scope (CLINC oos) | 300 | **98.3%** | 27.7% | 0.713 | 0.999 | 0.647 |
| unrelated domains (CLINC in-scope) | 300 | **98.3%** | 21.3% | 0.675 | 0.998 | 0.584 |
| adjacent domain (Banking77) | 300 | **78.7%** | 14.0% | 0.555 | 0.985 | 0.453 |

AUROC compares each out-of-domain set against the in-domain held-out tickets (1.0 = perfectly separated, 0.5 = chance).
The correctness gate alone barely notices out-of-domain text, while raw similarity separates it almost perfectly. The two signals do different jobs, hence two stages.

## Robustness to typos (~8% of letters dropped, doubled or substituted)

| | accuracy (answer all) | answered | error on answered |
|---|---|---|---|
| clean input | 81.5% | 86.0% | 8.7% |
| noisy, correctness gate only (first version) | 75.5% | 85.0% | 15.3% |
| noisy, two-stage gate (deployed) | 75.5% | 64.5% | 15.5% |
| noisy, two-stage gate trained with typo augmentation (tried, not kept) | 75.5% | 66.5% | 15.8% |

Typo augmentation did not reduce the error on answered queries, so the simpler gate trained on clean data is deployed. Robustness to noisy input remains an open weakness: 8% character noise costs about 6 points of accuracy and the gate does not fully compensate.

Example: `problems with my pass reset` becomes `problems wiith m pass reset`.

