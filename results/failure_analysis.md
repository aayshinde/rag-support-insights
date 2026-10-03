# Failure analysis: where triage goes wrong

Based on 4,000 leave-one-out queries from the knowledge base and the 200 held-out tickets, using the deployed two-stage gate.

## 1. Most of the errors are between near-synonym categories

| Most confused category pair (either direction) | wrong predictions |
|---|---|
| PRODUCT SUPPORT <-> TECHNICAL SUPPORT | 104 |
| IT SUPPORT <-> TECHNICAL SUPPORT | 90 |
| CUSTOMER SERVICE <-> TECHNICAL SUPPORT | 85 |
| CUSTOMER SERVICE <-> PRODUCT SUPPORT | 80 |
| IT SUPPORT <-> PRODUCT SUPPORT | 45 |
| CUSTOMER SERVICE <-> RETURNS AND EXCHANGES | 34 |
| CUSTOMER SERVICE <-> IT SUPPORT | 31 |
| PRODUCT SUPPORT <-> RETURNS AND EXCHANGES | 27 |

**57% of all wrong predictions are confusions among just 4 categories** (CUSTOMER SERVICE, IT SUPPORT, PRODUCT SUPPORT, TECHNICAL SUPPORT), out of 21.

Error rate by data source: bitext 0.0%, tickets 39.0%. All of the errors are in the free-text tickets, where one topic is filed under several overlapping categories.

## 2. Taxonomy experiment: merge the most-confused category pairs

If the errors were model failures, merging labels would not help much. If they are label ambiguity, merging should recover accuracy and let the gate answer more.

| Pairs merged | categories left | accuracy (answer all, calibration) | accuracy (held-out) | coverage at 10% error |
|---|---|---|---|---|
| 0 | 21 | 81.0% | 81.5% | 88.2% |
| 1 | 20 | 83.7% | 85.5% | 89.6% |
| 2 | 19 | 87.0% | 90.0% | 94.3% |
| 3 | 18 | 91.9% | 92.5% | 100.0% |
| 4 | 18 | 91.9% | 92.5% | 100.0% |
| 5 | 18 | 91.9% | 92.5% | 100.0% |

Caveat: merging labels makes the task easier by construction, so the accuracy gain alone proves little. The evidence is how concentrated the errors are: a few overlapping categories account for most of them, which points at the taxonomy, not the retriever.

Pairs merged cumulatively, most confused first: PRODUCT SUPPORT + TECHNICAL SUPPORT; IT SUPPORT + TECHNICAL SUPPORT; CUSTOMER SERVICE + TECHNICAL SUPPORT; CUSTOMER SERVICE + PRODUCT SUPPORT; IT SUPPORT + PRODUCT SUPPORT

## 3. Weakest categories

| Category | n | accuracy | escalated | error on answered |
|---|---|---|---|---|
| SALES AND PRE-SALES | 57 | 36.8% | 33.3% | 44.7% |
| GENERAL INQUIRY | 27 | 40.7% | 33.3% | 38.9% |
| RETURNS AND EXCHANGES | 93 | 40.9% | 38.7% | 35.1% |
| HUMAN RESOURCES | 39 | 46.2% | 33.3% | 30.8% |
| IT SUPPORT | 229 | 49.8% | 31.9% | 28.8% |
| CUSTOMER SERVICE | 314 | 53.8% | 27.7% | 27.8% |
| PRODUCT SUPPORT | 332 | 55.7% | 27.4% | 25.3% |
| SERVICE OUTAGES AND MAINTENANCE | 86 | 58.1% | 29.1% | 19.7% |

## 4. Confident misses in the held-out split

These are the cases the gate cannot catch, because the neighbours agree on the wrong label.

- (0.96) gold **IT SUPPORT**, predicted **TECHNICAL SUPPORT**: "Request for System Support
There has been an observed system slowdown impacting user login times. The probable cause is server overload and deployment issues. The team has restarted the servers, clear"
- (0.92) gold **TECHNICAL SUPPORT**, predicted **CUSTOMER SERVICE**: "Support Request for Crash Issue
The platform experienced a crash because of insufficient capacity."
- (0.75) gold **SALES AND PRE-SALES**, predicted **RETURNS AND EXCHANGES**: "Customer Support, I am reaching out to inquire about the digital marketing strategies your company employs to enhance brand growth across various tech products. I am interested in this field and belie"
- (0.69) gold **IT SUPPORT**, predicted **TECHNICAL SUPPORT**: "Reported Problem with Data Analytics System
System is operational"
- (0.66) gold **IT SUPPORT**, predicted **TECHNICAL SUPPORT**: "Secure Patient Information
Could you furnish the necessary documentation that secures patient information?"

## What I would do next

1. Fix the taxonomy before improving the model. Merging the top confused pairs is a cheaper win than any model change.
2. Add a human-review queue for the escalated tickets and feed their resolved labels back into the index, which is the loop the knowledge-gap clusters are meant to drive.
3. Evaluate on real tickets with a clean taxonomy, since half of this benchmark is templated and the other half has overlapping labels.
