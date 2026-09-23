# Analysis walkthrough

This is a runnable, prose-first analysis companion. Start from the README setup. Run commands from the repository root. All reported numbers refer to the committed full run, not the smoke run.

## 1. Establish the question and population

The question is whether personalization improves ranking over positive-rating popularity, while reaching more of the catalog with low single-request scoring time. The population is users with at least one eligible positive rating in the test period. It is not all visitors, all users, or users selected by future history depth.

The official MovieLens 100K data has 100,000 unique user/movie pairs. Inspect source provenance:

```bash
python -c "import json; r=json.load(open('reports/full/metrics.json',encoding='utf-8')); print(json.dumps(r['dataset'],indent=2))"
```

## 2. Partition by calendar time

Use fixed global cutoffs, not per-user leave-last-out: before February 1 for training; February 1 through March 14 for validation; March 15 onward for test, all UTC. No future-only user or item gets a trained embedding. Verify the 66,994 / 15,079 / 17,927 counts and train-only indexing with `tests/test_protocol.py`.

The training catalog has 1,566 items. The remaining 116 metadata movies do not have training interactions. The test positive-target filter removes 232 interactions involving train-unseen items. Seven of 257 users with positive test ratings have no eligible target; 250 remain. Recall is conditional on this eligibility, not an all-catalog recall estimate.

## 3. Fit and select without test feedback

Build a binary positive matrix using ratings ≥4; keep a separate all-ratings consumption mask. Fit item CF and PyTorch BPR, evaluate on validation, and write the selection log before opening the test evaluator. Negative sampling excludes all training-rated movies only; excluding future-rated movies would leak future information.

CF selects no co-occurrence shrinkage from 0/20/100. BPR selects 32 dimensions at epoch 10 from 16/32 dimensions and 10/30/60 epochs. Its validation NDCG is 0.26921 versus 0.25595 for selected CF and 0.25147 for popularity. Loss continues falling at later epochs without improved validation ranking: objective fit and ranking generalization are different.

```bash
python -c "import json; s=json.load(open('reports/full/selection.json',encoding='utf-8')); print([(t['model'],t.get('epoch'),t['ndcg']) for t in s['trials']])"
```

## 4. Audit metrics independently

For each user, retrieve top 10 eligible candidates, excluding every pre-cutoff rating. Recall is `hits / eligible target count`. NDCG uses binary relevance, discount `1/log2(rank+1)`, and ideal DCG from `min(10, target count)` positives. Macro aggregation gives each eligible user equal weight. Coverage is the number of distinct recommended movies across users divided by 1,566.

`scripts/audit_metrics.py` reconstructs the saved model's recommendations and independently implements the metric formulas without importing the production ranking or metric helpers. It checks aggregate Recall, NDCG, and coverage against the committed report. This is an audit of the fixed predictions, not a new model-selection round. The hand-calculated toy tests also check metric boundaries.

```bash
python scripts/audit_metrics.py
```

## 5. Read the evidence and its uncertainty

Selected item CF's test NDCG is 0.28031; BPR is 0.27249; popularity is 0.27423. CF versus popularity's interval spans zero, so the result does not establish that personalization improves over popularity. BPR versus CF is negative across its nominal interval, conditional on the single split/seed.

Catalog coverage increases from 4.66% for popularity to 8.30% for CF and 14.50% for BPR. More reach can expose irrelevant items; coverage alone is not catalog utility, discovery satisfaction, fairness, or diversity within a list.

| Training ratings | Users | Popularity NDCG | CF NDCG | BPR NDCG |
|---|---:|---:|---:|---:|
| 0 | 184 | .32177 | .32177 | .32177 |
| 1–19 | 2 | .23898 | .41556 | .33321 |
| 20–99 | 22 | .20083 | .25609 | .19317 |
| 100+ | 42 | .10609 | .10489 | .09523 |

Different cohorts have different target counts and candidate difficulty; the table is descriptive, not evidence that more user history causes worse recommendations. Cold users all share the same fallback. Paired resampling includes those zero-difference pairs rather than silently dropping them.

## 6. Separate demo inference from benchmark inference

The standard BPR benchmark uses a learned user embedding plus item bias. The browser has no user embedding: normalize each item vector, average the selected favorites, then compute dot products with unit candidate vectors. Stable index order breaks ties; selected favorites are excluded. The empty profile uses training positive counts. This heuristic is cheap and reproducible but not trained to reproduce the BPR user space. The separate fold-in test proxy scores 0.26568 NDCG and uses all positive training history, not 2–5 selected favorites.

Run Python/JavaScript parity tests to verify exported values, scores and ranks, including duplicate favorites, no favorites, and a fully consumed catalog:

```bash
python -m pytest tests/test_export.py -q
```

## 7. Make a product decision

Retain popularity as a credible default. Neither model establishes improvement over it here. Do not promote CF solely because it led on test after BPR led validation. Plan a fresh evaluation period, multiple training seeds and a properly instrumented online experiment with suitable data rights. Keep this test split sealed against future tuning. See the decision memo for the rollout proposal.
