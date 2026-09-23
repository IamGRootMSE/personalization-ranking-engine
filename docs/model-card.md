# Model card

**Name/version:** Personalization & Ranking Engine 1.0.0. **Type:** BPR matrix factorization with user/item embeddings and item bias; comparator models are positive-count popularity and item cosine CF. **Owner/use:** independent portfolio research, not an employer production system.

**Data:** official MovieLens 100K, stable 1998-04 release; 100,000 source ratings. Train before 1998-02-01 UTC, validate before 1998-03-15 UTC, test afterward. Training includes 66,994 ratings, 647 users, 1,566 observed items. No demographic attributes or pretrained external models. Ratings ≥4 define positives. Unrated entries are uncertain, not known negative feedback.

**Parameters:** selected 32 dimensions, epoch 10, Adam LR .01, regularization .0001, batch 2048, seed 42. Pairwise softplus objective on uniform unconsumed training-catalog negatives. Train and validation use identical temporal candidate policy; model and mappings remain frozen through test. No final refit on validation.

**Test:** Recall@10 .07781; NDCG@10 .27249; coverage .14496; 250 eligible users. Paired NDCG delta versus popularity −.00174, 95% bootstrap interval [−.00946,+.00605]. 184 users have no training history and receive fallback. Scores and cohorts are fully reported in `reports/full/metrics.json`; results are not general guarantees.

**Efficiency:** BPR p50 .0639 ms / p95 .0898 ms on recorded Windows/AMD hardware, one thread, single query, 20 warmups/300 timings. Training, data acquisition, evaluation and export completed in about 8.5 seconds in this environment, excluding dependency installation. Filesystem writes after timing are not all included in this elapsed field. No claims about production throughput or mobile-browser latency.

**Intended applications:** learning embedding optimization, reviewing temporal evaluation rigor, reproducing a research benchmark, exploring local recommendations. **Out of scope:** commercial or revenue-bearing use without rights, automated high-impact decisions, modeling individual visitors' sensitive traits, or attributing employer/business impact.

**New users/items:** unseen users and users without positive training history use popularity. Train-unseen items are ineligible. Train items without positive ratings have weak or negative-only model evidence; recommendations are still filtered consistently. All consumed items, including low ratings, are excluded. No propensity correction or exposure modeling is performed.

**Demo difference:** local browser inference uses average cosine similarity to favorite item vectors, not the learned BPR user vector and bias. The fold-in proxy has NDCG .26568 and is not evidence for short-profile onboarding. The public fictional sandbox's vectors are independently trained on synthetic taste profiles and do not enter benchmark results.

**Limitations/risks:** historical self-selection, exposure bias, limited cold-item coverage, dominance of cold users, small warm cohorts, one seed/split, stale frozen profiles, nonstationarity, popularity feedback loops, and potential memorization in learned weights. No causal business effect, demographic fairness, calibrated preference probabilities, or harm audit has been measured.

**Privacy and publication:** source ratings, per-user scores, item metadata and MovieLens learned assets are ignored by Git and excluded from the build. Public files contain aggregate evidence and original fictional assets only. Local file input remains in browser memory. There is no telemetry or backend.

**Monitoring if adapted:** schema and freshness, duplicate interactions, vocabulary churn, missing history, fallback rate, coverage, exposure concentration, qualified outcomes, latency, errors, and drift. Refresh and retrain on authorized data with a new untouched evaluation period. Document any material change to candidates, positives, objective or serving profile separately.
