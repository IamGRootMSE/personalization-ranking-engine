# Verified resume bullets

Label the entry **Independent Project — Personalization & Ranking Engine**. Do not place these under an employer or imply production adoption.

- Built a reproducible CPU-first recommendation pipeline on 100,000 MovieLens ratings, comparing popularity, item-based collaborative filtering, and a PyTorch BPR model with global temporal splits and full ranking over 1,566 eligible movies.
- Evaluated 250 held-out users with Recall@10, NDCG@10, catalog coverage, history cohorts, and 2,000 paired bootstrap resamples; reported that BPR reached 14.50% catalog coverage versus 4.66% for popularity without establishing an accuracy improvement.
- Delivered a static browser recommendation demo with local model import, learned-vector cold-start scoring, favorite exclusion, Python/JavaScript parity tests, CI smoke training, and a GitHub Pages deployment workflow; measured BPR CPU scoring/filtering/sorting p95 at 0.0898 ms under documented single-thread conditions.

Source of claims: `reports/full/metrics.json`, `reports/full/selection.json`, tests and the verification record. Deployment workflow existence is not a claim of a live deployment; consult the deployment record. Latency is a local warm microbenchmark, not an online service-level result.
