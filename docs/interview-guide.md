# Interview guide

## A 60-second opening

“I built an independent recommendation benchmark on MovieLens 100K with popularity, conventional item CF, and a PyTorch BPR embedding model. I used global time cutoffs, trained preprocessing only on history, and ranked the full eligible catalog with consumed-item exclusion. The neural model led validation but did not beat popularity on test. It expanded coverage, and the evidence showed that cold users dominated the test population. I also built a static browser demo with Python/JavaScript inference parity and a licensing-aware local bundle import.”

## How do embeddings learn?

Each row in the user or item table is a trainable vector. A positive item and an unknown sampled item form a triplet `(u,i,j)`. The score is `Uu·Vi+bi`. The loss encourages `s(u,i)` to exceed `s(u,j)`. Backpropagation updates the participating user, positive/negative item rows and biases. Vectors encode latent collaborative directions, not labeled genres. Initialization, sampling, optimizer and regularization all affect the learned geometry.

## Why BPR softplus, rather than rating MSE?

The product asks for a ranked shortlist, not a precise 1–5 rating. `softplus(-(s_pos-s_neg))` is numerically stable negative log-sigmoid pairwise loss. It rewards relative ordering, but it is not Recall/NDCG itself and does not create calibrated probabilities. MSE is a reasonable explicit-feedback comparator for a different objective. More sophisticated objectives should earn their complexity through validation.

## What does regularization do?

An L2 penalty on sampled embedding and bias rows restrains score magnitudes and helps generalization. This implementation averages row-norm penalties per batch and multiplies by .0001; it is not AdamW weight decay or a full-table L2 penalty every step. Frequent rows are regularized more often through sampling. The distinction matters for reproduction.

## What is a negative here?

A uniformly drawn train-catalog item that the user has not rated in training. Even low ratings are excluded from sampled unknowns. These are unobserved preferences, not verified dislikes. Excluding future positives from training negatives would leak future information. Uniform negative sampling is simple but can mismatch actual exposures and popularity.

## Why a global temporal split?

Random splits or per-user leave-last-out can train on interactions that occurred after another user's test recommendation time. Fixed global cutoffs emulate a frozen deployment snapshot. The cost is many unseen users and movies; that is disclosed rather than hidden by selecting only warm users. Validation histories mask test consumption but do not refit model taste, matching the declared frozen protocol.

## Why full-catalog evaluation?

There are only 1,566 eligible items, so ranking them all is inexpensive. A sampled set of easy negatives could overstate accuracy and make results depend on a sampling recipe. Eligibility itself still matters: movies absent from training are excluded and counted. Recall is therefore conditional on a restricted candidate catalog.

## Recall, NDCG, coverage: what does each tell us?

Recall measures the fraction of eligible future positives recovered. NDCG rewards putting hits near the top and normalizes by an ideal ranking. Macro means weight users equally. Coverage measures how many different catalog items appear anywhere in top-10 lists; it does not measure relevance, novelty or fairness. Higher coverage can coexist with worse accuracy, as BPR demonstrates here.

## How did you estimate uncertainty?

Resample the same test-user rows with replacement and recompute paired model differences 2,000 times. Pairing retains the correlation from users being easy or hard for every model. Percentile 95% intervals span zero for CF/BPR versus popularity. These intervals omit training randomness, split selection, catalog dependence, and multiple-comparison correction; I would repeat prospectively across seeds and time windows before a strong conclusion.

## What happens to unseen users and items?

Unseen users get training popularity. Train-unseen items have no fitted vectors, so every model excludes them and reports the missing targets. A production system would need freshness-aware content retrieval or new-item exploration. Using all movie metadata to construct train mappings would hide this failure mode.

## Does the demo perform the same inference as the benchmark?

No. Standard BPR uses a fitted user vector plus item bias. A visitor supplies favorites, so the demo averages their unit item vectors and ranks other unit vectors by dot product. It is an explicit fold-in heuristic, not a newly trained user embedding. The separate all-training-favorites proxy underperforms popularity. Tests compare exact JS and Python output on the exported JSON, including exclusion, fallback and numerical tolerance.

## Why can simpler models win?

The data is small and exposure-biased, popularity is a strong prior, and a pairwise loss can overfit unknown negatives. Many users get fallback anyway. Item co-occurrence can capture robust local signals without fitting a latent user space. Lower training loss is not proof of better ranking; later BPR checkpoints lose validation quality.

## What would you change at scale?

Replace the dense item-CF matrix with sparse top-neighbor storage. For embeddings, use approximate nearest-neighbor retrieval then a richer reranker, with separately evaluated retrieval recall. Cache normalized item vectors, batch queries, refresh histories and new-item features, instrument actual impressions and log serving versions. These are proposals; the project does not claim to have implemented large-scale serving.

## What did the project not prove?

No online lift, revenue benefit, calibrated preference scores, demographic fairness, production latency, or generalization to current users. MovieLens data rights also prevent an unrestricted public real-movie model bundle. Showing those boundaries is part of the engineering work.
