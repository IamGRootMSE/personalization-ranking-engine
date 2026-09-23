"""Shared full-catalog protocol, paired bootstrap and auditable aggregates."""
import numpy as np


def rank(scores, consumed=(), k=10):
    scores = np.asarray(scores, dtype=np.float64).copy()
    if scores.ndim != 1 or not np.isfinite(scores).all() or k < 1:
        raise ValueError("Expected finite score vector and positive k")
    allowed = np.ones(len(scores), dtype=bool)
    allowed[list(consumed)] = False
    ids = np.flatnonzero(allowed)
    return ids[np.lexsort((ids, -scores[ids]))[:k]]


def metrics(recommendations, relevant, k):
    relevant = set(relevant)
    if not relevant:
        raise ValueError("No relevant candidates: undefined recall")
    hits = np.array([int(i in relevant) for i in recommendations[:k]])
    discount = 1 / np.log2(np.arange(len(hits)) + 2)
    ideal = (1 / np.log2(np.arange(min(k, len(relevant))) + 2)).sum()
    return float(hits.sum() / len(relevant)), float((hits * discount).sum() / ideal)


def make_cases(target, previous, item_index, train_depth, threshold):
    positives = {}
    for u, i, rating, _ in target:
        if rating >= threshold:
            positives.setdefault(int(u), set()).add(int(i))
    cases, excluded_items, consumed_targets = [], 0, 0
    for u in sorted(positives):
        raw = positives[u]
        known = {item_index[i] for i in raw if i in item_index}
        excluded_items += len(raw) - len(known)
        relevant = known - previous.get(u, set())
        consumed_targets += len(known) - len(relevant)
        if relevant:
            cases.append({"user": u, "relevant": relevant, "seen": previous.get(u, set()),
                          "depth": train_depth.get(u, 0)})
    return cases, {"users_with_positive_ratings": len(positives), "evaluated_users": len(cases),
                   "users_without_eligible_targets": len(positives) - len(cases),
                   "positive_targets_for_unseen_items": excluded_items,
                   "already_consumed_positive_targets": consumed_targets}


def evaluate(score, cases, catalog_size, k):
    rows, recommended = [], set()
    for case in cases:
        rec = rank(score(case["user"]), case["seen"], k)
        recall, ndcg = metrics(rec, case["relevant"], k)
        rows.append([recall, ndcg])
        recommended.update(rec.tolist())
    values = np.asarray(rows)
    if not len(values):
        raise ValueError("No eligible evaluation users")
    result = {"recall": float(values[:, 0].mean()), "ndcg": float(values[:, 1].mean()),
              "coverage": len(recommended) / catalog_size, "users": len(cases), "by_history": {}}
    for name, lower, upper in [("0", 0, 0), ("1–19", 1, 19), ("20–99", 20, 99), ("100+", 100, 1000000)]:
        mask = [lower <= c["depth"] <= upper for c in cases]
        subset = values[mask]
        result["by_history"][name] = {"users": len(subset),
            "recall": float(subset[:, 0].mean()) if len(subset) else None,
            "ndcg": float(subset[:, 1].mean()) if len(subset) else None}
    return result, values


def paired_ci(a, b, samples=2000, seed=42):
    diff = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    means = np.array([diff[rng.integers(len(diff), size=len(diff))].mean(axis=0)
                      for _ in range(samples)])
    return {name: {"delta": float(diff[:, j].mean()),
                   "low": float(np.quantile(means[:, j], .025)),
                   "high": float(np.quantile(means[:, j], .975))}
            for j, name in enumerate(["recall", "ndcg"])}
