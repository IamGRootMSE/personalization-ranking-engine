"""Independent audit of frozen predictions. Does not fit or select any model."""
import json
from pathlib import Path
import numpy as np
import torch


def main():
    report = json.loads(Path("reports/full/metrics.json").read_text(encoding="utf-8"))
    rows = np.loadtxt("data/ml-100k/u.data", dtype=np.int64)
    from datetime import datetime
    train_end = int(datetime.fromisoformat(report["protocol"]["train_end_exclusive"]).timestamp())
    test_start = int(datetime.fromisoformat(report["protocol"]["validation_end_exclusive"]).timestamp())
    baselines = np.load("artifacts/full/baselines.npz")
    state = torch.load("artifacts/full/bpr.pt", weights_only=True)
    items, users = baselines["items"].tolist(), baselines["users"].tolist()
    item_idx = {i: j for j, i in enumerate(items)}
    user_idx = {u: j for j, u in enumerate(users)}
    x, pop, sim = baselines["positives"], baselines["popularity"], baselines["similarity"]
    assert set(items) == set(rows[rows[:, 3] < train_end, 1].tolist())
    assert set(users) == set(rows[rows[:, 3] < train_end, 0].tolist())
    assert int(pop.sum()) == int(((rows[:, 3] < train_end) & (rows[:, 2] >= 4)).sum())
    prior, truth = {}, {}
    for u, movie, rating, timestamp in rows.tolist():
        if timestamp < test_start:
            prior.setdefault(u, set()).add(movie)
        elif rating >= 4:
            truth.setdefault(u, set()).add(movie)
    U, V, bias = (state[key].numpy() for key in ["user.weight", "item.weight", "bias.weight"])
    vn = V.astype(np.float64)
    vn /= np.maximum(np.sqrt((vn * vn).sum(1, keepdims=True)), 1e-12)
    values = {name: [] for name in report["models"]}
    coverage = {name: set() for name in values}
    for u in sorted(truth):
        candidates = [i for i in items if i not in prior.get(u, set())]
        positives = truth[u].intersection(candidates)
        if not positives:
            continue
        n = user_idx.get(u)
        scores = {"popularity": pop, "item_cf": pop, "bpr": pop, "favorite_fold_in": pop}
        if n is not None and x[n].any():
            scores["item_cf"] = x[n] @ sim
            scores["bpr"] = U[n] @ V.T + bias.ravel()
            favorites = np.flatnonzero(x[n])
            scores["favorite_fold_in"] = vn @ vn[favorites].mean(0)
        for name, score in scores.items():
            recommended = sorted(candidates, key=lambda i: (-float(score[item_idx[i]]), item_idx[i]))[:10]
            hits = [j + 1 for j, i in enumerate(recommended) if i in positives]
            recall = len(hits) / len(positives)
            dcg = sum(1 / np.log2(r + 1) for r in hits)
            ideal = sum(1 / np.log2(r + 1) for r in range(1, min(10, len(positives)) + 1))
            values[name].append((recall, dcg / ideal))
            coverage[name].update(recommended)
    result = {}
    for name in values:
        means = np.mean(values[name], axis=0)
        actual = [*means, len(coverage[name]) / len(items)]
        expected = [report["models"][name][metric] for metric in ["recall", "ndcg", "coverage"]]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
        assert len(values[name]) == report["models"][name]["users"]
        result[name] = dict(zip(["recall", "ndcg", "coverage"], map(float, actual)))
    output = {"status": "passed", "users": len(values["popularity"]), "models": result,
              "tolerance": 1e-12, "method": "Independent Python sorting and scalar DCG sums; no production evaluation helpers"}
    Path("reports/full/audit.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
