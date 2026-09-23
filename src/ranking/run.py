"""Run once per report directory; test remains sealed until selection completes."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
import torch
from .data import acquire, load, split, matrix, histories
from .evaluation import make_cases, evaluate, paired_ci, rank
from .models import item_cf, train_bpr, cold_scores


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def run(config_path, output, data_dir, artifact_dir):
    cfg = json.loads(Path(config_path).read_text())
    out, assets = Path(output), Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.json").exists():
        raise FileExistsError("A completed test report already exists. Use a NEW report directory; do not tune on test.")
    start = time.perf_counter()
    torch.set_num_threads(cfg["threads"])
    torch.use_deterministic_algorithms(True)
    np.random.seed(cfg["seed"])
    root = acquire(data_dir)
    rows, titles = load(root)
    train, val, test = split(rows, cfg["train_end"], cfg["validation_end"])
    users, items, x, consumed = matrix(train, cfg["positive_rating"])
    ui, ii = {u: n for n, u in enumerate(users)}, {i: n for n, i in enumerate(items)}
    depth = {u: int(consumed[n].sum()) for u, n in ui.items()}
    val_cases, val_exclusions = make_cases(val, histories(train, ii), ii, depth, cfg["positive_rating"])
    popularity = x.sum(axis=0)
    validation, selected = [], {}

    def cf_scorer(sim):
        return lambda u: x[ui[u]] @ sim if u in ui and x[ui[u]].any() else popularity

    best_cf, sim_best = -1, None
    for shrinkage in cfg["cf_shrinkages"]:
        sim = item_cf(x, shrinkage)
        agg, _ = evaluate(cf_scorer(sim), val_cases, len(items), cfg["k"])
        validation.append({"model": "item_cf", "shrinkage": shrinkage, **agg})
        if agg["ndcg"] > best_cf:
            best_cf, sim_best = agg["ndcg"], sim.copy()
            selected["item_cf"] = {"shrinkage": shrinkage}

    def bpr_scorer(state):
        U = state["user.weight"].numpy()
        V = state["item.weight"].numpy()
        bias = state["bias.weight"].numpy().ravel()
        return lambda u: U[ui[u]] @ V.T + bias if u in ui and x[ui[u]].any() else popularity

    best_bpr, state_best = -1, None
    for dimension in cfg["bpr_dimensions"]:
        def checkpoint(model, epoch, losses):
            nonlocal best_bpr, state_best
            state = copy.deepcopy(model.state_dict())
            agg, _ = evaluate(bpr_scorer(state), val_cases, len(items), cfg["k"])
            validation.append({"model": "bpr", "dimension": dimension, "epoch": epoch,
                               "loss": losses[-1], **agg})
            print(f"validation BPR d={dimension} epoch={epoch}: NDCG={agg['ndcg']:.4f}", flush=True)
            if agg["ndcg"] > best_bpr:
                best_bpr, state_best = agg["ndcg"], state
                selected["bpr"] = {"dimension": dimension, "epoch": epoch, "losses": losses}
        train_bpr(x, consumed, cfg, dimension, checkpoint)
    pop_val, _ = evaluate(lambda u: popularity, val_cases, len(items), cfg["k"])
    validation.append({"model": "popularity", **pop_val})
    # Persist selection BEFORE test evaluation. No final train+validation refit:
    # this measures a fixed model snapshot with all pre-test consumption masked.
    write_json(out / "config.json", cfg)
    write_json(out / "selection.json", {"criterion": "validation macro NDCG@K; first wins ties",
                                       "selected": selected, "trials": validation})
    test_cases, test_exclusions = make_cases(test, histories(np.vstack([train, val]), ii), ii, depth, cfg["positive_rating"])
    vectors = state_best["item.weight"].numpy()
    scores = {"popularity": lambda u: popularity, "item_cf": cf_scorer(sim_best),
              "bpr": bpr_scorer(state_best),
              "favorite_fold_in": lambda u: cold_scores(vectors, popularity, np.flatnonzero(x[ui[u]])) if u in ui else popularity}
    results, per_user, latencies = {}, {}, {}
    for name, scorer in scores.items():
        results[name], per_user[name] = evaluate(scorer, test_cases, len(items), cfg["k"])
        # Batch size one; score + consumed filtering + stable full sort. Warm
        # users chosen to avoid measuring only the cold-user popularity fallback.
        warm = [c for c in test_cases if c["user"] in ui and x[ui[c["user"]]].any()]
        timing_cases = warm or test_cases
        for n in range(20):
            c = timing_cases[n % len(timing_cases)]
            rank(scorer(c["user"]), c["seen"], cfg["k"])
        times = []
        for n in range(cfg["latency_repeats"]):
            c = timing_cases[n % len(timing_cases)]
            t = time.perf_counter_ns()
            rank(scorer(c["user"]), c["seen"], cfg["k"])
            times.append((time.perf_counter_ns() - t) / 1e6)
        latencies[name] = {"median_ms": float(np.median(times)), "p95_ms": float(np.quantile(times, .95))}
    pairs = {f"{a}_minus_{b}": paired_ci(per_user[a], per_user[b], cfg["bootstrap_samples"], cfg["seed"])
             for a, b in [("item_cf", "popularity"), ("bpr", "popularity"), ("bpr", "item_cf"), ("favorite_fold_in", "popularity")]}
    report = {"dataset": json.loads((Path(data_dir) / "provenance.json").read_text()),
              "protocol": {"k": cfg["k"], "train_end_exclusive": cfg["train_end"],
                           "validation_end_exclusive": cfg["validation_end"], "positive_rating_min": cfg["positive_rating"],
                           "candidate_policy": "all train-observed movies, minus all prior ratings; fixed train-only model",
                           "train_rows": len(train), "validation_rows": len(val), "test_rows": len(test),
                           "train_users": len(users), "eligible_catalog": len(items), "total_catalog": len(titles),
                           "validation_exclusions": val_exclusions, "test_exclusions": test_exclusions},
              "models": results, "paired_bootstrap_95_ci": pairs, "latency": latencies,
              "timing_conditions": {"batch_size": 1, "warmups": 20, "repetitions": cfg["latency_repeats"],
                                    "population": "warm eligible test users cycled", "includes": "scoring + filtering + stable sort",
                                    "excludes": "disk, startup, network, browser rendering"},
              "hardware": {"platform": platform.platform(), "processor": platform.processor(),
                           "logical_cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(),
                           "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
              "elapsed_seconds": time.perf_counter() - start}
    # Private per-user arrays support independent audits; never publish IDs or arrays.
    np.savez(assets / "evaluation.npz", **per_user)
    torch.save(state_best, assets / "bpr.pt")
    np.savez(assets / "baselines.npz", popularity=popularity, similarity=sim_best, users=users, items=items, positives=x)
    bundle = {"schema": 1, "source": "MovieLens 100K — private local model",
              "restricted": True, "method": "mean-unit-item-cosine-v1",
              "movies": [{"id": i, "title": titles[i]} for i in items],
              "vectors": vectors.astype(float).tolist(), "popularity": popularity.astype(float).tolist()}
    write_json(assets / "private-bundle.json", bundle)
    report["private_bundle_sha256"] = hashlib.sha256((assets / "private-bundle.json").read_bytes()).hexdigest()
    write_json(out / "metrics.json", report)
    print(json.dumps({"models": results, "latency": latencies}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/full.json")
    p.add_argument("--output", default="reports/full")
    p.add_argument("--data", default="data")
    p.add_argument("--artifacts", default="artifacts/full")
    p.add_argument("--system-certificates", action="store_true", help="Use pip's bundled truststore for managed Windows certificates")
    args = p.parse_args()
    if args.system_certificates:
        from pip._vendor import truststore
        truststore.inject_into_ssl()
    run(args.config, args.output, args.data, args.artifacts)
