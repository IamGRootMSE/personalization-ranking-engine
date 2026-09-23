"""Count popularity, cosine item CF, and pairwise PyTorch matrix factorization."""
import numpy as np
import torch
from torch import nn


def item_cf(x, shrinkage):
    co = x.T @ x
    norm = np.sqrt(np.diag(co))
    sim = co / np.maximum(norm[:, None] * norm[None, :], 1e-12)
    sim *= co / np.maximum(co + shrinkage, 1e-12)
    np.fill_diagonal(sim, 0)
    return sim


class BPR(nn.Module):
    def __init__(self, users, items, dimension):
        super().__init__()
        self.user = nn.Embedding(users, dimension)
        self.item = nn.Embedding(items, dimension)
        self.bias = nn.Embedding(items, 1)
        nn.init.normal_(self.user.weight, std=0.05)
        nn.init.normal_(self.item.weight, std=0.05)
        nn.init.zeros_(self.bias.weight)

    def forward(self, u, i):
        return (self.user(u) * self.item(i)).sum(-1) + self.bias(i).squeeze(-1)


def negative_samples(users, consumed, rng):
    """Uniform unknowns within train catalog; reject ALL consumed ratings."""
    if np.any(consumed[users].all(axis=1)):
        raise ValueError("User has no eligible training negatives")
    negatives = rng.integers(consumed.shape[1], size=len(users))
    bad = consumed[users, negatives]
    while bad.any():
        negatives[bad] = rng.integers(consumed.shape[1], size=int(bad.sum()))
        bad = consumed[users, negatives]
    return negatives


def train_bpr(x, consumed, config, dimension, on_checkpoint):
    torch.manual_seed(config["seed"])
    rng = np.random.default_rng(config["seed"])
    model = BPR(*x.shape, dimension)
    opt = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    u, i = np.where(x > 0)
    valid = ~consumed[u].all(axis=1)
    u, i = u[valid], i[valid]
    if not len(u):
        raise ValueError("No positive pairs with eligible negatives")
    losses = []
    for epoch in range(1, max(config["bpr_epochs"]) + 1):
        order = rng.permutation(len(u))
        total = 0
        for start in range(0, len(order), config["batch_size"]):
            idx = order[start:start + config["batch_size"]]
            ub, ib = u[idx], i[idx]
            jb = negative_samples(ub, consumed, rng)
            ut, it, jt = map(torch.from_numpy, (ub, ib, jb))
            diff = model(ut, it) - model(ut, jt)
            reg = (model.user(ut).square().sum(-1) + model.item(it).square().sum(-1)
                   + model.item(jt).square().sum(-1) + model.bias(it).square().sum(-1)
                   + model.bias(jt).square().sum(-1)).mean()
            loss = torch.nn.functional.softplus(-diff).mean() + config["regularization"] * reg
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        losses.append(total / len(u))
        if epoch in config["bpr_epochs"]:
            on_checkpoint(model, epoch, list(losses))
    return model


def normalized(vectors):
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


def cold_scores(vectors, popularity, favorites):
    """Mean unit favorite vector dot unit candidate; no user ID or fitted bias."""
    favorites = sorted(set(favorites))
    if not favorites:
        return np.asarray(popularity, dtype=np.float64).copy()
    if min(favorites) < 0 or max(favorites) >= len(vectors):
        raise ValueError("Favorite index outside catalog")
    v = normalized(np.asarray(vectors, dtype=np.float64))
    return v @ v[favorites].mean(axis=0)
