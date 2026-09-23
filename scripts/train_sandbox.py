"""Original fictional movies + generated taste profiles, not MovieLens evidence."""
import json
from pathlib import Path
import numpy as np
import torch
from ranking.models import train_bpr

TITLES = [
    ["Orbit of Glass", "The Last Signal", "Moonward", "Station Eleven-Nine", "A Map of Other Suns", "The Quiet Satellite"],
    ["The Apricot Club", "Dinner for Strangers", "A Very Small Disaster", "Tuesday at Noon", "The Borrowed Bicycle", "Letters from the Bakery"],
    ["Fog on Platform Six", "The Missing Hour", "A Key Without a Door", "Night Ledger", "The Fifth Witness", "Under the Observatory"],
    ["Beyond the Cedar Ridge", "The River Cartographer", "North of Tomorrow", "The Blue Compass", "Across the Salt Plain", "Wind in the Atlas"],
    ["Paper Lantern Summer", "The Garden Upstairs", "A Song for Winter", "The Shape of Home", "After the Last Train", "Small Hours in June"]
]


def main():
    cfg = {"seed": 73, "learning_rate": .025, "regularization": .001, "batch_size": 512, "bpr_epochs": [45]}
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(73)
    x = np.zeros((300, 30), dtype=np.float32)
    for u in range(300):
        group = u % 5
        picks = rng.choice(np.arange(group * 6, group * 6 + 6), 4, replace=False)
        x[u, picks] = 1
        x[u, rng.integers(30)] = 1
    model = train_bpr(x, x > 0, cfg, 8, lambda *args: None)
    bundle = {"schema": 1, "source": "Original fictional sandbox — synthetic preferences",
              "restricted": False, "method": "mean-unit-item-cosine-v1",
              "movies": [{"id": j + 1, "title": title} for j, title in enumerate(sum(TITLES, []))],
              "vectors": model.item.weight.detach().numpy().astype(float).tolist(),
              "popularity": x.sum(0).astype(float).tolist(),
              "training": {**cfg, "users": 300, "items": 30, "dimension": 8,
                           "purpose": "Usability only; excluded from real-data benchmark claims"}}
    Path("site/sandbox.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
