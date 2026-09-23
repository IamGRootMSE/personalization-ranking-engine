"""Private JSON bundle inference, identical to the public JavaScript scorer."""
import argparse
import json
from pathlib import Path
from .models import cold_scores
from .evaluation import rank


def recommend(bundle, favorite_ids, k=10):
    index = {m["id"]: n for n, m in enumerate(bundle["movies"])}
    if any(i not in index for i in favorite_ids):
        raise ValueError("Unknown favorite movie ID")
    favorites = [index[i] for i in favorite_ids]
    scores = cold_scores(bundle["vectors"], bundle["popularity"], favorites)
    return [{**bundle["movies"][i], "score": float(scores[i])} for i in rank(scores, favorites, k)]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("bundle")
    p.add_argument("--favorites", nargs="*", type=int, default=[])
    args = p.parse_args()
    print(json.dumps(recommend(json.loads(Path(args.bundle).read_text(encoding="utf-8")), args.favorites), indent=2))
