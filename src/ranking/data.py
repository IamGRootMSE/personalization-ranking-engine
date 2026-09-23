"""Official acquisition, immutable temporal partitions and train-only indexing."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile
import numpy as np

URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
OFFICIAL_MD5 = "0e33842e24a9c977be4e0107933c0723"


def acquire(root="data"):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "ml-100k.zip"
    if not archive.exists():
        # Use normal TLS verification. Corporate certificate stores can be enabled
        # by the caller with truststore; never disable certificate verification.
        urllib.request.urlretrieve(URL, archive)
    raw = archive.read_bytes()
    if hashlib.md5(raw).hexdigest() != OFFICIAL_MD5:
        raise ValueError("MovieLens archive checksum mismatch; remove and reacquire")
    with zipfile.ZipFile(archive) as z:
        # Only the necessary files, never demographic data or arbitrary zip paths.
        for name in ["u.data", "u.item", "README"]:
            target = root / "ml-100k" / name
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(z.read("ml-100k/" + name))
    provenance = {
        "dataset": "MovieLens 100K", "release": "1998-04",
        "url": URL, "md5": OFFICIAL_MD5,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "terms": "https://files.grouplens.org/datasets/movielens/ml-100k-README.txt",
        "redistribution": "Separate permission required; raw data and derived model bundles excluded from publication."
    }
    path = root / "provenance.json"
    if not path.exists():
        path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return root / "ml-100k"


def load(root):
    rows = np.loadtxt(Path(root) / "u.data", dtype=np.int64)
    assert rows.shape == (100000, 4)
    assert len(np.unique(rows[:, :2], axis=0)) == len(rows)
    assert np.all((rows[:, 2] >= 1) & (rows[:, 2] <= 5))
    movies = {}
    for line in (Path(root) / "u.item").read_text(encoding="latin-1").splitlines():
        fields = line.split("|")  # actual file is pipe-delimited
        if len(fields) != 24:
            raise ValueError("Unexpected movie schema")
        movies[int(fields[0])] = fields[1]
    assert len(movies) == 1682
    return rows, movies


def split(rows, train_end, validation_end):
    a = int(datetime.fromisoformat(train_end).timestamp())
    b = int(datetime.fromisoformat(validation_end).timestamp())
    if a >= b:
        raise ValueError("Temporal cutoffs must increase")
    t = rows[:, 3]
    return rows[t < a], rows[(t >= a) & (t < b)], rows[t >= b]


def histories(rows, item_index):
    result = {}
    for u, i, _, _ in rows:
        if int(i) in item_index:
            result.setdefault(int(u), set()).add(item_index[int(i)])
    return result


def matrix(train, threshold=4):
    users = sorted(set(train[:, 0].tolist()))
    items = sorted(set(train[:, 1].tolist()))
    ui, ii = {u: n for n, u in enumerate(users)}, {i: n for n, i in enumerate(items)}
    x = np.zeros((len(users), len(items)), dtype=np.float32)
    consumed = np.zeros_like(x, dtype=bool)
    for u, i, r, _ in train:
        consumed[ui[int(u)], ii[int(i)]] = True
        if r >= threshold:
            x[ui[int(u)], ii[int(i)]] = 1
    return users, items, x, consumed
