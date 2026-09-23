"""Allowlisted static publish: never recursively copy data/model directories."""
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
FILES = ["index.html", "style.css", "app.mjs", "scorer.mjs", "favicon.svg", "sandbox.json"]


def build():
    sandbox = json.loads((ROOT / "site/sandbox.json").read_text())
    assert sandbox["restricted"] is False and "fictional" in sandbox["source"]
    report = json.loads((ROOT / "reports/full/metrics.json").read_text(encoding="utf-8"))
    assert report["protocol"]["train_rows"] > 1000 and report["models"]["bpr"]["users"] > 0
    dest = ROOT / "dist"
    if dest.exists():
        # Fixed known workspace-local build output, never a computed external path.
        assert dest.resolve().parent == ROOT.resolve()
        shutil.rmtree(dest)
    dest.mkdir()
    for name in FILES:
        shutil.copy2(ROOT / "site" / name, dest / name)
    shutil.copy2(ROOT / "reports/full/metrics.json", dest / "metrics.json")
    (dest / ".nojekyll").write_text("")
    assert sorted(p.name for p in dest.iterdir()) == sorted(FILES + ["metrics.json", ".nojekyll"])
    print(f"Built {len(FILES) + 2} allowlisted public files; no private MovieLens assets.")


if __name__ == "__main__":
    build()
