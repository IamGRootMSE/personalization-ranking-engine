import json
import os
from pathlib import Path
import shutil
import subprocess
import pytest
from ranking.infer import recommend

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("filename", ["site/sandbox.json", "artifacts/full/private-bundle.json"])
def test_python_matches_actual_js_export(filename):
    path = ROOT / filename
    if not path.exists():
        pytest.skip("Private bundle only available after local full training")
    node = os.environ.get("NODE_BINARY") or shutil.which("node")
    if not node:
        pytest.fail("Node is required to test exported inference")
    bundle = json.loads(path.read_text(encoding="utf-8"))
    fixtures = [[], [bundle["movies"][0]["id"]], [bundle["movies"][i]["id"] for i in [0, 3, 9]],
                [bundle["movies"][2]["id"]] * 2, [m["id"] for m in bundle["movies"]]]
    js = """import fs from 'node:fs';import {recommend} from './site/scorer.mjs';
const data=JSON.parse(fs.readFileSync(0,'utf8'));
console.log(JSON.stringify(data.fixtures.map(f=>recommend(data.bundle,f,10))));"""
    result = subprocess.run([node, "--input-type=module", "-e", js], cwd=ROOT,
                            input=json.dumps({"bundle": bundle, "fixtures": fixtures}),
                            text=True, capture_output=True, check=True, encoding="utf-8")
    for favorites, actual in zip(fixtures, json.loads(result.stdout)):
        expected = recommend(bundle, favorites)
        assert [m["id"] for m in actual] == [m["id"] for m in expected]
        assert [m["score"] for m in actual] == pytest.approx([m["score"] for m in expected], abs=1e-10)
        assert not set(favorites) & {m["id"] for m in actual}
