"""Stage-1 sanity checks as tests. The full checks run through
`python -m exp36.run sanity --dataset {visdrone,uavdt}` (they load every cache
and take a couple of minutes); this test asserts the saved reports, and the
slow variant recomputes them on a 30-image slice of each scene."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "experiments" / "exp36"


@pytest.mark.parametrize("dataset", ["visdrone", "uavdt"])
def test_saved_sanity_report_passes(dataset):
    p = REPORTS / f"sanity_{dataset}.json"
    if not p.exists():
        pytest.skip(f"run: uv run python -m exp36.run sanity --dataset {dataset}")
    rep = json.load(open(p))
    assert rep["pass"], rep["checks"]
    for name, ok in rep["checks"].items():
        assert ok, name
    # every model in every scene has the eval-threshold cache (not the deploy one)
    for scene in rep["scenes"].values():
        for m, v in scene["models"].items():
            assert v["min_score"] <= 0.01, (m, v["min_score"])


@pytest.mark.slow
@pytest.mark.parametrize("dataset", ["visdrone", "uavdt"])
def test_recompute_on_slice(dataset):
    from exp36.loaders.cache import load_dataset

    ds = load_dataset(dataset, max_images=30)
    for scene, sd in ds.scenes.items():
        for m, preds in sd.preds.items():
            assert len(preds) == sd.n_images == len(sd.whs)
            for p, (w, h) in zip(preds, sd.whs):
                assert p["boxes"].shape[0] == len(p["scores"]) == len(p["labels"])
                assert set(p["labels"].tolist()) <= set(range(ds.num_classes)), (m, scene)
                if len(p["boxes"]):
                    assert (p["boxes"][:, 2] <= w + 1).all() and (p["boxes"][:, 3] <= h + 1).all()
                    assert (p["boxes"][:, 2] >= p["boxes"][:, 0]).all()


@pytest.mark.slow
def test_grid_is_deterministic(tmp_path):
    """Spec §5: two independent runs on a slice must give byte-identical jsonl."""
    import hashlib
    import subprocess
    import sys

    hashes = []
    for run in ("a", "b"):
        out = tmp_path / run
        subprocess.run(
            [
                sys.executable,
                "-m",
                "exp36.run",
                "grid",
                "--dataset",
                "visdrone",
                "--pool",
                "hetero3",
                "--max-images",
                "20",
                "--out",
                str(out),
                "--cache-root",
                str(out / "cache"),
                "--workers",
                "2",
                "--ks",
                "1",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        hashes.append(
            hashlib.sha256((out / "grid_visdrone_hetero3.jsonl").read_bytes()).hexdigest()
        )
    assert hashes[0] == hashes[1]
