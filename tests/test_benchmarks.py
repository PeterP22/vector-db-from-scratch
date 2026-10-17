import json

import numpy as np
import pytest

from benchmarks import plot, run
from benchmarks.datasets import exact_knn, read_fvecs, read_ivecs, write_vecs
from benchmarks.run import recall_at_k


def test_recall_at_k_exact_and_disjoint():
    gt = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    assert recall_at_k([[3, 2, 1, 0], [7, 6, 5, 4]], gt, k=4) == 1.0
    assert recall_at_k([[8, 9, 10, 11], [0, 1, 2, 3]], gt, k=4) == 0.0


def test_recall_at_k_near_miss():
    gt = np.array([list(range(10)) + [10, 11]])
    # The 11th true neighbour does not count toward recall@10.
    assert recall_at_k([[0, 1, 2, 3, 4, 5, 6, 7, 8, 10]], gt, k=10) == pytest.approx(0.9)
    # Short result lists count missing slots as misses.
    assert recall_at_k([[0, 1, 2]], gt, k=10) == pytest.approx(0.3)


def test_fvecs_ivecs_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    f = rng.normal(size=(7, 5)).astype(np.float32)
    i = rng.integers(0, 1000, size=(4, 3)).astype(np.int32)
    write_vecs(tmp_path / "a.fvecs", f)
    write_vecs(tmp_path / "b.ivecs", i)
    np.testing.assert_array_equal(read_fvecs(tmp_path / "a.fvecs"), f)
    np.testing.assert_array_equal(read_ivecs(tmp_path / "b.ivecs"), i)
    assert read_fvecs(tmp_path / "a.fvecs").dtype == np.float32
    assert read_ivecs(tmp_path / "b.ivecs").dtype == np.int32


def test_fvecs_header_layout(tmp_path):
    path = tmp_path / "x.fvecs"
    np.array([2, 0, 0], dtype=np.int32).tofile(path)
    with open(path, "r+b") as fh:
        fh.seek(4)
        fh.write(np.array([1.5, -2.0], dtype=np.float32).tobytes())
    np.testing.assert_array_equal(read_fvecs(path), [[1.5, -2.0]])


def test_exact_knn_matches_numpy():
    rng = np.random.default_rng(3)
    base = rng.normal(size=(200, 8)).astype(np.float32)
    q = rng.normal(size=(5, 8)).astype(np.float32)
    expected = np.argsort(np.linalg.norm(base[None] - q[:, None], axis=2), axis=1)[:, :10]
    np.testing.assert_array_equal(exact_knn(base, q, 10), expected)


def test_run_synthetic_quick(tmp_path):
    out = tmp_path / "quick.json"
    run.main(["--data", "synthetic", "--quick", "--out", str(out)])
    data = json.loads(out.read_text())
    names = {ix["name"] for ix in data["indexes"]}
    assert {"LinearScan", "KDTree", "LSH", "HNSW", "IVF"} <= names
    for key in ("platform", "cpu_model", "cpu_count", "python", "numpy", "date"):
        assert key in data["metadata"]
    for ix in data["indexes"]:
        assert ix["points"], ix["name"]
        for p in ix["points"]:
            assert 0.0 <= p["recall"] <= 1.0
            assert p["qps"] > 0
    exact = [ix for ix in data["indexes"] if ix["name"] in ("LinearScan", "KDTree")]
    assert all(ix["points"][0]["recall"] == 1.0 for ix in exact)

    png = tmp_path / "recall_qps.png"
    plot.main([str(out), "--out", str(png)])
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
