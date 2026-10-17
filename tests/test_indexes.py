import numpy as np
import pytest

from benchmarks.datasets import exact_knn, make_synthetic
from benchmarks.run import recall_at_k
from src.core.distance import DistanceMetric
from src.indexes import HNSW, IVF, LSH, KDTree, LinearScan

K = 10


def search_all(index, queries, **kwargs):
    return [index.search(q, k=K, **kwargs) for q in queries]


def test_linear_scan_exact(small_data):
    base, queries = small_data
    index = LinearScan(base.shape[1])
    index.add(list(base))
    gt = exact_knn(base, queries, K)
    for res, expected in zip(search_all(index, queries), gt):
        assert res.indices == expected.tolist()
        assert res.distances == sorted(res.distances)


def test_kdtree_exact_vs_brute_force(small_data):
    base, queries = small_data
    index = KDTree(base.shape[1])
    index.add(list(base))
    gt = exact_knn(base, queries, K)
    for q, res, expected in zip(queries, search_all(index, queries), gt):
        assert res.indices == expected.tolist()
        brute = np.sort(np.linalg.norm(base - q, axis=1))[:K]
        np.testing.assert_allclose(res.distances, brute, rtol=1e-5)


def test_ivf_full_probe_is_exact(small_data):
    base, queries = small_data
    index = IVF(base.shape[1], n_clusters=8, kmeans_max_iter=20)
    index.add(list(base))
    gt = exact_knn(base, queries, K)
    results = search_all(index, queries, n_probe=index.n_clusters)
    assert [r.indices for r in results] == gt.tolist()


def test_hnsw_recall():
    base, queries, gt = make_synthetic(n_base=2000, n_queries=50, dim=32, seed=7)
    index = HNSW(32, M=16, ef_construction=100)
    index.add(list(base))
    results = [r.indices for r in search_all(index, queries, ef=64)]
    assert recall_at_k(results, gt, K) >= 0.9


def test_lsh_returns_k_sorted(small_data):
    base, queries = small_data
    index = LSH(base.shape[1], num_tables=8, num_bits=4, metric=DistanceMetric.L2)
    index.add(list(base))
    for q, res in zip(queries, search_all(index, queries)):
        assert len(res.indices) == K
        assert len(set(res.indices)) == K
        assert res.distances == sorted(res.distances)
        np.testing.assert_allclose(res.distances, np.linalg.norm(base[res.indices] - q, axis=1), rtol=1e-5)


@pytest.mark.parametrize(
    "make",
    [
        lambda d: LinearScan(d),
        lambda d: KDTree(d),
        lambda d: LSH(d, num_tables=2, num_bits=4),
        lambda d: HNSW(d, M=4, ef_construction=10),
        lambda d: IVF(d, n_clusters=4, kmeans_max_iter=5),
    ],
    ids=["LinearScan", "KDTree", "LSH", "HNSW", "IVF"],
)
def test_dimension_mismatch_raises(make, small_data):
    base, _ = small_data
    index = make(base.shape[1])
    with pytest.raises(ValueError):
        index.add([np.zeros(base.shape[1] + 1, dtype=np.float32)])
    index.add(list(base[:50]))
    with pytest.raises(ValueError):
        index.search(np.zeros(base.shape[1] - 1, dtype=np.float32), k=K)
