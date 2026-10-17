"""Benchmark datasets: TEXMEX .fvecs/.ivecs readers, SIFT-small loader, synthetic clusters."""

from pathlib import Path

import numpy as np


def _read_vecs(path, dtype) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        return np.zeros((0, 0), dtype=dtype)
    dim = int(raw[0])
    if dim <= 0 or raw.size % (dim + 1) != 0:
        raise ValueError(f"{path}: not a valid TEXMEX vecs file (dim header {dim})")
    rows = raw.reshape(-1, dim + 1)
    if not np.all(rows[:, 0] == dim):
        raise ValueError(f"{path}: inconsistent per-row dimension headers")
    return rows[:, 1:].copy().view(dtype)


def read_fvecs(path) -> np.ndarray:
    """Read a TEXMEX .fvecs file (each row: int32 dim, then dim float32)."""
    return _read_vecs(path, np.float32)


def read_ivecs(path) -> np.ndarray:
    """Read a TEXMEX .ivecs file (each row: int32 dim, then dim int32)."""
    return _read_vecs(path, np.int32)


def write_vecs(path, data: np.ndarray) -> None:
    """Write a 2-D float32 or int32 array in TEXMEX format."""
    data = np.ascontiguousarray(data)
    if data.dtype not in (np.float32, np.int32):
        raise ValueError(f"unsupported dtype {data.dtype}")
    n, dim = data.shape
    out = np.empty((n, dim + 1), dtype=np.int32)
    out[:, 0] = dim
    out[:, 1:] = data.view(np.int32)
    out.tofile(path)


def load_siftsmall(directory) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load SIFT-small (base 10000x128, queries 100x128, groundtruth 100x100).

    Accepts either the directory containing the siftsmall_*.?vecs files or its parent.
    """
    d = Path(directory)
    if not (d / "siftsmall_base.fvecs").exists() and (d / "siftsmall").is_dir():
        d = d / "siftsmall"
    base = read_fvecs(d / "siftsmall_base.fvecs")
    queries = read_fvecs(d / "siftsmall_query.fvecs")
    groundtruth = read_ivecs(d / "siftsmall_groundtruth.ivecs").astype(np.int64)
    return base, queries, groundtruth


def exact_knn(base: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Exact k nearest neighbours by L2 distance (brute force, float64)."""
    b = base.astype(np.float64)
    q = queries.astype(np.float64)
    d2 = (q * q).sum(1)[:, None] - 2.0 * q @ b.T + (b * b).sum(1)[None, :]
    k = min(k, base.shape[0])
    part = np.argpartition(d2, k - 1, axis=1)[:, :k]
    order = np.take_along_axis(d2, part, axis=1).argsort(axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)


def make_synthetic(
    n_base: int = 2000,
    n_queries: int = 50,
    dim: int = 32,
    n_clusters: int = 20,
    k_gt: int = 100,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seeded Gaussian clusters; queries drawn from the same mixture.

    Returns (base, queries, groundtruth) with exact L2 groundtruth.
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 10.0, size=(n_clusters, dim))

    def sample(n):
        labels = rng.integers(0, n_clusters, size=n)
        return (centers[labels] + rng.normal(0.0, 1.0, size=(n, dim))).astype(np.float32)

    base = sample(n_base)
    queries = sample(n_queries)
    return base, queries, exact_knn(base, queries, k_gt)
