"""Recall@10 vs single-query QPS benchmark for the indexes in src/indexes.

Usage:
    python benchmarks/run.py --data siftsmall --dir data --out results/siftsmall.json
    python benchmarks/run.py --data synthetic --quick --out results/synthetic_quick.json
"""

import argparse
import datetime as _dt
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.datasets import load_siftsmall, make_synthetic
from src.core.distance import DistanceMetric
from src.indexes import HNSW, IVF, LSH, KDTree, LinearScan

K = 10


def recall_at_k(results, groundtruth, k=K) -> float:
    """Mean fraction of each query's true top-k found in its returned top-k."""
    if len(results) == 0:
        return 0.0
    hits = 0
    for res, gt in zip(results, groundtruth):
        hits += len({int(i) for i in list(res)[:k]} & {int(i) for i in gt[:k]})
    return hits / (k * len(results))


def log(msg: str) -> None:
    print(msg, flush=True)


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def collect_metadata(faiss_mod) -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "date": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "cpu_model": cpu_model(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "faiss": getattr(faiss_mod, "__version__", None) if faiss_mod else None,
        "git_commit": commit,
    }


def sweep_config(data: str, quick: bool) -> dict:
    if quick:
        return {
            "max_queries": 20,
            "lsh": [(4, 8), (16, 12)],
            "hnsw_build": {"M": 16, "ef_construction": 40},
            "hnsw_ef": [10, 40, 160],
            "ivf_build": {"n_clusters": 32 if data == "synthetic" else 100},
            "ivf_n_probe": [1, 4, 16],
        }
    return {
        "max_queries": None,
        "lsh": [(t, b) for t in (2, 8, 32) for b in (8, 12, 16)],
        "hnsw_build": {"M": 16, "ef_construction": 100},
        "hnsw_ef": [10, 20, 40, 80, 160, 320],
        "ivf_build": {"n_clusters": 100},
        "ivf_n_probe": [1, 2, 4, 8, 16, 32],
    }


def time_queries(search_fn, queries, groundtruth, max_seconds):
    """Issue queries one at a time after one warm-up query.

    If the warm-up suggests the full query set would exceed max_seconds, only the
    first queries that fit the budget are timed (recorded as capped).
    """
    t0 = time.perf_counter()
    search_fn(queries[0])
    warmup = time.perf_counter() - t0

    n = len(queries)
    capped_reason = None
    if max_seconds is not None and warmup * n > max_seconds:
        n_fit = max(1, int(max_seconds / max(warmup, 1e-9)))
        if n_fit < n:
            capped_reason = (
                f"warm-up query took {warmup:.2f}s; timed only {n_fit}/{n} queries "
                f"to stay within the {max_seconds:.0f}s per-point budget"
            )
            n = n_fit

    results, latencies = [], []
    for q in queries[:n]:
        t = time.perf_counter()
        ids = search_fn(q)
        latencies.append(time.perf_counter() - t)
        results.append(ids)

    total = sum(latencies)
    return {
        "recall": recall_at_k(results, groundtruth[:n]),
        "qps": n / total if total > 0 else float("inf"),
        "mean_latency_ms": 1000.0 * total / n,
        "p50_latency_ms": 1000.0 * float(np.median(latencies)),
        "n_queries": n,
        "short_results": sum(1 for r in results if len(r) < K),
        "capped": capped_reason,
    }


def timed_build(make):
    t0 = time.perf_counter()
    index = make()
    return index, time.perf_counter() - t0


def build_ours(cls, base_list, dim, **params):
    def make():
        index = cls(dim, **params)
        index.add(base_list)
        index.build()
        return index

    return timed_build(make)


def point_log(name, params, r):
    p = ", ".join(f"{k}={v}" for k, v in params.items()) or "-"
    extra = f"  [{r['capped']}]" if r["capped"] else ""
    log(f"  {name:<14} {p:<28} recall@{K}={r['recall']:.3f}  qps={r['qps']:9.1f}  (n={r['n_queries']}){extra}")


def run_single_build(name, cls, build_params, knob, values, base_list, dim, queries, gt, budget, notes):
    log(f"[{name}] building with {build_params or 'defaults'} ...")
    index, build_s = build_ours(cls, base_list, dim, **build_params)
    log(f"[{name}] built in {build_s:.2f}s")
    points = []
    for v in values:
        params = {knob: v} if knob else {}

        def search(q, params=params):
            return index.search(q, k=K, **params).indices

        r = time_queries(search, queries, gt, budget)
        point_log(name, params, r)
        points.append({"params": params, "build_seconds": build_s, **r})
    return {
        "name": name,
        "family": "ours",
        "build_params": {k: getattr(v, "value", v) for k, v in build_params.items()},
        "build_seconds": build_s,
        "knob": knob,
        "points": points,
        "notes": notes,
    }


def run_lsh(combos, base_list, dim, queries, gt, budget):
    points = []
    for num_tables, num_bits in combos:
        params = {"num_tables": num_tables, "num_bits": num_bits}
        log(f"[LSH] building {params} ...")
        index, build_s = build_ours(
            LSH, base_list, dim, num_tables=num_tables, num_bits=num_bits, metric=DistanceMetric.L2
        )
        log(f"[LSH] built in {build_s:.2f}s")
        cand_before, q_before = index._total_candidates, index._total_queries

        def search(q, index=index):
            return index.search(q, k=K).indices

        r = time_queries(search, queries, gt, budget)
        # Averaged over every search issued, including the warm-up query.
        n_searches = index._total_queries - q_before
        r["avg_candidates"] = (index._total_candidates - cand_before) / n_searches
        point_log("LSH", params, r)
        points.append({"params": params, "build_seconds": build_s, **r})
    return {
        "name": "LSH",
        "family": "ours",
        "build_params": {"metric": "l2"},
        "build_seconds": None,
        "knob": "num_tables x num_bits (rebuilt per combo)",
        "points": points,
        "notes": (
            "Random-hyperplane (SimHash) hashing through the origin, i.e. buckets approximate "
            "cosine similarity; candidates from all tables are re-ranked by exact L2 "
            "(metric=L2). Queries with no candidates return no results."
        ),
    }


def run_faiss(faiss, base, queries, gt, cfg, budget):
    faiss.omp_set_num_threads(1)
    dim = base.shape[1]
    out = []

    log("[faiss Flat] building IndexFlatL2 ...")

    def make_flat():
        index = faiss.IndexFlatL2(dim)
        index.add(base)
        return index

    flat, build_s = timed_build(make_flat)

    def search_flat(q):
        return flat.search(q[None, :], K)[1][0]

    r = time_queries(search_flat, queries, gt, budget)
    point_log("faiss Flat", {}, r)
    out.append({
        "name": "faiss IndexFlatL2", "family": "faiss", "build_params": {}, "build_seconds": build_s,
        "knob": None, "points": [{"params": {}, "build_seconds": build_s, **r}],
        "notes": "Reference: exact brute force, omp_set_num_threads(1).",
    })

    m, efc = cfg["hnsw_build"]["M"], cfg["hnsw_build"]["ef_construction"]
    log(f"[faiss HNSW] building IndexHNSWFlat M={m} efConstruction={efc} ...")

    def make_hnsw():
        index = faiss.IndexHNSWFlat(dim, m)
        index.hnsw.efConstruction = efc
        index.add(base)
        return index

    hnsw, build_s = timed_build(make_hnsw)
    points = []
    for ef in cfg["hnsw_ef"]:
        hnsw.hnsw.efSearch = ef

        def search_hnsw(q):
            return hnsw.search(q[None, :], K)[1][0]

        r = time_queries(search_hnsw, queries, gt, budget)
        point_log("faiss HNSW", {"efSearch": ef}, r)
        points.append({"params": {"efSearch": ef}, "build_seconds": build_s, **r})
    out.append({
        "name": "faiss IndexHNSWFlat", "family": "faiss",
        "build_params": {"M": m, "efConstruction": efc}, "build_seconds": build_s,
        "knob": "efSearch", "points": points,
        "notes": "Reference: same M/efConstruction as ours, omp_set_num_threads(1).",
    })
    return out


def load_data(args):
    if args.data == "siftsmall":
        base, queries, gt = load_siftsmall(args.dir)
        name = "SIFT-small"
    else:
        if args.quick:
            base, queries, gt = make_synthetic(n_base=2000, n_queries=50, dim=32, seed=args.seed)
        else:
            base, queries, gt = make_synthetic(n_base=10000, n_queries=100, dim=128, seed=args.seed)
        name = "synthetic"
    return name, base, queries, gt


def import_faiss(disabled):
    if disabled:
        return None
    try:
        import faiss
    except ImportError:
        return None
    return faiss


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", choices=["siftsmall", "synthetic"], default="siftsmall")
    ap.add_argument("--dir", default=str(ROOT / "data"), help="directory with siftsmall files")
    ap.add_argument("--out", default=None, help="output JSON (default results/<data>[_quick].json)")
    ap.add_argument("--quick", action="store_true", help="fewer queries and sweep points (CI smoke)")
    ap.add_argument("--seed", type=int, default=0, help="synthetic data seed")
    ap.add_argument("--max-seconds-per-point", type=float, default=300.0,
                    help="cap queries timed per sweep point if the warm-up predicts more than this")
    ap.add_argument("--no-faiss", action="store_true", help="skip faiss reference lines")
    args = ap.parse_args(argv)

    cfg = sweep_config(args.data, args.quick)
    name, base, queries, gt = load_data(args)
    if cfg["max_queries"]:
        queries, gt = queries[: cfg["max_queries"]], gt[: cfg["max_queries"]]
    base = np.ascontiguousarray(base, dtype=np.float32)
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    dim = base.shape[1]
    base_list = list(base)
    budget = args.max_seconds_per_point
    faiss = import_faiss(args.no_faiss)

    meta = collect_metadata(faiss)
    log(f"dataset={name} base={base.shape} queries={queries.shape} k={K} quick={args.quick}")
    log(f"cpu={meta['cpu_model']} x{meta['cpu_count']} python={meta['python']} "
        f"numpy={meta['numpy']} faiss={meta['faiss']}")

    indexes = [
        run_single_build("LinearScan", LinearScan, {}, None, [None], base_list, dim, queries, gt, budget,
                         "Exact; Python loop over all vectors."),
        run_single_build("KDTree", KDTree, {}, None, [None], base_list, dim, queries, gt, budget,
                         "Exact; median-split KD-tree with branch-and-bound. In high dimensions "
                         "pruning is weak, so most nodes are visited."),
        run_lsh(cfg["lsh"], base_list, dim, queries, gt, budget),
        run_single_build("HNSW", HNSW, dict(cfg["hnsw_build"]), "ef", cfg["hnsw_ef"], base_list, dim,
                         queries, gt, budget,
                         "Layer-0 degree is 2*M (M_max defaults to 2*M when M_max <= M)."),
        run_single_build("IVF", IVF, dict(cfg["ivf_build"]), "n_probe", cfg["ivf_n_probe"], base_list, dim,
                         queries, gt, budget, "k-means++ coarse quantizer, exact L2 within probed lists."),
    ]
    if faiss is not None:
        indexes.extend(run_faiss(faiss, base, queries, gt, cfg, budget))
    else:
        log("faiss not importable (or --no-faiss); skipping reference lines")

    result = {
        "dataset": {"name": name, "n_base": int(base.shape[0]), "n_queries": int(queries.shape[0]),
                    "dim": int(dim), "k": K, "quick": args.quick},
        "metadata": meta,
        "config": {"sweeps": {k: v for k, v in cfg.items()}, "max_seconds_per_point": budget,
                   "timing": "single-query wall clock (time.perf_counter), 1 warm-up query excluded"},
        "indexes": indexes,
    }
    out = Path(args.out) if args.out else ROOT / "results" / f"{args.data}{'_quick' if args.quick else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    log(f"wrote {out}")
    return result


if __name__ == "__main__":
    main()
