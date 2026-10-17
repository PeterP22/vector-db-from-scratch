"""Plot recall@10 vs QPS from a benchmarks/run.py results JSON.

Usage:
    python benchmarks/plot.py results/siftsmall.json --out docs/recall_qps.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

SERIES_COLORS = {
    "LinearScan": "#2a78d6",
    "KDTree": "#eb6834",
    "LSH": "#1baf7a",
    "HNSW": "#eda100",
    "IVF": "#e87ba4",
}
REFERENCE_COLOR = "#6f6e69"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


def pareto_front(points):
    """Points not dominated in (recall, qps), sorted by recall."""
    pts = sorted(points, key=lambda p: (-p["recall"], -p["qps"]))
    front, best_qps = [], -1.0
    for p in pts:
        if p["qps"] > best_qps:
            front.append(p)
            best_qps = p["qps"]
    return sorted(front, key=lambda p: p["recall"])


def hardware_string(meta):
    cpu = meta.get("cpu_model") or "unknown CPU"
    return f"{cpu}, single-threaded queries"


def plot(results, out_path):
    plt.rcParams.update({
        "font.size": 14,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_SECONDARY,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
    })
    fig, ax = plt.subplots(figsize=(16, 10), dpi=100)

    for ix in results["indexes"]:
        name, pts = ix["name"], ix["points"]
        if not pts:
            continue
        is_ref = ix.get("family") == "faiss"
        color = REFERENCE_COLOR if is_ref else SERIES_COLORS.get(name, TEXT_SECONDARY)
        line, label = pts, name
        # LSH points come from a 2-D grid of rebuilt indexes, so there is no single knob order.
        if name == "LSH" and len(pts) > 1:
            line = pareto_front(pts)
            dominated = [p for p in pts if p not in line]
            if dominated:
                label = "LSH (Pareto front; other combos faint)"
                ax.scatter([p["recall"] for p in dominated], [p["qps"] for p in dominated],
                           s=60, color=color, alpha=0.35, edgecolors="none", zorder=2)
        if len(line) == 1:
            linestyle, marker = "none", ("D" if is_ref else "o")
        else:
            linestyle, marker = ("--" if is_ref else "-"), ("s" if is_ref else "o")
        ax.plot(
            [p["recall"] for p in line], [p["qps"] for p in line],
            linestyle=linestyle, linewidth=2, color=color, marker=marker, markersize=10,
            markeredgecolor="white", markeredgewidth=1.5, label=label, zorder=3,
        )

    ax.set_yscale("log")
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("Queries per second (log scale)")
    ax.grid(True, which="major", color=GRID, linewidth=1)
    ax.grid(True, which="minor", axis="y", color=GRID, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ds = results["dataset"]
    ax.set_title(
        f"Recall vs QPS on {ds['name']} ({ds['n_base']:,} x {ds['dim']}d, {ds['n_queries']} queries)\n"
        f"{hardware_string(results['metadata'])}",
        loc="left", fontsize=16, color=TEXT_PRIMARY, pad=16,
    )
    ax.legend(frameon=False, loc="best", fontsize=12, labelcolor=TEXT_PRIMARY)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", help="results JSON from benchmarks/run.py")
    ap.add_argument("--out", default=str(ROOT / "docs" / "recall_qps.png"))
    args = ap.parse_args(argv)
    results = json.loads(Path(args.results).read_text())
    out = plot(results, args.out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
