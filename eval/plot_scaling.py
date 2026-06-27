#!/usr/bin/env python3
"""Plot alltoallv benchmark scaling.

Parses the per-run report JSON files written by the benchmarks (one file per
configuration point, under ``<data-dir>/<suite>_<date>/output/*.json``) and draws a
line plot of *running time* (averaged over iterations) against the *number of
workers* (= MPI ranks x threads-per-rank). One line per variant; two panels, one
per input distribution (uniform / nonuniform).

A "variant" is the experiment suite (singlethreaded / funneled / multithreaded),
taken from the suite directory name -- note that the *singlethreaded* and *funneled*
suites both use the ``single-threaded-alltoallv`` algorithm and are only
distinguishable by the suite. When several per-worker element counts are present
they are drawn as separate lines (distinguished by line style/marker), since the
message size strongly affects the running time.

Usage:
    python3 eval/plot_scaling.py
    python3 eval/plot_scaling.py --data-dir eval/data/supermuc --out eval/scaling.png
"""

import argparse
import glob
import json
import os
import re

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Filename suffix written by kaval/the runner: ...-r<ranks>-t<threads>-c<cfg>-s<seed>.json
_RT_RE = re.compile(r"-r(\d+)-t(\d+)-c\d+-s\d+\.json$")
# Suite directory: <data-dir>/alltoallv-<variant>_<yy>_<mm>_<dd>/output/...
_VARIANT_RE = re.compile(r"/alltoallv-([a-zA-Z0-9]+)_\d\d_\d\d_\d\d/")

# Stable colors per variant and styles per element count, so the two panels are
# directly comparable.
_VARIANT_COLORS = {
    "singlethreaded": "tab:blue",
    "funneled": "tab:orange",
    "multithreaded": "tab:green",
}
_ELEM_STYLES = ["-", "--", ":", "-."]
_ELEM_MARKERS = ["o", "s", "^", "D"]

# Columns that identify one benchmark configuration (a row in the aggregated table).
_GROUP = ["variant", "distribution", "elements", "processors", "threads", "workers"]


def variant_of(path):
    m = _VARIANT_RE.search(path)
    return m.group(1) if m else "unknown"


def workers_of(filename):
    """Return (workers, ranks, threads) parsed from the file name, or (None, ...)."""
    m = _RT_RE.search(filename)
    if not m:
        return None, None, None
    ranks, threads = int(m.group(1)), int(m.group(2))
    return ranks * threads, ranks, threads


def exchange_times(timer):
    """Per-iteration exchange times (one value per recorded iteration).

    ``timer`` is the list of per-iteration aggregated trees; each entry is
    ``{"root": {"<exchange>": {"max": [t], ...}}}``. We take the top-level timed
    region under ``root`` (the whole alltoallv; for the single-threaded variants it
    has a nested ``MPI_Alltoallv`` child, which we ignore) and use its per-rank
    ``max`` (averaged across ranks) as that iteration's time.
    """
    times = []
    for entry in timer:
        root = entry.get("root", {})
        if not root:
            continue
        # The single top-level region recorded for this iteration.
        node = next(iter(root.values()))
        maxes = node.get("max", [])
        if maxes:
            times.append(sum(maxes) / len(maxes))
    return times


def collect(data_dir, suites=None):
    """Read the report files into a long DataFrame with one row per measurement.

    Columns: ``variant, distribution, elements, processors, threads, workers,
    iteration, time`` -- one row per timed iteration of every configuration.

    When ``suites`` is given, only those ``<suite>_<date>`` directories (resolved
    relative to ``data_dir`` if not absolute) are processed; otherwise every
    suite directory under ``data_dir`` is.
    """
    if suites:
        paths = []
        for suite in suites:
            suite_dir = suite if os.path.isabs(suite) else os.path.join(data_dir, suite)
            paths += glob.glob(os.path.join(suite_dir, "output", "*.json"))
    else:
        paths = glob.glob(os.path.join(data_dir, "*", "output", "*.json"))
    rows = []
    for path in sorted(paths):
        if os.path.basename(path) == "config.json":
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        config = data.get("config", {})
        workers, ranks, threads = workers_of(os.path.basename(path))
        if workers is None:
            continue
        for i, t in enumerate(exchange_times(data.get("timer", []))):
            rows.append(
                {
                    "variant": variant_of(path),
                    "distribution": config.get("distribution", "unknown"),
                    "elements": config.get("elements_per_worker"),
                    "processors": ranks,
                    "threads": threads,
                    "workers": workers,
                    "iteration": i,
                    "time": t,
                }
            )
    return pd.DataFrame(rows, columns=_GROUP + ["iteration", "time"])


def aggregate(df):
    """Aggregate the per-iteration measurements to min/max/avg time per config."""
    return (
        df.groupby(_GROUP, as_index=False)["time"]
        .agg(time_min="min", time_max="max", time_avg="mean")
    )


def _distribution_order(values):
    order = [d for d in ("uniform", "nonuniform") if d in values]
    return order + [d for d in sorted(values) if d not in order]


def plot(agg, out_path):
    """Draw running time vs. number of workers, one panel per distribution.

    Each (variant, elements) series is the average over iterations with a shaded
    band spanning the per-iteration min/max at each worker count.
    """
    distributions = _distribution_order(set(agg["distribution"]))
    elements = sorted(agg["elements"].unique())
    if not distributions or not elements:
        raise SystemExit("No data found to plot.")

    # Within a cell the input size is fixed, so threads (not elements) selects the
    # line style/marker; color still encodes the variant.
    threads = sorted(agg["threads"].unique())
    thr_style = {t: _ELEM_STYLES[i % len(_ELEM_STYLES)] for i, t in enumerate(threads)}
    thr_marker = {t: _ELEM_MARKERS[i % len(_ELEM_MARKERS)] for i, t in enumerate(threads)}

    fig, axes = plt.subplots(
        len(distributions),
        len(elements),
        figsize=(5.5 * len(elements), 4.5 * len(distributions)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for row, dist in enumerate(distributions):
        for col, elem in enumerate(elements):
            ax = axes[row][col]
            cell = agg[(agg["distribution"] == dist) & (agg["elements"] == elem)]
            groups = list(cell.groupby(["variant", "threads"]))
            # Spread series around each worker count so markers/bands at the same x
            # don't shadow each other (multiplicative on the log x-axis).
            for idx, ((variant, thr), g) in enumerate(groups):
                g = g.sort_values("workers")
                offset = (idx - (len(groups) - 1) / 2) * 0.03
                x = g["workers"] * (2.0 ** offset)
                color = _VARIANT_COLORS.get(variant, "tab:gray")
                ax.fill_between(
                    x, g["time_min"], g["time_max"], color=color, alpha=0.15
                )
                ax.plot(
                    x,
                    g["time_avg"],
                    color=color,
                    linestyle=thr_style.get(thr, "-"),
                    marker=thr_marker.get(thr, "o"),
                    label=f"{variant}, t={thr}",
                )
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.grid(True, which="both", linestyle=":", alpha=0.5)
            if row == 0:
                ax.set_title(f"n={elem:g}")
            if row == len(distributions) - 1:
                ax.set_xlabel("number of workers (ranks x threads)")
            if col == 0:
                ax.set_ylabel(f"{dist}\nrunning time [s] (avg; band=min/max)")

    # One combined legend (de-duplicated) for the whole figure.
    handles, labels = [], []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, fontsize=8, title="variant, threads",
               loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("alltoallv scaling")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=os.path.join(here, "data", "supermuc"),
        help="Directory holding <suite>_<date>/output/*.json (default: eval/data/supermuc)",
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        metavar="<suite>_<date>",
        help="Suite directory names (each holding output/*.json) to process, "
        "resolved relative to --data-dir. Default: all suites under --data-dir.",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(here, "alltoallv_scaling.png"),
        help="Output image path (default: eval/alltoallv_scaling.png)",
    )
    args = parser.parse_args()

    df = collect(args.data_dir, args.suites)
    source = ", ".join(args.suites) if args.suites else args.data_dir
    print(f"Parsed {len(df)} measurements from {source}")
    if df.empty:
        raise SystemExit("No data found.")

    agg = aggregate(df)
    df = df.sort_values(_GROUP + ["iteration"]).reset_index(drop=True)
    agg = agg.sort_values(_GROUP).reset_index(drop=True)
    with pd.option_context("display.max_rows", None, "display.max_colwidth", None):
        print("\nPer-iteration measurements:")
        print(df.to_string(index=False))
        print("\nAggregated over iterations (min/max/avg):")
        print(agg.to_string(index=False))

    plot(agg, args.out)


if __name__ == "__main__":
    main()
