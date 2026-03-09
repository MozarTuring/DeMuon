"""Generate publication figures from experiment CSV results.

Usage:
    python plot_results.py --datadir <path-to-output-dir> [--outdir <figure-dir>]

Produces:
  1. Training loss   (per topology, all algorithms)
  2. Validation loss  (per topology, all algorithms)
  3. Validation perplexity (per topology, all algorithms)
  4. Consensus error  (per topology, all algorithms)
  5. Ablation: DeMuon w/ vs w/o msgn — val loss per topology
  6. Ablation: DeMuon w/ vs w/o msgn — consensus error per topology
"""

import argparse
import csv
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

# ── style (matching draw_jw.py) ──
mpl.rcParams['lines.linewidth'] = 0.5
mpl.rcParams['lines.linestyle'] = '-'
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 18
plt.rcParams["savefig.bbox"] = "tight"

MARKERS = ["o", "s", "D", "^"]
COLORS = list(plt.cm.tab10.colors[:4])
DEFAULT_CYCLE = cycler(color=COLORS, marker=MARKERS)
plt.rcParams['axes.prop_cycle'] = DEFAULT_CYCLE

# Unified 4-algorithm labels (same for all topologies)
ALGORITHMS = {
    "dsgd":      {"label": "DSGD"},
    "dsgd_clip": {"label": "DSGD_Clip"},
    "gt_nsgdm":  {"label": "GT_NSGDm"},
    "demuon":    {"label": "DeMuon"},
}

# Per-topology mapping: logical name -> directory prefix used in CSV paths.
# For each method on each graph we pick the run with the lowest consensus error.
ALG_DIR_PREFIX = {
    "complete": {
        "dsgd":      "dsgd_dim2",
        "dsgd_clip": "dsgd_clip_dim2",
        "gt_nsgdm":  "gt_nsgdm2",
        "demuon":    "demuon3",
    },
    "exp": {
        "dsgd":      "dsgd_dim",
        "dsgd_clip": "dsgd_clip_dim",
        "gt_nsgdm":  "gt_nsgdm_dim3",
        "demuon":    "demuon_invt2",
    },
    "ring": {
        "dsgd":      "dsgd_dim",
        "dsgd_clip": "dsgd_clip_dim",
        "gt_nsgdm":  "gt_nsgdm_dim",
        "demuon":    "demuon_invt3",
    },
}

TOPOLOGIES = {
    "complete": "Complete Graph",
    "exp":      "Directed Exponential Graph",
    "ring":     "Ring Graph",
}

ABLATION_COLORS = list(plt.cm.tab10.colors[:2])
ABLATION_MARKERS = ["s", "o"]
ABLATION_VARIANTS = {
    "ablation": {"label": "DeMuon w/o msgn"},
    "demuon":   {"label": "DeMuon w/ msgn"},
}

MARKER_EVERY = 10


def load_csv(path):
    """Load a loss CSV into a dict of column_name -> list of floats.
    If the data doesn't start at round 0, prepend a synthetic round-0 row
    using the first row's val loss values and consensus_err = 0."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        cols = {h: [] for h in header}
        for row in reader:
            for h, v in zip(header, row):
                cols[h].append(float(v))

    return cols


def find_csv(datadirs, name):
    for datadir in datadirs:
        d = Path(datadir) / name
        for cand in ["loss.csv", "loss_seed42.csv"]:
            p = d / cand
            if p.exists():
                return p
    return None


def write_source(pdf_path, sources):
    """Write a source.txt next to a PDF listing the CSVs it used."""
    txt_path = pdf_path.with_suffix(".source.txt")
    with open(txt_path, "w") as f:
        for label, csv_path in sources:
            f.write(f"{label}: {csv_path}\n")


def get_dir_name(alg_key, topo):
    """Return the CSV directory name for a given algorithm and topology."""
    prefix = ALG_DIR_PREFIX.get(topo, {}).get(alg_key, alg_key)
    return f"{prefix}_{topo}"


def plot_metric_by_topology(datadir, outdir, metric_col, ylabel, filename_suffix,
                            log_scale=False,
):
    """One figure per topology with all algorithms overlaid."""
    for topo, topo_title in TOPOLOGIES.items():
        fig, ax = plt.subplots()
        ax.set_prop_cycle(DEFAULT_CYCLE)
        sources = []
        for alg_key, style in ALGORITHMS.items():
            csv_path = find_csv(datadir, get_dir_name(alg_key, topo))
            if csv_path is None:
                continue
            data = load_csv(csv_path)
            if metric_col not in data:
                continue
            ax.plot(data["round"], data[metric_col],
                    markevery=MARKER_EVERY, label=style["label"])
            sources.append((style["label"], csv_path))

        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        if log_scale:
            ax.set_yscale("log")
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(True)

        out_path = Path(outdir) / f"{topo}_{filename_suffix}.pdf"
        fig.savefig(out_path)
        plt.close(fig)
        write_source(out_path, sources)
        print(f"  Saved {out_path}")


def plot_training_loss_by_topology(datadir, outdir):
    """Average training loss per round (mean of w0_train..w7_train)."""
    for topo, topo_title in TOPOLOGIES.items():
        fig, ax = plt.subplots()
        ax.set_prop_cycle(DEFAULT_CYCLE)
        sources = []
        for alg_key, style in ALGORITHMS.items():
            csv_path = find_csv(datadir, get_dir_name(alg_key, topo))
            if csv_path is None:
                continue
            data = load_csv(csv_path)
            train_cols = [c for c in data if c.endswith("_train")]
            if not train_cols:
                continue
            n = len(data["round"])
            avg_train = []
            for i in range(n):
                avg_train.append(sum(data[c][i] for c in train_cols) / len(train_cols))
            ax.plot(data["round"], avg_train,
                    markevery=MARKER_EVERY, label=style["label"])
            sources.append((style["label"], csv_path))

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Training loss")
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(True)

        out_path = Path(outdir) / f"{topo}_Training loss.pdf"
        fig.savefig(out_path)
        plt.close(fig)
        write_source(out_path, sources)
        print(f"  Saved {out_path}")


def plot_ablation(datadir, outdir):
    """Ablation figures: DeMuon w/ vs w/o msgn for val loss and consensus error."""
    ablation_cycle = cycler(color=ABLATION_COLORS, marker=ABLATION_MARKERS)
    for metric_col, ylabel, suffix in [
        ("avg_val_loss", "Validation Loss", "ablation_val_loss"),
        ("consensus_err", "Consensus Error", "ablation_consensus"),
    ]:
        for topo, topo_title in TOPOLOGIES.items():
            fig, ax = plt.subplots()
            ax.set_prop_cycle(ablation_cycle)
            sources = []
            for alg_key, style in ABLATION_VARIANTS.items():
                csv_path = find_csv(datadir, f"{alg_key}_{topo}")
                if csv_path is None:
                    continue
                data = load_csv(csv_path)
                if metric_col not in data:
                    continue
                ax.plot(data["round"], data[metric_col],
                        markevery=MARKER_EVERY, label=style["label"])
                sources.append((style["label"], csv_path))

            ax.set_xlabel("Iteration")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=12, loc='upper right')
            ax.grid(True)

            out_path = Path(outdir) / f"{topo}_{suffix}.pdf"
            fig.savefig(out_path)
            plt.close(fig)
            write_source(out_path, sources)
            print(f"  Saved {out_path}")


def plot_wall_clock(datadir, outdir):
    """Validation loss vs cumulative wall-clock time."""
    for topo, topo_title in TOPOLOGIES.items():
        fig, ax = plt.subplots()
        ax.set_prop_cycle(DEFAULT_CYCLE)
        sources = []
        for alg_key, style in ALGORITHMS.items():
            csv_path = find_csv(datadir, get_dir_name(alg_key, topo))
            if csv_path is None:
                continue
            data = load_csv(csv_path)
            if "cumul_time_sec" not in data or "avg_val_loss" not in data:
                continue
            hours = [t / 3600 for t in data["cumul_time_sec"]]
            ax.plot(hours, data["avg_val_loss"],
                    markevery=MARKER_EVERY, label=style["label"])
            sources.append((style["label"], csv_path))

        ax.set_xlabel("Wall-Clock Time (hours)")
        ax.set_ylabel("Validation loss")
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(True)

        out_path = Path(outdir) / f"{topo}_val_loss_vs_time.pdf"
        fig.savefig(out_path)
        plt.close(fig)
        write_source(out_path, sources)
        print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot experiment results")
    parser.add_argument("--datadir", type=str, nargs='+', required=True,
                        help="One or more output/ directories (searched in order)")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Directory for figures (default: <first datadir>/../figures)")
    args = parser.parse_args()

    datadirs = args.datadir
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    default_outdir = f"/Users/maojingwei/baidu/project/zzzjwmoutput/DeMuon/draws/{timestamp}"
    outdir = args.outdir or default_outdir
    os.makedirs(outdir, exist_ok=True)
    print(f"Data dirs: {datadirs}")
    print(f"Figures will be saved to: {outdir}\n")

    print("[1/6] Training loss (avg across workers)")
    plot_training_loss_by_topology(datadirs, outdir)

    print("[2/6] Validation loss")
    plot_metric_by_topology(datadirs, outdir,
                            "avg_val_loss", "Validation Loss", "Validation loss")

    print("[3/6] Validation perplexity")
    plot_metric_by_topology(datadirs, outdir,
                            "avg_val_ppl", "Perplexity", "perplexity",
                            log_scale=True)

    print("[4/6] Consensus error")
    plot_metric_by_topology(datadirs, outdir,
                            "consensus_err", "Consensus Error", "consensus_error",
                            log_scale=True)

    print("[5/6] Ablation (val loss & consensus)")
    plot_ablation(datadirs, outdir)

    print("[6/6] Validation loss vs wall-clock time")
    plot_wall_clock(datadirs, outdir)

    print(f"\nDone. {len(list(Path(outdir).glob('*.pdf')))} PDF figures in {outdir}")
    print(f"Per-figure source files: {outdir}/*.source.txt")


if __name__ == "__main__":
    main()
