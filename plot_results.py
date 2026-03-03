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

MARKERS = ["o", "s", "D", "^", "v"]
COLORS = list(plt.cm.tab10.colors[:4])
plt.rcParams['axes.prop_cycle'] = cycler(color=COLORS, marker=MARKERS[:4])

ALGORITHMS = {
    "dsgd":      {"label": "DSGD"},
    "dsgd_clip": {"label": "DSGD_Clip"},
    "gt_nsgdm":  {"label": "GT_NSGDm"},
    "demuon":    {"label": "DeMuon"},
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
    """Load a loss CSV into a dict of column_name -> list of floats."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        cols = {h: [] for h in header}
        for row in reader:
            for h, v in zip(header, row):
                cols[h].append(float(v))
    return cols


def find_csv(datadir, name):
    d = Path(datadir) / name
    for cand in ["loss.csv", "loss_seed42.csv"]:
        p = d / cand
        if p.exists():
            return p
    return None


def plot_metric_by_topology(datadir, outdir, metric_col, ylabel, filename_suffix,
                            algorithms=ALGORITHMS, log_scale=False):
    """One figure per topology with all algorithms overlaid."""
    for topo, topo_title in TOPOLOGIES.items():
        fig, ax = plt.subplots()
        for alg_key, style in algorithms.items():
            csv_path = find_csv(datadir, f"{alg_key}_{topo}")
            if csv_path is None:
                continue
            data = load_csv(csv_path)
            if metric_col not in data:
                continue
            ax.plot(data["round"], data[metric_col],
                    markevery=MARKER_EVERY, label=style["label"])

        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        if log_scale:
            ax.set_yscale("log")
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(True)

        out_path = Path(outdir) / f"{topo}_{filename_suffix}.pdf"
        fig.savefig(out_path)
        plt.close(fig)
        print(f"  Saved {out_path}")


def plot_training_loss_by_topology(datadir, outdir):
    """Average training loss per round (mean of w0_train..w7_train)."""
    for topo, topo_title in TOPOLOGIES.items():
        fig, ax = plt.subplots()
        for alg_key, style in ALGORITHMS.items():
            csv_path = find_csv(datadir, f"{alg_key}_{topo}")
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

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Training loss")
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(True)

        out_path = Path(outdir) / f"{topo}_Training loss.pdf"
        fig.savefig(out_path)
        plt.close(fig)
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
            for alg_key, style in ABLATION_VARIANTS.items():
                csv_path = find_csv(datadir, f"{alg_key}_{topo}")
                if csv_path is None:
                    continue
                data = load_csv(csv_path)
                if metric_col not in data:
                    continue
                ax.plot(data["round"], data[metric_col],
                        markevery=MARKER_EVERY, label=style["label"])

            ax.set_xlabel("Iteration")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=12, loc='upper right')
            ax.grid(True)

            out_path = Path(outdir) / f"{topo}_{suffix}.pdf"
            fig.savefig(out_path)
            plt.close(fig)
            print(f"  Saved {out_path}")


def plot_wall_clock(datadir, outdir):
    """Validation loss vs cumulative wall-clock time."""
    for topo, topo_title in TOPOLOGIES.items():
        fig, ax = plt.subplots()
        for alg_key, style in ALGORITHMS.items():
            csv_path = find_csv(datadir, f"{alg_key}_{topo}")
            if csv_path is None:
                continue
            data = load_csv(csv_path)
            if "cumul_time_sec" not in data or "avg_val_loss" not in data:
                continue
            hours = [t / 3600 for t in data["cumul_time_sec"]]
            ax.plot(hours, data["avg_val_loss"],
                    markevery=MARKER_EVERY, label=style["label"])

        ax.set_xlabel("Wall-Clock Time (hours)")
        ax.set_ylabel("Validation loss")
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(True)

        out_path = Path(outdir) / f"{topo}_val_loss_vs_time.pdf"
        fig.savefig(out_path)
        plt.close(fig)
        print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot experiment results")
    parser.add_argument("--datadir", type=str, required=True,
                        help="Path to output/ directory containing experiment subdirs")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Directory for figures (default: <datadir>/../figures)")
    args = parser.parse_args()

    outdir = args.outdir or str(Path(args.datadir).parent / "figures")
    os.makedirs(outdir, exist_ok=True)
    print(f"Figures will be saved to: {outdir}\n")

    print("[1/6] Training loss (avg across workers)")
    plot_training_loss_by_topology(args.datadir, outdir)

    print("[2/6] Validation loss")
    plot_metric_by_topology(args.datadir, outdir,
                            "avg_val_loss", "Validation Loss", "Validation loss")

    print("[3/6] Validation perplexity")
    plot_metric_by_topology(args.datadir, outdir,
                            "avg_val_ppl", "Perplexity", "perplexity",
                            log_scale=True)

    print("[4/6] Consensus error")
    plot_metric_by_topology(args.datadir, outdir,
                            "consensus_err", "Consensus Error", "consensus_error",
                            log_scale=True)

    print("[5/6] Ablation (val loss & consensus)")
    plot_ablation(args.datadir, outdir)

    print("[6/6] Validation loss vs wall-clock time")
    plot_wall_clock(args.datadir, outdir)

    print(f"\nDone. {len(list(Path(outdir).glob('*.pdf')))} PDF figures in {outdir}")


if __name__ == "__main__":
    main()
