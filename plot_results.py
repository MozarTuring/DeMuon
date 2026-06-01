"""Generate publication figures from experiment CSV results.

Usage:
    python plot_results.py --datadir <path-to-output-dir> [--outdir <figure-dir>]
    python plot_results.py --sources <sources.txt> [--outdir <figure-dir>]

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

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
    "dsgd_clip": {"label": "DSGD-C"},
    "gt_nsgdm":  {"label": "DSGD-N"},
    "demuon":    {"label": "DeMuon"},
}

# Per-topology mapping: logical name -> directory prefix used in CSV paths.
# For each method on each graph we pick the run with the lowest consensus error.
ALG_DIR_PREFIX = {
    "complete": {
        "dsgd":      "dsgd_dim2",
        "dsgd_clip": "dsgd_clip_dim2",
        "gt_nsgdm":  "gt_nsgdm2",
        "demuon":    "demuon_decay",
    },
    "exp": {
        "dsgd":      "dsgd_lin1",
        "dsgd_clip": "dsgd_clip_lin2",
        "gt_nsgdm":  "gt_nsgdm_lin2",
        "demuon":    "demuon_lin1",
    },
    "ring": {
        "dsgd":      "dsgd_lin2",
        "dsgd_clip": "dsgd_clip_lin6",
        "gt_nsgdm":  "gt_nsgdm_lin2",
        "demuon":    "demuon_lin1",
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


def extract_datadirs_from_sources(sources_path):
    """Extract unique data directories from a sources.txt file.
    Each CSV path like .../output/subdir/loss.csv implies datadir = .../output/."""
    datadirs = []
    seen = set()
    with open(sources_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("["):
                continue
            if ":" not in line:
                continue
            csv_path = line.split(":", 1)[1].strip()
            if not csv_path:
                continue
            datadir = str(Path(csv_path).parent.parent)
            if datadir not in seen:
                seen.add(datadir)
                datadirs.append(datadir)
    return datadirs


def find_csv(datadirs, name):
    for datadir in datadirs:
        d = Path(datadir) / name
        for cand in ["loss.csv", "loss_seed42.csv"]:
            p = d / cand
            if p.exists():
                return p
    return None


def validate_datadirs(datadirs):
    """Check that at least some CSVs can be found. Abort early with a clear
    error message if none of the data directories contain readable data."""
    found_any = False
    for topo in TOPOLOGIES:
        for alg_key in ALGORITHMS:
            csv_path = find_csv(datadirs, get_dir_name(alg_key, topo))
            if csv_path is not None:
                found_any = True
                break
        if found_any:
            break

    if not found_any:
        print("\nERROR: No CSV data files found in any of these directories:")
        for d in datadirs:
            print(f"  {d}")
        print("\nLooked for sub-directories like:")
        for topo in TOPOLOGIES:
            for alg_key in ALGORITHMS:
                print(f"  {get_dir_name(alg_key, topo)}/loss.csv")
        print("\nPlease check that --datadir / --sources points to the correct "
              "output directory on this machine.")
        raise SystemExit(1)


def write_source(pdf_path, sources, merged_file):
    """Append this figure's sources to the merged source file."""
    with open(merged_file, "a") as f:
        f.write(f"[{pdf_path.name}]\n")
        for label, csv_path in sources:
            f.write(f"  {label}: {csv_path}\n")
        f.write("\n")


def get_dir_name(alg_key, topo):
    """Return the CSV directory name for a given algorithm and topology."""
    prefix = ALG_DIR_PREFIX.get(topo, {}).get(alg_key, alg_key)
    return f"{prefix}_{topo}"


def plot_metric_by_topology(datadir, outdir, metric_col, ylabel, filename_suffix,
                            log_scale=False, neg_log_ticks=False, merged_file=None,
):
    """One figure per topology with all algorithms overlaid."""
    for topo, topo_title in TOPOLOGIES.items():
        fig, ax = plt.subplots()
        ax.set_prop_cycle(DEFAULT_CYCLE)
        sources = []
        for alg_key, style in ALGORITHMS.items():
            csv_path = find_csv(datadir, get_dir_name(alg_key, topo))
            if csv_path is None:
                print(f"  WARNING: no CSV for {style['label']} / {topo} "
                      f"(looked for {get_dir_name(alg_key, topo)}/loss.csv)")
                continue
            data = load_csv(csv_path)
            if metric_col not in data:
                print(f"  WARNING: column '{metric_col}' not found in {csv_path}")
                continue
            ax.plot(data["round"], data[metric_col],
                    markevery=MARKER_EVERY, label=style["label"])
            sources.append((style["label"], csv_path))

        if not sources:
            plt.close(fig)
            print(f"  SKIPPED {topo}_{filename_suffix}.pdf (no data found)")
            continue

        ax.set_xlabel("Iteration")
        if log_scale:
            ax.set_yscale("log")
        if neg_log_ticks:
            def _exp_fmt(y, _):
                if y <= 0:
                    return ''
                exp = -math.log10(y)
                if abs(exp - round(exp)) < 0.01:
                    return f'{int(round(exp))}'
                return ''
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(_exp_fmt))
            ax.yaxis.set_minor_formatter(ticker.NullFormatter())
            ax.set_ylabel(f"{ylabel} " + r"$(-\log_{10})$")
        else:
            ax.set_ylabel(ylabel)
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(True)

        out_path = Path(outdir) / f"{topo}_{filename_suffix}.pdf"
        fig.savefig(out_path)
        plt.close(fig)
        write_source(out_path, sources, merged_file)
        print(f"  Saved {out_path}")


def plot_training_loss_by_topology(datadir, outdir, merged_file=None):
    """Average training loss per round (mean of w0_train..w7_train)."""
    for topo, topo_title in TOPOLOGIES.items():
        fig, ax = plt.subplots()
        ax.set_prop_cycle(DEFAULT_CYCLE)
        sources = []
        for alg_key, style in ALGORITHMS.items():
            csv_path = find_csv(datadir, get_dir_name(alg_key, topo))
            if csv_path is None:
                print(f"  WARNING: no CSV for {style['label']} / {topo}")
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

        if not sources:
            plt.close(fig)
            print(f"  SKIPPED {topo}_Training loss.pdf (no data found)")
            continue

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Training Loss")
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(True)

        out_path = Path(outdir) / f"{topo}_Training loss.pdf"
        fig.savefig(out_path)
        plt.close(fig)
        write_source(out_path, sources, merged_file)
        print(f"  Saved {out_path}")


def plot_ablation(datadir, outdir, merged_file=None):
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

            if not sources:
                plt.close(fig)
                print(f"  SKIPPED {topo}_{suffix}.pdf (no data found)")
                continue

            ax.set_xlabel("Iteration")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=14, loc='upper right')
            ax.grid(True)

            out_path = Path(outdir) / f"{topo}_{suffix}.pdf"
            fig.savefig(out_path)
            plt.close(fig)
            write_source(out_path, sources, merged_file)
            print(f"  Saved {out_path}")


def plot_wall_clock(datadir, outdir, merged_file=None):
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

        if not sources:
            plt.close(fig)
            print(f"  SKIPPED {topo}_val_loss_vs_time.pdf (no data found)")
            continue

        ax.set_xlabel("Wall-Clock Time (hours)")
        ax.set_ylabel("Validation loss")
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(True)

        out_path = Path(outdir) / f"{topo}_val_loss_vs_time.pdf"
        fig.savefig(out_path)
        plt.close(fig)
        write_source(out_path, sources, merged_file)
        print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot experiment results")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--datadir", type=str, nargs='+',
                       help="One or more output/ directories (searched in order)")
    group.add_argument("--sources", type=str,
                       help="Path to a sources.txt file to replay plots from")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Directory for figures (default: timestamped dir)")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    default_outdir = f"/Users/jinma63/project/zzzjwmoutput/DeMuon/draws/{timestamp}"
    outdir = args.outdir or default_outdir
    os.makedirs(outdir, exist_ok=True)
    merged_file = Path(outdir) / "sources.txt"
    merged_file.unlink(missing_ok=True)  # start fresh
    print(f"Figures will be saved to: {outdir}\n")

    if args.sources:
        datadirs = extract_datadirs_from_sources(args.sources)
        print(f"Extracted data dirs from {args.sources}:")
        for d in datadirs:
            print(f"  {d}")
    else:
        datadirs = args.datadir
    print(f"\nData dirs: {datadirs}\n")

    validate_datadirs(datadirs)

    print("[1/6] Training loss (avg across workers)")
    plot_training_loss_by_topology(datadirs, outdir, merged_file)

    print("[2/6] Validation loss")
    plot_metric_by_topology(datadirs, outdir,
                            "avg_val_loss", "Validation Loss", "Validation loss",
                            merged_file=merged_file)

    print("[3/6] Validation perplexity")
    plot_metric_by_topology(datadirs, outdir,
                            "avg_val_ppl", "Perplexity", "perplexity",
                            log_scale=True, merged_file=merged_file)

    print("[4/6] Consensus error")
    plot_metric_by_topology(datadirs, outdir,
                            "consensus_err", "Consensus Error", "consensus_error",
                            log_scale=True, neg_log_ticks=True, merged_file=merged_file)

    print("[5/6] Ablation (val loss & consensus)")
    plot_ablation(datadirs, outdir, merged_file)

    print("[6/6] Validation loss vs wall-clock time")
    plot_wall_clock(datadirs, outdir, merged_file)

    print(f"\nDone. {len(list(Path(outdir).glob('*.pdf')))} PDF figures in {outdir}")
    print(f"Sources: {merged_file}")


if __name__ == "__main__":
    main()

"""
cd /Users/jinma63/project && python DeMuon/plot_results.py --sources /Users/jinma63/project/zzzjwmoutput/DeMuon/draws/20260407_111213/sources.txt
"""
