"""Score the ten saved flat/patch runs and plot their entropy trajectories."""

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.score_entropy import score_checkpoint


LABELS = (0, 250, 500, 750, 1000)
SEEDS = set(range(42, 47))
CONDITIONS = ("flat", "patches")
METRICS = {
    "genome_shannon_bits": "Genome entropy",
    "energy_binned_entropy_bits": "Energy entropy",
    "infrastructure_binned_entropy_bits": "Infrastructure entropy",
}
COUNTS = ("alive_cells", "occupied_genome_ids")


def collect_rows(manifest, runs_dir):
    """Require complete coverage before scoring any of the 50 saved states."""
    if len(manifest) != 10 or len({run["run_id"] for run in manifest}) != 10:
        raise ValueError("Expected ten distinct runs in the manifest.")
    if {run["condition"] for run in manifest} != set(CONDITIONS):
        raise ValueError("Expected flat and patches conditions.")
    for condition in CONDITIONS:
        group = [run for run in manifest if run["condition"] == condition]
        if len(group) != 5 or {run["seed"] for run in group} != SEEDS:
            raise ValueError(f"Expected seeds 42–46 once each for {condition}.")

    jobs = []
    for run in manifest:
        if tuple(run["checkpoint_labels"]) != LABELS:
            raise ValueError(f"Unexpected checkpoint labels for {run['run_id']}.")
        for label in LABELS:
            checkpoint = (
                Path(runs_dir) / run["run_id"] / f"checkpoint_{label:07d}/substrate.pt"
            )
            jobs.append((run, label, checkpoint))
    missing = [str(path) for _, _, path in jobs if not path.is_file() or not path.stat().st_size]
    if missing:
        raise FileNotFoundError("Missing or empty checkpoints:\n" + "\n".join(missing))

    rows = []
    for run, label, checkpoint in jobs:
        result = score_checkpoint(checkpoint, bin_width=1.0)
        if result["grid_shape"] != run["shape"]:
            raise ValueError(f"Grid shape differs from the manifest: {checkpoint}")
        path = checkpoint.resolve()
        row = {
            "run_id": run["run_id"],
            "condition": run["condition"],
            "seed": run["seed"],
            "checkpoint_label": label,
            "checkpoint": (
                path.relative_to(ROOT).as_posix()
                if path.is_relative_to(ROOT) else path.as_posix()
            ),
            "source_commit_from_metadata": run["source_commit_from_metadata"],
            "bin_width": 1.0,
        }
        row.update({key: result[key] for key in (*COUNTS, *METRICS)})
        rows.append(row)
    return sorted(rows, key=lambda row: (row["condition"], row["seed"], row["checkpoint_label"]))


def summarize(rows):
    """Compute means and sample SDs across seeds, treating timepoints as repeated observations."""
    expected = {(condition, seed, label) for condition in CONDITIONS for seed in SEEDS for label in LABELS}
    observed = [(row["condition"], row["seed"], row["checkpoint_label"]) for row in rows]
    if len(observed) != 50 or set(observed) != expected:
        raise ValueError("Expected one observation for each condition, seed, and checkpoint.")
    summary = []
    for metric in METRICS:
        for condition in CONDITIONS:
            for label in LABELS:
                values = np.array([
                    row[metric] for row in rows
                    if row["condition"] == condition and row["checkpoint_label"] == label
                ], dtype=float)
                if not np.isfinite(values).all():
                    raise ValueError(f"Non-finite metric values: {metric}")
                summary.append({
                    "condition": condition,
                    "checkpoint_label": label,
                    "metric": metric,
                    "n_runs": len(values),
                    "mean": float(values.mean()),
                    "sample_sd": float(values.std(ddof=1)),
                })
    return summary


def verify_endpoints(rows, reference_path):
    """Check the ten final observations against the previously committed results."""
    with Path(reference_path).open(newline="", encoding="utf-8") as stream:
        reference = list(csv.DictReader(stream))
    expected = {(row["condition"], int(row["seed"])): row for row in reference}
    final = [row for row in rows if row["checkpoint_label"] == LABELS[-1]]
    if len(expected) != 10 or len(reference) != 10 or len(final) != 10:
        raise ValueError("Expected ten reference and final-checkpoint rows.")
    for row in final:
        key = row["condition"], row["seed"]
        if key not in expected:
            raise ValueError(f"Missing endpoint reference: {key}")
        old = expected[key]
        if int(old["checkpoint_label"]) != LABELS[-1] or float(old["bin_width"]) != 1.0:
            raise ValueError("Reference checkpoint label or bin width differs.")
        for metric in (*COUNTS, *METRICS):
            if not math.isclose(row[metric], float(old[metric]), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"Endpoint differs from the saved baseline: {key}, {metric}")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_entropy(summary, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {"flat": "#2563B8", "patches": "#F45B32"}
    ink, muted = "#253246", "#58677A"
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 16,
        "axes.labelsize": 16, "xtick.labelsize": 13, "ytick.labelsize": 13,
        "svg.fonttype": "none",
    }):
        figure = plt.figure(figsize=(16, 9), facecolor="white")
        figure.text(0.055, 0.94, "Entropy Across Saved Checkpoints", fontsize=30,
                    weight="bold", color=colors["patches"], va="top")
        figure.text(0.055, 0.865, "10 runs  |  5 seeds per environment  |  50 saved states",
                    fontsize=17, color=muted, va="top")
        figure.legend(
            handles=[Line2D([0], [0], color=colors[key], lw=3, marker="o", label=key.title())
                     for key in CONDITIONS],
            loc="upper right", bbox_to_anchor=(0.95, 0.875), frameon=False,
            ncol=2, fontsize=17, handlelength=1.6, columnspacing=1.4,
        )
        limits = [(5.4, 6.1), (0, 4.3), (0, 4.3)]
        for index, (metric, title) in enumerate(METRICS.items()):
            axis = figure.add_axes([0.065 + index * 0.318, 0.245, 0.267, 0.47])
            for condition in CONDITIONS:
                points = sorted(
                    [row for row in summary if row["metric"] == metric
                     and row["condition"] == condition],
                    key=lambda row: row["checkpoint_label"],
                )
                x = np.array([row["checkpoint_label"] for row in points])
                mean = np.array([row["mean"] for row in points])
                sd = np.array([row["sample_sd"] for row in points])
                axis.fill_between(x, mean - sd, mean + sd, color=colors[condition],
                                  alpha=0.15, linewidth=0)
                axis.plot(x, mean, color=colors[condition], lw=2.8, marker="o", markersize=5.5)
            axis.set_title(title, color=ink, pad=17, weight="bold", fontsize=19)
            axis.set_xlabel("Checkpoint label", color=ink, labelpad=12)
            axis.set_ylabel("Entropy (bits)", color=ink, labelpad=9)
            axis.set_xlim(-25, 1025)
            axis.set_ylim(*limits[index])
            axis.set_xticks(LABELS)
            axis.tick_params(colors=muted, length=0, pad=8)
            axis.grid(axis="y", color="#E3E8EF", linewidth=0.8)
            axis.set_axisbelow(True)
            for spine in ("top", "right"):
                axis.spines[spine].set_visible(False)
            for spine in ("left", "bottom"):
                axis.spines[spine].set_color("#CCD5E0")
        figure.text(0.055, 0.13,
                    "Lines: mean across 5 seeds. Shading: ±1 sample standard deviation.",
                    color=ink, fontsize=16)
        figure.text(0.055, 0.08,
                    "Genome: living cells. Resources: all cells; bin width 1.0. Panel y-axis ranges differ.",
                    color=muted, fontsize=14)
        figure.text(0.055, 0.04,
                    "Recorded checkpoint labels are used; legacy labels may differ from completed update counts.",
                    color=muted, fontsize=12)
        figure.savefig(output_dir / "entropy_over_time.png", dpi=150, facecolor="white")
        figure.savefig(output_dir / "entropy_over_time.svg", facecolor="white")
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "results/week6/run_manifest.json")
    parser.add_argument("--runs", type=Path, default=ROOT / "data/raw/runs")
    parser.add_argument("--reference", type=Path, default=ROOT / "results/week7/entropy_by_run.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/week8/entropy_over_time")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = collect_rows(manifest, args.runs)
    summary = summarize(rows)
    verify_endpoints(rows, args.reference)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "entropy_by_checkpoint.csv", rows)
    write_csv(args.output / "entropy_summary.csv", summary)
    plot_entropy(summary, args.output)
    print(f"Scored {len(rows)} checkpoints from {len(manifest)} runs.")
    print("All 10 endpoints match the saved baseline within 1e-12.")
    print("Final checkpoint: mean ± sample SD, in bits")
    for metric, title in METRICS.items():
        values = [row for row in summary if row["metric"] == metric
                  and row["checkpoint_label"] == LABELS[-1]]
        print(title + ": " + "; ".join(
            f"{row['condition']} {row['mean']:.3f} ± {row['sample_sd']:.3f}" for row in values
        ))
    print("Limitations: five seeds per condition; checkpoints within a run are repeated observations.")
    print("Legacy labels may differ from completed updates; original source commits are unknown.")
    print("Higher entropy does not establish better ecosystem quality or prompt alignment.")
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
