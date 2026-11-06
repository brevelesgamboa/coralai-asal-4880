"""Evaluate saved Week 9 cell occupancy alongside the validated Week 10 scores."""

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import shlex
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.score_entropy import score_checkpoint
from scripts.validate_saved_clip_siglip import (
    SEEDS, UPDATES, SIGLIP_MODEL, SIGLIP_REVISION, inspect_inputs, require, sha256,
)

# Only this inspected runner's log semantics are supported: its loop index is
# written AFTER executing that iteration, unlike screenshot/checkpoint labels.
RUNNER = "external/coralai/experiments/coral_dev/headless_coral.py"
RUNNER_SHA256 = "92bb59a99bfa7ad6b02a0431ef4520cda79e1cd01350867884cbb08ba1a977be"
SOURCE_FILES = (
    RUNNER,
    "external/coralai/experiments/coral_dev/experiment.py",
    "external/coralai/experiments/coral_dev/physics.py",
    "external/coralai/experiments/coral_dev/evolution.py",
    "external/coralai/experiments/coral_dev/environments.py",
    "external/coralai/coralai/evolver.py",
    "external/coralai/coralai/substrate.py",
    "external/coralai/coralai/experiment.py",
)


def relative(path):
    return os.path.relpath(Path(path).resolve(), ROOT)


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def load_population_log(path, grid_cells, completed_updates):
    """Convert the historical runner's loop labels to completed-update labels."""
    observations = {}
    for row in read_csv(path):
        logged_step = int(row["step"])
        update = logged_step + 1
        count = int(row["n_alive"])
        require(0 <= logged_step < completed_updates, f"Invalid log update: {path}")
        require(update not in observations, f"Duplicate log update: {path}")
        require(0 <= count <= grid_cells, f"Population outside grid: {path}")
        retained = int(row["n_genomes"])
        require(retained >= 0, f"Negative genome registry size: {path}")
        observations[update] = {
            "logged_step": logged_step, "living_cell_count": count,
            "living_cell_fraction": count / grid_cells,
            "population_source": "step_log", "population_artifact": relative(path),
            "occupied_genome_ids": None, "retained_genome_count": retained,
        }
    require(list(observations) == sorted(observations), f"Unordered log: {path}")
    return observations


def measure_state(path, shape, retained_genomes=None, image_path=None):
    """Reuse the existing tensor validator/count; also audit resource-only proxies."""
    import numpy as np
    import torch

    metrics = score_checkpoint(path)
    require(metrics["grid_shape"] == shape, f"State/grid mismatch: {path}")
    cells = torch.load(path, map_location="cpu", weights_only=True).numpy()[0]
    genome = cells[13]
    require(np.all((genome == -1) | (genome >= 0)), f"Unknown empty sentinel: {path}")
    if retained_genomes is not None:
        require(np.all(genome < retained_genomes), f"Invalid genome reference: {path}")
    occupied = genome >= 0
    resources = cells[0] + cells[1]
    grid_cells = math.prod(shape)
    if image_path is not None:
        from PIL import Image

        # Historical headless make_frame (not the differently scaled pilot
        # renderer): per-channel min/max, x/y transpose, and uint8 truncation.
        rgb = np.zeros((shape[1], shape[0], 3), dtype=np.float32)
        for color, channel_index in enumerate((0, 1, 13)):
            channel = cells[channel_index]
            lo, hi = channel.min(), channel.max()
            rgb[..., color] = ((channel - lo) / (hi - lo) if hi > lo else np.zeros_like(channel)).T
        rendered = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        with Image.open(image_path) as image:
            require(np.array_equal(rendered, np.asarray(image)), f"State/screenshot pixel mismatch: {image_path}")
    return {
        "logged_step": None, "living_cell_count": metrics["alive_cells"],
        "living_cell_fraction": metrics["alive_cells"] / grid_cells,
        "population_source": "checkpoint" if path.name == "substrate.pt" else "initial_state",
        "population_artifact": relative(path),
        "occupied_genome_ids": metrics["occupied_genome_ids"],
        "retained_genome_count": retained_genomes,
    }, {
        "unoccupied_cells_with_energy_plus_infra_above_death_threshold": int(((~occupied) & (resources > 0.05)).sum()),
        "occupied_cells_at_or_below_death_threshold": int((occupied & (resources <= 0.05)).sum()),
        "matches_saved_screenshot_pixels": True if image_path is not None else None,
    }


def zero_status(populations):
    """Describe observed zeroes without assuming an absorbing extinction state."""
    updates = sorted(populations)
    zeros = [u for u in updates if populations[u]["living_cell_count"] == 0]
    if not zeros:
        return None, "not_observed"
    recovered = any(u > zeros[0] and populations[u]["living_cell_count"] > 0 for u in updates)
    return zeros[0], "temporary_zero_observed" if recovered else "zero_through_last_observation"


def join_trajectory(run_id, seed, configuration, shape, populations, scores):
    """Outer join on completed update; never nearest-match or fill a missing value."""
    rows = []
    for update in sorted(populations.keys() | scores.keys()):
        pop, score = populations.get(update), scores.get(update)
        row = {
            "run_id": run_id, "seed": seed, "configuration": configuration,
            "update": update, "grid_width": shape[0], "grid_height": shape[1],
            "grid_cells": math.prod(shape),
            "match_status": "exact_match" if pop and score else "population_only" if pop else "score_only",
        }
        row.update(pop or dict.fromkeys(("logged_step", "living_cell_count", "living_cell_fraction",
                   "population_source", "population_artifact", "occupied_genome_ids", "retained_genome_count")))
        row.update({name: score[name] if score else None for name in
                    ("clip_alignment", "clip_loss", "siglip_logit", "image", "image_sha256")})
        rows.append(row)
    return rows


def inspect_scores(source, scores_dir, hashes):
    """Verify Week 9 reuse and Week 10 provenance without loading either model."""
    fresh = inspect_inputs(source)
    manifest = read_json(scores_dir / "manifest.json")
    summary = read_json(scores_dir / "summary.json")
    model = read_json(scores_dir / "siglip_model.json")
    execution = read_json(scores_dir / "execution.json")
    for name in ("manifest.json", "per_frame.csv", "per_seed.csv", "summary.json", "execution.json",
                 "siglip_model.json", "verification.json", "requirements-used.txt"):
        path = scores_dir / name
        hashes[relative(path)] = sha256(path)
    # Paths are resolved by run identity, so moving the checkout/input root is safe.
    old_runs = {(r["seed"], r["configuration"]): r for r in manifest["runs"]}
    require(set(old_runs) == {(r["seed"], r["configuration"]) for r in fresh["runs"]}, "Week 10 run identities differ")
    for path, digest in fresh["input_sha256"].items():
        hashes[path] = digest
        old_path = str(Path(manifest["source"]) / Path(path).relative_to(Path(fresh["source"])))
        require(manifest["input_sha256"].get(old_path) == digest, f"Week 10 input provenance differs: {path}")
    require(model["model"] == SIGLIP_MODEL and model["revision"] == SIGLIP_REVISION, "Unexpected SigLIP model")
    require(model["text"] == manifest["prompt"] == fresh["prompt"] == summary["prompt"], "Prompt mismatch")
    require(manifest["sampled_updates"] == summary["sampled_updates"] == UPDATES, "Scoring window differs")
    require(summary["n_frames"] == 120 and summary["n_runs"] == 6, "Unexpected score coverage")
    require(execution["script_sha256"] == sha256(ROOT / "scripts/validate_saved_clip_siglip.py"), "Scoring script provenance differs")
    rows = read_csv(scores_dir / "per_frame.csv")
    indexed = {}
    for row in rows:
        key = (int(row["seed"]), row["configuration"], int(row["step"]))
        require(key not in indexed, f"Duplicate score identity: {key}")
        for metric in ("clip_alignment", "clip_loss", "siglip_logit"):
            row[metric] = float(row[metric])
            require(math.isfinite(row[metric]), f"Nonfinite score: {key}")
        indexed[key] = row
    expected = set()
    for run in fresh["runs"]:
        old = old_runs[run["seed"], run["configuration"]]
        require(old["config"] == run["config"], "Scored configuration mismatch")
        require(old["clip_mean"] == run["clip_mean"], "Scored CLIP aggregate mismatch")
        require([f["step"] for f in old["frames"]] == UPDATES, "Week 10 manifest frame updates differ")
        for frame, old_frame in zip(run["frames"], old["frames"]):
            key = run["seed"], run["configuration"], frame["step"]
            expected.add(key)
            require(key in indexed, f"Missing score: {key}")
            row = indexed[key]
            require(row["image_sha256"] == frame["sha256"] == old_frame["sha256"], f"Score/image hash mismatch: {key}")
            require(row["clip_alignment"] == frame["clip_alignment"] == old_frame["clip_alignment"] and
                    row["clip_loss"] == frame["clip_loss"] == old_frame["clip_loss"], f"Changed cached CLIP: {key}")
            require(Path(row["image"]).name == Path(frame["image"]).name and
                    Path(row["image"]).parts[-5:-2] == (f"seed_{key[0]}", key[1], "run"), f"Score/image identity mismatch: {key}")
            row["image"] = frame["image"]
    require(set(indexed) == expected, "Unexpected score identities")
    per_seed = {int(r["seed"]): r for r in read_csv(scores_dir / "per_seed.csv")}
    require(set(per_seed) == set(SEEDS), "Week 10 summary seeds differ")
    require([r["seed"] for r in summary["per_seed"]] == list(SEEDS), "Week 10 summary seed identities differ")
    for pair in summary["per_seed"]:
        for configuration in ("default", "selected"):
            run = next(r for r in fresh["runs"] if (r["seed"], r["configuration"]) == (pair["seed"], configuration))
            clip = pair[f"clip_{configuration}_mean"]
            siglip = pair[f"siglip_{configuration}_mean"]
            require(clip == run["clip_mean"] == float(per_seed[pair["seed"]][f"clip_{configuration}_mean"]), "CLIP mean mismatch")
            actual = statistics.mean(indexed[pair["seed"], configuration, u]["siglip_logit"] for u in UPDATES)
            require(math.isclose(actual, siglip, abs_tol=1e-12) and siglip == float(per_seed[pair["seed"]][f"siglip_{configuration}_mean"]), "SigLIP mean mismatch")
        for model_name in ("clip", "siglip"):
            difference = pair[f"{model_name}_selected_mean"] - pair[f"{model_name}_default_mean"]
            preferred = "selected" if difference > 0 else "default" if difference < 0 else "tie"
            require(math.isclose(difference, pair[f"{model_name}_difference"], abs_tol=1e-12) and
                    preferred == pair[f"{model_name}_preferred"], "Score preference mismatch")
    return fresh, summary, indexed


def inspect_sources(execution, hashes):
    commit = execution["git_head"]
    require(execution["source_sha256"][RUNNER] == RUNNER_SHA256, "Unsupported runner log semantics")
    evidence = {}
    for name in SOURCE_FILES:
        blob = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
        digest = hashlib.sha256(blob).hexdigest()
        require(name not in execution["source_sha256"] or digest == execution["source_sha256"][name], f"Historical source hash mismatch: {name}")
        require(sha256(ROOT / name) == digest, f"Inspected source differs from historical simulator: {name}")
        hashes[name] = digest
        evidence[name] = {"git_commit": commit, "sha256": digest, "matches_current_source": True}
    return evidence


def plot_trajectories(rows, output):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/coralai-survival-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(12, 8.5), sharex=True, sharey="row", layout="constrained")
    colors = {"default": "#236ca4", "selected": "#cf6518"}
    for col, seed in enumerate(SEEDS):
        axes[0, col].set_title(f"Seed {seed}")
        for configuration, color in colors.items():
            group = [r for r in rows if r["seed"] == seed and r["configuration"] == configuration]
            for i, metric in enumerate(("living_cell_fraction", "clip_alignment", "siglip_logit")):
                available = [r for r in group if r[metric] is not None]
                axes[i, col].plot([r["update"] for r in available], [r[metric] for r in available],
                                  color=color, label=configuration, linewidth=1.5)
                if i == 0:
                    axes[i, col].scatter([r["update"] for r in available], [r[metric] for r in available], color=color, s=5)
                    final = next((r for r in available if r["update"] == 1000), None)
                    if final:
                        axes[i, col].scatter([1000], [final[metric]], color=color, marker="D", s=24, zorder=4)
        axes[2, col].set_xlabel("Completed updates")
        for axis in axes[:, col]:
            axis.grid(alpha=0.2)
            axis.axvline(1000, color="#555555", linestyle=":", linewidth=0.8)
            axis.set_xlim(-15, 1030)
        axes[0, col].set_ylim(0, 0.27)
    axes[0, 0].set_ylabel("Living-cell fraction\n(count / 10,000 grid cells)")
    axes[1, 0].set_ylabel("CLIP cosine alignment\n(higher is better)")
    axes[2, 0].set_ylabel("SigLIP image–text logit\n(higher is better)")
    axes[0, 2].legend(loc="lower right")
    fig.suptitle('"branching colonies" · six saved runs · separate model scales\n'
                 'Population: 0, 1, 11, …, 991, 1000; scores: 50, 100, …, 1000', fontsize=12)
    fig.supxlabel("Dots are observed populations; lines guide the eye, without imputing CSV values.\n"
                  "Diamond at 1000: sole exact population–score match per run; gaps do not exclude brief zeroes.", fontsize=9)
    fig.savefig(output / "trajectories.png", dpi=180)
    fig.savefig(output / "trajectories.svg")
    plt.close(fig)


def summary_table(summaries):
    lines = ["| Seed | Config | Initial cells | Final cells (u1000) | Final fraction | Zero observed | CLIP mean | SigLIP mean |",
             "|---|---|---:|---:|---:|---|---:|---:|"]
    for row in summaries:
        fraction = f"{row['final_living_cell_fraction']:.2%}" if row["final_living_cell_fraction"] is not None else "missing"
        zero = "No" if row["zero_observation_status"] == "not_observed" else row["zero_observation_status"]
        lines.append(f"| {row['seed']} | {row['configuration']} | {row['initial_living_cells']} | "
                     f"{row['final_living_cells']} | {fraction} | {zero} | "
                     f"{row['clip_mean_50_1000']:.6f} | {row['siglip_mean_50_1000']:.6f} |")
    return "\n".join(lines) + "\n\nMeans retain the original 20-frame window (updates 50–1000). CLIP uses the saved float32 aggregate; SigLIP uses the saved arithmetic mean.\n"


def survival_note(commit):
    return f"""# Survival definition

We measure survival using **occupied grid cells**:

- A cell counts as occupied when `genome >= 0` (state channel 13).
- `genome = -1` marks an empty or inactive location.
- **Population** is the number of occupied cells.
- **Occupied fraction** is population divided by all grid locations: 10,000 per run.

This measures occupancy, not the number of organisms or whether cells can
reproduce. Energy or bright pixels alone cannot tell us whether a cell is occupied.
The simulator marks cells inactive when energy + infrastructure is at most 0.05.

The historical logs have a timing offset: log step 50 describes completed
update 51. Only update 1,000 has population measured at exactly the same update
as the model scores. Other measurements stay at their recorded times; missing
values are left blank. Sparse observations cannot prove permanent extinction.

Verified against simulator commit `{commit}`:

- `external/coralai/experiments/coral_dev/experiment.py:32–43,152–164`: state fields and death rule.
- `external/coralai/experiments/coral_dev/physics.py:34–37`: inactive cells have disabled actions.
- `external/coralai/experiments/coral_dev/headless_coral.py:217–227`: recording timing and occupied-cell count.
"""


def interpret(summaries, pairs):
    by_id = {(r["seed"], r["configuration"]): r for r in summaries}
    differences = [p["final_cell_difference_selected_minus_default"] for p in pairs]
    all_positive = all(r["living_cells_remain_at_last_observation"] for r in summaries)
    default_declines = all(by_id[s, "default"]["final_minus_initial_cells"] < 0 for s in SEEDS)
    default_scores_rise = all(by_id[s, "default"][f"{m}_change_50_to_1000"] > 0 for s in SEEDS for m in ("clip", "siglip"))
    require(all_positive and default_declines and default_scores_rise, "Interpretation requires review for changed source data")
    require(all(r["zero_observation_status"] == "not_observed" for r in summaries), "Interpretation requires review of observed zeroes")
    return f"""All six saved ecosystems contain living cells at update 1,000, and none of the
recorded population observations is zero. Default ends with 1,912, 1,934, and
1,899 living cells on seeds 101, 202, and 303; selected ends with 2,304, 2,326,
and 2,400. Selected therefore retains {differences[0]}, {differences[1]}, and {differences[2]} more cells
than default on matched seeds. Selected populations grow from their initial
states, while default populations finish slightly below their initial counts.
Both configurations remain occupied at the final observation; greater selected
occupancy does not establish longer survival or uninterrupted persistence.

The original 20-frame means still favor default under CLIP on every seed.
SigLIP favors selected on seeds 101 and 202 and default on 303. Thus neither
model's preferences consistently rank final population size: both prefer the
less populated default run on seed 303, both in temporal means and at the
exactly matched final update. For every default run, both scores
increase from update 50 to 1,000 while population decreases between updates
0 and 1,000. These are different observation windows, so this is a descriptive
divergence, not a matched-frame association or evidence that score increases
cause population loss. No scored frame is confirmed to depict zero population.

The log's step label trails completed updates by one. Consequently, only the
six final checkpoints align exactly with the 120 scored frames; the other 114
frames lack same-update population measurements. Population and score curves
retain their actual sampling times. No values were interpolated. The study
contains six runs paired across three seeds, with temporally dependent frames,
separate model scales, and no causal or independent-frame statistical claims.
"""


def evaluate(args):
    source, scores_dir, output = args.source.resolve(), args.scores_dir.resolve(), args.output_dir.resolve()
    require(not output.exists(), f"Output already exists: {output}. Choose a fresh --output-dir.")
    hashes = {}
    for name in ("scripts/score_entropy.py", "scripts/validate_saved_clip_siglip.py"):
        hashes[name] = sha256(ROOT / name)
    fresh, score_summary, indexed_scores = inspect_scores(source, scores_dir, hashes)
    source_evidence = inspect_sources(fresh["original_clip_execution"], hashes)
    rows, summaries, runs, diagnostics, missing = [], [], [], [], []
    for run in fresh["runs"]:
        seed, configuration = run["seed"], run["configuration"]
        run_id = f"seed_{seed}/{configuration}"
        directory = source / "seed_comparison" / run_id / "run"
        metadata = read_json(directory / "meta.json")
        config = run["config"]
        shape = metadata["shape"]
        require(shape == [config["shape"], config["shape"]] and all(n > 0 for n in shape), f"Invalid denominator: {run_id}")
        require(metadata["experiment"] == "coral_dev" and metadata["seed"] == seed, f"State identity mismatch: {run_id}")
        completed = metadata["completed_steps"]
        populations, state_updates, log_updates = {}, [], []
        log_path = directory / "step_log.csv"
        if log_path.exists():
            hashes[relative(log_path)] = sha256(log_path)
            populations = load_population_log(log_path, math.prod(shape), completed)
            log_updates = sorted(populations)
        else:
            missing.append(relative(log_path))
        expected_logs = list(range(1, completed + 1, config["log_interval"]))
        missing_logs = sorted(set(expected_logs) - set(log_updates))
        require(set(log_updates) <= set(expected_logs), f"Unexpected log sampling: {run_id}")
        checkpoints = sorted(directory.glob("checkpoint_*/substrate.pt"))
        states = [(0, directory / "initial_state.pt", None)]
        for path in checkpoints:
            meta_path = path.parent / "meta.json"
            checkpoint = read_json(meta_path)
            hashes[relative(meta_path)] = sha256(meta_path)
            require(checkpoint["shape"] == shape and path.parent.name == f"checkpoint_{checkpoint['step']:07d}", f"Checkpoint identity mismatch: {path}")
            require(checkpoint["git_hash"] == fresh["original_clip_execution"]["git_head"], f"Simulator version mismatch: {path}")
            require(0 <= checkpoint["step"] <= completed, f"Checkpoint beyond run: {path}")
            population_path = path.parent / "population.pkl"
            if population_path.exists():
                hashes[relative(population_path)] = sha256(population_path)  # Never unpickle simulation objects.
            states.append((checkpoint["step"], path, checkpoint["n_genomes"]))
        final_path = directory / metadata["final_checkpoint"] / "substrate.pt"
        if not final_path.exists():
            missing.append(relative(final_path))
        for update, path, retained in states:
            if not path.exists():
                missing.append(relative(path))
                continue
            hashes[relative(path)] = sha256(path)
            image_path = directory / f"screenshots/step_{update:07d}.png"
            if image_path.exists():
                hashes[relative(image_path)] = sha256(image_path)
            population, diagnostic = measure_state(path, shape, retained, image_path if image_path.exists() else None)
            if update in populations:
                require(populations[update]["living_cell_count"] == population["living_cell_count"], f"State/log disagreement: {path}")
            populations[update] = population
            state_updates.append(update)
            diagnostics.append({"run_id": run_id, "update": update, **diagnostic})
        scores = {u: indexed_scores[seed, configuration, u] for u in UPDATES}
        trajectory = join_trajectory(run_id, seed, configuration, shape, populations, scores)
        rows.extend(trajectory)
        observed = sorted(populations)
        require(observed, f"No population observations: {run_id}")
        last = populations[observed[-1]]
        zero, status = zero_status(populations)
        minimum = min(observed, key=lambda u: populations[u]["living_cell_count"])
        pair = next(p for p in score_summary["per_seed"] if p["seed"] == seed)
        summary = {
            "run_id": run_id, "seed": seed, "configuration": configuration, "grid_cells": math.prod(shape),
            "initial_living_cells": populations.get(0, {}).get("living_cell_count"),
            "last_observed_update": observed[-1], "final_living_cells": last["living_cell_count"],
            "final_living_cell_fraction": last["living_cell_fraction"],
            "living_cells_remain_at_last_observation": last["living_cell_count"] > 0,
            "final_checkpoint_available": final_path.exists(),
            "final_minus_initial_cells": last["living_cell_count"] - populations[0]["living_cell_count"] if 0 in populations else None,
            "minimum_observed_living_cells": populations[minimum]["living_cell_count"], "minimum_observed_update": minimum,
            "first_observed_zero_update": zero, "zero_observation_status": status,
            "population_observations": len(observed), "missing_expected_log_observations": len(missing_logs),
            "scored_frames": len(scores), "exact_population_score_matches": len(set(observed) & set(scores)),
            "scored_frames_missing_population": len(set(scores) - set(observed)),
            "clip_mean_50_1000": pair[f"clip_{configuration}_mean"],
            "siglip_mean_50_1000": pair[f"siglip_{configuration}_mean"],
            "clip_change_50_to_1000": scores[1000]["clip_alignment"] - scores[50]["clip_alignment"],
            "siglip_change_50_to_1000": scores[1000]["siglip_logit"] - scores[50]["siglip_logit"],
            "clip_at_1000": scores[1000]["clip_alignment"], "siglip_at_1000": scores[1000]["siglip_logit"],
        }
        summaries.append(summary)
        runs.append({"run_id": run_id, "seed": seed, "configuration": configuration,
                     "config": config, "grid_shape": shape, "grid_cells": math.prod(shape),
                     "simulator_git_commit": fresh["original_clip_execution"]["git_head"],
                     "run_directory": relative(directory), "log_recorded_steps": [u - 1 for u in log_updates],
                     "log_completed_updates": log_updates, "tensor_updates": state_updates,
                     "population_updates": observed, "missing_expected_log_updates": missing_logs,
                     "score_updates": UPDATES, "exact_match_updates": sorted(set(observed) & set(scores)),
                     "score_updates_without_population": sorted(set(scores) - set(observed)),
                     "snapshotted_experiment_source_available": (directory / "snapshot").exists()})
        print(f"{run_id}: {len(observed)} population observations; last u{observed[-1]} = {last['living_cell_count']} cells; {summary['exact_population_score_matches']} exact score matches")
    pairs = []
    for seed in SEEDS:
        default, selected = [next(r for r in summaries if (r["seed"], r["configuration"]) == (seed, c)) for c in ("default", "selected")]
        require(default["last_observed_update"] == selected["last_observed_update"], "Unmatched final population times")
        old = next(r for r in score_summary["per_seed"] if r["seed"] == seed)
        pairs.append({"seed": seed, "population_comparison_update": default["last_observed_update"],
                      "final_cell_difference_selected_minus_default": selected["final_living_cells"] - default["final_living_cells"],
                      "final_fraction_difference_selected_minus_default": selected["final_living_cell_fraction"] - default["final_living_cell_fraction"],
                      "both_have_living_cells_at_last_observation": default["living_cells_remain_at_last_observation"] and selected["living_cells_remain_at_last_observation"],
                      "clip_at_1000_difference_selected_minus_default": selected["clip_at_1000"] - default["clip_at_1000"],
                      "siglip_at_1000_difference_selected_minus_default": selected["siglip_at_1000"] - default["siglip_at_1000"],
                      **{key: old[key] for key in ("clip_difference", "clip_preferred", "siglip_difference", "siglip_preferred")}})
    # Text below is specific to this six-run historical dataset, not a generic report generator.
    interpretation = interpret(summaries, pairs) if not missing and not any(r["missing_expected_log_updates"] for r in runs) else "Population coverage is incomplete; consult manifest.json and per_run_summary.csv. Same-update scores are retained only where measurements exist. No extinction conclusion is supported by missing observations.\n"
    for path, digest in hashes.items():
        require(sha256(ROOT / path) == digest, f"Input changed during analysis: {path}")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"source": relative(source), "scores_dir": relative(scores_dir), "prompt": fresh["prompt"],
                "survival_measure": "simulator occupancy: genome >= 0; living-cell fraction uses all grid cells",
                "log_update_rule": "completed_update = logged_step + 1, after all simulation operations",
                "join_keys": ["configuration", "seed", "completed_update"],
                "missing_population_policy": "blank/null; no interpolation, nearest match, or zero imputation",
                "source_code": source_evidence, "runs": runs, "input_sha256": hashes,
                "state_diagnostics": diagnostics, "missing_state_artifacts": missing,
                "missing_dependency": None,
                "historical_coverage_limitation": "No same-update population at scored updates 50..950; no intermediate tensors or frozen run/snapshot source. Historical source verified from the saved git commit.",
                "smallest_future_logging_change": "Use evolver.timestep for both the log interval condition and the recorded update label; preserve n_alive calculation."}
    write_csv(output / "trajectory.csv", rows)
    write_csv(output / "per_run_summary.csv", summaries)
    write_csv(output / "paired_comparison.csv", pairs)
    (output / "per_run_summary.md").write_text(summary_table(summaries))
    (output / "survival_definition.md").write_text(survival_note(fresh["original_clip_execution"]["git_head"]))
    (output / "interpretation.md").write_text(interpretation)
    save_json(output / "manifest.json", manifest)
    plot_trajectories(rows, output)
    command = shlex.join([sys.executable, "scripts/evaluate_survival.py", "--source", relative(source), "--scores-dir", relative(scores_dir), "--output-dir", relative(output)])
    reproduction = shlex.join([sys.executable, "scripts/evaluate_survival.py", "--source", relative(source),
                               "--scores-dir", relative(scores_dir), "--output-dir", "/tmp/coralai-survival-repro"])
    import torch

    execution = {"command": command, "reproduction_command": reproduction, "cwd": str(ROOT),
                 "python": sys.executable, "python_version": platform.python_version(), "platform": platform.platform(),
                 "packages": {p: version(p) for p in ("torch", "numpy", "pillow", "matplotlib")},
                 "torch_runtime_version": torch.__version__, "analysis_device": "cpu",
                 "script_sha256": sha256(__file__), "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                 "historical_simulator_environment": fresh["original_clip_execution"]["versions"],
                 "historical_scoring_environment": read_json(scores_dir / "execution.json"),
                 "models_recomputed": False, "simulations_rerun": False}
    save_json(output / "execution.json", execution)
    (output / "reproduce.txt").write_text("From the repository root; requires Python with Torch, NumPy, Pillow, Matplotlib, and git history.\nExisting outputs are protected: use a fresh output directory. No model weights or Coralai runtime installation required.\n\nOriginal execution:\n" + command + "\n\nReproduce to a fresh directory:\n" + reproduction + "\n\nChecks:\npython -m unittest discover -s tests -p 'test_survival_evaluation.py'\npython -m unittest discover -s tests -p 'test_entropy_over_time.py'\npython scripts/validate_saved_clip_siglip.py --check-inputs\n\nVersions and historical environments: execution.json. Source hashes and update coverage: manifest.json.\n")
    verification = {"status": "passed", "six_run_identities_verified": True, "grid_denominators_verified": True,
                    "all_population_ranges_valid": all(0 <= r["living_cell_fraction"] <= 1 for r in rows if r["living_cell_fraction"] is not None),
                    "trajectory_rows": len(rows), "population_observations": sum(r["living_cell_count"] is not None for r in rows),
                    "scored_frames": sum(r["clip_alignment"] is not None for r in rows),
                    "exact_population_score_matches": sum(r["match_status"] == "exact_match" for r in rows),
                    "scored_frames_missing_population": sum(r["match_status"] == "score_only" for r in rows),
                    "initial_and_final_tensors_validated": len(diagnostics),
                    "state_screenshot_pixel_matches": sum(d["matches_saved_screenshot_pixels"] is True for d in diagnostics),
                    "cached_clip_and_siglip_finite_and_means_verified": True,
                    "source_hashes_unchanged_after_analysis": True,
                    "missing_state_artifacts": missing,
                    "automated_checks_only": True, "visual_review": "Inspect trajectories.png; review recorded separately in review.json."}
    save_json(output / "verification.json", verification)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""From the repository root:
  python scripts/evaluate_survival.py
  python scripts/evaluate_survival.py --source results/week9/supervised_coralai_20261022 \\
    --scores-dir results/week10/model_validation --output-dir /tmp/survival_repro

Needs Torch, NumPy, Pillow, Matplotlib and the recorded simulator git commit.
Reuses cached scores; never imports simulation runtime or loads model weights.
Logs use loop indices: row step=s measures completed update s+1. Exact outer
join only. Missing populations/scores stay blank; original windows unchanged.
Output directory must not exist. Outputs: trajectory.csv, per_run_summary.csv/.md,
paired_comparison.csv, trajectories.png/.svg, survival_definition.md,
interpretation.md, manifest.json, execution.json, verification.json, reproduce.txt.
""")
    parser.add_argument("--source", type=Path, default=ROOT / "results/week9/supervised_coralai_20261022", help="Saved Week 9 experiment root")
    parser.add_argument("--scores-dir", type=Path, default=ROOT / "results/week10/model_validation", help="Validated Week 10 scores and provenance")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/week11/survival_evaluation", help="Fresh output directory")
    args = parser.parse_args()
    try:
        evaluate(args)
    except (ValueError, FileNotFoundError, KeyError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"Survival evaluation stopped: {error}\n")


if __name__ == "__main__":
    main()
