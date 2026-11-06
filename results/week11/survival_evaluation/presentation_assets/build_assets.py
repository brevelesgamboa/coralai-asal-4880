"""Build classroom views from the existing survival evaluation; no simulation/inference."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import PIL
import torch


ROOT = Path(__file__).resolve().parents[4]
LABELS = {"default": "Default", "selected": "ASAL-selected"}
EXPECTED = {"default": 1899, "selected": 2400}
PANEL_SIZE, SCALE = 500, 5
X_POSITIONS, CANVAS_WIDTH = (40, 580), 1120
INK, MUTED, BACKGROUND = "#16283c", "#4c5d6d", "#ffffff"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    return os.path.relpath(Path(path).resolve(), ROOT)


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def font(size, bold=False):
    return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)


def centered(draw, text, y, size=28, center=CANVAS_WIDTH // 2, bold=False, fill=INK):
    chosen = font(size, bold)
    box = draw.textbbox((0, 0), text, font=chosen)
    require(box[2] - box[0] <= CANVAS_WIDTH - 40, f"Label exceeds canvas: {text}")
    draw.text((center, y), text, font=chosen, fill=fill, anchor="mt")


def read_rgb(path):
    with Image.open(path) as source:
        require(source.mode == "RGB" and source.size == (100, 100), f"Unexpected source image: {path}")
        return source.copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, default=ROOT / "results/week11/survival_evaluation")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="Directory without existing generated assets; the builder may already be present")
    args = parser.parse_args()
    evaluation, output = args.evaluation_dir.resolve(), args.output_dir.resolve()
    asset_names = ("seed_303_rgb_comparison.gif", "seed_303_endpoint_comparison.png",
                   "endpoint_population_all_seeds.png", "manifest.json", "verification.json")
    require(not any((output / name).exists() for name in asset_names), "Generated assets already exist; use a fresh --output-dir")
    inputs = {}

    def track(path):
        inputs[relative(path)] = digest(path)
        return path

    manifest_path = track(evaluation / "manifest.json")
    summary_path = track(evaluation / "per_run_summary.csv")
    manifest = json.loads(manifest_path.read_text())
    with summary_path.open(newline="") as stream:
        summaries = list(csv.DictReader(stream))
    model_manifest_path = track(ROOT / manifest["scores_dir"] / "manifest.json")
    model_manifest = json.loads(model_manifest_path.read_text())
    renderer = track(ROOT / "external/coralai/experiments/coral_dev/headless_coral.py")
    require(digest(renderer) == manifest["source_code"][relative(renderer)]["sha256"], "Renderer differs from inspected historical source")
    runs = {run["configuration"]: run for run in manifest["runs"] if run["seed"] == 303}
    scored_runs = {run["configuration"]: run for run in model_manifest["runs"] if run["seed"] == 303}
    require(set(runs) == set(scored_runs) == set(LABELS), "Missing or ambiguous seed 303 runs")
    updates = model_manifest["sampled_updates"]
    require(updates == list(range(50, 1001, 50)), "Unexpected recorded frame sequence")
    images, frames, masks, counts, frame_sources = {}, [], {}, {}, []
    for configuration, run in runs.items():
        directory = ROOT / run["run_directory"]
        require(run["run_id"] == f"seed_303/{configuration}", "Manifest run identity mismatch")
        metadata = json.loads(track(directory / "meta.json").read_text())
        require(metadata["seed"] == 303 and metadata["shape"] == [100, 100], "State identity/grid mismatch")
        recorded_frames = scored_runs[configuration]["frames"]
        require([frame["step"] for frame in recorded_frames] == updates, "Unsynchronized source frame sequence")
        images[configuration] = {}
        for recorded in recorded_frames:
            update = recorded["step"]
            path = track(directory / "screenshots" / f"step_{update:07d}.png")
            require(digest(path) == recorded["sha256"] == manifest["input_sha256"][relative(path)], "Saved RGB frame hash mismatch")
            require((ROOT / recorded["image"]).resolve() == path.resolve(), "Frame manifest points to another run")
            images[configuration][update] = read_rgb(path)
            frame_sources.append({"seed": 303, "configuration": configuration, "completed_update": update,
                                  "path": relative(path), "sha256": digest(path)})
        state_path = track(directory / metadata["final_checkpoint"] / "substrate.pt")
        checkpoint = json.loads(track(state_path.parent / "meta.json").read_text())
        require(checkpoint["step"] == 1000 and checkpoint["shape"] == [100, 100], "Wrong endpoint checkpoint")
        require(digest(state_path) == manifest["input_sha256"][relative(state_path)], "State hash mismatch")
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        require(tuple(state.shape) == (1, 14, 100, 100) and torch.isfinite(state).all().item(), "Invalid endpoint state")
        genome = state[0, 13].numpy()
        require(np.equal(genome, np.floor(genome)).all(), "Noninteger genome IDs")
        mask = genome >= 0
        counts[configuration] = int(mask.sum())
        summary = next(row for row in summaries if int(row["seed"]) == 303 and row["configuration"] == configuration)
        require(int(summary["last_observed_update"]) == 1000 and counts[configuration] == int(summary["final_living_cells"]) == EXPECTED[configuration], "Endpoint count differs from verified summary")
        # Saved tensor axes are x,y; the RGB renderer transposes to row=y,col=x.
        masks[configuration] = Image.fromarray((mask.T * 255).astype(np.uint8)).convert("RGB")
        cells = state.numpy()[0]
        rendered = np.zeros((100, 100, 3), dtype=np.float32)
        for color, index in enumerate((0, 1, 13)):
            channel = cells[index]
            lo, hi = channel.min(), channel.max()
            rendered[..., color] = ((channel - lo) / (hi - lo) if hi > lo else np.zeros_like(channel)).T
        require(np.array_equal(np.clip(rendered * 255, 0, 255).astype(np.uint8), np.asarray(images[configuration][1000])), "Endpoint state does not reproduce its RGB screenshot")

    for update in updates:
        canvas = Image.new("RGB", (CANVAS_WIDTH, 760), BACKGROUND)
        draw = ImageDraw.Draw(canvas)
        centered(draw, "Coralai  |  Seed 303", 22, 34, bold=True)
        centered(draw, f"Completed update {update:,}", 76, 30)
        for configuration, x in zip(LABELS, X_POSITIONS):
            centered(draw, LABELS[configuration], 137, 30, x + PANEL_SIZE // 2, bold=True)
            canvas.paste(images[configuration][update].resize((PANEL_SIZE, PANEL_SIZE), Image.Resampling.NEAREST), (x, 188))
        centered(draw, "Original RGB screenshots  |  Saved every 50 updates", 714, 22, fill=MUTED)
        frames.append(canvas)
    # GIF supports at most 256 palette colors. A shared palette avoids adding
    # time-dependent palette choices; the source images and static PNG stay RGB.
    # Learn the palette from source pixels, with black/white text samples.
    atlas = Image.new("RGB", (200, 100 * len(updates) + 100), BACKGROUND)
    for i, update in enumerate(updates):
        for configuration, x in zip(LABELS, (0, 100)):
            atlas.paste(images[configuration][update], (x, i * 100))
    ImageDraw.Draw(atlas).rectangle((0, 100 * len(updates), 99, atlas.height), fill=INK)
    palette = atlas.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    paletted = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    output.mkdir(parents=True, exist_ok=True)
    gif_path = output / asset_names[0]
    paletted[0].save(gif_path, save_all=True, append_images=paletted[1:], duration=500,
                     loop=0, disposal=2, optimize=False)

    endpoint = Image.new("RGB", (CANVAS_WIDTH, 1420), BACKGROUND)
    draw = ImageDraw.Draw(endpoint)
    centered(draw, "Coralai endpoint comparison", 22, 36, bold=True)
    centered(draw, "Seed 303  |  Completed update 1,000  |  100 × 100 cells", 80, 27)
    for configuration, x in zip(LABELS, X_POSITIONS):
        center = x + PANEL_SIZE // 2
        centered(draw, LABELS[configuration], 138, 31, center, bold=True)
        centered(draw, "Original RGB screenshot", 187, 23, center)
        endpoint.paste(images[configuration][1000].resize((PANEL_SIZE, PANEL_SIZE), Image.Resampling.NEAREST), (x, 226))
        centered(draw, "Analysis view: occupied-cell mask", 754, 24, center, bold=True)
        endpoint.paste(masks[configuration].resize((PANEL_SIZE, PANEL_SIZE), Image.Resampling.NEAREST), (x, 800))
        centered(draw, f"{counts[configuration]:,} occupied cells  |  {counts[configuration] / 100:.2f}%", 1327, 27, center, bold=True)
    centered(draw, "Mask only: white = occupied (genome ≥ 0); black = unoccupied", 1380, 23, fill=MUTED)
    endpoint.save(output / asset_names[1])

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/coralai-presentation-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    require(len(summaries) == 6 and {(int(row["seed"]), row["configuration"]) for row in summaries} ==
            {(s, c) for s in (101, 202, 303) for c in LABELS}, "Endpoint chart needs all six runs")
    fig, axis = plt.subplots(figsize=(9, 5.3), layout="constrained")
    positions = np.arange(3)
    chart_values = {}
    for offset, (configuration, label), color in zip((-0.19, 0.19), LABELS.items(), ("#236ca4", "#cf6518")):
        values = []
        for seed in (101, 202, 303):
            summary = next(row for row in summaries if int(row["seed"]) == seed and row["configuration"] == configuration)
            require(int(summary["last_observed_update"]) == 1000 and int(summary["grid_cells"]) == 10000, "Incomparable endpoint chart denominator/update")
            require(float(summary["final_living_cell_fraction"]) == int(summary["final_living_cells"]) / 10000, "Summary fraction mismatch")
            values.append(int(summary["final_living_cells"]))
        chart_values[configuration] = values
        bars = axis.bar(positions + offset, values, width=0.35, label=label, color=color)
        axis.bar_label(bars, labels=[f"{n:,}" for n in values], padding=5, fontsize=12)
    axis.set_xticks(positions, ["101", "202", "303"])
    axis.set_xlabel("Matched simulation seed", fontsize=12)
    axis.set_ylabel("Occupied grid cells", fontsize=12)
    axis.set_ylim(0, 2900)
    axis.set_title("Population remaining at completed update 1,000", fontsize=16, pad=18)
    axis.legend(loc="upper left", frameon=False, ncols=2, fontsize=11)
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=11)
    fig.supxlabel("Each grid contains 10,000 cells. Cell counts are not organism counts.", fontsize=11)
    fig.savefig(output / asset_names[2], dpi=180)
    plt.close(fig)

    # Decode the actual GIF and compare every panel to the intended source pair
    # after the sole unavoidable palette conversion; check static PNGs losslessly.
    gif_durations = []
    with Image.open(gif_path) as gif:
        require(gif.n_frames == len(updates) and gif.info["loop"] == 0, "GIF frame/loop mismatch")
        for i, update in enumerate(updates):
            gif.seek(i)
            decoded = gif.convert("RGB")
            require(np.array_equal(np.asarray(decoded), np.asarray(paletted[i].convert("RGB"))), f"GIF pair/update mismatch at {update}")
            gif_durations.append(gif.info["duration"])
    require(gif_durations == [500] * 20 and sum(gif_durations) == 10000, "Wrong GIF duration")
    with Image.open(output / asset_names[1]) as saved_endpoint:
        for configuration, x in zip(LABELS, X_POSITIONS):
            rgb = saved_endpoint.crop((x, 226, x + PANEL_SIZE, 726))
            mask = saved_endpoint.crop((x, 800, x + PANEL_SIZE, 1300))
            require(np.array_equal(np.asarray(rgb.resize((100, 100), Image.Resampling.NEAREST)), np.asarray(images[configuration][1000])), "Static RGB pixels changed")
            recovered_mask = np.asarray(mask)[::SCALE, ::SCALE, 0] == 255
            require(int(recovered_mask.sum()) == counts[configuration] and np.array_equal(recovered_mask, np.asarray(masks[configuration])[:, :, 0] == 255), "Saved mask count/orientation mismatch")
    require(all(digest(ROOT / p) == sha for p, sha in inputs.items()), "An input file changed during generation")
    command = shlex.join([sys.executable, relative(__file__), "--evaluation-dir", relative(evaluation), "--output-dir", relative(output)])
    save_json(output / "manifest.json", {
        "command": command, "reproduce_to_fresh_directory": command.rsplit("--output-dir", 1)[0] + "--output-dir /tmp/coralai-classroom-repro",
        "input_sha256": inputs, "frame_sources": frame_sources,
        "seed": 303, "frame_completed_updates": updates, "frame_duration_ms": gif_durations, "loop_duration_seconds": 10,
        "image_scale": "5× nearest-neighbor; no spatial interpolation", "gif_color_conversion": "Shared 256-color palette, no dithering; original PNGs unchanged",
        "endpoint_masks": {"completed_update": 1000, "rule": "state[0,13] >= 0", "orientation": "tensor x,y transposed to image row=y,col=x", "counts": counts},
        "endpoint_population_by_seed_101_202_303": chart_values,
        "renderer": {"source": relative(renderer), "lines": "119–130", "channels": ["red=energy", "green=infrastructure", "blue=genome ID"], "normalization": "independent per-channel min-max at each frame; constant channels zero", "colors_are_mixed": True},
        "versions": {"python": platform.python_version(), "pillow": PIL.__version__, "numpy": np.__version__, "torch": torch.__version__, "matplotlib": matplotlib.__version__},
        "script_sha256": digest(__file__), "output_sha256": {name: digest(output / name) for name in asset_names[:3]},
    })
    save_json(output / "verification.json", {"status": "passed", "original_frame_hashes_verified": len(frame_sources),
        "decoded_gif_frames_verified": len(frames), "matching_completed_updates_on_both_sides": True, "loop_duration_ms": sum(gif_durations),
        "static_original_rgb_pixels_preserved": True, "endpoint_state_reproduces_original_rgb": True,
        "mask_counts_match_summary": counts, "mask_orientation_matches_rgb": True, "all_six_chart_values_verified": True,
        "input_hashes_unchanged": True, "simulation_rerun": False, "model_scores_recomputed": False,
        "intermediate_masks_created": False, "visual_review": "Record human-visible inspection in review.json."})
    print(f"Saved {len(asset_names)} outputs to {output}; loop=10.0 s; mask counts={counts}")


if __name__ == "__main__":
    main()
