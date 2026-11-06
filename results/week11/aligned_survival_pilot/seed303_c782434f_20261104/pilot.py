"""Pinned two-run aligned pilot. Run stages from the repository root; see --help."""

import argparse
import ast
import csv
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import shlex
import shutil
import statistics
import subprocess
import sys
import tarfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
BASELINE = "c782434f33807704d52982c353854771fa15591c"
UPDATES = list(range(50, 1001, 50))
LABELS = {"default": "Default", "selected": "ASAL-selected"}
PROMPT = "branching colonies"
CLIP = ("openai/clip-vit-base-patch32", "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268")
SIGLIP = ("google/siglip-base-patch16-224", "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed")
RUNNER = "external/coralai/experiments/coral_dev/headless_coral.py"
SOURCE = ROOT / "results/week9/supervised_coralai_20261022/seed_comparison"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/coralai-aligned-matplotlib")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HOME", "/tmp/coralai-aligned-huggingface")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def root_relative(path):
    return os.path.relpath(Path(path).resolve(), ROOT)


def source_check(source_tree):
    require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_tree, text=True).strip() == BASELINE, "Wrong simulator commit")
    require(subprocess.run(["git", "diff", "--quiet", BASELINE], cwd=source_tree).returncode == 0, "Pinned source has tracked changes")
    original = read_json(SOURCE / "execution.json")
    for name, digest in original["source_sha256"].items():
        if name.startswith("external/coralai/"):
            require(sha(source_tree / name) == digest, f"Changed original simulator source: {name}")
    for name in ("scripts/clip_scoring.py", "foundation_models/clip.py", "asal_metrics.py"):
        require(sha(ROOT / name) == original["source_sha256"][name], f"Validated scoring source unavailable: {name}")
    return original


def validate_keys(rows):
    keys = [(row["configuration"], int(row["seed"]), int(row["completed_update"])) for row in rows]
    require(len(keys) == len(set(keys)) == 40 and set(keys) == {(label, 303, u) for label in LABELS for u in UPDATES}, "Missing/duplicate/wrong update identities")


def renderer(source_tree):
    """Compile only the two historical rendering functions, without starting a sim."""
    import numpy as np
    import torch

    module = ast.parse((source_tree / RUNNER).read_text())
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in ("scalar_channel", "make_frame")]
    require(len(functions) == 2, "Original renderer functions missing")
    namespace = {"np": np, "torch": torch}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source_tree / RUNNER), "exec"), namespace)
    return namespace["make_frame"]


def measure_pair(state_path, image_path, meta, make_frame):
    import numpy as np
    import torch
    from PIL import Image

    state = torch.load(state_path, map_location="cpu", weights_only=True)
    require(tuple(state.shape) == (1, 14, 100, 100) and torch.isfinite(state).all().item(), "Invalid state shape/values")
    genome = state[0, 13]
    require(torch.equal(genome, genome.floor()) and bool(((genome == -1) | (genome >= 0)).all()), "Invalid genome values")
    require(bool((genome < meta["n_genomes"]).all()), "Invalid genome references")
    count = int((genome >= 0).sum())
    substrate = SimpleNamespace(mem=state, w=100, h=100, windex={"energy": 0, "infra": 1, "genome": 13})
    with Image.open(image_path) as image:
        require(image.mode == "RGB" and image.size == (100, 100), "Invalid RGB image")
        require(np.array_equal(np.asarray(image), make_frame(substrate)), f"Image/state mismatch: {image_path}")
    require(0 <= count <= 10000, "Count outside denominator")
    return count, count / 10000


def record(args):
    source_tree, output = args.source_tree.resolve(), args.output_dir.resolve()
    original = source_check(source_tree)
    require(not (output / "recording_manifest.json").exists(), "Recording already exists; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "protected_artifacts.json").exists():
        protected = {root_relative(path): sha(path)
                     for base in (ROOT / "results/week9/supervised_coralai_20261022", ROOT / "results/week10/model_validation", ROOT / "results/week11/survival_evaluation")
                     for path in base.rglob("*") if path.is_file()}
        save_json(output / "protected_artifacts.json", protected)
    scoring_hashes = {}
    for name in ("scripts/clip_scoring.py", "foundation_models/clip.py", "asal_metrics.py"):
        path = output / "validated_scoring" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
        scoring_hashes[name] = sha(path)
    # The source clone is isolated on its own branch, with the original commit
    # unchanged. No modified runner or new simulation loop is needed.
    source_hashes = {}
    tracked = subprocess.check_output(["git", "ls-files", "external/coralai/coralai", "external/coralai/experiments/coral_dev"], cwd=source_tree, text=True).splitlines()
    tracked = [name for name in tracked if Path(name).suffix in (".py", ".config", ".json", ".toml")]
    for name in tracked:
        if (source_tree / name).is_file():
            source_hashes[name] = sha(source_tree / name)
    if not (output / "pinned_source.tar.gz").exists():
        with tarfile.open(output / "pinned_source.tar.gz", "w:gz") as archive:
            for name in tracked:
                if (source_tree / name).is_file():
                    archive.add(source_tree / name, arcname=name, recursive=False)
    make_frame = renderer(source_tree)
    rows, commands, configurations = [], [], {}
    for label in LABELS:
        folder = output / label
        folder.mkdir(exist_ok=False)
        base = read_json(SOURCE / f"{label}_config.json")
        require(base == read_json(SOURCE.parent / "search" / f"{label}_config.json"), "Original config differs from saved search selection")
        effective = dict(base, seed=303, checkpoint_interval=50, output_dir=str(folder / "run"))
        save_json(folder / "original_config.json", base)
        save_json(folder / "effective_config.json", effective)
        configurations[label] = {"original": base, "effective": effective,
                                 "changes": {key: {"original": base[key], "effective": effective[key]} for key in base if base[key] != effective[key]}}
        command = [sys.executable, str(source_tree / RUNNER), "--config", str(folder / "effective_config.json")]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join((str(source_tree / "external/coralai"), str(source_tree / "external/coralai/coralai/dependencies/PyTorch-NEAT")))
        env["TI_OFFLINE_CACHE_FILE_PATH"] = "/tmp/coralai-aligned-taichi-cache"
        commands.append({"argv": command, "cwd": str(source_tree), "PYTHONPATH": env["PYTHONPATH"]})
        save_json(folder / "command.json", commands[-1])
        print(f"Running {label}: pinned unchanged runner, seed 303, 1000 updates", flush=True)
        with (folder / "simulation.log").open("w") as log:
            subprocess.run(command, cwd=source_tree, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        metadata = read_json(folder / "run/meta.json")
        require(metadata["seed"] == 303 and metadata["start_step"] == 0 and metadata["completed_steps"] == 1000, "Run identity/length mismatch")
        require(read_json(folder / "run/config.json") == effective, "Effective configuration mismatch")
        checkpoints = sorted((folder / "run").glob("checkpoint_*/meta.json"))
        require([read_json(p)["step"] for p in checkpoints] == UPDATES, "Missing/duplicate/unexpected checkpoints")
        for update in UPDATES:
            checkpoint = folder / "run" / f"checkpoint_{update:07d}"
            state_path, image_path = checkpoint / "substrate.pt", folder / "run/screenshots" / f"step_{update:07d}.png"
            meta = read_json(checkpoint / "meta.json")
            require(meta["step"] == update and meta["git_hash"] == BASELINE and meta["shape"] == [100, 100], "Checkpoint provenance mismatch")
            count, fraction = measure_pair(state_path, image_path, meta, make_frame)
            rows.append({"configuration": label, "seed": 303, "completed_update": update,
                         "occupied_cells": count, "occupied_fraction": fraction, "grid_cells": 10000,
                         "image": str(image_path.relative_to(output)), "image_sha256": sha(image_path),
                         "checkpoint": str(state_path.relative_to(output)), "checkpoint_sha256": sha(state_path),
                         "checkpoint_meta_sha256": sha(checkpoint / "meta.json"), "population_sha256": sha(checkpoint / "population.pkl")})
    write_csv(output / "population.csv", rows)
    save_json(output / "recording_manifest.json", {"simulator_git_commit": BASELINE, "source_tree": str(source_tree),
        "isolation_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=source_tree, text=True).strip(),
        "source_sha256": source_hashes, "configurations": configurations, "prompt": PROMPT, "seed": 303,
        "validated_scoring_sha256": scoring_hashes,
        "scoring_provenance_note": "Week 9 scoring adapters were not committed at the recorded simulator HEAD; exact files recovered by their recorded SHA256 and frozen under validated_scoring. Simulator files match the pinned commit.",
        "scheduled_completed_updates": UPDATES, "frames": rows, "commands": commands,
        "recording_method": "Unchanged pinned runner's supported screenshot_interval=50 and checkpoint_interval=50; population derived from each saved matching checkpoint, not step_log.csv.",
        "cycle_point": "After physics, evolution, environment, radiation/cleanup, and timestep=step+1; screenshot and checkpoint have no intervening state mutation.",
        "occupancy_definition": "genome >= 0; occupied grid cells, not organism count or reproductive viability",
        "code_references": {RUNNER: "195–231", "external/coralai/experiments/coral_dev/experiment.py": "152–164", "external/coralai/experiments/coral_dev/physics.py": "34–37,53–57"},
        "original_environment": original["versions"]})
    environment = {"python": sys.executable, "python_version": platform.python_version(),
        "platform": platform.platform(), "packages": {name: version(name) for name in ("torch", "numpy", "pillow", "taichi", "neat-python", "jax", "jaxlib", "flax", "transformers", "huggingface-hub", "safetensors", "sentencepiece", "protobuf", "matplotlib")}}
    save_json(output / "environment.json", environment)
    (output / "requirements-used.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))


def fetch(args):
    from huggingface_hub import snapshot_download

    paths = {}
    for name, (model, revision), patterns in (("clip", CLIP, ["*.json", "merges.txt", "vocab.json", "flax_model.msgpack"]),
                                            ("siglip", SIGLIP, ["*.json", "spiece.model", "model.safetensors"])):
        path = Path(snapshot_download(model, revision=revision, cache_dir=str(args.cache_dir), allow_patterns=patterns,
                                      local_files_only=args.local_files_only))
        paths[name] = {"model": model, "revision": revision, "path": str(path),
                       "file_sha256": {p.name: sha(p) for p in path.iterdir() if p.is_file()}}
    require(paths["clip"]["file_sha256"]["flax_model.msgpack"] == read_json(SOURCE / "scorer.json")["weights_sha256"], "CLIP weight hash mismatch")
    historical_siglip = read_json(ROOT / "results/week10/model_validation/siglip_model.json")
    for name, digest in historical_siglip["file_sha256"].items():
        if name in paths["siglip"]["file_sha256"]:
            require(paths["siglip"]["file_sha256"][name] == digest, f"SigLIP file hash mismatch: {name}")
    save_json(args.output_dir / "models.json", paths)
    print("Pinned model downloads and historical weight hashes verified", flush=True)


def score(args):
    import numpy as np
    import torch
    import jax
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    output = args.output_dir.resolve()
    require(not (output / "per_update.csv").exists(), "New frames already scored; preserve results")
    source_check(args.source_tree)
    sys.path.insert(0, str(output / "validated_scoring"))
    from scripts.clip_scoring import ClipScorer, load_clip_image

    models = read_json(output / "models.json")
    rows = read_json(output / "recording_manifest.json")["frames"]
    for name in ("clip", "siglip"):
        for file, digest in models[name]["file_sha256"].items():
            require(sha(Path(models[name]["path"]) / file) == digest, "Model cache changed")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    clip = ClipScorer("cpu", model_path=models["clip"]["path"])
    text = clip.text_embeddings([PROMPT])
    processor = AutoProcessor.from_pretrained(models["siglip"]["path"], local_files_only=True, use_fast=False)
    siglip = AutoModel.from_pretrained(models["siglip"]["path"], local_files_only=True, torch_dtype=torch.float32,
                                     attn_implementation="eager", use_safetensors=True).to("cpu").eval()

    def clip_score(paths):
        losses = np.asarray(jax.device_get(clip.losses([load_clip_image(p, resize=True) for p in paths], text)))[:, 0]
        require(np.isfinite(losses).all(), "Nonfinite CLIP")
        return [-float(value) for value in losses]

    def siglip_score(paths):
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        inputs = processor(text=[PROMPT], images=images, padding="max_length", max_length=64,
                           truncation=True, return_tensors="pt")
        require(inputs["input_ids"].shape == (1, 64), "SigLIP padding differs")
        with torch.inference_mode():
            result = siglip(**inputs).logits_per_image[:, 0].cpu()
        require(torch.isfinite(result).all().item(), "Nonfinite SigLIP")
        return result.tolist()

    # Controls validate procedures only; none of these old scores enter pilot rows.
    controls = [r for r in read_csv(ROOT / "results/week10/model_validation/per_frame.csv")
                if int(r["seed"]) == 303 and int(r["step"]) == 50]
    control_paths = [ROOT / r["image"] for r in controls]
    control_clip, control_siglip = clip_score(control_paths), siglip_score(control_paths)
    checks = []
    for row, c, s in zip(controls, control_clip, control_siglip):
        require(abs(c - float(row["clip_alignment"])) < 1e-6, "Historical CLIP control mismatch")
        require(abs(s - float(row["siglip_logit"])) < 1e-4, "Historical SigLIP control mismatch")
        checks.append({"configuration": row["configuration"], "clip_abs_error": abs(c - float(row["clip_alignment"])), "siglip_abs_error": abs(s - float(row["siglip_logit"]))})
    for start in range(0, len(rows), 4):
        batch = rows[start:start + 4]
        paths = [output / row["image"] for row in batch]
        clip_values, siglip_values = clip_score(paths), siglip_score(paths)
        for row, c, s in zip(batch, clip_values, siglip_values):
            row.update(clip_alignment=c, clip_loss=-c, siglip_logit=s)
        print(f"Scored {start + len(batch)}/{len(rows)} new synchronized frames", flush=True)
    repeated_clip, repeated_siglip = clip_score([output / rows[0]["image"]]), siglip_score([output / rows[0]["image"]])
    require(abs(repeated_clip[0] - rows[0]["clip_alignment"]) < 1e-6 and abs(repeated_siglip[0] - rows[0]["siglip_logit"]) < 1e-4, "Repeat scoring check failed")
    write_csv(output / "per_update.csv", rows)
    save_json(output / "scoring.json", {"prompt": PROMPT, "models": models,
        "clip_preprocessing": "scripts.clip_scoring.load_clip_image(resize=True): RGB, 224x224 LANCZOS, float32 /255; original CLIP mean/std and normalized embedding cosine; loss = -alignment",
        "siglip_preprocessing": "Pinned AutoProcessor, use_fast=False; unchanged prompt, padding=max_length=64, truncation; CPU float32 eager AutoModel.eval; raw logits_per_image, no sigmoid",
        "clip_parameter_dtypes": clip.parameter_dtypes, "torch_threads": 4, "siglip_batch_size": 4,
        "historical_procedure_controls": checks, "new_frames_scored": len(rows), "historical_scores_reused_for_new_runs": False,
        "repeat_check": {"clip_abs_error": abs(repeated_clip[0] - rows[0]["clip_alignment"]), "siglip_abs_error": abs(repeated_siglip[0] - rows[0]["siglip_logit"])},
        "siglip_processor": processor.image_processor.to_dict()})


def numeric_rows(output):
    rows = read_csv(output / "per_update.csv")
    for row in rows:
        for name in ("seed", "completed_update", "occupied_cells", "grid_cells"):
            row[name] = int(row[name])
        for name in ("occupied_fraction", "clip_alignment", "clip_loss", "siglip_logit"):
            row[name] = float(row[name])
            require(math.isfinite(row[name]), f"Nonfinite {name}")
    return rows


def report(args):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw, ImageFont

    output = args.output_dir.resolve()
    require(not (output / "summary.csv").exists(), "Report exists; preserve completed outputs")
    rows = numeric_rows(output)
    summaries = []

    def correlation(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else None

    for label in LABELS:
        group = [row for row in rows if row["configuration"] == label]
        require([row["completed_update"] for row in group] == UPDATES, "Report update coverage differs")
        population = np.asarray([row["occupied_cells"] for row in group])
        clip = np.asarray([row["clip_alignment"] for row in group])
        siglip = np.asarray([row["siglip_logit"] for row in group])
        dp, dc, ds = np.diff(population), np.diff(clip), np.diff(siglip)
        summary = {"configuration": label, "seed": 303, "sampled_updates": len(group),
            "first_update": 50, "last_update": 1000, "occupied_at_50": int(population[0]),
            "occupied_at_1000": int(population[-1]), "occupied_fraction_at_1000": float(population[-1] / 10000),
            "population_change_50_to_1000": int(population[-1] - population[0]),
            "min_observed_population": int(population.min()), "max_observed_population": int(population.max()),
            "first_observed_zero_update": next((row["completed_update"] for row in group if row["occupied_cells"] == 0), None),
            "clip_mean": float(clip.astype(np.float32).mean()), "siglip_mean": statistics.mean(siglip.tolist()),
            "clip_change_50_to_1000": float(clip[-1] - clip[0]), "siglip_change_50_to_1000": float(siglip[-1] - siglip[0]),
            "population_declining_intervals": int((dp < 0).sum()),
            "population_down_clip_up_intervals": int(((dp < 0) & (dc > 0)).sum()),
            "population_down_siglip_up_intervals": int(((dp < 0) & (ds > 0)).sum()),
            "pearson_population_clip": correlation(population, clip), "pearson_population_siglip": correlation(population, siglip),
            "pearson_delta_population_delta_clip": correlation(dp, dc), "pearson_delta_population_delta_siglip": correlation(dp, ds)}
        summaries.append(summary)
    write_csv(output / "summary.csv", summaries)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True, layout="constrained")
    colors = {"default": "#236ca4", "selected": "#cf6518"}
    for label, display in LABELS.items():
        group = [row for row in rows if row["configuration"] == label]
        for axis, metric in zip(axes, ("occupied_fraction", "clip_alignment", "siglip_logit")):
            axis.plot(UPDATES, [row[metric] for row in group], color=colors[label], label=display, marker="o", markersize=3)
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(bottom=0)
    axes[0].set_ylabel("Occupied-cell fraction\n(count / 10,000)")
    axes[0].legend(frameon=False, ncols=2)
    axes[1].set_ylabel("CLIP cosine alignment\n(higher is better)")
    axes[2].set_ylabel("SigLIP image–text logit\n(higher is better)")
    axes[2].set_xlabel("Completed simulation updates")
    axes[2].set_xticks(range(0, 1001, 100))
    fig.suptitle('Following population and alignment over time\nSeed 303 · "branching colonies" · new paired pilot', fontsize=14)
    fig.supxlabel("Each point links an image, checkpoint, population count, and fresh model scores.\nLines join observed samples; intermediate states are not inferred.", fontsize=10)
    fig.savefig(output / "trajectories.png", dpi=180)
    fig.savefig(output / "trajectories.svg")
    plt.close(fig)
    frames = []

    def text(draw, value, y, size, center=560, bold=False):
        chosen = ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
        require(draw.textbbox((0, 0), value, font=chosen)[2] <= 1080, "GIF label too wide")
        draw.text((center, y), value, anchor="mt", font=chosen, fill="#16283c")

    for update in UPDATES:
        canvas = Image.new("RGB", (1120, 830), "white")
        draw = ImageDraw.Draw(canvas)
        text(draw, "Coralai aligned pilot  |  Seed 303", 22, 32, bold=True)
        text(draw, f"Completed update {update:,}", 76, 30)
        for label, x in zip(LABELS, (40, 580)):
            row = next(row for row in rows if row["configuration"] == label and row["completed_update"] == update)
            text(draw, LABELS[label], 132, 30, x + 250, True)
            with Image.open(output / row["image"]) as image:
                canvas.paste(image.resize((500, 500), Image.Resampling.NEAREST), (x, 180))
            text(draw, f"{row['occupied_cells']:,} occupied  |  {row['occupied_fraction']:.2%}", 710, 27, x + 250, True)
        text(draw, "Original RGB  |  Population measured from this exact saved state", 780, 22)
        frames.append(canvas)
    # Shared palette avoids frame-specific palette changes; original RGB PNGs
    # remain byte-unchanged. No new/interpolated simulation frames are created.
    atlas = Image.new("RGB", (200, 2100), "white")
    for i, update in enumerate(UPDATES):
        for label, x in zip(LABELS, (0, 100)):
            row = next(row for row in rows if row["configuration"] == label and row["completed_update"] == update)
            with Image.open(output / row["image"]) as image:
                atlas.paste(image, (x, i * 100))
    ImageDraw.Draw(atlas).rectangle((0, 2000, 99, 2100), fill="#16283c")
    palette = atlas.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    encoded = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    encoded[0].save(output / "aligned_comparison.gif", save_all=True, append_images=encoded[1:],
                    duration=500, loop=0, disposal=2, optimize=False)
    with Image.open(output / "aligned_comparison.gif") as gif:
        require(gif.n_frames == 20, "GIF missing updates")
        for i in range(20):
            gif.seek(i)
            require(gif.info["duration"] == 500 and np.array_equal(np.asarray(gif.convert("RGB")), np.asarray(encoded[i].convert("RGB"))), "GIF synchronization/content mismatch")
    manifest = read_json(output / "recording_manifest.json")
    for name, digest in manifest["source_sha256"].items():
        require(sha(args.source_tree / name) == digest, f"Pinned runtime source changed: {name}")
    for name, digest in manifest["validated_scoring_sha256"].items():
        require(sha(output / "validated_scoring" / name) == digest, f"Frozen scoring source changed: {name}")
    manifest.update(frames=rows, models=read_json(output / "models.json"),
                    scoring_procedures="scoring.json", environment="environment.json", clip_and_siglip_scored_fresh=True,
                    gif={"frame_updates": UPDATES, "duration_per_frame_ms": 500, "loop_seconds": 10,
                         "population_labels": "Exact checkpoint count for the displayed completed update", "conversion": "5× nearest-neighbor enlargement and shared 256-color GIF palette; original PNGs preserved"})
    save_json(output / "manifest.json", manifest)


def verify(args):
    import torch

    output = args.output_dir.resolve()
    source_check(args.source_tree)
    make_frame = renderer(args.source_tree)
    manifest = read_json(output / "recording_manifest.json")
    validate_keys(manifest["frames"])
    for name, digest in manifest["source_sha256"].items():
        require(sha(args.source_tree / name) == digest, f"Pinned runtime source changed: {name}")
    for name, digest in manifest["validated_scoring_sha256"].items():
        require(sha(output / "validated_scoring" / name) == digest, f"Frozen scoring source changed: {name}")
    if "pinned_source_archive_sha256" in manifest:
        require(sha(output / "pinned_source.tar.gz") == manifest["pinned_source_archive_sha256"], "Frozen source archive changed")
    rows = numeric_rows(output)
    validate_keys(rows)
    historical_equal = {label: 0 for label in LABELS}
    for row, recorded in zip(rows, manifest["frames"]):
        require(all(row[name] == recorded[name] for name in recorded), "Scored row no longer matches original image/checkpoint measurement")
        state, image = output / row["checkpoint"], output / row["image"]
        meta_path = state.parent / "meta.json"
        require(sha(state) == row["checkpoint_sha256"] and sha(image) == row["image_sha256"] and
                sha(meta_path) == row["checkpoint_meta_sha256"] and sha(state.parent / "population.pkl") == row["population_sha256"], "Recorded artifact changed")
        meta = read_json(meta_path)
        require(meta["step"] == row["completed_update"] and meta["git_hash"] == BASELINE, "Checkpoint update/version mismatch")
        count, fraction = measure_pair(state, image, meta, make_frame)
        require(count == row["occupied_cells"] and fraction == row["occupied_fraction"] and row["grid_cells"] == 10000, "Occupancy recount/denominator differs")
        require(row["clip_loss"] == -row["clip_alignment"], "CLIP score definition mismatch")
        old = SOURCE / "seed_303" / row["configuration"] / "run/screenshots" / f"step_{row['completed_update']:07d}.png"
        historical_equal[row["configuration"]] += int(sha(old) == sha(image))
    protected = read_json(output / "protected_artifacts.json")
    for path, digest in protected.items():
        require(sha(ROOT / path) == digest, f"Historical artifact changed: {path}")
    save_json(output / "verification.json", {"status": "passed", "matched_image_state_population_score_rows": len(rows),
        "scheduled_updates_per_run": 20, "checkpoint_occupancy_recounts_match": 40, "checkpoint_rgb_pixel_matches": 40,
        "unique_complete_run_seed_update_keys": True, "valid_grid_denominators": 10000, "all_scores_finite": True,
        "population_values_interpolated": False, "historical_scores_reused": False, "historical_artifacts_unchanged": len(protected),
        "historical_rgb_hash_matches_out_of_20": historical_equal,
        "scoring_controls": read_json(output / "scoring.json")["historical_procedure_controls"],
        "visual_inspection": "Review figure and representative decoded GIF frames; record separately in review.json."})
    print(f"Verified all {len(rows)} exact-update image/state/population/score tuples; {len(protected)} historical artifacts unchanged", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Stages: record uses the unmodified pinned runner and checkpoint_interval=50;
fetch downloads pinned models; score performs fresh inference; report builds
tables/figure/GIF; verify recounts all checkpoints and reproduces their images.
Record/fetch/score can run separately after environment setup. No historical
scores enter new results. All writes stay in a fresh --output-dir or model cache.
""")
    parser.add_argument("stage", choices=("record", "fetch", "score", "report", "verify"))
    parser.add_argument("--source-tree", type=Path, default=Path("/tmp/coralai-aligned-pilot-source"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/coralai-aligned-models"))
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    globals()[args.stage](args)


if __name__ == "__main__":
    main()
