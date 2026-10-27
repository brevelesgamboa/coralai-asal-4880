"""Score the six saved Week 9 runs with SigLIP and compare their CLIP preferences."""

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
import time


ROOT = Path(__file__).resolve().parents[1]
CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
SIGLIP_MODEL = "google/siglip-base-patch16-224"
SIGLIP_REVISION = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
SEEDS = (101, 202, 303)
UPDATES = list(range(50, 1001, 50))
REFERENCES = [
    "https://huggingface.co/google/siglip-base-patch16-224",
    "https://huggingface.co/docs/transformers/v4.47.1/en/model_doc/siglip",
]


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path):
    return os.path.relpath(Path(path).resolve(), ROOT)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fixed_config(config):
    return {key: value for key, value in config.items() if key not in ("seed", "output_dir")}


def inspect_inputs(source):
    """Validate every frame and cached score before accepting any CLIP reuse."""
    import numpy as np
    from PIL import Image

    source = Path(source).resolve()
    comparison = source / "seed_comparison"
    inputs = {}

    def load(path):
        inputs[relative(path)] = sha256(path)
        return read_json(path)

    settings = load(comparison / "search_settings.json")
    scorer = load(comparison / "scorer.json")
    execution = load(comparison / "execution.json")
    old_pairs = load(comparison / "per_seed.json")
    old_summary = load(comparison / "summary.json")
    configs = {name: load(comparison / f"{name}_config.json") for name in ("default", "selected")}
    for name, config in configs.items():
        require(config == load(source / "search" / f"{name}_config.json"),
                f"Comparison {name} config differs from original search selection")
    require(settings["prompt"] == "branching colonies", "Unexpected Week 9 target prompt")
    require(settings["comparison_seeds"] == list(SEEDS), "Unexpected comparison seeds")
    require((settings["model"], settings["revision"]) == (CLIP_MODEL, CLIP_REVISION),
            "Unexpected original CLIP checkpoint")
    require((scorer["model"], scorer["revision"]) == (CLIP_MODEL, CLIP_REVISION),
            "Scorer metadata differs from search checkpoint")
    require(scorer["objective"] == "mean ASAL negative cosine over steps interval..steps; step 0 excluded",
            "Unexpected original temporal scoring objective")
    require(scorer["preprocessing"] == "RGB, 224x224 LANCZOS, float32 /255; existing CLIP normalization",
            "Unexpected original CLIP preprocessing")
    require(len(scorer["weights_sha256"]) == 64, "Missing original CLIP weights hash")
    require([row["seed"] for row in old_pairs] == list(SEEDS), "Saved paired scores have unexpected seeds")
    found = {(int(path.parent.parent.name.removeprefix("seed_")), path.parent.name): path
             for path in comparison.glob("seed_*/*/scores.json")}
    expected = {(seed, label) for seed in SEEDS for label in ("default", "selected")}
    require(set(found) == expected, f"Expected six runs; missing={expected - set(found)}, extra={set(found) - expected}")
    runs = []
    for seed in SEEDS:
        pair = next(row for row in old_pairs if row["seed"] == seed)
        for label in ("default", "selected"):
            score_path = found[seed, label]
            folder = score_path.parent
            scores = load(score_path)
            config = load(folder / "input_config.json")
            metadata = load(folder / "run/meta.json")
            effective = load(folder / "run/config.json")
            require(scores == pair[label], f"Cached CLIP scores disagree with paired record: {score_path}")
            require(fixed_config(config) == fixed_config(configs[label]), f"Changed {label} configuration: {folder}")
            require(fixed_config(effective) == fixed_config(config), f"Runner settings differ: {folder}")
            require(metadata["effective_config"] == effective, f"Runner metadata differs: {folder}")
            require(scores["seed"] == config["seed"] == metadata["seed"] == effective["seed"] == seed,
                    f"Seed mismatch: {folder}")
            require(config["steps"] == metadata["completed_steps"] == 1000 and config["screenshot_interval"] == 50,
                    f"Unexpected simulation length or sampling: {folder}")
            require(metadata["start_step"] == 0 and config["resume_from"] is None, f"Resumed input: {folder}")
            frames = scores["frame_scores"]
            require([frame["step"] for frame in frames] == UPDATES,
                    f"Missing, duplicate, or reordered sampled updates in {score_path}")
            verified = []
            for frame in frames:
                # Saved paths are absolute on the original machine. Resolve the same
                # screenshot within this checkout and verify its saved content hash.
                recorded = Path(frame["image"])
                require(recorded.name == f"step_{frame['step']:07d}.png", f"Unexpected image name: {recorded}")
                path = folder / "run/screenshots" / recorded.name
                digest = sha256(path)  # Missing images raise FileNotFoundError; none are skipped.
                require(digest == frame["sha256"], f"Image hash mismatch: {path}")
                require(math.isfinite(frame["loss"]) and math.isfinite(frame["alignment"]), f"Nonfinite CLIP score: {path}")
                require(frame["alignment"] == -frame["loss"], f"CLIP loss/alignment mismatch: {path}")
                with Image.open(path) as image:
                    require(image.size == (config["shape"], config["shape"]), f"Image/config shape mismatch: {path}")
                    image.verify()
                inputs[relative(path)] = digest
                verified.append({"step": frame["step"], "image": relative(path),
                                 "original_recorded_path": frame["image"], "sha256": digest,
                                 "clip_alignment": frame["alignment"], "clip_loss": frame["loss"]})
            # Week 9 aggregated float32 losses. Retain that exact convention.
            mean = -float(np.mean(np.asarray([frame["loss"] for frame in frames], dtype=np.float32)))
            require(mean == scores["mean_alignment"] == -scores["mean_loss"], f"Incorrect CLIP mean: {score_path}")
            runs.append({"seed": seed, "configuration": label, "config": config,
                         "config_file": relative(folder / "input_config.json"),
                         "scores_file": relative(score_path), "clip_mean": mean, "frames": verified})
        require(math.isclose(pair["difference"], pair["selected"]["mean_alignment"] - pair["default"]["mean_alignment"], abs_tol=1e-12),
                f"Incorrect original paired difference: seed {seed}")
    require(math.isclose(statistics.mean(row["difference"] for row in old_pairs),
                         old_summary["paired_difference_mean"], abs_tol=1e-12), "Original summary differs from seed scores")
    return {"source": relative(source), "prompt": settings["prompt"], "seeds": list(SEEDS),
            "sampled_updates": UPDATES, "step_zero_excluded": True, "runs": runs,
            "original_clip_scorer": scorer, "original_clip_execution": execution,
            "original_configurations": configs, "input_sha256": inputs,
            "clip_reuse": "All 120 image hashes, configs, seeds, prompt, checkpoint, cached scores, and temporal aggregates verified; CLIP not recomputed."}


def preference(difference, tolerance):
    if abs(difference) <= tolerance:
        return "tie"
    return "selected" if difference > 0 else "default"


def comparisons(manifest, logits, clip_atol, siglip_atol):
    rows = []
    for seed in SEEDS:
        row = {"seed": seed}
        for label in ("default", "selected"):
            run = next(run for run in manifest["runs"] if run["seed"] == seed and run["configuration"] == label)
            values = logits[seed, label]
            require(len(values) == len(UPDATES) and all(math.isfinite(value) for value in values),
                    f"Missing or nonfinite SigLIP scores: seed {seed}, {label}")
            row[f"clip_{label}_mean"] = run["clip_mean"]
            row[f"siglip_{label}_mean"] = statistics.mean(values)
        for model, tolerance in (("clip", clip_atol), ("siglip", siglip_atol)):
            difference = row[f"{model}_selected_mean"] - row[f"{model}_default_mean"]
            row[f"{model}_difference"] = difference
            row[f"{model}_preferred"] = preference(difference, tolerance)
            row[f"{model}_near_tie"] = abs(difference) <= tolerance
        row["models_agree"] = row["clip_preferred"] == row["siglip_preferred"]
        rows.append(row)
    return rows


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_differences(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), layout="constrained")
    for axis, model, label in zip(axes, ("clip", "siglip"), ("Cosine alignment", "Image–text logit")):
        values = [row[f"{model}_difference"] for row in rows]
        axis.bar([str(row["seed"]) for row in rows], values, color="#326d92", width=0.55)
        axis.axhline(0, color="#333333", linewidth=0.8)
        axis.set_title("CLIP" if model == "clip" else "SigLIP")
        axis.set_xlabel("Simulation seed")
        axis.set_ylabel(f"Selected − default\n(temporal mean {label.lower()})")
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    fig.suptitle('"branching colonies" · separate score scales\nPositive favors selected; negative favors default', fontsize=11)
    fig.savefig(output / "per_seed_differences.png", dpi=220)
    fig.savefig(output / "per_seed_differences.svg")
    plt.close(fig)


def evaluate(args, manifest):
    import torch
    from PIL import Image
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoProcessor

    output = args.output_dir.resolve()
    require(not any((output / name).exists() for name in ("manifest.json", "per_frame.csv", "summary.json")),
            f"Evaluation outputs already exist: {output}. Choose a fresh --output-dir.")
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "manifest.json", manifest)
    packages = {name: version(name) for name in ("torch", "transformers", "sentencepiece", "protobuf", "numpy", "pillow", "matplotlib", "huggingface-hub", "tokenizers", "safetensors")}
    (output / "requirements-used.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    command = shlex.join([sys.executable, *sys.argv])
    save_json(output / "execution.json", {"command": command, "cwd": str(Path.cwd()),
              "python": sys.executable, "python_version": platform.python_version(), "platform": platform.platform(),
              "cpu_threads": args.threads, "batch_size": args.batch_size, "device": "cpu", "dtype": "float32",
              "packages": packages, "script_sha256": sha256(__file__),
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})
    started = time.perf_counter()
    snapshot = Path(snapshot_download(SIGLIP_MODEL, revision=SIGLIP_REVISION,
                    cache_dir=str(args.cache_dir), local_files_only=args.local_files_only,
                    allow_patterns=["*.json", "spiece.model", "model.safetensors"]))
    download_seconds = time.perf_counter() - started
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True, use_fast=False)
    model = AutoModel.from_pretrained(snapshot, local_files_only=True, torch_dtype=torch.float32,
                                      attn_implementation="eager", use_safetensors=True).to("cpu").eval()
    model_load_seconds = time.perf_counter() - started
    require(all(parameter.device.type == "cpu" and parameter.dtype == torch.float32 for parameter in model.parameters()),
            "Expected CPU float32 model weights")
    model_info = {"model": SIGLIP_MODEL, "revision": SIGLIP_REVISION, "snapshot": str(snapshot.resolve()),
                  "file_sha256": {path.name: sha256(path) for path in snapshot.iterdir() if path.is_file()},
                  "processor_class": type(processor).__name__, "image_processor": processor.image_processor.to_dict(),
                  "tokenizer_class": type(processor.tokenizer).__name__, "text_padding": "max_length",
                  "text_max_length": model.config.text_config.max_position_embeddings,
                  "text": manifest["prompt"], "prompt_template": None,
                  "score": "outputs.logits_per_image[:, 0]; learned-scale cosine plus learned bias; higher is better",
                  "aggregation": "arithmetic mean of the 20 raw image-text logits for updates 50..1000",
                  "attention": "eager", "training": False, "autograd": False, "sigmoid_applied": False,
                  "interpretation": "Uncalibrated model scores; CLIP and SigLIP units are kept separate.",
                  "references": REFERENCES, "download_or_cache_seconds": download_seconds,
                  "model_load_seconds": model_load_seconds}
    save_json(output / "siglip_model.json", model_info)

    def score_frames(frames):
        images = [Image.open(ROOT / frame["image"]).convert("RGB") for frame in frames]
        try:
            inputs = processor(text=[manifest["prompt"]], images=images, padding="max_length",
                               max_length=model.config.text_config.max_position_embeddings,
                               truncation=True, return_tensors="pt")
        finally:
            for image in images:
                image.close()
        require(inputs["input_ids"].shape == (1, 64), "Unexpected SigLIP token padding")
        with torch.inference_mode():
            values = model(**inputs).logits_per_image[:, 0].cpu()
        require(torch.isfinite(values).all().item(), "SigLIP returned nonfinite logits")
        return values.tolist()

    logits, frame_rows, timings = {}, [], []
    first_frames, first_values = None, None
    for run in manifest["runs"]:
        values = []
        started = time.perf_counter()
        for start in range(0, len(run["frames"]), args.batch_size):
            batch = run["frames"][start:start + args.batch_size]
            batch_values = score_frames(batch)
            if first_frames is None:
                first_frames, first_values = batch, batch_values
            values.extend(batch_values)
        elapsed = time.perf_counter() - started
        logits[run["seed"], run["configuration"]] = values
        for frame, value in zip(run["frames"], values):
            frame_rows.append({"seed": run["seed"], "configuration": run["configuration"],
                               "step": frame["step"], "image": frame["image"], "image_sha256": frame["sha256"],
                               "clip_alignment": frame["clip_alignment"], "clip_loss": frame["clip_loss"], "siglip_logit": value})
        write_csv(output / "per_frame.csv", frame_rows)
        timings.append({"seed": run["seed"], "configuration": run["configuration"], "seconds": elapsed})
        print(f"seed {run['seed']} {run['configuration']}: SigLIP mean logit {statistics.mean(values):.6f}; {elapsed:.2f}s", flush=True)
    repeated = score_frames(first_frames)
    repeat_difference = max(abs(a - b) for a, b in zip(first_values, repeated))
    single_difference = abs(score_frames(first_frames[:1])[0] - first_values[0])
    rows = comparisons(manifest, logits, args.clip_tie_atol, args.siglip_tie_atol)
    write_csv(output / "per_seed.csv", rows)
    plot_differences(rows, output)
    summary = {"prompt": manifest["prompt"], "n_runs": len(manifest["runs"]), "n_frames": len(frame_rows),
               "seeds": list(SEEDS), "sampled_updates": UPDATES, "clip_scores_reused": True,
               "per_seed": rows, "agreement_count": sum(row["models_agree"] for row in rows),
               "differences": "selected minus default, within each model's own units",
               "tie_tolerances": {"clip_alignment": args.clip_tie_atol, "siglip_logit": args.siglip_tie_atol,
                                  "meaning": "Absolute numerical near-tie flags; these are not significance thresholds."},
               "repeat_check": {"frames": len(first_frames), "same_batch_max_abs_logit_difference": repeat_difference,
                                "single_vs_batch_abs_logit_difference": single_difference},
               "timings": {"per_run_inference_and_preprocessing": timings, "model_load_seconds": model_load_seconds},
               "model_statistics": {},
               "limitations": ["Three seeds from one fixed prompt and previously selected configurations.",
                               "Temporal frames within a run are correlated; no independent-frame significance claims.",
                               "Model agreement is supporting evidence; neither model establishes visual or biological ground truth.",
                               "Preprocessing follows each model's own processor; score magnitudes have different units."],
               "reproduction_command": command}
    for name in ("clip", "siglip"):
        values = [row[f"{name}_difference"] for row in rows]
        summary["model_statistics"][name] = {"mean_paired_difference": statistics.mean(values),
            "sample_std_paired_difference": statistics.stdev(values),
            "default_mean": statistics.mean(row[f"{name}_default_mean"] for row in rows),
            "selected_mean": statistics.mean(row[f"{name}_selected_mean"] for row in rows)}
    save_json(output / "summary.json", summary)
    print(json.dumps({"agreement_count": summary["agreement_count"], "per_seed": rows}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Example (from the repository root, in an environment with Torch, Transformers,
SentencePiece, NumPy, Pillow, and Matplotlib):
  python scripts/validate_saved_clip_siglip.py --check-inputs
  python scripts/validate_saved_clip_siglip.py --local-files-only \\
    --output-dir results/week10/model_validation_repro

Omit --local-files-only for the first model download. The pinned SigLIP processor
receives the unchanged raw prompt with padding=max_length (64 tokens). Its image
processor resizes/rescales/normalizes the saved RGB image. Raw logits_per_image
are averaged over the same 20 updates as CLIP; sigmoid is never applied.
CLIP is reused only after exact image hashes, metadata, and aggregates pass.
Missing frames and inconsistent scores stop evaluation. New output directories
preserve existing results. No simulations or search are run.
Outputs: manifest.json, per_frame.csv, per_seed.csv, summary.json,
per_seed_differences.png/.svg, siglip_model.json, execution.json,
requirements-used.txt. Input validation needs only NumPy and Pillow.
Official inference example: https://huggingface.co/google/siglip-base-patch16-224
""")
    parser.add_argument("--source", type=Path, default=ROOT / "results/week9/supervised_coralai_20261022", help="Saved Week 9 experiment root")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/week10/model_validation", help="New evaluation output directory")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/model_cache/siglip-validation", help="Hugging Face model cache")
    parser.add_argument("--local-files-only", action="store_true", help="Use only the cached pinned SigLIP files")
    parser.add_argument("--check-inputs", action="store_true", help="Verify all six saved runs and 120 cached CLIP scores, then exit")
    parser.add_argument("--batch-size", type=int, default=4, help="CPU inference batch size (default: 4)")
    parser.add_argument("--threads", type=int, default=4, help="Torch CPU threads (default: 4)")
    parser.add_argument("--clip-tie-atol", type=float, default=1e-6, help="Absolute near-tie tolerance in CLIP cosine units")
    parser.add_argument("--siglip-tie-atol", type=float, default=1e-4, help="Absolute near-tie tolerance in SigLIP logit units")
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1 or any(not math.isfinite(t) or t < 0 for t in (args.clip_tie_atol, args.siglip_tie_atol)):
        parser.error("Batch size and threads must be positive; tie tolerances must be finite and nonnegative")
    try:
        manifest = inspect_inputs(args.source)
        print(f"Verified {len(manifest['runs'])} runs and {sum(len(run['frames']) for run in manifest['runs'])} images; prompt={manifest['prompt']!r}", flush=True)
        if not args.check_inputs:
            evaluate(args, manifest)
    except (ValueError, FileNotFoundError, KeyError, OSError) as error:
        parser.exit(2, f"Evaluation stopped: {error}\n")


if __name__ == "__main__":
    main()
