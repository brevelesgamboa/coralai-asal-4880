"""Compare fixed-image ASAL CLIP scores and warmed scoring time on CPU and GPU."""

import argparse
import csv
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import shlex
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scripts.clip_scoring import ClipScorer, load_clip_image, select_device

MODEL = "openai/clip-vit-base-patch32"
REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def software():
    versions = {"python": platform.python_version()}
    for name in ("jax", "jaxlib", "flax", "transformers", "numpy", "pillow",
                 "huggingface-hub", "torch", "jax-rocm60-plugin", "jax-rocm60-pjrt",
                 "jax-rocm7-plugin", "jax-rocm7-pjrt", "rocm", "rocm-sdk-core",
                 "rocm-sdk-libraries", "rocm-sdk-device-gfx1030", "rocm-sdk-device-gfx1031",
                 "jax-cuda12-plugin", "jax-cuda12-pjrt", "jax-cuda13-plugin"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            pass
    return versions


def hardware():
    result = {"platform": platform.platform(), "logical_cpus": os.cpu_count()}
    for command in (["lscpu"], ["lspci", "-nn"], ["rocminfo"],
                    ["rocm-smi", "--showproductname", "--showdriverversion"],
                    ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"]):
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=15)
            lines = completed.stdout.splitlines()
            if command[0] == "lscpu":
                lines = [line for line in lines if any(key in line for key in ("Model name:", "Architecture:", "CPU(s):"))]
            elif command[0] == "lspci":
                lines = [line for line in lines if any(key in line.lower() for key in ("vga", "3d controller", "display controller"))]
            elif command[0] == "rocminfo":
                lines = [line.strip() for line in lines if any(key in line for key in ("Name:", "Device Type:"))]
            result[command[0]] = {"exit_code": completed.returncode, "output": lines,
                                   "stderr": completed.stderr.strip()}
        except (OSError, subprocess.TimeoutExpired) as exc:
            result[command[0]] = {"unavailable": str(exc)}
    return result


def benchmark_worker(job, device):
    result = {"requested_device": device, "status": "unavailable"}
    try:
        selected = select_device(device)
    except ValueError as exc:
        result["error"] = str(exc)
        return result
    import jax

    scorer = ClipScorer(device, model_path=job["snapshot"])
    started = time.perf_counter()
    images = jax.device_put(np.load(job["images_npy"], allow_pickle=False), selected)
    jax.block_until_ready(images)
    transfer_seconds = time.perf_counter() - started
    started = time.perf_counter()
    text = scorer.text_embeddings(job["prompts"])
    text_seconds = time.perf_counter() - started
    warmup_times, times, samples = [], [], []
    for index in range(job["warmup"] + job["repeats"]):
        started = time.perf_counter()
        losses = scorer.losses(images, text)
        # Synchronize each completed pass, including all GPU operations, before stopping time.
        jax.block_until_ready(losses)
        elapsed = time.perf_counter() - started
        if index < job["warmup"]:
            warmup_times.append(elapsed)
        else:
            times.append(elapsed)
            samples.append(np.asarray(jax.device_get(losses)))
    if not np.isfinite(samples).all():
        raise ValueError("CLIP returned non-finite scores")
    median = statistics.median(times)
    result.update(
        status="ok", actual_device=str(selected), platform=selected.platform,
        device_kind=selected.device_kind, parameter_dtypes=scorer.parameter_dtypes,
        model_load_seconds=scorer.model_load_seconds,
        input_transfer_seconds=transfer_seconds, text_setup_seconds=text_seconds,
        warmup_seconds=warmup_times, scoring_seconds=times,
        median_scoring_seconds=median, median_ms_per_image=median * 1000 / len(images),
        repeat_max_abs_score_difference=float(np.max(np.abs(np.stack(samples) - samples[0]))),
        scores=samples[0].tolist(), software=software(),
    )
    return result


def ranks(scores):
    return np.argsort(np.argsort(scores, kind="stable"), kind="stable") + 1


def near_ties(scores, tolerance):
    return [any(i != j and abs(float(a) - float(b)) <= tolerance
                for j, b in enumerate(scores)) for i, a in enumerate(scores)]


def compare_scores(cpu, gpu, prompts, tolerance):
    difference = gpu - cpu
    rankings = []
    for index, prompt in enumerate(prompts):
        left, right = cpu[:, index], gpu[:, index]
        inversions = []
        for i in range(len(left)):
            for j in range(i + 1, len(left)):
                cpu_gap, gpu_gap = float(left[i] - left[j]), float(right[i] - right[j])
                if cpu_gap * gpu_gap < 0:
                    inversions.append({"image_indices": [i, j], "cpu_gap": cpu_gap,
                                       "gpu_gap": gpu_gap,
                                       "near_tie": max(abs(cpu_gap), abs(gpu_gap)) <= tolerance})
        rankings.append({"prompt": prompt, "strict_order_agrees": bool(np.array_equal(ranks(left), ranks(right))),
                         "agrees_outside_near_ties": not any(not pair["near_tie"] for pair in inversions),
                         "inversions": inversions,
                         "cpu_near_tie_images": near_ties(left, tolerance),
                         "gpu_near_tie_images": near_ties(right, tolerance)})
    return {"max_absolute_difference": float(np.max(np.abs(difference))),
            "mean_absolute_difference": float(np.mean(np.abs(difference))),
            "near_tie_tolerance": tolerance, "rankings": rankings}


def write_tables(output, images, prompts, devices, tolerance):
    cpu = np.asarray(devices["cpu"]["scores"]) if devices.get("cpu", {}).get("status") == "ok" else None
    gpu = np.asarray(devices["gpu"]["scores"]) if devices.get("gpu", {}).get("status") == "ok" else None
    fields = ("image", "prompt", "cpu_loss", "gpu_loss", "gpu_minus_cpu", "abs_difference",
              "cpu_rank", "gpu_rank", "cpu_near_tie", "gpu_near_tie")
    with (output / "scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for p, prompt in enumerate(prompts):
            for i, image in enumerate(images):
                row = {"image": image, "prompt": prompt}
                for name, scores in (("cpu", cpu), ("gpu", gpu)):
                    if scores is not None:
                        row.update({f"{name}_loss": float(scores[i, p]), f"{name}_rank": int(ranks(scores[:, p])[i]),
                                    f"{name}_near_tie": near_ties(scores[:, p], tolerance)[i]})
                if cpu is not None and gpu is not None:
                    delta = float(gpu[i, p] - cpu[i, p])
                    row.update(gpu_minus_cpu=delta, abs_difference=abs(delta))
                writer.writerow(row)
    with (output / "timings.csv").open("w", newline="") as stream:
        fields = ("device", "status", "model_load_seconds", "median_scoring_seconds", "median_ms_per_image")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, result in devices.items():
            writer.writerow({key: (name if key == "device" else result.get(key, "")) for key in fields})
    return compare_scores(cpu, gpu, prompts, tolerance) if cpu is not None and gpu is not None else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-glob")
    parser.add_argument("--prompts", nargs="+", default=["green energy patches", "branching colonies"])
    parser.add_argument("--devices", nargs="+", choices=("cpu", "gpu"), default=["cpu", "gpu"])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--near-tie-tolerance", type=float, default=1e-5)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker", choices=("cpu", "gpu"), help=argparse.SUPPRESS)
    parser.add_argument("--job", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        job = json.loads(args.job.read_text())
        try:
            result = benchmark_worker(job, args.worker)
        except Exception as exc:
            result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        save_json(job["worker_output"], result)
        return 0 if result["status"] == "ok" else 2
    if not args.images_glob or args.output_dir is None:
        parser.error("--images-glob and --output-dir are required")
    if args.warmup < 1 or args.repeats < 2:
        parser.error("Use at least one warm-up and two timed repeats")
    if (not np.isfinite(args.near_tie_tolerance) or args.near_tie_tolerance < 0
            or any(not prompt.strip() for prompt in args.prompts)):
        parser.error("Prompts must be nonempty and the near-tie tolerance finite and nonnegative")
    import glob
    from huggingface_hub import snapshot_download

    paths = sorted(Path(path) for path in glob.glob(args.images_glob))
    if not paths:
        parser.error(f"No images matched: {args.images_glob}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    snapshot = Path(snapshot_download(MODEL, revision=args.revision, local_files_only=args.local_files_only,
                                     allow_patterns=["*.json", "*.txt", "flax_model.msgpack"]))
    resolution_seconds = time.perf_counter() - started
    started = time.perf_counter()
    images = np.stack([load_clip_image(path, resize=True) for path in paths])
    preprocessing_seconds = time.perf_counter() - started
    manifest = [{"path": str(path), "sha256": sha256(path)} for path in paths]
    report = {
        "command": shlex.join(["python", str(Path(__file__).relative_to(ROOT)), *sys.argv[1:]]),
        "cwd": str(Path.cwd()), "images": manifest, "prompts": args.prompts,
        "model": MODEL, "revision": snapshot.name,
        "model_files_sha256": {path.name: sha256(path) for path in sorted(snapshot.iterdir()) if path.is_file()},
        "preprocessing": "PIL RGB; 224x224 LANCZOS; float32 / 255; ASAL CLIP channel normalization",
        "preprocessed_sha256": hashlib.sha256(images.tobytes()).hexdigest(),
        "precision": "float32 parameters; highest JAX matmul precision",
        "timing_scope": "Sequential image encoding plus ASAL losses for all prompts; resident inputs and cached text embeddings; synchronized after each full pass. Disk I/O, preprocessing, input transfer, and text setup excluded.",
        "model_loading_scope": "Cached model/processor construction and parameter placement, synchronized; dependency imports and snapshot resolution/download excluded.",
        "snapshot_resolution_seconds": resolution_seconds, "preprocessing_seconds": preprocessing_seconds,
        "warmup_passes": args.warmup, "timed_passes": args.repeats,
        "hardware": hardware(), "software": software(),
        "environment": {key: os.environ.get(key) for key in (
            "JAX_PLATFORMS", "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
            "HSA_OVERRIDE_GFX_VERSION", "XLA_PYTHON_CLIENT_PREALLOCATE")},
        "devices": {}, "comparison": None,
    }
    with tempfile.TemporaryDirectory(prefix="coralai-clip-comparison-") as temporary:
        temporary = Path(temporary)
        np.save(temporary / "images.npy", images, allow_pickle=False)
        for device in dict.fromkeys(args.devices):
            worker_output = output / f"{device}.json"
            job = {"images_npy": str(temporary / "images.npy"), "snapshot": str(snapshot),
                   "prompts": args.prompts, "warmup": args.warmup, "repeats": args.repeats,
                   "worker_output": str(worker_output)}
            save_json(temporary / "job.json", job)
            print(f"Benchmarking {device}: {len(paths)} images, {len(args.prompts)} prompts", flush=True)
            started = time.perf_counter()
            with (output / f"{device}.log").open("w") as log:
                completed = subprocess.run([sys.executable, str(Path(__file__)), "--worker", device,
                                            "--job", str(temporary / "job.json")], stdout=log,
                                           stderr=subprocess.STDOUT)
            result = json.loads(worker_output.read_text()) if worker_output.exists() else {
                "status": "error", "error": f"Worker exited {completed.returncode}; see {device}.log"}
            result["worker_wall_seconds"] = time.perf_counter() - started
            result["worker_exit_code"] = completed.returncode
            report["devices"][device] = result
            print(device, result["status"], result.get("error", result.get("median_scoring_seconds")), flush=True)
            save_json(output / "comparison.json", report)
    report["comparison"] = write_tables(output, [str(path) for path in paths], args.prompts,
                                         report["devices"], args.near_tie_tolerance)
    report["status"] = "complete" if all(r["status"] == "ok" for r in report["devices"].values()) else "partial"
    report["inputs_unchanged"] = all(sha256(item["path"]) == item["sha256"] for item in manifest)
    save_json(output / "comparison.json", report)
    print(f"Saved scores.csv, timings.csv, comparison.json in {output}", flush=True)
    return 0 if report["status"] == "complete" and report["inputs_unchanged"] else 2


if __name__ == "__main__":
    sys.exit(main())
