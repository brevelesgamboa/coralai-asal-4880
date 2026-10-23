# scripts/score_coralai_progress.py
from pathlib import Path
import argparse
import glob
import json
import re
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scripts.clip_scoring import ClipScorer, load_clip_image as prepare_clip_image


def step_number(path: Path) -> int:
    """Extract the number from names such as step_0000100.png."""
    match = re.search(r"step_(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def load_clip_image(path: Path) -> np.ndarray:
    """Load a PNG and convert it to CLIP's expected 224 x 224 RGB input."""
    return prepare_clip_image(path, resize=True)


parser = argparse.ArgumentParser(
    description="Score Coralai headless screenshots against a text prompt."
)
parser.add_argument(
    "--images-glob",
    required=True,
    help='Screenshot pattern, e.g. "runs/.../screenshots/step_*.png"',
)
parser.add_argument("--prompt", default="branching colonies")
parser.add_argument(
    "--output",
    default="results/week7/coralai_clip_progress.json",
)
parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
args = parser.parse_args()

image_paths = sorted(
    (Path(path) for path in glob.glob(args.images_glob)),
    key=step_number,
)

if not image_paths:
    raise FileNotFoundError(f"No images matched: {args.images_glob}")

print(f"Loading CLIP on {args.device}. First run may download model weights.", flush=True)
try:
    scorer = ClipScorer(args.device)
except ValueError as exc:
    parser.exit(2, f"{exc}\n")

print(f"Embedding prompt: {args.prompt!r}", flush=True)
text_embedding = scorer.text_embeddings([args.prompt])

output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)

results = []
best_loss = float("inf")
initial_loss = None

for index, image_path in enumerate(image_paths, start=1):
    rgb = load_clip_image(image_path)
    loss = float(scorer.losses([rgb], text_embedding)[0, 0])
    if not np.isfinite(loss):
        raise ValueError(f"Invalid score for {image_path}")

    if initial_loss is None:
        initial_loss = loss

    is_best = loss < best_loss
    if is_best:
        best_loss = loss

    result = {
        "step": step_number(image_path),
        "image": str(image_path),
        "asal_target_loss": loss,
        "alignment": -loss,
        "change_from_initial_loss": loss - initial_loss,
        "best_loss_so_far": best_loss,
        "is_best_so_far": is_best,
    }
    results.append(result)

    improvement = initial_loss - loss
    status = "NEW BEST" if is_best else ""
    print(
        f"[{index:>4}/{len(image_paths)}] "
        f"step {result['step']:>7} | "
        f"loss {loss:.6f} | "
        f"improvement {improvement:+.6f} | {status}",
        flush=True,
    )

    # Save after every frame so progress survives an interrupted run.
    output.write_text(
        json.dumps(
            {
                "prompt": args.prompt,
                "model": "ASAL CLIP",
                "device": args.device,
                "interpretation": (
                    "Lower target loss / higher alignment indicates stronger "
                    "alignment with the prompt."
                ),
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "n_images_scored": len(results),
                "best_loss": best_loss,
                "progress": results,
            },
            indent=2,
        )
        + "\n"
    )

print(f"\nFinished. Best loss: {best_loss:.6f}")
print(f"Saved progress: {output}")
