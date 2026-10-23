from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.clip_scoring import ClipScorer, load_clip_image

parser = argparse.ArgumentParser()
parser.add_argument("--image", default="results/week7/coralai_frame.png")
parser.add_argument("--prompt", default="branching colonies")
parser.add_argument(
    "--output", default="results/week7/coralai_clip_score.json"
)
parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
args = parser.parse_args()

rgb = load_clip_image(args.image)

print(f"Loading CLIP on {args.device}. First run may download model weights.", flush=True)
try:
    scorer = ClipScorer(args.device)
except ValueError as exc:
    parser.exit(2, f"{exc}\n")

print("Scoring the image...", flush=True)
loss = scorer.score(rgb, args.prompt)

result = {
    "image": args.image,
    "prompt": args.prompt,
    "model": "ASAL CLIP",
    "asal_target_loss": loss,
    "alignment": -loss,
    "interpretation": "Lower target loss means stronger prompt alignment.",
    "device": args.device,
}

output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2) + "\n")

print(json.dumps(result, indent=2))
print("Saved:", output)
