"""Build a 10-second GIF from saved seed-202 images and frame-level scores.

Requires Pillow. Run from the repository root; use --output-dir for a fresh copy.
No simulations or model inference. GIF's 256-color palettes approximate RGB colors;
nearest-neighbor enlargement and disabled dithering preserve the pixel geometry.
"""

import argparse
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import shlex
import sys

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageStat, __version__

ROOT = Path(__file__).resolve().parents[5]
BASE = ROOT / "results/week10/model_validation"
UPDATES = list(range(50, 1001, 50))
BOXES = {"default": (60, 145, 460, 545), "selected": (660, 145, 1060, 545)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preference(default, selected):
    a, b = Decimal(default), Decimal(selected)
    if not a.is_finite() or not b.is_finite():
        raise ValueError("Scores must be finite")
    return "Exact tie" if a == b else "Search-selected" if b > a else "Default"


def font(size, bold=False):
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for folder in ("/usr/share/fonts/dejavu-sans-fonts", "/usr/share/fonts/truetype/dejavu"):
        path = Path(folder) / filename
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.truetype(filename, size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent,
                        help="New output directory; existing GIFs are preserved")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    gif_path = out / "seed_202_model_preferences.gif"
    names = [gif_path.name, "animation.json", *[f"preview_update_{step:04d}.png" for step in (50, 550, 1000)]]
    if any((out / name).exists() for name in names):
        parser.error("Outputs already exist; choose a fresh --output-dir")
    before = {str(p.relative_to(ROOT)): sha(p) for p in BASE.rglob("*")
              if p.is_file() and "animations" not in p.relative_to(BASE).parts}
    manifest = json.loads((BASE / "manifest.json").read_text())
    assert manifest["sampled_updates"] == UPDATES
    with (BASE / "per_frame.csv").open() as stream:
        rows = [r for r in csv.DictReader(stream) if int(r["seed"]) == 202]
    lookup = {(r["configuration"], int(r["step"])): r for r in rows}
    assert len(rows) == len(lookup) == 40
    assert set(lookup) == {(label, step) for label in BOXES for step in UPDATES}
    frames, records = [], []
    for step in UPDATES:
        canvas = Image.new("RGB", (1120, 800), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((32, 18), f"Seed 202  |  Update {step:,} / 1,000", font=font(35, True), fill="black")
        draw.text((32, 66), f'Target: "{manifest["prompt"]}"', font=font(24), fill="#444444")
        originals, record = [], {"seed": 202, "update": step, "images": {}, "models": {}}
        for label, box in BOXES.items():
            row = lookup[label, step]
            run = next(r for r in manifest["runs"] if r["seed"] == 202 and r["configuration"] == label)
            source = next(f for f in run["frames"] if f["step"] == step)
            assert source["image"] == row["image"] and source["sha256"] == row["image_sha256"]
            assert source["clip_alignment"] == float(row["clip_alignment"])
            assert sha(ROOT / row["image"]) == source["sha256"]
            with Image.open(ROOT / row["image"]) as image:
                image = image.convert("RGB")
                assert image.size == (100, 100)
                originals.append(image.copy())
                tile = image.resize((400, 400), Image.Resampling.NEAREST)
                canvas.paste(tile, box[:2])
                assert canvas.crop(box).tobytes() == tile.tobytes()
            title = "Default" if label == "default" else "Search-selected"
            draw.text((box[0], 105), title, font=font(28, True), fill="black")
            record["images"][label] = {"path": row["image"], "sha256": source["sha256"]}
        for model, key, top, unit in (("CLIP", "clip_alignment", 565, "cosine alignment"),
                                      ("SigLIP", "siglip_logit", 660, "image-text logit")):
            a, b = (lookup[label, step][key] for label in BOXES)
            choice = preference(a, b)
            delta = Decimal(b) - Decimal(a)
            draw.rounded_rectangle((32, top, 1088, top + 82), radius=8, fill="#eeeeee")
            draw.text((48, top + 6), f"{model} | Current-frame preference: {choice}", font=font(26, True), fill="black")
            draw.text((48, top + 46), f"Default {float(a):.6f}   |   Selected {float(b):.6f}   |   Delta {float(delta):+.6f}",
                      font=font(23), fill="#333333")
            record["models"][model] = {"unit": unit, "default": a, "selected": b,
                                       "selected_minus_default": str(delta), "preference": choice}
        draw.text((32, 767), "Saved frames only. Higher scores are better within each model; scales are separate.",
                  font=font(21), fill="#444444")
        # Quantize the image pair rather than letting white margins dominate the palette.
        colors = Image.new("RGB", (200, 100))
        for index, image in enumerate(originals):
            colors.paste(image, (index * 100, 0))
        palette = colors.quantize(colors=224, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        gray = [channel for index in range(32) for channel in (round(index * 255 / 31),) * 3]
        palette.putpalette(palette.getpalette()[:224 * 3] + gray)
        encoded = canvas.quantize(palette=palette, dither=Image.Dither.NONE)
        record["palette_mean_absolute_rgb_error_0_to_255"] = {
            label: sum(ImageStat.Stat(ImageChops.difference(original, encoded.convert("RGB").crop(BOXES[label]).resize(
                (100, 100), Image.Resampling.NEAREST))).mean) / 3
            for label, original in zip(BOXES, originals)}
        frames.append(encoded)
        records.append(record)
    durations = [950, *([450] * 18), 950]
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, disposal=2, optimize=False)
    with Image.open(gif_path) as decoded:
        assert decoded.n_frames == 20 and decoded.info["loop"] == 0 and decoded.size == (1120, 800)
        actual_durations = []
        for index, step in enumerate(UPDATES):
            decoded.seek(index)
            rgb = decoded.convert("RGB")
            assert rgb.tobytes() == frames[index].convert("RGB").tobytes()
            actual_durations.append(decoded.info["duration"])
            if step in (50, 550, 1000):
                rgb.save(out / f"preview_update_{step:04d}.png")
    assert actual_durations == durations and sum(durations) == 10000
    assert all(sha(ROOT / name) == digest for name, digest in before.items())
    metadata = {"prompt": manifest["prompt"], "seed": 202, "updates": UPDATES,
                "size": [1120, 800], "duration_ms": sum(durations), "frame_durations_ms": durations,
                "loop": "infinite", "preference_rule": "Compare the exact saved decimal scores for each frame; equal values display Exact tie.",
                "image_handling": "4x nearest-neighbor enlargement; no crop, color grading, dithering, or interpolated simulation frames. GIF's 256-color palettes approximate RGB colors.",
                "pillow_version": __version__, "frames": records, "existing_files_sha256": before,
                "command": shlex.join([sys.executable, *sys.argv]), "gif_sha256": sha(gif_path),
                "checks": {"decoded_frames_match_rendered_frames": True, "all_inputs_hash_verified": True,
                           "existing_files_unchanged": True, "exact_tie_rule_tested": preference("1", "1.0") == "Exact tie",
                           "inspection_updates": [50, 550, 1000]}}
    (out / "animation.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(f"{gif_path}\n20 frames, 10 seconds, 1120 x 800, infinite loop")


if __name__ == "__main__":
    main()
