# Optional feeding cycles

Exploratory results from one matched starting world, pending review. Four
400×400, seed-42, 1,000-update conditions compare fixed/moving reserves and
steady/pulsed release. Food timing can reverse occupied-cell rankings between
feeding and paused phases. Occupancy means `genome >= 0`.

From the repository root, create the ignored Python 3.11 environment using the
existing dependency pins (CPU PyTorch is sufficient):

```sh
python3.11 -m venv optional/feeding_cycles/.local/venv
. optional/feeding_cycles/.local/venv/bin/activate
python -m pip install 'torch==2.7.1+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r configs/food_movement_requirements.txt
python optional/feeding_cycles/run.py --output-dir optional/feeding_cycles/results/seed42_reproduction
```

Install system `ffmpeg`/`ffprobe` for animation export. Repository-source imports
include vendored PyTorch-NEAT; no package installation of Coralai is needed.

Regenerate the figure and verify the compact evidence without simulations or raw
captures:

```sh
python optional/feeding_cycles/run.py --output-dir optional/feeding_cycles/results/seed42_20261127 --analyze-only
```

Add `--render-media` to regenerate animation/fallback from retained ignored
`local/` captures. Outputs include configs, CSV, summary/provenance, figure, MP4,
and fallback. Feeding updates 1–100 precede pauses 101–200; repeat five times.
