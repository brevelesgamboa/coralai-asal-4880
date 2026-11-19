#!/bin/sh
set -eu
# Reuse the existing scientific Python environment.
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/../../../.." && pwd)
cd "$repo_dir"
# Essential packages and exact versions are recorded in requirements-used.txt.
# No CLIP/model download or ASAL search is needed.
python_bin=${CORALAI_PYTHON:-.venv-novelty/bin/python}
exec "$python_bin" -B scripts/compare_food_movement.py --output-dir "${1:?Pass a fresh output directory}"
