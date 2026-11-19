#!/bin/sh
set -eu
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/../../../.." && pwd)
cd "$repo_dir"
python_bin=${CORALAI_PYTHON:-.venv-novelty/bin/python}
exec "$python_bin" -B scripts/compare_food_direction.py --output-dir "${1:?Pass a fresh output directory}"
