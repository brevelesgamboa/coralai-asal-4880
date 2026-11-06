#!/usr/bin/env bash
set -euo pipefail

# From the repository root, using the isolated environment and source checkout:
# bash results/week11/aligned_survival_pilot/seed303_c782434f_20261104/reproduce.sh \
#   results/week11/aligned_survival_pilot/seed303_reproduction
pilot_output=${1:?Provide a fresh output directory}
pilot_environment=${2:-/tmp/coralai-aligned-pilot-env}
pilot_source=${3:-/tmp/coralai-aligned-pilot-source}
pilot_code=results/week11/aligned_survival_pilot/seed303_c782434f_20261104/pilot.py
if [[ -e "$pilot_output" ]]; then
    printf 'Output already exists; choose a fresh path: %s\n' "$pilot_output" >&2
    exit 2
fi
"$pilot_environment/bin/python" "$pilot_code" record --source-tree "$pilot_source" --output-dir "$pilot_output"
"$pilot_environment/bin/python" "$pilot_code" fetch --local-files-only --output-dir "$pilot_output"
"$pilot_environment/bin/python" "$pilot_code" score --source-tree "$pilot_source" --output-dir "$pilot_output"
"$pilot_environment/bin/python" "$pilot_code" report --source-tree "$pilot_source" --output-dir "$pilot_output"
"$pilot_environment/bin/python" "$pilot_code" verify --source-tree "$pilot_source" --output-dir "$pilot_output"
