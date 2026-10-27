# Coralai + ASAL

CSC 4880 project connecting Coralai ecosystem simulations to ASAL supervised target search.

## Current progress
- Added Coralai source under external/coralai/.
- Organized existing experiment results under results/.
- Collected project documentation under docs/.
- Defined the [Coralai pilot configuration and result contract](docs/coralai_pilot_contract.md), with JSON schemas, a default configuration, and an illustrative result.

## Pilot format checks

The shared format can be validated without running Coralai or downloading CLIP.
Use a Python environment with the existing `jsonschema==4.23.0` dependency:

```sh
python scripts/validate_coralai_contract.py config configs/coralai_pilot.default.json
python scripts/validate_coralai_contract.py result examples/coralai_pilot_result.json
python -m unittest discover -s tests -p 'test_coralai_contract.py'
```

## Verified headless CPU run

From the repository root, using Python 3.11 (the version checked here):

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c results/week9/headless_runner_verification_20261022/requirements-used.txt external/coralai external/coralai/coralai/dependencies/PyTorch-NEAT
python external/coralai/experiments/coral_dev/headless_coral.py \
  --config configs/headless_cpu_repro.json --seed 42 \
  --output-dir data/raw/headless_repro/run_1
```

Use a fresh `--output-dir` on every invocation; repeat with `run_2` to compare.
The config preserves the existing sanity check: 51 steps, 50×50 grid, CPU, seed 42.
Each run saves `config.json`, `meta.json`, screenshots, and checkpoints at updates 50 and 51.
Verification outputs: `data/raw/headless_runner_verification_20261022/{run_a,run_b,isolated_run}/`.
Commands, versions, hashes, and comparisons: [verification records](results/week9/headless_runner_verification_20261022/).
All four frames matched across these runs; repeat tensors differed by at most `7.1e-6`.

See [pilot validation](docs/coralai_pilot_validation.md) for saved-artifact checks,
reproduction commands, and [the recorded checks](results/week8/validation/README.md).

## Next steps
- Compute Shannon and discretized entropy on saved runs.
- Connect Coralai simulation frames to ASAL scoring.

## Upstream projects
- ASAL: https://github.com/SakanaAI/asal
- Coralai: https://github.com/aidanbx/coralai
