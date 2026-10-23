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
The clean source export/fresh environment check was performed by Codex; independent teammate reproduction is pending.

## CLIP device comparison

Use the existing ASAL environment with JAX, Flax, Transformers, NumPy, Pillow, and einops.
Both scoring scripts accept `--device cpu` or `--device gpu`; CPU remains the default.
From the repository root:

```sh
python scripts/compare_clip_devices.py \
  --images-glob 'external/coralai/runs/coral_dev_headless_20261021_143009/screenshots/step_*.png' \
  --prompts 'green energy patches' 'branching colonies' \
  --warmup 2 --repeats 5 --local-files-only \
  --output-dir results/week9/clip_device_comparison_repro
```

Use a new output directory each time. Omit `--local-files-only` if model weights need downloading.
The comparison pins the existing CLIP revision and shares identical RGB/LANCZOS preprocessing and float32 weights.
Outputs: `scores.csv`, `timings.csv`, per-device JSON, and `comparison.json` with hashes, hardware, versions, and rank/tie checks.
Scoring times exclude loading, disk/preprocessing, transfers, and cached prompt setup; every pass synchronizes completion.
Recorded run: `results/week9/clip_device_comparison_20261022/` (21 existing frames, two prompts).
CPU-only reference: 4.36 s model loading, 3.45 s median scoring pass. That original ASAL environment exposes CPU only; the separate GPU environment below has now been verified.
An unavailable device is reported without CPU fallback; the command saves available results and exits 2 for a partial comparison.

## AMD RX 6700 XT scoring setup

Create a separate Python 3.11 environment; keep the original ASAL environment unchanged.

```sh
python3.11 -m venv data/venvs/asal-rocm-gfx1031
. data/venvs/asal-rocm-gfx1031/bin/activate
python -m pip install --extra-index-url https://repo.amd.com/rocm/whl-multi-arch/ \
  -r results/week9/clip_gpu_gfx1031_20261022/requirements-gpu.txt
HSA_OVERRIDE_GFX_VERSION=10.3.0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python scripts/compare_clip_devices.py \
  --images-glob 'external/coralai/runs/coral_dev_headless_20261021_143009/screenshots/step_*.png' \
  --prompts 'green energy patches' 'branching colonies' --warmup 2 --repeats 5 --local-files-only \
  --output-dir results/week9/clip_gpu_repro
```

The existing tested environment is `data/venvs/asal-rocm-gfx1031-20261022/`; activate it to skip installation.
This [AMD TheRock](https://github.com/ROCm/TheRock/blob/main/RELEASES.md) stack uses JAX 0.9.1, ROCm 7.14.1, and Flax 0.12.0 for scoring only.
Native gfx1031 is rejected by this PJRT build; the command uses process-local gfx1030 compatibility on the actual RX 6700 XT.
Results: `results/week9/clip_gpu_gfx1031_20261022/comparison/` (`scores.csv`, `timings.csv`, `comparison.json`). Use a new reproduction output directory each time.
Verified GPU arithmetic and CLIP; maximum CPU/GPU score difference `2.54e-7`, identical rankings. Median 21-frame/two-prompt passes: CPU 3.35 s, GPU 5.84 s; loading: 6.86 s / 30.24 s.

See [pilot validation](docs/coralai_pilot_validation.md) for saved-artifact checks,
reproduction commands, and [the recorded checks](results/week8/validation/README.md).

## Supervised Coralai search (CPU)

Activate the validated CPU ASAL environment (`. /home/sparky/.venvs/coralai-asal/bin/activate` on this machine).
Requires its existing evosax 0.1.6 and Coralai dependencies; the adapter adds vendored import paths without installing packages.
If CLIP is not already cached, download the pinned weights once:
`python -c 'from huggingface_hub import snapshot_download; snapshot_download("openai/clip-vit-base-patch32", revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268")'`
From the repository root, use fresh output directories:

```sh
JAX_PLATFORMS=cpu python main_opt.py --substrate coralai \
  --coralai_config configs/coralai_supervised.default.json \
  --coralai_search_config configs/coralai_supervised.search.json \
  --pop_size 4 --n_iters 2 --sigma 0.25 --seed 0 \
  --save_dir results/week9/supervised_coralai_repro/search
JAX_PLATFORMS=cpu python scripts/coralai_search.py compare \
  --experiment results/week9/supervised_coralai_repro/search \
  --output-dir results/week9/supervised_coralai_repro/seed_comparison
```

ASAL's Sep-CMA-ES evaluates eight candidates plus the default, using one reused CPU scorer and serial headless rollouts.
The 100×100, eight-patch pilot runs 1,000 updates, sampling every 50; selection seed 42, held-out seeds 101/202/303.
Selection minimizes mean ASAL loss over updates 50–1,000; alignment is its negative (higher is better). Step 0 is excluded.
Best sampled-frame and final-frame scores are separate columns; `selected_config.json` is the saved winner.
Recorded evidence: `results/week9/supervised_coralai_20261022/{search,seed_comparison}/` (CSV/JSON scores, configs, logs, frames, checkpoints, versions, commands).
Taichi CPU scheduling can change runs slightly; the three-seed comparison measures this small experiment, not a general improvement guarantee.

## Next steps
- Compute Shannon and discretized entropy on saved runs.
- Connect Coralai simulation frames to ASAL scoring.

## Upstream projects
- ASAL: https://github.com/SakanaAI/asal
- Coralai: https://github.com/aidanbx/coralai
