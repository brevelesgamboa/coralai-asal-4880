# Coralai + ASAL

CSC 4880 project using ASAL to search Coralai simulation settings with text prompts.

## Setup

Use Python 3.11 and run commands from the repository root. The examples below use CPU.

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c results/week9/supervised_coralai_20261022/requirements-used.txt \
  jax jaxlib flax transformers evosax einops tqdm dotmap jsonschema \
  external/coralai external/coralai/coralai/dependencies/PyTorch-NEAT
```

Download the CLIP weights once before running a search:

```sh
python -c 'from huggingface_hub import snapshot_download; snapshot_download("openai/clip-vit-base-patch32", revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268")'
```

## Run a simulation

```sh
python external/coralai/experiments/coral_dev/headless_coral.py \
  --config configs/headless_cpu_repro.json --seed 42 \
  --output-dir data/raw/example_run
```

This config runs 51 steps on a 50×50 grid. Each run saves its settings, screenshots, logs, and checkpoints.
Use a new output directory for each run. Change the JSON config to adjust the simulation.

## Run supervised search

```sh
JAX_PLATFORMS=cpu python main_opt.py --substrate coralai \
  --coralai_config configs/coralai_supervised.default.json \
  --coralai_search_config configs/coralai_supervised.search.json \
  --pop_size 4 --n_iters 2 --sigma 0.25 --seed 0 \
  --save_dir results/supervised_search/search
```

ASAL's Sep-CMA-ES generates and evaluates eight candidates, plus the default configuration.
Each simulation runs 1,000 steps on a 100×100 grid with screenshots every 50 steps.
The search config sets the prompt, parameter ranges, and simulation seeds.

Selection uses mean CLIP alignment across steps 50–1,000. Higher alignment is better.
Results include separate scores for the best sampled frame and the final frame.
The selected settings are saved in `results/supervised_search/search/selected_config.json`.

Compare the selected settings with the default on three additional seeds:

```sh
JAX_PLATFORMS=cpu python scripts/coralai_search.py compare \
  --experiment results/supervised_search/search \
  --output-dir results/supervised_search/seed_comparison
```

The comparison saves per-seed scores, paired differences, averages, and variation.
CPU runs can vary slightly with the same seed.

## Score saved screenshots

```sh
python scripts/score_coralai_progression.py \
  --images-glob 'data/raw/example_run/screenshots/step_*.png' \
  --prompt 'branching colonies' --device cpu \
  --output results/example_scores.json
```

For a CPU/GPU score and timing comparison, use an environment with GPU-enabled JAX:

```sh
python scripts/compare_clip_devices.py \
  --images-glob 'data/raw/example_run/screenshots/step_*.png' \
  --prompts 'branching colonies' --warmup 2 --repeats 5 \
  --output-dir results/device_comparison
```

## Results and checks

Saved experiments are in [results/](results/). The [supervised search results](results/week9/supervised_coralai_20261022/summary.json)
include configs, seeds, scores, runtimes, and reproduction commands.
That experiment improved the selection-seed score but scored lower than the default on all three additional seeds.
Hardware details and device benchmarks are saved with the [GPU comparison results](results/week9/clip_gpu_gfx1031_20261022/).

```sh
python -m unittest discover -s tests -p 'test_coralai*.py'
python -m unittest discover -s tests -p 'test_headless_coral_runner.py'
python -m unittest discover -s tests -p 'test_clip_devices.py'
```

See the [pilot format guide](docs/coralai_pilot_contract.md) and [validation instructions](docs/coralai_pilot_validation.md) for artifact checks.

## Upstream projects

- [ASAL](https://github.com/SakanaAI/asal)
- [Coralai](https://github.com/aidanbx/coralai)
