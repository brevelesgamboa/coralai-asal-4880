# Coralai + ASAL

CSC 4880 project connecting Coralai ecosystem simulations to ASAL automated search and evaluation.

## Current progress

- Config-based headless runner with explicit seeds, saved settings, screenshots, and checkpoints.
- CLIP scoring of simulation frames over time, with CPU/GPU score agreement verified.
- ASAL Sep-CMA-ES integration for supervised configuration search.
- Independent evaluation of saved runs using pretrained SigLIP.
- Shared [configuration and result contract](docs/coralai_pilot_contract.md).

## Latest results

| Experiment | Finding | Records |
|---|---|---|
| Supervised search | Eight candidates evaluated. The selected configuration slightly improved selection-seed alignment, but scored below the default on all three held-out seeds under CLIP. | [Search results](results/week9/supervised_coralai_20261022/) |
| Second-model evaluation | Evaluated 120 saved frames. CLIP preferred the default on all three seeds; SigLIP preferred the selected configuration on two. | [Model validation](results/week10/model_validation/) |
| CPU/GPU comparison | Scores closely matched. CPU scoring was faster in this benchmark: 3.35 s versus 5.84 s per pass, excluding model loading. | [Device comparison](results/week9/clip_gpu_gfx1031_20261022/) |

The search and second-model findings cover one target prompt and three held-out seeds. CLIP and SigLIP use different score scales, so we compare their configuration preferences.

## Verified headless CPU run

From the repository root, using Python 3.11:

```sh
python3.11 -m venv .venv
. .venv/bin/activate

python -m pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c results/week9/headless_runner_verification_20261022/requirements-used.txt external/coralai external/coralai/coralai/dependencies/PyTorch-NEAT

python external/coralai/experiments/coral_dev/headless_coral.py \
  --config configs/headless_cpu_repro.json --seed 42 \
  --output-dir data/raw/headless_repro/run_1
```

Use a fresh output directory for each run. This verification config runs 51 steps on a 50×50 grid and saves settings, screenshots, and checkpoints at updates 50 and 51.

Repeated runs produced identical screenshots and also ran successfully in a clean environment. See the [verification records](results/week9/headless_runner_verification_20261022/) for commands, versions, and numerical comparisons.

This setup covers the Coralai CPU runner. ASAL search and SigLIP evaluation use their separately recorded environments and commands in the linked experiment folders.

## Pilot format checks

These checks do not require running Coralai or downloading CLIP. They require `jsonschema==4.23.0`.

```sh
python scripts/validate_coralai_contract.py config configs/coralai_pilot.default.json
python scripts/validate_coralai_contract.py result examples/coralai_pilot_result.json
python -m unittest discover -s tests -p 'test_coralai_contract.py'
```

See [pilot validation](docs/coralai_pilot_validation.md) for saved-artifact checks and reproduction details.

## Next steps

- Check repeated-run variation and improve search evaluation across seeds.
- Compare entropy trends, genome diversity, and resource distributions across environments.
- Investigate GPU scoring performance through profiling and batching.

## Upstream projects

- [ASAL](https://github.com/SakanaAI/asal)
- [Coralai](https://github.com/aidanbx/coralai)