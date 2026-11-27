# Coralai + ASAL

CSC 4880 project studying Coralai ecosystem simulations and working toward ASAL supervised target search.

## Completed work

- Integrated Coralai under `external/coralai/` and organized documentation and experiment results.
- Verified reproducible headless CPU runs with saved configurations, screenshots, and checkpoints.
- Defined a shared pilot configuration and result format with JSON validation.
- Compared fixed, slower, and faster food movement, including optional direction reversal, using matched settings and saved-state visualizations.

## Documentation and results

- [Pilot configuration and result format](docs/coralai_pilot_contract.md)
- [Pilot validation and reproduction instructions](docs/coralai_pilot_validation.md)
- [Headless CPU verification records](results/week9/headless_runner_verification_20261022/)
- [Week 13 food-movement experiments, evidence, and slide assets](results/week13/presentation_ready/START_HERE.txt)

## Next steps

- Compute Shannon and discretized entropy on saved simulation runs.
- Connect frames from the verified Coralai NEAT simulator to ASAL scoring and target search.

## Upstream projects

- [ASAL](https://github.com/SakanaAI/asal)
- [Coralai](https://github.com/aidanbx/coralai)