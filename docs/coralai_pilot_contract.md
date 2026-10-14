# Coralai pilot configuration and results, version 1

This is the shared interface for the fresh headless runner in [issue #3](https://github.com/brevelesgamboa/coralai-asal-4880/issues/3)
and supervised target search in [issue #4](https://github.com/brevelesgamboa/coralai-asal-4880/issues/4).
It completes the format definition in [issue #2](https://github.com/brevelesgamboa/coralai-asal-4880/issues/2).
The schemas, examples, and validator work now. The config-driven simulation runner
and search are planned implementations; this document does not claim they exist.
Real runs, command checks, and checkpoint comparisons belong to [issue #5](https://github.com/brevelesgamboa/coralai-asal-4880/issues/5).

## Files and validation

- [Default configuration](../configs/coralai_pilot.default.json)
- [Configuration schema](../schemas/coralai_pilot_config.schema.json)
- [Result schema](../schemas/coralai_pilot_result.schema.json)
- [Illustrative runner result](../examples/coralai_pilot_result.json)
- [Shared Python validator](../scripts/validate_coralai_contract.py)

Both documents are UTF-8 JSON objects with `schema_version: 1` and `snake_case`
keys. All settings and result fields shown in the schemas are required. Unknown
fields and versions are rejected, including inside sections. JSON booleans are
not integers. Following JSON Schema, whole-valued numbers such as `100.0` count
as integers; the runner must use integer values for CLI flags and tensor sizes.
Duplicate keys, `NaN`, infinities, and numeric overflow are rejected
by the shared validator. Copy the default file and edit values; the validator
does not fill defaults, coerce values, or mutate documents. JSON Schema `default`
annotations are not needed to load a configuration.

Run these commands from the repository root in a Python environment with
`jsonschema==4.23.0`, already listed in `requirements.txt`:

```sh
python -m pip install 'jsonschema==4.23.0'
python scripts/validate_coralai_contract.py config configs/coralai_pilot.default.json
python scripts/validate_coralai_contract.py result examples/coralai_pilot_result.json
python -m unittest discover -s tests -p 'test_coralai_contract.py'
```

Validation needs no Torch, Taichi, JAX, GPU, model download, or saved checkpoint.
Schemas use Draft 2020-12. Their `urn:` identifiers are registered from local
files by the validator, so validation does not fetch schema references online.
For Python callers, use the same functions at both boundaries:

```python
from scripts.validate_coralai_contract import load_json, validate_config, validate_result

config = load_json("configs/coralai_pilot.default.json")
validate_config(config)
# The issue #3 runner consumes config and returns a manifest.
# validate_result(manifest) before saving it or scoring its final image.
```

Schema validation checks structure and ranges. `validate_result` also checks
related values, such as the requested versus completed steps, checkpoint
schedule, and score signs.
It does not check artifact existence, tensor contents, available hardware, or
whether the simulation actually received the settings; those are runner and
integration-validation responsibilities.

## Configuration accepted by the pilot runner

Version 1 always runs the `coral_dev` experiment from a fresh state without a GUI.
It evaluates one final frame and one text prompt. These choices are fixed rather
than additional switches. The runner validates the entire configuration before
initializing the simulation, forwards every `simulation` setting, and records
the complete configuration in its result. Scoring consumes `evaluation`; the
runner preserves that section without loading a foundation model.

### Simulation settings

Field names below are inside `simulation`. CLI mappings refer to the existing
`external/coralai/experiments/coral_dev/run.py`, which currently accepts flags,
not a pilot config file.

| Field | Type and allowed values | Pilot default | Existing CLI flag / meaning |
|---|---|---|---|
| `seed` | Integer, 0 through 4294967295 | `42` | `--seed`; seed Python, NumPy, and Torch |
| `shape` | Positive integer | `100` | `--shape`; square grid side, yielding `[100, 100]` |
| `steps` | Positive integer | `1000` | `--steps`; number of completed simulation updates |
| `env` | `"flat"` or `"patches"` | `"patches"` | `--env`; pilot environment |
| `env_param` | Positive integer for patches; `null` for flat | `8` | `--env-param`; patch count; omit the flag for flat |
| `backend` | `"cpu"`, `"cuda"`, or `"metal"` | `"cpu"` | `--backend`; Taichi backend |
| `device` | `"cpu"`, `"cuda"`, or `"mps"` | `"cpu"` | `--device`; Torch device |
| `infra_decay` | Number in `[0, 1]` | `0.0` | `--infra-decay`; infrastructure fraction lost per update; zero disables |
| `defense_coeff` | Nonnegative number | `0.0` | `--defense-coeff`; infrastructure defense multiplier; zero disables |
| `radiate_interval` | Positive integer | `50` | `--radiate-interval`; updates between radiation mutations |
| `cleanup_interval` | Positive integer | `100` | `--cleanup-interval`; updates between extinct-genome cleanup |
| `emergency_pop_cap` | Nonnegative integer | `1000` | `--emergency-pop-cap`; emergency live-genome cap; zero disables |
| `checkpoint_interval` | Nonnegative integer | `100` | `--checkpoint-interval`; periodic checkpoint spacing; see final-save rule below |
| `log_interval` | Positive integer | `10` | `--log-interval`; updates between log entries |

Supported backend/device pairs are `cpu/cpu`, `cuda/cuda`, and `metal/mps`.
Unavailable requested hardware must produce an explicit error, never silently
fall back. CPU is an explicit baseline, not a performance guarantee. The default
grid size, seed, and patch count match the existing week 6 patch baseline;
other defaults make a finite pilot and frequent checkpoints convenient. They
are not a claim that this configuration has been benchmarked or reproduced.

For patches, `env_param` sets the initial patch count and the count per spawn wave
(`n_spawn` defaults to `n_patches`). Remaining environment parameters stay at
the experiment defaults: radius fraction `0.07`, capacity `30.0`, injection rate
`0.06`, spawn interval `250`, and seed infrastructure `0.4`. Record the resolved
environment in run metadata and the experiment snapshot so these defaults can
be inspected. To use flat, change **both** `env` to `"flat"` and `env_param` to
`null`; do not pass a numeric parameter to the flat environment constructor.

The initial interface deliberately covers the two existing comparison
environments. Resume, GUI, profiling, benchmarking, Vulkan, other environments,
multiple prompts, temporal sampling, and configurable rendering are outside v1.
Extend the contract deliberately if either implementation needs them.

### Evaluation and output settings

| Field | Type and allowed values | Pilot default | Consumer |
|---|---|---|---|
| `evaluation.prompt` | String containing non-whitespace text | `"branching colonies"` | ASAL supervised target scoring |
| `evaluation.bin_width` | Finite positive number | `1.0` | Existing fixed-width resource entropy functions |
| `output.root` | Relative directory path using forward slashes | `"data/raw/pilot"` | Runner artifact destination |

`output.root` is resolved relative to the project repository root, independent
of the caller's working directory. The runner creates a unique `run_id` and a
new directory at `<repo>/<output.root>/<run_id>/`; existing runs must not be
overwritten. Absolute paths, backslashes, colons, control characters, empty
components, and `.` or `..` components are rejected. Use forward slashes without
a trailing slash. Generated raw artifacts remain under ignored `data/`; commit
small summaries separately under `results/` when carrying out issue #5.

Search copies this full configuration for each candidate and changes only the
chosen parameter. Trial counts, search bounds, search RNG state, and the search
algorithm belong to search's own settings, not this runner config. Simulation
seed, shape, step budget, rendering, model, prompt, and entropy bin width must
match when comparing the default and best candidate. A search RNG seed does not
implicitly replace the simulation seed.

## Result returned by the runner

The planned Python interface is `run(config) -> manifest`, where `config` and
`manifest` are dictionaries validated by the shared functions. The runner saves
the same dictionary as `<run_dir>/result.json`. A future CLI should read a config
file, save that manifest, and exit zero for success or nonzero for failure;
its entry-point name will be documented in issue #3. No config-runner command is
advertised here as currently executable.

The checked-in example is **illustrative**, not evidence of a simulation. Its
artifact paths are placeholders relative to the example manifest; the files do
not exist. `null` scores mean not evaluated, never a measured zero. Unknown
provenance is `null` and the example has no recorded package versions.

| Result field | Meaning |
|---|---|
| `schema_version` | `1`, matching the effective configuration |
| `run_id` | Unique run-directory name; letters, digits, underscores, and hyphens; first character alphanumeric |
| `status` | `"success"` or `"failed"` for simulation and artifact generation |
| `effective_config` | Complete validated settings actually used, not a patch or the original filename |
| `completed_steps` | Actual number of completed updates, starting from zero |
| `artifacts.checkpoints` | Ordered list of `{completed_steps, directory}` for saved checkpoints |
| `artifacts.final_checkpoint` | Final checkpoint directory, or `null` if unavailable |
| `artifacts.final_image` | PNG rendered from the final checkpoint, or `null` |
| `artifacts.log` | Step log CSV path, or `null` |
| `artifacts.metadata` | Run metadata JSON path, or `null` |
| `provenance` | Main-repo Git commit, dirty flag, actual backend/device, and package versions |
| `evaluation.clip` | ASAL scores added by evaluation/search, or `null` |
| `evaluation.entropy` | Existing entropy metrics added by evaluation, or `null` |
| `error` | `null` on success; `{code, message}` on failure |

Artifact paths are relative to the directory containing `result.json`, **not**
the repository root or `output.root`. Checkpoint directories use
`checkpoint_<completed_steps padded to at least 7 digits>`, containing
`substrate.pt`, `population.pkl`, and `meta.json`. Consumers append
`substrate.pt` to `final_checkpoint` to render or calculate entropy.

Checkpoint steps must strictly increase, with no duplicates or future steps.
With a positive checkpoint interval, save step zero, every completed multiple
of the interval, and the final state. With interval zero, skip initial and
periodic saves but **still save the final state**. Save an interval-aligned final
state once. `final_checkpoint` must identify the last listed checkpoint and
match `completed_steps`, and any final image must derive from that state.

A success completes exactly `effective_config.simulation.steps`, saves the final
checkpoint and image plus logs and metadata, and returns `error: null`. Do not
set success until required artifact writes finish. Record actual package
versions (for example `python`, `torch`, and `taichi`) in `provenance.versions`;
its keys are package names. Use a full lowercase Git SHA and a dirty flag when
available, or `null` when unavailable. Backend/device must match the effective
configuration; v1 allows no automatic hardware fallback.

A failure after configuration validation returns `status: "failed"`, an error
code such as `simulation_failed` or `render_failed` and a useful message, the
actual completed count, and any artifacts saved before failure. Missing paths
are `null`, and both evaluation entries are `null`. A failure can have completed
all updates if rendering or saving failed. A last periodic checkpoint that is
earlier than the actual completed count stays in the checkpoint list; it is not
reported as `final_checkpoint`. If a final-state checkpoint was saved before a
rendering failure, it can be reported with a `null` final image. Invalid configs
are rejected before a run starts and do not produce a manifest. If the output
destination itself cannot be written, return/report the failure even when
`result.json` cannot be persisted.

The current external runner uses a loop index for periodic filenames after
updating the state, so a label can differ from its saved evolver timestep. It
also does not guarantee a final save. Issue #3 must normalize completed-update
counts and implement the final-save rule. Do not relabel old checkpoints by
assuming their filename is an exact completed count.

## Rendering and scores

Use the existing `substrates/coralai.py` renderer on the final `substrate.pt`.
The saved finite cell tensor has shape `(1, 14, W, H)` for `coral_dev`. Red is
energy (channel 0) divided by 10 and clipped to `[0, 1]`; green is infrastructure
(channel 1) treated identically; blue is `0.25` where genome ID (channel 13) is
nonnegative, otherwise zero. Convert x/y cell coordinates to image row/column,
round to uint8, and resize with nearest-neighbor sampling to an RGB PNG of
224 x 224 pixels. ASAL receives float RGB values in `[0, 1]`.

The runner returns `evaluation: {"clip": null, "entropy": null}`. Evaluation
validates the result, scores only successful runs, and fills either entry
independently. A scoring failure leaves that entry `null` and is recorded by
search/evaluation; it does not change a successful runner status. Do not rank an
unscored or failed candidate as zero loss.

`evaluation.clip` contains `model: "ASAL CLIP"`, `asal_target_loss`, and
`alignment`, copying those fields from the existing `scripts/score_coralai.py`
output. The fixed model is `openai/clip-vit-base-patch32` through the existing
ASAL foundation-model code. Score the final image against
`effective_config.evaluation.prompt`. Lower `asal_target_loss` is better;
`alignment = -asal_target_loss` (floating point tolerance 1e-9) is higher-is-better.
Search should minimize the loss directly.

`evaluation.entropy` copies these fields from
`scripts/score_entropy.py::score_checkpoint` on the final state:

- `alive_cells` and `occupied_genome_ids`: integer counts.
- `genome_shannon_bits`: Shannon entropy of genome counts in living cells only.
- `energy_binned_entropy_bits` and `infrastructure_binned_entropy_bits`: entropy
  over all grid cells, with fixed-width bins anchored at zero and no clipping.
- `bin_width`: exactly `effective_config.evaluation.bin_width`.

These entropies are measured in bits, using the existing functions. Do not
replace them with entropy of rendered pixel colors. Use the final artifact paths
and effective config as the source of image, checkpoint, prompt, and grid shape,
instead of duplicating those values inside score objects.

For issue #5, compare settings and completed counts before comparing scores.
Fixed seeds and recorded software/hardware support an auditable rerun, but do
not promise bitwise replay: the current Coralai kernels have known scheduling
and floating point nondeterminism. Record variation and reproduction problems
with real results rather than treating structural validation as reproduction.
