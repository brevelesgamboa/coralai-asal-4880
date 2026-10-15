# Coralai pilot validation

Issue #5 checks saved artifacts and reproduction of scores using the
[shared format](coralai_pilot_contract.md). The first checks and remaining
dependencies are recorded in [the validation results](../results/week8/validation/README.md).

## Setup and format tests

Run from the repository root in a Python environment containing Torch, NumPy,
Pillow, and `jsonschema`. Artifact validation uses CPU checkpoint loading and
does not initialize Taichi or load a foundation model.

```sh
python -m unittest discover -s tests -p 'test_coralai*.py'
python scripts/validate_coralai_contract.py config configs/coralai_pilot.default.json
python scripts/validate_coralai_contract.py result examples/coralai_pilot_result.json
```

The example manifest demonstrates the format. It has no real artifact files,
so use a manifest produced by the runner for the following checks.

## Check a real pilot run

Replace `RUN_ID` with the actual run directory after issue #3 provides the
config-driven runner and its documented run command:

```sh
python scripts/validate_coralai_pilot.py \
  data/raw/pilot/RUN_ID/result.json \
  --output results/week8/validation/RUN_ID.json
```

The checker validates the manifest, required checkpoint files, completed-step
metadata, grid shape, finite 14-channel tensors, integer genome IDs, and the
step log's presence and data rows. It regenerates the final image with the
existing renderer and requires exact pixel agreement. It recomputes entropy
with the existing functions and compares stored metrics when present (relative
and absolute tolerance `1e-9`). It also checks settings recorded in run metadata
and lists which settings it could verify. Missing metadata settings are not
evidence that the requested values reached the simulation.

The output report records `simulation_rerun: false` and `clip_recomputed: false`:
this command checks saved artifacts. `population.pkl` is checked for presence
as a file, not unpickled; successful resume is a separate integration check.
Keep report outputs outside input run directories. Exit code zero means these
artifact checks passed; exit code one records a failed check in the report.

## Reproduce the existing saved results

These commands work on the local week 6 checkpoints and preserve committed
week 7 baselines. They evaluate saved data rather than start fresh simulations.

```sh
python substrates/coralai.py \
  data/raw/runs/coral_dev_20261001_174547/checkpoint_0001000/substrate.pt \
  --output data/raw/validation/week8/coralai_frame.png

python scripts/score_entropy.py \
  data/raw/runs/coral_dev_20261001_174547/checkpoint_0001000/substrate.pt \
  --output data/raw/validation/week8/coralai_entropy.json

python scripts/compare_entropy.py --output data/raw/validation/week8/entropy

python scripts/score_coralai.py \
  --image data/raw/validation/week8/coralai_frame.png \
  --prompt "branching colonies" \
  --output data/raw/validation/week8/coralai_clip_score.json
```

The CLIP command requires the pinned ASAL/JAX/Flax/Transformers dependencies and
access to model weights. It currently runs on CPU. `compare_entropy.py` accepts
`--runs` if the saved run directories are stored elsewhere. The baseline files
are in `results/week7/`; the checked-in reproduction report records comparisons
and tolerances. Checkpoint label 1000 is a legacy filename, not proof of exactly
1000 completed updates: the existing external runner's loop and checkpoint
labels can differ by one. Preserve those labels when reporting old data.

## Compare the default and searched candidate

After issue #4 produces a candidate, rerun the default and candidate with the
same seed, grid size, steps, hardware settings, prompt, and entropy bin width.
Keep separate run directories. Pass the parameter actually searched (for
example, `infra_decay`); this example does not choose the search parameter for #4.

```sh
python scripts/validate_coralai_pilot.py \
  data/raw/pilot/CANDIDATE_RUN/result.json \
  --baseline data/raw/pilot/DEFAULT_RUN/result.json \
  --parameter infra_decay \
  --output results/week8/validation/default_vs_candidate.json
```

The comparison rejects configuration differences beyond the searched parameter
and output root. It validates both runs and reports freshly computed entropy
deltas as candidate minus baseline. If both manifests have CLIP scores, it also
reports their stored target-loss delta; a negative delta favors the candidate.
Unscored runs retain a `null` loss delta. Provenance-record equality is reported
separately; equal unknown fields do not establish identical software versions.
Use matching source/software versions and record differences when rerunning.

## Checks needed before closing #5

- Run the documented config-runner command on a fresh default configuration and
  confirm the requested settings reach the simulation.
- Validate its real manifest, final checkpoint, and image.
- Repeat the run and record score/entropy variation; saved-artifact recomputation
  alone does not establish simulation reproducibility.
- If search finishes, compare its candidate against the default using matching
  settings and the existing entropy functions.
- Record runner, model setup, and reproduction problems with the measured results.
