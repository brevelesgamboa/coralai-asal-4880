# Issue #4: Supervised Target-Search Pilot — `radiate_interval` Sweep

Swept parameter: `radiate_interval` (steps between radiation/mutation events).
Target prompt: `"branching colonies"` (same prompt used for the week 7 baseline, for comparability).

## Method

Five configs, identical except for `radiate_interval`, per the pilot contract's rule
that only the searched parameter changes between candidates:

- Fixed: `seed=42`, `shape=100` (100x100 grid), `env=patches`, `env_param=8` (8 energy
  patches), `steps=1001`, `infra_decay=0.0`, `defense_coeff=0.0`, `cleanup_interval=100`.
  (`steps=1001` with `checkpoint_interval=1000` is a deliberate choice, not a contract
  default — see Notes below.)
- Varied: `radiate_interval` in `{10, 25, 50, 100, 200}`. `50` is Coralai's own default,
  used here as the baseline.

For each run: ran `headless_coral_dev.py` (from Ctm31's issue #3 branch,
`headless-coral-runs`, not yet merged) to produce a final checkpoint, rendered the
checkpoint to a 224x224 image with `substrates/coralai.py`, then scored that image
against the target prompt with `scripts/score_coralai.py` (ASAL CLIP,
`openai/clip-vit-base-patch32`).

Configs: `configs/issue4_radiate_interval/`.
Rendered frames + scores: `results/week8/issue4_radiate_interval/`.
Raw run data (checkpoints, screenshots, step logs): `data/raw/issue4_radiate_interval/`
(not committed — gitignored, same as the rest of `data/raw/`).

## Results

| `radiate_interval` | ASAL target loss | alignment | living cells (final) | genomes (final) |
|---|---|---|---|---|
| 25  | -0.25967 | 0.25967 (best)  | 1,958 | 87  |
| 10  | -0.25532 | 0.25532         | 2,233 | 120 |
| 200 | -0.25087 | 0.25087         | 1,832 | 54  |
| 100 | -0.24787 | 0.24787         | 1,777 | 59  |
| 50 (baseline) | -0.24727 | 0.24727 (worst) | 1,934 | 71  |

Lower loss (= higher alignment) means the rendered image matched "branching colonies"
more strongly. All five runs completed successfully — no failures to report.

## What worked / what we found

- **`radiate_interval=25` best matched the target prompt**, beating both more frequent
  (`10`) and less frequent (`100`, `200`) mutation, and beating Coralai's own default
  (`50`), which scored worst of the five. The relationship between mutation frequency
  and visual alignment to the target prompt is not monotonic — there's no simple
  "more mutation = better" or "less mutation = better" rule here.
- **Genome diversity scales monotonically with mutation frequency**, independent of
  the CLIP score: final genome count was 120 / 87 / 71 / 59 / 54 for
  `radiate_interval` = 10 / 25 / 50 / 100 / 200 respectively. More frequent radiation
  reliably produces more distinct genomes, but that diversity doesn't translate
  directly into a better prompt-alignment score.
- **Baseline sanity check**: this sweep's `radiate_interval=50` run (a fresh
  simulation, not the team's saved reference checkpoint) scored -0.24727, very close
  to the team's already-validated reference score of -0.24683 (issue #5). Good
  independent confirmation that render+score reproduces consistent results.

## Notes / things worth knowing

- Ctm31's `headless_coral_dev.py` (issue #3) never explicitly saves a checkpoint at
  the final step — only at exact multiples of `checkpoint_interval` strictly before
  the last step. With the script's own defaults (300 steps, checkpoint every 1000) it
  would save zero checkpoints. Worked around here by setting `steps=1001` with
  `checkpoint_interval=1000`, which lands a checkpoint exactly on the last step. Not a
  permanent fix — worth flagging on issue #3.
- `headless_coral_dev.py` was pulled directly from Ctm31's open branch
  (`headless-coral-runs`), not yet merged to `main`. It's also missing a `.py`
  extension there (committed as `headless_coral_dev`, not `headless_coral_dev.py`).
