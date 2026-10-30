# Issue #20: Check Repeatability Across Seeds

Independent reproduction, on a second machine, of the Week 9 held-out seed
comparison (`results/week9/supervised_coralai_20261022/seed_comparison`),
which found the search-selected config (`radiate_interval=10`) scoring lower
than the default (`radiate_interval=50`) on all 3 held-out seeds.

## Command

Read-only against the existing finished search in
`results/week9/supervised_coralai_20261022/search` (not modified); output
written to this task's own folder.

```
$env:JAX_PLATFORMS = "cpu"
$env:PYTHONIOENCODING = "utf-8"
python scripts/coralai_search.py compare --experiment results/week9/supervised_coralai_20261022/search --output-dir results/week10/issue20_seed_repeatability/seed_comparison_millz
```

`PYTHONIOENCODING=utf-8` is required on Windows — see Bug below.

## Result

Reproduced on different hardware (Windows ARM64 laptop vs. the original
machine) using the same 3 held-out seeds (101, 202, 303):

| metric | original | reproduced here | difference |
|---|---|---|---|
| default mean alignment | 0.25010 | 0.25009 | 0.00002 |
| selected mean alignment | 0.24443 | 0.24438 | 0.00006 |
| paired difference (selected - default) | -0.00567 | -0.00571 | 0.00004 |
| seeds where selected beat default | 0 / 3 | 0 / 3 | match |

**Repeatability confirmed.** The cross-machine difference (~0.00002-0.00006)
is smaller than the same-machine rerun noise the original search already
reported (~0.0002, per `same_config_repeat` in
`results/week9/supervised_coralai_20261022/summary.json`, since Taichi CPU
scheduling isn't bitwise reproducible even on one machine). The Week 9
finding is not a one-machine fluke: the selected config loses to the default
on held-out seeds consistently.

## Bug found: Windows console encoding crash

`coralai/evolver.py`'s `save_checkpoint` prints a Unicode arrow (`→`)
on every checkpoint save. On Windows, when stdout is redirected to a file
(not a live console) — which is exactly what `scripts/coralai_search.py`
does for every evaluation — Python defaults to the `cp1252` codec, which
can't encode that character, and the whole run crashes
(`UnicodeEncodeError`) right after the final checkpoint is otherwise
written successfully. Worked around here by setting `PYTHONIOENCODING=utf-8`
before running; not fixed in `evolver.py` itself.

## Output location

Full run data (6 simulations: default + selected x 3 seeds, checkpoints,
screenshots, scores): `results/week10/issue20_seed_repeatability/seed_comparison_millz/`.
