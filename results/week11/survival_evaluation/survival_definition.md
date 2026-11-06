# Survival definition

We measure survival using **occupied grid cells**:

- A cell counts as occupied when `genome >= 0` (state channel 13).
- `genome = -1` marks an empty or inactive location.
- **Population** is the number of occupied cells.
- **Occupied fraction** is population divided by all grid locations: 10,000 per run.

This measures occupancy, not the number of organisms or whether cells can
reproduce. Energy or bright pixels alone cannot tell us whether a cell is occupied.
The simulator marks cells inactive when energy + infrastructure is at most 0.05.

The historical logs have a timing offset: log step 50 describes completed
update 51. Only update 1,000 has population measured at exactly the same update
as the model scores. Other measurements stay at their recorded times; missing
values are left blank. Sparse observations cannot prove permanent extinction.

Verified against simulator commit `c782434f33807704d52982c353854771fa15591c`:

- `external/coralai/experiments/coral_dev/experiment.py:32–43,152–164`: state fields and death rule.
- `external/coralai/experiments/coral_dev/physics.py:34–37`: inactive cells have disabled actions.
- `external/coralai/experiments/coral_dev/headless_coral.py:217–227`: recording timing and occupied-cell count.
