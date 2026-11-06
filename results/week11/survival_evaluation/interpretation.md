All six saved ecosystems contain living cells at update 1,000, and none of the
recorded population observations is zero. Default ends with 1,912, 1,934, and
1,899 living cells on seeds 101, 202, and 303; selected ends with 2,304, 2,326,
and 2,400. Selected therefore retains 392, 392, and 501 more cells
than default on matched seeds. Selected populations grow from their initial
states, while default populations finish slightly below their initial counts.
Both configurations remain occupied at the final observation; greater selected
occupancy does not establish longer survival or uninterrupted persistence.

The original 20-frame means still favor default under CLIP on every seed.
SigLIP favors selected on seeds 101 and 202 and default on 303. Thus neither
model's preferences consistently rank final population size: both prefer the
less populated default run on seed 303, both in temporal means and at the
exactly matched final update. For every default run, both scores
increase from update 50 to 1,000 while population decreases between updates
0 and 1,000. These are different observation windows, so this is a descriptive
divergence, not a matched-frame association or evidence that score increases
cause population loss. No scored frame is confirmed to depict zero population.

The log's step label trails completed updates by one. Consequently, only the
six final checkpoints align exactly with the 120 scored frames; the other 114
frames lack same-update population measurements. Population and score curves
retain their actual sampling times. No values were interpolated. The study
contains six runs paired across three seeds, with temporally dependent frames,
separate model scales, and no causal or independent-frame statistical claims.
