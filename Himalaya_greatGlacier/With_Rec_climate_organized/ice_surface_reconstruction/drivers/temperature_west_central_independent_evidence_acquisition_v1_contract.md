# West/Central HMA Independent Temperature Evidence Acquisition v1

## Purpose

Admit only genuinely new, machine-readable temperature evidence relevant to the
1880-1950 residual-bias question. This is an evidence-acquisition experiment,
not a reconstruction correction experiment.

## Frozen constraints

- Original project and downloaded source files are read-only.
- Results are written only to the experiment-specific output directory.
- Years 1921-1939 are sealed. They are never converted to numeric values,
  summarized, plotted, correlated, or used in any decision.
- No production temperature file is modified.
- No EnKF or OGGM run is permitted.
- CRU, Berkeley Earth, and other station-derived grids are background products,
  not independent votes.

## Admission classes

- `A`: direct observations or documentary evidence independent of the current
  proxy network.
- `B`: a reconstruction with machine-readable values and source material shown
  not to overlap the current assimilated proxy network.
- `C`: useful context with unresolved overlap, unsuitable seasonality, or
  insufficient spatial support. It cannot vote for a correction.
- `X`: excluded because of locked-period exposure, source duplication, or an
  unresolvable provenance failure.

## Candidate-v3 gate

Candidate v3 may be designed only if at least two independent `A` or `B`
sources agree on the direction and low-frequency temporal shape of the
1880-1920 residual. Otherwise retain the original production temperature and
register the unresolved structural uncertainty.

