# Round 115 Temperature Low-Rank OOF Contract

## Question

Can the metadata-ready, documented-temperature proxy9 evidence groups recover
transferable HMA annual temperature fields with a method family that is
materially different from the closed pointwise/global-increment EnKF routes?

## Frozen Data Boundary

- Instrumental and proxy values are read only for 1940-2000.
- The consumed 1921-1939 interval is never requested or reopened.
- Round 114 external temperature series remain sealed and do not enter fitting,
  ranking, hyperparameter selection, or threshold selection.
- Original MAT, NetCDF, CSV, and source files are read-only. Every generated
  artifact is written to the independent Round 115 result directory.

## Proxy Network

- Use proxy9 columns joined one-to-one to the Round 110 v4 master ledger by
  `network_record_1based`.
- Retain only `documented_network_role_v4 == documented_temperature` and
  `operator_metadata_ready_v4 == True`.
- One proxy9 column represents one evidence group; no cross-network duplicate
  weighting is introduced.
- In each fit, proxy means and scales use training years only. Missing proxy
  values are set to the corresponding training mean after standardization.
  Columns with fewer than 20 finite training years or zero training variance
  are inactive for that fit.

## Candidate Method

- Target: the unweighted five-product mean of Berkeley Earth, CRU TS, GISTEMP,
  HadCRUT5, and NOAA annual temperature anomalies on the established 336-state
  grid, evaluated only on the HMA mask.
- Five outer blocks: 1940-1951, 1952-1963, 1964-1975, 1976-1987, 1988-2000.
- For every outer block, the target mean, spatial EOFs, proxy normalization,
  regression coefficients, rank, and ridge penalty are learned without that
  block.
- EOFs are area weighted and fitted only on the current training years.
- Rank grid: 1, 2, 3, 5, 8. Ridge grid: 0.1, 1, 10, 100, 1000.
- Rank and ridge are selected by four inner blocked validations within each
  outer training set. The inner objective is the median regional RMSE over all
  four HMA longitude regions and all five instrumental products. Ties prefer
  lower rank and then stronger regularization.
- The stitched OOF candidate is recentered to 1961-1990 using predictions only.

## Frozen Comparators And Gates

- Frozen legacy control: `tas_ensemble_mean_rec_result`, the equal-six project
  reconstruction. `tas_ensemble_mean_6` is not the control.
- Absolute control: the zero-anomaly field.
- Apply all required Round 113 modern gates without relaxation:
  median RMSE improvement over equal-six at least 5%; worst pair regression no
  worse than 5%; at least 75% of block-region-product paths improve; median
  absolute skill at least 5%; worst absolute regression no worse than 5%; at
  least 75% positive regional skill; median annual regional correlation at
  least 0.30; median spatial skill positive; worst spatial regression no worse
  than 5%; at least 75% positive spatial product skill; and median annual
  spatial pattern correlation at least 0.20.

## Decision

Only a pass of every required modern gate advances this method to a separately
contracted full 1000-2000 temperature reconstruction. Failure closes this
specific low-rank annual proxy9 method; it does not authorize tuning against
Round 114 external evidence, changing gates, reopening 1921-1939, or running
OGGM.
