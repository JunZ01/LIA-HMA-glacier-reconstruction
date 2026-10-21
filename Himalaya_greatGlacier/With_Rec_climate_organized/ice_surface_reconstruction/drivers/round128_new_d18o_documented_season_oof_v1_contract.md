# Round 128 New d18O Documented-Season OOF

## Objective

Determine whether the four new annual tree-ring d18O records are represented
more faithfully by their published climate seasons than by annual temperature
and precipitation predictors.

## Fixed candidates and seasons

- MiMei: June-August temperature and precipitation.
- Xinpu: June-August temperature and precipitation.
- Ganesh: June-September temperature and precipitation.
- Keylong: current June-August temperature and previous-October through
  current-April precipitation.

## Frozen boundaries

- Candidate values and climate predictors for 1921-1939 are not decoded or read.
- Keylong year 1940 is left missing because its documented winter predictor
  would require October-December 1939.
- Modern outer blocks remain 1940-1951, 1952-1963, 1964-1975, 1976-1987,
  and 1988-2000; fits use only nonmissing training rows.
- Temperature products are Berkeley Earth and CRU. Precipitation products are
  GPCC and CRU, using the same source files as frozen Round 118.
- Annual bivariate and documented-season bivariate models use identical OLS,
  folds, sites, response values, and product migration paths.
- Wache remains quarantined. The sealed final-test reserve is not opened.
- Original inputs are read-only and no OGGM experiment is run.

## Decision

A candidate advances to a seasonal reconstruction operator only if the median
documented-season improvement over its annual bivariate model is at least 1%,
and the documented-season model has positive median OOF skill and correlation
across the four self/cross-product paths.
