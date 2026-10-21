# Round 167 Qamdo 2 standard chronology modern OOF V2

## Objective

Determine whether the newly recovered CHIN074 raw measurements contain transferable eastern-HMA temperature information and are not an alias of proxy6/proxy9.

## Frozen chronology

- Input: 144 CHIN074 raw ring-width series from the official NOAA/Mainz archive.
- Parse Tucson rows at byte-token level. Values assigned to 1921-1939 are discarded before numeric conversion.
- Detrend with OpenDendro `dplPy 0.1.5`, `fit=ModNegEx`, `method=residual`.
- Build one standard chronology with Tukey biweight, without prewhitening.
- The published `Qamdo_Recon.txt` is used only as a lineage/orientation check, never as the candidate chronology.
- No chronology-method, lag, season, site, or sign search is permitted.

## Identity gate

- Compare both the derived standard chronology and published reconstruction with every proxy6 and proxy9 sequence over allowed modern years.
- Exact affine alias: at least 50 overlaps, absolute correlation at least 0.999999, and maximum affine residual at most 1e-8.
- Near alias: at least 50 overlaps and absolute correlation at least 0.995.
- Any exact or near alias closes the candidate before OOF.

## Modern OOF gate

- Documented signal: positive response to previous-calendar-year January-December temperature.
- Pair ring year 1941-2000 with climate target year 1940-1999; this avoids the locked 1939 target.
- Products: Berkeley Earth and CRU TS4.04 at 31 N, 97 E.
- Five target-year blocks: 1940-1951, 1952-1963, 1964-1975, 1976-1987, 1988-1999.
- Fit only an intercept and one chronology slope on the other four blocks.
- Require for both products: pooled OOF correlation at least 0.30, positive skill against training-fold climatology, at least four positive fold correlations, all five slopes positive, and positive early/late split correlations.
- Require Berkeley/CRU OOF prediction correlation at least 0.80.

## Limits

Passing authorizes only a separate signal-free RCS recovery/replication stage. No paleo trajectory, production reconstruction, EnKF, Jionglacuo, 1921-1939 value, or OGGM run is permitted here.
