# Round 146 New Temperature Proxy Modern OOF V1

## Objective

Test whether the two genuinely new chronologies from Round 145 contain transferable modern temperature information before either is used in a paleo reconstruction.

## Frozen design

- Haizi: use the raw `std` chronology, positive response, and the paper's mean-annual-temperature target at Daocheng station coordinates 29.03 N, 100.18 E. The primary article defines mean annual temperature; this supersedes the preliminary inventory label of previous-October-current-September.
- Hengduan: use the raw `SsfStb` regional chronology, positive response, and May-September mean temperature averaged across the five published sampling sites: DLL, FGH, KQH, XYS, and YPL.
- Instrumental products: Berkeley Earth and CRU TS4.04, 1940-2000 only.
- Outer folds: 1940-1951, 1952-1963, 1964-1975, 1976-1987, and 1988-2000. Fit an intercept and one proxy slope on the other four blocks and predict the held-out block.
- No smoothing, proxy selection, sign constraint, season search, site search, regularization, or hyperparameter is used.

## Candidate gates

Each candidate advances separately only if, for both products: pooled OOF correlation is at least 0.30; OOF skill versus training-fold climatology is positive; at least four of five fold correlations are positive; all five fitted slopes are positive; and Berkeley/CRU OOF prediction correlation is at least 0.80.

The sealed Jionglacuo record remains unopened. It may be used only after at least one candidate passes every modern gate. No 1921-1939 value, paleo validation value, EnKF, or OGGM use is permitted.
