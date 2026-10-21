# Round 274 independent Hengduan cluster scPDSI OOF contract

- Use only the nine new 2018 chronologies admitted by Round 273: northern
  HLS/JYC/YZC and southern BLX/CYS/FNPTS/GDCTS/GDCYS/LGQ.
- Read chronology values only for 1940-2000 with the tested year-first parser.
- Build the exact published hydrological-year target as the area-weighted mean
  monthly CRU-TS 3.26 scPDSI from previous May through current April.
- Northern target box: 28-32 N, 96-103 E. Southern target box: 25-28 N,
  96-102 E. Evaluate 1954-2000 only.
- Use five fixed blocked folds: 1954-1962, 1963-1971, 1972-1980,
  1981-1989, 1990-2000.
- Within every fold and cluster, estimate proxy mean/scale, response direction,
  equal-weight composite, and linear calibration from training years only.
- Advance only if both regional pooled OOF correlations are at least 0.30,
  both regional skills are positive, each region has at least four positive
  fold correlations, all ten fitted composite slopes are positive, the
  north-minus-south OOF correlation is at least 0.20 with positive skill, and
  observed north/south target correlation is below 0.90.
- Do not read the published paleo reconstruction, use old ITRDB chronologies,
  run EnKF, modify production climate, or run OGGM.

