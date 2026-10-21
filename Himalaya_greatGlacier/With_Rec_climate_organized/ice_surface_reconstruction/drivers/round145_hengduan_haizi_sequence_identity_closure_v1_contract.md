# Round 145 Hengduan/Haizi Sequence Identity Closure V1

## Objective

Determine whether the newly released Hengduan `SsfStb` and Haizi `std` chronologies are genuinely new proxy sequences relative to the complete archived proxy6 and proxy9 networks.

## Frozen rules

- Use only the Round 144 V3 allowed-period exports: 1000-1920 and 1940-2000.
- Compare each raw chronology against every proxy6 and proxy9 observation on exact common years.
- An exact affine alias requires at least 20 common values, absolute correlation at least 0.999999, and maximum residual after intercept-plus-slope fit no greater than 1e-8.
- A near-exact standardized sequence requires at least 20 common values, absolute correlation at least 0.995, and normalized affine RMSE no greater than 0.05.
- Also test each raw chronology against its same-workbook published temperature reconstruction. A derived reconstruction is internal source evidence and can never validate the source chronology.
- Do not open the sealed Jionglacuo file or any 1921-1939 value. Do not fit a PSM, reconstruct climate, run EnKF, or run OGGM.

Only chronology candidates with no exact alias may advance to source-site/measurement closure and modern blocked validation.
