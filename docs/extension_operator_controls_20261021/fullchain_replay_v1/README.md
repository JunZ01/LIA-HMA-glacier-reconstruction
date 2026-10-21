# Full-chain modern replay, three frozen controls (2026-10-21)

Produced by `replay_modern_fullchain_20261021.py` from the `fullchain_inputs_<uid>.npz`
exports only. Self-consistency replay against an OGGM reference field; not validation,
not a temporal-transfer test; no LIA reconstruction, climate or dynamic run.

## Harness check (step 0)

The legacy tau calibration was replayed with the production `_evaluate`, the 1-D
inversion bed, native grid and 132 m exclusion. All three archived values are
reproduced exactly (`legacy_tau_harness_check.csv`): 84.217577, 133.608663,
95.27864 kPa, with identical RMSE and point counts. The run stops if this fails.

## Chain (steps 1-2)

Saved LIA grid; unified bed B = topo - H_ref, bilinear on projected modern OGGM
flowlines; explicit connectors (alpha = beta = 0, avoid existing cells); 16x
refinement; production constraint collection (verbatim function); 100 m raster
exclusion; legacy harmonic thickness and operator A. Factors: tau source (legacy,
refit on this chain with the legacy loss and search), outlet (H0 production, Href
modern DEM surface via the new optional `outlet_surface_m`), operator.

Departures from production: raster instead of vector distance for the 100 m
exclusion; modern footprint and modern flowlines (node spacing 122 m) instead of
LIA ones. Operator A receives the same constraint thickness as the legacy operator
(S_cell = B_cell + H); the bilinear-versus-cell bed mismatch this absorbs is
recorded (RMSE 1.4-2.1 m, max 6.6-10.9 m).

## Results, outlet H0 (`fullchain_summary.csv`)

| system | tau source | tau kPa | flowline RMSE m | legacy op. volume % | A volume % |
|---|---|---|---|---|---|
| 3632 | legacy | 84.2 | 20.7 | -38.7 | -5.2 |
| 3632 | refit | 92.9 | 16.4 | -32.9 | +0.6 |
| 3131 | legacy | 133.6 | 61.7 | -36.1 | -6.0 |
| 3131 | refit | 105.9 | 59.0 | -45.5 | -15.1 |
| 2906 | legacy | 95.3 | 22.1 | -47.4 | -10.5 |
| 2906 | refit | 107.2 | 20.3 | -41.9 | -5.3 |

- Outlet convention changes A volume by at most 0.07 percentage points.
- The legacy operator stays 33-47 % low at every tau that fits the flowlines.
- A volume moves about 7-13 percentage points per 20 kPa of tau
  (`tau_curves.csv`); tau identification, not the operator, now dominates A error.
- 3131: the loss is flat between 100 and 140 kPa (0.79-0.84) and the best flowline
  RMSE is 59 m, against 21 m on the 1-D bed. Along the 3131 flowlines the 1-D and
  2-D reference thickness differ by 55 m RMSE (correlation 0.80) and the beds by up
  to 217 m; on trunk segment fl4:seg2 the 2-D bed is on average 135 m below the 1-D
  bed. A single-tau plastic profile cannot reproduce the 2-D field there, so tau is
  weakly identified and the refit is not a reliable transfer value for this system.
