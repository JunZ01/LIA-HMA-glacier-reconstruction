# Climate reconstruction improvement localization pilot v1

## Question

Test whether the archived Gaspari-Cohn covariance-localization option can
improve twentieth-century low-frequency temperature evolution without
degrading held-out proxy performance.

## Frozen design

- proxy6 / EnKF / archived prior experiment 1;
- archived proxy holdout repeat 1;
- exact RegEM-completed residual matrix;
- inflation fixed at 1;
- archived PSM coefficients and precipitation auxiliary field;
- years 1000-2000;
- no glacier trajectory, geometry, length, area, volume, or mass balance in
  calibration or selection.

The branches are no localization and Gaspari-Cohn radii of 3000, 4500, and
6000 km. The historical radius CSV is boundary evidence only; it is not reused
as an experiment result because its script depended on interactive workspace
state and did not jointly constrain held-out proxy skill.

## Acceptance

A finite-radius branch advances only if all declared gates in
`experiment_config.json` pass. Proxy and instrumental metrics are reported
separately; there is no weighted score that can hide failure in one evidence
class. The 3584 pixel is a named diagnostic, not the sole selection target.

## Isolation

All original D- and E-drive sources are read only. Prepared inputs, logs,
reconstruction means, figures, and conclusions are written only below the new
project result directory
`validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_localization_pilot_v1/`.
