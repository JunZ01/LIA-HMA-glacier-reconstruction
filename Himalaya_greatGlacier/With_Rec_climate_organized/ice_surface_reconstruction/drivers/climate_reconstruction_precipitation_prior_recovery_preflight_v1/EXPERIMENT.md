# HMA Precipitation Prior Recovery Preflight v1

This independent no-OGGM experiment asks whether the surviving precipitation
loader can produce a reproducible `336 x 848` EnKF prior member pool in MATLAB
R2024b.

The experiment freezes:

- model: `cesm_lme010`;
- state: annual precipitation;
- member years: 1001-1848, corresponding to model indices 152-999;
- HMA bounds: 24-51 N and 62-110 E;
- loader: the existing `load_S_720360.m`;
- source precipitation: merged CESM PRECC + PRECL;
- source conversion: m/s to mm/day;
- bias climatology: CRU CL 1.1 precipitation divided by 30;
- nonpositive-value replacement: `gamrnd(1, 0.05, ...)`;
- RNG: `rng(4236, 'twister')` before each loader call.

The loader is called twice with the same RNG state. The two outputs must agree
element by element and have shape `336 x 848`. The first output is frozen as a
new experiment artifact in mm/day.

The later `prior_fields.precipitation` export is compared only over its
1001-1848 overlap after dividing by 30. It is not used as the recovered member
pool. The comparison diagnoses unit and version closure, including zero-mask
agreement and the fraction of values affected by stochastic replacement or
another unarchived runtime difference.

No EnKF update is run. Original MATLAB, MAT, NetCDF, and CRU files are read-only.

