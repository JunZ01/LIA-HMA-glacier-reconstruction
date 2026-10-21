# HMA Precipitation Provenance, Unit, and Coverage Audit v1

This independent no-OGGM audit determines whether the archived precipitation
reconstruction is ready for a controlled EnKF replay.

The audit answers four questions without modifying any source file:

1. What precipitation observations are actually present in the combined
   0.5-degree NetCDF, and over which years and cells are they finite?
2. What configuration, fixed prior-member indices, and posterior summaries are
   preserved in the archived precipitation experiment?
3. What precipitation source, conversion, bias correction, grid, and
   nonpositive-value treatment are present in the surviving prior loaders?
4. Is a frozen 336 x 848 precipitation prior pool available for a member-level
   replay, or is a prior-recovery preflight still required?

`rec_ref` is inventoried only as a comparison product and is never treated as
truth. The compact `prior_fields.mat` is treated as a later evaluation export,
not as the archived member pool. CRU and GPCP are not admitted to validation
until their provenance and time coverage pass the audit.

Original MAT, NetCDF, MATLAB, and archived result files are read-only. Outputs
are written only to the independent result directory
`validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_provenance_audit_v1/`.

