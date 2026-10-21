# Round 278 Baiyu-Haizi Anchor Identity Audit Contract

## Purpose

Acquire byte-preserved official Baiyu 2022 and Haizi 2024 reconstruction
sources into a new result directory, audit their location, sample lineage, and
allowed-year sequences against the current proxy6/proxy9 network, and decide
whether each record is admissible as an independent eastern-HMA temperature
anchor.

## Frozen safety boundary

- Source MAT, HDF5, XLSX, RAR, HTML, CSV, and project files are read-only.
- Historical worksheet rows are decoded year first.
- Only 1850-1920, 1940-1950, and each paper's documented modern calibration
  period may expose reconstruction or chronology values.
- Years 1921-1939 are rejected before the value cell or HDF5 value element is
  accessed.
- A synthetic XLSX and HDF5 poison test must pass before official data are
  parsed.
- No image digitization, unofficial numerical file, production climate edit,
  EnKF run, or OGGM run is permitted.

## Frozen identity rules

- Exact affine alias: at least 20 allowed common years, absolute raw
  correlation >= 0.999999999, and maximum affine residual <= 1e-10.
- Near alias: at least 30 allowed common years, absolute raw correlation >=
  0.995, and affine RMSE <= 10% of the candidate anchor standard deviation.
- Location proximity alone cannot establish identity. Distances of 35 km and
  75 km are reported as audit bands, not deletion thresholds.
- proxy6 records are audited through their frozen one-to-one evidence-group
  mapping to the 297 proxy9 records. The audit must retain both network IDs.

## Outputs

- byte-preserved `raw_unopened` source copies and SHA256 manifest;
- allowed-period reconstruction curves and plot;
- sequence and location identity tables;
- source/sample/season/unit/calibration metadata table;
- poison-test result, machine-readable summary, and `FINAL_RESULT.md`.
