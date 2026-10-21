# 03 Velocity Data Decision

Status: **confirmed product/QC policy; 1,205-member gate extraction and Farinotti turnover scan complete**

Decision date: 2026-07-19

## 1. Decision

Use the NASA MEaSUREs ITS_LIVE **v2.1 RGI14A static 2014-2024 mosaic** as the primary modern surface-velocity product.

- Resolution: 120 m
- Projection: ESRI:102027
- Primary observable: median surface-speed magnitude in a fixed cross-glacier gate near RGI60 `Zmed`
- Direction diagnostic: project `vx` and `vy` onto the downstream inversion-flowline tangent
- Millan 2022 velocity: cross-check or alternative data scenario only; it must not be combined as an independent likelihood with ITS_LIVE when the errors are not independent
- Observation reuse policy: velocity is used only for the Stage 1 turnover constraint. It is not reused to calibrate `glen_a` in Stage 0.

Official source references:

- `https://its-live.jpl.nasa.gov/`
- `https://its-live-data.s3.amazonaws.com/velocity_mosaic/v2.1/static/ITS_LIVE_velocity_120m_RGI14A_0000_V02.1_{layer}.tif`
- `s3://its-live-data/velocity_mosaic/v2.1/static/ITS_LIVE_velocity_120m_RGI14A_0000_V02.1.nc`

## 2. Frozen Download Scope

The data scope follows the complete LIA-to-modern membership manifest, not only the 245 main RGI glaciers.

| Item | Count | Status |
|---|---:|---|
| LIA systems | 245 | frozen |
| Unique RGI60 glaciers downloaded | 1,205 | frozen |
| Scientific members retained with QC flags | 1,205 | all retained |
| RGI region 13 | 23 | downloaded |
| RGI region 14 | 406 | downloaded |
| RGI region 15 | 776 | downloaded |

`UniqueID=2692` declares 25 modern members, while spatial recovery yields 26.
All 26 remain in the workflow with the review flag. No candidate is silently
discarded at the data or turnover stage.

All project data, clips, hashes, temporary files, and outputs are on `E:`. Programs installed on other drives may be executed, but they are not project-data destinations.

## 3. Download and Integrity Result

The target extent is the union of all 1,205 RGI outlines with a 5 km buffer.

- GDAL source window: `5211 8275 14346 8184`
- Target raster size: `14346 x 8184`
- Target bounds: `-1698367.5, -230632.5, 23152.5, 751447.5`
- Pixel size: `120 x -120 m`
- Layers: `v`, `vx`, `vy`, `v_error`, `vx_error`, `vy_error`, `count`, `outlier_percent`

All eight target layers are readable, have identical dimensions, origin, transform, and CRS, and have non-zero GDAL checksums. SHA-256 values and acquisition paths are recorded in `data_raw/velocity/its_live/v2.1/download_registry.json`.

The first remote `count` extraction was interrupted and produced a raster with zero valid pixels. It was moved to `static/failed_downloads` and is excluded from analysis. The valid `count` clip was regenerated from the complete local source COG.

## 4. Whole-Outline Coverage Audit

All 1,205 target outlines were sampled against all eight aligned layers.

| Check | Result |
|---|---:|
| RGI outlines audited | 1,205 |
| Outlines with no valid `v` pixel | 0 |
| Outlines with less than 50% `v` coverage | 0 |
| Median `v` coverage | 100% |
| Outlines requiring all-touched fallback | 1 |
| Glaciers smaller than 1 km2 | 641 |

Coverage is therefore not the limiting factor. Slow motion, directional coherence, and representation at the gate are the limiting factors.

Whole-outline speed is a QC diagnostic only. It is **not** the turnover observation because it mixes accumulation, ablation, tributary, and terminus regimes.

## 5. Gate Sampling Operator

The gate operator was tested read-only on the 245 existing modern OGGM glacier directories. These directories match the 245 manifest `MainRGI` records exactly.

1. Select the main inversion flowline: the highest-order flowline with no downstream recipient.
2. Use RGI60 `Zmed` as the fixed geometric anchor and select the closest main-flowline point.
3. Construct a gate that is 480 m long in the along-flow direction and one local inversion-flowline width in the cross-flow direction.
4. Clip the gate to the RGI outline.
5. Use the median ITS_LIVE speed magnitude `v` as the calibration observation.
6. Retain P25 and P75 as the robust spatial observation interval.
7. Project `vx/vy` onto the downstream tangent only as a direction-coherence diagnostic.

`Zmed` is an explicit ELA proxy, not a measured ELA and not a modeled MB zero crossing. This avoids using the MB parameters being calibrated to choose their own velocity observation. If an independent snowline/ELA product is added later, it may replace the anchor as a documented data scenario.

The initial 245-main-glacier gate audit produced:

| Check | Result |
|---|---:|
| Successful gate audits | 245 / 245 |
| Gates with no valid velocity | 0 |
| Median gate pixels | 39 |
| Minimum gate pixels | 6 |
| Median gate speed | 11.74 m/yr |
| Median downstream alignment ratio | 0.923 |
| Gates with quantized-zero formal error median | 210 |

After Stage 0 initialized all modern members, the same operator completed all
1,205 gates with no extraction failures. The full-member result is:

| Class | Count |
|---|---:|
| usable | 304 |
| weak-slow | 313 |
| weak-direction-or-sampling | 346 |
| unusable | 242 |

For the 191 Stage 0 MB-success systems (950 members), 174 systems have at least
one evaluable `usable` or `weak-slow` gate. Seventeen have no evaluable strong
gate and are routed to weak identifiability.

## 6. Error and QC Policy

The error layers are unsigned integer rasters with 1 m/yr steps and no scale factor. For 210 of 245 audited gates, the formal-error median is zero. This is quantization of sub-unit formal errors, not proof of zero uncertainty.

Use the following observation representation:

- Central value: median gate `v`
- Robust interval: gate P25 to P75
- Formal-error floor: 0.5 m/yr
- Effective interval for calibration: `[max(0, P25 - formal_floor), P75 + formal_floor]`
- High `outlier_percent`: uncertainty-broadening flag, not automatic rejection

Do not convert the gate IQR into an independent-pixel standard error. The cross-glacier velocity gradient is real structure, not random measurement noise.

Frozen preliminary QC classes for the 245 available gates:

| Class | Rule | Count | Treatment |
|---|---|---:|---|
| usable | at least 10 pixels; coverage at least 0.8; median alignment at least 0.5; positive downstream fraction at least 0.75; median speed at least 5 m/yr | 172 | normal turnover likelihood |
| weak-slow | direction and sampling pass, but median speed below 5 m/yr | 38 | retain with broad posterior |
| weak-direction-or-sampling | low pixel count, low coverage, weak alignment, or low positive-downstream fraction | 23 | retain only as weak constraint; inspect geometry |
| unusable | no velocity, coverage below 0.5, or non-positive downstream median/alignment | 12 | exclude from turnover likelihood and record diagnostic reason |

Additional non-exclusive flags:

- Low pixel count: 2
- Weak direction: 34
- Reversed/non-positive direction: 12
- Slow speed: 65
- Median outlier fraction above 20%: 26

Rank-deficient or weak-velocity glaciers are degraded in precision, not removed from the scientific inventory. Their wider posterior must propagate into the climate-sensitivity analysis.

## 7. Independence and Provenance

Velocity carries information about annual turnover and instantaneous flow, but this workflow spends that information once:

- Stage 0 `glen_a`: thickness product only, interpreted as effective rheology
- Stage 1 turnover: ITS_LIVE gate velocity
- Millan velocity: cross-check/scenario axis only

Every derived gate record retains the RGI ID, LIA system membership, product/version, gate definition, formal-error floor, QC flags, and source paths. Later parameter-source stamps must reference this decision and the exact gate-QC row.

The old `F:/oggm_related/Rec_climate/Rec_rgi_calibration` directories were read only to audit gate feasibility. Their existing MB parameters were not used to define the gate or the new velocity observation.

## 8. Decision Outcome

The velocity data decision is closed at the product and policy level:

- primary product confirmed;
- complete 1,205-glacier raster scope confirmed;
- observation-reuse policy confirmed;
- gate operator confirmed;
- error interpretation and QC policy confirmed.

The full-member validation hook is complete. A missing valid gate remains a
weak-identifiability diagnostic, not permission to retune `glen_a` or use LIA
geometry in the calibration objective.

## 9. Artifacts

- `data_raw/velocity/its_live/v2.1/download_registry.json`
- `outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/velocity_work/velocity_glacier_raw_stats.csv`
- `outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/velocity_work/velocity_raw_summary.json`
- `outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/velocity_work/velocity_gate_245_stats.csv`
- `outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/velocity_work/velocity_gate_245_summary.json`
- `outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/velocity_work/velocity_gate_1205_stats.csv`
- `outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/velocity_work/velocity_gate_1205_summary.json`
- `outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/velocity_work/velocity_baseline_farinotti_stats.csv`
- `outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/velocity_work/velocity_baseline_farinotti_summary.json`
- `outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/velocity_work/velocity_baseline_millan_stats.csv`
- `outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/velocity_work/velocity_baseline_millan_summary.json`
- `outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/03_velocity_coverage_qc.xlsx`

## 10. Farinotti turnover scan outcome

The isolated Farinotti `fs=0` full scan evaluated 191 systems, 950 members,
2,518 modern-MB-equivalent candidates, and 6,201 candidate-gate residuals.
Stage 0 key-file hashes were unchanged. Nine strong-gate members had a zero
fixed Farinotti section and remain explicit operator failures; no system failed.

At each system's minimum interval-loss candidate:

| Diagnostic | Systems |
|---|---:|
| all strong gates inside the observation interval | 22 |
| some strong gates inside | 42 |
| no strong gate inside; model still too fast at the low-turnover edge | 108 |
| mixed/interior unresolved residual | 2 |
| no evaluable strong gate | 17 |

Of the 174 velocity-evaluable systems, 166 select the lowest-turnover candidate
edge: 138 at the registered precipitation lower bound and 28 at the physical
`melt_f=1.5` bound. The median system-level observed/modeled speed ratio after
this refinement is 0.231. These are structural/turnover diagnostics, not
permission to widen priors and not frozen point posteriors.

## 11. Millan structural scenario

The Millan `fs=0` scan completed 186 systems, 932 members, 2,452 candidates,
and 6,149 candidate-gate residuals. Nine strong-gate members had zero Millan
section; no system failed. The diagnostic classes are:

| Diagnostic | Systems |
|---|---:|
| all strong gates inside the observation interval | 28 |
| some strong gates inside | 49 |
| no strong gate inside; model still too fast at the low-turnover edge | 94 |
| mixed/interior unresolved residual | 2 |
| no evaluable strong gate | 13 |

On the 186 common systems, 17 categorical reachability results improve, 155
are unchanged, and one changes from partial to no gate inside while its total
interval loss still decreases. Median observed/modeled speed increases from
0.234 under Farinotti to 0.293 under Millan. The comparison is conditional and
cannot select Millan as truth because Millan thickness is velocity-driven.

Both structures remain dominated by low-turnover boundary solutions. These
rows must be routed as parameter-bound or structural-incompatibility
diagnostics before any Stage 2 parameter artifact is frozen.
