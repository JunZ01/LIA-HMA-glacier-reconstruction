# 06 Debris-Cover Data Decision

Status: **confirmed primary product, thresholds, classifications, and cross-product uncertainty**

Decision date: 2026-07-19

## 1. Dataset Decision

Use **Scherler, Wulf and Gorelick (2018), Supraglacial Debris Cover v1.0**
as the primary debris dataset.

- Dataset DOI: `10.5880/GFZ.3.3.2018.005`
- Article DOI: `10.1029/2018GL080158`
- RGI basis: RGI 6.0
- Primary sensor/algorithm: `LS8_2013-2017_RATIO`
- Observation period: 2013-2017
- Native pixel size: 30 m
- Primary fields: `RGIId`, `DC_Area`, `DC_BgnDate`, `DC_EndDate`,
  `DC_CTSmean`

The primary algorithm is `RATIO`, not because it is assumed to be universally
best, but because the independent Herreid and Pellicciotti comparison explicitly
uses the Scherler Landsat ratio product with `DC_CTSmean >= 10`. This provides a
registered cross-product validation domain.

Retain Scherler `LS8_2013-2017_FDC` as the within-dataset algorithm scenario.
Use **Herreid and Pellicciotti (2020) v1.0**, DOI
`10.5281/zenodo.3866466`, as the independent updated-map comparison.

## 2. Local Acquisition

All files are on the `E:` project drive. No project data or cache is stored on
`C:`.

Scherler source archives support HTTP byte ranges. Only the nine `.dbf`, `.shp`,
and `.shx` members for RGI regions 13, 14, and 15 were downloaded from each
global product archive. Each member passed its ZIP CRC32, uncompressed-size,
and local SHA-256 checks.

| Product | Network bytes selected | Uncompressed target bytes | Role |
|---|---:|---:|---|
| LS8_RATIO | 61,451,390 | 107,602,316 | Primary classification |
| LS8_FDC | 78,397,638 | 134,849,202 | Algorithm sensitivity |

Herreid's 1.37 GB outer archive contains nested data archives. The acquisition
used resumable 16 MiB byte ranges to extract only `S3.zip`:

- compressed outer member: 354,737,493 bytes;
- verified `S3.zip`: 356,230,205 bytes;
- extracted HMA comparison components: 96;
- full outer-archive MD5 registered by Zenodo:
  `b85422798d80fa0c04d3c0f21284c1a2`.

The Herreid S3 layers define Scherler false positives, false negatives, and true
positives over the Herreid coverage domain and Scherler pixels with
`DC_CTSmean >= 10`.

## 3. Frozen Classification Rule

For each RGI glacier:

```text
debris_fraction = DC_Area_m2 / (RGI_Area_km2 * 1e6)
```

Use these fixed classes:

| Debris fraction | Class |
|---:|---|
| `< 0.01` | `zero-or-trace` |
| `0.01 to < 0.10` | `minor` |
| `0.10 to < 0.25` | `debris-covered` |
| `>= 0.25` | `heavy-debris` |

The binary Stage 5 threshold is **10%**. A mapped Ratio glacier must also have
`DC_CTSmean >= 10` to receive a confident binary classification. Mapped
glaciers below that count are retained as `weak-observation`; their mapped area
is not silently changed to zero.

For a 245-system aggregate, member debris areas and glacier areas are summed
before taking the fraction. The system is `weak-observation` only if debris
mapped from low-count members is at least 1% of total system area. This prevents
a tiny uncertain tributary from invalidating a large, well-observed system.

An RGI ID absent from a Scherler debris shapefile is registered as
`zero-detected`, not as missing regional coverage. The product documentation
states global RGI 6.0 coverage; the downloaded shapefiles contain mapped
debris-area features rather than a row for every zero-debris glacier.

## 4. Target Results

### 1,205 modern RGI glaciers

- Ratio debris detected: 1,065
- zero detected: 140
- low observation count: 68
- raw debris fraction `>=10%`: 408
- raw debris fraction `>=25%`: 159
- confident debris-covered: 385
- weak-observation classification: 68

FDC and Ratio disagree on the four-bin class for 257 glaciers. The median
absolute debris-fraction difference is 0.0109; the 90th percentile is 0.0691.
This is large enough to retain FDC as an explicit algorithm scenario.

### 245 LIA systems

- raw Ratio debris fraction `>=10%`: 168
- raw Ratio debris fraction `>=25%`: 72
- confident debris-covered: 165
- weak-observation: 4
- FDC/Ratio four-bin class disagreement: 21 systems

### Herreid versus Scherler over the 1,205 targets

| Comparison component | Area (km2) |
|---|---:|
| Scherler false positive | 181.649 |
| Scherler false negative | 207.090 |
| True positive | 1122.756 |
| Union | 1511.495 |

- Jaccard agreement: 0.7428
- disagreement fraction of the union: 0.2572
- Herreid-minus-Scherler mapped debris: +25.441 km2
- target glaciers with a non-zero comparison union: 541

These quantities are comparison-domain statistics, not a new calibration
likelihood. They demonstrate that debris-map choice is a material structural
uncertainty.

## 5. Workflow Use

Debris data are not allowed to rewrite a clean-ice `melt_f` after Stage 2.

- Stage 1/2: debris class may flag weak MB identifiability, but LIA or terminal
  residuals cannot use it to tune annual-scale parameters.
- Stage 4: stratify terminal residuals by the frozen debris class.
- Stage 5 beta loop: compare no-debris and debris-aware structural scenarios.
- Product uncertainty: rerun the classification or structural scenario with
  FDC and/or the Herreid comparison, preserving the source-product stamp.

## 6. Registered Artifacts

- `data_raw/debris/scherler_2018/v1.0/LS8_2013-2017_RATIO/download_registry.json`
- `data_raw/debris/scherler_2018/v1.0/LS8_2013-2017_FDC/download_registry.json`
- `data_raw/debris/herreid_pellicciotti_2020/v1.0/download_registry.json`
- `outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/debris_work/debris_glacier_registry_1205.csv`
- `outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/debris_work/debris_system_registry_245.csv`
- `outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/debris_work/debris_summary.json`

The source shapefiles do not include `.prj` files, but inherit RGI 6.0 WGS84
geometry. This decision used their registered area attributes for the primary
fractions. Any future polygon operation must assign EPSG:4326 explicitly and
record that assumption. Herreid comparison layers include ESRI:102025 metadata
and were intersected with the exact 1,205 target outlines in that equal-area
CRS.
