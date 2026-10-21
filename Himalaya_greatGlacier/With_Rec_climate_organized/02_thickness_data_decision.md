# Thickness data decision

Status: confirmed for the thickness decision only.

The full constraint inventory is not frozen by this document. Velocity,
precipitation prior, LIA timing, and debris cover remain open decisions.

## Scope

- 245 LIA systems.
- 1,205 unique RGI60 glaciers in the conservative download superset.
- 1,204 scientific members expected from the source `RGICount` fields.
- The one-member excess for LIA `UniqueID=2692` remains review-only and must
  not enter scientific system aggregation until resolved.
- RGI60 regional counts are 23 in region 13, 406 in region 14, and 776 in
  region 15.

## Products and roles

### Farinotti 2019

- Dataset DOI: `10.3929/ethz-b-000315707`.
- License: `CC-BY-NC-SA-4.0`.
- All 1,205 target glacier TIFFs were found and extracted; none are missing.
- Official MD5 checksums for the RGI 13, 14, and 15 archives match exactly.
- Decision role: primary operational thickness product for the first
  effective-rheology calibration scenario.

This role does not make Farinotti an independent direct observation. The
derived `glen_a` remains an effective parameter conditional on a consensus
model product.

### Millan 2022

- Dataset DOI: `10.6096/1007`.
- Paper DOI: `10.1038/s41561-021-00885-z`.
- Product version: `v1.1`.
- Resolution: 50 m.
- Variables used: `THICKNESS` and `ERRTHICKNESS`.
- The DataCite record and product page do not name a license. The product page
  states that the data are freely open for download; provenance records this
  access statement without inventing a license.
- Decision role: required structural alternative and cross-check, never an
  additional independent term in the same likelihood.

Millan thickness is velocity-driven. Its errors are therefore not independent
of a later velocity constraint.

## Processing and QC

- Farinotti glacier support is defined by positive thickness pixels.
- Millan pixels are included when their centers are inside the audited RGI
  polygon and thickness is positive.
- Volume is `sum(thickness_m * projected_pixel_area_m2) / 1e9` in km3.
- Millan candidate tiles are selected by positive support and closest raster
  area match to the RGI area. No glacier had multiple positive candidates.
- All 1,205 glaciers have paired Farinotti and Millan statistics.
- Median raster-area absolute error is 0.17% for Farinotti and 0.67% for
  Millan.
- Millan raster-area error exceeds 5% for 121 glaciers; Farinotti exceeds 5%
  for 5 glaciers.
- Comparisons are assigned low interpretation weight when glacier area is
  below 1 km2 or Millan area error exceeds 5%. This flags 641 glaciers.
- `ERRTHICKNESS` integration is retained only as a diagnostic proxy. Pixel
  errors are not treated as independent random errors.

## Paired results

- Farinotti total volume: 642.50 km3.
- Millan total volume: 748.77 km3.
- Millan total volume is 16.54% higher.
- Median glacier mean-thickness difference is +30.65% (Millan minus
  Farinotti).
- Median LIA-system volume difference is +18.12%.
- Region-total Millan volume differences are +21.43% for RGI 13, +16.27% for
  RGI 14, and +16.36% for RGI 15.
- For glaciers at least 10 km2, total-volume differences remain about
  15%; the result is therefore not only a small-glacier rasterization effect.
- Relative differences are larger for glaciers below 1 km2, where the median
  mean-thickness difference is +43.70% and the 50 m grid has a larger area
  mismatch.

## Decision

1. Use Farinotti 2019 as the primary operational product in the first
   thickness-conditioned effective-rheology run.
2. Run Millan 2022 v1.1 as a separate structural scenario.
3. Treat the product choice as an explicit structure axis. The observed
   16-18% aggregate volume difference is large enough that it cannot be
   hidden inside one `glen_a` point estimate.
4. Do not combine Farinotti and Millan as if they were two independent direct
   observations.
5. Carry low-weight and area-error flags into uncertainty reporting, with
   special care for sub-km2 glaciers.

## Auditable outputs

- Raw archive provenance and hashes:
  `data_raw/thickness/download_registry.json`
- Target manifest:
  `outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/00_target_download_manifest.xlsx`
- Product comparison workbook:
  `outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/02_thickness_product_comparison.xlsx`
- Machine-readable intermediate tables:
  `outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/thickness_work/`

## Next inventory decision

The velocity decision and the first two fixed-thickness turnover scans are now
complete. On 468 common strong gates, the Millan-conditioned modeled speed is
0.829 times the Farinotti-conditioned speed at the median; 420 gates become
slower and 48 become faster because thickness is redistributed spatially.

For the 186 common systems, Millan improves the categorical velocity
reachability of 17 systems, leaves 155 unchanged, and worsens one categorical
reachability result. Full compatibility rises from 22 to 28 systems and the
median system observed/modeled speed ratio rises from 0.234 to 0.293.

This confirms that thickness-product choice materially affects turnover, but
does not identify a preferred truth. Millan thickness is velocity-driven and
therefore dependent on the later velocity diagnostic. The comparison remains
a structural sensitivity axis, not a product-selection likelihood.
