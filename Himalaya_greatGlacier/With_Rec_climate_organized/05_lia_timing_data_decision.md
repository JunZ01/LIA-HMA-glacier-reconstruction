# 05 LIA Timing Data Decision

Status: **confirmed regional-window decision; glacier-specific dates remain latent**

Decision date: 2026-07-19

## 1. Scope and Boundary

The 245 selected LIA systems contain four source regions, not three. The source
field `region` is retained as the timing stratum. No region is merged merely to
fit a prior workflow description.

LIA timing is used only in Stage 3 to identify the centennial initial state. It
must never enter the Stage 1/2 likelihood and must never rewrite `glen_a`,
`prcp_fac`, `melt_f`, or `temp_bias`.

The available chronology does not identify an exact maximum year for every
outline. A regional probability window is therefore more defensible than
assigning a single date such as 1500 CE to all glaciers.

## 2. Source Geometry Registration

Read-only source:

`E:\3spyderProject\1_LIA_himalaya_oggm\Himalaya_greatGlacier\Him_10_lia_with_large.shp`

The source contains 245 unique `UniqueID` values and 245 unique `MainRGI`
values. Its geometry CRS is ESRI:102025. The main `.shp` SHA-256 is:

`17fa6bda255542e71a1d803d9940436237382147980db35e56a79d7f98d20cef`

Hashes for all five shapefile components are stored in
`lia_timing_summary.json`. The source files were read only and were not copied,
rewritten, or normalized in place.

## 3. Evidence and Interpretation

The timing decision uses four evidence statements:

1. Rowan (2017), DOI `10.1177/0959683616658530`, synthesizes 66 ages younger
   than 2000 years and reports peak Himalayan moraine building mainly during
   1300-1600 CE. The review also emphasizes spatial asynchrony and climate
   conditions favorable to glacier advance around both 1400 and 1800 CE.
2. Yang et al. (2022), DOI `10.5194/tc-16-3739-2022`, resolves eastern
   Himalayan/Bhutan advances in the 1270s, 1470s, 1710s, and 1850s. Most mapped
   maximum extents occur in the early LIA, while a late-LIA tail must remain
   admissible.
3. Qiao and Yi (2017), DOI `10.1016/j.quaint.2016.11.049`, provides a central
   and western Himalayan LIA reconstruction. It supports the spatial strata but
   does not justify a glacier-specific exact maximum year for this inventory.
4. The Hunza Basin reconstruction, DOI `10.1016/j.palaeo.2020.110086`, supports
   retaining Karakoram/Hindu-Kush as a separate stratum. It is not treated as a
   precise date for all 82 Region 4 systems.

The pan-Himalayan synthesis is the common timing anchor. Regional studies alter
the plateau or tail but do not remove the wide outer window. This deliberately
avoids false precision.

## 4. Frozen Regional Windows

| Region | Label | Systems | Core plateau (CE) | Outer admissible window (CE) |
|---:|---|---:|---:|---:|
| 1 | Eastern Himalaya / Bhutan-SE Tibet | 21 | 1250-1550 | 1200-1900 |
| 2 | Central Himalaya | 79 | 1300-1600 | 1200-1900 |
| 3 | Western Himalaya | 63 | 1300-1650 | 1200-1900 |
| 4 | Karakoram / Hindu-Kush | 82 | 1300-1650 | 1200-1900 |

The core plateau is the interval with full timing weight. The outer interval is
an uncertainty envelope, not a claim that all years are equally likely.

## 5. Stage 3 Weighting Operator

For outer bounds `a, d` and core bounds `b, c`, use the trapezoid:

```text
w(t) = 0                         for t < a or t > d
w(t) = (t - a) / (b - a)        for a <= t < b
w(t) = 1                         for b <= t <= c
w(t) = (d - t) / (d - c)        for c < t <= d
```

For each candidate initial state, Stage 3 evaluates the LIA geometry score
throughout the admissible window and retains the maximum weighted score:

```text
score_state = max_t [w(t) * score_geometry(t)]
```

`score_geometry` must be defined separately from timing and must use the
pre-registered area, main-flowline length, and hypsometry covariance. A timing
window cannot compensate for a geometry miss. If no state reaches the geometry
tolerance anywhere inside the outer window, the result is a classified Stage 3
failure, not a reason to retune annual-scale parameters.

## 6. Uncertainty and Re-entry

- `timing_status = regional-window; glacier-date-latent` for all 245 systems.
- New glacier-specific moraine dates may replace the regional prior for that
  glacier only if their sample, calibration, and uncertainty provenance are
  registered.
- Region-level window revisions require a new decision version and Stage 3
  re-entry. Stage 0-2 parameter artifacts remain frozen.
- `UniqueID=2692` retains `crosswalk_status=review-required` because its source
  membership declaration and candidate count disagree. That crosswalk issue is
  independent of the timing decision.

## 7. Registered Artifacts

- `outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/lia_timing_work/lia_timing_registry_245.csv`
- `outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/lia_timing_work/lia_timing_summary.json`

These files close the LIA timing decision for inventory v1. They do not run an
OGGM task or alter any notebook.
