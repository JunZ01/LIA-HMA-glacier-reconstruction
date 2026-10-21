# Round 244 contract: NPL065 published CRU-box target geometry

## Question

How much of the gap between the Round 238 nearest-cell result and the published
NPL065 relationship is caused by using the wrong CRU target geometry?

## Frozen design

- Reuse the Round 238 v4 modern-only chronology without reopening raw rings.
- Preserve 1960-2000, the five blocked folds, regression, and metrics.
- Baseline target: exported nearest-cell CRU TS4.01 DJF Tmin.
- Published-geometry target: cosine-latitude weighted mean of all CRU TS4.01
  half-degree cells whose centers lie in 27.5-28.5 N, 86-87 E, for previous
  December plus current January-February.
- Verify all CRU NetCDF hashes against the Round 208 manifest.
- Report raw chronology-target correlation and blocked OOF skill for both
  geometries.
- Define a material target-geometry effect prospectively as box-minus-point
  OOF correlation at least 0.05.

The published in-sample 1960-2013 correlation is not treated as directly
comparable with this 1960-2000 OOF result. This round changes no chronology
method and cannot reproduce the paper by itself.

Original files are read-only. No 1921-1939 value, paleo proxy value,
Jionglacuo, EnKF, production reconstruction, correction, or OGGM run is
allowed.
