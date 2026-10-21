# Round 234 Contract: Hengduan Individual-Site Modern OOF

## Purpose

Test whether the weak five-site Hengduan regional result hides one transferable
individual-site warm-season temperature signal. This is a discovery screen,
not a reconstruction or a replacement for the published RCSigFree chronology.

## Frozen inputs and targets

- Raw sites: DLL, FGH, KQH, XYS, and YPL from Mendeley dataset
  `10.17632/889n6ccx4z.1`.
- Build modern-only exploratory STD chronologies with dplpy 0.1.5,
  ModNegEx residual detrending, Tukey biweight mean, and no prewhitening.
- Decode ring-width values only for 1940--2000. Skip 1921--1939 and all paleo
  values before numeric conversion.
- Target: local May--September mean temperature at each published site.
- Products: Berkeley Earth and CRU TS4.04.
- Target years: 1940--1999.
- Five blocked 12-year folds: 1940--1951, 1952--1963, 1964--1975,
  1976--1987, and 1988--1999.

## Frozen site gates

- Complete 1940--2000 chronology and minimum modern sample depth at least six.
- OOF correlation at least 0.30 for both products.
- Positive skill against training-fold climatology for both products.
- At least four of five fold correlations positive for both products.
- All ten fitted slopes positive.
- Positive early and late split correlations for both products.
- Berkeley/CRU OOF predictions correlate at least 0.80.

## Family-wise discovery gate

Use the maximum across five sites of the minimum dual-product OOF correlation,
with a score of zero when either product has a non-positive fitted slope. Apply
all 59 non-zero circular shifts jointly to the five chronologies, preserving
their cross-site dependence. The empirical family-wise p-value must be at most
0.05 in addition to one site passing every frozen site gate.

## Interpretation boundary

A passing site advances only to source-lineage and exact chronology-method
validation. The dplpy chronology must not be described as the published
RCSigFree chronology and must not enter a PSM, EnKF, production reconstruction,
or glacier simulation.

## Hard boundaries

- Do not inspect any paleo trajectory.
- Do not decode numeric values for 1921--1939.
- Do not open or hash Jionglacuo.
- Do not tune season, lag, sign, detrending, folds, station weights, products,
  family statistic, or gates after seeing results.
- Preserve all source-file hashes.

