# Round 156 Regional JJA Low-Frequency Temperature OOF Contract

## Question

The archived all-HMA JJA proxy common mode failed the purged low-frequency
test, while later annual low-rank residual fields recovered regional mean skill
but failed spatial gates. Does one intermediate change, splitting the formally
documented JJA temperature network into fixed west, central, and east HMA
regions, recover transferable low-frequency regional temperature information?

## Frozen Design

- Reuse the Round 16 source-native observations, locations, Berkeley Earth and
  CRU JJA targets, years 1940-2000, centered 11-year mean, and three purged
  outer blocks: 1940-1959, 1960-1979, and 1980-2000.
- Retain only records whose formal season is JJA. Define regions before fitting
  as west `<80 E`, central `80-90 E`, and east `>=90 E`.
- Candidate for each region: the equal-weight mean of training-standardized
  proxy records inside that region, linearly calibrated to the mean JJA target
  at the unique proxy sites in that region.
- Global control for the same regional target: the identical operator using all
  HMA JJA proxy records. The absolute control is the fold-training target mean.
- Standardization, feature eligibility, smoothing support, intercept, and slope
  are fit only from windows that do not overlap the held-out block. No target
  value from a held-out window enters fitting.
- No proxy deletion, reliability weighting, longitude-boundary search,
  alternate season, smoothing-width search, regularization, product weighting,
  trend term, EnKF, localization, or reconstruction-field generation is
  allowed.

## Frozen Gates

All gates must pass to advance a three-region JJA operator:

1. Median pooled RMSE improvement over the all-HMA common-mode control is at
   least 5%, the worst region-product regression is no worse than 5%, and at
   least 75% of block-region-product paths improve.
2. Median pooled skill over the fold-training-mean control is at least 5%, the
   worst pooled skill is no worse than -5%, and at least 75% of block paths
   have positive absolute skill.
3. Median pooled regional correlation is at least 0.30 and no pooled
   region-product correlation is negative.
4. Median candidate-versus-global improvement is nonnegative in both the first
   and third outer blocks, and at least 75% of region-product branches retain
   positive fitted proxy slopes in all three folds.

Region-specific results are attribution only. A partial pass cannot be used to
assemble a full HMA reconstruction post hoc.

## Boundary

Only 1940-2000 values may be read. The locked 1921-1939 period, 1901-1920
trajectory, external paleo benchmarks, Jionglacuo, EnKF production outputs, and
OGGM are excluded. Original sources remain read-only and all outputs are written
to a new independent result directory.
