# Temperature Fold-Native OEnKF Increment CV V1

## Question

Does the ordinary-EnKF temperature assimilation increment retain transferable
skill when every PSM coefficient and diagonal residual variance is estimated
without the evaluated time block?

## Data Boundary

- The only instrumental and proxy years read are 1940-2000.
- The 1921-1939 values are never requested from NetCDF or MAT inputs.
- Original files are read-only; all generated files live in one independent
  result directory.
- MATLAB execution is restricted to R2024b with `-nojvm`.

## Frozen Design

- Networks: proxy6 and proxy9.
- Assimilation: ordinary EnKF, experiment-1 fixed 500-member prior only.
- No AO/HGAO analogue selection and no localization.
- Outer blocks: 1940-1951, 1952-1963, 1964-1975, 1976-1987, 1988-2000.
- Each outer PSM and residual variance excludes its outer block.
- Alpha is selected inside each outer training set using four inner blocked
  fits that exclude both the outer and inner blocks.
- Alpha grid: 0, 0.25, 0.5, 0.75, 1.
- Candidate state: background + alpha * ordinary-EnKF increment.

## Decision Gates

The cross-fitted proxy6/proxy9 mean advances only if it improves median
region-product RMSE by at least 5% over alpha=1, has no region-product
regression worse than 5%, improves at least 75% of block-region-product cells,
and passes the frozen Round-68 absolute regional and spatial gates. Both proxy
networks must improve directionally and alpha selections must be stable in at
least four of five outer folds. Passing remains exploratory and cannot restore
independent validation from the consumed 1921-1939 period.
