# Round 158 BT021 Low-Frequency Modern OOF Contract

## Question

Does the source-distinct BT021 Ura-Dhur chronology contain transferable
11-year JJA temperature information at its documented Bhutan location?

## Frozen Design

- Predictor: the Round 157 allowed `BT021` standardized growth-index chronology.
- Targets: Berkeley Earth and CRU JJA mean temperature at `27.4223 N,
  90.9675 E` over 1940-2000.
- Use a centered 11-year mean and the three Round 16 outer blocks:
  1940-1959, 1960-1979, and 1980-2000.
- Test centers require their complete 11-year window inside the held-out block.
  Training centers and predictor standardization support are purged whenever
  their 11-year windows overlap the held-out block.
- Fit only an intercept and one BT021 slope. The absolute control is the
  fold-training target mean.

## Gates

Both products must have pooled OOF correlation at least 0.30, absolute skill at
least 5%, positive test correlation in at least two of three blocks, positive
fit slopes in all three blocks, and positive first- and third-block test
correlations. Berkeley/CRU stitched OOF predictions must correlate at least
0.80.

A pass advances BT021 only to a separately contracted incremental east-network
OOF test. It does not authorize a paleo curve, production reconstruction, EnKF,
or OGGM.

## Boundary

Only the already isolated 1940-2000 BT021 values and modern instrumental targets
may be read. The locked interval, 1901-1920, Bhutan 2015 paleo values, other
external benchmarks, and Jionglacuo are excluded. All sources remain read-only.
