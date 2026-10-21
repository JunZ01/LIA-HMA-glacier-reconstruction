# Round 246 NPL065 calibration-endpoint attribution contract

## Question

How much of the published 1960-2013 NPL065 DJF Tmin correlation can be
explained by extending calibration beyond 2000 and by reporting in-sample
correlation rather than blocked out-of-fold skill?

## Frozen design

- Parse only ring-width slots from 1941 through 2013. Skip all earlier slots,
  including 1921-1939, before numeric conversion.
- Build one exploratory modern-only STD chronology over 1941-2013 using the
  Round 238 dplpy ModNegEx/residual procedure, linear fallback on convergence
  failure, Tukey biweight mean, and no prewhitening.
- Use the published CRU TS4.01 DJF Tmin box: 27.5-28.5 N, 86-87 E.
- Compare 1960-2000 with the existing five blocked folds.
- Compare 1960-2013 with six contiguous nine-year blocked folds:
  1960-1968, 1969-1977, 1978-1986, 1987-1995, 1996-2004, 2005-2013.
- Report in-sample correlation, blocked OOF correlation and skill, and the
  2001-2013 endpoint correlation.
- Compare the rebuilt chronology with Round 238 on their 1941-2000 overlap.

## Interpretation boundary

This is a calibration-period and reporting-metric attribution using an
approximate modern-only chronology. It is not the paper's exact signal-free,
age-dependent 30-year spline chronology and cannot validate or reject that
chronology. It must not access paleo proxy values, locked 1921-1939 values,
Jionglacuo, production EnKF, production reconstruction, or OGGM.
