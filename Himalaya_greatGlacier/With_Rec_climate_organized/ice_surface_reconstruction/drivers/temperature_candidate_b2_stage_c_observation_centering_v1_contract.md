# Candidate B2 Stage C observation-centering supplement v1

## Purpose

Freeze the proxy-observation zero point before any Candidate B2 real proxy
observation is read. This supplement changes no PSM coefficient, residual
variance, model weight, network weight, state prior, or time gate.

## Frozen zero point

For each selected bivariate proxy record in proxy6 and proxy9, define

`center_i = mean(HXb_i[1500:1848])`

from the archived CESM PSM operator. Candidate B2 assimilates

`y_centered_i(year) = y_raw_i(year) - center_i`.

The constants are calculated only from archived model-operator values. Proxy
`data` values are not read while the constants are prepared. No observed
period, instrumental target, independent reconstruction, or validation metric
enters the centering calculation.

## Rationale

The Candidate B2 operator means are anomalies relative to the same archived
1500-1848 operator baseline. Subtracting the archived operator mean therefore
preserves the innovation zero point used by the original CESM PSM while
avoiding an observation-fitted intercept. Centering each proxy on its own
observed coverage, on an instrumental interval, or on 1850-1920 is forbidden.

## Data and authorization locks

- Selected records remain the 201 proxy6 and 290 proxy9 complete bivariate PSM
  rows frozen by Stage B.
- Allowed years remain 1500-1920 and 1940-2000.
- Years 1921-1939 may not be requested, read, filtered, scored, or plotted.
- 20CRv3 may not be accessed.
- Static-package preparation may read `HXb`, `location`, and metadata, but not
  the proxy `data` dataset.
- Reading real proxy observations or executing the Candidate B2 EnKF still
  requires the user's separate explicit authorization.
- Production files and OGGM remain untouched.
