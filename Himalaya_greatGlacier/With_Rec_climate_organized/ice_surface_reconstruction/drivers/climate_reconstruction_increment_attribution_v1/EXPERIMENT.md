# Climate reconstruction increment attribution v1

## Question

This read-only diagnostic asks:

1. Which of the six archived reconstruction members create the negative
   twentieth-century low-frequency signal in the final equal-weight mean?
2. For the exactly replayable OEnKF-proxy6 member, which individual proxy
   posterior increments create its 1901-2000 trend and early-to-late change?

## Frozen scope

- Six archived members: AO/HGAO/OEnKF crossed with proxy datasets 6 and 9.
- Exact replay branch: OEnKF, proxy dataset 6, prior experiment 1.
- Target diagnostic: 33.75 N, 75.75 E (glacier system 3584 pixel).
- Regional diagnostic: the frozen 336-cell HMA replay state grid.
- Instrumental interval: 1901-2000.
- Anomaly reference: 1961-1990.
- Low-frequency window: 30 years.
- No glacier simulation and no glacier curve enters any climate decision.

## Hard preflight gates

- The six-member equal-weight mean must close to the archived ensemble mean.
- The replayed OEnKF-proxy6 posterior mean must remain within 0.024 degC of
  the archived replay target.
- Prior mean plus the sum of per-proxy K times innovation contributions must
  close to the replayed posterior at state, target, and regional scales.
- Trend and early-to-late contribution sums must close to the corresponding
  posterior-increment metrics.

No parameter is changed. Failure of any gate stops interpretation.
