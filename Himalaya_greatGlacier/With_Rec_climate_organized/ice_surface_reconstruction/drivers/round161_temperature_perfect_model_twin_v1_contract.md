# Round 161 Temperature Perfect-Model Twin Contract

## Objective

Test whether the archived temperature prior and reconstruction operators can
recover a known HMA state when proxy quality is ideal. This separates a minimum
method/prior error from errors caused by real proxy records and PSM
nonstationarity.

## Frozen Design

- Use the archived 848-member temperature prior and experiment-1 pool of 500
  unique members.
- Select 24 deterministic truth members, evenly spaced through the 348 members
  excluded from that pool. Truth members never enter the reconstruction ensemble.
- Use the full-period proxy6 and proxy9 PSM operators and unchanged RegEM
  diagonal residual variances.
- Generate synthetic observations from each truth member's exact `HXb`.
- Run two conditions: noise-free and five fixed-seed residual-noise replicates per
  truth member (`seed=16120260820`).
- Compare the all-500 ordinary EnKF, archived top-100 global-pattern adaptive AO,
  and the all-500 background mean. Evaluate proxy6, proxy9, and their equal mean.
- Score HMA, west `<80E`, central `80-90E`, and east `>=90E` using the existing
  HMA mask and cosine-latitude weights.

## Gates For The Equal-Network Ordinary EnKF

- Noise-free HMA median state-RMSE skill over the background must be at least
  `0.50`, median spatial-pattern correlation at least `0.50`, HMA regional-value
  correlation at least `0.50`, and each subregional value correlation at least
  `0.30`.
- Residual-noise HMA median state-RMSE skill must be at least `0.20`, median
  spatial-pattern correlation at least `0.30`, and HMA regional-value correlation
  at least `0.30`.

Passing shows only that the method can work under its own prior/PSM assumptions
and releases a separately contracted instrumental-truth twin. Failure identifies
a structural prior/operator limitation and blocks that external stage until the
failure is explained.

## Boundaries

- Do not use actual proxy observations.
- Do not read instrumental climate values, 1901-1939, or Jionglacuo.
- Do not alter R, localize, tune member count, inspect paleo trajectories, run a
  production reconstruction, or run OGGM.
