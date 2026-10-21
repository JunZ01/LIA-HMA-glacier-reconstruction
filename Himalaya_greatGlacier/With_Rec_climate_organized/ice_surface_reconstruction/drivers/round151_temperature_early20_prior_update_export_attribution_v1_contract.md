# Round 151 Early-20th-Century Temperature Attribution Contract

## Question

Which frozen production component creates the 1901--1920 positive temperature
trend in the current eastern-HMA public curve: the selected prior/background,
the proxy/PSM EnKF update, or the public spatial export?

## Frozen inputs and boundary

- Use the archived six temperature branches (`AO/HGAO/O` x `proxy6/proxy9`),
  the recovered 336 x 848 prior, archived experiment indices, and the Round 122
  allowed-year public regional curves.
- Read proxy/posterior years 1800--1920 and 1961--1990 only. Never request,
  tokenize, or decode 1921--1939.
- Keep the 1961--1990 anomaly baseline and the exact Round 25 background
  reconstruction definitions.
- Aggregate native fields with the same 2-degree HMA and longitude-region masks
  used to create the Round 122 public curves.
- Define `export_residual = public_curve - native_six_member_posterior`.
- Require both identities to close within `1e-12 degC` annually:
  `posterior = background + increment` and
  `public = background + increment + export_residual`.
- Report within-period 1901--1920 slopes and the 1901--1920 minus 1881--1900
  mean change for HMA, west, central, and east.
- No EnKF rerun, PSM refit, parameter tuning, OGGM, locked-period access, or
  Jionglacuo access is permitted.

## Interpretation

The background term includes the archived adaptive/fixed prior-member selection.
The increment term includes proxy observations, calibrated PSMs, R, covariance,
and EnKF gain. The export residual includes the public field's spatial export,
resampling, and aggregation difference. This round locates the production stage;
it does not yet assign a physical error to an individual proxy.
