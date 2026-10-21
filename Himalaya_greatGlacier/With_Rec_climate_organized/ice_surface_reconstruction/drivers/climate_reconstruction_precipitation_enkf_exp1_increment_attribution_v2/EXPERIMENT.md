# Precipitation EnKF Experiment-1 Increment Attribution V2

Date: 2026-08-13

V2 supersedes the failed V1 metadata implementation. V1 stopped before
MATLAB because it assumed every proxy had three PSM coefficients. Proxy6
actually contains 201 three-coefficient bivariate tree records and one
two-coefficient glacier-ice record.

V2 retains the frozen V1 scientific design and records coefficients only as
generic `b0`, `b1`, and optional `b2`. It evaluates 1901-1920 only; 1921-1939
remains locked. No proxy, PSM, prior, EnKF setting, or source file is changed.
