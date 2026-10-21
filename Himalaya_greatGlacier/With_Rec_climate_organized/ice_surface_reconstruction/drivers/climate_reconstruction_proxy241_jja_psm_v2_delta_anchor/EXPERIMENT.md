# Proxy 241 JJA PSM Delta-Anchor v2

Question: does replacing only proxy9 record 241's Jan-Dec bivariate PSM with a JJA bivariate PSM reduce the 1901-1950 phase conflict at target 3584?

Frozen quantities: proxy observations, all other proxy operators, prior, localization, RegEM-completed error matrix, target extraction, and comparison reference.

## Resolved replay issue

The archived annual operator was produced in an older MATLAB environment. The current contract requires MATLAB R2024b, whose `load_S.m` uses a stochastic `gamrnd(1, 0.05, ...)` replacement for nonpositive precipitation after bias correction. Therefore the annual temperature predictor, chronology, coefficients, and residuals reproduce, but the fresh precipitation predictor and fresh annual `HXb` cannot be expected to match the archived file bit-for-bit.

This v2 experiment uses a common-random-number construction: reset MATLAB R2024b to the same fixed RNG state before the fresh Jan-Dec and JJA PSM calls, compute `fresh_jja_HXb - fresh_annual_HXb`, and apply that difference to the archived annual `HXb`. The EnKF baseline is therefore exactly the archived annual baseline, while the replacement contains only the controlled seasonal operator perturbation under a shared precipitation realization.

Preflight gates:

- deterministic annual replay fields match the archived record;
- repeated annual calls with the same R2024b RNG state match each other;
- anchored annual `HXb` matches the archive exactly;
- all source hashes remain unchanged.

No OGGM is used. All outputs are written to a new independent result directory.