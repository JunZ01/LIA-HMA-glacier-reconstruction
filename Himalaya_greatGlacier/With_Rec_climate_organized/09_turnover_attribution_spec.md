# Turnover attribution preflight

**Status:** executable diagnostic preflight v0; not a flux correction and not a posterior

## Purpose

The Farinotti and Millan scans show a broad velocity mismatch. Before defining
a formal velocity likelihood, this preflight asks whether a deliberately broad
geodetic-imbalance envelope can explain the observed gate speed and records
debris and gate-position strata for the same rows.

The output is a D-line diagnostic product. It does not modify Stage 0 glacier
directories, MB parameters, thickness, or velocity observations.

## Gate-level input row

Each selected best-candidate gate row is joined with:

- ITS_LIVE gate speed and effective interval;
- candidate apparent-balance flux and fixed cross-section;
- normalized flowline position and gate elevation relative to `Zmed`;
- Scherler Ratio debris fraction/class for the member outline;
- system Hugonnet MB, fully correlated error, area, region, and precipitation class.

The debris field is explicitly an **outline-member proxy**, not a 30 m debris
fraction sampled inside the velocity gate. This distinction remains in the
machine-readable output and must not be lost in later interpretation.

## Pressure-list three-way audit

The pressure list is defined before this preflight by
`candidate_edge_class == low-turnover-edge`. It is not recomputed from the
attribution result. Each evaluable pressure-list system is assigned once, in
this priority order:

1. `imbalance-envelope-explains`: every selected strong gate is inside the
   extreme imbalance envelope;
2. `debris-stratified-residual`: at least one selected gate remains outside the
   envelope and the frozen system debris fraction is at least 10%;
3. `residual-outside-envelope`: at least one gate remains outside the envelope
   and the system debris fraction is below 10%.

Systems without an evaluable strong gate remain
`no-evaluable-strong-gate`; they are recorded but cannot enter the scientific
three-way denominator. The second class is an association stratum, not proof
that a debris process has already closed the flux budget. Causal attribution
requires a later debris-aware structural scenario.

## Coarse imbalance envelope

The current candidate flux is the upper, apparent-balance edge:

```text
Q_upper = Q_balance
```

For a negative system geodetic MB, the first-pass extreme lower edge assigns
the entire system volume-loss rate to a reduction in the gate flux:

```text
I_nominal = max(0, -MB) * area / rho / seconds_per_year
I_max      = max(0, -(MB - MB_error)) * area / rho / seconds_per_year
Q_lower_nominal = max(0, Q_balance - I_nominal)
Q_lower_extreme = max(0, Q_balance - I_max)
```

The conservative envelope uses `[Q_lower_extreme, Q_upper]`. It is intentionally
wide and does not claim a spatially correct along-flowline allocation. A gate
inside this envelope is classified as `imbalance-envelope-explains` for the
preflight only. A gate outside remains a residual requiring further diagnosis.

The next exact operator must replace this extreme allocation with an
along-flowline/geodetic-height-change model. It must retain the envelope as a
diagnostic bound and propagate Hugonnet covariance.

## Conversion sensitivity

The current deformation conversion uses `u_surface = u_integrated / 0.8`.
The preflight also evaluates the global sensitivity `u_surface = u_integrated`
(`surface_factor=1.0`) as a sliding-dominated lower-speed scenario. It is a
global operator sensitivity, not a per-glacier fitted parameter.

## Parallel execution and writes

System/scenario groups are independent. A configurable Windows-safe process
pool evaluates groups in parallel. Workers return rows only; the parent process
is the sole writer of CSV, JSON, failure, and progress artifacts. This prevents
concurrent writes to shared OGGM directories and makes checkpoints atomic.

## Acceptance criteria

The first attribution pass is complete only when:

1. every system in both thickness scenarios is represented, including no-gate systems;
2. every gate row carries debris scope, gate position, MB/uncertainty, and envelope fields;
3. worker exceptions are retained in `failures.csv` and do not stop other groups;
4. progress checkpoints and the dashboard agree with the final row counts;
5. the three-way pressure list is available: envelope-explained, debris-stratified,
   and residual-outside-envelope.

No formal posterior is published from this preflight.
