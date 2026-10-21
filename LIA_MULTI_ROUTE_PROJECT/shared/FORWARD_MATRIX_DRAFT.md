# Forward Matrix Draft

Status: prospective design draft; not launch authorization.

## Purpose

The forward matrix evaluates how an immutable converged spin-up state responds
to a declared historical forcing contract. It is downstream of the spin-up
library: a forward cell may not alter, refit, or relabel its source state.

The cell key is `(spinup_state_hash, model_family, forward_forcing_id,
forward_parameter_policy, observation_operator_version, solver_version)`.
The `model_family` must match the source state. Parameter priors may be
compared across families, but states and outcomes may not be pooled or
transferred.

## Prospective Axes

Every future versioned manifest must freeze these axes explicitly:

1. spin-up state ID and hash;
2. forward temperature-anomaly product and reference period;
3. equilibrium-to-forward temperature-offset policy;
4. precipitation-anomaly mode, independently from `prcp_fac`;
5. physical-parameter carryover or explicitly declared forward-only change;
6. time interval, checkpoint cadence, and observation operator;
7. model-family-specific solver, geometry, domain, and resource contract.

No axis may be inferred from a filename, an old output, or a different route.

## Required Baseline Cell

Before a new forcing or forward-only policy is evaluated, the source branch
must be replayed from the saved state. Its registered checkpoints must meet the
prospective reproduction tolerances for length, area, and volume. A failed
reproduction stops every dependent cell; it is not a scientific response.

## Observation and Scoring Layer

Observations are applied only after a technically valid forward trajectory is
complete. Each summary must retain raw annual/checkpoint length, area, volume,
mass-balance, domain, and boundary diagnostics before scoring.

- 1-D length uses its registered observation-domain operator and must exclude
  synthetic safety-tail ice.
- 2-D length and separation require their own registered planform operator;
  no 1-D node-count or flowline-length gate is imported by analogy.
- Endpoint, amplitude, corridor, timing, modern-MB, and shape diagnostics are
  separate named fields. A diagnostic is not promoted to a gate after results
  exist.
- Cross-family comparisons are stratified analyses, never silent pooled
  acceptance evidence.

## Cell Outcomes

Each cell receives exactly one terminal class:

- `completed-scientific-result`;
- `technical-failure`;
- `scientific-domain-failure`;
- `duplicate-not-launched`;
- `stopped-by-contract`;
- `not-authorized`.

A promising trajectory remains a route-specific candidate until its formal
gates, independent reproduction, and a canonical transition transaction are
complete.

## Stop Rules

Stop a cell or dependent block when any of the following occurs:

- the controller validator fails or canonical state is `decision-required`;
- the manifest, source-state hash, forcing identity, or observation operator
  is missing or ambiguous;
- the baseline reproduction gate fails;
- the fingerprint already exists;
- finite/nonnegative, mass-closure, boundary, domain, or checkpoint integrity
  fails;
- a preregistered class invalid-rate ceiling is reached;
- a bounded response loses the declared bracket or reaches a closed family;
- the result would require a threshold, parameter bound, forcing, geometry,
  or solver change not present in the approved contract.

Exact numerical tolerances and class invalid-rate ceilings remain open
decisions. Until a versioned user-approved transaction freezes them, this
document authorizes zero forward cells.
