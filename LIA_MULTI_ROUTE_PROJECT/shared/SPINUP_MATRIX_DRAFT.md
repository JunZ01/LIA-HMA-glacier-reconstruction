# Adaptive Spin-Up Matrix Draft

Status: prospective design draft; not launch authorization.

This document defines shared semantics, not shared numerical evidence. Every
state key is `(lia_unique_id, model_family, geometry_version,
spinup_forcing_id, parameter_vector, solver_version)`. A state produced by one
model family cannot be reused as the state, reproduction, or acceptance
evidence of another family.

## Purpose and Non-Purpose

Spin-up maps a declared parameter vector to a technically valid equilibrium-
state candidate. It does not fit the LIA or RGI length, does not score a
historical curve, and does not change accepted counts. Equilibrium length,
area, volume, and shape are outputs retained for the later forward matrix.

## Input Parameter Blocks

1. Equilibrium temperature correction: `delta_t_eq`
2. Dynamics: `glen_a`
3. Mass balance: `prcp_fac x melt_f`
4. Melt threshold: `temp_melt`

`base_temp_bias` is inherited provenance. The effective equilibrium bias is
`base_temp_bias + delta_t_eq` and is not a second independent search axis.

## Registered Draft Bounds

- `delta_t_eq`: primary `[-1.5, +1.5] degC`; conditional cold extension to
  `-3.0 degC` only under a registered signed response.
- `glen_a`: Stage-2 baseline multipliers `[0.5, 1.0, 2.0, 3.0]`.
- `prcp_fac`: Stage-2 baseline multipliers `[0.8, 1.0, 1.2]`.
- `melt_f`: Stage-2 baseline multipliers `[0.8, 1.0, 1.2]`.
- `temp_melt`: first block `[-1.5, -1.0, -0.5] degC`; outer hard bounds
  `[-2.0, 0.0] degC` only under a registered trend.

## Time Design

- Repeated climate period: 30 years.
- Resume block: 300 years.
- Checkpoints: 300, 600, ..., 3000 years.
- Convergence window: the final 90 years at identical climate phase.
- Maximum: 3000 years.
- Equilibrium length is an output state coordinate, not a target gate.

At each checkpoint, compare like climate phases only. Record drift over each
of the last three 30-year cycles and over the full 90-year window. Raw and
normalized changes must both be stored; no phase-misaligned endpoint
comparison may be called convergence.

## Prospective Convergence Logic

A state may be labeled `converged` only when all of the following are true:

1. the registered minimum duration has elapsed;
2. finite, nonnegative, boundary, domain, and mass-closure gates pass;
3. every registered drift metric passes at identical climate phase;
4. the registered confirmation count passes without an intervening failure;
5. an atomic state and its complete fingerprint are written and hash-checked.

The recommended confirmation structure is two consecutive 300-year checks.
The exact metric thresholds, minimum duration, and whether this recommendation
is adopted remain `decision-required`. Existing 1-D 200-year stability and
2-D steady-volume gates are evidence for route design; neither silently
becomes the common threshold.

Possible terminal labels are `converged`, `max-duration-nonconverged`,
`technical-failure`, `scientific-domain-failure`, and
`stopped-by-contract`. Only `converged` states enter a forward-matrix
proposal.

## Required Outputs

- parameter vector and source hashes;
- full equilibrium state and state hash;
- equilibrium length, area, volume, and thickness diagnostics;
- same-phase 30-year and 90-year changes;
- simulated duration and stop reason;
- numerical, domain, and boundary diagnostics.

The output package must also record `model_family`, geometry and observation-
domain definitions, executable and dependency versions, checkpoint hashes,
and whether the state is eligible for forward evaluation. Eligibility does
not imply that a forward run is authorized.

## Matrix Construction Rules

- Enumerate prospective cells before results exist and version the manifest.
- Never launch an undeclared full Cartesian product. Use registered staged
  blocks with a stated question, representative gate, and class stop rule.
- An existing fingerprint is a no-run duplicate, not a replicate.
- Parameter bounds are hard until a new prospective contract is approved.
- A boundary-hitting response is reported; it does not silently expand a bound.
- 1-D and 2-D manifests, output roots, thresholds, and summaries remain
  separate even when their parameter names are analogous.

## Open Contract Items

- exact same-phase convergence thresholds;
- whether one passing 90-year window or two consecutive 300-year checks are
  required;
- technical boundary margin by model family;
- representative long-run policy for continuation to 3000 years;
- per-route preflight and concurrency limits.

Until these items are frozen in a user-approved canonical transaction, this
document authorizes zero spin-up cells.
