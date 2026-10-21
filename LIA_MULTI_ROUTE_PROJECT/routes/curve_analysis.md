# Curve Analysis Route

Purpose: read-only comparison and publication analysis.

Initial evidence groups:

- 58 formally accepted curves;
- 32 retained non-near-monotonic accepted curves;
- 26 temporarily excluded near-monotonic accepted curves;
- unresolved and failed systems, stratified by failure class;
- future 1-D and 2-D outputs, kept model-family specific.

Planned products:

- length, area, volume, and mass-balance ensemble curves;
- parameter and equilibrium-state distributions;
- accepted-versus-excluded response comparisons;
- model-family sensitivity and representative 2-D validation figures.

Status: read-only analysis is allowed; no simulation launch authority.

Every table and figure must carry `model_family`, `route_id`, evidence
version, and acceptance status. The 58 canonical accepted systems are the only
accepted population unless a later canonical transaction changes that count.
Proposed, provisional, short-qualified, steady-passing, historical-passing,
excluded-shape, technical-failure, and scientific-failure groups remain
distinct.

Cross-family products may align variables and show stratified sensitivity, but
must not pool states, denominators, gates, or acceptance counts. A missing
route label or incompatible observation definition is a hard stop for the
affected comparison.
