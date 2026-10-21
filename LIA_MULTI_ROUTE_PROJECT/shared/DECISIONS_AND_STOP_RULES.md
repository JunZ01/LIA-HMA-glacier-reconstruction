# Decisions and Stop Rules

Status: organizational register; no launch authorization.

## Decision Register

| ID | Status | Decision | Effect now |
|---|---|---|---|
| D-001 | canonical | Authority order is facts, execution contract, current state, versioned evidence, conversation context. | Organizational files cannot override canonical control. |
| D-002 | canonical | Current state is v10.188, `decision-required`, with accepted 58, active 0, queued 0. | All model launches remain blocked. |
| D-003 | canonical | 1-D and modified planform-2D are different model families. | Evidence, states, gates, and accepted-count claims stay separate. |
| D-004 | proposal | Spin-up creates a state library; equilibrium length is an output coordinate rather than an observational target. | Design work only. |
| D-005 | proposal | Inspect three same-phase 30-year cycles and prefer two consecutive 300-year confirmation checks. | Thresholds and confirmation count are not frozen. |
| D-006 | proposal | Forward evaluation is a separate state-by-forcing matrix with mandatory baseline reproduction. | No cell is authorized. |
| D-007 | decision-required | Exact model-family-specific convergence, reproduction, boundary, and invalid-rate thresholds. | Must be frozen prospectively before preflight. |
| D-008 | decision-required | Whether to authorize one exact-contract Stage B retry for 2086 and 2534. | No 2-D retry, steady, or historical launch. |
| D-009 | decision-required | Which route may supply the primary approximately-100-curve population and under which acceptance/reproduction contract. | No route gains accepted-count authority. |
| D-010 | decision-required | Versioned matrix manifests, representative systems, output roots, resources, and one-chain preflights. | No spin-up or forward matrix launch. |
| D-011 | proposal | Minimal GPU spin-up benchmark v1 fixes system 705, scientific parameters, batch sizes 1/2/4/8/16, five 0.1-year technical batches, and one batch-size-1 300-year checkpoint/resume chain. | Draft only; no benchmark or 300-year chain is authorized. |

## Global Stop Rules

| ID | Trigger | Required action |
|---|---|---|
| S-001 | Required authority file unread, hash conflict, or validator failure. | Stop all affected work; reconcile evidence before a new route snapshot. |
| S-002 | Canonical state is `decision-required` or `launch_authorized=false`. | Permit read-only audit and organizational drafting only. |
| S-003 | Duplicate fingerprint or colliding output root. | Do not launch or overwrite; point to existing evidence. |
| S-004 | Missing identity, geometry, forcing, solver, parameter, threshold, or observation-operator provenance. | Stop before execution. |
| S-005 | A proposed action changes a frozen threshold, bound, forcing, geometry, operator, or solver. | Create a disclosed new contract proposal and obtain explicit approval. |
| S-006 | A technical failure occurs. | Preserve evidence; allow no retry without identified cause and a versioned exact-contract retry decision. |
| S-007 | A scientific failure closes the tested mechanism or class. | Do not reopen it under a new filename or glacier-wise micro-search. |
| S-008 | 1-D evidence is used to initialize, reproduce, accept, or count a 2-D result, or vice versa. | Reject the inference and restore route-stratified reporting. |
| S-009 | A result would change accepted counts or canonical route state. | Require formal evidence, independent reproduction, explicit user approval, and the full canonical transaction. |

Route contracts may add stricter stop rules. They may not weaken these global
rules. A future result may resolve a decision only through immutable versioned
evidence; an agent message alone resolves none of them.
