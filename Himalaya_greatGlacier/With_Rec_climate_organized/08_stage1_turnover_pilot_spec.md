# Stage 1 turnover pilot specification

**Status:** executable pilot v2, not a frozen parameter posterior  
**Structure:** Farinotti 2019, `fs=0`, Stage 0 effective `glen_a` frozen  
**Observation:** ITS_LIVE v2.1 gate velocity, used once

## Purpose

This pilot tests the Stage 1 turnover operator before a 191-system run. It does
not use LIA geometry and it does not modify Stage 0 glacier directories.

The v2 operator first solves the feasible `melt_f` interval implied by the two
registered precipitation-prior bounds. It samples 13 values inside that
system-specific interval. For every candidate `melt_f`, the precipitation
multiplier is solved so that the 2000-2020 Hugonnet system MB remains fixed.
`temp_bias` remains at its Stage 0 value. Candidate apparent MB and flux are
then recomputed, while the Farinotti cross-section is held fixed.

The v0 coarse global grid is retained as an audit result. Its out-of-ridge grid
points are not counted as model failures; v1 removed them from the candidate
set by construction. v2 additionally treats a near-zero prior-bound residual
as an endpoint root, which is required for systems whose Stage 0 solution lies
exactly on the registered precipitation bound.

## Fixed and free quantities

| Quantity | Pilot treatment |
|---|---|
| Farinotti thickness and cross-section | fixed Stage 0 input |
| Product-conditional `glen_a` | fixed; never rewritten |
| `temp_bias` | fixed at Stage 0 value |
| `melt_f` | sampled on the system-specific feasible equivalence interval |
| precipitation multiplier | solved on the modern-MB equivalence surface |
| LIA geometry or reconstructed climate | forbidden |
| ITS_LIVE velocity | Stage 1 turnover diagnostic, used once |

The flux-to-velocity conversion is

```text
u_integrated = flux / fixed cross-section
u_surface = u_integrated / 0.8
```

for Glen `n=3` and `fs=0`. This is conditional on the frozen structural
scenario; it is not a deformation/sliding partition inferred from velocity.

## Velocity objective

`usable` and `weak-slow` gates are evaluated. A gate contributes zero interval
loss when modeled speed lies inside its registered effective interval. Outside
the interval, the residual is the logarithmic distance to the nearest interval
edge. `weak-slow` gates receive weight 0.25 and `usable` gates weight 1.0.

The resulting minimum-loss candidate is a pilot diagnostic only. All feasible
candidates remain in the output table; this pilot does not publish a point-only
posterior.

## Isolation and validation

- Selected Stage 0 glacier directories are copied to the pilot work directory.
- Candidate apparent-MB tasks write only to those copies.
- Source key-file hashes are checked before and after the run.
- The exact Stage 0 candidate must reproduce the saved baseline speed within
  relative tolerance `1e-6` before the operator is considered valid.
- Every feasible candidate must preserve system MB within `0.1 kg m-2 yr-1`.
- Missing or zero fixed cross-sections are retained as operator failures.
- Systems without a strong velocity gate remain `weak-identifiability`; they
  are not deleted and no fallback parameter is assigned.

## Pilot systems

The eight systems in
`config/stage1_turnover_pilot_farinotti_v2.json` cover the three permanent
review systems, a near-match baseline, a severe speed excess, weak-slow-only
gates, unusable-only gates, and a known zero-section failure.

## Scale-up gate

The full 191-system scan may start only after:

1. source immutability passes;
2. baseline speed reproduction passes for every evaluable pilot system;
3. modern-MB equivalence residuals pass for every feasible candidate; and
4. all data/operator failures are present in machine-readable output.

Reaching the velocity interval is not a scale-up requirement. Failure to reach
it is scientific evidence about the MB parameterization or missing structure,
not permission to change `glen_a` or use LIA information.

The first full application is configured in
`config/stage1_turnover_full_farinotti_v0.json`. It reads all 191 Stage 0 MB
success systems (950 members), checkpoints after every system, and resumes only
when the code, configuration, input-table, and Stage 0 source hashes match.
Partial rows from an interrupted system are discarded before that system is
re-entered.

## v2 outcome

The audited v2 pilot completed all eight systems and 58 members:

- 105 feasible modern-MB-equivalent candidates;
- 303 candidate-gate residual rows;
- maximum absolute modern-MB residual below `7e-10 kg m-2 yr-1`;
- maximum baseline-speed reproduction error below `2e-13`;
- Stage 0 key-file hashes unchanged;
- one retained fixed-section operator failure in system 3082;
- one system (1856) routed to weak identifiability because it has no evaluable
  strong gate.

Six of the seven velocity-evaluable systems select the lower precipitation
prior boundary. Only system 5180 places all strong gates inside their effective
observation intervals. The remaining speed excess is therefore retained as a
turnover/structure incompatibility diagnostic; it is not removed by widening
the registered parameter priors.

## Farinotti full-scan outcome

The resumable full scan completed 191 systems and 950 members in 14.5 minutes.
It produced 2,518 feasible equivalence candidates and 6,201 gate residuals.
All semantic validation checks passed, 1,900 source/clone fixed-input file
pairs matched byte-for-byte, and the only nine operator failures were positive
velocity gates placed at zero Farinotti cross-section.

The system diagnostic classes are:

- 22 velocity-compatible;
- 42 partially velocity-compatible;
- 108 still too fast at the lowest-turnover edge;
- 2 mixed or interior unresolved residuals;
- 17 weak-identifiability systems with no evaluable strong gate.

This result closes the Farinotti operator test but does not freeze Layer 1
parameters.

## Millan full-scan and structure comparison

The Millan scenario completed 186 systems and 932 members in 12.7 minutes. It
produced 2,452 candidates and 6,149 gate residuals; 1,864 source/clone
fixed-input pairs matched byte-for-byte. Its system classes are 28 compatible,
49 partially compatible, 94 still too fast at the low-turnover edge, 2
mixed/interior residuals, and 13 without an evaluable strong gate.

Among 173 common velocity-evaluable systems, Millan improves categorical
reachability for 17, leaves 155 unchanged, and worsens one categorical result.
It lowers the best interval loss for 130 systems and raises it for 21. This is
a structural sensitivity result, not model selection, because Millan thickness
is velocity-driven.

The attribution audit supersedes velocity-weighted posterior publication. The
next implementation step is deterministic routing plus an unweighted Stage 2
freeze of low/middle/high representatives along each modern-MB equivalence
surface. Velocity-compatible, boundary, no-gate, and incompatible cases retain
their diagnostic labels as soft evidence, but none selects a parameter point.
No LIA information enters this decision.
