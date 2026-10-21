# Stage 3 routing and Stage 2 parameter freeze

**Status:** executable readiness gate v0; no LIA search or transient run

## Policy after the velocity attribution audit

ITS_LIVE velocity does not enter a Stage 1/2 parameter likelihood. It remains
registered once as:

1. soft structural evidence;
2. a deterministic routing/identifiability signal; and
3. auxiliary evidence for interpreting Stage 4 terminal residuals.

It cannot weight, select, or contract the modern-MB equivalence surface. This
is a demotion from hard likelihood, not deletion of the observation.

## The 245-row routing ledger

`stage3_routing_245.csv` has exactly one row per LIA system. Routing dimensions
remain separate columns because they are not mutually exclusive:

- primary Farinotti readiness (`core-191` or classified modern-MB failure);
- Millan structural-scenario availability;
- velocity soft-evidence class and strict-anomaly flag;
- weak-identifiability and no-strong-gate flags;
- debris class/fraction and Hugonnet imbalance/error;
- immutable parameter-source stamps and allowed downstream use.

The 54 non-core systems remain in the ledger with their original error text and
the sign pattern of the two failed temperature-bracket residuals. The five
additional Millan failures among the core 191 retain the separate
`thickness-target-outside-glen-a-scan` route.

## Unsqueezed Stage 2 ensemble

The parent scan contains a complete 13-point regular grid for 35 systems. For
156 systems, the Stage 0 baseline already occupies one regular-grid node, so
the stored grid has 12 rows; that baseline is added back to restore 13 nodes.
An extra baseline is ignored when the regular grid is already complete.
Candidates are then sorted along the precipitation/melt equivalence coordinate.
Stage 2 freezes positions 1, 7, and 13 as:

- `low-turnover`;
- `mid-turnover`;
- `high-turnover`.

Selection uses no velocity field, loss, compatibility class, or LIA data. The
three representatives carry equal weight. Farinotti effective `glen_a` and
`fs=0` are fixed across the three representatives. Member precipitation is:

```text
raw_prcp_fac = winter_prcp_fac_center * precip_prior_multiplier
prcp_fac = clip(raw_prcp_fac, 0.1, 10.0)
```

matching the OGGM bounds used by the parent candidate evaluation.

## Outputs

- `stage3_routing_245.csv`: one-row-per-system Stage 3 work ledger;
- `stage2_system_ensemble.csv`: 191 x 3 frozen system parameters;
- `stage2_member_ensemble.csv`: 950 x 3 frozen member parameters;
- `excluded_54_details.csv`: retained modern-MB failure branch;
- `routing_summary.json`, `run_manifest.json`, and `failures.csv`.

Stage 3 may read these files but may not rewrite them. LIA geometry and timing
first enter after this gate, in the initial-state objective.

## Acceptance

The gate passes only with exactly 245 routing rows, 191 core systems, 54
classified exclusions, 186 Millan systems, 573 system-ensemble rows, and 2,850
member-ensemble rows. Every core system must contribute exactly three roles,
all selected candidates must preserve modern MB within the parent tolerance,
all parameter values must carry source stamps, and velocity selection weight
must equal zero.
