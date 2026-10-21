# HMA Climate Reconstruction Mainline Ledger

Last updated: 2026-08-26
Status: ACTIVE
Authority: This file records workflow decisions. `LIA_PROJECT_FACTS.md` remains the scientific and operational contract.

## Mandatory startup protocol

Before any HMA climate analysis, plotting, or experiment:

1. Read `LIA_PROJECT_FACTS.md` in full.
2. Read this ledger in full.
3. State the single active question and verify that the proposed work is not listed under Closed Questions or Rejected Routes.
4. Do not reopen a closed question unless a new independent observation directly contradicts its recorded decision. Record that evidence and the reason for reopening first.
5. After an experiment, update this ledger with the result, decision, and exactly one next step before starting another experiment.

## Project objective

Produce a verifiable HMA temperature and precipitation reconstruction while preserving original inputs and keeping every experiment independent.

Current temperature objective: determine and, only if independently supported, correct the residual 1850/1860-1950 temperature discrepancy by separating direction, amplitude, spatial distribution, season mapping, proxy-network behavior, and weak early instrumental coverage.

## Non-negotiable constraints

- Never modify original source data or the production reconstruction during diagnostics.
- Use an independent result directory for every experiment.
- Keep 1921-1939 sealed and exclude it from method selection.
- Do not run EnKF or OGGM without explicit authorization.
- Do not count related gridded products as independent evidence.
- CRU and Berkeley may be used as instrumental references or sensitivity targets, but their agreement is not multiple independent validation.

## Current scientific state

### Closed: uniform cooling of 1500-1879

Decision: retain the current production reconstruction for 1500-1879. Do not apply a uniform cold shift, a 50:50 background blend, an optimized four-model blend, or a time-segmented correction.

Evidence already completed:

- CESM, ModE-Sim, MPI-ESM1-2-LR, and MIROC-ES2L low-frequency background comparisons.
- 50:50 blending and four-background weighting experiments.
- Independent Huang, Baiyu, and Haizi reconstruction comparisons.
- GHCN-M, CRU, and Berkeley instrumental-period checks.
- Four-prior strategy-C replay for 1500-1879.

Interpretation: the four external background families are generally colder, but independent records do not justify uniformly cooling the project reconstruction. The archived posterior-reference warm gap is systematic, yet the central proxy6/proxy9 selection pathway explains only a small fraction of it (median approximately 6.8%). Reinvestigating why the external model families are cold is not the active mainline.

Reopen only if: a new independent, dated, regionally relevant temperature record covering substantial parts of 1500-1879 contradicts the retained production curve and passes source-independence and seasonality checks.

### Open: residual discrepancy during 1850/1860-1950

The period is not treated as one tunable segment. It is a diagnostic interval with uneven evidence. The locked years 1921-1939 remain excluded from selection.

Established 1901-1920 result:

- The eastern HMA warming projection is not unique to the CESM prior.
- Full eastern background trends are positive under CESM, ModE-Sim, MPI, and MIROC for both proxy6 and proxy9.
- Omitting central proxy9 reverses or removes the eastern warming in all four priors.
- Omitting central proxy6 reduces the warming strongly but less consistently.
- Therefore the effect is associated with the central proxy-network/operator/selection pathway, especially proxy9, rather than a CESM-only covariance geometry.
- This does not establish that the raw proxy observations are wrong.

### Instrumental evidence status

- Fifteen GHCN-Monthly v4 stations are the official direct-station panel currently available.
- `qcf` is primary; `qcu` is a homogenization sensitivity check.
- Station and reconstruction comparisons use anomalies relative to each station's 1961-1990 baseline and matched months.
- CRU and Berkeley are background instrumental products and alternative reference targets, not independent station networks.
- Annual proxy reconstructions must be evaluated with correlations, trends, bias, and low-frequency/decadal metrics; monthly-scale precision must not be overinterpreted.
- Round 277's `/100` GHCN integer scaling correction is required at input preflight.

## Rejected or completed routes

Do not propose these as the next step without qualifying new evidence:

- Uniformly cooling 1500-1879 because four model backgrounds are colder.
- Treating the old 50:50 blend as a validated correction.
- Hand-tuning a 75% or other fixed blend to match one reference curve.
- Creating a correction specifically around 1850-1920 by time segmentation.
- Repeating broad CRU/Berkeley/GHCN comparisons without a new discrimination target.
- Blaming proxy9 raw data before separating record identity, PSM mapping, and selection effects.
- Returning to localization as the primary explanation.
- Running OGGM to diagnose this temperature-reconstruction mechanism.

## Single active question

Which exact central-HMA records account for the 1901-1920 eastward warming projection: records shared by proxy6 and proxy9, proxy6-only records, or records added in proxy9?

## Next decisive experiment

Create a read-only central proxy identity audit and four-prior replay for 1901-1920.

Fixed elements:

- proxy observations
- archived PSM definitions
- top-100 selection rule
- common 1500-1848 state pools for CESM, ModE-Sim, MPI, and MIROC
- eastern-HMA target and all non-central proxy operators

Required identity groups:

- central records shared by proxy6 and proxy9
- central records unique to proxy6
- central records added in proxy9

Required branches:

- full network
- each identity group acting alone within the central contribution
- leave-one-group-out sensitivity

Required reporting:

- exact identity-match rules and ambiguous matches
- counts and active years for each group
- eastern-HMA trend for each model, network, and branch
- background-member selection overlap
- separate operator and selection effects where identifiable
- non-additivity warning; leave-group-out effects must not be summed as if the selector were linear

Decision gate:

- Shared records dominate across four priors: inspect shared records' season target and PSM mapping.
- Proxy9-added records dominate: inspect those added records and the proxy9 network construction.
- Effects vary strongly by prior: quantify prior-structure uncertainty and reject a single-prior correction.
- No identity group robustly dominates: stop record-level attribution and retain the production temperature pending stronger independent evidence.

## Current authoritative outputs

- `validation_data/results/paleo_surface_validation_v0/climate_reconstruction_temperature_four_prior_strategy_c_replay_1901_1920_v1`
- `validation_data/results/paleo_surface_validation_v0/climate_reconstruction_temperature_four_prior_strategy_c_replay_1500_1879_v3`

## Progress rule

The assistant must not end a work cycle with a list of several speculative next steps. It must record one decision and one next action here. If blocked, record the exact missing input or permission. Routine read-only analysis, plotting, and independent experiment creation proceed without asking for repeated authorization.
