# HMA Climate Reconstruction Mainline Ledger v2

Last updated: 2026-08-26
Status: ACTIVE AND AUTHORITATIVE
Supersedes: `CLIMATE_RECONSTRUCTION_MAINLINE_LEDGER.md`
Contract: `LIA_PROJECT_FACTS.md`

## Mandatory startup

Before climate analysis, read `LIA_PROJECT_FACTS.md` and this ledger in full. Do not use the superseded ledger's proxy6/proxy9 identity split as the mainline.

## Objective

Diagnose the common temperature discrepancy between the project reconstruction and `rec_ref` during 1850/1860-1950. Determine whether the discrepancy is real and whether it concerns direction, mean level, trend, amplitude, spatial structure, or seasonal mapping. Keep 1921-1939 sealed from method selection.

## Correct scale of the problem

Proxy6 and proxy9 are close to each other compared with their shared difference from `rec_ref`. Therefore:

- `proxy6 - proxy9` is a secondary network sensitivity.
- `proxy6 - rec_ref` and `proxy9 - rec_ref` are the primary residuals.
- Splitting proxy6/proxy9 record identities cannot explain their common-mode difference from `rec_ref`.
- A record-identity replay is deferred unless independent evidence first shows that the 1901-1920 central-to-eastern projection explains a material part of a real reconstruction error.

## Closed routes

- Do not uniformly cool 1500-1879.
- Do not treat the old 50:50 blend as validated.
- Do not hand-tune fixed background proportions to match `rec_ref`.
- Do not reopen the four-background coldness comparison without new independent evidence.
- Do not use proxy6-versus-proxy9 differences as a substitute for evaluating the common `rec_ref` residual.
- Do not count CRU and Berkeley as multiple independent station networks.
- Do not run EnKF, OGGM, or modify production during this diagnostic.

## Single active question

During 1850/1860-1950, are the project reconstruction and `rec_ref` different in a way independently supported as an error, and if so, which component is wrong?

## Next decisive analysis

Build one common-residual evidence matrix using existing read-only outputs and observations.

Compare on the same plots and scales:

- proxy6, proxy9, and `rec_ref`
- proxy6 minus proxy9
- proxy6 minus `rec_ref`
- proxy9 minus `rec_ref`
- GHCN-M direct stations where available
- CRU and Berkeley as instrumental reference alternatives
- available independent regional reconstructions with explicit season and region labels

Report separately for western, central, and eastern HMA and for annual or physically matched seasons:

- overlap years and coverage
- mean residual
- linear trend
- correlation
- variance and amplitude ratio
- decadal residual behavior
- source independence and limitations

The 1921-1939 values may not be used to select, fit, weight, or tune a method.

## Decision gate

- Independent observations favor the project reconstruction: retain production and classify much of the `rec_ref` difference as reference uncertainty.
- Independent observations favor `rec_ref` consistently: design a physically constrained Candidate v2, then validate outside its fitting evidence.
- Evidence differs by region or season: reject a uniform correction and address only the supported component.
- Evidence is inadequate before 1901: do not tune that interval; report unresolved uncertainty.

## Progress rule

Every experiment must end with one recorded decision and one next action. Closed questions stay closed unless new independent contradictory evidence is documented first.
