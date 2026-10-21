# Round 125 Ganesh and Keylong d18O Proxy Qualification

## Objective

Qualify two public NOAA tree-ring d18O series that are absent by name from the
current proxy ledger: Ganesh (Nepal) and Keylong (India).

## Boundaries

- Original project and downloaded source files are read-only.
- All results are written to an independent Round 125 directory.
- Candidate value fields for 1921-1939 are skipped before numeric conversion.
- Wache is excluded because its locked-period proxy values were inadvertently
  rendered by an external web preview and it is no longer a blind candidate.
- The sealed climate final-test reserve is not opened.
- Modern tests use only the five frozen 1940-2000 blocks.
- No OGGM experiment is run.

## Tests and decision

Each candidate undergoes identity/spatial novelty checking and blocked OOF
temperature-only, precipitation-only, and bivariate annual PSM tests against
the frozen multi-product modern consensuses. A spatially nonduplicate record
advances only if at least one documented branch has positive pooled OOF
correlation and positive skill against the training-mean baseline.
