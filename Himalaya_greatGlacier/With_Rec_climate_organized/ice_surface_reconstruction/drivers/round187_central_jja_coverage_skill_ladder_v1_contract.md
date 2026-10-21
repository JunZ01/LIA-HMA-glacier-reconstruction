# Round 187 Central-JJA Coverage-Skill Ladder Contract

## Purpose

Diagnose whether the strong modern OOF skill of the 41-record Round 156
central-JJA branch depends on records that do not cover the early reconstruction
period.

## Frozen cohorts

For each cutoff in `1550, 1600, ..., 1850`, retain a record only when it has at
least 40 finite years in every fixed 50-year window from that cutoff through
1899 and at least 40 finite years in 1940-2000. Add the complete 41-record
modern-only branch as a reference. Cohorts are defined from Round 185 coverage
counts only; proxy values and skill never affect membership.

## Frozen evaluation

Run the exact Round 156 three-block purged 11-year low-frequency OOF test for
Berkeley Earth and CRU against the unchanged 41-site central-band JJA target.
A cohort is transport-capable only when both products have pooled correlation
at least 0.30, absolute skill at least 0.05, and positive slopes in all folds.

The first passing cutoff is diagnostic, not an authorized reconstruction start
date or a post-hoc production candidate.

## Boundaries

- Use the already-derived Round 185 coverage table; do not reopen historical
  proxy values or the locked period.
- Do not use Chongce values, Jionglacuo, EnKF, a production correction, or OGGM.
- Preserve all source fingerprints.

