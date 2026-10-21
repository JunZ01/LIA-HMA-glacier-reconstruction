# Round 234 V2 Contract: Hengduan Individual-Site Modern OOF

V1 stopped at the climate-target preflight before chronology construction or
skill calculation. The monthly target reader returned the expected 61 by 5
matrix, but the aggregation call supplied one season definition for five site
columns, causing unassigned columns to remain non-finite.

V2 changes only the aggregation argument from one May--September tuple to five
identical May--September tuples, matching the verified Round 180 v2 interface.
The raw sites, modern-only parser, dplpy chronology method, years, folds,
products, site gates, family-wise circular-shift test, interpretation boundary,
source-hash checks, and all locked-data prohibitions remain exactly as frozen in
the Round 234 v1 contract.

