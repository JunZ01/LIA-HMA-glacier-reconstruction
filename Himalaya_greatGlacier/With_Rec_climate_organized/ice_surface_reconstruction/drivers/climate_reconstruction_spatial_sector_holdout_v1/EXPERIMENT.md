# HMA Spatial-Sector Holdout Pilot v1

This independent no-OGGM experiment tests whether instrumental anchoring
propagates temperature information into HMA regions that were not assimilated.
It uses only 1901-1980 and does not open the 1981-2000 outer test.

The 86 common HMA state cells are divided into four contiguous longitude
sectors: 69-77E, 79-87E, 89-95E, and 97-103E. Four held-out products, four
held-out 20-year blocks, and four held-out sectors yield 64 validation cells.
The held-out product and held-out sector are absent from the instrumental
observations. Instrument error statistics are estimated only at assimilated
cells from the other 60 years.

Three branches are compared on the unassimilated sector:

- fold-5 proxy-only;
- one regional-mean observation computed only from assimilated cells, using
  the previously frozen regional scale `0.0625`;
- spatial observations at assimilated cells, using shrinkage `0.25` and scale
  `0.00390625` from the still-internal spatial v2 boundary branch.

The spatial branch is a mechanism pilot, not a frozen production model: its
scale remains boundary-selected in the same-grid experiment. It may advance
only if it improves unassimilated-sector skill over both controls. Original
inputs remain read-only and `rec_ref` is not used.
