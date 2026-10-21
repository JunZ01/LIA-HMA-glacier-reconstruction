# Round 142 Southern Tibetan Plateau Transfer-Failure Attribution V1

## Objective

Attribute the failed Round 141 modern transfer to one of two predeclared mechanisms: disagreement between GPCC and CRU regional seasonal targets, or temporal instability of the fixed proxy-target relationships.

## Frozen inputs and calculations

- Reuse exactly the Round 141 region, 38 proxy records, 19 spatial groups, two documented cross-year seasons, 30 grid cells, and 1941-2000 modern data.
- Use the same two disjoint periods, 1941-1970 and 1971-2000, and 11-year centered smoothing confined inside each period.
- For each season and period, calculate the GPCC-CRU target correlation and standardized trend signs.
- For every proxy, season, and product, calculate its early- and late-period correlation with the corresponding target. Report sign agreement, absolute correlation change, and product agreement. No proxy is removed or reweighted.
- The attribution is diagnostic only. It does not create a reconstruction and cannot tune a season, box, proxy subset, weight, or acceptance threshold.

## Frozen interpretation

- Instrumental targets are called coherent only if all four season-period GPCC-CRU correlations are at least 0.80 and all four trend signs agree.
- A season/product proxy relation is called temporally stable only if at least 60 percent of finite proxy correlations retain sign and the median absolute early-late correlation change is no greater than 0.40.
- If instrumental targets are coherent but any season fails proxy stability for both products, proxy-target temporal instability is the dominant diagnosed mechanism and the next admissible method is a proxy-specific cross-validated PSM redesign.
- If instrumental targets are not coherent, the regional target/product problem must be resolved before any PSM redesign.

No 1921-1939 value, frozen external reserve, EnKF, or OGGM use is permitted. Source files remain read-only.
