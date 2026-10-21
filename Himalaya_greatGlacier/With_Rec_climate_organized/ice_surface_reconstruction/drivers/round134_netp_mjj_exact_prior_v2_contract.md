# Round 134 NETP MJJ Exact-Prior Preflight V2

V1 stopped because its diagnostic gate incorrectly required nonzero variance in all 336 state cells. Read-only diagnosis found exactly three constant southwest-corner cells at 25 N and 63/65/67 E. The same three-cell mask occurs in the already verified Round 129 annual and JJA fields; all 32 NETP cells are variable. V2 changes no exported value and does not rerun the loader.

V2 reuses the immutable V1 MAT export and requires:

1. Exact zero-difference annual closure against both the recovered production prior and Round 129 V3.
2. Exact grid-coordinate and member-year agreement with Round 129 V3.
3. Finite 336 x 848 MJJ data.
4. Exact equality of the MJJ, annual, and JJA zero-variance masks; exactly 333 variable cells and three known constant cells.
5. Nonzero MJJ variance in every one of the 32 NETP cells.
6. MJJ is not value-identical to annual or JJA precipitation.
7. All source fingerprints remain unchanged.

The V1 stop and V2 correction concern only QA interpretation. No proxy, instrumental, QT-Prec, Shenqi, 1921-1939, or frozen-reserve value is read. No EnKF or OGGM run is allowed.

