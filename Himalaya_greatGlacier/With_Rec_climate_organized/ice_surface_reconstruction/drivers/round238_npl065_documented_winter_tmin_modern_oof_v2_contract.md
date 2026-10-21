# Round 238 v2 contract: NPL065 documented winter Tmin modern OOF

This contract preserves every scientific choice and threshold from
`round238_npl065_documented_winter_tmin_modern_oof_v1_contract.md`.

The only implementation change is that the Tucson RWL source is read as raw
bytes. Lines without a valid year in columns 9-12 are skipped as descriptive
headers. Data-line sample IDs and tokens are decoded as ASCII only after that
structural test. This is required because the official NOAA file contains a
non-ASCII character in its descriptive header.

The failed v1 directory is retained. V1 stopped on header decoding before any
ring-width or climate magnitude was processed and has no scientific result.

All v1 boundaries remain frozen: original files are read-only; 1921-1939 and
pre-1941 ring-width values are skipped before numeric conversion; no paleo
curve, Jionglacuo, EnKF, production reconstruction, correction, or OGGM run is
allowed.
