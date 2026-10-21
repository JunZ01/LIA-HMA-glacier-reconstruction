# Round 197 East Asia temperature-field metadata audit

## Objective

Determine whether Zenodo record `10.5281/zenodo.18477496`, *A gridded
temperature reconstruction for East Asia during 1400-2000 CE*, is a materially
new external evidence family that can later adjudicate the HMA 1500-1899 warm
bias.

## Frozen boundary

- Read the Zenodo record metadata and `Code.zip` only.
- Do not download or open `Data.zip` or `Results.zip`.
- Do not parse any annual temperature, proxy, model, or reconstruction value.
- Do not access 1921-1939 values or the prohibited Jionglacuo source.
- Do not run reconstruction, correction, EnKF, MATLAB, or OGGM.
- Preserve downloaded files byte-for-byte and write only to the independent
  Round 197 result directory.

## Questions

1. Does the declared space-time domain cover HMA and 1500-1899?
2. Is the output an absolute-temperature field or an anomaly/index?
3. Does the deposited code expose a reproducible workflow and identify its
   proxy, model, calibration, and validation sources?
4. Is the product materially distinct from the project's existing
   four-family comparison and six-member reconstruction, or is it a repackaged
   member/source family?

## Admission decision

Advance to a separate locked-safe numeric preflight only if the metadata and
code establish all of the following:

- HMA spatial coverage and continuous coverage of 1500-1899;
- an interpretable temperature quantity and spatial coordinates;
- enough source and method information to audit overlap;
- no evidence that the final product is a direct copy or deterministic
  repackaging of an existing project reconstruction.

Source overlap may permit comparison-only use, but it must be reported and
prevents an unqualified independent-validation claim.
