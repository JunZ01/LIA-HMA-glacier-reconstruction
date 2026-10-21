# Round 198 remote Data.zip inventory

## Objective

Inventory the file names, compressed sizes, checksums, and directory structure
inside Zenodo record `18477496` `Data.zip` without downloading or opening any
member payload. Use the inventory to identify the model ensemble and proxy
workbook needed for a later source-overlap audit.

## Boundary

- Read only HTTP headers, the ZIP end record, and central-directory bytes.
- Do not download the 1.2 GB archive or any ZIP member.
- Do not parse climate, proxy, model, reconstruction, 1921-1939, or Jionglacuo
  values.
- Do not run MATLAB, reconstruction, EnKF, correction, or OGGM.
- Write only to the independent Round 198 directory.

This round cannot admit the numeric reconstructed field. It may only decide
whether a targeted, separately contracted source-identity extraction is useful.

