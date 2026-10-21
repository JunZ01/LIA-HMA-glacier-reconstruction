# Proxy9 Record Provenance v1

Purpose: trace proxy9 records 203 and 241 to their public proxy identities using exact time-series evidence, not coordinate-only guesses.

Hard gates:

- exact coordinate candidate in each `combined_proxydata6.mat` copy;
- sequence correlation at least 0.999999999;
- maximum residual after an affine transform at most 1e-10;
- source hashes unchanged;
- no OGGM and no use of `rec_ref`.

The experiment writes only to its independent result directory. Public species and citation fields are kept separate from the numerical identity proof because the local processed MAT files do not retain those fields.
