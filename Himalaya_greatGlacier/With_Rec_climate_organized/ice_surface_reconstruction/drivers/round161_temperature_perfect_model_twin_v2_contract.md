# Round 161 Temperature Perfect-Model Twin v2 Contract

V1 stopped before any twin reconstruction because it paired the 14-by-24 prior
state coordinates with the 43-by-78 instrumental-grid HMA mask. V2 changes only
that technical input: native `grid_lat`, `grid_lon`, and `hma_mask` are read from
the already verified `heldout_oof_fields.npz` archive. All truth members, PSMs,
R values, methods, random seed, metrics, gates, and scientific boundaries remain
identical to the v1 contract.
