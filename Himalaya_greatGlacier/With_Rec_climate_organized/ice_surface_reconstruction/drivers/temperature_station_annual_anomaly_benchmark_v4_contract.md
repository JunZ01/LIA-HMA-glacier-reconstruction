# Temperature Station Annual Anomaly Benchmark v4

This contract inherits v1-v3. V4 corrects one discovered spatial-support
assumption: a valid GHCN-M station may lie outside the finite production-field
mask. Such a station remains in the observational panel, while its project
comparison is explicitly missing. No extrapolation across the production mask
and no station deletion are allowed. Product-specific station coverage must be
reported.
