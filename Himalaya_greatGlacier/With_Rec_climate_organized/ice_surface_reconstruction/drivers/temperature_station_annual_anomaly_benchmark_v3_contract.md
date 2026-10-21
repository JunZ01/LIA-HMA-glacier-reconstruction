# Temperature Station Annual Anomaly Benchmark v3

This contract inherits the v1 scientific contract and the v2 missing-row
correction. V3 changes only the Windows runtime import order: Rasterio/GDAL is
loaded before h5py to avoid an environment-specific DLL collision. Scientific
inputs, years, gates, calculations, and outputs are unchanged.
