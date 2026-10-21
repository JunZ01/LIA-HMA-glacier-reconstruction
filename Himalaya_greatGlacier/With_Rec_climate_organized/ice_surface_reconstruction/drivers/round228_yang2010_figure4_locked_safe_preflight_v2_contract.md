# Round 228 v2: in-memory locked-safe Figure 4 preflight

Round 228 v1 stopped before rendering because the Poppler Windows wrapper did
not support the requested stdout path. Keep that failed directory untouched.
Use pypdfium2 to render PDF page 6 in memory, cover the rightmost 30 percent,
and save only the masked raster. No unmasked raster, curve values, 1921-1939
values, Jionglacuo material, or production files may be read or written.
