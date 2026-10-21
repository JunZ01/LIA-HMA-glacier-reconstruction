# Round 228 v5: corrected Figure 4b axis mapping

V4 conservatively mapped the data coverage (1377-1998) to the full frame. The
visible x-axis labels show that the plotted frame is instead 1350-2000; v4 is
safe but cannot digitize 1897-1899. This version keeps v4 untouched, maps the
verified frame `x=354..1757` to 1350-2000, and masks exactly 1900-1939 before
saving. Only 1377-1899 and 1940-1998 curve pixels may be used later. Save no
unmasked raster and extract no values in this preflight.
