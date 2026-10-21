# Round 229: Yang 2010 11-year curve digitization

Digitize only the red 11-year FFT low-pass curve from the Round 228 v5 safe
image. The image exposes data years 1377-1899 and 1940-1998 and has 1900-1939
masked before saving. Map the plot frame to axis years 1350-2000; derive the
zero line and -1.0 axis line from black border pixels; identify red pixels by a
fixed color rule; and extract one median y position per allowed year.

Convert the published whole-series departures to 1961-1990 anomalies using
only visible 1961-1990 values. Do not inspect the PDF or any masked pixel, infer
or interpolate 1900-1939, access Jionglacuo, modify source/production files, or
use the digitized series as production forcing. This is a trajectory benchmark
with pixel and publication-figure uncertainty.
