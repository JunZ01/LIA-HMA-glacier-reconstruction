# Round 129 Exact Seasonal-Prior Coupling Contract V3

V2 stopped because a direct raw-CESM extraction did not reproduce the archived annual precipitation prior after the production loader's monthly bias correction and nonpositive-value replacement. V3 does not relax that closure gate.

V3 must call the original `load_S_720360` implementation with CESM-LME010, HMA bounds 24-51 N and 62-110 E, member years 1001-1848, and RNG seed 4236 reset before every load. It exports annual temperature, JJA temperature, annual precipitation, JJA precipitation, and previous-October to current-April precipitation. The cross-year precipitation call uses start-year indices 1000-1847 so its output is labelled by ending year 1001-1848.

The exported annual precipitation field must close exactly against the recovered 336 x 848 prior. The proxy operators, modern products, OOF errors, candidate gates, locked-period boundary, and interpretation boundary remain those declared in the V2 contract. No source file may be modified, no 1921-1939 candidate/current climate value may be decoded, the frozen external reserve remains sealed, and OGGM is out of scope.
