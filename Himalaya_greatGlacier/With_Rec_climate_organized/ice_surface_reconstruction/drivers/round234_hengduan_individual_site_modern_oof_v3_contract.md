# Round 234 V3 Contract: Hengduan Individual-Site Modern OOF

V2 passed the five-site climate-target preflight and stopped before chronology
or skill calculation because dplpy received source series with no observations
in 1940--2000. A modern-coverage-only audit found seven such columns across the
five sites. Every nonempty modern series has at least 18 observations.

V3 drops only columns that are entirely empty over 1940--2000 before calling
dplpy. It records both the source sample count and the retained modern sample
count. No partially observed series is removed. The V2 climate aggregation fix
and every V1 scientific choice, gate, family test, interpretation boundary,
hash check, and locked-data prohibition remain unchanged.

