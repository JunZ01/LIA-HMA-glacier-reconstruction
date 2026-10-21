# Round 131 New-Proxy Huber Innovation Contract

Use the completed Round 130 inputs and run only combined Xinpu+Keylong BEA+GPCC and CRU branches for years 1000-1920. All archived proxies, prior members, operators, R values, and `M_update` remain frozen.

For each active new proxy only, compute its fixed-prior innovation standard deviation as `sqrt(var(HXb)+R)`. If the raw innovation exceeds three standard deviations, retain the source observation unchanged on disk but clip the value passed to `M_update` to the prior operator mean plus or minus three standard deviations. Archived proxy innovations are never clipped. This is a robust-likelihood sensitivity, not source cleaning.

Do not access 1921-1939, open the frozen external validation reserve, modify source files, alter localization, or run OGGM. Advancement requires BEA/CRU robust increments to remain coherent, 31-year robust increments to correlate at least 0.95 with their standard Round 130 counterparts, and maximum HMA increments to fall below half the baseline HMA standard deviation.
