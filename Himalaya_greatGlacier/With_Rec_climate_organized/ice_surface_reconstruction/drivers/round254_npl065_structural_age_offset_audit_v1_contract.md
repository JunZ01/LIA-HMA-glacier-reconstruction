# Round 254 NPL065 structural age-offset audit contract

Read the Tucson RWL file as bytes and inspect only sample IDs, line start-year
fields, six-character field occupancy, and end markers. Do not convert any
ring-width token to a number.

Recover each sample's first and last occupied calendar year and calculate its
structural ring-index offset at 1941. Verify global 1787-2013 coverage and
modern 1941-2013 depth. These offsets may be used in a later modern-only ADS
sensitivity experiment, but this round computes no chronology or climate
metric.
