# Round 138 NETP Structural Sensitivity Envelope Contract

## Purpose

Package the verified Round 134, 136, and 137 temporal sensitivity branches into
one transparent structural envelope.

## Members

Use exactly 18 product-marginalized 31-year curves:

- The full Round 134 candidate.
- Fifteen Round 136 leave-one-spatial-group-out curves.
- The product mean of the two Round 137 early-calibration curves.
- The product mean of the two Round 137 late-calibration curves.

No member selection, weighting, rescaling, or tuning is allowed. The annual
median, 5/25/75/95 percentiles, and full range are sensitivity summaries, not
probabilistic confidence or credible intervals.

## Gates

- All upstream rounds must have passed their frozen gates.
- All 18 members must be finite over 1515-1900.
- Every member must correlate at least 0.90 with the full candidate over both
  1515-1900 and 1800-1900.
- Every member must retain a positive 1800-1900 trend.
- The ensemble median must remain closer to withheld proxy 15 than QT-Prec by
  the frozen 0.10 correlation-difference rule over 1800-1900.

Passing establishes a structurally robust NETP MJJ temporal candidate and an
explicit sensitivity envelope. It does not establish spatial fields or permit
production replacement, EnKF, or OGGM.

No 1921-1939 value and no frozen external-reserve value may be read.
