"""Append verified Round 98-99 temperature mechanism findings."""

from __future__ import annotations

import hashlib
from pathlib import Path


EXPECTED_SHA256 = "d523009867d5481aed48e5c3f622eb2727848da5cfa9ca2be194d63cc78efce3"


def main() -> int:
    root = next(parent for parent in Path(__file__).resolve().parents if (parent / "LIA_PROJECT_FACTS.md").exists())
    facts = root / "LIA_PROJECT_FACTS.md"
    before = facts.read_bytes()
    actual = hashlib.sha256(before).hexdigest()
    if actual != EXPECTED_SHA256:
        raise RuntimeError(f"facts-hash-conflict:{actual}")
    addition = r"""

## 150. Western early-temperature member attribution (2026-08-16)

- Loop-v2 Round 98 sampled all six corrected project temperature members on the exact Round 97 western station cells and station-year support, then tested individual members, method composites, proxy-network-size composites, and every leave-one-member-out ensemble.
- All six members have the wrong 1901-1920 trend sign. Their trends range from -3.905 to -1.923 degC per century versus +1.878 degC per century in the robust station network. The best member, AO_6, has correlation 0.082 and negative RMSE skill -0.258.
- No method, network-size, or leave-one-member-out composite passes the frozen route requiring at least +0.15 correlation gain, at least 10% lower RMSE, and the observed trend sign. In 1917 the station network is +0.456 degC while every member lies between -1.690 and -1.115 degC.
- Decision: close nonnegative member weighting as a repair mechanism. A convex combination cannot reverse a trend shared by all six members. Route to their common reconstruction pathway rather than tuning ensemble weights.
- All source hashes are unchanged, no 1921-1939 value was accessed, and the curve figure is nonblank. Evidence is in validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_98_project_temperature_member_attribution_v1/.

## 151. Western early-temperature prior/increment mechanism (2026-08-16)

- Loop-v2 Round 99 reused the verified Round 25 strategy-C recovery and decomposed each 1901-1920 station-support member as posterior = background + assimilation increment. It requested only 1901-1920 plus the 1961-1990 anomaly baseline.
- Component closure is 2.22e-16 degC. Recovered native posterior curves reproduce the exported-member curves with median/minimum correlation 0.9951/0.9936, so the mechanism result is not a replay or downscaling artifact.
- The station trend is +1.878 degC per century. The recovered six-member posterior trend is -2.435, its background trend is +0.702, and its increment trend is -3.137 degC per century. The increment has the posterior sign and equals 128.8% of its absolute negative trend, overwhelming a background that points in the observed direction.
- Both ordinary EnKF members have fixed zero-trend backgrounds and wrong-sign increment trends (-2.327 and -3.372 degC per century). Thus adaptive analog selection is not necessary for the failure. In 1917 the ensemble posterior/background/increment decomposition is -1.371/-0.122/-1.249 degC.
- Decision: the proxy-assimilation increment is the dominant and sufficient source of the early-20th-century western phase/trend failure. Keep prior replacement, localization, and member weighting closed; advance to station-targeted proxy innovation-times-gain attribution and then a network-wide PSM/chronology repair only if the contribution pattern is broad.
- All source hashes are unchanged, no 1921-1939 value was accessed, and the curve figure is nonblank. Evidence is in validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_99_temperature_prior_increment_station_decomposition_v1/.
"""
    text = before.decode("utf-8")
    if "## 150." in text or "## 151." in text:
        raise RuntimeError("target-sections-already-present")
    facts.write_bytes(before + addition.encode("utf-8"))
    print(hashlib.sha256(facts.read_bytes()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
