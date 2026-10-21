"""Append verified Round 96-97 temperature findings to the project contract."""

from __future__ import annotations

import hashlib
from pathlib import Path


EXPECTED_SHA256 = "04067a626ca11d5fb2fb2e510ad7f7db51e6e2fbc83e1a686baba529cd21618a"


def main() -> int:
    root = next(parent for parent in Path(__file__).resolve().parents if (parent / "LIA_PROJECT_FACTS.md").exists())
    facts = root / "LIA_PROJECT_FACTS.md"
    before = facts.read_bytes()
    actual = hashlib.sha256(before).hexdigest()
    if actual != EXPECTED_SHA256:
        raise RuntimeError(f"facts-hash-conflict:{actual}")
    addition = r"""

## 148. GHCN-M 1901-1920 direct-temperature support boundary (2026-08-16)

- Loop-v2 Round 96 audited official GHCN-M v4 QCU monthly TAVG availability without retaining or using any temperature magnitude. A valid month required a nonmissing value and blank quality flag; a final station required all 12 valid months in at least 15 of the 20 years.
- The HMA inventory contains 218 stations, but only eight satisfy the final rule: seven western, one central, and zero eastern stations across four countries. Full-HMA support fails, with only four stations in the least-supported year and no annual eastern support.
- The western subset alone passes its preregistered route: seven final stations across four countries and at least four complete stations every year, exceeding the required five stations, two countries, and three stations per year.
- Decision: direct 1901-1920 station phase adjudication is permitted only for western HMA (longitude below 80 E). Do not generalize it to central, eastern, or full HMA.
- No 1921-1939 value was accessed. Evidence is in validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_96_ghcnm_1901_1920_temperature_coverage_readiness_v1/.

## 149. Western 1901-1920 direct-temperature phase adjudication (2026-08-16)

- Loop-v2 Round 97 compared the seven-station western GHCN-M network with the corrected six-member project reconstruction and the Round 21 family-balanced public reference. Products were sampled on the same public-grid station cells and used exactly the same station-year support; every series was independently centered, so the experiment adjudicates timing and trend but not absolute level.
- The observed network is robust: leave-one-country-out correlations with the full network have median 0.981 and minimum 0.906. Its 1901-1920 linear trend is +1.878 degC per century.
- The project six-member curve has annual correlation 0.047, centered RMSE 0.661 degC, negative RMSE skill -0.342, trend -2.789 degC per century, and nine opposite-sign years. Its trend is opposite to the direct station network.
- The family-balanced reference is closer but not validated: annual correlation 0.280, centered RMSE 0.476 degC, positive RMSE skill 0.035, trend -0.106 degC per century, and ten opposite-sign years. Its trend also has the wrong sign.
- The reference improves correlation by 0.233 and RMSE by 28.1% relative to the project, but it fails the preregistered trend-sign gate. Decision: the direct western adjudication is inconclusive between products; neither is accepted as a validated 1901-1920 western temperature curve. The evidence nevertheless confirms a material early-20th-century phase/trend failure in the current project ensemble.
- A source-grid latitude offset of 0.004881 degrees initially allowed an adjacent missing row to propagate into one public-grid station cell. The frozen data, metrics, and decision rules were unchanged; finite-neighbor bilinear weights were renormalized as recorded in EXECUTION_ADDENDUM.md. All source hashes remained unchanged and the result figure is nonblank.
- Advance to a frozen member/method/proxy-network attribution of the project ensemble. Do not tune against 1921-1939 or authorize production/OGGM forcing. Evidence is in validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_97_western_ghcnm_temperature_phase_adjudication_v1/.
"""
    text = before.decode("utf-8")
    if "## 148." in text or "## 149." in text:
        raise RuntimeError("target-sections-already-present")
    facts.write_bytes(before + addition.encode("utf-8"))
    print(hashlib.sha256(facts.read_bytes()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
