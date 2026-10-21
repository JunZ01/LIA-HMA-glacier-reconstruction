"""Append the verified western seasonal-temperature PSM result to project facts."""

from pathlib import Path


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
MARKER = "## 271. Western bivariate documented-season temperature OOF closure (2026-08-19)"

TEXT = f"""

{MARKER}

- A frozen modern mechanism test used the already identified west-of-80-E bivariate records that are formally documented temperature proxies with nonannual seasons: 37 proxy6 records (36 JJA, one April) and 49 proxy9 records (48 JJA, one April). It used only 1940-2000, the five Round-08 blocked folds, and both Berkeley-Earth-plus-GPCC and CRU self/cross-product paths.
- The annual control was proxy on annual temperature plus annual precipitation. The candidate changed only the temperature predictor to the documented season and retained the identical annual precipitation covariate. It reused the established Round-118 monthly readers and aggregation implementation.
- Every median skill comparison is negative. Pooled seasonal-versus-annual RMSE changes are -0.4841% for BEA self, -0.8563% for CRU self, -0.5116% for BEA-to-CRU, and -0.1136% for CRU-to-BEA. Only 24.32% of proxy6 and 20.41% of proxy9 records improve on both self-product paths.
- The seasonal temperature coefficient has 70.96% BEA/CRU sign agreement, so the coefficient-sign gate alone passes, but all ten skill and improvement-fraction gates fail. Documented-season temperature mismatch is rejected as the primary cause of the western bivariate PSM anomaly. Do not run an 1861-1900 seasonal EnKF replay or promote a seasonal operator from this outcome-defined subset.
- Original source hashes are unchanged. No 1901-1939 value, EnKF, localization, or OGGM run was used. Evidence is in `validation_data/results/paleo_surface_validation_v0/climate_reconstruction_temperature_west_bivariate_documented_season_oof_v1/`.
"""


def main() -> None:
    original = FACTS.read_text(encoding="utf-8")
    if MARKER in original:
        raise RuntimeError("facts-marker-already-present")
    FACTS.write_text(original.rstrip() + TEXT + "\n", encoding="utf-8")
    print("TEMPERATURE_WEST_BIVARIATE_DOCUMENTED_SEASON_FACTS_UPDATED")


if __name__ == "__main__":
    main()
