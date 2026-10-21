"""Append verified Round 94-95 ERA-20C conclusions."""

from __future__ import annotations

import hashlib
from pathlib import Path


EXPECTED_BEFORE = "9a74f57d1aa43044ecdbba0db0e14b7e74370be3e31f02189b9eee3df8266c87"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    project = Path(__file__).resolve()
    while not (project / "LIA_PROJECT_FACTS.md").exists():
        if project.parent == project:
            raise RuntimeError("project-root-not-found")
        project = project.parent
    facts = project / "LIA_PROJECT_FACTS.md"
    before = digest(facts)
    if before != EXPECTED_BEFORE:
        raise RuntimeError(f"facts-hash-conflict:{before}")
    text = facts.read_text(encoding="utf-8")
    if "## 146. ERA-20C metadata and untouched-target readiness" in text:
        raise RuntimeError("round94-95-facts-already-present")
    addition = r'''

## 146. ERA-20C metadata and untouched-target readiness (2026-08-16)

- Loop-v2 Round 94 audited official ECMWF, JAMSTEC, and WDCC metadata without reading climate values. V1 stopped on an ECMWF HTTP 401, V2 on a JAMSTEC DAP HTTP 500, and V3 on two overly literal metadata string checks. V4 changed only transport endpoints and equivalent text matching; failed versions are preserved.
- ECMWF documents ERA-20C as a 1900-2010 atmospheric reanalysis assimilating only surface pressure and surface marine winds. The public JAMSTEC monthly 2 m temperature catalog contains exactly 1,332 months from January 1900 through December 2010, and its NCSS metadata and HTTP file endpoints are publicly accessible.
- Round 94 froze GHCN-M v4 QCU 1981-2000 as a previously unparsed validation target with at least 16 complete station years and the unchanged Round 92 support/performance gates. ERA-20C 1900-1920 remained inaccessible unless every validation gate passed.
- EKF400v2 remains a scientifically relevant 1602-2003 monthly 2-degree candidate, but its 3.31 GiB ensemble-mean file requires registered WDCC access and was deferred.
- Evidence is in `validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_94_era20c_temperature_metadata_readiness_v4/`.

## 147. ERA-20C versus GHCN-M v4 validation (2026-08-16)

- Loop-v2 Round 95 executed the frozen 1981-2000 validation. V1 exposed a Windows path-length issue, V2 exposed the NetCDF4/HDF5 container, and V3 stopped on two unused missing polar rows. V4 retained all data and scientific gates, added the verified HDF5 reader, allowed only the two audited polar rows to be missing, and still required the complete HMA box to be finite.
- The final support contains 69 stations across 6 countries: 27 western, 7 central, and 35 eastern. Median station monthly correlation is 0.742, median monthly RMSE skill is 0.190, and median annual station-pattern correlation is 0.496.
- All regional annual gates pass. Correlation/RMSE skill is 0.907/0.393 in the west, 0.845/0.393 in the central region, 0.756/0.242 in the east, and 0.811/0.271 for HMA overall.
- The candidate nevertheless fails its frozen station-consistency gate: only 68.1% of stations have positive monthly RMSE skill, below the required 75%. Do not weaken this gate after access.
- Decision: reject ERA-20C as authorized early-20th-century HMA temperature evidence. Its 1900-1920 values remain unread, and 1921-1939 was not accessed. No production or OGGM forcing is authorized.
- All 240 ERA-20C monthly requests and hashes are recorded, all local source hashes are unchanged, and the result figures are nonblank. Evidence is in `validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_95_era20c_ghcnm_temperature_value_validation_v4/`.
'''
    facts.write_text(text.rstrip() + addition + "\n", encoding="utf-8")
    print(f"before_sha256={before}")
    print(f"after_sha256={digest(facts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
