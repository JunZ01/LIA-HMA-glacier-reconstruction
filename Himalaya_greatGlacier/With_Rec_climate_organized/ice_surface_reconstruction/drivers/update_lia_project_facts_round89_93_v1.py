"""Append verified Round 89-93 temperature-evidence conclusions."""

from __future__ import annotations

import hashlib
from pathlib import Path


EXPECTED_BEFORE = "cb876c486c43fbb097d8f43dcc5b8c1289616fe2ebff0316a60711d9d6ea30a9"


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
    if "## 143. 20CRv3 temperature metadata eligibility" in text:
        raise RuntimeError("round89-93-facts-already-present")
    addition = r'''

## 143. 20CRv3 temperature metadata eligibility (2026-08-16)

- Loop-v2 Round 89 selected NOAA/CIRES/DOE 20CRv3 as a materially new atmospheric-reanalysis method candidate. Official metadata identify monthly ensemble-mean 2 m air temperature for 1806-2015 and surface-pressure observations as the assimilated atmospheric observations. No temperature value was read in this metadata round.
- The official GHCN-D inventory supplied 74 HMA stations with catalog overlap for paired TMAX/TMIN validation during 2001-2015: 37 western, 8 central, and 29 eastern stations across eight countries. V2 replaced only an unreliable dynamic-webpage text check with equivalent official DAS attributes; all metadata gates then passed.
- Decision: freeze a 2001-2015 station-value validation before accessing 20CRv3 temperature values. This was a candidate-evidence test only and did not authorize production or OGGM forcing.
- Evidence is in `validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_89_20crv3_ghcnd_temperature_metadata_eligibility_v2/`.

## 144. GHCN-D paired-daily target support closure (2026-08-16)

- Loop-v2 Rounds 90-91 froze and executed a paired-daily GHCN-D target: daily TAVG required valid blank-QC TMAX and TMIN, a month required at least 80% paired days, a station year required all 12 valid months, and a station required at least 12 complete years during 2001-2015.
- All 74 official station files were preserved, but only 25 stations met the complete-year rule. Region support was 0 western, 2 central, and 23 eastern stations, so the predeclared support gates failed before any 20CRv3 temperature value was downloaded or read.
- Decision: close the GHCN-D paired-daily target without lowering thresholds. Advance only through a separately frozen, scientifically distinct monthly station product rather than retrofitting the failed target.
- Evidence is in `validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_91_20crv3_ghcnd_temperature_value_validation_v1/`.

## 145. 20CRv3 versus GHCN-M v4 temperature validation (2026-08-16)

- Loop-v2 Round 92 froze a distinct validation against NOAA GHCN-M v4 QCU monthly TAVG over 2001-2015. Round 93 V1 exposed a three-character versus documented two-character GHCN-M country-code parser error; V2-V3 then exposed two NetCDF `time.units` representation differences. The failed versions are preserved. V4 changed only those verified I/O parsers and retained every scientific support and performance gate.
- V4 retained 64 stations across 7 countries: 17 western, 8 central, and 39 eastern. The median station monthly correlation is 0.760, median monthly RMSE skill is 0.320, 90.6% of stations have positive monthly skill, and median annual station-pattern correlation is 0.466. These station-scale and spatial gates pass.
- Regional annual performance is uneven. Correlation/RMSE skill is 0.934/0.634 in the west, 0.776/0.361 in the central region, but only 0.416/-0.046 in the east. The frozen requirements that every region have correlation at least 0.50 and positive RMSE skill therefore fail.
- Decision: reject 20CRv3 as new external whole-HMA temperature-change evidence. Do not weaken the east-region gates, acquire the 1806-2000 field for candidate construction, or use this result to revisit the consumed 1921-1939 test. The result may describe regional modern behavior but does not authorize production or OGGM forcing.
- All local source hashes were unchanged. Only 2001-2015 temperature values were parsed; no 1921-1939 value was accessed. Evidence is in `validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_93_20crv3_ghcnm_v4_temperature_value_validation_v4/`.
'''
    facts.write_text(text.rstrip() + addition + "\n", encoding="utf-8")
    print(f"before_sha256={before}")
    print(f"after_sha256={digest(facts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
