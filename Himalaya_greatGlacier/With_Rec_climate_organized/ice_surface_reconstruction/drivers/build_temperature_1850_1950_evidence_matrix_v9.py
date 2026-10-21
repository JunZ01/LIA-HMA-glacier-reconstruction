from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
V8 = BASE / "temperature_1850_1950_low_frequency_evidence_matrix_v8"
OUTPUT = BASE / "temperature_1850_1950_low_frequency_evidence_matrix_v9"
DRIVER = Path(__file__).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_row(matrix: pd.DataFrame, row: dict[str, object]) -> pd.DataFrame:
    unknown = set(row) - set(matrix.columns)
    if unknown:
        raise RuntimeError(f"unknown-columns:{sorted(unknown)}")
    complete = {column: row.get(column, "") for column in matrix.columns}
    return pd.concat([matrix, pd.DataFrame([complete])], ignore_index=True)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    matrix = pd.read_csv(V8 / "evidence_matrix.csv")

    xiada = matrix.index[matrix.evidence_id == "XIADA_CO_BRGDGT_PENDING_NUMERIC"]
    if len(xiada) != 1:
        raise RuntimeError(f"xiada-identity-failed:{len(xiada)}")
    matrix.loc[xiada[0], "machine_readable"] = False
    matrix.loc[xiada[0], "finding"] = (
        "The Xiada Co record is scientifically well placed for 1500-1879, but no no-login "
        "machine-readable age-temperature table was found. The previously linked TPDC DOI "
        "10.11888/Paleoenv.tpdc.271728 is a Dagze modern calibration dataset and is not Xiada."
    )
    matrix.loc[xiada[0], "next_action"] = "author_or_repository_request_if_the_series_becomes_decision_critical"
    matrix.loc[xiada[0], "source_url"] = "https://doi.org/10.1016/j.quaint.2018.12.018"

    common = {"sealed_period_values_used": False, "production_modified": False}
    matrix = add_row(
        matrix,
        {
            "evidence_id": "KUSAI_LAKE_SUMMER_TEMPERATURE_PENDING_NUMERIC",
            "region": "northern_Tibetan_Plateau",
            "longitude": 93.0,
            "coverage": "approximately_350-2010_CE",
            "target": "JJA_temperature_from_varve_light_layer_thickness",
            "measurement_family": "lake_sediment_varve_calibrated_temperature",
            "machine_readable": False,
            "numeric_now": False,
            "source_independence": "independent_non_tree_lake_sediment",
            "spatial_relevance": "northern_HMA_boundary",
            "season_match": "JJA",
            "admitted_role": "priority_pending_seasonal_temperature_record",
            "finding": (
                "The open primary paper reports a calibrated 350-2010 JJA reconstruction, but its "
                "public supplement does not expose the original age-temperature series as a table."
            ),
            "next_action": "author_or_repository_request_if_needed;do_not_digitize_for_candidate_selection",
            "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4205841/",
            **common,
        },
    )
    matrix = add_row(
        matrix,
        {
            "evidence_id": "YAMZHOG_YUMCO_POLLEN_TEMPERATURE_PENDING_NUMERIC",
            "region": "southern_Tibetan_Plateau",
            "longitude": 91.0,
            "coverage": "approximately_100-1996_CE",
            "target": "July_temperature_or_pollen_temperature_index",
            "measurement_family": "lake_sediment_pollen",
            "machine_readable": False,
            "numeric_now": False,
            "source_independence": "independent_non_tree_lake_sediment",
            "spatial_relevance": "central_Himalaya_southern_margin",
            "season_match": "July_or_qualitative_warm_cold_state_requires_method_audit",
            "admitted_role": "pending_directional_or_quantitative_record",
            "finding": (
                "The primary paper supports a cold/moist interval from about 1400 to 1890 and later "
                "dry conditions, but no public calibrated age-temperature table was found."
            ),
            "next_action": "author_or_repository_request_if_needed;retain_published_state_only_as_context",
            "source_url": "https://doi.org/10.1016/j.palaeo.2018.04.005",
            **common,
        },
    )

    if matrix.evidence_id.duplicated().any():
        raise RuntimeError("duplicate-evidence-id")
    matrix.to_csv(OUTPUT / "evidence_matrix.csv", index=False)

    summary = json.loads((V8 / "summary.json").read_text(encoding="utf-8"))
    summary.update(
        {
            "experiment": "temperature_1850_1950_low_frequency_evidence_matrix_v9",
            "supersedes": "temperature_1850_1950_low_frequency_evidence_matrix_v8",
            "row_count": int(len(matrix)),
            "identity_correction": (
                "Xiada now points to DOI 10.1016/j.quaint.2018.12.018; TPDC DOI "
                "10.11888/Paleoenv.tpdc.271728 is recorded as Dagze modern calibration, not Xiada."
            ),
            "new_pending_records": ["Kusai Lake", "Yamzhog Yumco"],
            "prohibited_source": "20CRv3",
            "prohibited_source_accessed": False,
        }
    )
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    report = """# HMA low-frequency temperature evidence matrix v9

## Identity correction

The Xiada Co row now points to the Xiada paper DOI `10.1016/j.quaint.2018.12.018`.
The former TPDC link `10.11888/Paleoenv.tpdc.271728` is a Dagze modern calibration
dataset, not a Xiada age-temperature series.

## Added search outcomes

- Kusai Lake has a published calibrated JJA temperature reconstruction, but no
  public machine-readable original age-temperature table was found.
- Yamzhog Yumco supplies a published cold/warm environmental history, but no
  public calibrated age-temperature table was found.

These are pending or contextual records, not numerical votes. The v8 scientific
decision is unchanged: a uniform HMA temperature correction is rejected, while
Naqu, PX202, and Qinghai remain the stable conflict channels requiring targeted
operator or calibration audits.

The sealed period was not read. Production files were not modified. EnKF and
OGGM were not run. 20CRv3 was not accessed.
"""
    (OUTPUT / "FINAL_RESULT.md").write_text(report, encoding="utf-8")
    manifest = {
        "inputs": [
            {"path": str(V8 / "evidence_matrix.csv"), "sha256": sha256(V8 / "evidence_matrix.csv")},
            {"path": str(V8 / "summary.json"), "sha256": sha256(V8 / "summary.json")},
        ],
        "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
        "prohibited_source": "20CRv3",
        "prohibited_source_accessed": False,
    }
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
