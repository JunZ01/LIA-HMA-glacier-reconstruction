"""Read-only observation-domain audit for all saved Core-191 score sources.

No OGGM model is advanced. The audit reads saved scalar time series and the
Core-191 elevation-band audits, then classifies each deduplicated chain as
valid, definite overflow, or unresolved legacy overflow thickness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .build_lia_191_iterative_loop import score_sources
from .lia_observation_domain import saved_series_observation_domain_assessment


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_output_dir() -> Path:
    return (
        project_root()
        / "validation_data"
        / "results"
        / "paleo_surface_validation_v0"
        / "lia_191_controller_v1"
        / "diagnostics"
        / "observation_domain_gate_v1"
    )


def elevation_band_root() -> Path:
    return (
        project_root()
        / "validation_data"
        / "results"
        / "paleo_surface_validation_v0"
        / "lia_elevation_band_core191_length_corrected_v1"
    )


def _as_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _node_contract(system_id: int) -> tuple[int, float]:
    path = elevation_band_root() / str(system_id) / "elevation_band_audit.json"
    audit = json.loads(path.read_text(encoding="utf-8"))
    return int(audit["ice_node_count"]), float(audit["node_dx_m"])


def _load_source_rows() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for source in score_sources():
        if not source.path.exists():
            continue
        frame = pd.read_csv(source.path)
        if frame.empty:
            continue
        frame["score_source"] = source.label
        frame["score_priority"] = source.priority
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("no-saved-score-sources")
    rows = pd.concat(frames, ignore_index=True, sort=False)
    rows["lia_unique_id"] = pd.to_numeric(
        rows["lia_unique_id"], errors="raise"
    ).astype(int)
    rows = rows.sort_values("score_priority").drop_duplicates(
        "fingerprint", keep="last"
    )
    return rows.reset_index(drop=True)


def audit_saved_chains() -> pd.DataFrame:
    rows = _load_source_rows()
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows.to_dict(orient="records"), start=1):
        system_id = int(row["lia_unique_id"])
        node_count, dx_m = _node_contract(system_id)
        output_dir = Path(str(row.get("output_dir", "")))
        series_path = output_dir / "timeseries.csv"
        record: dict[str, Any] = {
            "fingerprint": row.get("fingerprint"),
            "task_id": row.get("task_id"),
            "lia_unique_id": system_id,
            "score_source": row.get("score_source"),
            "score_priority": row.get("score_priority"),
            "delta_t": row.get("delta_t"),
            "old_accepted": _as_bool(row.get("accepted")),
            "old_acceptance_state": row.get("acceptance_state"),
            "old_forward_peak_m": row.get("forward_peak_m"),
            "old_forward_2000_m": row.get("forward_2000_m"),
            "output_dir": str(output_dir),
            "observation_domain_node_count": node_count,
            "observation_domain_length_m": node_count * dx_m,
            "timeseries_found": series_path.exists(),
        }
        if not series_path.exists():
            record.update(
                {
                    "observation_domain_assessed": False,
                    "observation_domain_overflow": False,
                    "observation_domain_unresolved": True,
                    "length_observation_valid": False,
                    "observation_domain_state": "missing-timeseries",
                }
            )
            records.append(record)
            continue

        series = pd.read_csv(series_path)
        assessment = saved_series_observation_domain_assessment(
            series, node_count, node_dx_m=dx_m
        )
        outside = assessment["outside_rows"]
        overflow = assessment["overflow_rows"]
        unresolved = assessment["unresolved_rows"]
        record.update(
            {
                "observation_domain_assessed": assessment["assessed"],
                "observation_domain_overflow": assessment["overflow"],
                "observation_domain_unresolved": assessment["unresolved"],
                "length_observation_valid": assessment["valid"],
                "observation_domain_state": assessment["state"],
                "explicit_new_gate_columns": assessment["explicit"],
                "any_ice_outside_observation_domain": bool(outside.any()),
                "outside_saved_row_count": int(outside.sum()),
                "overflow_saved_row_count": int(overflow.sum()),
                "unresolved_saved_row_count": int(unresolved.sum()),
                "first_outside_phase": (
                    None if not outside.any() else str(series.loc[outside, "phase"].iloc[0])
                ),
                "first_outside_year": (
                    None if not outside.any() else int(series.loc[outside, "year"].iloc[0])
                ),
                "first_overflow_phase": (
                    None if not overflow.any() else str(series.loc[overflow, "phase"].iloc[0])
                ),
                "first_overflow_year": (
                    None if not overflow.any() else int(series.loc[overflow, "year"].iloc[0])
                ),
                "maximum_raw_length_m": float(
                    pd.to_numeric(series.get("length_m"), errors="coerce").max()
                ),
                "maximum_length_gt_10m_m": float(
                    pd.to_numeric(series.get("length_gt_10m"), errors="coerce").max()
                ),
            }
        )
        records.append(record)
        if index % 250 == 0 or index == len(rows):
            print(f"audited {index}/{len(rows)} saved chains", flush=True)
    return pd.DataFrame(records)


def write_outputs(audit: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "chain_observation_domain_audit.csv"
    audit.to_csv(audit_path, index=False)

    system = (
        audit.groupby("lia_unique_id", as_index=False)
        .agg(
            saved_chain_count=("fingerprint", "count"),
            valid_chain_count=("length_observation_valid", "sum"),
            overflow_chain_count=("observation_domain_overflow", "sum"),
            unresolved_chain_count=("observation_domain_unresolved", "sum"),
            any_ice_outside_chain_count=("any_ice_outside_observation_domain", "sum"),
            old_accepted_chain_count=("old_accepted", "sum"),
        )
        .sort_values("lia_unique_id")
    )
    system["system_has_overflow"] = system["overflow_chain_count"].gt(0)
    system["system_has_unresolved"] = audit.groupby("lia_unique_id")[
        "observation_domain_state"
    ].apply(lambda values: values.astype(str).eq("observation-domain-unresolved").any()).reindex(
        system["lia_unique_id"]
    ).to_numpy()
    system.to_csv(output_dir / "system_observation_domain_summary.csv", index=False)

    source = (
        audit.groupby("score_source", as_index=False)
        .agg(
            saved_chain_count=("fingerprint", "count"),
            valid_chain_count=("length_observation_valid", "sum"),
            overflow_chain_count=("observation_domain_overflow", "sum"),
            unresolved_chain_count=("observation_domain_unresolved", "sum"),
        )
        .sort_values("score_source")
    )
    source.to_csv(output_dir / "source_observation_domain_summary.csv", index=False)

    impacted = audit[
        audit["old_accepted"]
        & ~audit["length_observation_valid"]
    ].copy()
    impacted.to_csv(output_dir / "old_accepted_candidates_invalidated.csv", index=False)

    summary = {
        "mode": "read_only_saved_timeseries_audit",
        "saved_chain_count": int(len(audit)),
        "system_count": int(audit["lia_unique_id"].nunique()),
        "valid_chain_count": int(audit["length_observation_valid"].sum()),
        "overflow_chain_count": int(audit["observation_domain_overflow"].sum()),
        "unresolved_saved_row_evidence_chain_count": int(
            audit["observation_domain_unresolved"].sum()
        ),
        "exclusive_unresolved_chain_count": int(
            audit["observation_domain_state"]
            .astype(str)
            .eq("observation-domain-unresolved")
            .sum()
        ),
        "system_with_overflow_count": int(system["system_has_overflow"].sum()),
        "system_with_unresolved_count": int(system["system_has_unresolved"].sum()),
        "old_accepted_candidate_count": int(audit["old_accepted"].sum()),
        "old_accepted_invalidated_count": int(len(impacted)),
        "thickness_threshold_m": 10.0,
        "legacy_rule": (
            "definite overflow is inferred from saved scalar evidence; chains "
            "with ice outside the domain but insufficient thickness evidence "
            "are unresolved and cannot enter length scoring"
        ),
        "chain_audit": str(audit_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = audit_saved_chains()
    summary = write_outputs(audit, args.output_dir)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
