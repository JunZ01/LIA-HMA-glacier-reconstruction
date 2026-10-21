"""Build reference values used by the reusable LIA diagnostic plots.

Area and modern volume are physical observations/estimates.  The length
reference is deliberately expressed on the OGGM model-flowline scale: the
registered modern-to-LIA length ratio is applied to the terminal LIA model
length.  This avoids plotting incompatible absolute length definitions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build(
    chain_dir: Path,
    lia_targets: Path,
    system_manifest: Path,
    modern_tau: Path,
    output: Path,
) -> pd.DataFrame:
    chain = pd.read_csv(chain_dir / "selected_chain_summary.csv")
    spin = pd.read_csv(chain_dir / "selected_spinup_timeseries.csv")
    targets = pd.read_csv(lia_targets)
    manifest = pd.read_csv(system_manifest)
    tau = pd.read_csv(modern_tau)
    tau = tau[tau["bed_scenario"].eq("farinotti2019")].copy()
    tau = tau.drop_duplicates("rgi_id")

    ids = chain["lia_unique_id"].astype(int).tolist()
    terminal = (
        spin[spin["lia_unique_id"].astype(int).isin(ids)]
        .sort_values("year")
        .groupby("lia_unique_id", as_index=False)
        .tail(1)[["lia_unique_id", "length_m"]]
        .rename(columns={"length_m": "lia_model_length_m"})
    )
    target_cols = targets[[
        "lia_unique_id", "target_lia_area_km2", "target_main_flowline_length_m"
    ]].rename(columns={
        "target_lia_area_km2": "lia_area_km2",
        "target_main_flowline_length_m": "lia_registered_length_m",
    })
    manifest_cols = manifest[[
        "lia_unique_id", "main_rgi_id", "modern_rgi_area_sum_km2",
        "farinotti_volume_km3",
    ]].rename(columns={
        "main_rgi_id": "rgi_id",
        "modern_rgi_area_sum_km2": "rgi_area_km2",
        "farinotti_volume_km3": "rgi_volume_km3",
    })
    tau_cols = tau[["rgi_id", "modern_main_centerline_length_km"]]

    refs = chain[["lia_unique_id"]].copy()
    refs["lia_unique_id"] = refs["lia_unique_id"].astype(int)
    refs = refs.merge(target_cols, on="lia_unique_id", validate="one_to_one")
    refs = refs.merge(manifest_cols, on="lia_unique_id", validate="one_to_one")
    refs = refs.merge(terminal, on="lia_unique_id", validate="one_to_one")
    refs = refs.merge(tau_cols, on="rgi_id", validate="one_to_one")
    refs["rgi_registered_length_m"] = refs["modern_main_centerline_length_km"] * 1000.0
    refs["rgi_model_length_m"] = (
        refs["lia_model_length_m"]
        * refs["rgi_registered_length_m"]
        / refs["lia_registered_length_m"]
    )
    refs["lia_area_m2"] = refs["lia_area_km2"] * 1e6
    refs["rgi_area_m2"] = refs["rgi_area_km2"] * 1e6
    refs["rgi_volume_m3"] = refs["rgi_volume_km3"] * 1e9
    refs["lia_length_m"] = refs["lia_model_length_m"]
    refs["rgi_length_m"] = refs["rgi_model_length_m"]
    refs["length_reference_mode"] = "model-scale (registered ratio applied)"
    refs = refs.sort_values("lia_unique_id")
    output.parent.mkdir(parents=True, exist_ok=True)
    refs.to_csv(output, index=False)
    return refs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chain-dir", required=True, type=Path)
    parser.add_argument("--lia-targets", required=True, type=Path)
    parser.add_argument("--system-manifest", required=True, type=Path)
    parser.add_argument("--modern-tau", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    frame = build(args.chain_dir, args.lia_targets, args.system_manifest, args.modern_tau, args.output)
    print(f"references:{len(frame)} output:{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
