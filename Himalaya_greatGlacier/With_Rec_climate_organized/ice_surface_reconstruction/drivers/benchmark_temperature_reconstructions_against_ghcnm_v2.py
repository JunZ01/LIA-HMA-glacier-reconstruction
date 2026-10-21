from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import benchmark_temperature_reconstructions_against_ghcnm_v1 as core


core.OUT = (
    Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
    / "temperature_reconstruction_external_instrumental_benchmark_v2"
)


def extract_reference_families_common_support(
    stations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    with h5py.File(core.PUBLIC, "r") as handle:
        years = np.asarray(handle["time"], dtype=int).ravel()
        selected_indices = np.flatnonzero((years >= 1951) & (years <= 2000))
        selected_years = years[selected_indices]
        if not np.array_equal(selected_years, np.arange(1951, 2001)):
            raise RuntimeError("Reference reconstruction 1951-2000 year contract failed")
        baseline_full_indices = np.flatnonzero((years >= 1961) & (years <= 1990))
        baseline_selected = (selected_years >= 1961) & (selected_years <= 1990)
        lat = np.asarray(handle["lat"], dtype=float).ravel()
        lon = np.asarray(handle["lon"], dtype=float).ravel()

        common_support = np.ones((len(lat), len(lon)), dtype=bool)
        for variable in core.MEMBERS.values():
            baseline_values = core.finite(handle[variable][baseline_full_indices, :, :])
            common_support &= np.isfinite(baseline_values).sum(axis=0) >= 20
        if not int(common_support.sum()):
            raise RuntimeError("No common 1961-1990 support across public reconstruction members")

        station_rows = []
        location_rows = []
        for station in stations.itertuples(index=False):
            distance = np.hypot(
                lat[:, None] - float(station.latitude),
                lon[None, :] - float(station.longitude),
            )
            distance[~common_support] = np.inf
            row, col = np.unravel_index(np.argmin(distance), distance.shape)
            source_distance = float(distance[row, col])
            if not np.isfinite(source_distance) or source_distance > 2.0:
                raise RuntimeError(f"No common reference support within 2 degrees for {station.station_id}")
            member_values = {}
            for member, variable in core.MEMBERS.items():
                values = core.finite(handle[variable][selected_indices, row, col])
                member_values[member] = values - np.nanmean(values[baseline_selected])
            local_enkf = core.finite_mean([member_values["HGAO"], member_values["O"]])
            neukom = core.finite_mean([
                values for name, values in member_values.items() if name.startswith("Nk_")
            ])
            families = {
                "reference_local_enkf_degC": local_enkf,
                "reference_lmr_degC": member_values["LMR"],
                "reference_neukom_degC": neukom,
                "reference_phyda_degC": member_values["PHYDA"],
            }
            families["reference_family_equal_degC"] = core.finite_mean(list(families.values()))
            location_rows.append({
                "station_id": station.station_id,
                "station_name": station.station_name,
                "station_latitude": float(station.latitude),
                "station_longitude": float(station.longitude),
                "source_latitude": float(lat[row]),
                "source_longitude": float(lon[col]),
                "source_distance_deg": source_distance,
                "common_support_cell_count": int(common_support.sum()),
            })
            for index, year in enumerate(selected_years):
                station_rows.append({
                    "station_id": station.station_id,
                    "year": int(year),
                    **{name: float(values[index]) for name, values in families.items()},
                })
    return pd.DataFrame(station_rows), pd.DataFrame(location_rows)


core.extract_reference_families = extract_reference_families_common_support


if __name__ == "__main__":
    core.main()
