from __future__ import annotations

import concurrent.futures
import hashlib
import io
import json
from datetime import date
from pathlib import Path
from xml.etree import ElementTree

import matplotlib
import numpy as np
import pandas as pd
import requests
from scipy.io import netcdf_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
OUT = LOOP / "round_221_regen_netp_mjj_data_recovery_v1"
RAW = OUT / "official_ncss_subsets_unmodified"
CONTRACT = OUT / "ROUND_CONTRACT.md"
DRIVER = Path(__file__).resolve()
BASE = "https://dapds00.nci.org.au/thredds"
CATALOG_URL = f"{BASE}/catalog/ks32/CLEX_Data/REGEN_AllStns/v1-2019/catalog.xml"
README_URL = f"{BASE}/fileServer/ks32/CLEX_Data/REGEN_AllStns/readme.txt"
LICENSE_URL = f"{BASE}/fileServer/ks32/CLEX_Data/REGEN_AllStns/license.txt"
QUALITY_URL = f"{BASE}/fileServer/ks32/CLEX_Data/REGEN_AllStns/v1-2019/REGEN_AllStns_V1-2019_1950-2016_QualityMask.nc"
TARGET_LAT = np.asarray([35, 37, 39, 41] * 8, dtype=float)
TARGET_LON = np.repeat(np.arange(93, 109, 2), 4).astype(float)
YEARS = np.arange(1950, 2001, dtype=int)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "sha256": sha256(path)}


def fetch(url: str, path: Path) -> None:
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    path.write_bytes(response.content)


def year_url(year: int) -> str:
    filename = f"REGEN_AllStns_V1-2019_{year}.nc"
    path = f"ks32/CLEX_Data/REGEN_AllStns/v1-2019/{filename}"
    return (
        f"{BASE}/ncss/grid/{path}?var=p&north=42&west=92&east=108&south=34"
        f"&horizStride=1&time_start={year}-01-01T00%3A00%3A00Z"
        f"&time_end={year}-12-31T23%3A59%3A59Z&accept=netcdf"
    )


def brackets(axis: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    upper = np.searchsorted(axis, target, side="right")
    lower = upper - 1
    fraction = (target - axis[lower]) / (axis[upper] - axis[lower])
    return lower, upper, fraction


def interpolate(values: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    il, iu, fy = brackets(lat, TARGET_LAT)
    jl, ju, fx = brackets(lon, TARGET_LON)
    output = np.empty((values.shape[0], 32), dtype=float)
    for column in range(32):
        output[:, column] = (
            values[:, il[column], jl[column]] * (1 - fy[column]) * (1 - fx[column])
            + values[:, il[column], ju[column]] * (1 - fy[column]) * fx[column]
            + values[:, iu[column], jl[column]] * fy[column] * (1 - fx[column])
            + values[:, iu[column], ju[column]] * fy[column] * fx[column]
        )
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    for path in (CONTRACT, DRIVER, ROOT / "LIA_PROJECT_FACTS.md"):
        if not path.exists():
            raise FileNotFoundError(path)

    fixed_downloads = {
        "catalog.xml": CATALOG_URL,
        "readme.txt": README_URL,
        "license.txt": LICENSE_URL,
        "REGEN_AllStns_V1-2019_1950-2016_QualityMask.nc": QUALITY_URL,
    }
    for name, url in fixed_downloads.items():
        fetch(url, RAW / name)
    ElementTree.parse(RAW / "catalog.xml")
    readme = (RAW / "readme.txt").read_text(encoding="utf-8")
    if "GPCC Full Data Monthly Version 2018" not in readme or "GHCN-Daily" not in readme:
        raise RuntimeError("REGEN source-family statement changed")

    def fetch_year(year: int) -> tuple[int, str]:
        path = RAW / f"REGEN_AllStns_V1-2019_{year}_NETP_subset.nc"
        fetch(year_url(year), path)
        return year, sha256(path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        downloaded = dict(executor.map(fetch_year, [int(year) for year in YEARS]))

    quality_path = RAW / "REGEN_AllStns_V1-2019_1950-2016_QualityMask.nc"
    with netcdf_file(quality_path, "r", mmap=False) as handle:
        quality_lat = np.asarray(handle.variables["lat"][:], dtype=float).copy()
        quality_lon = np.asarray(handle.variables["lon"][:], dtype=float).copy()
        quality = np.asarray(handle.variables["p"][:], dtype=int).copy()
        flags = np.asarray(getattr(handle.variables["p"], "flag_values"), dtype=int)
        meanings = getattr(handle.variables["p"], "flag_meanings", b"")
    if not np.array_equal(flags, np.asarray([-99999, 0, 1])) or meanings != b"ocean land_low_quality land_high_quality":
        raise RuntimeError("Unexpected REGEN quality flags")
    q_il, q_iu, _ = brackets(quality_lat, TARGET_LAT)
    q_jl, q_ju, _ = brackets(quality_lon, TARGET_LON)
    eligible = np.asarray([
        np.all(quality[[q_il[i], q_iu[i]]][:, [q_jl[i], q_ju[i]]] == 1)
        for i in range(32)
    ])
    if int(eligible.sum()) < 20:
        raise RuntimeError("Too few high-quality REGEN target cells")

    annual = np.full((len(YEARS), 32), np.nan)
    source_rows = []
    for row, year in enumerate(YEARS):
        path = RAW / f"REGEN_AllStns_V1-2019_{year}_NETP_subset.nc"
        with netcdf_file(path, "r", mmap=False) as handle:
            lat = np.asarray(handle.variables["lat"][:], dtype=float).copy()
            lon = np.asarray(handle.variables["lon"][:], dtype=float).copy()
            precipitation = np.asarray(handle.variables["p"][:], dtype=float).copy()
            units = getattr(handle.variables["p"], "units", b"")
        expected_days = (date(int(year), 12, 31) - date(int(year), 1, 1)).days + 1
        if precipitation.shape != (expected_days, 9, 17) or units != b"mm/day":
            raise RuntimeError(f"Unexpected REGEN subset structure: {year}")
        target_daily = interpolate(precipitation, lat, lon)
        start = date(int(year), 5, 1).timetuple().tm_yday - 1
        stop = date(int(year), 7, 31).timetuple().tm_yday
        mjj = target_daily[start:stop]
        if mjj.shape[0] != 92 or not np.isfinite(mjj[:, eligible]).all():
            raise RuntimeError(f"Incomplete eligible MJJ data: {year}")
        annual[row] = np.mean(mjj, axis=0)
        source_rows.append({"year": int(year), "file": path.name, "bytes": path.stat().st_size, "sha256": downloaded[int(year)]})

    np.savez_compressed(
        OUT / "regen_netp_mjj_1950_2000.npz",
        years=YEARS,
        latitude=TARGET_LAT,
        longitude=TARGET_LON,
        high_quality_target=eligible,
        mjj_mm_day=annual,
    )
    pd.DataFrame(source_rows).to_csv(OUT / "official_subset_manifest.csv", index=False)
    pd.DataFrame({"latitude": TARGET_LAT, "longitude": TARGET_LON, "high_quality_target": eligible, "mjj_mean_mm_day": annual.mean(axis=0), "mjj_sd_mm_day": annual.std(axis=0, ddof=1)}).to_csv(OUT / "regen_target_cell_inventory.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), gridspec_kw={"wspace": 0.26})
    mask_map = eligible.reshape(8, 4).T
    axes[0].imshow(mask_map, origin="lower", aspect="auto", cmap="Greys", vmin=0, vmax=1, extent=[92, 108, 34, 42])
    axes[0].set(title="REGEN four-neighbor quality gate", xlabel="Longitude", ylabel="Latitude")
    regional = np.average(annual[:, eligible], axis=1, weights=np.cos(np.deg2rad(TARGET_LAT[eligible])))
    axes[1].plot(YEARS, regional, color="#222222", linewidth=1.2)
    axes[1].plot(YEARS, pd.Series(regional).rolling(11, center=True, min_periods=11).mean(), color="#B33A3A", linewidth=1.8)
    axes[1].set(title="REGEN NETP MJJ high-quality support", xlabel="Year", ylabel="mm/day")
    axes[1].grid(axis="y", color="#DDDDDD", linewidth=0.6)
    figure.savefig(OUT / "regen_netp_mjj_recovery.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    tracked = {"contract": CONTRACT, "driver": DRIVER, "facts": ROOT / "LIA_PROJECT_FACTS.md", **{name: RAW / name for name in fixed_downloads}}
    source_states = {name: state(path) for name, path in tracked.items()}
    summary = {
        "round": 221,
        "decision": "admit_regen_as_source_correlated_independent_gridding_diagnostic",
        "doi": "10.25914/5ca4c380b0d44",
        "license": "CC BY-NC-SA 4.0",
        "source_role": "independent_gridding_method_but_source_correlated_with_gpcc_and_ghcn",
        "years": [1950, 2000],
        "downloaded_year_count": len(source_rows),
        "target_cell_count": 32,
        "high_quality_target_count": int(eligible.sum()),
        "all_eligible_cell_years_have_92_finite_mjj_days": True,
        "source_states": source_states,
        "locked_1901_1939_values_read": False,
        "proxy_or_project_climate_values_read": False,
        "jionglacuo_accessed": False,
        "enkf_run": False,
        "production_changed": False,
        "oggm_run": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (OUT / "FINAL_RESULT.md").write_text(
        "# Round 221 Result\n\n"
        "`admit_regen_as_source_correlated_independent_gridding_diagnostic`\n\n"
        f"Recovered 51 official NETP daily subsets for 1950-2000. `{int(eligible.sum())}/32` target cells pass the strict four-neighbor high-quality mask. "
        "REGEN is an independent gridding method, but its station and GPCC ancestry prevents treating it as a fully independent amplitude target.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
